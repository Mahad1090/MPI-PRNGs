/*
 * File: cuda_philox.cu
 * Project: Parallel Random Number Generation
 * Paper: "Parallel Random Numbers: As Easy as 1,2,3"
 *         Salmon et al., SC11, 2011
 * Course: CS-3006 Parallel and Distributed Computing
 * Purpose: GPU-accelerated counter-based PRNG using Philox-4x32-10.
 *          Each CUDA thread independently performs Philox calls using its
 *          global thread ID as the counter, storing all four generated words.
 * Compile: nvcc -O2 -std=c++17 -I../Random123/include -o cuda_philox cuda_philox.cu
 * Run:     ./cuda_philox [N]
 *          Example: ./cuda_philox 10000000
 */

// ── CUDA Runtime & Standard Includes ─────────────────────────────────────────
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <filesystem>

// ── Using Namespace ───────────────────────────────────────────────────────────
using namespace std;

// ── Random123 Header: Philox ──────────────────────────────────────────────────
// Random123 supports BOTH host (CPU) and device (GPU) execution with the
// SAME headers. The library detects __CUDA_ARCH__ at compile time and
// emits the appropriate machine instructions automatically.
//
// Why Philox for GPU? (Paper Section 4.3, Salmon et al. SC11 2011)
// ------------------------------------------------------------------
// Philox uses multiply-based "S-boxes" instead of Threefish rotations.
// On GPU:
//   - 32-bit integer multiply is a NATIVE single-cycle operation on all
//     NVIDIA GPU architectures from Fermi onward.
//   - Philox-4x32-10 uses only 32-bit multiplies → maximum throughput.
//   - Threefry uses 64-bit rotations which require 2 instructions on 32-bit
//     GPU hardware → ~2x slower than Philox on GPU.
// Conclusion: Philox is the paper's recommended choice for GPU PRNG.
//
// Philox-4x32-10 meaning:
//   4  : produces 4 output words per call
//   x32: each word is 32 bits
//   10 : uses 10 rounds of the S-box network
#include <Random123/philox.h>
#include <Random123/threefry.h>

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr long long DEFAULT_N          = 10'000'000LL;
static constexpr int       BLOCK_SIZE         = 256;    // threads per block
static constexpr int       NUM_WARMUP_RUNS    = 5;      // warm-up iterations
static constexpr int       NUM_TIMED_RUNS     = 5;      // timed iterations
static constexpr int       PHILOX_OUTPUT_WORDS = 4;     // Philox-4x32 returns 4 words
static constexpr uint32_t  SHARED_KEY_WORD_0  = 0xDEAD'BEEFu; // fixed key
static constexpr uint32_t  SHARED_KEY_WORD_1  = 0xCAFE'BABEu; // fixed key

// ── CUDA Error Checking Macro ──────────────────────────────────────────────────
// Wraps every CUDA API call to detect and report errors immediately.
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t cuda_error_code = (call);                                    \
        if (cuda_error_code != cudaSuccess) {                                    \
            cerr << "[CUDA ERROR] " << cudaGetErrorString(cuda_error_code)      \
                      << "  at " << __FILE__ << ":" << __LINE__ << "\n";         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// ── Why This Eliminates cuRAND ─────────────────────────────────────────────────
//
// cuRAND requires:
//   1. cudaMalloc a curandState array (one state object per thread)
//   2. curandInit kernel — each thread initializes its own state (slow!)
//   3. curandGenerate kernel — each thread generates values from its state
//   4. State arrays consume significant global memory
//
// With Philox (Random123):
//   1. No state initialization whatsoever
//   2. Thread ID maps to the Philox call counter — computation is immediate
//   3. Key is a compile-time constant
//   4. Zero global memory for state
//   5. Better statistical quality (counter-based bijection vs. LFSR)
//
// PDC Concept: Stateless computation — Philox output depends only on
// (key, counter), both of which are known at generation time without
// any prior computation. This is the key insight of Salmon et al. 2011.

// ── GPU Kernel: Philox Random Number Generation ───────────────────────────────
//
// PDC Concept: SIMT (Single Instruction, Multiple Thread) execution
// All threads execute the SAME instruction sequence (call philox4x32_10 with
// the same key) — the ONLY difference is the counter value derived from the
// thread ID. Each Philox call produces four 32-bit output words.
//
// Branching:
//   The main path is uniform. Only the final partial Philox call needs bounds
//   checks when N is not divisible by four.
//
// Memory coalescing:
//   Thread k writes to output_device_buffer[4k..4k+3]. Consecutive threads
//   still write consecutive 32-bit addresses, so warp writes remain coalesced.
//
// Counter-to-thread-ID mapping:
//   global_thread_id = blockIdx.x * blockDim.x + threadIdx.x
//   This maps each thread to a unique, deterministic Philox call counter.
//   Thread 0 → counter 0 → outputs 0..3
//   Thread 1 → counter 1 → outputs 4..7
//   Any thread can compute ANY value independently without knowing
//   what its neighbors computed.
__global__ void philox_generate_kernel(
    uint32_t* __restrict__ output_device_buffer,
    long long total_numbers_to_generate,
    uint32_t  key_word_0,
    uint32_t  key_word_1)
{
    // Compute this thread's unique global ID
    // PDC Concept: Thread indexing in a 1D grid
    long long global_thread_id = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);

    // Grid-stride loop: handles N > num_threads cases
    // PDC Concept: Grid-stride loops allow a fixed-size kernel launch to
    // process arbitrarily large N without launching one thread per element.
    long long grid_stride = static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    long long total_philox_calls =
        (total_numbers_to_generate + PHILOX_OUTPUT_WORDS - 1) / PHILOX_OUTPUT_WORDS;

    // Build Philox key once per thread. The key identifies the random stream.
    philox4x32_key_t philox_key = {{key_word_0, key_word_1}};

    for (long long philox_call_index = global_thread_id;
         philox_call_index < total_philox_calls;
         philox_call_index += grid_stride){
        // Build Philox counter: call index = position in Philox block sequence
        // Using 64-bit counter split across two 32-bit words to support
        // N > 2^32 without counter collisions.
        philox4x32_ctr_t philox_counter = {{
            static_cast<uint32_t>(philox_call_index & 0xFFFFFFFFULL),   // low 32 bits
            static_cast<uint32_t>(philox_call_index >> 32),             // high 32 bits
            0u,
            0u
        }};

        // Core computation: 10-round Philox bijection
        // This is a pure function — deterministic, no side effects,
        // no shared mutable state, no communication with other threads.
        philox4x32_ctr_t philox_output =
            philox4x32_10(philox_counter, philox_key);

        long long base_output_index = philox_call_index * PHILOX_OUTPUT_WORDS;

        // Fast path for complete 4-word blocks; tail path handles N % 4.
        if (base_output_index + 3 < total_numbers_to_generate) {
            output_device_buffer[base_output_index + 0] = philox_output.v[0];
            output_device_buffer[base_output_index + 1] = philox_output.v[1];
            output_device_buffer[base_output_index + 2] = philox_output.v[2];
            output_device_buffer[base_output_index + 3] = philox_output.v[3];
        } 
        else {
            if (base_output_index + 0 < total_numbers_to_generate)
                output_device_buffer[base_output_index + 0] = philox_output.v[0];
            if (base_output_index + 1 < total_numbers_to_generate)
                output_device_buffer[base_output_index + 1] = philox_output.v[1];
            if (base_output_index + 2 < total_numbers_to_generate)
                output_device_buffer[base_output_index + 2] = philox_output.v[2];
            if (base_output_index + 3 < total_numbers_to_generate)
                output_device_buffer[base_output_index + 3] = philox_output.v[3];
        }
    }
}

// ── Helper: Format Large Numbers with Commas ─────────────────────────────────
static string format_with_commas(long long value) {
    string number_string = to_string(value);
    int insert_position = static_cast<int>(number_string.size()) - 3;
    while (insert_position > 0) {
        number_string.insert(static_cast<size_t>(insert_position), ",");
        insert_position -= 3;
    }
    return number_string;
}

// ── Helper: Read Baseline Time from CSV ──────────────────────────────────────
static double read_baseline_time_seconds(const string& csv_path) {
    ifstream csv_input_file(csv_path);
    if (!csv_input_file.is_open()) return -1.0;
    string header_line;
    getline(csv_input_file, header_line);
    string data_line;
    if (!getline(csv_input_file, data_line)) return -1.0;
    istringstream line_stream(data_line);
    string token;
    getline(line_stream, token, ',');
    getline(line_stream, token, ',');
    try { return stod(token); } catch (...) { return -1.0; }
}

// ── CPU Single Core Threefry Benchmark (same machine comparison) ─────────────
// Runs Threefry-4x64-20 for the same N on the host CPU to provide a
// direct apples-to-apples comparison on the same hardware.
//
// We use Threefry (not MT) because:
//   - Threefry is the authors' baseline from Random123 (same paper as Philox)
//   - Both Threefry and Philox are counter-based PRNGs from the same library
//   - This comparison isolates the CPU-vs-GPU dimension, not algorithm
//   - MT is only background motivation, not a benchmark target
static double cpu_threefry_benchmark_seconds(long long total_numbers) {
    vector<uint64_t> cpu_buffer(static_cast<size_t>(total_numbers));
    static const threefry4x64_key_t tf_key = {{ 0ULL, 0ULL, 0ULL, 0ULL }};

    auto cpu_start = chrono::high_resolution_clock::now();

    long long full_batches    = total_numbers / 4;
    long long leftover_values = total_numbers % 4;
    long long write_idx       = 0;

    for (long long batch = 0; batch < full_batches; ++batch) {
        threefry4x64_ctr_t ctr = {{ static_cast<uint64_t>(batch * 4), 0ULL, 0ULL, 0ULL }};
        threefry4x64_ctr_t out = threefry4x64(ctr, tf_key);
        cpu_buffer[static_cast<size_t>(write_idx + 0)] = out.v[0];
        cpu_buffer[static_cast<size_t>(write_idx + 1)] = out.v[1];
        cpu_buffer[static_cast<size_t>(write_idx + 2)] = out.v[2];
        cpu_buffer[static_cast<size_t>(write_idx + 3)] = out.v[3];
        write_idx += 4;
    }
    if (leftover_values > 0) {
        threefry4x64_ctr_t ctr = {{ static_cast<uint64_t>(full_batches * 4), 0ULL, 0ULL, 0ULL }};
        threefry4x64_ctr_t out = threefry4x64(ctr, tf_key);
        for (long long i = 0; i < leftover_values; ++i)
            cpu_buffer[static_cast<size_t>(write_idx++)] = out.v[static_cast<size_t>(i)];
    }

    auto cpu_end = chrono::high_resolution_clock::now();

    volatile uint64_t sink = cpu_buffer[0];
    (void)sink;

    return chrono::duration<double>(cpu_end - cpu_start).count();
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Parse Command-Line Arguments ──────────────────────────────────────────
    long long total_numbers_to_generate = DEFAULT_N;
    if (argc >= 2) {
        try {
            total_numbers_to_generate = stoll(argv[1]);
            if (total_numbers_to_generate <= 0) {
                cerr << "[ERROR] N must be positive. Got: " << argv[1] << "\n";
                return EXIT_FAILURE;
            }
        } catch (const exception& ex) {
            cerr << "[ERROR] Invalid N: " << argv[1] << " — " << ex.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    cout << "==========================================================\n";
    cout << "  GPU Philox-4x32-10 CUDA PRNG\n";
    cout << "  CS-3006 Parallel and Distributed Computing\n";
    cout << "==========================================================\n\n";

    // ── Check GPU Availability ────────────────────────────────────────────────
    int cuda_device_count = 0;
    cudaError_t device_query_error = cudaGetDeviceCount(&cuda_device_count);
    if (device_query_error != cudaSuccess || cuda_device_count == 0) {
        cerr << "[ERROR] No CUDA-capable GPU found. "
                  << "Install CUDA drivers or run on a GPU machine.\n";
        return EXIT_FAILURE;
    }

    // ── Query & Print GPU Properties ─────────────────────────────────────────
    cudaDeviceProp gpu_device_properties;
    CUDA_CHECK(cudaGetDeviceProperties(&gpu_device_properties, 0));
    CUDA_CHECK(cudaSetDevice(0));

    size_t gpu_free_memory_bytes  = 0;
    size_t gpu_total_memory_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&gpu_free_memory_bytes, &gpu_total_memory_bytes));

    cout << "  GPU Device      : " << gpu_device_properties.name << "\n";
    cout << "  Compute Cap.    : " << gpu_device_properties.major
              << "." << gpu_device_properties.minor << "\n";
    cout << "  SM count        : " << gpu_device_properties.multiProcessorCount << "\n";
    cout << "  Total VRAM      : "
              << gpu_total_memory_bytes / (1024 * 1024) << " MB\n";
    cout << "  Free  VRAM      : "
              << gpu_free_memory_bytes  / (1024 * 1024) << " MB\n";
    cout << "  Max threads/blk : "
              << gpu_device_properties.maxThreadsPerBlock << "\n\n";

    // ── Compute Grid Dimensions ───────────────────────────────────────────────
    // PDC Concept: Thread hierarchy — CUDA executes threads in blocks,
    // blocks in a grid. We choose BLOCK_SIZE=256 (common optimal value)
    // and compute grid size to cover ceil(N / 4) Philox calls.
    int  threads_per_block = BLOCK_SIZE;
    long long total_philox_calls =
        (total_numbers_to_generate + PHILOX_OUTPUT_WORDS - 1) / PHILOX_OUTPUT_WORDS;
    long long num_blocks_raw =
        (total_philox_calls + threads_per_block - 1) / threads_per_block;

    // Cap grid size to device maximum to avoid launch errors
    int max_grid_dim = gpu_device_properties.maxGridSize[0];
    int num_blocks   = static_cast<int>(
        min(num_blocks_raw, static_cast<long long>(max_grid_dim)));

    long long total_gpu_threads =
        static_cast<long long>(num_blocks) * threads_per_block;

    cout << "  N               : "
              << format_with_commas(total_numbers_to_generate) << "\n";
    cout << "  Philox calls    : "
              << format_with_commas(total_philox_calls) << "\n";
    cout << "  Threads / block : " << threads_per_block << "\n";
    cout << "  Grid blocks     : " << num_blocks << "\n";
    cout << "  Total threads   : "
              << format_with_commas(total_gpu_threads) << "\n\n";

    // ── Allocate Device Memory ────────────────────────────────────────────────
    // We generate uint32_t (4-byte) values because Philox-4x32 produces
    // 32-bit words natively. Using the native width avoids packing overhead.
    size_t output_buffer_size_bytes =
        static_cast<size_t>(total_numbers_to_generate) * sizeof(uint32_t);

    uint32_t* output_device_buffer = nullptr;
    cudaError_t malloc_error =
        cudaMalloc(reinterpret_cast<void**>(&output_device_buffer),
                   output_buffer_size_bytes);
    if (malloc_error != cudaSuccess) {
        cerr << "[ERROR] cudaMalloc failed ("
                  << output_buffer_size_bytes / (1024*1024) << " MB): "
                  << cudaGetErrorString(malloc_error) << "\n";
        return EXIT_FAILURE;
    }

    // ── Create CUDA Timing Events ─────────────────────────────────────────────
    // PDC Concept: GPU timing — cudaEventElapsedTime gives precise GPU-side
    // timing in milliseconds, independent of CPU overhead. This is more
    // accurate than std::chrono for GPU kernels because it measures exactly
    // the time the GPU spent executing, not including CPU→GPU scheduling delay.
    cudaEvent_t cuda_event_start, cuda_event_stop;
    CUDA_CHECK(cudaEventCreate(&cuda_event_start));
    CUDA_CHECK(cudaEventCreate(&cuda_event_stop));

    // ── Warm-Up Iterations ────────────────────────────────────────────────────
    // GPU kernels incur a one-time JIT compilation and initialization overhead
    // on the first launch. We run NUM_WARMUP_RUNS un-timed iterations to bring
    // the GPU into steady state before measuring performance.
    cout << "  Running " << NUM_WARMUP_RUNS << " warm-up iterations...\n";
    for (int warmup_index = 0; warmup_index < NUM_WARMUP_RUNS; ++warmup_index) {
        philox_generate_kernel<<<num_blocks, threads_per_block>>>(
            output_device_buffer,
            total_numbers_to_generate,
            SHARED_KEY_WORD_0,
            SHARED_KEY_WORD_1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed Iterations ──────────────────────────────────────────────────────
    cout << "  Running " << NUM_TIMED_RUNS << " timed iterations...\n\n";
    float total_gpu_time_ms = 0.0f;

    for (int timed_run_index = 0;
         timed_run_index < NUM_TIMED_RUNS;
         ++timed_run_index)
    {
        // Place start event in the default CUDA stream
        CUDA_CHECK(cudaEventRecord(cuda_event_start, 0));

        // Launch Philox kernel
        philox_generate_kernel<<<num_blocks, threads_per_block>>>(
            output_device_buffer,
            total_numbers_to_generate,
            SHARED_KEY_WORD_0,
            SHARED_KEY_WORD_1);

        // Place stop event and wait for GPU to complete
        CUDA_CHECK(cudaEventRecord(cuda_event_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(cuda_event_stop));

        // Retrieve elapsed time in milliseconds (GPU-side measurement)
        float iteration_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iteration_time_ms,
                                        cuda_event_start,
                                        cuda_event_stop));

        total_gpu_time_ms += iteration_time_ms;
        cout << "  Timed run " << (timed_run_index + 1) << "/"
                  << NUM_TIMED_RUNS << "  →  "
                  << fixed << setprecision(3)
                  << iteration_time_ms << " ms\n";
    }

    double average_gpu_time_ms      = total_gpu_time_ms / NUM_TIMED_RUNS;
    double average_gpu_time_seconds = average_gpu_time_ms / 1000.0;

    // GPU throughput: N × 4 bytes / time
    double gpu_throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint32_t)) /
        (average_gpu_time_seconds * 1.0e9);

    // ── Copy Results Back to Host ─────────────────────────────────────────────
    // Transfer generated values from GPU global memory to CPU RAM.
    // In a real application this would be avoided if the data is consumed
    // on the GPU (e.g., Monte Carlo simulation entirely on GPU).
    vector<uint32_t> host_output_buffer(
        static_cast<size_t>(total_numbers_to_generate));

    CUDA_CHECK(cudaMemcpy(host_output_buffer.data(),
                          output_device_buffer,
                          output_buffer_size_bytes,
                          cudaMemcpyDeviceToHost));

    // ── GPU Baseline: Single Thread Philox <<<1, 1>>> ───────────────────────────
    // PDC Concept: GPU parallelization baseline.
    // Runs the SAME Philox kernel on the SAME GPU hardware with <<<1, 1>>>.
    // The grid-stride loop in philox_generate_kernel processes all N Philox
    // calls sequentially in that single thread.
    //
    // PRIMARY SPEEDUP DEFINITION:
    //   speedup = gpu_baseline_time / gpu_parallel_time
    //   → Same hardware + same algorithm + same data, only thread count differs.
    //   → This cleanly isolates the GPU PARALLELIZATION benefit.
    //   → No hardware differences, no algorithm differences contaminate it.
    cout << "\n  GPU baseline warmup (1 CUDA thread, " << NUM_WARMUP_RUNS << " iters)...\n";
    for (int warmup = 0; warmup < NUM_WARMUP_RUNS; ++warmup) {
        philox_generate_kernel<<<1, 1>>>(
            output_device_buffer, total_numbers_to_generate,
            SHARED_KEY_WORD_0, SHARED_KEY_WORD_1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cout << "  GPU baseline timed (" << NUM_TIMED_RUNS << " iters)...\n";
    float total_gpu_baseline_ms = 0.0f;
    for (int base_run = 0; base_run < NUM_TIMED_RUNS; ++base_run) {
        CUDA_CHECK(cudaEventRecord(cuda_event_start, 0));
        philox_generate_kernel<<<1, 1>>>(
            output_device_buffer, total_numbers_to_generate,
            SHARED_KEY_WORD_0, SHARED_KEY_WORD_1);
        CUDA_CHECK(cudaEventRecord(cuda_event_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(cuda_event_stop));
        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, cuda_event_start, cuda_event_stop));
        total_gpu_baseline_ms += iter_ms;
        cout << "  Baseline run " << (base_run + 1) << "/" << NUM_TIMED_RUNS
                  << "  →  " << fixed << setprecision(3) << iter_ms << " ms\n";
    }
    double average_gpu_baseline_ms      = total_gpu_baseline_ms / NUM_TIMED_RUNS;
    double average_gpu_baseline_seconds = average_gpu_baseline_ms / 1000.0;
    double gpu_baseline_throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint32_t)) /
        (average_gpu_baseline_seconds * 1.0e9);

    // ── CPU Single Core Threefry (informational — different hardware) ─────────
    cout << "\n  Running CPU single core Threefry benchmark (informational)...\n";
    double cpu_elapsed_seconds = cpu_threefry_benchmark_seconds(total_numbers_to_generate);
    double cpu_throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
        (cpu_elapsed_seconds * 1.0e9);

    // ── Primary Speedup: GPU Parallel vs GPU Single Thread ────────────────────
    double gpu_parallel_speedup = average_gpu_baseline_ms / average_gpu_time_ms;

    // Hardware ratio (informational only — different CPU vs GPU hardware)
    double hardware_ratio = cpu_elapsed_seconds / average_gpu_time_seconds;

    // ── Format helper (avoids repeated ostringstream boilerplate) ─────────────
    auto fmt = [](double v, int prec) -> string {
        ostringstream oss;
        oss << fixed << setprecision(prec) << v;
        return oss.str();
    };

    // ── Print Results Table ─────────────────────────────────────────────
    // Inner box width: 49 chars. Format:
    //   Header rows: "│  " + setw(47) + "│"
    //   Data rows:   "│  " + setw(21) label + ": " + setw(24) value + "│"
    //   Indented:    "│    " + setw(19) label + ": " + setw(24) value + "│"
    cout << "\n┌─────────────────────────────────────────────────┐\n";
    cout <<   "│  GPU Philox Results                             │\n";
    cout <<   "├─────────────────────────────────────────────────┤\n";
    cout <<   "│  " << left << setw(21) << "GPU device"
              << ": " << left << setw(24) << string(gpu_device_properties.name).substr(0,24) << "│\n";
    cout <<   "│  " << left << setw(21) << "N generated"
              << ": " << left << setw(24) << format_with_commas(total_numbers_to_generate) << "│\n";
    cout <<   "├─────────────────────────────────────────────────┤\n";
    cout <<   "│  GPU Baseline (1 CUDA thread, Philox)           │\n";
    cout <<   "│    " << left << setw(19) << "Time"
              << ": " << left << setw(24) << (fmt(average_gpu_baseline_ms,2) + " ms") << "│\n";
    cout <<   "│    " << left << setw(19) << "Throughput"
              << ": " << left << setw(24) << (fmt(gpu_baseline_throughput_GBps,2) + " GB/s") << "│\n";
    cout <<   "├─────────────────────────────────────────────────┤\n";
    cout <<   "│  " << left << setw(47)
              << ("GPU Parallel (" + format_with_commas(total_gpu_threads) + " threads, Philox)")
              << "│\n";
    cout <<   "│    " << left << setw(19) << "Threads used"
              << ": " << left << setw(24) << format_with_commas(total_gpu_threads) << "│\n";
    cout <<   "│    " << left << setw(19) << "Grid dims"
              << ": " << left << setw(24)
              << (to_string(num_blocks) + "b x " + to_string(threads_per_block) + "t")
              << "│\n";
    cout <<   "│    " << left << setw(19) << "Time"
              << ": " << left << setw(24) << (fmt(average_gpu_time_ms,2) + " ms") << "│\n";
    cout <<   "│    " << left << setw(19) << "Throughput"
              << ": " << left << setw(24) << (fmt(gpu_throughput_GBps,2) + " GB/s") << "│\n";
    cout <<   "│    " << left << setw(19) << "Speedup vs baseline"
              << ": " << left << setw(24) << (fmt(gpu_parallel_speedup,2) + "x") << "│\n";
    cout <<   "├─────────────────────────────────────────────────┤\n";
    cout <<   "│  Hardware Observation (informational only)       │\n";
    cout <<   "│    " << left << setw(19) << "CPU Single Core"
              << ": " << left << setw(24) << (fmt(cpu_throughput_GBps,2) + " GB/s (Threefry)") << "│\n";
    cout <<   "│    " << left << setw(19) << "GPU Parallel"
              << ": " << left << setw(24) << (fmt(gpu_throughput_GBps,2) + " GB/s (Philox)") << "│\n";
    cout <<   "│    " << left << setw(19) << "Hardware Ratio"
              << ": " << left << setw(24) << (fmt(hardware_ratio,2) + "x (NOT primary speedup)") << "│\n";
    cout <<   "└─────────────────────────────────────────────────┘\n\n";

    // ── Save Results to CSV ─────────────────────────────────────────────
    // New CSV format:
    // N, gpu_baseline_time_ms, gpu_baseline_throughput_GBps,
    // gpu_parallel_time_ms, gpu_parallel_throughput_GBps,
    // gpu_speedup_vs_single_thread, cpu_threefry_throughput_GBps, num_cuda_threads
    const string results_directory = "results";
    const string csv_file_path     = results_directory + "/gpu_results.csv";

    try { filesystem::create_directories(results_directory); } catch (...) {}

    ofstream csv_output_file(csv_file_path);
    if (!csv_output_file.is_open()) {
        cerr << "[ERROR] Cannot open " << csv_file_path << " for writing.\n";
    } else {
        csv_output_file << "N,gpu_baseline_time_ms,gpu_baseline_throughput_GBps,"
                           "gpu_parallel_time_ms,gpu_parallel_throughput_GBps,"
                           "gpu_speedup_vs_single_thread,"
                           "cpu_threefry_throughput_GBps,num_cuda_threads\n";
        csv_output_file << fixed << setprecision(6)
            << total_numbers_to_generate        << ","
            << average_gpu_baseline_ms          << ","
            << gpu_baseline_throughput_GBps     << ","
            << average_gpu_time_ms              << ","
            << gpu_throughput_GBps              << ","
            << gpu_parallel_speedup             << ","
            << cpu_throughput_GBps              << ","
            << total_gpu_threads                << "\n";
        csv_output_file.close();
        cout << "  Results saved → " << csv_file_path << "\n\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(cuda_event_start));
    CUDA_CHECK(cudaEventDestroy(cuda_event_stop));
    CUDA_CHECK(cudaFree(output_device_buffer));

    // Sanity check: print first generated value
    cout << "  [Sanity] First GPU-generated value: 0x"
              << hex << host_output_buffer[0] << dec << "\n\n";

    return EXIT_SUCCESS;
}
