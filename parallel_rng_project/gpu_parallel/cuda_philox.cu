/*
 * Philox-4x32-10 on the GPU. Each Philox call produces four 32-bit outputs,
 * and the call index is the counter, so threads never coordinate.
 *
 * Speedup is reported against the same kernel launched as <<<1, 1>>> on the
 * same GPU. The single-core CPU Threefry time is printed for reference only.
 *
 * Build: nvcc -O2 -std=c++17 -I../random123/include -o cuda_philox cuda_philox.cu
 * Run:   ./cuda_philox [N]
 */

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

using namespace std;

#include <Random123/philox.h>
#include <Random123/threefry.h>

static constexpr long long DEFAULT_N          = 10'000'000LL;
static constexpr int       BLOCK_SIZE         = 256;
static constexpr int       NUM_WARMUP_RUNS    = 5;
static constexpr int       NUM_TIMED_RUNS     = 5;
static constexpr int       PHILOX_OUTPUT_WORDS = 4;
static constexpr uint32_t  SHARED_KEY_WORD_0  = 0xDEAD'BEEFu;
static constexpr uint32_t  SHARED_KEY_WORD_1  = 0xCAFE'BABEu;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t cuda_error_code = (call);                                    \
        if (cuda_error_code != cudaSuccess) {                                    \
            cerr << "[CUDA ERROR] " << cudaGetErrorString(cuda_error_code)      \
                      << "  at " << __FILE__ << ":" << __LINE__ << "\n";         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Philox call k writes outputs 4k..4k+3. Adjacent threads therefore write
// adjacent 16-byte chunks, which keeps warp stores coalesced. Only the final
// call needs bounds checks, when N is not a multiple of four.
__global__ void philox_generate_kernel(
    uint32_t* __restrict__ output_device_buffer,
    long long total_numbers_to_generate,
    uint32_t  key_word_0,
    uint32_t  key_word_1)
{
    long long global_thread_id = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) + static_cast<long long>(threadIdx.x);

    // Grid-stride loop: the grid is capped at the device limit, and the
    // <<<1, 1>>> baseline reuses this kernel to do every call in one thread.
    long long grid_stride = static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    long long total_philox_calls =
        (total_numbers_to_generate + PHILOX_OUTPUT_WORDS - 1) / PHILOX_OUTPUT_WORDS;

    philox4x32_key_t philox_key = {{key_word_0, key_word_1}};

    for (long long philox_call_index = global_thread_id;
         philox_call_index < total_philox_calls;
         philox_call_index += grid_stride){
        // The 64-bit call index is split across two counter words so N can
        // exceed 2^32 without counters repeating.
        philox4x32_ctr_t philox_counter = {{
            static_cast<uint32_t>(philox_call_index & 0xFFFFFFFFULL),
            static_cast<uint32_t>(philox_call_index >> 32),
            0u,
            0u
        }};

        philox4x32_ctr_t philox_output =
            philox4x32_10(philox_counter, philox_key);

        long long base_output_index = philox_call_index * PHILOX_OUTPUT_WORDS;

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

static string format_with_commas(long long value) {
    string number_string = to_string(value);
    int insert_position = static_cast<int>(number_string.size()) - 3;
    while (insert_position > 0) {
        number_string.insert(static_cast<size_t>(insert_position), ",");
        insert_position -= 3;
    }
    return number_string;
}

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

// Same loop as baseline/sequential_threefry.cpp, run on this host so the CPU
// and GPU numbers in the report come from one machine.
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

int main(int argc, char* argv[])
{
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

    int cuda_device_count = 0;
    cudaError_t device_query_error = cudaGetDeviceCount(&cuda_device_count);
    if (device_query_error != cudaSuccess || cuda_device_count == 0) {
        cerr << "[ERROR] No CUDA-capable GPU found. "
                  << "Install CUDA drivers or run on a GPU machine.\n";
        return EXIT_FAILURE;
    }

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

    // One thread per Philox call, i.e. ceil(N / 4) threads in total.
    int  threads_per_block = BLOCK_SIZE;
    long long total_philox_calls =
        (total_numbers_to_generate + PHILOX_OUTPUT_WORDS - 1) / PHILOX_OUTPUT_WORDS;
    long long num_blocks_raw =
        (total_philox_calls + threads_per_block - 1) / threads_per_block;

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

    // CUDA events time the kernel on the device, excluding host launch overhead.
    cudaEvent_t cuda_event_start, cuda_event_stop;
    CUDA_CHECK(cudaEventCreate(&cuda_event_start));
    CUDA_CHECK(cudaEventCreate(&cuda_event_stop));

    // Untimed launches absorb context creation and module load costs.
    cout << "  Running " << NUM_WARMUP_RUNS << " warm-up iterations...\n";
    for (int warmup_index = 0; warmup_index < NUM_WARMUP_RUNS; ++warmup_index) {
        philox_generate_kernel<<<num_blocks, threads_per_block>>>(
            output_device_buffer,
            total_numbers_to_generate,
            SHARED_KEY_WORD_0,
            SHARED_KEY_WORD_1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cout << "  Running " << NUM_TIMED_RUNS << " timed iterations...\n\n";
    float total_gpu_time_ms = 0.0f;

    for (int timed_run_index = 0;
         timed_run_index < NUM_TIMED_RUNS;
         ++timed_run_index)
    {
        CUDA_CHECK(cudaEventRecord(cuda_event_start, 0));

        philox_generate_kernel<<<num_blocks, threads_per_block>>>(
            output_device_buffer,
            total_numbers_to_generate,
            SHARED_KEY_WORD_0,
            SHARED_KEY_WORD_1);

        CUDA_CHECK(cudaEventRecord(cuda_event_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(cuda_event_stop));

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

    double gpu_throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint32_t)) /
        (average_gpu_time_seconds * 1.0e9);

    // Copied back only for the sanity print at the end; not timed.
    vector<uint32_t> host_output_buffer(
        static_cast<size_t>(total_numbers_to_generate));

    CUDA_CHECK(cudaMemcpy(host_output_buffer.data(),
                          output_device_buffer,
                          output_buffer_size_bytes,
                          cudaMemcpyDeviceToHost));

    // Same kernel, same GPU, one thread: the reported speedup isolates the
    // effect of thread count from hardware and algorithm differences.
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

    cout << "\n  Running CPU single core Threefry benchmark (informational)...\n";
    double cpu_elapsed_seconds = cpu_threefry_benchmark_seconds(total_numbers_to_generate);
    double cpu_throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
        (cpu_elapsed_seconds * 1.0e9);

    double gpu_parallel_speedup = average_gpu_baseline_ms / average_gpu_time_ms;

    // CPU vs GPU compares different hardware and word sizes; reported, not
    // used as the speedup.
    double hardware_ratio = cpu_elapsed_seconds / average_gpu_time_seconds;

    auto fmt = [](double v, int prec) -> string {
        ostringstream oss;
        oss << fixed << setprecision(prec) << v;
        return oss.str();
    };

    // Inner box width is 49 columns:
    //   header rows   "│  " + setw(47) + "│"
    //   data rows     "│  " + setw(21) label + ": " + setw(24) value + "│"
    //   indented rows "│    " + setw(19) label + ": " + setw(24) value + "│"
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

    // Column order is what roofline_analysis.cpp and generate_plots.py expect.
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

    CUDA_CHECK(cudaEventDestroy(cuda_event_start));
    CUDA_CHECK(cudaEventDestroy(cuda_event_stop));
    CUDA_CHECK(cudaFree(output_device_buffer));

    cout << "  [Sanity] First GPU-generated value: 0x"
              << hex << host_output_buffer[0] << dec << "\n\n";

    return EXIT_SUCCESS;
}
