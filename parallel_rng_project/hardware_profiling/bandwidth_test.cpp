/*
 * File: bandwidth_test.cpp
 * Project: Parallel Random Number Generation
 * Paper: "Parallel Random Numbers: As Easy as 1,2,3"
 *         Salmon et al., SC11, 2011
 * Course: CS-3006 Parallel and Distributed Computing
 * Purpose: Measures CPU peak memory bandwidth and compute throughput,
 *          then performs roofline arithmetic intensity analysis to classify
 *          Threefry and Mersenne Twister as compute-bound or memory-bound.
 * Compile: g++ -O2 -std=c++17 -o bandwidth_test bandwidth_test.cpp
 * Run:     ./bandwidth_test
 */

// ── Standard Library Includes ─────────────────────────────────────────────────
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>

// ── Using Namespace ───────────────────────────────────────────────────────────
using namespace std;

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr size_t ARRAY_SIZE_BYTES      = 512ULL * 1024ULL * 1024ULL; // 512 MB
static constexpr size_t ARRAY_ELEMENT_COUNT   = ARRAY_SIZE_BYTES / sizeof(double);
static constexpr int    BANDWIDTH_ITERATIONS  = 10;
static constexpr int    COMPUTE_ITERATIONS    = 10;
static constexpr long long FMA_LOOP_COUNT     = 500'000'000LL; // 500M FMA ops per iteration

// ── Threefry Arithmetic Intensity (from paper analysis) ───────────────────────
//
// Threefry-4x64-20 operation count:
//   20 rounds × (4 additions + 4 XORs + 4 rotations) = 20 × 12 = 240 operations
//
// Memory traffic per call:
//   Input:  counter (4 × 8 bytes = 32 bytes) + key (4 × 8 bytes = 32 bytes) = 64 bytes
//   Output: 4 × 8 bytes = 32 bytes
//   Total:  64 + 32 = 96 bytes
//
// Arithmetic Intensity = 240 ops / 96 bytes = 2.5 FLOP/byte
//
// Mersenne Twister:
//   State array: 624 × 32-bit = 2496 bytes  (must be read and partially written each call)
//   Operations per output: ~6 (twist + temper)
//   Arithmetic Intensity ≈ 6 / 2496 ≈ 0.0024 FLOP/byte
//   (extremely memory-bound — the bottleneck is loading the huge state)
static constexpr double THREEFRY_OPERATIONS_PER_CALL = 240.0;
static constexpr double THREEFRY_BYTES_PER_CALL       =  96.0;
static constexpr double THREEFRY_ARITHMETIC_INTENSITY =
    THREEFRY_OPERATIONS_PER_CALL / THREEFRY_BYTES_PER_CALL;  // = 2.5

static constexpr double MT_OPS_PER_OUTPUT    =   6.0;
static constexpr double MT_STATE_BYTES       = 2496.0;
static constexpr double MT_ARITHMETIC_INTENSITY =
    MT_OPS_PER_OUTPUT / MT_STATE_BYTES;  // ≈ 0.0024

// ── Helper: Timing using chrono ──────────────────────────────────────────────
static double get_elapsed_seconds(
    const chrono::high_resolution_clock::time_point& start,
    const chrono::high_resolution_clock::time_point& end)
{
    return chrono::duration<double>(end - start).count();
}

// ── Read-Only Bandwidth Benchmark ─────────────────────────────────────────────
// Reads every element of the array and accumulates a sum.
// The volatile sink prevents the compiler from eliminating the loop.
// PDC Concept: Memory bandwidth is often the limiting factor for
// memory-bound algorithms. This measures achievable read bandwidth.
static double benchmark_read_bandwidth_GBps(const vector<double>& read_array) {
    vector<double> iteration_times(BANDWIDTH_ITERATIONS);

    for (int iteration = 0; iteration < BANDWIDTH_ITERATIONS; ++iteration) {
        double accumulator = 0.0;
        auto time_start = chrono::high_resolution_clock::now();

        for (size_t element_index = 0;
             element_index < read_array.size();
             ++element_index)
        {
            accumulator += read_array[element_index];
        }

        auto time_end = chrono::high_resolution_clock::now();
        iteration_times[iteration] = get_elapsed_seconds(time_start, time_end);

        // Prevent dead-code elimination
        volatile double sink = accumulator;
        (void)sink;
    }

    double best_time_seconds = *min_element(
        iteration_times.begin(), iteration_times.end());

    return static_cast<double>(ARRAY_SIZE_BYTES) / (best_time_seconds * 1.0e9);
}

// ── Write-Only Bandwidth Benchmark ────────────────────────────────────────────
// Fills the array with a constant value to measure write bandwidth.
static double benchmark_write_bandwidth_GBps(vector<double>& write_array) {
    vector<double> iteration_times(BANDWIDTH_ITERATIONS);

    for (int iteration = 0; iteration < BANDWIDTH_ITERATIONS; ++iteration) {
        auto time_start = chrono::high_resolution_clock::now();

        for (size_t element_index = 0;
             element_index < write_array.size();
             ++element_index)
        {
            write_array[element_index] = static_cast<double>(iteration);
        }

        auto time_end = chrono::high_resolution_clock::now();
        iteration_times[iteration] = get_elapsed_seconds(time_start, time_end);
    }

    double best_time_seconds = *min_element(
        iteration_times.begin(), iteration_times.end());

    return static_cast<double>(ARRAY_SIZE_BYTES) / (best_time_seconds * 1.0e9);
}

// ── Copy Bandwidth Benchmark ──────────────────────────────────────────────────
// Copies source array to destination array (STREAM Copy benchmark).
// This is the classic STREAM benchmark that measures DRAM bandwidth.
// Reads source (512 MB) + writes destination (512 MB) = 1024 MB total.
static double benchmark_copy_bandwidth_GBps(const vector<double>& source_array,
                                             vector<double>&       destination_array)
{
    vector<double> iteration_times(BANDWIDTH_ITERATIONS);

    for (int iteration = 0; iteration < BANDWIDTH_ITERATIONS; ++iteration) {
        auto time_start = chrono::high_resolution_clock::now();

        for (size_t element_index = 0;
             element_index < source_array.size();
             ++element_index)
        {
            destination_array[element_index] = source_array[element_index];
        }

        auto time_end = chrono::high_resolution_clock::now();
        iteration_times[iteration] = get_elapsed_seconds(time_start, time_end);

        volatile double sink = destination_array[0];
        (void)sink;
    }

    double best_time_seconds = *min_element(
        iteration_times.begin(), iteration_times.end());

    // Total bytes = read (512 MB) + write (512 MB)
    double total_bytes_transferred =
        static_cast<double>(ARRAY_SIZE_BYTES) * 2.0;

    return total_bytes_transferred / (best_time_seconds * 1.0e9);
}

// ── Compute Throughput Benchmark ──────────────────────────────────────────────
// Runs a chain of fused multiply-add (FMA) operations to measure peak GFLOPS.
// FMA counts as 2 FLOPS (one multiply + one add) per instruction.
// PDC Concept: Compute ceiling in the roofline model.
static double benchmark_compute_GFLOPS() {
    vector<double> iteration_times(COMPUTE_ITERATIONS);

    for (int iteration = 0; iteration < COMPUTE_ITERATIONS; ++iteration) {
        // Use multiple independent accumulators to expose instruction-level
        // parallelism (ILP) and keep the FPU pipelines full.
        double acc0 = 1.0, acc1 = 1.0, acc2 = 1.0, acc3 = 1.0;
        const double multiplier = 1.0000001;
        const double addend     = 0.0000001;

        auto time_start = chrono::high_resolution_clock::now();

        for (long long fma_index = 0;
             fma_index < FMA_LOOP_COUNT / 4;
             ++fma_index)
        {
            acc0 = acc0 * multiplier + addend;
            acc1 = acc1 * multiplier + addend;
            acc2 = acc2 * multiplier + addend;
            acc3 = acc3 * multiplier + addend;
        }

        auto time_end = chrono::high_resolution_clock::now();
        iteration_times[iteration] = get_elapsed_seconds(time_start, time_end);

        volatile double sink = acc0 + acc1 + acc2 + acc3;
        (void)sink;
    }

    double best_time_seconds = *min_element(
        iteration_times.begin(), iteration_times.end());

    // FMA_LOOP_COUNT iterations × 4 accumulators × 2 FLOPS per FMA
    double total_flops = static_cast<double>(FMA_LOOP_COUNT) * 2.0;

    return total_flops / (best_time_seconds * 1.0e9);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    cout << "==========================================================\n";
    cout << "  Hardware Profiling: Memory Bandwidth & Roofline Analysis\n";
    cout << "  CS-3006 Parallel and Distributed Computing\n";
    cout << "==========================================================\n\n";

    // ── Allocate Test Arrays ──────────────────────────────────────────────────
    cout << "  Allocating 2 × " << (ARRAY_SIZE_BYTES / (1024 * 1024))
              << " MB arrays...\n\n";

    vector<double> array_source(ARRAY_ELEMENT_COUNT);
    vector<double> array_destination(ARRAY_ELEMENT_COUNT);

    // Initialize source with non-trivial values to prevent compiler folding
    for (size_t idx = 0; idx < ARRAY_ELEMENT_COUNT; ++idx)
        array_source[idx] = static_cast<double>(idx) * 0.000001;

    // ── Run Bandwidth Benchmarks ──────────────────────────────────────────────
    // PDC Concept: Memory hierarchy — DRAM bandwidth is typically the
    // bottleneck for memory-bound algorithms. Measuring it lets us place
    // algorithms correctly on the roofline model.
    cout << "── Memory Bandwidth Benchmarks ──────────────────────────\n";
    cout << "  Running " << BANDWIDTH_ITERATIONS
              << " iterations each, reporting best...\n\n";

    cout << "  [1/3] Read-only bandwidth...";
    cout.flush();
    double peak_read_bandwidth_GBps =
        benchmark_read_bandwidth_GBps(array_source);
    cout << "  " << fixed << setprecision(2)
              << peak_read_bandwidth_GBps << " GB/s\n";

    cout << "  [2/3] Write-only bandwidth...";
    cout.flush();
    double peak_write_bandwidth_GBps =
        benchmark_write_bandwidth_GBps(array_destination);
    cout << " " << fixed << setprecision(2)
              << peak_write_bandwidth_GBps << " GB/s\n";

    cout << "  [3/3] Copy (STREAM) bandwidth...";
    cout.flush();
    double peak_copy_bandwidth_GBps =
        benchmark_copy_bandwidth_GBps(array_source, array_destination);
    cout << " " << fixed << setprecision(2)
              << peak_copy_bandwidth_GBps << " GB/s\n\n";

    // Free arrays to reclaim memory
    array_source.clear();
    array_source.shrink_to_fit();
    array_destination.clear();
    array_destination.shrink_to_fit();

    // ── Run Compute Benchmark ─────────────────────────────────────────────────
    cout << "── Compute Throughput Benchmark ─────────────────────────\n";
    cout << "  Running " << COMPUTE_ITERATIONS
              << " FMA iterations (" << FMA_LOOP_COUNT / 1'000'000LL
              << "M FMAs each)...\n\n";

    cout << "  Peak GFLOPS...";
    cout.flush();
    double peak_compute_GFLOPS = benchmark_compute_GFLOPS();
    cout << "        " << fixed << setprecision(2)
              << peak_compute_GFLOPS << " GFLOPS\n\n";

    // ── Arithmetic Intensity Analysis ─────────────────────────────────────────
    // PDC Concept: Arithmetic Intensity = FLOPS / Bytes = the ratio of
    // computation to memory traffic. This determines whether an algorithm
    // is limited by memory bandwidth or compute throughput.
    cout << "── Arithmetic Intensity Analysis ────────────────────────\n\n";

    cout << "  Threefry-4x64-20 (from Salmon et al. SC11 2011):\n";
    cout << "  ┌─────────────────────────────────────────────────┐\n";
    cout << "  │  20 rounds × (4 add + 4 XOR + 4 rotate)         │\n";
    cout << "  │  = 20 × 12 = " << setw(3) << static_cast<int>(THREEFRY_OPERATIONS_PER_CALL)
              << " operations per call             │\n";
    cout << "  │                                                 │\n";
    cout << "  │  Input:  counter 32 B + key 32 B = 64 B         │\n";
    cout << "  │  Output: 4 × 8 B = 32 B                         │\n";
    cout << "  │  Total memory traffic = 96 bytes per call        │\n";
    cout << "  │                                                 │\n";
    cout << "  │  Arithmetic Intensity = 240 / 96 = "
              << fixed << setprecision(2) << THREEFRY_ARITHMETIC_INTENSITY
              << " FLOP/B  │\n";
    cout << "  └─────────────────────────────────────────────────┘\n\n";

    cout << "  Mersenne Twister (mt19937_64):\n";
    cout << "  ┌─────────────────────────────────────────────────┐\n";
    cout << "  │  State: 624 × 32-bit words = 2496 bytes          │\n";
    cout << "  │  Each output: ~6 operations (twist + temper)     │\n";
    cout << "  │  Memory traffic ≈ state size = 2496 bytes        │\n";
    cout << "  │                                                 │\n";
    cout << "  │  Arithmetic Intensity = 6 / 2496 ≈ "
              << fixed << setprecision(4) << MT_ARITHMETIC_INTENSITY
              << " FLOP/B │\n";
    cout << "  │  (100× lower than Threefry — extremely          │\n";
    cout << "  │   memory-bound at scale)                        │\n";
    cout << "  └─────────────────────────────────────────────────┘\n\n";

    // ── Roofline Model ────────────────────────────────────────────────────────
    // PDC Concept: Roofline Model (Williams et al. 2009)
    // Performance = min(peak_compute, bandwidth × arithmetic_intensity)
    //
    // Ridge point: the arithmetic intensity at which bandwidth roof meets
    // compute ceiling. Below the ridge → memory-bound. Above → compute-bound.
    double ridge_point_FLOP_per_byte = peak_compute_GFLOPS /
                                        peak_copy_bandwidth_GBps;

    // Achievable performance for each algorithm (GFLOPS)
    double threefry_achievable_GFLOPS =
        min(peak_compute_GFLOPS,
                 peak_copy_bandwidth_GBps * THREEFRY_ARITHMETIC_INTENSITY);

    double mt_achievable_GFLOPS =
        min(peak_compute_GFLOPS,
                 peak_copy_bandwidth_GBps * MT_ARITHMETIC_INTENSITY);

    cout << "── Roofline Model Analysis ──────────────────────────────\n\n";
    cout << "  Peak memory bandwidth (STREAM copy) : "
              << fixed << setprecision(2)
              << peak_copy_bandwidth_GBps  << " GB/s\n";
    cout << "  Peak compute (FMA throughput)        : "
              << fixed << setprecision(2)
              << peak_compute_GFLOPS       << " GFLOPS\n";
    cout << "  Ridge point                          : "
              << fixed << setprecision(3)
              << ridge_point_FLOP_per_byte << " FLOP/byte\n\n";

    cout << "  Threefry-4x64-20:\n";
    cout << "    Arithmetic intensity : "
              << THREEFRY_ARITHMETIC_INTENSITY << " FLOP/byte\n";
    cout << "    Achievable perf      : "
              << fixed << setprecision(2)
              << threefry_achievable_GFLOPS << " GFLOPS\n";
    if (THREEFRY_ARITHMETIC_INTENSITY > ridge_point_FLOP_per_byte) {
        cout << "    Verdict              : COMPUTE-BOUND\n";
        cout << "      → Threefry is limited by CPU throughput, not memory.\n";
        cout << "      → Scales linearly with core count (ideal for MPI).\n";
    } else {
        cout << "    Verdict              : MEMORY-BOUND\n";
        cout << "      → Threefry is limited by memory bandwidth.\n";
        cout << "      → Approaches memory bandwidth ceiling.\n";
    }
    cout << "\n";

    cout << "  Mersenne Twister:\n";
    cout << "    Arithmetic intensity : "
              << fixed << setprecision(6)
              << MT_ARITHMETIC_INTENSITY << " FLOP/byte\n";
    cout << "    Achievable perf      : "
              << fixed << setprecision(4)
              << mt_achievable_GFLOPS << " GFLOPS\n";
    cout << "    Verdict              : SEVERELY MEMORY-BOUND\n";
    cout << "      → MT's 2496-byte state causes cache pressure at scale.\n";
    cout << "      → With T threads: " << (2496*8/1024)
              << " KB state per 8 threads → cache overflow.\n";
    cout << "      → Scaling MT across cores degrades performance.\n\n";

    // ── Save Hardware Profile to CSV ──────────────────────────────────────────
    const string results_directory = "results";
    const string csv_file_path     = results_directory + "/hardware_profile.csv";

    try { filesystem::create_directories(results_directory); } catch (...) {}

    ofstream csv_output_file(csv_file_path);
    if (!csv_output_file.is_open()) {
        cerr << "[ERROR] Cannot open " << csv_file_path << " for writing.\n";
        return EXIT_FAILURE;
    }

    csv_output_file << "metric,value,unit\n";
    csv_output_file << fixed << setprecision(6);
    csv_output_file << "peak_read_bandwidth,"  << peak_read_bandwidth_GBps   << ",GB/s\n";
    csv_output_file << "peak_write_bandwidth," << peak_write_bandwidth_GBps  << ",GB/s\n";
    csv_output_file << "peak_copy_bandwidth,"  << peak_copy_bandwidth_GBps   << ",GB/s\n";
    csv_output_file << "peak_compute,"         << peak_compute_GFLOPS        << ",GFLOPS\n";
    csv_output_file << "ridge_point,"          << ridge_point_FLOP_per_byte  << ",FLOP/byte\n";
    csv_output_file << "threefry_arithmetic_intensity,"
                    << THREEFRY_ARITHMETIC_INTENSITY << ",FLOP/byte\n";
    csv_output_file << "mt_arithmetic_intensity,"
                    << MT_ARITHMETIC_INTENSITY << ",FLOP/byte\n";
    csv_output_file.close();

    cout << "  Hardware profile saved → " << csv_file_path << "\n\n";

    return EXIT_SUCCESS;
}
