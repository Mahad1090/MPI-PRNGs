/*
 * File: sequential_threefry.cpp
 * Project: Parallel Random Number Generation
 * Paper: "Parallel Random Numbers: As Easy as 1,2,3"
 *         Salmon et al., SC11, 2011
 * Course: CS-3006 Parallel and Distributed Computing
 * Purpose: Sequential single-core baseline using Threefry-4x64-20 from the
 *          Random123 library. This is the AUTHORS' OWN implementation running
 *          on a single CPU core with no MPI and no CUDA.
 *
 *          This is the CORRECT baseline for this project:
 *            - The paper introduces counter-based PRNGs; Threefry is their impl.
 *            - Mersenne Twister is only background context, NOT the baseline.
 *            - Speedup of MPI and GPU versions is measured AGAINST this file.
 *
 * Compile: g++ -O2 -std=c++17 -I../random123/include \
 *              -o sequential_threefry sequential_threefry.cpp
 * Run:     ./sequential_threefry [N]
 *          Example: ./sequential_threefry 10000000
 */

// ── Standard Library Includes ────────────────────────────────────────────────
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <filesystem>

// ── Using Namespace ───────────────────────────────────────────────────────────
using namespace std;

// ── Random123 Header: Threefry ────────────────────────────────────────────────
// Random123 is HEADER-ONLY — no build step needed.
// Threefry-4x64-20 generates 4 × 64-bit values per call using 20 rounds
// of the Threefish block cipher key schedule.
//
// Why Threefry for CPU? (Paper Section 4.1, Salmon et al. SC11 2011)
// -----------------------------------------------------------------
// Threefry uses 64-bit integer additions, XORs, and rotations.
// Modern CPUs handle these natively and efficiently.
// Threefry is the paper's recommended choice for general-purpose CPU PRNG.
//
// Counter-based model:
//   output(n) = bijection(key, counter_n)
//   Pure function, ZERO mutable state, trivially parallelizable.
#include <Random123/threefry.h>

// ── Constants ────────────────────────────────────────────────────────────────
static constexpr long long DEFAULT_N       = 10'000'000LL;  // 10 million values
static constexpr int       NUM_TIMING_RUNS = 5;             // runs to average

// Fixed key: all zeros — reproducible. Any constant works; the key
// distinguishes this stream from other parameterizations.
// (In mpi_threefry.cpp each rank uses its rank number as word 0 of the key.)
static const threefry4x64_key_t THREEFRY_KEY = {{ 0ULL, 0ULL, 0ULL, 0ULL }};

// ── Project Baseline Story ────────────────────────────────────────────────────
//
// This file is Step 1 of the project story:
//
//   Step 1 — THIS FILE: Single core Threefry-4x64-20
//             The authors' own implementation from Random123.
//             Serves as the reference point (speedup = 1.00x).
//
//   Step 2 — cpu_parallel/mpi_threefry.cpp:
//             Same Threefry-4x64-20, parallelized with MPI.
//             Each rank uses rank number as unique key.
//             Speedup measured AGAINST this single-core baseline.
//
//   Step 3 — gpu_parallel/cuda_philox.cu:
//             Philox-4x32-10 on NVIDIA GPU via CUDA.
//             Philox is Threefry's GPU-native sibling from the same paper.
//             Speedup measured AGAINST this single-core baseline.
//
// Role of Mersenne Twister in this project:
//   MT is NOT a baseline. It is mentioned only in comments and the report
//   as background motivation — to show WHY counter-based PRNGs were needed.
//   MT has a 2496-byte sequential state; Threefry/Philox have ZERO state.

// ── Helper: Format Large Numbers with Commas ─────────────────────────────────
static string format_with_commas(long long value) {
    string s = to_string(value);
    int pos = static_cast<int>(s.size()) - 3;
    while (pos > 0) {
        s.insert(static_cast<size_t>(pos), ",");
        pos -= 3;
    }
    return s;
}

// ── Helper: Print Formatted Results Table ────────────────────────────────────
static void print_results_table(long long total_numbers_generated,
                                double    elapsed_seconds,
                                double    throughput_GBps)
{
    cout << "\n┌─────────────────────────────────────────────────┐\n";
    cout <<   "│   Single Core Threefry-4x64-20 Results          │\n";
    cout <<   "│   (Authors Baseline — Random123 Library)         │\n";
    cout <<   "├─────────────────────────────────────────────────┤\n";
    cout <<   "│  N generated  : " << left << setw(32)
              << format_with_commas(total_numbers_generated) << "│\n";
    cout <<   "│  Timing runs  : " << left << setw(32)
              << NUM_TIMING_RUNS << "│\n";
    cout <<   "│  Avg time     : " << left << setw(32)
              << (to_string(elapsed_seconds).substr(0,6) + " seconds") << "│\n";
    cout <<   "│  Throughput   : " << left << setw(32)
              << (to_string(throughput_GBps).substr(0,6) + " GB/s") << "│\n";
    cout <<   "│  Speedup      : " << left << setw(32)
              << "1.00x (authors baseline)" << "│\n";
    cout <<   "└─────────────────────────────────────────────────┘\n\n";
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Parse Command-Line Arguments ─────────────────────────────────────────
    long long total_numbers_to_generate = DEFAULT_N;
    if (argc >= 2) {
        try {
            total_numbers_to_generate = stoll(argv[1]);
            if (total_numbers_to_generate <= 0) {
                cerr << "[ERROR] N must be a positive integer. Got: "
                          << argv[1] << "\n";
                return EXIT_FAILURE;
            }
        } catch (const exception& ex) {
            cerr << "[ERROR] Invalid argument for N: " << argv[1]
                      << " — " << ex.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    cout << "==========================================================\n";
    cout << "  Single Core Threefry-4x64-20 Baseline\n";
    cout << "  (Authors Implementation from Random123 Library)\n";
    cout << "  CS-3006 Parallel and Distributed Computing\n";
    cout << "==========================================================\n";
    cout << "  N = " << format_with_commas(total_numbers_to_generate)
              << "  |  Runs = " << NUM_TIMING_RUNS << "\n\n";

    // ── Allocate Output Buffer ────────────────────────────────────────────────
    // Threefry-4x64 produces 4 × uint64_t (8 bytes each) per call.
    // We store all N values to measure real memory pressure, not just compute.
    vector<uint64_t> generated_values;
    try {
        generated_values.resize(static_cast<size_t>(total_numbers_to_generate));
    } catch (const bad_alloc& alloc_error) {
        cerr << "[ERROR] Could not allocate "
                  << total_numbers_to_generate * 8 / (1024 * 1024)
                  << " MB for output buffer: " << alloc_error.what() << "\n";
        return EXIT_FAILURE;
    }

    // ── Timing Loop ──────────────────────────────────────────────────────────
    // Run NUM_TIMING_RUNS independent timed trials and average to reduce
    // measurement variance from OS scheduling and cache cold-start effects.
    vector<double> run_times_seconds(NUM_TIMING_RUNS, 0.0);

    for (int run_index = 0; run_index < NUM_TIMING_RUNS; ++run_index) {

        auto time_start = chrono::high_resolution_clock::now();

        // ── Core Generation Loop ──────────────────────────────────────────
        // Threefry-4x64 produces 4 values per call. We process in batches
        // of 4 for maximum throughput. The counter encodes the absolute
        // position in the sequence — no state to carry between calls.
        //
        // PDC Concept: This is the single-core sequential version.
        // The parallelization insight: any rank/thread can independently
        // compute its values by setting counter = its_start_index.
        // This is what mpi_threefry.cpp exploits with zero communication.

        long long full_batches    = total_numbers_to_generate / 4;
        long long leftover_values = total_numbers_to_generate % 4;
        long long write_index     = 0;

        for (long long batch = 0; batch < full_batches; ++batch) {
            threefry4x64_ctr_t ctr = {{
                static_cast<uint64_t>(batch * 4),  // counter = absolute index
                0ULL, 0ULL, 0ULL
            }};
            threefry4x64_ctr_t out = threefry4x64(ctr, THREEFRY_KEY);

            generated_values[static_cast<size_t>(write_index + 0)] = out.v[0];
            generated_values[static_cast<size_t>(write_index + 1)] = out.v[1];
            generated_values[static_cast<size_t>(write_index + 2)] = out.v[2];
            generated_values[static_cast<size_t>(write_index + 3)] = out.v[3];
            write_index += 4;
        }

        // Handle remainder (0–3 leftover values when N is not a multiple of 4)
        if (leftover_values > 0) {
            threefry4x64_ctr_t ctr = {{
                static_cast<uint64_t>(full_batches * 4),
                0ULL, 0ULL, 0ULL
            }};
            threefry4x64_ctr_t out = threefry4x64(ctr, THREEFRY_KEY);
            for (long long i = 0; i < leftover_values; ++i)
                generated_values[static_cast<size_t>(write_index++)] =
                    out.v[static_cast<size_t>(i)];
        }

        auto time_end = chrono::high_resolution_clock::now();

        run_times_seconds[run_index] =
            chrono::duration<double>(time_end - time_start).count();

        cout << "  Run " << (run_index + 1) << "/" << NUM_TIMING_RUNS
                  << "  →  " << fixed << setprecision(4)
                  << run_times_seconds[run_index] << " s\n";
    }

    // ── Compute Average Time & Throughput ────────────────────────────────────
    double total_time_seconds = 0.0;
    for (int i = 0; i < NUM_TIMING_RUNS; ++i)
        total_time_seconds += run_times_seconds[i];
    double average_elapsed_seconds = total_time_seconds / NUM_TIMING_RUNS;

    // Throughput: N × sizeof(uint64_t) bytes / time
    double throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
        (average_elapsed_seconds * 1.0e9);

    // ── Prevent Dead-Code Elimination ────────────────────────────────────────
    cout << "\n  [Sanity] First generated value : 0x"
              << hex << generated_values[0] << dec << "\n";

    // ── Print Results Table ───────────────────────────────────────────────────
    print_results_table(total_numbers_to_generate,
                        average_elapsed_seconds,
                        throughput_GBps);

    // ── Save Results to CSV ───────────────────────────────────────────────────
    // CSV format: N, time_seconds, throughput_GBps
    // This file is read by mpi_threefry.cpp and cuda_philox.cu to compute
    // speedup relative to THIS single-core Threefry baseline.
    const string results_directory = "results";
    const string csv_file_path     = results_directory + "/baseline_results.csv";

    try {
        filesystem::create_directories(results_directory);
    } catch (const filesystem::filesystem_error& fs_error) {
        cerr << "[WARNING] Could not create results directory: "
                  << fs_error.what() << "\n";
    }

    ofstream csv_output_file(csv_file_path);
    if (!csv_output_file.is_open()) {
        cerr << "[ERROR] Cannot open " << csv_file_path << " for writing.\n";
        return EXIT_FAILURE;
    }

    csv_output_file << "N,time_seconds,throughput_GBps\n";
    csv_output_file << fixed << setprecision(9)
                    << total_numbers_to_generate << ","
                    << average_elapsed_seconds    << ","
                    << throughput_GBps            << "\n";

    csv_output_file.close();
    cout << "  Results saved → " << csv_file_path << "\n\n";

    return EXIT_SUCCESS;
}
