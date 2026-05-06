/*
 * File: sequential_mt.cpp
 * Project: Parallel Random Number Generation
 * Paper: "Parallel Random Numbers: As Easy as 1,2,3"
 *         Salmon et al., SC11, 2011
 * Course: CS-3006 Parallel and Distributed Computing
 * Purpose: Sequential baseline using Mersenne Twister (std::mt19937_64).
 *          Demonstrates the throughput of a traditional PRNG that CANNOT
 *          be efficiently parallelized due to its sequential state machine.
 * Compile: g++ -O2 -std=c++17 -o baseline sequential_mt.cpp
 * Run:     ./baseline [N]
 *          Example: ./baseline 10000000
 */

// ── Standard Library Includes ────────────────────────────────────────────────
// All required headers for I/O, timing, file output, and random generation
#include <iostream>       // std::cout, std::cerr
#include <fstream>        // std::ofstream for CSV output
#include <chrono>         // high_resolution_clock for accurate timing
#include <random>         // std::mt19937_64 Mersenne Twister 64-bit
#include <vector>         // std::vector for storing generated values
#include <string>         // std::string for formatting
#include <cstdint>        // uint64_t
#include <iomanip>        // std::setprecision, std::fixed
#include <numeric>        // std::accumulate
#include <filesystem>     // std::filesystem::create_directories

// ── Using Namespace ───────────────────────────────────────────────────────────
using namespace std;

// ── Constants ────────────────────────────────────────────────────────────────
// Default values used when no command-line arguments are provided
static constexpr long long DEFAULT_N         = 10'000'000LL;  // 10 million values
static constexpr int       NUM_TIMING_RUNS   = 5;             // runs to average over
static constexpr uint64_t  MT_SEED           = 42ULL;         // reproducible seed

// ── Why Mersenne Twister Cannot Be Parallelized ───────────────────────────────
//
// 1. SEQUENTIAL STATE DEPENDENCY
//    MT maintains an internal state array of 624 × 32-bit integers (2496 bytes).
//    Each call to mt19937_64::operator() updates that state in-place.
//    To generate the k-th value you MUST have already generated values 0..k-1
//    because the state at step k is derived from the state at step k-1.
//    This creates an inherent sequential chain: x[k] = f(state[k-1]).
//
// 2. THREAD-UNSAFE BY DESIGN
//    Splitting MT across threads requires either:
//    (a) One generator per thread → each thread has its own 2496-byte state.
//        But then outputs from different threads are correlated because they
//        are all seeded from the same initial state (or nearby states).
//    (b) One shared generator + mutex → serializes all access, no speedup.
//
// 3. CACHE PRESSURE AT SCALE
//    With T threads, total MT state = T × 2496 bytes.
//    At 8 threads: 19,968 bytes ≈ 20 KB — exceeds L1 cache (typically 32 KB).
//    At 32 threads: ~78 KB — exceeds L2 cache on many CPUs.
//    Counter-based PRNGs (Threefry, Philox) have ZERO state — the output is
//    computed directly from (key, counter) with no stored mutable state.
//
// 4. NO SKIP-AHEAD SUPPORT
//    MT does not support efficient skip-ahead (jumping to the k-th value
//    without computing all previous values). Threefry/Philox support this
//    trivially: just set counter = k.
//
// PDC Concept: Amdahl's Law — the sequential state dependency is the
// serial fraction that limits the theoretical maximum parallel speedup.

// ── Helper: Format Large Numbers with Commas ─────────────────────────────────
// Formats a number like 10000000 → "10,000,000" for readable output
static string format_with_commas(long long value) {
    string number_string = to_string(value);
    int         insert_position = static_cast<int>(number_string.size()) - 3;
    while (insert_position > 0) {
        number_string.insert(static_cast<size_t>(insert_position), ",");
        insert_position -= 3;
    }
    return number_string;
}

// ── Helper: Print Formatted Results Table ────────────────────────────────────
// Prints a UTF-8 box-drawing table to stdout, matching project spec
static void print_results_table(long long total_numbers_generated,
                                double    elapsed_seconds,
                                double    throughput_GBps)
{
    cout << "\n┌─────────────────────────────────────────────────┐\n";
    cout <<   "│       Sequential Mersenne Twister Results       │\n";
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
              << "1.00x (baseline)" << "│\n";
    cout <<   "└─────────────────────────────────────────────────┘\n\n";
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Parse Command-Line Arguments ─────────────────────────────────────────
    // Accept optional N as first argument; fall back to DEFAULT_N
    long long total_numbers_to_generate = DEFAULT_N;
    if (argc >= 2) {
        try {
            total_numbers_to_generate = stoll(argv[1]);
            if (total_numbers_to_generate <= 0) {
                cerr << "[ERROR] N must be a positive integer. Got: "
                          << argv[1] << "\n";
                return EXIT_FAILURE;
            }
        } catch (const exception& exception) {
            cerr << "[ERROR] Invalid argument for N: " << argv[1]
                      << " — " << exception.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    cout << "==========================================================\n";
    cout << "  Sequential Mersenne Twister Baseline\n";
    cout << "  CS-3006 Parallel and Distributed Computing\n";
    cout << "==========================================================\n";
    cout << "  N = " << format_with_commas(total_numbers_to_generate)
              << "  |  Runs = " << NUM_TIMING_RUNS << "\n\n";

    // ── Allocate Output Buffer ────────────────────────────────────────────────
    // PDC concept: Memory bandwidth — we must actually store values to memory
    // so the measurement reflects real memory pressure, not just compute.
    // Using uint64_t (8 bytes) matches the 64-bit MT output width.
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
    // Run NUM_TIMING_RUNS independent timed trials and average them.
    // PDC concept: Measurement noise — a single run can be distorted by OS
    // scheduling, cache cold-start, or memory bus contention. Averaging
    // multiple runs gives a stable estimate of steady-state throughput.
    vector<double> run_times_seconds(NUM_TIMING_RUNS, 0.0);

    for (int run_index = 0; run_index < NUM_TIMING_RUNS; ++run_index) {

        // Re-seed the generator before every run so each run is identical.
        // This ensures we measure the same workload each time.
        mt19937_64 mersenne_twister_engine(MT_SEED);

        // Record wall-clock start time with the highest available resolution
        auto time_start = chrono::high_resolution_clock::now();

        // ── Core Generation Loop ──────────────────────────────────────────
        // This is the sequential bottleneck: each call to the engine
        // internally updates 624 × 32-bit words of state (the twist step
        // every 624 values) before returning a tempered output word.
        // There is NO way to vectorize across calls because call k reads
        // the state written by call k-1.
        for (long long value_index = 0;
             value_index < total_numbers_to_generate;
             ++value_index)
        {
            generated_values[static_cast<size_t>(value_index)] =
                mersenne_twister_engine();
        }

        auto time_end = chrono::high_resolution_clock::now();

        // Convert to floating-point seconds for throughput calculation
        run_times_seconds[run_index] =
            chrono::duration<double>(time_end - time_start).count();

        cout << "  Run " << (run_index + 1) << "/" << NUM_TIMING_RUNS
                  << "  →  " << fixed << setprecision(4)
                  << run_times_seconds[run_index] << " s\n";
    }

    // ── Compute Average Time & Throughput ────────────────────────────────────
    // Average over all runs to reduce measurement variance
    double total_time_seconds = 0.0;
    for (int run_index = 0; run_index < NUM_TIMING_RUNS; ++run_index) {
        total_time_seconds += run_times_seconds[run_index];
    }
    double average_elapsed_seconds = total_time_seconds / NUM_TIMING_RUNS;

    // Throughput formula from project specification:
    // bytes transferred = N * sizeof(uint64_t) = N * 8
    // throughput GB/s   = bytes / (time_s * 1e9)
    double throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
        (average_elapsed_seconds * 1.0e9);

    // ── Prevent Dead-Code Elimination ────────────────────────────────────────
    // Print one value so the compiler cannot eliminate the generation loop
    // as dead code (since we never "use" generated_values otherwise).
    cout << "\n  [Sanity] First generated value : 0x"
              << hex << generated_values[0] << dec << "\n";

    // ── Print Results Table ───────────────────────────────────────────────────
    print_results_table(total_numbers_to_generate,
                        average_elapsed_seconds,
                        throughput_GBps);

    // ── Save Results to CSV ───────────────────────────────────────────────────
    // PDC concept: Reproducibility — store raw results so downstream analysis
    // (roofline_analysis.cpp, generate_plots.py) can read them without
    // re-running the experiment.
    //
    // CSV format: N, time_seconds, throughput_GBps
    const string results_directory = "results";
    const string csv_file_path     = results_directory + "/baseline_results.csv";

    // Create results/ directory if it does not already exist
    try {
        filesystem::create_directories(results_directory);
    } catch (const filesystem::filesystem_error& fs_error) {
        cerr << "[WARNING] Could not create results directory: "
                  << fs_error.what() << " — CSV will not be saved.\n";
    }

    ofstream csv_output_file(csv_file_path);
    if (!csv_output_file.is_open()) {
        cerr << "[ERROR] Cannot open " << csv_file_path
                  << " for writing. Check directory permissions.\n";
        return EXIT_FAILURE;
    }

    // Write header row (required by project specification)
    csv_output_file << "N,time_seconds,throughput_GBps\n";

    // Write data row with full precision
    csv_output_file << fixed << setprecision(9)
                    << total_numbers_to_generate << ","
                    << average_elapsed_seconds    << ","
                    << throughput_GBps            << "\n";

    csv_output_file.close();
    cout << "  Results saved → " << csv_file_path << "\n\n";

    return EXIT_SUCCESS;
}
