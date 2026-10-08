/*
 * Single-core Threefry-4x64-20 baseline. All MPI and GPU speedups are
 * measured against the time this program writes to results/baseline_results.csv.
 *
 * Build: g++ -O2 -std=c++17 -I../random123/include -o sequential_threefry sequential_threefry.cpp
 * Run:   ./sequential_threefry [N]
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <filesystem>

using namespace std;

#include <Random123/threefry.h>

static constexpr long long DEFAULT_N       = 10'000'000LL;
static constexpr int       NUM_TIMING_RUNS = 5;

// mpi_threefry.cpp puts the rank in key word 0 instead of using an all-zero key.
static const threefry4x64_key_t THREEFRY_KEY = {{ 0ULL, 0ULL, 0ULL, 0ULL }};

static string format_with_commas(long long value) {
    string s = to_string(value);
    int pos = static_cast<int>(s.size()) - 3;
    while (pos > 0) {
        s.insert(static_cast<size_t>(pos), ",");
        pos -= 3;
    }
    return s;
}

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

int main(int argc, char* argv[])
{
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

    // Every value is stored so the benchmark pays for memory writes, not just
    // the cipher rounds.
    vector<uint64_t> generated_values;
    try {
        generated_values.resize(static_cast<size_t>(total_numbers_to_generate));
    } catch (const bad_alloc& alloc_error) {
        cerr << "[ERROR] Could not allocate "
                  << total_numbers_to_generate * 8 / (1024 * 1024)
                  << " MB for output buffer: " << alloc_error.what() << "\n";
        return EXIT_FAILURE;
    }

    vector<double> run_times_seconds(NUM_TIMING_RUNS, 0.0);

    for (int run_index = 0; run_index < NUM_TIMING_RUNS; ++run_index) {

        auto time_start = chrono::high_resolution_clock::now();

        // The counter is the absolute output index, so any slice of this loop
        // can be computed independently; mpi_threefry.cpp relies on that.
        long long full_batches    = total_numbers_to_generate / 4;
        long long leftover_values = total_numbers_to_generate % 4;
        long long write_index     = 0;

        for (long long batch = 0; batch < full_batches; ++batch) {
            threefry4x64_ctr_t ctr = {{
                static_cast<uint64_t>(batch * 4),
                0ULL, 0ULL, 0ULL
            }};
            threefry4x64_ctr_t out = threefry4x64(ctr, THREEFRY_KEY);

            generated_values[static_cast<size_t>(write_index + 0)] = out.v[0];
            generated_values[static_cast<size_t>(write_index + 1)] = out.v[1];
            generated_values[static_cast<size_t>(write_index + 2)] = out.v[2];
            generated_values[static_cast<size_t>(write_index + 3)] = out.v[3];
            write_index += 4;
        }

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

    double total_time_seconds = 0.0;
    for (int i = 0; i < NUM_TIMING_RUNS; ++i)
        total_time_seconds += run_times_seconds[i];
    double average_elapsed_seconds = total_time_seconds / NUM_TIMING_RUNS;

    double throughput_GBps =
        (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
        (average_elapsed_seconds * 1.0e9);

    // Reading a generated value keeps the compiler from discarding the loop.
    cout << "\n  [Sanity] First generated value : 0x"
              << hex << generated_values[0] << dec << "\n";

    print_results_table(total_numbers_to_generate,
                        average_elapsed_seconds,
                        throughput_GBps);

    // mpi_threefry and cuda_philox read the time column from this file.
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
