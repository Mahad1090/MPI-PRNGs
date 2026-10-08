/*
 * Threefry-4x64-20 split across MPI ranks. Ranks share nothing while
 * generating; the only communication is a barrier before timing and a
 * single MPI_Reduce of per-rank times afterwards.
 *
 * Build: mpic++ -O2 -std=c++17 -I../random123/include -o mpi_threefry mpi_threefry.cpp
 * Run:   mpirun -np 4 ./mpi_threefry [N] [seed]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <filesystem>

using namespace std;

#include <mpi.h>
#include <Random123/threefry.h>

static constexpr long long DEFAULT_N    = 10'000'000LL;
static constexpr uint64_t  DEFAULT_SEED = 42ULL;

// Stream layout: key = (rank, seed, 0, 0), counter word 0 = global output index.
// The rank in the key already makes every rank's stream distinct, so the
// counter offset is not needed for independence. It is kept so a value can be
// located by its position in the full N-element sequence.

static string format_with_commas(long long value) {
    string number_string = to_string(value);
    int         insert_position = static_cast<int>(number_string.size()) - 3;
    while (insert_position > 0) {
        number_string.insert(static_cast<size_t>(insert_position), ",");
        insert_position -= 3;
    }
    return number_string;
}

// Returns -1.0 when the baseline has not been run yet.
static double read_baseline_time_seconds(const string& csv_path) {
    ifstream csv_input_file(csv_path);
    if (!csv_input_file.is_open()) {
        return -1.0;
    }

    string header_line;
    getline(csv_input_file, header_line);

    string data_line;
    if (!getline(csv_input_file, data_line)) {
        return -1.0;
    }

    istringstream line_stream(data_line);
    string token;
    getline(line_stream, token, ',');   // N
    getline(line_stream, token, ',');   // time_seconds
    try {
        return stod(token);
    } catch (...) {
        return -1.0;
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_rank_number     = 0;
    int total_mpi_processes = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_number);
    MPI_Comm_size(MPI_COMM_WORLD, &total_mpi_processes);

    long long total_numbers_to_generate = DEFAULT_N;
    uint64_t  user_seed                 = DEFAULT_SEED;

    if (argc >= 2) {
        try {
            total_numbers_to_generate = stoll(argv[1]);
            if (total_numbers_to_generate <= 0) {
                if (mpi_rank_number == 0)
                    cerr << "[ERROR] N must be positive. Got: " << argv[1] << "\n";
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        } catch (...) {
            if (mpi_rank_number == 0)
                cerr << "[ERROR] Invalid N argument: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    if (argc >= 3) {
        try {
            user_seed = static_cast<uint64_t>(stoull(argv[2]));
        } catch (...) {
            if (mpi_rank_number == 0)
                cerr << "[WARNING] Invalid seed argument, using default: "
                          << DEFAULT_SEED << "\n";
        }
    }

    // Each rank takes floor(N / P) values; the last rank also takes N % P.
    long long base_count_per_rank =
        total_numbers_to_generate / static_cast<long long>(total_mpi_processes);
    long long remainder_values    =
        total_numbers_to_generate % static_cast<long long>(total_mpi_processes);

    long long this_rank_count =
        (mpi_rank_number == total_mpi_processes - 1)
            ? base_count_per_rank + remainder_values
            : base_count_per_rank;

    long long this_rank_counter_start =
        static_cast<long long>(mpi_rank_number) * base_count_per_rank;

    if (mpi_rank_number == 0) {
        cout << "==========================================================\n";
        cout << "  MPI Threefry-4x64-20 Parallel PRNG\n";
        cout << "  CS-3006 Parallel and Distributed Computing\n";
        cout << "==========================================================\n";
        cout << "  MPI processes : " << total_mpi_processes << "\n";
        cout << "  Total N       : " << format_with_commas(total_numbers_to_generate) << "\n";
        cout << "  Seed          : " << user_seed << "\n\n";
    }

    vector<uint64_t> local_generated_values;
    try {
        local_generated_values.resize(static_cast<size_t>(this_rank_count));
    } catch (const bad_alloc& alloc_error) {
        cerr << "[ERROR] Rank " << mpi_rank_number
                  << " could not allocate memory: " << alloc_error.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    threefry4x64_key_t threefry_key = {{
        static_cast<uint64_t>(mpi_rank_number),
        user_seed,
        0ULL,
        0ULL
    }};

    auto generation_start_time = chrono::high_resolution_clock::now();

    long long full_batches        = this_rank_count / 4;
    long long leftover_values     = this_rank_count % 4;
    long long output_write_index  = 0;

    for (long long batch_index = 0; batch_index < full_batches; ++batch_index) {

        threefry4x64_ctr_t threefry_counter = {{
            static_cast<uint64_t>(this_rank_counter_start + batch_index * 4),
            0ULL,
            0ULL,
            0ULL
        }};

        threefry4x64_ctr_t threefry_output =
            threefry4x64(threefry_counter, threefry_key);

        local_generated_values[static_cast<size_t>(output_write_index + 0)] =
            threefry_output.v[0];
        local_generated_values[static_cast<size_t>(output_write_index + 1)] =
            threefry_output.v[1];
        local_generated_values[static_cast<size_t>(output_write_index + 2)] =
            threefry_output.v[2];
        local_generated_values[static_cast<size_t>(output_write_index + 3)] =
            threefry_output.v[3];

        output_write_index += 4;
    }

    if (leftover_values > 0) {
        threefry4x64_ctr_t threefry_counter = {{
            static_cast<uint64_t>(this_rank_counter_start + full_batches * 4),
            0ULL,
            0ULL,
            0ULL
        }};
        threefry4x64_ctr_t threefry_output =
            threefry4x64(threefry_counter, threefry_key);

        for (long long leftover_index = 0;
             leftover_index < leftover_values;
             ++leftover_index)
        {
            local_generated_values[static_cast<size_t>(output_write_index++)] =
                threefry_output.v[static_cast<size_t>(leftover_index)];
        }
    }

    auto generation_end_time = chrono::high_resolution_clock::now();

    double this_rank_elapsed_seconds =
        chrono::duration<double>(
            generation_end_time - generation_start_time).count();

    // The run is only finished when the slowest rank is, so wall time is the max.
    double wall_clock_time_seconds = 0.0;
    MPI_Reduce(&this_rank_elapsed_seconds,
               &wall_clock_time_seconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    if (mpi_rank_number == 0) {

        double total_throughput_GBps =
            (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
            (wall_clock_time_seconds * 1.0e9);

        const string baseline_csv_path = "results/baseline_results.csv";
        double baseline_elapsed_seconds =
            read_baseline_time_seconds(baseline_csv_path);

        double speedup_vs_baseline  = -1.0;
        double parallel_efficiency  = -1.0;
        bool   baseline_available   = (baseline_elapsed_seconds > 0.0);

        if (baseline_available) {
            speedup_vs_baseline =
                baseline_elapsed_seconds / wall_clock_time_seconds;
            parallel_efficiency =
                (speedup_vs_baseline /
                 static_cast<double>(total_mpi_processes)) * 100.0;
        }

        cout << "\n┌──────────────────────────────────────────────────────┐\n";
        cout <<   "│          MPI Threefry-4x64-20 Results                │\n";
        cout <<   "├──────────────────────────────────────────────────────┤\n";
        cout <<   "│  MPI processes  : " << left << setw(34)
                  << total_mpi_processes << "│\n";
        cout <<   "│  N generated    : " << left << setw(34)
                  << format_with_commas(total_numbers_to_generate) << "│\n";
        cout <<   "│  Wall time      : " << left << setw(34)
                  << (to_string(wall_clock_time_seconds).substr(0,8) + " seconds") << "│\n";
        cout <<   "│  Throughput     : " << left << setw(34)
                  << (to_string(total_throughput_GBps).substr(0,6) + " GB/s") << "│\n";

        if (baseline_available) {
            cout << "│  Speedup        : " << left << setw(34)
                      << (to_string(speedup_vs_baseline).substr(0,6) + "x") << "│\n";
            cout << "│  Efficiency     : " << left << setw(34)
                      << (to_string(parallel_efficiency).substr(0,6) + "%") << "│\n";
        } else {
            cout << "│  Speedup        : " << left << setw(34)
                      << "N/A (run baseline first)" << "│\n";
            cout << "│  Efficiency     : " << left << setw(34)
                      << "N/A" << "│\n";
        }
        cout << "└──────────────────────────────────────────────────────┘\n\n";

        // Appended so a sweep over process counts accumulates in one file.
        const string results_directory = "results";
        const string csv_file_path     = results_directory + "/mpi_results.csv";

        try {
            filesystem::create_directories(results_directory);
        } catch (...) {}

        bool file_exists = ifstream(csv_file_path).good();
        ofstream csv_output_file(csv_file_path, ios::app);

        if (!csv_output_file.is_open()) {
            cerr << "[ERROR] Cannot open " << csv_file_path << " for writing.\n";
        } else {
            if (!file_exists) {
                csv_output_file
                    << "num_processes,N,time_seconds,throughput_GBps,speedup,efficiency\n";
            }
            csv_output_file << fixed << setprecision(9)
                << total_mpi_processes         << ","
                << total_numbers_to_generate   << ","
                << wall_clock_time_seconds     << ","
                << total_throughput_GBps       << ","
                << (baseline_available ? speedup_vs_baseline : 0.0) << ","
                << (baseline_available ? parallel_efficiency : 0.0) << "\n";
            csv_output_file.close();
            cout << "  Results saved → " << csv_file_path << "\n\n";
        }
    }

    // Keeps the compiler from discarding the generation loop.
    if (local_generated_values.size() > 0) {
        volatile uint64_t sink =
            local_generated_values[0] ^
            local_generated_values[local_generated_values.size() - 1];
        (void)sink;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
