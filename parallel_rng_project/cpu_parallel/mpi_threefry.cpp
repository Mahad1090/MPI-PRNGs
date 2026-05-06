/*
 * File: mpi_threefry.cpp
 * Project: Parallel Random Number Generation
 * Paper: "Parallel Random Numbers: As Easy as 1,2,3"
 *         Salmon et al., SC11, 2011
 * Course: CS-3006 Parallel and Distributed Computing
 * Purpose: Demonstrates embarrassingly parallel RNG using Threefry-4x64-20
 *          counter-based PRNG across multiple MPI processes. Each rank
 *          generates its partition completely independently with zero
 *          inter-rank communication during number generation.
 * Compile: mpic++ -O2 -std=c++17 -I../Random123/include -o mpi_threefry mpi_threefry.cpp
 * Run:     mpirun -np 4 ./mpi_threefry [N] [seed]
 *          Example: mpirun -np 4 ./mpi_threefry 10000000 42
 */

// ── Standard Library Includes ────────────────────────────────────────────────
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

// ── Using Namespace ───────────────────────────────────────────────────────────
using namespace std;

// ── MPI Header ───────────────────────────────────────────────────────────────
// OpenMPI / MPICH C++ bindings
#include <mpi.h>

// ── Random123 Header: Threefry ────────────────────────────────────────────────
// Random123 is a HEADER-ONLY library — no linking step needed.
// Threefry-4x64-20 generates 4 × 64-bit values per call using 20 rounds
// of the Threefish block cipher key schedule. It is the best portable
// choice for CPU because modern CPUs handle 64-bit integer operations
// natively and efficiently.
//
// Paper reference: Section 4.1, Salmon et al., SC11 2011
// "Threefry is based on the Threefish cipher and is particularly well
//  suited for general-purpose CPUs."
#include <Random123/threefry.h>

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr long long DEFAULT_N    = 10'000'000LL;
static constexpr uint64_t  DEFAULT_SEED = 42ULL;

// ── Why Rank Number as Key Guarantees Independence ────────────────────────────
//
// Threefry implements the counter-based PRNG model from the paper:
//   output(n) = bijection( key, counter_n )
//
// where bijection is a cryptographic-quality keyed permutation.
//
// KEY INSIGHT: If two ranks use DIFFERENT keys, even if their counters
// overlap completely (e.g., both use counter 0..N/P), the outputs are
// STATISTICALLY INDEPENDENT because a change in any bit of the key
// propagates through all 20 rounds and produces completely uncorrelated
// output.
//
// We set:  key = (mpi_rank_number, user_seed)
//          counter = [rank_start_offset, rank_start_offset+1, ...]
//
// This guarantees:
// (a) No two ranks ever produce the same output stream.
// (b) No rank needs to know what any other rank generated.
// (c) Any value can be reproduced exactly by any rank at any time.
//
// PDC Concept: Data Parallelism — the same operation (generate one
// random value) is applied to different data (different counter values
// on different ranks) simultaneously.

// ── Multistream vs Substream Distinction ──────────────────────────────────────
//
// MULTISTREAM APPROACH (what we use here):
//   Each rank uses a DIFFERENT KEY but can use the same counter range.
//   The key uniquely identifies the stream. Streams are provably
//   independent because they come from different bijections.
//   Advantage: Any rank can start generating from counter 0 independently.
//   Used in: Our implementation — key = rank number.
//
// SUBSTREAM APPROACH (alternative):
//   All ranks use the SAME KEY but non-overlapping counter ranges.
//   Rank 0: counter 0..(N/P-1)
//   Rank 1: counter N/P..(2N/P-1)
//   etc.
//   Advantage: The full counter sequence is one contiguous block.
//   Disadvantage: Ranks must coordinate to know their counter offset.
//
// We chose multistream because it is simpler, completely stateless,
// and requires absolutely no coordination between ranks.

// ── Helper: Format Large Numbers with Commas ──────────────────────────────────
static string format_with_commas(long long value) {
    string number_string = to_string(value);
    int         insert_position = static_cast<int>(number_string.size()) - 3;
    while (insert_position > 0) {
        number_string.insert(static_cast<size_t>(insert_position), ",");
        insert_position -= 3;
    }
    return number_string;
}

// ── Helper: Read Baseline Time from CSV ──────────────────────────────────────
// Reads results/baseline_results.csv and returns the baseline time in seconds.
// Returns -1.0 if the file cannot be read (speedup will be shown as N/A).
static double read_baseline_time_seconds(const string& csv_path) {
    ifstream csv_input_file(csv_path);
    if (!csv_input_file.is_open()) {
        return -1.0;   // File not found — caller handles this
    }

    string header_line;
    getline(csv_input_file, header_line);   // Skip header

    string data_line;
    if (!getline(csv_input_file, data_line)) {
        return -1.0;   // Empty file
    }

    // Parse CSV: N, time_seconds, throughput_GBps
    istringstream line_stream(data_line);
    string token;
    getline(line_stream, token, ',');   // N — discard
    getline(line_stream, token, ',');   // time_seconds
    try {
        return stod(token);
    } catch (...) {
        return -1.0;
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── MPI Initialization ────────────────────────────────────────────────────
    // MPI_Init must be the first MPI call. It sets up the communication
    // infrastructure and assigns each process a rank within MPI_COMM_WORLD.
    MPI_Init(&argc, &argv);

    int mpi_rank_number     = 0;   // This process's rank (0-indexed)
    int total_mpi_processes = 1;   // Total number of MPI processes launched

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_number);
    MPI_Comm_size(MPI_COMM_WORLD, &total_mpi_processes);

    // ── Parse Command-Line Arguments ──────────────────────────────────────────
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

    // ── Partition Work Across Ranks ───────────────────────────────────────────
    // PDC Concept: Data partitioning — divide the N values evenly among P ranks.
    // Each rank is responsible for generating exactly its own share.
    //
    // Base count: every rank generates at least floor(N / P) values.
    // Remainder:  the last rank absorbs any leftover values when N % P != 0.
    // This avoids communication while ensuring all N values are covered.
    long long base_count_per_rank =
        total_numbers_to_generate / static_cast<long long>(total_mpi_processes);
    long long remainder_values    =
        total_numbers_to_generate % static_cast<long long>(total_mpi_processes);

    // Last rank gets the extra remainder values
    long long this_rank_count =
        (mpi_rank_number == total_mpi_processes - 1)
            ? base_count_per_rank + remainder_values
            : base_count_per_rank;

    // Counter start offset for this rank (using substream-style offset within
    // our multistream design to make the full sequence contiguous if desired)
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

    // ── Allocate Per-Rank Output Buffer ───────────────────────────────────────
    // Each rank allocates only its share of memory. With P ranks this reduces
    // per-process memory from O(N) to O(N/P) — a key scaling advantage.
    vector<uint64_t> local_generated_values;
    try {
        local_generated_values.resize(static_cast<size_t>(this_rank_count));
    } catch (const bad_alloc& alloc_error) {
        cerr << "[ERROR] Rank " << mpi_rank_number
                  << " could not allocate memory: " << alloc_error.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // ── Barrier Before Timing ─────────────────────────────────────────────────
    // Synchronize all ranks so timing starts from the same moment.
    // This is the ONLY synchronization point before number generation.
    // PDC Concept: Barrier synchronization for fair timing comparison.
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Core Generation: EMBARRASSINGLY PARALLEL ──────────────────────────────
    //
    // PDC Concept: Embarrassingly Parallel Computation
    // A problem is "embarrassingly parallel" when it can be split into
    // independent subtasks with ZERO communication between them.
    // This is the ideal case for parallel scaling.
    //
    // Why zero synchronization is needed during generation:
    // Each rank independently computes output(key_rank, counter) for its
    // assigned counter range. Since the bijection is deterministic and
    // depends only on (key, counter), no rank needs information from
    // any other rank. The results are correct by construction.
    //
    // Threefry-4x64 setup:
    //   threefry4x64_key_t  : 4 × 64-bit key words
    //   threefry4x64_ctr_t  : 4 × 64-bit counter words
    //   threefry4x64        : function that maps (key, counter) → 4 × 64-bit output
    //
    // We use word 0 of the key for the rank number and word 1 for the user seed.
    // We use word 0 of the counter for the sequential index.

    // Build the key: rank number + user seed → unique stream per rank
    threefry4x64_key_t threefry_key = {{
        static_cast<uint64_t>(mpi_rank_number),  // unique per rank
        user_seed,                                // user-supplied entropy
        0ULL,                                     // padding
        0ULL                                      // padding
    }};

    // Record start time using high-resolution wall clock
    auto generation_start_time = chrono::high_resolution_clock::now();

    // ── Inner Generation Loop ─────────────────────────────────────────────────
    // Each call to threefry4x64 produces 4 × 64-bit values.
    // We process in batches of 4 to maximize throughput.
    // PDC Concept: SIMD-friendly computation — the 4-wide interface maps
    // naturally to AVX2 256-bit registers (4 × 64 bits).

    long long full_batches        = this_rank_count / 4;
    long long leftover_values     = this_rank_count % 4;
    long long output_write_index  = 0;

    for (long long batch_index = 0; batch_index < full_batches; ++batch_index) {

        // Counter encodes the absolute position in the global sequence.
        // Even though ranks use different keys, using the absolute counter
        // makes it easy to reconstruct any value given (rank, position).
        threefry4x64_ctr_t threefry_counter = {{
            static_cast<uint64_t>(this_rank_counter_start + batch_index * 4),
            0ULL,
            0ULL,
            0ULL
        }};

        // Call Threefry: pure function, no shared mutable state,
        // no communication, completely independent of other ranks/threads
        threefry4x64_ctr_t threefry_output =
            threefry4x64(threefry_counter, threefry_key);

        // Store the 4 output words to the local buffer
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

    // Handle leftover values (0, 1, 2, or 3) when this_rank_count is not
    // a multiple of 4. Generate one final batch and use only what we need.
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

    // Record end time immediately after generation completes
    auto generation_end_time = chrono::high_resolution_clock::now();

    double this_rank_elapsed_seconds =
        chrono::duration<double>(
            generation_end_time - generation_start_time).count();

    // ── Reduce to Wall-Clock Time ─────────────────────────────────────────────
    // PDC Concept: Collective communication — MPI_Reduce gathers one value
    // from every rank and applies an operation (MPI_MAX here).
    //
    // We use MPI_MAX to find the SLOWEST rank because the experiment's
    // wall-clock time is determined by the last rank to finish.
    // Using MPI_MIN or MPI_SUM would under-report or over-report.
    double wall_clock_time_seconds = 0.0;
    MPI_Reduce(&this_rank_elapsed_seconds,   // send buffer
               &wall_clock_time_seconds,     // receive buffer (valid on rank 0)
               1,                            // count
               MPI_DOUBLE,                   // datatype
               MPI_MAX,                      // operation: slowest rank
               0,                            // root rank receives result
               MPI_COMM_WORLD);

    // ── Rank 0: Compute Metrics and Report ────────────────────────────────────
    if (mpi_rank_number == 0) {

        // Throughput over ALL N values (total work / wall time)
        double total_throughput_GBps =
            (static_cast<double>(total_numbers_to_generate) * sizeof(uint64_t)) /
            (wall_clock_time_seconds * 1.0e9);

        // Read sequential baseline for speedup calculation
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

        // Print formatted results table
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

        // ── Save Results to CSV ───────────────────────────────────────────────
        // Append mode so multiple runs (1, 2, 4, 8 processes) all accumulate
        // in the same file, making it easy for the plotting script to read.
        const string results_directory = "results";
        const string csv_file_path     = results_directory + "/mpi_results.csv";

        try {
            filesystem::create_directories(results_directory);
        } catch (...) {}

        // Check if file exists to decide whether to write header
        bool file_exists = ifstream(csv_file_path).good();
        ofstream csv_output_file(csv_file_path, ios::app);

        if (!csv_output_file.is_open()) {
            cerr << "[ERROR] Cannot open " << csv_file_path << " for writing.\n";
        } else {
            if (!file_exists) {
                // Write CSV header only once
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

    // ── Prevent compiler dead-code elimination ────────────────────────────────
    // Use first and last value so generation loop cannot be optimized away
    if (local_generated_values.size() > 0) {
        volatile uint64_t sink =
            local_generated_values[0] ^
            local_generated_values[local_generated_values.size() - 1];
        (void)sink;
    }

    // ── MPI Finalization ──────────────────────────────────────────────────────
    // MPI_Finalize must be the last MPI call. It releases all MPI resources.
    MPI_Finalize();
    return EXIT_SUCCESS;
}
