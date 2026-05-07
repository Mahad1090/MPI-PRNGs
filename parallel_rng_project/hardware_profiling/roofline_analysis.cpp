/*
 * File: roofline_analysis.cpp
 * Project: Parallel Random Number Generation
 * Paper: "Parallel Random Numbers: As Easy as 1,2,3"
 *         Salmon et al., SC11, 2011
 * Course: CS-3006 Parallel and Distributed Computing
 * Purpose: Reads all experiment CSV files, prints a complete performance
 *          summary table, performs roofline and scaling analysis, and
 *          saves a readable final report to results/final_analysis.txt.
 * Compile: g++ -O2 -std=c++17 -o roofline_analysis roofline_analysis.cpp
 * Run:     ./roofline_analysis
 */

// ── Standard Library Includes ─────────────────────────────────────────────────
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <cmath>
#include <algorithm>

// ── Using Namespace ───────────────────────────────────────────────────────────
using namespace std;

// ── Data Structures for CSV Results ───────────────────────────────────────────

struct BaselineResult {
    long long total_N           = 0;
    double    time_seconds      = 0.0;
    double    throughput_GBps   = 0.0;
    bool      loaded            = false;
};

struct MpiResult {
    int       num_processes     = 0;
    long long total_N           = 0;
    double    time_seconds      = 0.0;
    double    throughput_GBps   = 0.0;
    double    speedup           = 0.0;
    double    efficiency        = 0.0;
};

struct GpuResult {
    long long total_N                       = 0;
    long long num_threads                   = 0;
    double    gpu_baseline_time_ms          = 0.0;
    double    gpu_baseline_throughput_GBps  = 0.0;
    double    gpu_parallel_time_ms          = 0.0;
    double    gpu_parallel_throughput_GBps  = 0.0;
    double    speedup                       = 0.0;  // gpu_speedup_vs_single_thread
    double    cpu_throughput_GBps           = 0.0;
    bool      loaded                        = false;
};

struct HardwareProfile {
    double peak_read_bandwidth_GBps   = 0.0;
    double peak_write_bandwidth_GBps  = 0.0;
    double peak_copy_bandwidth_GBps   = 0.0;
    double peak_compute_GFLOPS        = 0.0;
    double ridge_point                = 0.0;
    double threefry_ai                = 2.5;
    double philox_ai                  = 1.0;
    bool   loaded                     = false;
};

// ── Helper: Trim Whitespace ────────────────────────────────────────────────────
static string trim_string(const string& input_string) {
    size_t start_pos = input_string.find_first_not_of(" \t\r\n");
    size_t end_pos   = input_string.find_last_not_of(" \t\r\n");
    if (start_pos == string::npos) return "";
    return input_string.substr(start_pos, end_pos - start_pos + 1);
}

// ── CSV Readers ────────────────────────────────────────────────────────────────

static BaselineResult read_baseline_csv(const string& file_path) {
    BaselineResult result;
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "[WARNING] Cannot open: " << file_path << "\n";
        return result;
    }
    string header_line, data_line;
    getline(file, header_line);
    if (!getline(file, data_line)) return result;

    istringstream stream(data_line);
    string token;
    try {
        getline(stream, token, ','); result.total_N         = stoll(token);
        getline(stream, token, ','); result.time_seconds    = stod(token);
        getline(stream, token, ','); result.throughput_GBps = stod(token);
        result.loaded = true;
    } catch (...) {
        cerr << "[WARNING] Failed to parse " << file_path << "\n";
    }
    return result;
}

static vector<MpiResult> read_mpi_csv(const string& file_path) {
    vector<MpiResult> results;
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "[WARNING] Cannot open: " << file_path << "\n";
        return results;
    }
    string header_line;
    getline(file, header_line);

    string data_line;
    while (getline(file, data_line)) {
        if (trim_string(data_line).empty()) continue;
        istringstream stream(data_line);
        string token;
        MpiResult row;
        try {
            getline(stream, token, ','); row.num_processes   = stoi(token);
            getline(stream, token, ','); row.total_N         = stoll(token);
            getline(stream, token, ','); row.time_seconds    = stod(token);
            getline(stream, token, ','); row.throughput_GBps = stod(token);
            getline(stream, token, ','); row.speedup         = stod(token);
            getline(stream, token, ','); row.efficiency      = stod(token);
            results.push_back(row);
        } catch (...) {
            cerr << "[WARNING] Skipping malformed MPI CSV row.\n";
        }
    }
    return results;
}

static GpuResult read_gpu_csv(const string& file_path) {
    GpuResult result;
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "[WARNING] Cannot open: " << file_path << "\n";
        return result;
    }
    string header_line, data_line;
    getline(file, header_line);
    if (!getline(file, data_line)) return result;

    istringstream stream(data_line);
    string token;
    try {
        getline(stream, token, ','); result.total_N                      = stoll(token);
        getline(stream, token, ','); result.gpu_baseline_time_ms         = stod(token);
        getline(stream, token, ','); result.gpu_baseline_throughput_GBps = stod(token);
        getline(stream, token, ','); result.gpu_parallel_time_ms         = stod(token);
        getline(stream, token, ','); result.gpu_parallel_throughput_GBps = stod(token);
        getline(stream, token, ','); result.speedup                      = stod(token);
        getline(stream, token, ','); result.cpu_throughput_GBps          = stod(token);
        getline(stream, token, ','); result.num_threads                  = stoll(token);
        result.loaded = true;
    } catch (...) {
        cerr << "[WARNING] Failed to parse " << file_path << "\n";
    }
    return result;
}

static HardwareProfile read_hardware_profile_csv(const string& file_path) {
    HardwareProfile profile;
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "[WARNING] Cannot open: " << file_path << "\n";
        return profile;
    }
    string header_line;
    getline(file, header_line);

    string data_line;
    while (getline(file, data_line)) {
        if (trim_string(data_line).empty()) continue;
        istringstream stream(data_line);
        string metric_name, value_token, unit_token;
        getline(stream, metric_name,  ',');
        getline(stream, value_token,  ',');
        getline(stream, unit_token,   ',');

        metric_name = trim_string(metric_name);
        double value = 0.0;
        try { value = stod(trim_string(value_token)); } catch (...) { continue; }

        if (metric_name == "peak_read_bandwidth")           profile.peak_read_bandwidth_GBps  = value;
        else if (metric_name == "peak_write_bandwidth")     profile.peak_write_bandwidth_GBps = value;
        else if (metric_name == "peak_copy_bandwidth")      profile.peak_copy_bandwidth_GBps  = value;
        else if (metric_name == "peak_compute")             profile.peak_compute_GFLOPS       = value;
        else if (metric_name == "ridge_point")              profile.ridge_point               = value;
        else if (metric_name == "threefry_arithmetic_intensity") profile.threefry_ai           = value;
        else if (metric_name == "philox_arithmetic_intensity")  profile.philox_ai               = value;
    }
    profile.loaded = (profile.peak_copy_bandwidth_GBps > 0.0);
    return profile;
}

// ── Helper: Format Number with Commas ─────────────────────────────────────────
static string format_N(long long value) {
    string s = to_string(value);
    int pos = static_cast<int>(s.size()) - 3;
    while (pos > 0) { s.insert(static_cast<size_t>(pos), ","); pos -= 3; }
    return s;
}

// ── Helper: Fixed-Width Column String ─────────────────────────────────────────
static string col(const string& text, int width) {
    if (static_cast<int>(text.size()) >= width)
        return text.substr(0, static_cast<size_t>(width));
    return text + string(static_cast<size_t>(width - static_cast<int>(text.size())), ' ');
}

// ── Print Performance Summary Table ───────────────────────────────────────────
static void print_summary_table(const BaselineResult&         baseline,
                                 const vector<MpiResult>& mpi_results,
                                 const GpuResult&              gpu_result,
                                 ostream&                       output_stream)
{
    output_stream << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    output_stream <<   "║                  PERFORMANCE SUMMARY TABLE                      ║\n";
    output_stream <<   "╠══════════════════════════════════════════════════════════════════╣\n";
    output_stream <<   "║ Implementation     │ N          │ Time(s)  │ GB/s   │ Speedup   ║\n";
    output_stream <<   "╠══════════════════════════════════════════════════════════════════╣\n";

    auto print_row = [&](const string& impl,
                          long long          n_val,
                          double             time_s,
                          double             gbps,
                          double             speedup)
    {
        ostringstream row;
        row << "║ "
            << col(impl, 18) << " │ "
            << col(format_N(n_val), 10) << " │ "
            << col(to_string(time_s).substr(0,8), 8) << " │ "
            << col(to_string(gbps).substr(0,6),  6) << " │ "
            << col(to_string(speedup).substr(0,6) + "x", 9) << " ║\n";
        output_stream << row.str();
    };

    if (baseline.loaded)
        print_row("Single Core TF",
                  baseline.total_N, baseline.time_seconds,
                  baseline.throughput_GBps, 1.0);
    else
        output_stream << "║ Single Core TF     │ (no data)  │          │        │           ║\n";

    for (const auto& mpi : mpi_results) {
        string label = "MPI " + to_string(mpi.num_processes) +
                            (mpi.num_processes == 1 ? " process" : " processes");
        print_row(label, mpi.total_N, mpi.time_seconds,
                  mpi.throughput_GBps, mpi.speedup);
    }

    if (gpu_result.loaded) {
        print_row("GPU Philox 1T",
                  gpu_result.total_N,
                  gpu_result.gpu_baseline_time_ms / 1000.0,
                  gpu_result.gpu_baseline_throughput_GBps, 1.0);
        print_row("GPU Philox Parallel",
                  gpu_result.total_N,
                  gpu_result.gpu_parallel_time_ms / 1000.0,
                  gpu_result.gpu_parallel_throughput_GBps, gpu_result.speedup);
    } else {
        output_stream << "║ GPU Philox 1T      │ (no data)  │          │        │           ║\n";
        output_stream << "║ GPU Philox Parallel│ (no data)  │          │        │           ║\n";
    }

    output_stream << "╚══════════════════════════════════════════════════════════════════╝\n\n";
}

// ── Print Hardware Profiles Comparison ───────────────────────────────────────────
static void print_hardware_profiles(const HardwareProfile& local_hw,
                                     const HardwareProfile& remote_hw,
                                     ostream&               output_stream)
{
    output_stream << "── Hardware Profiles ─────────────────────────────────────────────\n\n";

    auto print_profile = [&](const string& label, const HardwareProfile& hw) {
        output_stream << "  " << label << ":\n";
        if (!hw.loaded) { output_stream << "    [no data]\n\n"; return; }
        output_stream << "    Peak copy BW  : " << fixed << setprecision(2)
                      << hw.peak_copy_bandwidth_GBps << " GB/s\n";
        output_stream << "    Peak compute  : " << fixed << setprecision(2)
                      << hw.peak_compute_GFLOPS << " GFLOPS\n";
        output_stream << "    Ridge point   : " << fixed << setprecision(3)
                      << hw.ridge_point << " FLOP/byte\n";
        output_stream << "    Threefry bound: "
                      << (hw.threefry_ai > hw.ridge_point ? "COMPUTE-BOUND" : "MEMORY-BOUND")
                      << " (AI=" << fixed << setprecision(2) << hw.threefry_ai << ")\n\n";
    };

    print_profile("Local machine",  local_hw);
    print_profile("Remote machine", remote_hw);
}

// ── Print Hardware Utilisation ─────────────────────────────────────────────────
static void print_utilization_analysis(const BaselineResult&    baseline,
                                        const vector<MpiResult>& mpi_results,
                                        const HardwareProfile&   local_hw,
                                        const HardwareProfile&   remote_hw,
                                        ostream&                 output_stream)
{
    output_stream << "── Hardware Utilisation ──────────────────────────────────────────\n\n";

    if (!local_hw.loaded) {
        output_stream << "  [NO DATA] Run bandwidth_test first.\n\n";
        return;
    }

    const double ai               = local_hw.threefry_ai;
    const double local_ceil_GBps  = local_hw.peak_compute_GFLOPS / ai;
    const bool   has_remote       = remote_hw.loaded;
    const double remote_ceil_GBps = has_remote ? remote_hw.peak_compute_GFLOPS / ai : 0.0;

    output_stream << "  Threefry AI = " << fixed << setprecision(2) << ai
                  << " FLOP/byte  (COMPUTE-BOUND: ceiling = peak_GFLOPS / AI)\n";
    output_stream << "    Local  ceiling per rank: " << fixed << setprecision(2)
                  << local_hw.peak_compute_GFLOPS << " / " << ai
                  << " = " << local_ceil_GBps << " GB/s\n";
    if (has_remote)
        output_stream << "    Remote ceiling per rank: " << fixed << setprecision(2)
                      << remote_hw.peak_compute_GFLOPS << " / " << ai
                      << " = " << remote_ceil_GBps << " GB/s\n";
    output_stream << "\n";

    const int wI = 22, wP = 4, wG = 9, wR = 9, wL = 9, wRmt = 9;
    int total_w = wI + wP + wG + wR + wL + (has_remote ? wRmt : 0);
    string sep(total_w, '-');

    output_stream << "  " << left
                  << setw(wI) << "Implementation"
                  << setw(wP) << "P"
                  << setw(wG) << "GB/s"
                  << setw(wR) << "GB/s/rk"
                  << setw(wL) << "%Local";
    if (has_remote) output_stream << setw(wRmt) << "%Remote";
    output_stream << "\n  " << sep << "\n";

    auto util_row = [&](const string& label, int p, double gbps) {
        double per_rank    = (p > 0) ? gbps / static_cast<double>(p) : 0.0;
        double pct_local   = (local_ceil_GBps  > 0) ? (per_rank / local_ceil_GBps)  * 100.0 : 0.0;
        double pct_remote  = (has_remote && remote_ceil_GBps > 0)
                             ? (per_rank / remote_ceil_GBps) * 100.0 : 0.0;
        ostringstream sg, sr, pl, pr;
        sg << fixed << setprecision(2) << gbps;
        sr << fixed << setprecision(2) << per_rank;
        pl << fixed << setprecision(1) << pct_local  << "%";
        if (has_remote) pr << fixed << setprecision(1) << pct_remote << "%";
        output_stream << "  " << left
                      << setw(wI) << label
                      << setw(wP) << p
                      << setw(wG) << sg.str()
                      << setw(wR) << sr.str()
                      << setw(wL) << pl.str();
        if (has_remote) output_stream << setw(wRmt) << pr.str();
        output_stream << "\n";
    };

    if (baseline.loaded) util_row("Single Core TF", 1, baseline.throughput_GBps);
    for (const auto& m : mpi_results) {
        string lbl = "MPI " + to_string(m.num_processes) +
                     (m.num_processes == 1 ? " process" : " processes");
        util_row(lbl, m.num_processes, m.throughput_GBps);
    }

    output_stream << "  " << sep << "\n\n";
    output_stream << "  GB/s/rk = total throughput / process count (per-rank throughput).\n";
    output_stream << "  %Local  = (GB/s/rk × AI) / local peak_compute × 100.\n";
    if (has_remote)
        output_stream << "  %Remote = same formula using remote machine peak_compute.\n";
    output_stream << "  Multi-node MPI rows span both machines; per-rank % is approximate.\n\n";
}

// ── Print Roofline Analysis ────────────────────────────────────────────────────
static void print_roofline_analysis(const HardwareProfile& hw,
                                     ostream&                output_stream)
{
    output_stream << "── Roofline Analysis ─────────────────────────────────────────────\n\n";

    if (!hw.loaded) {
        output_stream << "  [NO DATA] Run bandwidth_test first to generate hardware_profile.csv\n\n";
        return;
    }

    output_stream << "  Peak memory bandwidth (STREAM copy) : "
                  << fixed << setprecision(2)
                  << hw.peak_copy_bandwidth_GBps << " GB/s\n";
    output_stream << "  Peak compute (FMA)                  : "
                  << fixed << setprecision(2)
                  << hw.peak_compute_GFLOPS << " GFLOPS\n";
    output_stream << "  Ridge point                         : "
                  << fixed << setprecision(3)
                  << hw.ridge_point << " FLOP/byte\n\n";

    output_stream << "  Single Core Threefry-4x64-20 (Authors Baseline):\n";
    output_stream << "    Arithmetic Intensity  = "
                  << fixed << setprecision(2)
                  << hw.threefry_ai << " FLOP/byte\n";
    if (hw.threefry_ai > hw.ridge_point) {
        output_stream << "    Status: COMPUTE-BOUND (AI > ridge point)\n";
        output_stream << "    → Threefry is limited by CPU throughput, not memory bandwidth.\n";
        output_stream << "    → Adding more MPI processes increases throughput proportionally.\n";
        output_stream << "    → This is why near-linear MPI speedup is expected.\n";
    } else {
        output_stream << "    Status: MEMORY-BOUND (AI < ridge point)\n";
        output_stream << "    → Approaches peak memory bandwidth ceiling.\n";
        output_stream << "    → Stateless design still gives superior parallel scaling.\n";
    }
    output_stream << "\n";

    output_stream << "  MPI Threefry (Our CPU Contribution):\n";
    output_stream << "    Arithmetic Intensity  = "
                  << fixed << setprecision(2)
                  << hw.threefry_ai << " FLOP/byte (same as single core)\n";
    output_stream << "    MPI adds negligible overhead (one MPI_Reduce of P doubles).\n";
    output_stream << "    → Each rank independently runs Threefry with zero inter-rank comms.\n";
    output_stream << "    → Embarrassingly parallel — near-linear speedup with process count.\n\n";

    output_stream << "  GPU Philox-4x32-10 (Our GPU Contribution):\n";
    output_stream << "    Arithmetic Intensity  = "
                  << fixed << setprecision(2)
                  << hw.philox_ai << " FLOP/byte\n";
    output_stream << "    Status: MEMORY-BOUND on CPU roof, but GPU has far higher BW.\n";
    output_stream << "    → Philox 32-bit multiply is a native single-cycle GPU operation.\n";
    output_stream << "    → Thousands of CUDA threads saturate GPU memory bandwidth.\n\n";
}

// ── Print Scaling Analysis ────────────────────────────────────────────────────
static void print_scaling_analysis(const vector<MpiResult>& mpi_results,
                                    ostream&                   output_stream)
{
    output_stream << "── MPI Scaling Analysis ──────────────────────────────────────────\n\n";

    if (mpi_results.empty()) {
        output_stream << "  [NO DATA] Run mpi_threefry experiments first.\n\n";
        return;
    }

    output_stream << left
                  << "  " << setw(10) << "Processes"
                  << setw(14) << "Actual Speedup"
                  << setw(14) << "Ideal Speedup"
                  << setw(12) << "Efficiency"
                  << "Deviation\n";
    output_stream << "  " << string(60, '-') << "\n";

    for (const auto& mpi_row : mpi_results) {
        double ideal_speedup    = static_cast<double>(mpi_row.num_processes);
        double deviation_pct    = ((mpi_row.speedup - ideal_speedup) / ideal_speedup) * 100.0;
        string deviation_label =
            (abs(deviation_pct) < 5.0)  ? "Near ideal"    :
            (deviation_pct < -20.0)           ? "Significant loss (MPI overhead)" :
            (deviation_pct > 5.0)             ? "Super-linear (cache effects)"    :
                                                "Minor overhead";

        output_stream << "  " << left
                      << setw(10) << mpi_row.num_processes
                      << setw(14) << (to_string(mpi_row.speedup).substr(0,6) + "x")
                      << setw(14) << (to_string(ideal_speedup).substr(0,4)   + "x")
                      << setw(12) << (to_string(mpi_row.efficiency).substr(0,5) + "%")
                      << deviation_label << "\n";
    }
    output_stream << "\n";

    output_stream << "  Interpretation:\n";
    output_stream << "  • Efficiency > 90%: excellent scaling, MPI overhead is minimal.\n";
    output_stream << "  • Efficiency < 70%: excessive synchronization or load imbalance.\n";
    output_stream << "  • Threefry has zero communication during generation — any overhead\n";
    output_stream << "    comes only from MPI_Init, MPI_Barrier, and MPI_Reduce.\n";
    output_stream << "  • Super-linear speedup is possible if per-rank data fits in cache\n";
    output_stream << "    while the full N does not (cache-size effect).\n\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    cout << "==========================================================\n";
    cout << "  Roofline & Performance Summary Analysis\n";
    cout << "  CS-3006 Parallel and Distributed Computing\n";
    cout << "==========================================================\n\n";

    // ── Read All Result CSV Files ─────────────────────────────────────────────
    BaselineResult         baseline      = read_baseline_csv("results/baseline_results.csv");
    vector<MpiResult>      mpi_rows      = read_mpi_csv("results/mpi_results.csv");
    vector<MpiResult>      mpi_remote    = read_mpi_csv("results/mpi_results_remote.csv");
    GpuResult              gpu_result    = read_gpu_csv("results/gpu_results.csv");
    HardwareProfile        hw_local      = read_hardware_profile_csv("results/hardware_profile.csv");
    HardwareProfile        hw_remote     = read_hardware_profile_csv("results/hardware_profile_remote.csv");

    auto sort_mpi = [](vector<MpiResult>& v) {
        sort(v.begin(), v.end(), [](const MpiResult& a, const MpiResult& b) {
            return a.num_processes < b.num_processes;
        });
    };
    sort_mpi(mpi_rows);
    sort_mpi(mpi_remote);

    // ── Print to Terminal ─────────────────────────────────────────────────────
    print_summary_table(baseline, mpi_rows, gpu_result, cout);
    print_hardware_profiles(hw_local, hw_remote, cout);

    cout << "  [Local machine MPI performance]\n";
    print_utilization_analysis(baseline, mpi_rows, hw_local, hw_remote, cout);

    if (!mpi_remote.empty()) {
        cout << "  [Remote machine MPI performance]\n";
        print_utilization_analysis(baseline, mpi_remote, hw_remote, hw_local, cout);
    }

    print_roofline_analysis(hw_local, cout);
    print_scaling_analysis(mpi_rows, cout);

    if (!mpi_remote.empty()) {
        cout << "  [Remote machine scaling]\n";
        print_scaling_analysis(mpi_remote, cout);
    }

    // ── Save Full Analysis to Text File ───────────────────────────────────────
    const string results_directory = "results";
    const string report_file_path  = results_directory + "/final_analysis.txt";

    try { filesystem::create_directories(results_directory); } catch (...) {}

    ofstream report_file(report_file_path);
    if (!report_file.is_open()) {
        cerr << "[ERROR] Cannot open " << report_file_path << " for writing.\n";
        return EXIT_FAILURE;
    }

    report_file << "==========================================================\n";
    report_file << "  FINAL PERFORMANCE ANALYSIS REPORT\n";
    report_file << "  Project: Parallel Random Number Generation\n";
    report_file << "  Paper: Salmon et al. SC11 2011\n";
    report_file << "  Course: CS-3006 Parallel and Distributed Computing\n";
    report_file << "==========================================================\n\n";
    report_file << "  Baseline: Single Core Threefry-4x64-20\n";
    report_file << "  (Authors' implementation from Random123 library)\n";
    report_file << "  Our contributions:\n";
    report_file << "    1. MPI parallelization of Threefry across multiple CPU processes\n";
    report_file << "    2. GPU acceleration using Philox-4x32-10 via CUDA\n";
    report_file << "  Speedup numbers show improvement over the single-core Threefry baseline.\n\n";

    print_summary_table     (baseline, mpi_rows, gpu_result, report_file);
    print_hardware_profiles (hw_local, hw_remote, report_file);

    report_file << "  [Local machine MPI performance]\n";
    print_utilization_analysis(baseline, mpi_rows, hw_local, hw_remote, report_file);

    if (!mpi_remote.empty()) {
        report_file << "  [Remote machine MPI performance]\n";
        print_utilization_analysis(baseline, mpi_remote, hw_remote, hw_local, report_file);
    }

    print_roofline_analysis(hw_local, report_file);
    print_scaling_analysis (mpi_rows,  report_file);

    if (!mpi_remote.empty()) {
        report_file << "  [Remote machine scaling]\n";
        print_scaling_analysis(mpi_remote, report_file);
    }

    report_file << "── Key Conclusions ──────────────────────────────────────────────\n\n";
    report_file << "  1. Counter-based PRNGs (Threefry, Philox) are embarrassingly\n";
    report_file << "     parallel — each rank/thread independently computes its\n";
    report_file << "     output from (key, counter) with zero communication.\n\n";
    report_file << "  2. Single Core Threefry is the correct baseline — it is the\n";
    report_file << "     authors' own implementation. MPI and GPU speedups measure\n";
    report_file << "     pure parallelization benefit, not algorithm differences.\n\n";
    report_file << "  3. Philox is optimal for GPU because it uses 32-bit multiply,\n";
    report_file << "     which maps perfectly to GPU hardware units with zero\n";
    report_file << "     warp divergence (all threads execute identical instructions).\n\n";
    report_file << "  4. MPI Threefry achieves near-linear speedup because the\n";
    report_file << "     parallel fraction is essentially 100% — communication\n";
    report_file << "     cost is only two collective calls (Barrier + Reduce).\n\n";
    report_file << "  Reference: Salmon, Moraes, Dror, Shaw.\n";
    report_file << "  'Parallel Random Numbers: As Easy as 1, 2, 3'\n";
    report_file << "  SC11, November 2011, Seattle, Washington.\n\n";

    report_file.close();
    cout << "  Full report saved → " << report_file_path << "\n\n";

    return EXIT_SUCCESS;
}
