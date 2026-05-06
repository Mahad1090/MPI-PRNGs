#!/usr/bin/env bash
# ==============================================================================
# File: run_all_experiments.sh
# Project: Parallel Random Number Generation
# Paper: "Parallel Random Numbers: As Easy as 1,2,3"
#         Salmon et al., SC11, 2011
# Course: CS-3006 Parallel and Distributed Computing
# Purpose: Compiles all programs, runs all experiments in the correct order,
#          generates plots, and produces a final summary.
# Run:    chmod +x run_all_experiments.sh && ./run_all_experiments.sh
#
# NOTE (macOS): Replace 'mpirun' with 'mpiexec' if OpenMPI was installed
#               via Homebrew. Also add '--oversubscribe' if process count
#               exceeds physical core count.
# ==============================================================================

# ── Strict Error Handling ─────────────────────────────────────────────────────
# Exit immediately if any command returns a non-zero status
set -e
# Treat unset variables as errors
set -u
# Pipe failures propagate correctly
set -o pipefail

# ── Script Root Directory ─────────────────────────────────────────────────────
# All paths are relative to the directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Results and Log Setup ─────────────────────────────────────────────────────
RESULTS_DIR="${SCRIPT_DIR}/results"
PLOTS_DIR="${SCRIPT_DIR}/plots"
LOG_FILE="${RESULTS_DIR}/experiment_log.txt"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${PLOTS_DIR}"

# Tee: write to both terminal and log file simultaneously
exec > >(tee -a "${LOG_FILE}") 2>&1

# ── Experiment Configuration ──────────────────────────────────────────────────
N=10000000          # Total random numbers per experiment
SEED=42             # Reproducible seed for MPI Threefry

# ── Tracking Arrays ───────────────────────────────────────────────────────────
declare -a EXPERIMENT_NAMES=()
declare -a EXPERIMENT_STATUS=()

pass() { EXPERIMENT_NAMES+=("$1"); EXPERIMENT_STATUS+=("PASSED"); }
fail() { EXPERIMENT_NAMES+=("$1"); EXPERIMENT_STATUS+=("FAILED"); }

# ── Color Codes ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Helper: Section Header ────────────────────────────────────────────────────
section() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════${RESET}"
    echo -e "${CYAN}  $1${RESET}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════${RESET}"
    echo ""
}

# ── Helper: Step Done/Fail ────────────────────────────────────────────────────
done_msg()  { echo -e "  ${GREEN}✓  DONE: $1${RESET}"; }
fail_msg()  { echo -e "  ${RED}✗  FAILED: $1${RESET}"; }
warn_msg()  { echo -e "  ${YELLOW}⚠  WARNING: $1${RESET}"; }
info_msg()  { echo -e "  ${BOLD}→  $1${RESET}"; }

# ── Run a command and track success/failure without stopping the script ────────
# Usage: run_step "label" command [args...]
run_step() {
    local label="$1"
    shift
    echo ""
    info_msg "Running: $*"
    if "$@"; then
        done_msg "${label}"
        pass "${label}"
    else
        fail_msg "${label}"
        fail "${label}"
        # Do NOT exit — continue running remaining experiments
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Parallel Random Number Generation — CS-3006 SP 2026    ║${RESET}"
echo -e "${BOLD}║   Paper: Salmon et al., SC11 2011                        ║${RESET}"
echo -e "${BOLD}║   Implementations: MT (baseline) | Threefry MPI | Philox ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo "  Log file: ${LOG_FILE}"
echo "  Started : $(date)"
echo "  N       : ${N}"

# ── Step 1: Dependency Checks ─────────────────────────────────────────────────
section "STEP 1 — Dependency Checks"

check_dep() {
    local tool="$1"
    local install_hint="$2"
    if command -v "${tool}" &>/dev/null; then
        echo -e "  ${GREEN}✓${RESET}  ${tool} found: $(command -v "${tool}")"
    else
        echo -e "  ${RED}✗${RESET}  ${tool} NOT FOUND. ${install_hint}"
        echo ""
        echo "  Aborting: install missing dependencies then re-run."
        exit 1
    fi
}

check_dep "g++"     "Install: sudo apt install build-essential"
check_dep "mpic++"  "Install: sudo apt install libopenmpi-dev openmpi-bin"
check_dep "python3" "Install: sudo apt install python3 python3-pip"

# nvcc is optional — GPU experiment will be skipped if not present
HAVE_NVCC=false
if command -v nvcc &>/dev/null; then
    echo -e "  ${GREEN}✓${RESET}  nvcc found: $(command -v nvcc)"
    HAVE_NVCC=true
else
    warn_msg "nvcc not found — GPU experiment will be skipped."
fi

# Random123 headers check
RANDOM123_INCLUDE="${SCRIPT_DIR}/random123/include"
if [ -f "${RANDOM123_INCLUDE}/Random123/threefry.h" ]; then
    echo -e "  ${GREEN}✓${RESET}  Random123 headers found at ${RANDOM123_INCLUDE}"
else
    echo -e "  ${RED}✗${RESET}  Random123 headers NOT FOUND at ${RANDOM123_INCLUDE}"
    echo ""
    echo "  Fix: git clone https://github.com/DEShawResearch/random123"
    echo "       (run from ${SCRIPT_DIR})"
    exit 1
fi

echo ""
done_msg "All required dependencies present."

# ── Step 2: Compilation ───────────────────────────────────────────────────────
section "STEP 2 — Compilation"

# Disable set -e during compilation so one failure does not abort everything
set +e

echo "  Compiling baseline/sequential_mt.cpp..."
g++ -O2 -std=c++17 \
    -o "${SCRIPT_DIR}/baseline/baseline" \
    "${SCRIPT_DIR}/baseline/sequential_mt.cpp"
if [ $? -eq 0 ]; then
    done_msg "baseline/sequential_mt.cpp"
    COMPILE_BASELINE=true
else
    fail_msg "baseline/sequential_mt.cpp"
    COMPILE_BASELINE=false
fi

echo "  Compiling cpu_parallel/mpi_threefry.cpp..."
mpic++ -O2 -std=c++17 \
    -I"${RANDOM123_INCLUDE}" \
    -o "${SCRIPT_DIR}/cpu_parallel/mpi_threefry" \
    "${SCRIPT_DIR}/cpu_parallel/mpi_threefry.cpp"
if [ $? -eq 0 ]; then
    done_msg "cpu_parallel/mpi_threefry.cpp"
    COMPILE_MPI=true
else
    fail_msg "cpu_parallel/mpi_threefry.cpp"
    COMPILE_MPI=false
fi

COMPILE_GPU=false
if [ "${HAVE_NVCC}" = true ]; then
    echo "  Compiling gpu_parallel/cuda_philox.cu..."
    nvcc -O2 -std=c++17 \
        -I"${RANDOM123_INCLUDE}" \
        -o "${SCRIPT_DIR}/gpu_parallel/cuda_philox" \
        "${SCRIPT_DIR}/gpu_parallel/cuda_philox.cu"
    if [ $? -eq 0 ]; then
        done_msg "gpu_parallel/cuda_philox.cu"
        COMPILE_GPU=true
    else
        fail_msg "gpu_parallel/cuda_philox.cu"
    fi
else
    warn_msg "Skipping CUDA compilation (nvcc not available)."
fi

echo "  Compiling hardware_profiling/bandwidth_test.cpp..."
g++ -O2 -std=c++17 \
    -o "${SCRIPT_DIR}/hardware_profiling/bandwidth_test" \
    "${SCRIPT_DIR}/hardware_profiling/bandwidth_test.cpp"
if [ $? -eq 0 ]; then
    done_msg "hardware_profiling/bandwidth_test.cpp"
    COMPILE_BW=true
else
    fail_msg "hardware_profiling/bandwidth_test.cpp"
    COMPILE_BW=false
fi

echo "  Compiling hardware_profiling/roofline_analysis.cpp..."
g++ -O2 -std=c++17 \
    -o "${SCRIPT_DIR}/hardware_profiling/roofline_analysis" \
    "${SCRIPT_DIR}/hardware_profiling/roofline_analysis.cpp"
if [ $? -eq 0 ]; then
    done_msg "hardware_profiling/roofline_analysis.cpp"
    COMPILE_ROOF=true
else
    fail_msg "hardware_profiling/roofline_analysis.cpp"
    COMPILE_ROOF=false
fi

# Re-enable strict exit
set -e

# ── Step 3: Experiments ───────────────────────────────────────────────────────
section "STEP 3 — Running Experiments"

# All experiments run from the project root so relative path "results/" resolves
cd "${SCRIPT_DIR}"

# Disable set -e so a failed experiment does not abort the rest
set +e

# --- 3.1 Sequential Baseline ---
if [ "${COMPILE_BASELINE}" = true ]; then
    section "Experiment 1/8 — Sequential Mersenne Twister Baseline"
    run_step "Sequential MT baseline" \
        "${SCRIPT_DIR}/baseline/baseline" "${N}"
else
    warn_msg "Skipping baseline (compilation failed)."
    fail "Sequential MT baseline"
fi

# --- 3.2 MPI Experiments ---
for NP in 1 2 4 8; do
    if [ "${COMPILE_MPI}" = true ]; then
        section "Experiment — MPI Threefry with ${NP} process(es)"
        run_step "MPI Threefry (${NP} processes)" \
            mpirun --oversubscribe -np "${NP}" \
            "${SCRIPT_DIR}/cpu_parallel/mpi_threefry" "${N}" "${SEED}"
    else
        warn_msg "Skipping MPI ${NP}-process run (compilation failed)."
        fail "MPI Threefry (${NP} processes)"
    fi
done

# --- 3.3 GPU Experiment ---
if [ "${COMPILE_GPU}" = true ]; then
    section "Experiment — GPU Philox CUDA"
    run_step "GPU Philox CUDA" \
        "${SCRIPT_DIR}/gpu_parallel/cuda_philox" "${N}"
else
    warn_msg "Skipping GPU experiment (nvcc not available or compilation failed)."
    fail "GPU Philox CUDA"
fi

# --- 3.4 Bandwidth Test ---
if [ "${COMPILE_BW}" = true ]; then
    section "Experiment — Hardware Bandwidth & Compute Profiling"
    run_step "Bandwidth test" \
        "${SCRIPT_DIR}/hardware_profiling/bandwidth_test"
else
    warn_msg "Skipping bandwidth test (compilation failed)."
    fail "Bandwidth test"
fi

# --- 3.5 Roofline Analysis ---
if [ "${COMPILE_ROOF}" = true ]; then
    section "Experiment — Roofline & Performance Summary Analysis"
    run_step "Roofline analysis" \
        "${SCRIPT_DIR}/hardware_profiling/roofline_analysis"
else
    warn_msg "Skipping roofline analysis (compilation failed)."
    fail "Roofline analysis"
fi

# --- 3.6 Python Plots ---
section "Experiment — Generating Plots"
run_step "Python plot generation" \
    python3 "${SCRIPT_DIR}/plots/generate_plots.py"

set -e

# ── Step 4: Final Summary ─────────────────────────────────────────────────────
section "STEP 4 — Final Summary"

echo -e "  ${BOLD}Experiment Results:${RESET}"
echo "  ┌────────────────────────────────────────┬──────────┐"
echo "  │ Experiment                             │ Status   │"
echo "  ├────────────────────────────────────────┼──────────┤"

ALL_PASSED=true
for i in "${!EXPERIMENT_NAMES[@]}"; do
    name="${EXPERIMENT_NAMES[$i]}"
    status="${EXPERIMENT_STATUS[$i]}"
    if [ "${status}" = "PASSED" ]; then
        status_colored="${GREEN}PASSED${RESET}"
    else
        status_colored="${RED}FAILED${RESET}"
        ALL_PASSED=false
    fi
    # Pad name to 38 chars
    printf "  │ %-38s │ " "${name}"
    echo -e "${status_colored}   │"
done

echo "  └────────────────────────────────────────┴──────────┘"
echo ""

echo -e "  ${BOLD}Output Locations:${RESET}"
echo "  ┌──────────────────────────────────────────────────────┐"
echo "  │  CSV Results  → ${RESULTS_DIR}/"
echo "  │  Plot Images  → ${PLOTS_DIR}/"
echo "  │  Full Report  → ${RESULTS_DIR}/final_analysis.txt"
echo "  │  Experiment Log → ${LOG_FILE}"
echo "  └──────────────────────────────────────────────────────┘"
echo ""

echo "  CSV files generated:"
for csv_file in "${RESULTS_DIR}"/*.csv; do
    [ -f "${csv_file}" ] && echo "    • $(basename "${csv_file}")"
done

echo ""
echo "  Plot PNG files generated:"
for png_file in "${PLOTS_DIR}"/*.png; do
    [ -f "${png_file}" ] && echo "    • $(basename "${png_file}")"
done

echo ""
if [ "${ALL_PASSED}" = true ]; then
    echo -e "  ${GREEN}${BOLD}All experiments completed successfully!${RESET}"
else
    echo -e "  ${YELLOW}${BOLD}Some experiments failed — check log: ${LOG_FILE}${RESET}"
fi

echo ""
echo "  Finished: $(date)"
echo ""
