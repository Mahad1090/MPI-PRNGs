# Parallel Random Number Generation — CS-3006 Spring 2026

> Demonstrating the parallel scaling benefits of the authors' counter-based PRNGs (Threefry, Philox) from the Random123 library, using a single-core Threefry baseline as the reference point.
---

## Authors

| Name | Role |
|------|------|
| Member 1 | [Mahad Malik] |
| Member 2 | [Rayyan Imran] |

---

## Paper Reference

> Salmon, J.K., Moraes, M.A., Dror, R.O., Shaw, D.E.
> **"Parallel Random Numbers: As Easy as 1, 2, 3"**
> *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC11)*, November 2011, Seattle, Washington, USA.
> DOI: [10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Compilation](#compilation)
6. [Running Experiments](#running-experiments)
7. [Expected Output](#expected-output)
8. [Results Interpretation](#results-interpretation)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What This Project Demonstrates

This project implements and benchmarks three paradigms for generating random numbers in parallel computing:

1. **Sequential Baseline** — Single-core Threefry-4x64-20 (Random123), the **authors' own implementation**, used as the reference point. This is what the paper provides as its starting point.
2. **CPU Parallel** — Same Threefry-4x64-20 wrapped with MPI across multiple processes. Speedup measured against the single-core Threefry baseline. This demonstrates our MPI parallelization contribution.
3. **GPU Parallel** — Philox-4x32-10 (Random123) on NVIDIA GPU via CUDA. Speedup measured against the single-core Threefry baseline. This demonstrates our GPU acceleration contribution.
4. **Hardware Profiling** — Measures memory bandwidth and compute throughput, placing each algorithm on the Roofline Model.

### Why Counter-Based PRNGs Are Better for Parallel Computing

Traditional PRNGs like Mersenne Twister maintain **mutable internal state** (2496 bytes for MT). This creates two fundamental problems for parallel computing:

- **Sequential dependency**: Value `x[k]` depends on `x[k-1]`. You cannot compute the k-th value without first computing all prior values.
- **Cache pressure**: With T threads, total MT state = T × 2496 bytes. At 8 threads this is ~20 KB — already straining L1 cache.

Counter-based PRNGs (Threefry, Philox) implement:

```
output(n) = bijection(key, counter_n)
```

This is a **pure function** with **zero mutable state**. Thread k can directly compute output k by setting `counter = k` — no prior computation required. This is the key insight of Salmon et al. 2011.

### What Are Threefry and Philox?

| Algorithm | Type | Best For | Outputs |
|-----------|------|----------|---------|
| **Threefry-4x64-20** | Counter-based, Threefish cipher | CPU (64-bit integer ops) | 4 × 64-bit per call |
| **Philox-4x32-10** | Counter-based, multiply S-box | GPU (32-bit multiply) | 4 × 32-bit per call |

Both come from the [Random123](https://github.com/DEShawResearch/random123) header-only library.

---

## System Requirements

| Dependency | Minimum Version | Purpose |
|-----------|----------------|---------|
| **g++** | 7.0+ with C++17 | Compile C++ programs |
| **OpenMPI** (`mpicc`, `mpic++`) | 3.0+ | MPI parallel execution |
| **CUDA Toolkit** (`nvcc`) | 11.0+ | GPU kernel compilation |
| **Random123** | Latest (header-only) | Threefry & Philox headers |
| **Python 3** | 3.8+ | Plotting script |
| **numpy** | 1.18+ | Array operations in plots |
| **matplotlib** | 3.3+ | Plot generation |
| **seaborn** | 0.11+ | Plot styling |
| **pandas** | 1.1+ | CSV data loading |

---

## Project Structure

```
parallel_rng_project/
├── baseline/
│   └── sequential_threefry.cpp    # Single-core Threefry-4x64-20 (authors baseline)
├── cpu_parallel/
│   └── mpi_threefry.cpp           # MPI + Threefry-4x64-20
├── gpu_parallel/
│   └── cuda_philox.cu             # CUDA + Philox-4x32-10
├── hardware_profiling/
│   ├── bandwidth_test.cpp         # Memory bandwidth & GFLOPS measurement
│   └── roofline_analysis.cpp      # Reads CSVs, prints summary, saves report
├── results/                       # Generated CSV files go here
│   ├── baseline_results.csv
│   ├── mpi_results.csv
│   ├── gpu_results.csv
│   ├── hardware_profile.csv
│   └── final_analysis.txt
├── plots/                         # Generated PNG plots go here
│   ├── generate_plots.py
│   ├── speedup_vs_processes.png
│   ├── efficiency_vs_processes.png
│   ├── throughput_vs_N.png
│   ├── cpu_vs_gpu_bar.png
│   ├── roofline_model.png
│   └── scaling_analysis.png
├── Random123/                     # Random123 headers (cloned separately)
├── README.md
└── run_all_experiments.sh         # One-command full experiment runner
```

---

## Setup Instructions

### Step 1 — Clone or Download This Project

```bash
git clone <your-repo-url>
cd parallel_rng_project
```

### Step 2 — Download Random123 Headers

Random123 is header-only — no build step needed. Clone it into the project root:

```bash
git clone https://github.com/DEShawResearch/random123
```

After this step you should have `random123/include/Random123/threefry.h` available.

### Step 3 — Install Python Dependencies

```bash
pip install numpy matplotlib seaborn pandas
```

### Step 4 — Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

If CUDA is not installed: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### Step 5 — Verify MPI Installation

```bash
mpirun --version
mpic++ --version
```

If MPI is not installed (Ubuntu):

```bash
sudo apt update && sudo apt install -y libopenmpi-dev openmpi-bin
```

---

## Compilation

Run from the `parallel_rng_project/` directory.

### Sequential Baseline

```bash
cd baseline
g++ -O2 -std=c++17 -I../random123/include -o sequential_threefry sequential_threefry.cpp
cd ..
```

### MPI Threefry

```bash
cd cpu_parallel
mpic++ -O2 -std=c++17 -I../random123/include -o mpi_threefry mpi_threefry.cpp
cd ..
```

### GPU Philox (CUDA)

```bash
cd gpu_parallel
nvcc -O2 -std=c++17 -I../random123/include -o cuda_philox cuda_philox.cu
cd ..
```

### Bandwidth Test

```bash
cd hardware_profiling
g++ -O2 -std=c++17 -o bandwidth_test bandwidth_test.cpp
cd ..
```

### Roofline Analysis

```bash
cd hardware_profiling
g++ -O2 -std=c++17 -o roofline_analysis roofline_analysis.cpp
cd ..
```

---

## Running Experiments

Run experiments **in this exact order** from the `parallel_rng_project/` directory.
Results of earlier experiments are read by later ones (e.g., MPI reads baseline CSV for speedup).

### Step 1 — Sequential Baseline (Authors Implementation)

```bash
cd baseline && ./sequential_threefry 10000000 && cd ..
```

### Step 2 — MPI Experiments

```bash
cd cpu_parallel
mpirun -np 1 ./mpi_threefry 10000000
mpirun -np 2 ./mpi_threefry 10000000
mpirun -np 4 ./mpi_threefry 10000000
mpirun -np 8 ./mpi_threefry 10000000
cd ..
```

### Step 3 — GPU Experiment

```bash
cd gpu_parallel && ./cuda_philox 10000000 && cd ..
```

### Step 4 — Hardware Profiling

```bash
cd hardware_profiling && ./bandwidth_test && cd ..
```

### Step 5 — Summary & Analysis

```bash
cd hardware_profiling && ./roofline_analysis && cd ..
```

### Step 6 — Generate Plots

```bash
cd plots && python3 generate_plots.py && cd ..
```

### Or: Run Everything Automatically

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## Expected Output

### Single Core Threefry Baseline

```
┌─────────────────────────────────────────────────┐
│   Single Core Threefry-4x64-20 Results          │
│   (Authors Baseline — Random123 Library)         │
├─────────────────────────────────────────────────┤
│  N generated  : 10,000,000                      │
│  Timing runs  : 5                               │
│  Avg time     : 0.0420 seconds                  │
│  Throughput   : 1.905 GB/s                      │
│  Speedup      : 1.00x (authors baseline)        │
└─────────────────────────────────────────────────┘
```

### MPI Threefry (4 processes)

```
┌──────────────────────────────────────────────────────┐
│          MPI Threefry-4x64-20 Results                │
├──────────────────────────────────────────────────────┤
│  MPI processes  : 4                                  │
│  N generated    : 10,000,000                         │
│  Wall time      : 0.025000 seconds                   │
│  Throughput     : 3.2000 GB/s                        │
│  Speedup        : 3.80x                              │
│  Efficiency     : 95.00%                             │
└──────────────────────────────────────────────────────┘
```

### GPU Philox

```
┌──────────────────────────────────────────────────────┐
│         GPU Philox-4x32-10 Results                  │
├──────────────────────────────────────────────────────┤
│  GPU device     : NVIDIA GeForce RTX 3060            │
│  CUDA threads   : 10,240                             │
│  Grid dims      : 40 blocks × 256 threads            │
│  N generated    : 10,000,000                         │
│  GPU avg time   : 2.50000 ms                         │
│  GPU throughput : 16.000 GB/s                        │
│  GPU speedup    : 18.5x vs Single Core Threefry      │
└──────────────────────────────────────────────────────┘
```

---

## Results Interpretation

### Speedup Numbers

- **Speedup = single_core_threefry_time / parallel_time**
- All speedup numbers are relative to the single-core Threefry baseline.
- Speedup of `4.0x` with 4 MPI processes = perfect linear scaling.
- Speedup > process count = super-linear (rare, usually cache effects).
- Speedup < process count = overhead from synchronization or load imbalance.

### Efficiency Numbers

- **Efficiency = (speedup / num_processes) × 100%**
- 100% = perfectly efficient (ideal).
- 80–100% = good scaling.
- < 70% = significant overhead.

### Roofline Plot

The roofline model (Plot 5) shows:
- **X-axis**: Arithmetic Intensity (FLOP/byte) — how compute-heavy an algorithm is relative to its memory traffic.
- **Y-axis**: Achievable performance (GFLOPS).
- **Memory bandwidth line** (diagonal): performance limited by DRAM bandwidth.
- **Compute ceiling** (horizontal): performance limited by CPU/GPU throughput.
- **Ridge point**: where the two lines meet. Algorithms to the left are memory-bound; to the right are compute-bound.

| Algorithm | Arithmetic Intensity | Classification |
|-----------|---------------------|----------------|
| Single Core Threefry-4x64-20 | 2.5 FLOP/byte | Compute-bound (on most CPUs) |
| MPI Threefry | 2.5 FLOP/byte | Same as single core (MPI adds negligible overhead) |
| GPU Philox-4x32-10 | 1.0 FLOP/byte | Near ridge point (GPU bandwidth compensates) |

### Compute-Bound vs Memory-Bound

- **Compute-bound**: Adding more cores improves performance linearly (great for MPI scaling).
- **Memory-bound**: Adding more cores fights over the same memory bus (limited scaling).

Threefry sits in the compute-bound regime → it scales near-linearly with MPI process count. MPI adds arithmetic intensity while keeping the same AI, so throughput increases proportionally with process count.

### Why Threefry Scales (Embarrassingly Parallel)

Each MPI rank uses its rank number as a unique cryptographic key. Because the bijection is a pure function of (key, counter), **no rank needs data from any other rank during generation**. The parallel fraction approaches 100%, meaning Amdahl's Law predicts near-linear speedup.

### Role of Mersenne Twister in This Project

Mersenne Twister is **NOT** used as a baseline in our experiments. It is mentioned only as background context to explain why counter-based PRNGs were needed. The paper shows MT fails BigCrush statistical tests and cannot be parallelized effectively (2496-byte sequential state, no skip-ahead). Our project focuses on demonstrating the parallel scaling benefits of the authors' counter-based approach.

---

## Troubleshooting

### CUDA Not Found

```
error: nvcc: not found
```

**Fix**: Install CUDA Toolkit from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and add to PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### MPI Not Found

```
error: mpic++: not found
```

**Fix** (Ubuntu/Debian):

```bash
sudo apt install -y libopenmpi-dev openmpi-bin
```

### Random123 Headers Not Found

```
fatal error: Random123/threefry.h: No such file or directory
```

**Fix**: Ensure you cloned Random123 into the project root:

```bash
git clone https://github.com/DEShawResearch/random123
```

Then verify:

```bash
ls random123/include/Random123/threefry.h   # should exist
```

The compile command uses `-I../random123/include` — adjust path if you cloned elsewhere.

### Permission Error on Shell Script

```
bash: ./run_all_experiments.sh: Permission denied
```

**Fix**:

```bash
chmod +x run_all_experiments.sh
```

### MPI Oversubscription Warning

If you run more processes than CPU cores:

```bash
mpirun --oversubscribe -np 8 ./mpi_threefry 10000000
```

### macOS MPI Note

On macOS, replace `mpirun` with `mpiexec` and you may need:

```bash
brew install open-mpi
```
