#!/usr/bin/env python3
"""
Plots for the Tesla T4 runs in ../results/gpu_results_t4_sweep.csv.

GPU throughput is kernel time only: the output stays in device memory and the
device-to-host copy is not timed. CPU numbers include writing to host RAM.

Run: python3 generate_gpu_plots.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({"savefig.dpi": 300, "lines.linewidth": 2.0, "lines.markersize": 8})

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")

T4_PEAK_BANDWIDTH_GBps = 320.0  # NVIDIA T4 datasheet, GDDR6

gpu = pd.read_csv(os.path.join(RESULTS_DIR, "gpu_results_t4_sweep.csv")).sort_values("N")
baseline = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
mpi = pd.read_csv(os.path.join(RESULTS_DIR, "mpi_results.csv"))

n_labels = [f"{n / 1e6:.0f}M" for n in gpu["N"]]


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# Same kernel on the same GPU, launched with one thread vs a full grid.
fig, ax = plt.subplots(figsize=(9, 6))
x = range(len(gpu))
w = 0.38
ax.bar([i - w / 2 for i in x], gpu["gpu_baseline_time_ms"], w,
       label="<<<1, 1>>> (one thread)", color="tomato")
ax.bar([i + w / 2 for i in x], gpu["gpu_parallel_time_ms"], w,
       label="Full grid (one thread per Philox call)", color="mediumseagreen")
ax.set_yscale("log")
for i, row in enumerate(gpu.itertuples()):
    ax.text(i, row.gpu_baseline_time_ms * 1.6, f"{row.gpu_speedup_vs_single_thread:,.0f}×",
            ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(n_labels)
ax.set_ylim(top=gpu["gpu_baseline_time_ms"].max() * 5)
ax.set_xlabel("Values generated (N)")
ax.set_ylabel("Kernel time (ms, log scale)")
ax.set_title("Tesla T4 Philox-4x32-10: one thread vs full grid")
ax.legend(loc="upper left")
save(fig, "gpu_t4_single_vs_grid.png")


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(n_labels, gpu["gpu_parallel_throughput_GBps"], marker="o",
        color="mediumseagreen", label="Achieved (kernel only)")
ax.axhline(T4_PEAK_BANDWIDTH_GBps, color="dimgray", linestyle="--",
           label=f"T4 spec memory bandwidth ({T4_PEAK_BANDWIDTH_GBps:.0f} GB/s)")
for label, gbps in zip(n_labels, gpu["gpu_parallel_throughput_GBps"]):
    ax.annotate(f"{gbps:.1f} GB/s\n({gbps / T4_PEAK_BANDWIDTH_GBps:.0%} of spec)",
                xy=(label, gbps), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=10)
ax.set_ylim(0, T4_PEAK_BANDWIDTH_GBps * 1.1)
ax.margins(x=0.12)
ax.set_xlabel("Values generated (N)")
ax.set_ylabel("Output written (GB/s)")
ax.set_title("Tesla T4 Philox throughput vs N")
ax.legend(loc="lower right")
save(fig, "gpu_t4_throughput_vs_N.png")


# Different generators and hardware, so this is a rate of random bytes
# produced, not a speedup of one implementation over another.
N = 300_000_000
t4 = gpu[gpu["N"] == N].iloc[0]
mpi_n = mpi[mpi["N"] == N]
best_mpi = mpi_n.loc[mpi_n["throughput_GBps"].idxmax()]
entries = [
    ("Colab host CPU\n1 core, Threefry", t4["cpu_threefry_throughput_GBps"], "silver"),
    ("Lab CPU\n1 core, Threefry", float(baseline["throughput_GBps"].iloc[0]), "tomato"),
    (f"2-node MPI\n{int(best_mpi['num_processes'])} ranks, Threefry", best_mpi["throughput_GBps"], "steelblue"),
    ("Tesla T4\nPhilox", t4["gpu_parallel_throughput_GBps"], "mediumseagreen"),
]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar([e[0] for e in entries], [e[1] for e in entries], color=[e[2] for e in entries])
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
            f"{bar.get_height():.2f} GB/s", ha="center", fontsize=11)
ax.set_yscale("log")
ax.set_ylim(top=max(e[1] for e in entries) * 4)
ax.set_ylabel("Random output (GB/s, log scale)")
ax.set_title(f"Random bytes produced per second, N = {N:,}")
ax.text(0.01, 0.97,
        "Different machines and generators (64-bit Threefry on CPU, 32-bit Philox on GPU).\n"
        "GPU time is kernel only; output is not copied back to the host.",
        transform=ax.transAxes, va="top", fontsize=9, color="dimgray")
save(fig, "random_bytes_per_second.png")
