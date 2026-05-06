#!/usr/bin/env python3
"""
File: generate_plots.py
Project: Parallel Random Number Generation
Paper: "Parallel Random Numbers: As Easy as 1,2,3"
        Salmon et al., SC11, 2011
Course: CS-3006 Parallel and Distributed Computing
Purpose: Reads all CSV result files from results/ directory and generates
         six publication-quality plots saved to the plots/ directory.
Run:    python3 generate_plots.py
"""

# ── Standard & Third-Party Imports ───────────────────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Plot Style Configuration ──────────────────────────────────────────────────
# Use seaborn's clean whitegrid style for professional appearance
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "lines.linewidth":    2.0,
    "lines.markersize":   8,
})

# ── Path Constants ────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.dirname(__file__)      # save plots in the plots/ folder
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── CSV Loading Helpers ───────────────────────────────────────────────────────

def load_csv(filename: str) -> pd.DataFrame | None:
    """Load a CSV from the results directory; return None if missing."""
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(file_path):
        print(f"  [WARNING] Not found: {file_path}  — skipping plots that need it.")
        return None
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as exc:
        print(f"  [ERROR] Failed to read {file_path}: {exc}")
        return None


def get_scalar(df: pd.DataFrame, column: str, row: int = 0):
    """Safely retrieve a scalar value from a DataFrame."""
    try:
        return float(df.iloc[row][column])
    except Exception:
        return None

# ── Load All Data ─────────────────────────────────────────────────────────────
print("Loading CSV result files...")
df_baseline  = load_csv("baseline_results.csv")
df_mpi       = load_csv("mpi_results.csv")
df_gpu       = load_csv("gpu_results.csv")
df_hw        = load_csv("hardware_profile.csv")

# ── Derived Quantities ────────────────────────────────────────────────────────

def get_hw_value(hw_df: pd.DataFrame, metric_name: str) -> float | None:
    """Look up a value from the hardware_profile metric→value table."""
    if hw_df is None:
        return None
    mask = hw_df["metric"].str.strip() == metric_name
    if mask.sum() == 0:
        return None
    try:
        return float(hw_df.loc[mask, "value"].values[0])
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Speedup vs MPI Process Count
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] speedup_vs_processes.png")

if df_mpi is not None and "num_processes" in df_mpi.columns:
    df_mpi_sorted = df_mpi.sort_values("num_processes").drop_duplicates("num_processes")
    process_counts  = df_mpi_sorted["num_processes"].tolist()
    actual_speedups = df_mpi_sorted["speedup"].tolist()
    ideal_speedups  = process_counts  # ideal linear speedup = process count

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(process_counts, actual_speedups,
            color="steelblue", marker="o", label="Actual Speedup")
    ax.plot(process_counts, ideal_speedups,
            color="tomato", linestyle="--", marker="s", label="Ideal Linear Speedup")

    # Annotate each actual speedup point
    for xv, yv in zip(process_counts, actual_speedups):
        ax.annotate(f"{yv:.2f}x",
                    xy=(xv, yv),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=10,
                    color="steelblue")

    ax.set_title("MPI Threefry-4x64-20: Speedup vs Process Count")
    ax.set_xlabel("Number of MPI Processes")
    ax.set_ylabel("Speedup (×)")
    ax.set_xticks(process_counts)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "speedup_vs_processes.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   Saved → {save_path}")
else:
    print("   Skipped — no MPI data available.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Parallel Efficiency vs MPI Process Count
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] efficiency_vs_processes.png")

if df_mpi is not None and "num_processes" in df_mpi.columns:
    df_mpi_sorted   = df_mpi.sort_values("num_processes").drop_duplicates("num_processes")
    process_counts  = df_mpi_sorted["num_processes"].tolist()
    efficiencies    = df_mpi_sorted["efficiency"].tolist()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(process_counts, efficiencies,
            color="mediumseagreen", marker="^", label="Parallel Efficiency")
    ax.axhline(y=100.0, color="tomato", linestyle="--", label="Ideal 100% Efficiency")

    for xv, yv in zip(process_counts, efficiencies):
        ax.annotate(f"{yv:.1f}%",
                    xy=(xv, yv),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=10,
                    color="mediumseagreen")

    ax.set_title("MPI Threefry-4x64-20: Parallel Efficiency vs Process Count")
    ax.set_xlabel("Number of MPI Processes")
    ax.set_ylabel("Efficiency (%)")
    ax.set_ylim(0, max(efficiencies) * 1.25 if efficiencies else 120)
    ax.set_xticks(process_counts)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "efficiency_vs_processes.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   Saved → {save_path}")
else:
    print("   Skipped — no MPI data available.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Throughput vs N (log scale X axis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] throughput_vs_N.png")

has_any_data = (df_baseline is not None or df_mpi is not None or df_gpu is not None)

if has_any_data:
    fig, ax = plt.subplots(figsize=(9, 6))

    if df_baseline is not None:
        baseline_N    = df_baseline["N"].values
        baseline_gbps = df_baseline["throughput_GBps"].values
        ax.semilogx(baseline_N, baseline_gbps,
                    color="tomato", marker="o",
                    label="Sequential Mersenne Twister")

    if df_mpi is not None:
        # Use the best (highest throughput) MPI run for each N
        best_mpi = (df_mpi.sort_values("throughput_GBps", ascending=False)
                          .drop_duplicates("N")
                          .sort_values("N"))
        ax.semilogx(best_mpi["N"].values, best_mpi["throughput_GBps"].values,
                    color="steelblue", marker="s",
                    label="MPI Threefry (best process count)")

    if df_gpu is not None:
        ax.semilogx(df_gpu["N"].values, df_gpu["gpu_throughput_GBps"].values,
                    color="mediumseagreen", marker="^",
                    label="GPU Philox CUDA")

    ax.set_title("Throughput vs Input Size N")
    ax.set_xlabel("Input Size N (log scale)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "throughput_vs_N.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   Saved → {save_path}")
else:
    print("   Skipped — no throughput data available.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: CPU MPI vs GPU Throughput (grouped bar chart)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] cpu_vs_gpu_bar.png")

if df_mpi is not None and df_gpu is not None:
    # Use the single GPU result N and find the closest MPI N
    gpu_n = int(df_gpu["N"].iloc[0])
    gpu_gbps = float(df_gpu["gpu_throughput_GBps"].iloc[0])

    # Best MPI throughput at closest N
    df_mpi_at_n = df_mpi[df_mpi["N"] == gpu_n]
    if df_mpi_at_n.empty:
        df_mpi_at_n = df_mpi  # fall back to all MPI data

    mpi_groups = (df_mpi_at_n.sort_values("num_processes")
                              .drop_duplicates("num_processes"))
    labels     = [f"MPI {p}p" for p in mpi_groups["num_processes"].tolist()]
    mpi_gbps   = mpi_groups["throughput_GBps"].tolist()

    x          = np.arange(len(labels))
    bar_width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_mpi = ax.bar(x - bar_width / 2, mpi_gbps, bar_width,
                       label="CPU MPI Threefry", color="steelblue")
    bars_gpu = ax.bar(x + bar_width / 2, [gpu_gbps] * len(labels), bar_width,
                       label="GPU Philox CUDA", color="mediumseagreen")

    # Value labels on top of each bar
    for bar in bars_mpi:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars_gpu:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(f"CPU MPI vs GPU Throughput  (N = {gpu_n:,})")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "cpu_vs_gpu_bar.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   Saved → {save_path}")
else:
    print("   Skipped — need both MPI and GPU data.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Roofline Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] roofline_model.png")

peak_bw    = get_hw_value(df_hw, "peak_copy_bandwidth")
peak_gflops= get_hw_value(df_hw, "peak_compute")
ridge_pt   = get_hw_value(df_hw, "ridge_point")
tf_ai      = get_hw_value(df_hw, "threefry_arithmetic_intensity") or 2.5
mt_ai      = get_hw_value(df_hw, "mt_arithmetic_intensity")       or 0.0024

# Fall back to sensible defaults if hardware profile not available
peak_bw     = peak_bw    or 20.0   # 20 GB/s typical DRAM
peak_gflops = peak_gflops or 50.0  # 50 GFLOPS typical CPU
ridge_pt    = ridge_pt    or (peak_gflops / peak_bw)

fig, ax = plt.subplots(figsize=(10, 7))

# Arithmetic intensity x-axis (log scale)
ai_range = np.logspace(-3, 3, 500)

# Memory bandwidth roof:  GFLOPS = bandwidth * AI  (capped at compute ceiling)
bw_roof    = peak_bw * ai_range
compute_ceil = np.full_like(ai_range, peak_gflops)
roofline   = np.minimum(bw_roof, compute_ceil)

# Shade regions
memory_bound_mask  = ai_range <= ridge_pt
compute_bound_mask = ai_range >= ridge_pt

ax.fill_between(ai_range, roofline, alpha=0.12, color="steelblue",
                where=memory_bound_mask, label="Memory-bound region")
ax.fill_between(ai_range, roofline, alpha=0.12, color="mediumseagreen",
                where=compute_bound_mask, label="Compute-bound region")

# Draw the roofline itself
ax.loglog(ai_range, roofline,
          color="black", linewidth=2.5, label="Roofline boundary")

# Ridge point vertical line
ax.axvline(x=ridge_pt, color="dimgray", linestyle=":", linewidth=1.5,
           label=f"Ridge point = {ridge_pt:.2f} FLOP/byte")

# --- Algorithm points ---
# Threefry MPI achievable performance
threefry_perf = min(peak_gflops, peak_bw * tf_ai)
ax.plot(tf_ai, threefry_perf,
        marker="D", color="steelblue", markersize=12,
        zorder=5, label=f"Threefry MPI  AI={tf_ai:.1f}")
ax.annotate(f"Threefry\n{threefry_perf:.1f} GFLOPS",
            xy=(tf_ai, threefry_perf), xytext=(tf_ai * 2, threefry_perf * 0.7),
            fontsize=10, color="steelblue",
            arrowprops=dict(arrowstyle="->", color="steelblue"))

# GPU Philox — uses 32-bit ops so slightly different AI (≈ 1.25 for Philox-4x32-10)
philox_ai   = 1.25
philox_perf = min(peak_gflops, peak_bw * philox_ai)
ax.plot(philox_ai, philox_perf,
        marker="^", color="mediumseagreen", markersize=12,
        zorder=5, label=f"Philox GPU  AI={philox_ai:.2f}")
ax.annotate(f"Philox GPU\n{philox_perf:.1f} GFLOPS",
            xy=(philox_ai, philox_perf), xytext=(philox_ai * 3, philox_perf * 1.5),
            fontsize=10, color="mediumseagreen",
            arrowprops=dict(arrowstyle="->", color="mediumseagreen"))

# Mersenne Twister
mt_perf = min(peak_gflops, peak_bw * mt_ai)
ax.plot(mt_ai, max(mt_perf, 1e-3),
        marker="o", color="tomato", markersize=12,
        zorder=5, label=f"Mersenne Twister  AI={mt_ai:.4f}")
ax.annotate(f"Mersenne\nTwister",
            xy=(mt_ai, max(mt_perf, 1e-3)),
            xytext=(mt_ai * 8, max(mt_perf, 1e-3) * 3),
            fontsize=10, color="tomato",
            arrowprops=dict(arrowstyle="->", color="tomato"))

ax.set_title("Roofline Model: Counter-based PRNGs vs Mersenne Twister")
ax.set_xlabel("Arithmetic Intensity (FLOP / byte)  [log scale]")
ax.set_ylabel("Performance (GFLOPS)  [log scale]")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roofline_model.png")
fig.savefig(save_path)
plt.close(fig)
print(f"   Saved → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: Combined Speedup & Efficiency (dual Y-axis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] scaling_analysis.png")

if df_mpi is not None and "num_processes" in df_mpi.columns:
    df_mpi_sorted  = df_mpi.sort_values("num_processes").drop_duplicates("num_processes")
    process_counts = df_mpi_sorted["num_processes"].tolist()
    speedups       = df_mpi_sorted["speedup"].tolist()
    efficiencies   = df_mpi_sorted["efficiency"].tolist()
    ideal_su       = process_counts

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    line1, = ax1.plot(process_counts, speedups,
                      color="steelblue", marker="o",
                      label="Actual Speedup")
    line2, = ax1.plot(process_counts, ideal_su,
                      color="steelblue", marker="o", linestyle="--", alpha=0.4,
                      label="Ideal Speedup")
    line3, = ax2.plot(process_counts, efficiencies,
                      color="darkorange", marker="s",
                      label="Efficiency (%)")
    ax2.axhline(y=100.0, color="darkorange", linestyle=":", alpha=0.5)

    ax1.set_title("MPI Threefry: Combined Speedup & Efficiency Scaling")
    ax1.set_xlabel("Number of MPI Processes")
    ax1.set_ylabel("Speedup (×)", color="steelblue")
    ax2.set_ylabel("Efficiency (%)", color="darkorange")
    ax1.set_xticks(process_counts)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    combined_lines  = [line1, line2, line3]
    combined_labels = [l.get_label() for l in combined_lines]
    ax1.legend(combined_lines, combined_labels, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "scaling_analysis.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   Saved → {save_path}")
else:
    print("   Skipped — no MPI data available.")

print("\n✓ All plots generated successfully.")
print(f"  Output directory: {os.path.abspath(PLOTS_DIR)}\n")
