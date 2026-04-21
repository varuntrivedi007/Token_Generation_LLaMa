"""
Phase 6: Generate all report figures from merged results.
Professional styling with consistent palette, annotations, and typography.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.merge_results import load_all


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PALETTE = {
    "mps":  "#1f3a5f",   # deep navy
    "cpu":  "#c44536",   # coral red
    "cuda": "#2a9d8f",   # teal
    "wincpu": "#e9c46a", # gold (windows)
}
ACCENT = "#264653"       # dark slate
GRID   = "#d9d9d9"
TEXT   = "#1a1a1a"

DEVICE_LABEL = {"mps": "M4 Pro (MPS)", "cpu": "M4 Pro (CPU)", "cuda": "NVIDIA T4 (CUDA)"}

PLATFORM_LABEL = {
    "m4pro_mps":  "M4 Pro (MPS)",
    "m4pro_cpu":  "M4 Pro (CPU)",
    "colab_t4":   "NVIDIA T4 (CUDA)",
    "windows_cpu": "Windows CPU (i7)",
}

PLATFORM_COLOR = {
    "m4pro_mps":  PALETTE["mps"],
    "m4pro_cpu":  PALETTE["cpu"],
    "colab_t4":   PALETTE["cuda"],
    "windows_cpu": PALETTE["wincpu"],
}

PRECISION_COLOR = {"fp16": "#1f3a5f", "q8": "#2a9d8f", "q4": "#e9c46a"}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "lines.markeredgewidth": 1.0,
        "lines.markeredgecolor": "white",
    })


apply_style()


def color(device):
    return PALETTE.get(device, ACCENT)


def label(device):
    return DEVICE_LABEL.get(device, device)


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")




def fig_ttft_vs_plen(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in sorted(df["device"].unique()):
        d = df[df["device"] == device].sort_values("prompt_length")
        ax.plot(d["prompt_length"], d["ttft_ms_median"],
                marker="o", label=label(device), color=color(device))
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Time to first token (ms)")
    ax.set_title("Prefill Latency — TTFT vs Prompt Length")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(title="Platform", loc="upper left")
    save(fig, "01_ttft_vs_plen.png")


def fig_decode_vs_plen(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in sorted(df["device"].unique()):
        d = df[df["device"] == device].sort_values("prompt_length")
        ax.plot(d["prompt_length"], d["decode_ms_median"],
                marker="s", label=label(device), color=color(device))
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Per-token decode latency (ms)")
    ax.set_title("Decode Latency vs Context Length")
    ax.set_xscale("log", base=2)
    ax.legend(title="Platform")
    save(fig, "02_decode_vs_plen.png")


def fig_throughput(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in sorted(df["device"].unique()):
        d = df[df["device"] == device].sort_values("prompt_length")
        ax.plot(d["prompt_length"], d["throughput_median"],
                marker="^", label=label(device), color=color(device))
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Generation Throughput vs Prompt Length")
    ax.set_xscale("log", base=2)
    ax.legend(title="Platform")
    save(fig, "03_throughput.png")


def fig_platform_summary(df):
    pivot = df.groupby("device").agg(
        TTFT=("ttft_ms_median", "mean"),
        Decode=("decode_ms_median", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot.index))
    w = 0.38
    ax.bar(x - w / 2, pivot["TTFT"], w, label="TTFT (ms)",
           color=[color(d) for d in pivot.index], edgecolor="white")
    ax.bar(x + w / 2, pivot["Decode"], w, label="Decode / token (ms)",
           color=[color(d) for d in pivot.index], edgecolor="white",
           alpha=0.55, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([label(d) for d in pivot.index], rotation=15)
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("Cross-Platform Latency Summary")
    ax.legend()
    save(fig, "04_platform_summary.png")


def fig_decomposition_stacked():
    decomp_files = list(Path("results").rglob("decomp_*.json"))
    if not decomp_files:
        print("[skip] no decomposition results")
        return
    rows = []
    for p in decomp_files:
        with open(p) as f:
            d = json.load(f)
        for comp, summ in d["component_summary_ms"].items():
            rows.append({
                "device": d["device"],
                "prompt_length": d["prompt_length"],
                "component": comp,
                "ms": summ["median"] or 0,
            })
    df = pd.DataFrame(rows)
    
    df = df[~df["component"].isin(["other", "attention_full", "mlp_full"])]

    comp_order = ["embedding", "rmsnorm", "qkv_projection", "attn_output_proj",
                  "mlp_projections", "lm_head", "framework_overhead"]
    comp_colors = ["#264653", "#2a9d8f", "#1f3a5f", "#5a8fb8",
                   "#e9c46a", "#f4a261", "#c44536"]

    for device in df["device"].unique():
        dd = df[df["device"] == device]
        pivot = dd.pivot_table(
            index="prompt_length", columns="component", values="ms", aggfunc="sum"
        ).fillna(0)
        pivot = pivot.reindex(columns=[c for c in comp_order if c in pivot.columns])
        fig, ax = plt.subplots(figsize=(9, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=comp_colors[:pivot.shape[1]], edgecolor="white", width=0.7)
        ax.set_ylabel("Latency (ms, per full generation)")
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_title(f"Latency Decomposition — {label(device)}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=9, title="Component")
        ax.tick_params(axis="x", rotation=0)
        save(fig, f"05_decomp_{device}.png")


def fig_kv_optimization():
    files = list(Path("results").rglob("kvq_*.json"))
    if not files:
        return
    for p in files:
        with open(p) as f:
            d = json.load(f)
        base = d["results"]["baseline"]
        quant = d["results"]["quantized"]
        plens = sorted(int(k) for k in base.keys())
        base_ms = [base[str(k)]["total_ms"]["median"] for k in plens]
        q_ms = [quant[str(k)]["total_ms"]["median"] for k in plens]
        base_kv = [base[str(k)]["kv_bytes_fp16_mean"] / 1e6 for k in plens]
        q_kv = [quant[str(k)]["kv_bytes_int8_mean"] / 1e6 for k in plens]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(plens))
        w = 0.38
        ax1.bar(x - w / 2, base_ms, w, label="FP16 KV (baseline)", color="#1f3a5f", edgecolor="white")
        ax1.bar(x + w / 2, q_ms, w, label="INT8 KV (ours)", color="#2a9d8f", edgecolor="white")
        ax1.set_xticks(x); ax1.set_xticklabels(plens)
        ax1.set_xlabel("Prompt length (tokens)")
        ax1.set_ylabel("Total generation time (ms)")
        ax1.set_title("Latency: FP16 vs INT8 KV-cache")
        ax1.legend()

        ax2.bar(x - w / 2, base_kv, w, label="FP16 KV (baseline)", color="#1f3a5f", edgecolor="white")
        ax2.bar(x + w / 2, q_kv, w, label="INT8 KV (ours)", color="#2a9d8f", edgecolor="white")
        for i, (b, q) in enumerate(zip(base_kv, q_kv)):
            ax2.text(i, max(b, q) * 1.03, f"{b/q:.2f}×", ha="center", fontsize=9, color="#444")
        ax2.set_xticks(x); ax2.set_xticklabels(plens)
        ax2.set_xlabel("Prompt length (tokens)")
        ax2.set_ylabel("KV-cache memory (MB)")
        ax2.set_title("Memory: FP16 vs INT8 KV-cache")
        ax2.legend()

        fig.suptitle(f"KV-Cache Quantization — {label(d['device'])}", fontsize=14, fontweight="bold")
        save(fig, f"06_kvq_{d['device']}.png")


def fig_energy():
    files = list(Path("results").rglob("energy_*.json"))
    if not files:
        return
    rows = []
    for p in files:
        with open(p) as f:
            d = json.load(f)
        rows.append({"device": d["env"]["device"],
                     "prompt_length": d["prompt_length"],
                     "energy_per_token_mJ": d["energy_per_token_mJ"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in df["device"].unique():
        d = df[df["device"] == device].sort_values("prompt_length")
        ax.plot(d["prompt_length"], d["energy_per_token_mJ"],
                marker="D", color=color(device), label=label(device))
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Energy per token (mJ)")
    ax.set_title("Energy Efficiency — mJ per Generated Token")
    ax.legend()
    save(fig, "07_energy.png")


def fig_throughput_vs_bw(df):
    bw = {"mps": 273, "cpu": 100, "cuda": 320}
    agg = df.groupby("device")["throughput_median"].mean().reset_index()
    agg["bandwidth"] = agg["device"].map(bw)
    agg = agg.dropna()
    if len(agg) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in agg.iterrows():
        ax.scatter(r["bandwidth"], r["throughput_median"],
                   s=180, color=color(r["device"]),
                   edgecolor="white", linewidth=1.5, zorder=3,
                   label=label(r["device"]))
        ax.annotate(label(r["device"]),
                    (r["bandwidth"], r["throughput_median"]),
                    xytext=(10, 8), textcoords="offset points",
                    fontsize=10, color=TEXT)

    if len(agg) >= 2:
        slope, intercept = np.polyfit(agg["bandwidth"], agg["throughput_median"], 1)
        xs = np.linspace(agg["bandwidth"].min() * 0.9, agg["bandwidth"].max() * 1.1, 50)
        ys = slope * xs + intercept
        ss_res = np.sum((agg["throughput_median"] - (slope * agg["bandwidth"] + intercept)) ** 2)
        ss_tot = np.sum((agg["throughput_median"] - agg["throughput_median"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.plot(xs, ys, linestyle="--", color=ACCENT, linewidth=1.5,
                label=f"Linear fit (R² = {r2:.3f})")

    ax.set_xlabel("Memory bandwidth (GB/s)")
    ax.set_ylabel("Mean throughput (tokens/sec)")
    ax.set_title("Throughput Scales with Memory Bandwidth")
    ax.legend(loc="lower right")
    save(fig, "08_throughput_vs_bw.png")


def fig_precision_comparison(df):
    """CUDA-only: FP16 vs Q8 vs Q4 throughput (legacy, kept for backwards compat)."""
    d = df[df["device"] == "cuda"]
    if d.empty or d["precision"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for prec in ["fp16", "q8", "q4"]:
        dp = d[d["precision"] == prec].sort_values("prompt_length")
        if dp.empty:
            continue
        ax.plot(dp["prompt_length"], dp["throughput_median"],
                marker="o", label=prec.upper(),
                color=PRECISION_COLOR.get(prec, ACCENT), linewidth=2)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("T4 — Weight-Quantization Precision Comparison")
    ax.set_xscale("log", base=2)
    ax.legend(title="Precision")
    save(fig, "11_cuda_precision.png")


def fig_precision_per_platform(df):
    """Per-platform FP16 vs Q8 vs Q4 throughput (one subplot per platform)."""
    platforms = [p for p in ["m4pro_mps", "m4pro_cpu", "colab_t4", "windows_cpu"]
                 if p in df["platform"].unique()]
    if not platforms:
        return
    n = len(platforms)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, plat in zip(axes, platforms):
        dp = df[df["platform"] == plat]
        for prec in ["fp16", "q8", "q4"]:
            d = dp[dp["precision"] == prec].sort_values("prompt_length")
            if d.empty:
                continue
            ax.plot(d["prompt_length"], d["throughput_median"],
                    marker="o", label=prec.upper(),
                    color=PRECISION_COLOR.get(prec, ACCENT), linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_title(PLATFORM_LABEL.get(plat, plat), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, title="Precision")
    axes[0].set_ylabel("Throughput (tokens / sec)")
    fig.suptitle("Weight-Quantization Throughput Across Platforms",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "12_precision_per_platform.png")


def fig_quant_speedup_heatmap(df):
    """Heatmap of Q4/Q8 speedup over FP16 per platform, averaged over prompt lens."""
    platforms = [p for p in ["m4pro_mps", "m4pro_cpu", "colab_t4", "windows_cpu"]
                 if p in df["platform"].unique()]
    if not platforms:
        return
    precisions = ["fp16", "q8", "q4"]
    matrix = np.full((len(platforms), len(precisions)), np.nan)
    for i, plat in enumerate(platforms):
        dp = df[df["platform"] == plat]
        base = dp[dp["precision"] == "fp16"]["throughput_median"].mean()
        if not np.isfinite(base) or base == 0:
            continue
        for j, prec in enumerate(precisions):
            tp = dp[dp["precision"] == prec]["throughput_median"].mean()
            if np.isfinite(tp):
                matrix[i, j] = tp / base

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.0)
    ax.set_xticks(range(len(precisions)))
    ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_yticks(range(len(platforms)))
    ax.set_yticklabels([PLATFORM_LABEL.get(p, p) for p in platforms])
    for i in range(len(platforms)):
        for j in range(len(precisions)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Throughput ratio vs FP16")
    ax.set_title("Quantization Speedup Grid (tokens/sec, normalized to FP16)")
    save(fig, "13_quant_speedup_heatmap.png")


def main():
    df = load_all()
    if df.empty:
        print("[warn] no benchmark results yet")
    else:
        
        df_fp16 = df[df["precision"] == "fp16"]
        fig_ttft_vs_plen(df_fp16)
        fig_decode_vs_plen(df_fp16)
        fig_throughput(df_fp16)
        fig_platform_summary(df_fp16)
        fig_throughput_vs_bw(df_fp16)
        fig_precision_comparison(df)
        fig_precision_per_platform(df)
        fig_quant_speedup_heatmap(df)
    fig_decomposition_stacked()
    fig_kv_optimization()
    fig_energy()

    from analysis.roofline import plot_roofline, bandwidth_utilization
    plot_roofline(str(FIG_DIR / "09_roofline.png"))
    bandwidth_utilization(out_path=str(FIG_DIR / "10_bw_utilization.png"))


if __name__ == "__main__":
    main()
