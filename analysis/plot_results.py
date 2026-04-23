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
    "mps":    "#3d405b",   
    "cpu":    "#e07a5f",   
    "cuda":   "#81b29a",   
    "wincpu": "#f2cc8f",   
    "wingpu": "#9c6ade",   
}
ACCENT    = "#2d3047"     
ACCENT2   = "#6c757d"     

AXISCOLOR = "#5a5f6a"
TEXT      = "#1e2029"
BG        = "#fdfdfb"     
PANEL_BG  = "#f6f5f0"     

DEVICE_LABEL = {"mps": "M4 Pro (MPS)", "cpu": "M4 Pro (CPU)", "cuda": "NVIDIA T4 (CUDA)"}

PLATFORM_LABEL = {
    "m4pro_mps":   "M4 Pro (MPS)",
    "m4pro_cpu":   "M4 Pro (CPU)",
    "colab_t4":    "NVIDIA T4 (CUDA)",
    "windows_cpu": "Windows CPU (i7-11370H)",
    "windows_3070": "RTX 3070 Laptop",
}

PLATFORM_COLOR = {
    "m4pro_mps":   PALETTE["mps"],
    "m4pro_cpu":   PALETTE["cpu"],
    "colab_t4":    PALETTE["cuda"],
    "windows_cpu": PALETTE["wincpu"],
    "windows_3070": PALETTE["wingpu"],
}

PLATFORM_ORDER = ["m4pro_mps", "m4pro_cpu", "colab_t4", "windows_3070", "windows_cpu"]

PRECISION_COLOR = {
    "fp16": "#3d405b",   
    "q8":   "#81b29a",   
    "q4":   "#e07a5f",   
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi":       160,
        "figure.facecolor": BG,
        "savefig.dpi":      220,
        "savefig.bbox":     "tight",
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
        "font.family":      "serif",
        "font.serif":       ["Palatino", "Palatino Linotype", "Book Antiqua",
                             "Times New Roman", "DejaVu Serif"],
        "font.size":         11,
        "axes.facecolor":    BG,
        "axes.titlesize":    14,
        "axes.titleweight":  "semibold",
        "axes.titlepad":     14,
        "axes.titlecolor":   TEXT,
        "axes.labelsize":    11,
        "axes.labelweight":  "normal",
        "axes.labelcolor":   AXISCOLOR,
        "axes.labelpad":     8,
        "axes.edgecolor":    AXISCOLOR,
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom": True,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "axes.prop_cycle":   mpl.cycler(color=[PALETTE[k] for k in
                                               ["mps","cpu","cuda","wingpu","wincpu"]]),
        "grid.color":        GRID,
        "grid.linewidth":    0.7,
        "grid.linestyle":    "-",
        "grid.alpha":        1.0,
        "xtick.color":       AXISCOLOR,
        "ytick.color":       AXISCOLOR,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#dadde2",
        "legend.facecolor":  BG,
        "legend.fancybox":   True,
        "legend.fontsize":   9.5,
        "legend.title_fontsize": 10,
        "legend.borderpad":  0.8,
        "legend.columnspacing": 1.2,
        "lines.linewidth":   2.2,
        "lines.markersize":  7.5,
        "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": BG,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.edgecolor":   BG,
        "patch.linewidth":   0.8,
        "hatch.linewidth":   0.6,
    })


def ethereal_title(ax, title, subtitle=None):
    
    if subtitle:
        ax.set_title(title, loc="left", fontsize=14, fontweight="semibold",
                     color=TEXT, pad=24)
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=10, color=ACCENT2, ha="left", va="bottom",
                style="italic")
    else:
        ax.set_title(title, loc="left", fontsize=14, fontweight="semibold",
                     color=TEXT, pad=14)


def soft_spines(ax):
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXISCOLOR)
        ax.spines[spine].set_linewidth(0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


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




def _platforms_in_order(df):
    present = set(df["platform"].unique())
    return [p for p in PLATFORM_ORDER if p in present]


def _plot_lines(ax, df, ycol, marker="o"):
    for plat in _platforms_in_order(df):
        d = df[df["platform"] == plat].sort_values("prompt_length")
        col = PLATFORM_COLOR.get(plat, ACCENT)
        # glow halo
        ax.plot(d["prompt_length"], d[ycol], color=col, linewidth=6.0,
                alpha=0.10, solid_capstyle="round", zorder=2)
        # main line
        ax.plot(d["prompt_length"], d[ycol], color=col, linewidth=2.2,
                zorder=3)
        # markers
        ax.plot(d["prompt_length"], d[ycol], marker=marker, linestyle="none",
                markerfacecolor=col, markeredgecolor=BG, markeredgewidth=1.4,
                markersize=8, label=PLATFORM_LABEL.get(plat, plat), zorder=4)


def fig_ttft_vs_plen(df):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _plot_lines(ax, df, "ttft_ms_median", marker="o")
    ax.set_xlabel("Prompt length  (tokens)")
    ax.set_ylabel("Time to first token  (ms)")
    ethereal_title(ax, "Prefill Latency",
                   "TTFT scales with prompt length across all platforms (FP16)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(title="Platform", loc="upper left", frameon=True)
    soft_spines(ax)
    save(fig, "01_ttft_vs_plen.png")


def fig_decode_vs_plen(df):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _plot_lines(ax, df, "decode_ms_median", marker="s")
    ax.set_xlabel("Context length  (tokens)")
    ax.set_ylabel("Per-token decode latency  (ms)")
    ethereal_title(ax, "Decode Latency vs Context",
                   "Per-token cost grows with KV-cache size (FP16)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(title="Platform", loc="upper left", frameon=True)
    soft_spines(ax)
    save(fig, "02_decode_vs_plen.png")


def fig_throughput(df):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _plot_lines(ax, df, "throughput_median", marker="^")
    ax.set_xlabel("Prompt length  (tokens)")
    ax.set_ylabel("Throughput  (tokens / sec)")
    ethereal_title(ax, "Generation Throughput",
                   "End-to-end decode throughput, 128-token generation (FP16)")
    ax.set_xscale("log", base=2)
    ax.legend(title="Platform", loc="upper right", frameon=True)
    soft_spines(ax)
    save(fig, "03_throughput.png")


def fig_platform_summary(df):
    plats = _platforms_in_order(df)
    if not plats:
        return
    means = df.groupby("platform").agg(
        TTFT=("ttft_ms_median", "mean"),
        Decode=("decode_ms_median", "mean"),
    ).reindex(plats)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(plats))
    w = 0.34
    cols = [PLATFORM_COLOR.get(p, ACCENT) for p in plats]

    bars1 = ax.bar(x - w / 2 - 0.02, means["TTFT"], w, label="TTFT  (ms)",
                   color=cols, edgecolor=BG, linewidth=1.2, zorder=3)
    bars2 = ax.bar(x + w / 2 + 0.02, means["Decode"], w, label="Decode / token  (ms)",
                   color=cols, edgecolor=BG, linewidth=1.2, alpha=0.55, zorder=3)

    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08,
                f"{b.get_height():.1f}", ha="center", va="bottom",
                fontsize=8.5, color=AXISCOLOR)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08,
                f"{b.get_height():.1f}", ha="center", va="bottom",
                fontsize=8.5, color=AXISCOLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([PLATFORM_LABEL.get(p, p) for p in plats],
                       rotation=18, ha="right", fontsize=9.5)
    ax.set_ylabel("Latency  (ms, log scale)")
    ax.set_yscale("log")
    top = float(max(means["TTFT"].max(), means["Decode"].max()))
    bot = float(min(means["TTFT"].min(), means["Decode"].min()))
    ax.set_ylim(max(1, bot * 0.5), top * 3.5)
    ax.yaxis.grid(True, which="both", color=GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ethereal_title(ax, "Cross-Platform Latency Profile",
                   "Prefill dominates for short prompts; decode dominates over longer runs (FP16)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=True, borderaxespad=0)
    soft_spines(ax)
    save(fig, "04_platform_summary.png")


DECOMP_DIR_TO_PLATFORM = {
    "decomposition_mps":      "m4pro_mps",
    "decomposition_cpu":      "m4pro_cpu",
    "decomposition_cuda":     "colab_t4",
    "decomposition_cpu_win":  "windows_cpu",
    "decomposition_cuda_win": "windows_3070",
}


def fig_decomposition_stacked():
    decomp_files = list(Path("results").rglob("decomp_*.json"))
    if not decomp_files:
        print("[skip] no decomposition results")
        return
    rows = []
    for p in decomp_files:
        with open(p) as f:
            d = json.load(f)
        plat = DECOMP_DIR_TO_PLATFORM.get(p.parent.name, p.parent.name)
        for comp, summ in d["component_summary_ms"].items():
            rows.append({
                "platform": plat,
                "device": d["device"],
                "prompt_length": d["prompt_length"],
                "component": comp,
                "ms": summ["median"] or 0,
            })
    df = pd.DataFrame(rows)
    df = df[~df["component"].isin(["other", "attention_full", "mlp_full"])]

    comp_order = ["embedding", "rmsnorm", "qkv_projection", "attn_output_proj",
                  "mlp_projections", "lm_head", "framework_overhead"]
    comp_labels = {
        "embedding": "Embedding",
        "rmsnorm": "RMSNorm",
        "qkv_projection": "QKV projection",
        "attn_output_proj": "Attention output",
        "mlp_projections": "MLP",
        "lm_head": "LM head",
        "framework_overhead": "Framework overhead",
    }
    
    comp_colors = ["#3d405b", "#5b6b94", "#81b29a", "#bcd8c1",
                   "#f2cc8f", "#e07a5f", "#c44d3b"]

    for plat in df["platform"].unique():
        dd = df[df["platform"] == plat]
        pivot = dd.pivot_table(
            index="prompt_length", columns="component", values="ms", aggfunc="sum"
        ).fillna(0)
        pivot = pivot.reindex(columns=[c for c in comp_order if c in pivot.columns])
        pivot.columns = [comp_labels.get(c, c) for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=comp_colors[:pivot.shape[1]],
                   edgecolor=BG, linewidth=1.4, width=0.68, zorder=3)

        
        totals = pivot.sum(axis=1)
        for i, t in enumerate(totals):
            ax.text(i, t * 1.02, f"{t:.0f} ms", ha="center", va="bottom",
                    fontsize=9, color=AXISCOLOR, fontweight="semibold")

        ax.set_ylabel("Latency  (ms, per full generation)")
        ax.set_xlabel("Prompt length  (tokens)")
        ethereal_title(
            ax,
            "Latency Decomposition",
            f"Per-component time, {PLATFORM_LABEL.get(plat, plat)} (FP16)",
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  title="Component", frameon=True)
        ax.tick_params(axis="x", rotation=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.xaxis.grid(False)
        soft_spines(ax)
        save(fig, f"05_decomp_{plat}.png")


OPT_DIR_TO_PLATFORM = {
    "optimization_mps":      "m4pro_mps",
    "optimization_cpu":      "m4pro_cpu",
    "optimization_cuda":     "colab_t4",
    "optimization_cpu_win":  "windows_cpu",
    "optimization_cuda_win": "windows_3070",
}


def fig_kv_optimization():
    files = list(Path("results").rglob("kvq_*.json"))
    if not files:
        return
    for p in files:
        with open(p) as f:
            d = json.load(f)
        plat = OPT_DIR_TO_PLATFORM.get(p.parent.name, p.parent.name)
        base = d["results"]["baseline"]
        quant = d["results"]["quantized"]
        plens = sorted(int(k) for k in base.keys())
        base_ms = [base[str(k)]["total_ms"]["median"] for k in plens]
        q_ms = [quant[str(k)]["total_ms"]["median"] for k in plens]
        base_kv = [base[str(k)]["kv_bytes_fp16_mean"] / 1e6 for k in plens]
        q_kv = [quant[str(k)]["kv_bytes_int8_mean"] / 1e6 for k in plens]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
        x = np.arange(len(plens))
        w = 0.36
        fp16_col = "#3d405b"
        int8_col = "#81b29a"

        ax1.bar(x - w / 2 - 0.02, base_ms, w, label="FP16 KV  (baseline)",
                color=fp16_col, edgecolor=BG, linewidth=1.2, zorder=3)
        ax1.bar(x + w / 2 + 0.02, q_ms, w, label="INT8 KV  (ours)",
                color=int8_col, edgecolor=BG, linewidth=1.2, zorder=3)
        ax1.set_xticks(x); ax1.set_xticklabels(plens)
        ax1.set_xlabel("Prompt length  (tokens)")
        ax1.set_ylabel("Total generation time  (ms)")
        ethereal_title(ax1, "Latency", "FP16 vs INT8 KV-cache")
        ax1.legend(loc="upper left", frameon=True)
        ax1.yaxis.grid(True, color=GRID, linewidth=0.5); ax1.xaxis.grid(False)
        soft_spines(ax1)

        ax2.bar(x - w / 2 - 0.02, base_kv, w, label="FP16 KV  (baseline)",
                color=fp16_col, edgecolor=BG, linewidth=1.2, zorder=3)
        ax2.bar(x + w / 2 + 0.02, q_kv, w, label="INT8 KV  (ours)",
                color=int8_col, edgecolor=BG, linewidth=1.2, zorder=3)
        for i, (b, q) in enumerate(zip(base_kv, q_kv)):
            ax2.text(i, max(b, q) * 1.04, f"{b/q:.2f}×", ha="center",
                     fontsize=9.5, color=TEXT, fontweight="semibold")
        ax2.set_xticks(x); ax2.set_xticklabels(plens)
        ax2.set_xlabel("Prompt length  (tokens)")
        ax2.set_ylabel("KV-cache memory  (MB)")
        ethereal_title(ax2, "Memory", "KV-cache footprint reduction")
        ax2.legend(loc="upper left", frameon=True)
        ax2.yaxis.grid(True, color=GRID, linewidth=0.5); ax2.xaxis.grid(False)
        soft_spines(ax2)

        fig.suptitle(f"KV-Cache Quantization  ·  {PLATFORM_LABEL.get(plat, plat)}",
                     fontsize=15, fontweight="semibold", color=TEXT, y=1.02)
        save(fig, f"06_kvq_{plat}.png")


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
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for device in df["device"].unique():
        d = df[df["device"] == device].sort_values("prompt_length")
        col = color(device)
        ax.plot(d["prompt_length"], d["energy_per_token_mJ"],
                color=col, linewidth=6.0, alpha=0.10, zorder=2)
        ax.plot(d["prompt_length"], d["energy_per_token_mJ"],
                color=col, linewidth=2.2, zorder=3)
        ax.plot(d["prompt_length"], d["energy_per_token_mJ"],
                marker="D", linestyle="none",
                markerfacecolor=col, markeredgecolor=BG, markeredgewidth=1.3,
                markersize=7.5, label=label(device), zorder=4)
    ax.set_xlabel("Prompt length  (tokens)")
    ax.set_ylabel("Energy per token  (mJ)")
    ethereal_title(ax, "Energy Efficiency",
                   "Joules-per-token across platforms")
    ax.legend(title="Platform", loc="best", frameon=True)
    soft_spines(ax)
    save(fig, "07_energy.png")


PLATFORM_BW_GBPS = {
    "m4pro_mps":    273,
    "m4pro_cpu":    100,
    "colab_t4":     320,
    "windows_cpu":   50,
    "windows_3070": 448,
}


CLEAN_FIT_PLATFORMS = {"m4pro_mps", "m4pro_cpu", "colab_t4"}


def fig_throughput_vs_bw(df):
    agg = df.groupby("platform")["throughput_median"].mean().reset_index()
    agg["bandwidth"] = agg["platform"].map(PLATFORM_BW_GBPS)
    agg = agg.dropna()
    if len(agg) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    
    for _, r in agg.iterrows():
        col = PLATFORM_COLOR.get(r["platform"], ACCENT)
        ax.scatter(r["bandwidth"], r["throughput_median"],
                   s=520, color=col, alpha=0.15, zorder=2, edgecolor="none")
        ax.scatter(r["bandwidth"], r["throughput_median"],
                   s=220, color=col, edgecolor=BG, linewidth=1.8, zorder=4,
                   label=PLATFORM_LABEL.get(r["platform"], r["platform"]))

    offsets = {
        "m4pro_mps":    (12, 10),
        "m4pro_cpu":    (12, 10),
        "colab_t4":     (12, 10),
        "windows_3070": (-18, -22),
        "windows_cpu":  (12, 10),
    }
    for _, r in agg.iterrows():
        dx, dy = offsets.get(r["platform"], (12, 10))
        ax.annotate(PLATFORM_LABEL.get(r["platform"], r["platform"]),
                    (r["bandwidth"], r["throughput_median"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, color=TEXT, fontweight="semibold")

    clean = agg[agg["platform"].isin(CLEAN_FIT_PLATFORMS)]
    if len(clean) >= 2:
        slope, intercept = np.polyfit(clean["bandwidth"], clean["throughput_median"], 1)
        xs = np.linspace(agg["bandwidth"].min() * 0.9, agg["bandwidth"].max() * 1.1, 120)
        ys = slope * xs + intercept
        ss_res = np.sum((clean["throughput_median"] - (slope * clean["bandwidth"] + intercept)) ** 2)
        ss_tot = np.sum((clean["throughput_median"] - clean["throughput_median"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.plot(xs, ys, linestyle=(0, (4, 3)), color=ACCENT, linewidth=1.6,
                zorder=1, alpha=0.7,
                label=f"BW-bound fit  ·  R² = {r2:.3f}")

    win_cpu = agg[agg["platform"] == "windows_cpu"]
    if not win_cpu.empty:
        r = win_cpu.iloc[0]
        ax.annotate("PyTorch CPU FP16\nno AVX-FP16 path",
                    xy=(r["bandwidth"], r["throughput_median"]),
                    xytext=(110, 12), textcoords="offset points",
                    fontsize=9, color="#a04830", style="italic",
                    arrowprops=dict(arrowstyle="-", color="#a04830",
                                    lw=0.7, alpha=0.7, connectionstyle="arc3,rad=0.1"))
    win_gpu = agg[agg["platform"] == "windows_3070"]
    if not win_gpu.empty:
        r = win_gpu.iloc[0]
        ax.annotate("thermal-throttled\nlaptop GPU",
                    xy=(r["bandwidth"], r["throughput_median"]),
                    xytext=(-140, 50), textcoords="offset points",
                    fontsize=9, color="#a04830", style="italic",
                    arrowprops=dict(arrowstyle="-", color="#a04830",
                                    lw=0.7, alpha=0.7, connectionstyle="arc3,rad=-0.15"))

    ax.set_xlabel("Memory bandwidth  (GB/s)")
    ax.set_ylabel("Mean throughput  (tokens / sec)")
    ethereal_title(ax, "Throughput vs Memory Bandwidth",
                   "Clean platforms align with BW-bound theory; outliers expose implementation gaps")
    ax.legend(loc="lower right", frameon=True)
    soft_spines(ax)
    save(fig, "08_throughput_vs_bw.png")


def _plot_precision_lines(ax, dp):
    for prec in ["fp16", "q8", "q4"]:
        d = dp[dp["precision"] == prec].sort_values("prompt_length")
        if d.empty:
            continue
        col = PRECISION_COLOR.get(prec, ACCENT)
        ax.plot(d["prompt_length"], d["throughput_median"],
                color=col, linewidth=6.0, alpha=0.10, zorder=2)
        ax.plot(d["prompt_length"], d["throughput_median"],
                color=col, linewidth=2.2, zorder=3)
        ax.plot(d["prompt_length"], d["throughput_median"],
                marker="o", linestyle="none",
                markerfacecolor=col, markeredgecolor=BG, markeredgewidth=1.3,
                markersize=7.5, label=prec.upper(), zorder=4)


def fig_precision_comparison(df):
    """T4-only FP16 vs Q8 vs Q4."""
    d = df[df["platform"] == "colab_t4"]
    if d.empty or d["precision"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _plot_precision_lines(ax, d)
    ax.set_xlabel("Prompt length  (tokens)")
    ax.set_ylabel("Throughput  (tokens / sec)")
    ethereal_title(ax, "Weight Quantization on T4",
                   "FP16 vs Q8 vs Q4 throughput across prompt lengths")
    ax.set_xscale("log", base=2)
    ax.legend(title="Precision", loc="best", frameon=True)
    soft_spines(ax)
    save(fig, "11_cuda_precision.png")


def fig_precision_per_platform(df):
    """Per-platform FP16 vs Q8 vs Q4 — one subplot per platform."""
    platforms = [p for p in PLATFORM_ORDER if p in df["platform"].unique()]
    if not platforms:
        return
    n = len(platforms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, plat in zip(axes, platforms):
        dp = df[df["platform"] == plat]
        _plot_precision_lines(ax, dp)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Prompt length  (tokens)")
        ax.set_title(PLATFORM_LABEL.get(plat, plat),
                     fontsize=11, color=TEXT, fontweight="semibold", pad=10)
        ax.legend(fontsize=8.5, title="Precision", frameon=True, loc="best")
        soft_spines(ax)
    axes[0].set_ylabel("Throughput  (tokens / sec)")
    fig.suptitle("Weight Quantization — Throughput Across Platforms",
                 fontsize=14, fontweight="semibold", color=TEXT, y=1.03)
    save(fig, "12_precision_per_platform.png")


def fig_quant_speedup_heatmap(df):
    """Heatmap of Q4/Q8/FP16 throughput ratios per platform."""
    platforms = [p for p in PLATFORM_ORDER if p in df["platform"].unique()]
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

    from matplotlib.colors import LinearSegmentedColormap
    # muted ethereal diverging: terracotta → warm paper → sage
    cmap = LinearSegmentedColormap.from_list(
        "ethereal_div", ["#c44d3b", "#e07a5f", "#f4ead5", "#a8c9b3", "#4f8973"], N=256
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.grid(False)
    log_matrix = np.log2(matrix)
    im = ax.imshow(log_matrix, cmap=cmap, aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(precisions)))
    ax.set_xticklabels([p.upper() for p in precisions], fontsize=11, color=TEXT)
    ax.set_yticks(range(len(platforms)))
    ax.set_yticklabels([PLATFORM_LABEL.get(p, p) for p in platforms],
                       fontsize=10, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(platforms)):
        for j in range(len(precisions)):
            v = matrix[i, j]
            if np.isfinite(v):
                lv = np.clip(log_matrix[i, j], -1.5, 1.5)
                txt_color = "#ffffff" if abs(lv) > 0.9 else TEXT
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                        color=txt_color, fontsize=11, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, label="Throughput ratio vs FP16 (log₂ scale)",
                        pad=0.02, shrink=0.85)
    cbar.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cbar.set_ticklabels(["0.35×", "0.5×", "0.71×", "1×", "1.41×", "2×", "2.83×"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9, labelcolor=AXISCOLOR)
    ethereal_title(ax, "Quantization Speedup Grid",
                   "Mean throughput ratio vs FP16; >1× means quant is faster")
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
