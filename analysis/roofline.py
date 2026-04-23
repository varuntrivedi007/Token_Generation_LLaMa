import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_results import (
    apply_style, PALETTE, ACCENT, ACCENT2, BG, GRID, TEXT, AXISCOLOR,
    PLATFORM_LABEL, PLATFORM_COLOR, ethereal_title, soft_spines,
)


PLATFORMS = {
    "m4pro_mps":    {"bw_gbps": 273, "peak_tflops": 4.5,  "label": "M4 Pro (MPS)",     "color": PALETTE["mps"]},
    "m4pro_cpu":    {"bw_gbps": 100, "peak_tflops": 1.5,  "label": "M4 Pro (CPU)",     "color": PALETTE["cpu"]},
    "colab_t4":     {"bw_gbps": 320, "peak_tflops": 65.0, "label": "NVIDIA T4 (CUDA)", "color": PALETTE["cuda"]},
    "windows_3070": {"bw_gbps": 448, "peak_tflops": 20.0, "label": "RTX 3070 Laptop",  "color": PALETTE["wingpu"]},
    "windows_cpu":  {"bw_gbps": 50,  "peak_tflops": 1.0,  "label": "Windows CPU (i7)", "color": PALETTE["wincpu"]},
}

apply_style()


def model_params_and_kv(d_model=2048, n_layers=22, n_heads=32, head_dim=64, seq=512):
    """Rough TinyLlama-1.1B dimensions for analytical roofline points."""
    weight_bytes = n_layers * (4 * d_model * d_model + 3 * d_model * 4 * d_model) * 2
    kv_bytes = n_layers * 2 * n_heads * head_dim * seq * 2
    params = n_layers * (4 * d_model * d_model + 3 * d_model * 4 * d_model)
    flops = 2 * params
    return weight_bytes, kv_bytes, flops


def plot_roofline(out_path="figures/roofline.png"):
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ai = np.logspace(-2, 3, 400)

    DECODE_AI = 0.1

    ridges = [p["peak_tflops"] * 1000 / p["bw_gbps"] for p in PLATFORMS.values()]
    
    ax.axvspan(1e-2, min(ridges), alpha=0.05, color="#c44d3b", zorder=0)
    ax.axvspan(max(ridges), 1e3, alpha=0.05, color="#81b29a", zorder=0)

    for key, p in PLATFORMS.items():
        bw = p["bw_gbps"]
        peak = p["peak_tflops"] * 1000
        ridge_ai = peak / bw
        roof = np.minimum(ai * bw, peak)
        col = p["color"]
       
        ax.loglog(ai, roof, color=col, linewidth=5.5, alpha=0.10, zorder=2)
        
        ax.loglog(ai, roof,
                  label=f"{p['label']}  ·  {bw} GB/s  ·  {p['peak_tflops']} TFLOPS",
                  color=col, linewidth=2.2, zorder=3)

        
        ax.plot(ridge_ai, peak, marker="o", color=col,
                markersize=7.5, markeredgecolor=BG,
                markeredgewidth=1.4, zorder=4)

        
        y_decode = DECODE_AI * bw
        ax.plot(DECODE_AI, y_decode, marker="*", color=col,
                markersize=17, markeredgecolor=BG,
                markeredgewidth=1.2, zorder=5)

    ax.axvline(DECODE_AI, color=ACCENT, linestyle=(0, (3, 3)), linewidth=1.1,
               alpha=0.55, zorder=2)

    ax.text(0.014, 1.4e4, "MEMORY-BOUND\nLLM decode regime",
            fontsize=10, color="#8c2f28", fontweight="semibold",
            ha="left", va="top", alpha=0.85, linespacing=1.3)
    ax.text(260, 2.2, "COMPUTE-BOUND",
            fontsize=10, color="#1a6f63", fontweight="semibold",
            ha="center", va="bottom", alpha=0.85)
    ax.annotate("LLM decode\nAI ≈ 0.1 FLOP/byte",
                xy=(DECODE_AI, 14), xytext=(0.38, 2.6),
                fontsize=10, color=ACCENT, style="italic",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1,
                                connectionstyle="arc3,rad=0.12"))

    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(5e-1, 2e5)
    ax.set_xlabel("Arithmetic Intensity  (FLOPs / byte)")
    ax.set_ylabel("Attainable Performance  (GFLOPs / sec)")
    ethereal_title(ax, "Roofline Model",
                   "LLM decode sits well below every platform's ridge point")
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=1)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.7, zorder=1)
    soft_spines(ax)

    from matplotlib.lines import Line2D
    legend_markers = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AXISCOLOR,
               markersize=7, markeredgecolor=BG, markeredgewidth=1.2,
               label="Ridge point  (peak)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=AXISCOLOR,
               markersize=14, markeredgecolor=BG, markeredgewidth=1.2,
               label="Decode operating point"),
    ]
    leg1 = ax.legend(loc="lower right", fontsize=9, title="Platform  ·  BW  ·  Peak",
                     frameon=True, framealpha=0.94)
    leg1.get_title().set_fontsize(9.5)
    ax.add_artist(leg1)
    ax.legend(handles=legend_markers, loc="upper left",
              fontsize=9.5, framealpha=0.94, frameon=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[saved] {out_path}")


def bandwidth_utilization(results_csv="results/merged.csv", out_path="figures/bw_utilization.png"):
    import pandas as pd
    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        print(f"[skip] {results_csv} not found. Run merge_results.py first.")
        return

    df = df[df["precision"] == "fp16"]

    wbytes, _, _ = model_params_and_kv()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    key_col = "platform" if "platform" in df.columns else "device"
    platforms = df[key_col].dropna().unique()

    for plat_key in platforms:
        d = df[df[key_col] == plat_key].sort_values("prompt_length")
        _, kv_bytes_per_len, _ = zip(*[model_params_and_kv(seq=s) for s in d["prompt_length"]])
        bytes_per_tok = wbytes + np.array(kv_bytes_per_len)
        secs = d["decode_ms_median"].values / 1000.0
        achieved_bw = bytes_per_tok / secs / 1e9
        plat = PLATFORMS.get(plat_key, {})
        peak = plat.get("bw_gbps", 100)
        col = plat.get("color", ACCENT)
        lbl = plat.get("label", plat_key)
        util = 100 * achieved_bw / peak
        # halo
        ax.plot(d["prompt_length"], util, color=col, linewidth=6.0,
                alpha=0.10, zorder=2)
        ax.plot(d["prompt_length"], util, color=col, linewidth=2.2, zorder=3)
        ax.plot(d["prompt_length"], util, marker="o", linestyle="none",
                markerfacecolor=col, markeredgecolor=BG, markeredgewidth=1.3,
                markersize=7.5, label=lbl, zorder=4)

    ax.axhline(100, color=ACCENT2, linestyle=(0, (3, 3)), linewidth=1,
               alpha=0.7, zorder=1)
    ax.set_ylim(top=115)
    ax.text(ax.get_xlim()[1] * 0.92, 102, "peak BW",
            fontsize=9, color=ACCENT2, style="italic",
            ha="right", va="bottom")

    ax.set_xlabel("Prompt length  (tokens)")
    ax.set_ylabel("Bandwidth utilization  (% of peak)")
    ethereal_title(ax, "Achieved Memory Bandwidth vs Platform Peak",
                   "Higher = closer to silicon limit; gaps expose implementation inefficiency")
    ax.set_xscale("log", base=2)
    ax.legend(title="Platform", loc="best", frameon=True)
    soft_spines(ax)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    plot_roofline()
    bandwidth_utilization()
