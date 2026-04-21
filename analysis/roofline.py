"""
Phase 3: Roofline model analysis.

For each platform, plots arithmetic intensity (FLOPs/byte) vs achieved
performance (GFLOPs/s), overlaid with the platform's memory-bandwidth and
peak-compute ceilings.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_results import apply_style, PALETTE, ACCENT


PLATFORMS = {
    "m4pro_mps":    {"bw_gbps": 273, "peak_tflops": 4.5,  "label": "M4 Pro (MPS)",    "color": PALETTE["mps"]},
    "m4pro_cpu":    {"bw_gbps": 100, "peak_tflops": 1.5,  "label": "M4 Pro (CPU)",    "color": PALETTE["cpu"]},
    "colab_t4":     {"bw_gbps": 320, "peak_tflops": 65.0, "label": "NVIDIA T4 (CUDA)","color": PALETTE["cuda"]},
    "windows_cpu": {"bw_gbps": 50,  "peak_tflops": 1.0,  "label": "Windows CPU",     "color": PALETTE["wincpu"]},
}

apply_style()


def model_params_and_kv(d_model=2048, n_layers=22, n_heads=32, head_dim=64, seq=512):
    """Rough TinyLlama-1.1B dimensions for analytical roofline points."""
    
    weight_bytes = n_layers * (4 * d_model * d_model + 3 * d_model * 4 * d_model) * 2
    kv_bytes = n_layers * 2 * n_heads * head_dim * seq * 2  # fp16 K and V
    
    params = n_layers * (4 * d_model * d_model + 3 * d_model * 4 * d_model)
    flops = 2 * params
    return weight_bytes, kv_bytes, flops


def plot_roofline(out_path="figures/roofline.png"):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ai = np.logspace(-2, 3, 400)

    DECODE_AI = 0.1  # FLOPs/byte — typical LLM decode operating point

   
    ridges = [p["peak_tflops"] * 1000 / p["bw_gbps"] for p in PLATFORMS.values()]
    ax.axvspan(1e-2, min(ridges), alpha=0.06, color="#c44536", zorder=0)
    ax.axvspan(max(ridges), 1e3, alpha=0.06, color="#2a9d8f", zorder=0)

    
    for key, p in PLATFORMS.items():
        bw = p["bw_gbps"]
        peak = p["peak_tflops"] * 1000
        ridge_ai = peak / bw
        roof = np.minimum(ai * bw, peak)
        ax.loglog(ai, roof, label=f"{p['label']}  ({bw} GB/s, {p['peak_tflops']} TFLOPS)",
                  color=p["color"], linewidth=2.4, zorder=3)

        # Ridge point marker
        ax.plot(ridge_ai, peak, marker="o", color=p["color"],
                markersize=7, markeredgecolor="white",
                markeredgewidth=1.4, zorder=4)

        
        y_decode = DECODE_AI * bw
        ax.plot(DECODE_AI, y_decode, marker="*", color=p["color"],
                markersize=16, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)

    
    ax.axvline(DECODE_AI, color=ACCENT, linestyle=":", linewidth=1.3,
               alpha=0.7, zorder=2)

    
    ax.text(0.02, 1.8e4, "MEMORY-BOUND\n(LLM decode regime)",
            fontsize=10, color="#8c2f28", fontweight="bold",
            ha="left", va="top", alpha=0.85)
    ax.text(200, 2.5, "COMPUTE-BOUND",
            fontsize=10, color="#1a6f63", fontweight="bold",
            ha="center", va="bottom", alpha=0.85)
    ax.annotate("LLM decode\n(AI ≈ 0.1 FLOP/byte)",
                xy=(DECODE_AI, 12), xytext=(0.35, 3),
                fontsize=10, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))

    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(5e-1, 2e5)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Attainable Performance (GFLOPs / s)")
    ax.set_title("Roofline Model — LLM Decode Sits Well Below Every Platform's Ridge",
                 pad=12)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)

    
    from matplotlib.lines import Line2D
    legend_markers = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333",
               markersize=7, markeredgecolor="white", label="Ridge point (peak)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#333",
               markersize=14, markeredgecolor="white", label="Decode operating point"),
    ]
    leg1 = ax.legend(loc="lower right", fontsize=9.5, title="Platform (BW, Peak)")
    ax.add_artist(leg1)
    ax.legend(handles=legend_markers, loc="upper left",
              fontsize=9.5, framealpha=0.95)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in df["device"].dropna().unique():
        d = df[df["device"] == device].sort_values("prompt_length")
        _, kv_bytes_per_len, _ = zip(*[model_params_and_kv(seq=s) for s in d["prompt_length"]])
        bytes_per_tok = wbytes + np.array(kv_bytes_per_len)
        secs = d["decode_ms_median"].values / 1000.0
        achieved_bw = bytes_per_tok / secs / 1e9
        key = {"mps": "m4pro_mps", "cpu": "m4pro_cpu", "cuda": "colab_t4"}.get(device, device)
        plat = PLATFORMS.get(key, {})
        peak = plat.get("bw_gbps", 100)
        col = plat.get("color", ACCENT)
        lbl = plat.get("label", device)
        ax.plot(d["prompt_length"], 100 * achieved_bw / peak,
                marker="o", color=col, label=lbl, linewidth=2.0)
    ax.axhline(100, color="#999", linestyle=":", linewidth=1)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Bandwidth utilization (% of peak)")
    ax.set_title("Achieved Memory Bandwidth vs Platform Peak")
    ax.set_xscale("log", base=2)
    ax.legend(title="Platform")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    plot_roofline()
    bandwidth_utilization()
