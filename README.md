# LLM Token-Generation Latency Benchmark

Cross-platform benchmark for decomposing and analyzing autoregressive token-generation latency in LLaMA-family models. Tests TTFT, per-token latency, throughput, and memory-bandwidth utilization across five hardware platforms using both PyTorch and llama.cpp (GGUF) backends.

## Hardware Tested

| Platform | Backend | Memory BW |
|---|---|---|
| Apple M4 Pro | MPS (Metal) | 273 GB/s |
| Apple M4 Pro | CPU | ~100 GB/s |
| NVIDIA T4 (Colab) | CUDA | 320 GB/s HBM2 |
| NVIDIA RTX 3070 Laptop | CUDA | 448 GB/s spec |
| Intel i7-11370H | CPU | ~50 GB/s DDR4 |

## Models

- **TinyLlama-1.1B-Chat** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) — fits all platforms
- **OpenLLaMA-3B** (`openlm-research/open_llama_3b`) — for model-scaling experiments

## Precisions

- **FP16** — PyTorch baseline on all platforms
- **Q8_0 / Q4_K_M** — GGUF via llama.cpp on all platforms
- **INT8 / INT4 weight** — bitsandbytes on CUDA only

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**llama.cpp (GGUF backend) — platform-specific install:**

```bash
# macOS Metal
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Linux CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Linux / Windows CPU
pip install llama-cpp-python
```

**CUDA weight quantization (optional):**
```bash
pip install bitsandbytes>=0.43
```

## Running Benchmarks

### PyTorch backend (FP16, INT8/INT4 on CUDA)

```bash
# M4 Pro — MPS
python benchmark.py --device mps --model tinyllama --precision fp16

# M4 Pro — CPU
python benchmark.py --device cpu --model tinyllama --precision fp16

# NVIDIA GPU — CUDA
python benchmark.py --device cuda --model tinyllama --precision fp16
python benchmark.py --device cuda --model tinyllama --precision q8
python benchmark.py --device cuda --model tinyllama --precision q4

# Smoke test (single length, few trials)
python benchmark.py --device mps --prompt-lengths 128 --trials 3 --warmup 1
```

Results land in `results/<platform>/<model>_<precision>_<device>.json`.

### GGUF backend (Q8_0 + Q4_K_M via llama.cpp)

Download GGUF model files first (e.g. from Hugging Face Hub), then:

```bash
# macOS MPS
bash run_gguf.sh mps

# macOS / Linux CPU
bash run_gguf.sh cpu

# Windows CPU
bash run_gguf.sh windows

# CUDA
bash run_gguf.sh cuda
```

Or run directly:

```bash
python benchmark_gguf.py --device mps --precision q8
python benchmark_gguf.py --device mps --precision q4
```

### Full pipeline (benchmark → decompose → KV optimize → plots)

```bash
bash run_all.sh <device> <model> <precision>
# e.g.
bash run_all.sh mps tinyllama fp16
```

## Latency Decomposition

Measures time in each model component (MLP, QKV projection, attention, RMSNorm, LM head, embedding) using PyTorch forward hooks — not estimated from architecture proportions.

```bash
python decomposition.py --device mps --model tinyllama
python decomposition.py --device cuda --model tinyllama
```

Output: `results/<platform>/decomp_<model>_<device>.json` with per-component time fractions.

## KV-Cache INT8 Quantization

Implements symmetric per-(head, token) INT8 quantization of the KV-cache; measures latency before/after and perplexity delta at 512 and 1024 token contexts.

```bash
python optimization.py --device mps --model tinyllama
python optimization.py --device cuda --model tinyllama
```

Output: `results/optimization_<platform>/kvq_<model>_<device>.json`.

**Finding:** INT8 KV halves cache memory (1.94× reduction) but yields no latency benefit — KV traffic is ~1% of total memory bandwidth (dominated by 2.2 GB FP16 weight streaming). Perplexity delta < 0.1 on all platforms.

## Analysis & Plots

Merge results across platforms, then generate all figures:

```bash
python analysis/merge_results.py          # produces results/merged.csv
python analysis/plot_results.py           # produces figures/0*_*.png
python analysis/roofline.py               # produces figures/09_roofline.png
```

Figures generated:
- `01_ttft_vs_plen.png` — TTFT vs prompt length
- `02_decode_vs_plen.png` — per-token latency vs context length
- `03_throughput.png` — throughput vs prompt length
- `04_platform_summary.png` — cross-platform bar chart
- `05_decomp_<platform>.png` — latency decomposition stacked bars
- `06_kvq_<platform>.png` — KV quantization before/after
- `08_throughput_vs_bw.png` — throughput vs memory bandwidth scatter (R² = 0.978)
- `09_roofline.png` — roofline model (log-log)
- `10_bw_utilization.png` — bandwidth utilization % vs context length
- `11_cuda_precision.png` — CUDA precision comparison
- `12_precision_per_platform.png` — quantization speedup per platform
- `13_quant_speedup_heatmap.png` — quantization speedup heatmap

## Methodology

- **Trials:** 3 warmup (excluded) + 10 timed trials per configuration
- **Outlier filtering:** IQR-based
- **Synchronization:**
  - MPS: `torch.mps.synchronize()` before/after
  - CUDA: `torch.cuda.Event(enable_timing=True)`
  - CPU: `time.perf_counter_ns()`
- **Batch size:** 1 (decode regime)
- **Prompt lengths:** 32, 64, 128, 256, 512, 1024 tokens
- **Output tokens:** 128 per trial

## Project Layout

```
benchmark.py          # Phase 1: timing harness (TTFT, per-token, throughput)
benchmark_gguf.py     # Phase 1b: GGUF/llama.cpp harness (Q8_0, Q4_K_M)
decomposition.py      # Phase 2: forward-hook latency decomposition
optimization.py       # Phase 4: INT8 KV-cache quantization
energy.py             # Phase 5: powermetrics energy-per-token (macOS)
run_all.sh            # Full PyTorch pipeline
run_gguf.sh           # GGUF Q8+Q4 runner
analysis/
  merge_results.py    # Merge cross-platform JSON → merged.csv
  plot_results.py     # Generate all figures
  roofline.py         # Roofline model analysis
results/              # Raw JSON per platform
figures/              # Generated plots
```

## Key Results

| Platform | FP16 Throughput | BW Utilization |
|---|---|---|
| NVIDIA T4 (CUDA) | 31.4 tok/s | ~55% |
| RTX 3070 Laptop (CUDA) | 14.3–30.4 tok/s | variable (DVFS) |
| M4 Pro MPS | 17.9 tok/s | ~38% |
| M4 Pro CPU | 4.9 tok/s | ~29% |
| Intel i7-11370H | 0.36 tok/s | <5% (ISA fallback) |

87× throughput gap (T4 vs i7-11370H) explained by AVX-512 absence on i7 forcing PyTorch scalar FP32 fallback. Roofline fit R² = 0.978 confirms memory-bandwidth as the universal bottleneck across mature platforms.
