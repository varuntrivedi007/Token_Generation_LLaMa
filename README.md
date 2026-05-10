# Token-Generation Latency Benchmarking in LLaMA

**CECS 574 · Advanced Computer Architecture · CSULB · Spring 2026**
Avadh Maheshbhai Joshi · Eva Mewada · Varun Dushyant Trivedi

End-to-end reproducible study of autoregressive decode latency in TinyLlama-1.1B across 5 hardware platforms (Apple M4 Pro MPS/CPU, NVIDIA T4, RTX 3070 Laptop, Intel i7-11370H), 3 precisions (FP16, INT8/Q8_0, INT4/Q4_K_M), and 6 prompt lengths (32–1024 tokens). Validates the roofline memory-bandwidth bound, decomposes per-component latency via PyTorch forward hooks, and quantifies weight-vs-KV-cache quantization tradeoffs.

---

## 1. Repository Layout

```
Token_Generation/
├── benchmark.py             # Phase 1  PyTorch harness (FP16, INT8/INT4 via bitsandbytes)
├── benchmark_gguf.py        # Phase 1b llama.cpp/GGUF harness (Q8_0, Q4_K_M)
├── decomposition.py         # Phase 2  forward-hook component-level latency
├── optimization.py          # Phase 3  symmetric per-(head,token) INT8 KV-cache quant
├── energy.py                # Phase 4  powermetrics J/token (macOS only)
├── run_all.sh               # Full PyTorch pipeline
├── run_gguf.sh              # GGUF Q8 + Q4 sweep
├── analysis/
│   ├── merge_results.py     # Cross-platform JSON → results/merged.csv
│   ├── plot_results.py      # Figures 01–06, 11–13
│   └── roofline.py          # Figures 09 (roofline) + 10 (bandwidth utilization)
├── results/
│   ├── <platform>/                          # raw timing JSON
│   ├── decomposition_<device>[_win]/        # decomp JSON
│   ├── optimization_<device>[_win]/         # KV-quant JSON
│   └── merged.csv                           # produced by analysis/merge_results.py
├── figures/                 # final figures (20 PNGs)
├── requirements.txt
└── README.md
```

---

## 2. Hardware Targets

| Platform tag      | Device  | Hardware                | Peak BW (GB/s) | Peak FP16 (TFLOPS) | Backend(s)                |
| ----------------- | ------- | ----------------------- | -------------- | ------------------ | ------------------------- |
| `m4pro_mps`       | mps     | Apple M4 Pro            | 273            | 4.5                | PyTorch MPS, llama.cpp Metal |
| `m4pro_cpu`       | cpu     | Apple M4 Pro (P-cores)  | 100            | 1.5                | PyTorch CPU, llama.cpp NEON |
| `colab_t4`        | cuda    | NVIDIA T4 (Colab)       | 320            | 65.0               | PyTorch CUDA, llama.cpp CUDA, bitsandbytes |
| `windows_3070`    | cuda    | NVIDIA RTX 3070 Laptop  | 448            | 20.0               | PyTorch CUDA, llama.cpp CUDA |
| `windows_cpu`     | cpu     | Intel i7-11370H (Tiger Lake) | 50         | 1.0                | PyTorch CPU, llama.cpp AVX2 |

All five not required — each platform writes an independent JSON tree, and `analysis/merge_results.py` concatenates whatever is present.

### 2.1 Default output-directory mapping

`benchmark.py`, `decomposition.py`, `optimization.py` derive the `results/` subdir from `--device`:

| `--device` | `benchmark.py` default          | `decomposition.py` default        | `optimization.py` default        |
| ---------- | ------------------------------- | --------------------------------- | -------------------------------- |
| `mps`      | `results/m4pro_mps`             | `results/decomposition_mps`       | `results/optimization_mps`       |
| `cpu`      | `results/m4pro_cpu`             | `results/decomposition_cpu`       | `results/optimization_cpu`       |
| `cuda`     | `results/colab_t4`              | `results/decomposition_cuda`      | `results/optimization_cuda`      |

For Windows (RTX 3070 + i7-11370H), pass `--output-dir <path>` so they don't collide with Colab T4 / M4 CPU buckets — see §6.4–6.5. For `benchmark_gguf.py`, use `--platform-tag windows_3070` / `windows_cpu` (different flag; the GGUF script supports it).

---

## 3. Prerequisites

* Python ≥ 3.10
* `git`, `bash`, `curl` or `wget`
* Hugging Face account + access token (`huggingface-cli login`) — TinyLlama gated on some mirrors
* Platform compilers/SDKs for `llama-cpp-python`:
  * macOS: Xcode CLT, Metal headers (`xcode-select --install`)
  * Linux/CUDA: NVIDIA CUDA Toolkit ≥ 11.8, matching driver
  * Windows: MSVC build tools 2019+ (or run inside WSL2)

---

## 4. Setup

### 4.1 Clone and create venv

```bash
git clone <repo-url> Token_Generation
cd Token_Generation
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Install `llama-cpp-python` (per platform)

```bash
# macOS Metal (M4 Pro)
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir llama-cpp-python

# Linux CUDA (T4 / 3070)
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python

# Linux / Windows CPU
pip install llama-cpp-python
```

### 4.3 Optional: bitsandbytes (CUDA INT8/INT4 weight quant)

```bash
pip install "bitsandbytes>=0.43"      # CUDA only
```

### 4.4 Authenticate with Hugging Face

```bash
huggingface-cli login                 # paste user-access token
```

### 4.5 Model assets

* **PyTorch FP16 weights** — pulled automatically by `transformers.AutoModelForCausalLM.from_pretrained` on first run; cached under `~/.cache/huggingface/`.
* **GGUF Q8_0 / Q4_K_M** — `benchmark_gguf.py` calls `huggingface_hub.hf_hub_download` against `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` automatically. No manual fetch required. Override with `--gguf-path <local_file>` if already present.

---

## 5. Reproduction Pipeline (single platform, fastest path)

Quickest reproduction for the platform you sit at (Apple M4 Pro, Mac CPU, or Linux CUDA — Windows: skip to §6.4–6.5):

```bash
bash run_all.sh mps  tinyllama fp16          # macOS M4 Pro Metal
bash run_all.sh cpu  tinyllama fp16          # Mac CPU (writes m4pro_cpu)
bash run_all.sh cuda tinyllama fp16          # Linux CUDA (writes colab_t4)

# GGUF Q8 + Q4 sweep on same machine
bash run_gguf.sh mps                         # mps | cpu | cuda | windows
```

Then merge + plot (see §7).

---

## 6. Reproduction Pipeline (full 5-platform matrix)

Run on each machine separately, then collect `results/` trees onto a single host before merging.

### 6.1 Apple M4 Pro · MPS

```bash
python benchmark.py      --device mps --model tinyllama --precision fp16
bash   run_gguf.sh       mps
python decomposition.py  --device mps --model tinyllama --prompt-length 128
python decomposition.py  --device mps --model tinyllama --prompt-length 512
python optimization.py   --device mps --model tinyllama
```

### 6.2 Apple M4 Pro · CPU

Same as 6.1 with `--device cpu` and `bash run_gguf.sh cpu`.

### 6.3 NVIDIA T4 (Google Colab)

Colab runtime: *Runtime → Change runtime type → T4 GPU*. After cloning:

```bash
python benchmark.py      --device cuda --model tinyllama --precision fp16
python benchmark.py      --device cuda --model tinyllama --precision q8     # bitsandbytes INT8
python benchmark.py      --device cuda --model tinyllama --precision q4     # bitsandbytes INT4 (NF4)
bash   run_gguf.sh       cuda
python decomposition.py  --device cuda --model tinyllama --prompt-length 128
python decomposition.py  --device cuda --model tinyllama --prompt-length 512
python optimization.py   --device cuda --model tinyllama
```

Move `results/colab_t4/`, `results/decomposition_cuda/`, `results/optimization_cuda/` back to your aggregation host.

### 6.4 NVIDIA RTX 3070 Laptop · Windows

Both Colab T4 and the RTX 3070 use `--device cuda`, so `benchmark.py` writes to the same `results/colab_t4/` dir by default. Override with `--output-dir`:

```powershell
.venv\Scripts\activate

python benchmark.py     --device cuda --model tinyllama --precision fp16 
                        --output-dir results/windows_3070

# run_gguf.sh hardcodes PLATFORM_TAG=colab_t4 for cuda, so call benchmark_gguf.py directly:
python benchmark_gguf.py --device cuda --precision q8 --platform-tag windows_3070
python benchmark_gguf.py --device cuda --precision q4 --platform-tag windows_3070

python decomposition.py --device cuda --model tinyllama --prompt-length 128 
                        --output-dir results/decomposition_cuda_win
python decomposition.py --device cuda --model tinyllama --prompt-length 512 
                        --output-dir results/decomposition_cuda_win
python optimization.py  --device cuda --model tinyllama 
                        --output-dir results/optimization_cuda_win
```

### 6.5 Intel i7-11370H · Windows CPU

```powershell
.venv\Scripts\activate

python benchmark.py     --device cpu --model tinyllama --precision fp16 
                        --output-dir results/windows_cpu

bash run_gguf.sh windows                     # this branch correctly sets PLATFORM_TAG=windows_cpu

python decomposition.py --device cpu --model tinyllama --prompt-length 128 
                        --output-dir results/decomposition_cpu_win
python decomposition.py --device cpu --model tinyllama --prompt-length 512 
                        --output-dir results/decomposition_cpu_win
python optimization.py  --device cpu --model tinyllama ^
                        --output-dir results/optimization_cpu_win
```

> **Note** — Tiger Lake lacks AVX-512 BF16 used by recent PyTorch CPU kernels, so FP16 falls back to scalar FP32. Produces the deliberate ~50× outlier (≈0.45 tok/s).

---

## 7. Analysis Pipeline

Once `results/` populated:

```bash
python -m analysis.merge_results          # → results/merged.csv  (91 rows for full 5-platform matrix)
python -m analysis.plot_results           # → figures/01–06, 11–13
python -m analysis.roofline               # → figures/09_roofline.png, 10_bw_utilization.png
```

Re-runs idempotent. Figures written at 300 DPI.

---

## 8. Output Schemas

### 8.1 Per-cell timing JSON (`benchmark.py`, `benchmark_gguf.py`)

```jsonc
{
  "env": { "platform": "...", "torch": "...", "cuda": "..." },
  "device": "mps", "backend": "pytorch", "model": "tinyllama",
  "precision": "fp16", "platform_tag": "m4pro_mps",
  "prompt_lengths": [32, 64, 128, 256, 512, 1024],
  "trials": [
    {
      "prompt_length": 128,
      "ttft_ms":  { "raw": [...], "filtered": [...], "median": ... },
      "decode_ms":{ "raw": [...], "filtered": [...], "median": ... },
      "throughput_tps": ...,
      "total_ms": ...
    }
  ]
}
```

### 8.2 Decomposition JSON (`decomposition.py`)

7 buckets (per layer, summed): `embedding`, `rmsnorm`, `qkv_projection`, `attention_full`, `attn_output_proj`, `mlp_full`, `lm_head`. Stored as ms-per-component plus normalized fractions.

### 8.3 KV-quant JSON (`optimization.py`)

Baseline FP16 KV vs symmetric per-(head, token) INT8 KV. Records latency, KV bytes, perplexity at p ∈ {512, 1024}.

### 8.4 `results/merged.csv`

```
platform_dir, platform, file, device, backend, model, precision,
prompt_length, ttft_ms_median, decode_ms_median, throughput_median, total_ms_median
```

---

## 9. Reference Numbers (sanity check)

After running the full matrix, `merged.csv` should reproduce these FP16 means (averaged across 6 prompt lengths):

| platform        | TTFT mean (ms) | decode mean (ms/tok) | throughput mean (tok/s) |
| --------------- | -------------: | -------------------: | ----------------------: |
| `colab_t4`      |          86.77 |                24.07 |                   39.24 |
| `m4pro_mps`     |         264.06 |                31.35 |                   31.56 |
| `windows_3070`  |         164.28 |                47.39 |                   19.00 |
| `m4pro_cpu`     |        3602.46 |                42.38 |                   17.44 |
| `windows_cpu`   |      309188.88 |              1049.10 |                    0.45 |

Roofline regression on the three mature platforms (colab_t4, m4pro_mps, m4pro_cpu): **R² = 0.978**.
KV-cache INT8 quant: **1.94×** memory reduction, **0–36 % decode-latency regression** depending on platform.
Q4_K_M weight quant on Tiger Lake: **57.6×** speedup vs FP16 baseline.

---

## 10. Methodology Summary

* 3 warmup trials (discarded) + 10 timed trials per `(platform, precision, prompt_length)` cell.
* 1.5× IQR outlier fence applied per cell, then median.
* Sync primitives:
  * CUDA → `torch.cuda.Event(enable_timing=True)`
  * MPS → `torch.mps.synchronize()` flanking `time.perf_counter_ns`
  * CPU → `time.perf_counter_ns`
* Deterministic decoding: greedy (top-k = 1, temperature = 0).
* Output token count fixed at 128.
* Decomposition uses PyTorch forward hooks classified by module name pattern (see `decomposition.py:23`).
* KV-cache quant is post-training symmetric INT8 per `(head, token)`; scale stored in FP16.

---

## 11. Troubleshooting

| Symptom                                             | Fix                                                                                                |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `RuntimeError: MPS backend out of memory`           | Drop `--prompt-length 1024` to 512, or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.                 |
| `bitsandbytes` import fails on CUDA                 | Match CUDA toolkit ↔ wheel: `pip install bitsandbytes --extra-index-url <cuXXX>`.                  |
| `llama_cpp` segfault on macOS                       | `pip install --force-reinstall --no-cache-dir CMAKE_ARGS="-DGGML_METAL=on" llama-cpp-python`.      |
| Decode times wildly variable on RTX 3070 Laptop     | DVFS / thermal — plug in, set Windows power plan to *Best Performance*, raise `--trials` to 20.    |
| `merge_results.py` skips a platform                 | Confirm directory naming: `results/<platform>/` matches one of the five tags in §2.                |
| Tiger Lake FP16 throughput ≈ 0.4 tok/s              | Expected — AVX-512 absent, scalar FP32 fallback.                                                   |
| Two CUDA boxes overwrite each other's results       | Pass `--output-dir results/windows_3070` to `benchmark.py`, `decomposition.py`, `optimization.py`. |
| `benchmark_gguf.py` writes to wrong platform dir    | Pass `--platform-tag <tag>` directly (not via `run_gguf.sh`, which hardcodes `colab_t4` for cuda). |
