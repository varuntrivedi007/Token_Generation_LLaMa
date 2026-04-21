# LLM Token-Generation Latency Benchmark

Cross-platform benchmark (MPS / CUDA / CPU) for decomposing and analyzing
token-generation latency in LLaMA-family models. See [CLAUDE.md](CLAUDE.md)
for the full project plan.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# CUDA only, for q4/q8:
# pip install bitsandbytes
```

## Run the benchmark

```bash
# M4 Pro MPS
python benchmark.py --device mps --model tinyllama --precision fp16

# M4 Pro CPU
python benchmark.py --device cpu --model tinyllama --precision fp16

# Colab T4
python benchmark.py --device cuda --model tinyllama --precision fp16
python benchmark.py --device cuda --model tinyllama --precision q8
python benchmark.py --device cuda --model tinyllama --precision q4
```

Single-length smoke test:

```bash
python benchmark.py --device mps --prompt-lengths 128 --trials 3 --warmup 1
```

Results land in `results/<platform>/<model>_<precision>_<device>.json`.

## Layout

```
benchmark.py        # Phase 1: timing harness (TTFT, per-token, throughput)
decomposition.py    # Phase 2: forward-hook latency decomposition (TODO)
optimization.py     # Phase 4: INT8 KV-cache quantization (TODO)
energy.py           # Phase 5: powermetrics energy capture (TODO)
analysis/           # plotting, roofline, cross-platform merge (TODO)
results/            # raw JSON per platform
figures/            # generated plots
report/             # write-up
```
