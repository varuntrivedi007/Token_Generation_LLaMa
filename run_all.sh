#!/usr/bin/env bash
# Full benchmark sweep. Run on each platform; results land in results/.
set -euo pipefail

DEVICE="${1:-mps}"         # mps | cpu | cuda
MODEL="${2:-tinyllama}"    # tinyllama | openllama3b
PRECISION="${3:-fp16}"     # fp16 | fp32 | q8 | q4

echo "=== Phase 1: benchmark sweep ==="
python benchmark.py --device "$DEVICE" --model "$MODEL" --precision "$PRECISION"

echo "=== Phase 2: decomposition (p=128, p=512) ==="
python decomposition.py --device "$DEVICE" --model "$MODEL" --precision "$PRECISION" --prompt-length 128
python decomposition.py --device "$DEVICE" --model "$MODEL" --precision "$PRECISION" --prompt-length 512

echo "=== Phase 4: KV-cache quantization ==="
python optimization.py --device "$DEVICE" --model "$MODEL"

echo "=== Merge + plots ==="
python -m analysis.merge_results
python -m analysis.plot_results

echo "Done. See figures/ and results/merged.csv"
