set -e

TARGET="${1:-mps}"
PY="${PY:-.venv/bin/python}"
PLENS="${PLENS:-32 64 128 256 512 1024}"
OUT_TOK="${OUT_TOK:-128}"
WARMUP="${WARMUP:-3}"
TRIALS="${TRIALS:-10}"

case "$TARGET" in
    mps)
        DEVICE=mps; PLATFORM_TAG=m4pro_mps ;;
    cpu)
        DEVICE=cpu; PLATFORM_TAG=m4pro_cpu ;;
    windows)
        DEVICE=cpu; PLATFORM_TAG=windows_cpu ;;
    cuda)
        DEVICE=cuda; PLATFORM_TAG=colab_t4 ;;
    *)
        echo "Unknown target: $TARGET"; exit 1 ;;
esac

echo "GGUF benchmark: target=$TARGET device=$DEVICE platform_tag=$PLATFORM_TAG "

for PRECISION in q8 q4; do
    echo ""
    echo "--- precision=$PRECISION ---"
    $PY benchmark_gguf.py \
        --device "$DEVICE" \
        --platform-tag "$PLATFORM_TAG" \
        --precision "$PRECISION" \
        --prompt-lengths $PLENS \
        --output-tokens "$OUT_TOK" \
        --warmup "$WARMUP" \
        --trials "$TRIALS"
done

echo ""
echo "GGUF benchmark done for $TARGET"
