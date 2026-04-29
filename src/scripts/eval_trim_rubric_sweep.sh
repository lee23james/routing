#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${1:-math}"
DEVICE="${DEVICE:-cpu}"

case "$DATASET" in
    math|math1k)
        EP_FILE="${EP_FILE:-data/episodes/math500_test_100_episodes.jsonl}"
        CKPT_ROOT="${CKPT_ROOT:-checkpoints/trim_rubric_math1k_sweep}"
        OUT_ROOT="${OUT_ROOT:-results/trim_rubric_math1k_sweep}"
        LAM_VALUES="${LAM_VALUES:-8e-4 6e-4 5e-4 4e-4 3e-4}"
        ;;
    aime|aime_official)
        EP_FILE="${EP_FILE:-data/episodes/aime_test_episodes.jsonl}"
        CKPT_ROOT="${CKPT_ROOT:-checkpoints/trim_rubric_aime_official_sweep}"
        OUT_ROOT="${OUT_ROOT:-results/trim_rubric_aime_official_sweep}"
        LAM_VALUES="${LAM_VALUES:-3e-4 2.5e-4 2e-4 1.5e-4 1e-4 8e-5}"
        ;;
    *)
        echo "Unknown dataset: $DATASET (expected math or aime)" >&2
        exit 1
        ;;
esac

if [ ! -f "$EP_FILE" ]; then
    echo "Missing eval episodes: $EP_FILE" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT" logs

echo "TRIM-Rubric eval sweep"
echo "  dataset: $DATASET"
echo "  eval episodes: $EP_FILE"
echo "  checkpoints: $CKPT_ROOT"
echo "  output: $OUT_ROOT"
echo "  lambdas: $LAM_VALUES"

for LAM in $LAM_VALUES; do
    CKPT="$CKPT_ROOT/$LAM/best.pt"
    OUT_DIR="$OUT_ROOT/$LAM"
    LOG_FILE="logs/eval_trim_rubric_${DATASET}_${LAM}.log"

    if [ ! -f "$CKPT" ]; then
        echo "[skip] missing checkpoint: $CKPT"
        continue
    fi
    if [ -f "$OUT_DIR/comparison_summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] eval exists: $OUT_DIR/comparison_summary.json"
        continue
    fi

    mkdir -p "$OUT_DIR"
    echo "[eval] $DATASET cost=$LAM"
    PYTHONUNBUFFERED=1 python -u -m eval.evaluate \
        --mode offline \
        --episodes_path "$EP_FILE" \
        --checkpoint_rubric "$CKPT" \
        --lam_values "$LAM" \
        --device "$DEVICE" \
        --output_dir "$OUT_DIR" \
        2>&1 | tee "$LOG_FILE"
done

echo "TRIM-Rubric eval sweep complete."
