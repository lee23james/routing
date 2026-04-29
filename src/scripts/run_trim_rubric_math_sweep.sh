#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-0}"
EP_FILE="${2:-data/episodes/math_train_1k_episodes.jsonl}"
RUBRIC_WEIGHTS="${3:-data/rubrics/math_train_1k/rubric_weights.json}"
EPOCHS="${4:-200}"
LAM_RUBRIC="${LAM_RUBRIC:-0.3}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-checkpoints/trim_rubric_math1k_sweep}"
LAM_VALUES="${LAM_VALUES:-8e-4 6e-4 5e-4 4e-4 3e-4}"

if [ ! -f "$EP_FILE" ]; then
    echo "Missing episodes: $EP_FILE"
    echo "Generate it with: PYTHONPATH=src python -m data.generate_episodes --dataset math_train_1k --output_dir data/episodes"
    exit 1
fi

if [ ! -f "$RUBRIC_WEIGHTS" ]; then
    echo "Missing rubric weights: $RUBRIC_WEIGHTS"
    echo "Generate them with: PYTHONPATH=src python -m rubric.generate_rubrics --episodes_path $EP_FILE --output_dir $(dirname "$RUBRIC_WEIGHTS")"
    exit 1
fi

mkdir -p "$OUT_ROOT" logs

echo "Math TRIM-Rubric sweep"
echo "  GPU: $GPU"
echo "  episodes: $EP_FILE"
echo "  rubric_weights: $RUBRIC_WEIGHTS"
echo "  lam_rubric: $LAM_RUBRIC"
echo "  epochs: $EPOCHS"
echo "  lambdas: $LAM_VALUES"

for LAM in $LAM_VALUES; do
    SAVE_DIR="$OUT_ROOT/$LAM"
    LOG_FILE="logs/trim_rubric_math1k_${LAM}.log"
    if [ -f "$SAVE_DIR/best.pt" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "Skip $LAM, checkpoint exists: $SAVE_DIR/best.pt"
        continue
    fi

    echo "Training Math cost=$LAM -> $SAVE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python -u -m router.train_ppo \
        --episodes_path "$EP_FILE" \
        --lam "$LAM" \
        --lam_rubric "$LAM_RUBRIC" \
        --rubric_weights "$RUBRIC_WEIGHTS" \
        --num_epochs "$EPOCHS" \
        --device "$DEVICE" \
        --save_dir "$SAVE_DIR" \
        --tag "trim_rubric_math1k_${LAM}" \
        2>&1 | tee "$LOG_FILE"
done
