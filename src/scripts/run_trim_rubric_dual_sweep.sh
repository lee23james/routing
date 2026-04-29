#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

MATH_GPU="${1:-0}"
AIME_GPU="${2:-1}"
EPOCHS="${3:-200}"
MATH_EP="${MATH_EP:-data/episodes/math_train_1k_episodes.jsonl}"
AIME_EP="${AIME_EP:-data/episodes/aime_train_episodes.jsonl}"
MATH_RUBRIC="${MATH_RUBRIC:-data/rubrics/math_train_1k/rubric_weights.json}"
AIME_RUBRIC="${AIME_RUBRIC:-data/rubrics/aime_train/rubric_weights.json}"

mkdir -p logs

echo "Launching dual TRIM-Rubric sweeps"
echo "  Math: GPU $MATH_GPU, episodes=$MATH_EP"
echo "  AIME: GPU $AIME_GPU, episodes=$AIME_EP"
echo "  Epochs: $EPOCHS"

bash scripts/run_trim_rubric_math_sweep.sh \
    "$MATH_GPU" "$MATH_EP" "$MATH_RUBRIC" "$EPOCHS" \
    > logs/trim_rubric_math_dual.log 2>&1 &
MATH_PID=$!

bash scripts/run_trim_rubric_aime_sweep.sh \
    "$AIME_GPU" "$AIME_EP" "$AIME_RUBRIC" "$EPOCHS" \
    > logs/trim_rubric_aime_dual.log 2>&1 &
AIME_PID=$!

echo "  Math PID: $MATH_PID -> logs/trim_rubric_math_dual.log"
echo "  AIME PID: $AIME_PID -> logs/trim_rubric_aime_dual.log"

MATH_STATUS=0
AIME_STATUS=0
wait "$MATH_PID" || MATH_STATUS=$?
wait "$AIME_PID" || AIME_STATUS=$?

echo "Dual sweep finished"
echo "  Math status: $MATH_STATUS"
echo "  AIME status: $AIME_STATUS"

if [ "$MATH_STATUS" -ne 0 ] || [ "$AIME_STATUS" -ne 0 ]; then
    exit 1
fi

if [ "${RUN_EVAL:-1}" = "1" ]; then
    echo "Launching TRIM-Rubric eval sweeps"
    bash scripts/eval_trim_rubric_sweep.sh math \
        > logs/eval_trim_rubric_math_dual.log 2>&1 &
    MATH_EVAL_PID=$!

    bash scripts/eval_trim_rubric_sweep.sh aime \
        > logs/eval_trim_rubric_aime_dual.log 2>&1 &
    AIME_EVAL_PID=$!

    MATH_EVAL_STATUS=0
    AIME_EVAL_STATUS=0
    wait "$MATH_EVAL_PID" || MATH_EVAL_STATUS=$?
    wait "$AIME_EVAL_PID" || AIME_EVAL_STATUS=$?

    echo "Dual eval finished"
    echo "  Math eval status: $MATH_EVAL_STATUS"
    echo "  AIME eval status: $AIME_EVAL_STATUS"

    if [ "$MATH_EVAL_STATUS" -ne 0 ] || [ "$AIME_EVAL_STATUS" -ne 0 ]; then
        exit 1
    fi
fi
