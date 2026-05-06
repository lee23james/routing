#!/usr/bin/env bash
# Search MATH-trained TRIM-Rubric PPO checkpoints on the local 4-GPU box.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
EPISODES_PATH="${EPISODES_PATH:-data/episodes/math_train_200_episodes.jsonl}"
RUBRIC_DIR="${RUBRIC_DIR:-data/rubrics/math200}"
RUBRIC_WEIGHTS="${RUBRIC_WEIGHTS:-$RUBRIC_DIR/rubric_weights.json}"
LAM_RUBRIC="${LAM_RUBRIC:-0.3}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-64}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-1}"

LOG_DIR="logs/trim_rubric_math200_point_search"
CKPT_ROOT="checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$RUBRIC_DIR"

LAMBDAS=(
  "0"
  "5e-6"
  "2e-5"
  "1e-4"
)
GPUS=(0 1 2 3)

sanitize_lam() {
  printf "%s" "$1" | sed 's/+//g'
}

if [ ! -f "$EPISODES_PATH" ]; then
  echo "episodes file not found: $EPISODES_PATH" >&2
  exit 1
fi

if [ ! -f "$RUBRIC_WEIGHTS" ]; then
  echo "[$(date '+%F %T')] generating rubric weights -> $RUBRIC_WEIGHTS"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m rubric.generate_rubrics \
    --episodes_path "$EPISODES_PATH" \
    --output_dir "$RUBRIC_DIR" \
    > "$LOG_DIR/generate_rubrics.log" 2>&1
fi

if ! "$PYTHON_BIN" - <<PY
import json
path = "$RUBRIC_WEIGHTS"
data = json.load(open(path))
active = data.get("active_rubrics", [])
print(f"active_rubrics={len(active)} {active}")
raise SystemExit(0 if active else 1)
PY
then
  echo "no active rubrics in $RUBRIC_WEIGHTS" >&2
  exit 1
fi

run_one() {
  local lam="$1"
  local gpu="$2"
  local lam_tag
  local rub_tag
  lam_tag="$(sanitize_lam "$lam")"
  rub_tag="$(sanitize_lam "$LAM_RUBRIC")"
  local tag="trim_rubric_math200_point_search_lam${lam_tag}_rub${rub_tag}_seed${SEED}"
  local save_dir="${CKPT_ROOT}/${tag}"
  local log_file="${LOG_DIR}/train_lam${lam_tag}_rub${rub_tag}_seed${SEED}_gpu${gpu}.log"

  echo "[$(date '+%F %T')] GPU ${gpu} lam=${lam} lam_rubric=${LAM_RUBRIC} -> ${save_dir}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m router.train_ppo \
    --episodes_path "$EPISODES_PATH" \
    --lam "$lam" \
    --lam_rubric "$LAM_RUBRIC" \
    --rubric_weights "$RUBRIC_WEIGHTS" \
    --num_epochs "$NUM_EPOCHS" \
    --episodes_per_epoch "$EPISODES_PER_EPOCH" \
    --device cuda:0 \
    --save_dir "$save_dir" \
    --save_every "$SAVE_EVERY" \
    --save_epoch_checkpoints \
    --seed "$SEED" \
    > "$log_file" 2>&1
}

active=0
for idx in "${!LAMBDAS[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${LAMBDAS[$idx]}" "$gpu" &
  active=$((active + 1))
  if [ "$active" -ge "${#GPUS[@]}" ]; then
    wait -n
    active=$((active - 1))
  fi
done

wait

echo "[$(date '+%F %T')] all MATH TRIM-Rubric point-search training jobs finished"
