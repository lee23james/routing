#!/usr/bin/env bash
# Search TRIM-Agg PPO checkpoints across low lambda values on GPUs 4/5/6/7.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/honghudata/deepseek_VG/maker/anaconda3/envs/routing/bin/python}"
EPISODES_PATH="${EPISODES_PATH:-data/episodes/omnimath_episodes.jsonl}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-64}"
SAVE_EVERY="${SAVE_EVERY:-20}"
SEED="${SEED:-1}"

LOG_DIR="logs/trim_agg_point_search"
CKPT_ROOT="checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_ROOT"

LAMBDAS=(
  "0"
  "1e-6"
  "2e-6"
  "5e-6"
  "8e-6"
  "1e-5"
  "1.5e-5"
  "2e-5"
  "3e-5"
  "5e-5"
  "8e-5"
  "1e-4"
)
GPUS=(4 5 6 7)

sanitize_lam() {
  printf "%s" "$1" | sed 's/+//g'
}

run_one() {
  local lam="$1"
  local gpu="$2"
  local lam_tag
  lam_tag="$(sanitize_lam "$lam")"
  local tag="trim_agg_point_search_lam${lam_tag}_seed${SEED}"
  local save_dir="${CKPT_ROOT}/${tag}"
  local log_file="${LOG_DIR}/train_lam${lam_tag}_seed${SEED}_gpu${gpu}.log"

  echo "[$(date '+%F %T')] GPU ${gpu} lam=${lam} -> ${save_dir}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m router.train_ppo \
    --episodes_path "$EPISODES_PATH" \
    --lam "$lam" \
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

echo "[$(date '+%F %T')] all point-search training jobs finished"
