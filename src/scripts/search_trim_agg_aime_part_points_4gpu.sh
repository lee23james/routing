#!/usr/bin/env bash
# Search AIME Part I-trained TRIM-Agg PPO checkpoints on the local 4-GPU box.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
EPISODES_PATH="${EPISODES_PATH:-data/episodes/aime_2010_2024_part1_train_episodes.jsonl}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-64}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-1}"

LOG_DIR="logs/trim_agg_aime_part1_204_point_search"
CKPT_ROOT="checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_ROOT"

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

run_one() {
  local lam="$1"
  local gpu="$2"
  local lam_tag
  lam_tag="$(sanitize_lam "$lam")"
  local tag="trim_agg_aime_part1_204_point_search_lam${lam_tag}_seed${SEED}"
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

echo "[$(date '+%F %T')] all AIME TRIM-Agg point-search training jobs finished"
