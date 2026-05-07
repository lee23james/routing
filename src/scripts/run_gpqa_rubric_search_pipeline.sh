#!/usr/bin/env bash
# Reuse GPQA train/test episodes, then run Agg/Rubric point search and final eval.

set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_EPISODES="${TRAIN_EPISODES:-data/episodes/gpqa_main_train_200_episodes.jsonl}"
TEST_EPISODES="${TEST_EPISODES:-data/episodes/gpqa_diamond_test_100_episodes.jsonl}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-200}"
EXPECTED_TEST="${EXPECTED_TEST:-100}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
LOG_DIR="logs/trim_gpqa_main200_diamond100_search"

mkdir -p "$LOG_DIR"

line_count() {
  local path="$1"
  if [ -f "$path" ]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

wait_for_episodes() {
  while true; do
    local train_count
    local test_count
    train_count="$(line_count "$TRAIN_EPISODES")"
    test_count="$(line_count "$TEST_EPISODES")"
    echo "[$(date '+%F %T')] episodes: train=${train_count}/${EXPECTED_TRAIN}, test=${test_count}/${EXPECTED_TEST}"
    if [ "$train_count" -ge "$EXPECTED_TRAIN" ] && [ "$test_count" -ge "$EXPECTED_TEST" ]; then
      return
    fi
    sleep "$SLEEP_SECONDS"
  done
}

wait_for_episodes

echo "[$(date '+%F %T')] starting GPQA TRIM-Agg point search"
bash scripts/search_trim_agg_gpqa_points_4gpu.sh \
  > "$LOG_DIR/pipeline_train_agg.log" 2>&1

echo "[$(date '+%F %T')] starting GPQA TRIM-Rubric point search"
bash scripts/search_trim_rubric_gpqa_points_4gpu.sh \
  > "$LOG_DIR/pipeline_train_rubric.log" 2>&1

echo "[$(date '+%F %T')] starting GPQA final eval/plot"
bash scripts/eval_gpqa_search_final.sh \
  > "$LOG_DIR/pipeline_eval_final.log" 2>&1

echo "[$(date '+%F %T')] GPQA Rubric-search pipeline finished"
