#!/usr/bin/env bash
# Wait for AIME Part-search episodes, then run Agg/Rubric quick search and final eval.

set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_EPISODES="${TRAIN_EPISODES:-data/episodes/aime_2010_2024_part1_train_episodes.jsonl}"
TEST_EPISODES="${TEST_EPISODES:-data/episodes/aime_2020_2024_part2_test_episodes.jsonl}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-204}"
EXPECTED_TEST="${EXPECTED_TEST:-74}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
LOG_DIR="logs/trim_aime_part_search"

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

echo "[$(date '+%F %T')] starting AIME TRIM-Agg point search"
bash scripts/search_trim_agg_aime_part_points_4gpu.sh \
  > "$LOG_DIR/pipeline_train_agg.log" 2>&1

echo "[$(date '+%F %T')] starting AIME TRIM-Rubric point search"
bash scripts/search_trim_rubric_aime_part_points_4gpu.sh \
  > "$LOG_DIR/pipeline_train_rubric.log" 2>&1

echo "[$(date '+%F %T')] starting AIME final eval/plot"
bash scripts/eval_aime_part_search_final.sh \
  > "$LOG_DIR/pipeline_eval_final.log" 2>&1

echo "[$(date '+%F %T')] AIME Part-search pipeline finished"
