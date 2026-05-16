#!/usr/bin/env bash
# Reuse OmniMath 1-3 train episodes and OmniMath 3-4 test episodes, then run search/eval.

set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_EPISODES="${TRAIN_EPISODES:-data/episodes/omnimath_diff1_3_train_200_episodes.jsonl}"
TEST_EPISODES="${TEST_EPISODES:-data/episodes/omnimath_diff3_4_test_200_episodes.jsonl}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-200}"
EXPECTED_TEST="${EXPECTED_TEST:-200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
LOG_DIR="logs/trim_omnimath13_to34_search"

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

wait_for_generation_services_to_exit() {
  while pgrep -f 'generate_omnimath13_to34_episodes_20k|server_vllm|data.generate_episodes' >/dev/null; do
    echo "[$(date '+%F %T')] generation/vLLM services still running; waiting before offline training"
    sleep "$SLEEP_SECONDS"
  done
}

wait_for_episodes
wait_for_generation_services_to_exit

echo "[$(date '+%F %T')] starting OmniMath 1-3 TRIM-Agg point search"
TRAIN_EPISODES="$TRAIN_EPISODES" \
EPISODES_PATH="$TRAIN_EPISODES" \
bash scripts/search_trim_agg_omnimath13_to34_points_4gpu.sh \
  > "$LOG_DIR/pipeline_train_agg.log" 2>&1

echo "[$(date '+%F %T')] starting OmniMath 1-3 TRIM-Rubric point search"
TRAIN_EPISODES="$TRAIN_EPISODES" \
EPISODES_PATH="$TRAIN_EPISODES" \
bash scripts/search_trim_rubric_omnimath13_to34_points_4gpu.sh \
  > "$LOG_DIR/pipeline_train_rubric.log" 2>&1

echo "[$(date '+%F %T')] starting OmniMath 3-4 final eval/plot"
TEST_EPISODES="$TEST_EPISODES" \
N_SELECTED_POINTS="${N_SELECTED_POINTS:-11}" \
bash scripts/eval_omnimath13_to34_search_final.sh \
  > "$LOG_DIR/pipeline_eval_final.log" 2>&1

echo "[$(date '+%F %T')] OmniMath 1-3 -> 3-4 search pipeline finished"
