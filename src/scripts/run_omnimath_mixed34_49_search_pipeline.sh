#!/usr/bin/env bash
# Reuse OmniMath 1-3 trained checkpoints, build mixed 3-4/4-9 test, then run final eval.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd2/chencheng/~/envs/trim/bin/python}"
OMNIMATH34_EPISODES="${OMNIMATH34_EPISODES:-data/episodes/omnimath_diff3_4_test_200_episodes.jsonl}"
OMNIMATH49_EPISODES="${OMNIMATH49_EPISODES:-data/episodes/omnimath_diff4_9_stratified_test_100_episodes.jsonl}"
MIXED_EPISODES="${MIXED_EPISODES:-data/episodes/omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl}"
MANIFEST="${MANIFEST:-results/trim_omnimath13_to_mixed34_49_search/mixed_test_manifest.json}"
EXPECTED_34="${EXPECTED_34:-200}"
EXPECTED_49="${EXPECTED_49:-100}"
EXPECTED_MIXED="${EXPECTED_MIXED:-200}"
MIXED_SEED="${MIXED_SEED:-1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-logs/trim_omnimath13_to_mixed34_49_search}"

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
    local n34
    local n49
    n34="$(line_count "$OMNIMATH34_EPISODES")"
    n49="$(line_count "$OMNIMATH49_EPISODES")"
    echo "[$(date '+%F %T')] episodes: omnimath34=${n34}/${EXPECTED_34}, omnimath49=${n49}/${EXPECTED_49}"
    if [ "$n34" -ge "$EXPECTED_34" ] && [ "$n49" -ge "$EXPECTED_49" ]; then
      return
    fi
    sleep "$SLEEP_SECONDS"
  done
}

wait_for_generation_services_to_exit() {
  while pgrep -f 'generate_omnimath49_test_episodes_20k|server_vllm|data.generate_episodes' >/dev/null; do
    echo "[$(date '+%F %T')] generation/vLLM services still running; waiting before offline eval"
    sleep "$SLEEP_SECONDS"
  done
}

wait_for_episodes
wait_for_generation_services_to_exit

echo "[$(date '+%F %T')] building mixed OmniMath test episodes"
"$PYTHON_BIN" scripts/build_omnimath_mixed34_49_test.py \
  --omnimath34_episodes "$OMNIMATH34_EPISODES" \
  --omnimath49_episodes "$OMNIMATH49_EPISODES" \
  --output "$MIXED_EPISODES" \
  --manifest "$MANIFEST" \
  --seed "$MIXED_SEED" \
  > "$LOG_DIR/build_mixed.log" 2>&1

if [ "$(line_count "$MIXED_EPISODES")" -lt "$EXPECTED_MIXED" ]; then
  echo "Mixed OmniMath test episodes are incomplete: $MIXED_EPISODES has $(line_count "$MIXED_EPISODES") rows" >&2
  exit 1
fi

echo "[$(date '+%F %T')] starting mixed OmniMath final eval/plot"
TEST_EPISODES="$MIXED_EPISODES" \
N_SELECTED_POINTS="${N_SELECTED_POINTS:-11}" \
bash scripts/eval_omnimath_mixed34_49_search_final.sh \
  > "$LOG_DIR/pipeline_eval_final.log" 2>&1

echo "[$(date '+%F %T')] OmniMath 1-3 -> mixed 3-4/4-9 eval pipeline finished"
