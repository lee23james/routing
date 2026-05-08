#!/usr/bin/env bash
# End-to-end MATH TRIM-RubricV2 search + final plot/table generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
TRAIN_EPISODES="${TRAIN_EPISODES:-$SRC_DIR/data/episodes/math_train_200_episodes.jsonl}"
TEST_EPISODES="${TEST_EPISODES:-$SRC_DIR/data/episodes/math500_episodes.jsonl}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

cd "$SRC_DIR"

require_rows() {
  local path="$1"
  local expected="$2"
  if [ ! -f "$path" ]; then
    echo "missing episode file: $path" >&2
    exit 1
  fi
  local rows
  rows="$(wc -l < "$path")"
  if [ "$rows" -lt "$expected" ]; then
    echo "episode file has too few rows: $path has $rows, expected at least $expected" >&2
    exit 1
  fi
  echo "$path rows=$rows"
}

require_rows "$TRAIN_EPISODES" 200
require_rows "$TEST_EPISODES" 169

if [ "$SKIP_TRAIN" != "true" ]; then
  PYTHON_BIN="$PYTHON_BIN" EPISODES_PATH="$TRAIN_EPISODES" \
    bash scripts/search_trim_rubric_v2_math_points_4gpu.sh
fi

if [ "$SKIP_EVAL" != "true" ]; then
  PYTHON_BIN="$PYTHON_BIN" TEST_EPISODES="$TEST_EPISODES" \
    bash scripts/eval_math_rubric_v2_final.sh
fi
