#!/usr/bin/env bash
# Evaluate AIME Part-search Agg/Rubric checkpoints and write MATH-style plots/tables.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
N_SELECTED_POINTS="${N_SELECTED_POINTS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-results/trim_aime_part1_204_part2_74_search/final}"
AIME_PART2_EPISODES="${AIME_PART2_EPISODES:-data/episodes/aime_2020_2024_part2_test_episodes.jsonl}"
AGG_CHECKPOINT_GLOB="${AGG_CHECKPOINT_GLOB:-checkpoints/trim_agg_aime_part1_204_point_search_*/*.pt}"
RUBRIC_CHECKPOINT_GLOB="${RUBRIC_CHECKPOINT_GLOB:-checkpoints/trim_rubric_aime_part1_204_point_search_*/*.pt}"

if [ ! -f "$AIME_PART2_EPISODES" ]; then
  echo "AIME Part II episode file not found: $AIME_PART2_EPISODES" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m eval.plot_trim_agg_baseline \
  --datasets aime_2020_2024_part2_test \
  --aime_part2_episodes "$AIME_PART2_EPISODES" \
  --agg_checkpoint_glob "$AGG_CHECKPOINT_GLOB" \
  --rubric_checkpoint_glob "$RUBRIC_CHECKPOINT_GLOB" \
  --output_dir "$OUTPUT_DIR" \
  --n_selected_points "$N_SELECTED_POINTS" \
  --device "$DEVICE"
