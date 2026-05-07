#!/usr/bin/env bash
# Final compare/eval for GPQA TRIM-Agg + TRIM-Rubric point search.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
TEST_EPISODES="${TEST_EPISODES:-data/episodes/gpqa_diamond_test_100_episodes.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/trim_gpqa_main200_diamond100_search/final}"
AGG_GLOB="${AGG_GLOB:-checkpoints/trim_agg_gpqa_main200_point_search_*/*.pt}"
RUBRIC_GLOB="${RUBRIC_GLOB:-checkpoints/trim_rubric_gpqa_main200_point_search_*/*.pt}"
N_SELECTED_POINTS="${N_SELECTED_POINTS:-8}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m eval.plot_trim_agg_baseline \
  --datasets gpqa_diamond_test_100 \
  --gpqa_diamond_episodes "$TEST_EPISODES" \
  --agg_checkpoint_glob "$AGG_GLOB" \
  --rubric_checkpoint_glob "$RUBRIC_GLOB" \
  --checkpoint_glob "$AGG_GLOB" \
  --output_dir "$OUTPUT_DIR" \
  --n_selected_points "$N_SELECTED_POINTS" \
  --device "$DEVICE"
