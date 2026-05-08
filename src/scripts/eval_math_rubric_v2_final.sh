#!/usr/bin/env bash
# Final compare/eval for MATH TRIM-Agg + TRIM-Rubric + TRIM-RubricV2 point search.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
TEST_EPISODES="${TEST_EPISODES:-data/episodes/math500_episodes.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/trim_rubric_v2_math200_point_search/final}"
AGG_GLOB="${AGG_GLOB:-checkpoints/trim_agg_math200_point_search_*/*.pt}"
RUBRIC_GLOB="${RUBRIC_GLOB:-checkpoints/trim_rubric_math200_point_search_*/*.pt}"
RUBRIC_V2_GLOB="${RUBRIC_V2_GLOB:-checkpoints/trim_rubric_v2_math200_point_search_*/*.pt}"
N_SELECTED_POINTS="${N_SELECTED_POINTS:-8}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m eval.plot_trim_agg_baseline \
  --datasets math500 \
  --math500_episodes "$TEST_EPISODES" \
  --agg_checkpoint_glob "$AGG_GLOB" \
  --rubric_checkpoint_glob "$RUBRIC_GLOB" \
  --rubric_v2_checkpoint_glob "$RUBRIC_V2_GLOB" \
  --checkpoint_glob "$AGG_GLOB" \
  --output_dir "$OUTPUT_DIR" \
  --n_selected_points "$N_SELECTED_POINTS" \
  --device "$DEVICE"
