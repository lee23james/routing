#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
PYTHON_BIN="/home/chencheng/miniconda3/envs/routing/bin/python"
MANIFEST_PATH="${ROOT_DIR}/scripts/trim_agg_table1_manifest.json"
OUTPUT_DIR="${ROOT_DIR}/table1_trim_agg/output"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/compute_trim_agg_table1.py \
    --manifest "${MANIFEST_PATH}" \
    --output_dir "${OUTPUT_DIR}"
