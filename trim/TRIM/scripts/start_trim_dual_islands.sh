#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
START_SCRIPT="${ROOT_DIR}/scripts/start_trim_island.sh"

if [[ ! -x "${START_SCRIPT}" ]]; then
    echo "missing executable script: ${START_SCRIPT}" >&2
    exit 1
fi

run_island() {
    env "$@" bash "${START_SCRIPT}"
}

run_island \
    ISLAND_NAME=island_a \
    TARGET_GPU=0 \
    AUX_GPU=1 \
    TARGET_PORT=30000 \
    DRAFT_PORT=30001 \
    PRM_PORT=30002

run_island \
    ISLAND_NAME=island_b \
    TARGET_GPU=2 \
    AUX_GPU=3 \
    TARGET_PORT=31000 \
    DRAFT_PORT=31001 \
    PRM_PORT=31002
