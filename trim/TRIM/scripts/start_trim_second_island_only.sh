#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
START_SCRIPT="${ROOT_DIR}/scripts/start_trim_island.sh"

if [[ ! -x "${START_SCRIPT}" ]]; then
    echo "missing executable script: ${START_SCRIPT}" >&2
    exit 1
fi

env \
    ISLAND_NAME=island_b \
    TARGET_GPU="${TARGET_GPU:-2}" \
    AUX_GPU="${AUX_GPU:-3}" \
    TARGET_PORT="${TARGET_PORT:-31000}" \
    DRAFT_PORT="${DRAFT_PORT:-31001}" \
    PRM_PORT="${PRM_PORT:-31002}" \
    bash "${START_SCRIPT}"
