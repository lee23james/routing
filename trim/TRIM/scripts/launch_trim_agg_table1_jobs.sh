#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"

MATH_SESSION="${MATH_SESSION:-trim_agg_table1_math}"
AIME_SESSION="${AIME_SESSION:-trim_agg_table1_aime}"

cd "${ROOT_DIR}"

if screen -ls | grep -q "\\.${MATH_SESSION}[[:space:]]"; then
    echo "[skip] screen session already exists: ${MATH_SESSION}"
else
    screen -dmS "${MATH_SESSION}" bash -lc \
        'cd /home/chencheng/routing/trim/TRIM && bash scripts/run_qwen3_math_trim_agg_table1_prep.sh'
    echo "[start] launched ${MATH_SESSION}"
fi

if screen -ls | grep -q "\\.${AIME_SESSION}[[:space:]]"; then
    echo "[skip] screen session already exists: ${AIME_SESSION}"
else
    screen -dmS "${AIME_SESSION}" bash -lc \
        'cd /home/chencheng/routing/trim/TRIM && bash scripts/run_qwen3_aime_trim_agg_table1_prep.sh'
    echo "[start] launched ${AIME_SESSION}"
fi

echo "[info] current screens:"
screen -ls
