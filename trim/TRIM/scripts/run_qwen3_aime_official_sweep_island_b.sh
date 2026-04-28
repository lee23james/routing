#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"

export TARGET_SERVER_URL="${TARGET_SERVER_URL:-http://localhost:31000/v1}"
export DRAFT_SERVER_URL="${DRAFT_SERVER_URL:-http://localhost:31001/v1}"
export PRM_SERVER_URL="${PRM_SERVER_URL:-http://localhost:31002}"
export MAX_WORKERS="${MAX_WORKERS:-16}"

exec bash "${ROOT_DIR}/scripts/run_qwen3_aime_official_sweep.sh"
