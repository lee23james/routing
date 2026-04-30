#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
PYTHON_BIN="/home/chencheng/miniconda3/envs/routing/bin/python"
LOG_DIR="${ROOT_DIR}/logs"
TABLE1_DIR="${ROOT_DIR}/table1_trim_agg"
BASELINE_DIR="${TABLE1_DIR}/baselines/math500_test100"
LOG_PATH="${LOG_DIR}/trim_agg_table1_math_baselines.log"

TARGET_MODEL="/mnt/hdd2/chengcheng/qwen3-14b"
DRAFT_MODEL="/mnt/hdd2/chengcheng/qwen3-1.7b"
PRM_MODEL="/mnt/hdd2/chengcheng/qwen2.5-math-prm-7b"

TARGET_SERVER_URL="${TARGET_SERVER_URL:-http://localhost:30000/v1}"
DRAFT_SERVER_URL="${DRAFT_SERVER_URL:-http://localhost:30001/v1}"
PRM_SERVER_URL="${PRM_SERVER_URL:-http://localhost:30002}"

EVAL_DATASET_NAME="math500"
EVAL_SPLIT="test_100"
THRESHOLDS="0,1"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-10}"

mkdir -p "${LOG_DIR}" "${BASELINE_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${LOG_PATH}"
}

assert_server_ready() {
    local url
    local health_url
    for url in "${TARGET_SERVER_URL}" "${DRAFT_SERVER_URL}"; do
        health_url="${url%/v1}/health"
        if ! curl -sf "${health_url}" > /dev/null; then
            log "[error] server unavailable: ${health_url}"
            exit 1
        fi
    done
}

main() {
    if [[ -f "${BASELINE_DIR}/thr_0_00_metrics.jsonl" && -f "${BASELINE_DIR}/thr_1_00_metrics.jsonl" ]]; then
        log "[skip] MATH baselines already exist in ${BASELINE_DIR}"
        exit 0
    fi

    assert_server_ready
    log "[start] MATH baselines"
    log "[info] dataset=${EVAL_DATASET_NAME}/${EVAL_SPLIT}"
    log "[info] servers: target=${TARGET_SERVER_URL}, draft=${DRAFT_SERVER_URL}, prm=${PRM_SERVER_URL}"
    log "[info] output_dir=${BASELINE_DIR}"

    (
        cd "${ROOT_DIR}"
        "${PYTHON_BIN}" TRIM_Thr.py \
            --draft_model_name "${DRAFT_MODEL}" \
            --draft_server_url "${DRAFT_SERVER_URL}" \
            --target_model_name "${TARGET_MODEL}" \
            --target_server_url "${TARGET_SERVER_URL}" \
            --prm_model_name "${PRM_MODEL}" \
            --prm_server_url "${PRM_SERVER_URL}" \
            --eval_dataset_name "${EVAL_DATASET_NAME}" \
            --eval_split "${EVAL_SPLIT}" \
            --thresholds "${THRESHOLDS}" \
            --batch_size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --output_dir "${BASELINE_DIR}" \
            --target_disable_thinking true
    ) 2>&1 | tee -a "${LOG_PATH}"

    if [[ ! -f "${BASELINE_DIR}/thr_0_00_metrics.jsonl" || ! -f "${BASELINE_DIR}/thr_1_00_metrics.jsonl" ]]; then
        log "[error] MATH baseline run finished without both endpoint files"
        exit 1
    fi

    log "[done] MATH baselines complete"
}

main "$@"
