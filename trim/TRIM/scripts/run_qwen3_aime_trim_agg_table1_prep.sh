#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
PYTHON_BIN="/home/chencheng/miniconda3/envs/routing/bin/python"
LOG_DIR="${ROOT_DIR}/logs"
TABLE1_DIR="${ROOT_DIR}/table1_trim_agg"
BASELINE_DIR="${TABLE1_DIR}/baselines/aime_test"
LOG_PATH="${LOG_DIR}/trim_agg_table1_aime_prep.log"

TARGET_MODEL="/mnt/hdd2/chengcheng/qwen3-14b"
DRAFT_MODEL="/mnt/hdd2/chengcheng/qwen3-1.7b"
PRM_MODEL="/mnt/hdd2/chengcheng/qwen2.5-math-prm-7b"

TARGET_SERVER_URL="${TARGET_SERVER_URL:-http://localhost:31000/v1}"
DRAFT_SERVER_URL="${DRAFT_SERVER_URL:-http://localhost:31001/v1}"
PRM_SERVER_URL="${PRM_SERVER_URL:-http://localhost:31002}"

EVAL_DATASET_NAME="aime"
EVAL_SPLIT="test"
THRESHOLDS="0,1"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-10}"

FILL_COSTS_CSV="${FILL_COSTS_CSV:-8.5e-5,9e-5,9.5e-5}"
WORKER_NAME="${WORKER_NAME:-table1_fill}"
MAX_WORKERS="${MAX_WORKERS:-16}"

mkdir -p "${LOG_DIR}" "${BASELINE_DIR}" "${TABLE1_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

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

run_baselines() {
    if [[ -f "${BASELINE_DIR}/thr_0_00_metrics.jsonl" && -f "${BASELINE_DIR}/thr_1_00_metrics.jsonl" ]]; then
        log "[skip] AIME baselines already exist in ${BASELINE_DIR}"
        return 0
    fi

    assert_server_ready
    log "[start] AIME baselines"
    log "[info] dataset=${EVAL_DATASET_NAME}/${EVAL_SPLIT}"
    log "[info] servers: target=${TARGET_SERVER_URL}, draft=${DRAFT_SERVER_URL}, prm=${PRM_SERVER_URL}"

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
        log "[error] AIME baseline run finished without both endpoint files"
        exit 1
    fi

    log "[done] AIME baselines complete"
}

run_fill_points() {
    log "[start] AIME fill sweep: ${FILL_COSTS_CSV}"
    (
        cd "${ROOT_DIR}"
        TARGET_SERVER_URL="${TARGET_SERVER_URL}" \
        DRAFT_SERVER_URL="${DRAFT_SERVER_URL}" \
        PRM_SERVER_URL="${PRM_SERVER_URL}" \
        COSTS_CSV="${FILL_COSTS_CSV}" \
        WORKER_NAME="${WORKER_NAME}" \
        MAX_WORKERS="${MAX_WORKERS}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        bash scripts/run_qwen3_aime_official_sweep_worker.sh
    ) 2>&1 | tee -a "${LOG_PATH}"
    log "[done] AIME fill sweep complete"
}

main() {
    run_baselines
    run_fill_points
}

main "$@"
