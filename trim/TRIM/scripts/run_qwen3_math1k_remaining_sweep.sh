#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
PYTHON_BIN="/home/chencheng/miniconda3/envs/routing/bin/python"
LOG_DIR="${ROOT_DIR}/logs"
SAVE_DIR="${ROOT_DIR}/rlpolicy_checkpoints_qwen3_math1k_sweep"
OUTPUT_DIR="${ROOT_DIR}/rlpolicy_results_qwen3_math1k_sweep"
SUPERVISOR_LOG="${LOG_DIR}/trim_agg_math1k_remaining_sweep.log"

TARGET_MODEL="/mnt/hdd2/chengcheng/qwen3-14b"
DRAFT_MODEL="/mnt/hdd2/chengcheng/qwen3-1.7b"
PRM_MODEL="/mnt/hdd2/chengcheng/qwen2.5-math-prm-7b"
TARGET_SERVER_URL="${TARGET_SERVER_URL:-http://localhost:30000/v1}"
DRAFT_SERVER_URL="${DRAFT_SERVER_URL:-http://localhost:30001/v1}"
PRM_SERVER_URL="${PRM_SERVER_URL:-http://localhost:30002}"

TRAIN_DATASET_NAME="math"
TRAIN_SPLIT="train_1k"
EVAL_DATASET_NAME="math500"
EVAL_SPLIT="test_100"
TARGET_DISABLE_THINKING="true"
NUM_EPOCHS="10"
BATCH_SIZE="64"
VAL_FRACTION="0.09090909090909091"
VAL_EVERY="50"
EVAL_EVERY="50"
SEED="10"

WAIT_SECONDS="${WAIT_SECONDS:-30}"
MAX_WORKERS="${MAX_WORKERS:-16}"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}" "${OUTPUT_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${SUPERVISOR_LOG}"
}

cost_tag() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
cost = float(sys.argv[1])
mantissa, exp = f"{cost:.15e}".split("e")
mantissa = mantissa.rstrip("0").rstrip(".")
print(f"{mantissa}e{int(exp)}")
PY
}

checkpoint_path() {
    local cost="$1"
    local tag
    tag="$(cost_tag "${cost}")"
    printf '%s/%s/policy_best.pt\n' "${SAVE_DIR}" "${tag}"
}

metrics_path() {
    local cost="$1"
    local tag
    tag="$(cost_tag "${cost}")"
    printf '%s/%s/eval_metrics.jsonl\n' "${OUTPUT_DIR}" "${tag}"
}

train_log_path() {
    printf '%s/trim_agg_math1k_%s.log\n' "${LOG_DIR}" "$1"
}

eval_log_path() {
    printf '%s/trim_agg_math1k_eval_%s.log\n' "${LOG_DIR}" "$1"
}

assert_vllm_ready() {
    local url
    local health_url
    for url in "${TARGET_SERVER_URL}" "${DRAFT_SERVER_URL}" "${PRM_SERVER_URL}"; do
        health_url="${url%/v1}/health"
        if ! curl -sf "${health_url}" > /dev/null; then
            log "[error] vLLM server is unavailable: ${health_url}"
            exit 1
        fi
    done
}

wait_for_existing_eval_or_result() {
    local cost="$1"
    local metrics
    metrics="$(metrics_path "${cost}")"

    while true; do
        if [[ -f "${metrics}" ]]; then
            log "[skip] eval ${cost} already finished: ${metrics}"
            return 0
        fi

        if pgrep -af "TRIM_Agg.py --mode eval" | grep -F -- "--cost_per_token ${cost}" > /dev/null; then
            log "[wait] eval ${cost} is already running elsewhere; sleeping ${WAIT_SECONDS}s"
            sleep "${WAIT_SECONDS}"
            continue
        fi

        return 0
    done
}

run_train() {
    local cost="$1"
    local ckpt
    local train_log
    ckpt="$(checkpoint_path "${cost}")"
    train_log="$(train_log_path "${cost}")"

    if [[ -f "${ckpt}" ]]; then
        log "[skip] train ${cost} already has checkpoint: ${ckpt}"
        return 0
    fi

    assert_vllm_ready
    log "[start] train ${cost}"

    (
        cd "${ROOT_DIR}"
        "${PYTHON_BIN}" TRIM_Agg.py \
            --mode train \
            --draft_model_name "${DRAFT_MODEL}" \
            --draft_server_url "${DRAFT_SERVER_URL}" \
            --target_model_name "${TARGET_MODEL}" \
            --target_server_url "${TARGET_SERVER_URL}" \
            --prm_model_name "${PRM_MODEL}" \
            --prm_server_url "${PRM_SERVER_URL}" \
            --train_dataset_name "${TRAIN_DATASET_NAME}" \
            --train_split "${TRAIN_SPLIT}" \
            --eval_dataset_name "${EVAL_DATASET_NAME}" \
            --eval_split "${EVAL_SPLIT}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --val_fraction "${VAL_FRACTION}" \
            --val_every "${VAL_EVERY}" \
            --eval_every "${EVAL_EVERY}" \
            --save_dir "${SAVE_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --seed "${SEED}" \
            --target_disable_thinking "${TARGET_DISABLE_THINKING}" \
            --cost_per_token "${cost}" \
            --max_workers "${MAX_WORKERS}" \
            --resume true \
            --use_wandb false
    ) 2>&1 | tee -a "${train_log}"

    if [[ ! -f "${ckpt}" ]]; then
        log "[error] train ${cost} finished without checkpoint: ${ckpt}"
        exit 1
    fi

    log "[done] train ${cost}"
}

run_eval() {
    local cost="$1"
    local ckpt
    local metrics
    local eval_log
    ckpt="$(checkpoint_path "${cost}")"
    metrics="$(metrics_path "${cost}")"
    eval_log="$(eval_log_path "${cost}")"

    if [[ -f "${metrics}" ]]; then
        log "[skip] eval ${cost} already finished: ${metrics}"
        return 0
    fi

    if [[ ! -f "${ckpt}" ]]; then
        log "[error] eval ${cost} missing checkpoint: ${ckpt}"
        exit 1
    fi

    wait_for_existing_eval_or_result "${cost}"
    if [[ -f "${metrics}" ]]; then
        return 0
    fi

    assert_vllm_ready
    log "[start] eval ${cost}"

    (
        cd "${ROOT_DIR}"
        "${PYTHON_BIN}" TRIM_Agg.py \
            --mode eval \
            --draft_model_name "${DRAFT_MODEL}" \
            --draft_server_url "${DRAFT_SERVER_URL}" \
            --target_model_name "${TARGET_MODEL}" \
            --target_server_url "${TARGET_SERVER_URL}" \
            --prm_model_name "${PRM_MODEL}" \
            --prm_server_url "${PRM_SERVER_URL}" \
            --eval_dataset_name "${EVAL_DATASET_NAME}" \
            --eval_split "${EVAL_SPLIT}" \
            --checkpoint "${ckpt}" \
            --output_dir "${OUTPUT_DIR}" \
            --target_disable_thinking "${TARGET_DISABLE_THINKING}" \
            --cost_per_token "${cost}" \
            --max_workers "${MAX_WORKERS}"
    ) 2>&1 | tee -a "${eval_log}"

    if [[ ! -f "${metrics}" ]]; then
        log "[error] eval ${cost} finished without metrics: ${metrics}"
        exit 1
    fi

    log "[done] eval ${cost}"
}

main() {
    log "[info] remaining sweep supervisor started"
    log "[info] train dataset: ${TRAIN_DATASET_NAME}/${TRAIN_SPLIT}; eval dataset: ${EVAL_DATASET_NAME}/${EVAL_SPLIT}"
    log "[info] train config: num_epochs=${NUM_EPOCHS}, batch_size=${BATCH_SIZE}, val_fraction=${VAL_FRACTION}, val_every=${VAL_EVERY}, eval_every=${EVAL_EVERY}, seed=${SEED}"
    log "[info] models: target=${TARGET_MODEL}, draft=${DRAFT_MODEL}, prm=${PRM_MODEL}"
    log "[info] servers: target=${TARGET_SERVER_URL}, draft=${DRAFT_SERVER_URL}, prm=${PRM_SERVER_URL}, max_workers=${MAX_WORKERS}"

    run_eval "6e-4"

    for cost in "5e-4" "4e-4" "3e-4"; do
        run_train "${cost}"
        run_eval "${cost}"
    done

    log "[done] remaining sweep complete"
}

main "$@"
