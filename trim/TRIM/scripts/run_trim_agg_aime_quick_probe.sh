#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/deepseek_VG/deepseek_VG/routing/routing/trim/TRIM}"
PYTHON_BIN="${PYTHON_BIN:-/home/honghudata/deepseek_VG/maker/anaconda3/envs/routing/bin/python}"
HELPER="${ROOT_DIR}/scripts/trim_agg_quick_probe.py"

MODEL_ROOT="${MODEL_ROOT:-/home/deepseek_VG/deepseek_VG/routing/routing/models}"
TARGET_MODEL="${TARGET_MODEL:-${MODEL_ROOT}/qwen3-14b}"
DRAFT_MODEL="${DRAFT_MODEL:-${MODEL_ROOT}/qwen3-1.7b}"
PRM_MODEL="${PRM_MODEL:-${MODEL_ROOT}/qwen2.5-math-prm-7b}"

TARGET_SERVER_URL="${TARGET_SERVER_URL:-http://localhost:32000/v1}"
DRAFT_SERVER_URL="${DRAFT_SERVER_URL:-http://localhost:32001/v1}"
PRM_SERVER_URL="${PRM_SERVER_URL:-http://localhost:32002}"

RUN_NAME="${RUN_NAME:-aime_quick_probe_4e-4_to_9e-4_2ep_eval128}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/rlpolicy_checkpoints_${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/rlpolicy_results_${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/quick_probe_data/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_NAME}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/summary.jsonl}"

COSTS_CSV="${COSTS_CSV:-4e-4,5e-4,6e-4,7e-4,8e-4,9e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
VAL_EVERY="${VAL_EVERY:-4}"
EVAL_EVERY="${EVAL_EVERY:-9999}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_TOKENS_PER_STEP="${MAX_TOKENS_PER_STEP:-768}"
MAX_STEPS="${MAX_STEPS:-18}"
TRAIN_LIMIT="${TRAIN_LIMIT:-all}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
SEED="${SEED:-10}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"

SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${ROOT_DIR}/math_eval/data}"
EPISODE_BASELINE="${EPISODE_BASELINE:-/home/deepseek_VG/deepseek_VG/routing/routing/src/data/episodes/aime_test_episodes.jsonl}"

mkdir -p "${SAVE_DIR}" "${OUTPUT_DIR}" "${DATA_DIR}" "${LOG_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export TOKENIZERS_PARALLELISM="false"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${LOG_DIR}/run.log"
}

health_url() {
    local url="$1"
    printf '%s/health\n' "${url%/v1}"
}

assert_server_ready() {
    local url
    for url in "${TARGET_SERVER_URL}" "${DRAFT_SERVER_URL}" "${PRM_SERVER_URL}"; do
        if ! curl -sf "$(health_url "${url}")" >/dev/null; then
            log "[error] server unavailable: $(health_url "${url}")"
            exit 1
        fi
    done
}

cost_tag() {
    "${PYTHON_BIN}" - "$1" <<'PY'
from decimal import Decimal
import sys
d = Decimal(sys.argv[1]).normalize()
m, e = f"{d:.15E}".split("E")
print(f"{m.rstrip('0').rstrip('.')}e{int(e)}")
PY
}

run_summary() {
    "${PYTHON_BIN}" "${HELPER}" summary \
        --result_root "${OUTPUT_DIR}" \
        --costs "${COSTS_CSV}" \
        --srm_acc 0.16597510373443983 \
        --lrm_acc 0.1908713692946058 \
        --lrm_avg_tokens 3059.4211618257264 \
        --episode_baseline "${EPISODE_BASELINE}" \
        --eval_limit "${EVAL_LIMIT}" \
        --out "${SUMMARY_PATH}" \
        | tee -a "${LOG_DIR}/run.log"
}

run_train_eval() {
    local cost="$1"
    local tag ckpt metrics train_log eval_log
    tag="$(cost_tag "${cost}")"
    ckpt="${SAVE_DIR}/${tag}/policy_best.pt"
    metrics="${OUTPUT_DIR}/${tag}/eval_metrics.jsonl"
    train_log="${LOG_DIR}/train_${tag}.log"
    eval_log="${LOG_DIR}/eval_${tag}.log"

    if [[ ! -f "${ckpt}" ]]; then
        log "[start] train cost=${cost} tag=${tag} epochs=${NUM_EPOCHS}"
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
                --data_dir "${DATA_DIR}" \
                --train_dataset_name aime \
                --train_split train \
                --eval_dataset_name aime \
                --eval_split test \
                --num_epochs "${NUM_EPOCHS}" \
                --batch_size "${BATCH_SIZE}" \
                --mini_batch_size "${MINI_BATCH_SIZE}" \
                --val_fraction "${VAL_FRACTION}" \
                --val_every "${VAL_EVERY}" \
                --eval_every "${EVAL_EVERY}" \
                --save_dir "${SAVE_DIR}" \
                --output_dir "${OUTPUT_DIR}" \
                --seed "${SEED}" \
                --target_disable_thinking true \
                --draft_disable_thinking true \
                --cost_per_token "${cost}" \
                --max_workers "${MAX_WORKERS}" \
                --max_tokens_per_step "${MAX_TOKENS_PER_STEP}" \
                --max_steps "${MAX_STEPS}" \
                --policy_device "${POLICY_DEVICE}" \
                --resume true \
                --use_wandb false
        ) 2>&1 | tee -a "${train_log}"
    else
        log "[skip] train cost=${cost}; checkpoint exists: ${ckpt}"
    fi

    if [[ ! -f "${ckpt}" ]]; then
        log "[error] missing checkpoint after train: ${ckpt}"
        exit 1
    fi

    if [[ ! -f "${metrics}" ]]; then
        log "[start] eval cost=${cost} tag=${tag}"
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
                --data_dir "${DATA_DIR}" \
                --eval_dataset_name aime \
                --eval_split test \
                --checkpoint "${ckpt}" \
                --output_dir "${OUTPUT_DIR}" \
                --target_disable_thinking true \
                --draft_disable_thinking true \
                --cost_per_token "${cost}" \
                --max_workers "${MAX_WORKERS}" \
                --max_tokens_per_step "${MAX_TOKENS_PER_STEP}" \
                --max_steps "${MAX_STEPS}" \
                --batch_size "${BATCH_SIZE}" \
                --policy_device "${POLICY_DEVICE}"
        ) 2>&1 | tee -a "${eval_log}"
    else
        log "[skip] eval cost=${cost}; metrics exists: ${metrics}"
    fi

    run_summary
}

main() {
    log "[info] TRIM-Agg AIME quick probe started"
    log "[info] run_name=${RUN_NAME}"
    log "[info] costs=${COSTS_CSV}"
    log "[info] train config: epochs=${NUM_EPOCHS}, batch_size=${BATCH_SIZE}, val_every=${VAL_EVERY}, eval_limit=${EVAL_LIMIT}"
    log "[info] servers: target=${TARGET_SERVER_URL}, draft=${DRAFT_SERVER_URL}, prm=${PRM_SERVER_URL}"
    assert_server_ready

    log "[info] preparing quick-probe data in ${DATA_DIR}"
    "${PYTHON_BIN}" "${HELPER}" prepare-data \
        --source_root "${SOURCE_DATA_ROOT}" \
        --output_root "${DATA_DIR}" \
        --train_limit "${TRAIN_LIMIT}" \
        --eval_limit "${EVAL_LIMIT}" \
        | tee -a "${LOG_DIR}/run.log"

    IFS=',' read -r -a costs <<< "${COSTS_CSV}"
    local cost
    for cost in "${costs[@]}"; do
        cost="${cost//[[:space:]]/}"
        [[ -z "${cost}" ]] && continue
        run_train_eval "${cost}"
    done

    log "[done] TRIM-Agg AIME quick probe complete"
    log "[done] summary: ${SUMMARY_PATH}"
}

main "$@"
