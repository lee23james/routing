#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
PYTHON_BIN="/home/chencheng/miniconda3/envs/routing/bin/python"
LOG_DIR="${ROOT_DIR}/logs"
SAVE_DIR="${ROOT_DIR}/rlpolicy_checkpoints_qwen3_aime_official_sweep"
OUTPUT_DIR="${ROOT_DIR}/rlpolicy_results_qwen3_aime_official_sweep"
WORKER_NAME="${WORKER_NAME:-worker}"
SUPERVISOR_LOG="${LOG_DIR}/trim_agg_aime_official_sweep_${WORKER_NAME}.log"

TARGET_MODEL="/mnt/hdd2/chengcheng/qwen3-14b"
DRAFT_MODEL="/mnt/hdd2/chengcheng/qwen3-1.7b"
PRM_MODEL="/mnt/hdd2/chengcheng/qwen2.5-math-prm-7b"

TRAIN_DATASET_NAME="aime"
TRAIN_SPLIT="train"
EVAL_DATASET_NAME="aime"
EVAL_SPLIT="test"
TARGET_DISABLE_THINKING="true"
NUM_EPOCHS="10"
BATCH_SIZE="64"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
VAL_EVERY="50"
EVAL_EVERY="50"
SEED="10"
COSTS_CSV="${COSTS_CSV:-3e-4,2.5e-4,2e-4,1.5e-4,1e-4,8e-5}"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}" "${OUTPUT_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export TOKENIZERS_PARALLELISM="false"

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
    printf '%s/trim_agg_aime_%s.log\n' "${LOG_DIR}" "$1"
}

eval_log_path() {
    printf '%s/trim_agg_aime_eval_%s.log\n' "${LOG_DIR}" "$1"
}

assert_vllm_ready() {
    local port
    for port in 30000 30001 30002; do
        if ! curl -sf "http://localhost:${port}/health" > /dev/null; then
            log "[error] vLLM server on port ${port} is unavailable."
            exit 1
        fi
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
            --target_model_name "${TARGET_MODEL}" \
            --prm_model_name "${PRM_MODEL}" \
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

    assert_vllm_ready
    log "[start] eval ${cost}"

    (
        cd "${ROOT_DIR}"
        "${PYTHON_BIN}" TRIM_Agg.py \
            --mode eval \
            --draft_model_name "${DRAFT_MODEL}" \
            --target_model_name "${TARGET_MODEL}" \
            --prm_model_name "${PRM_MODEL}" \
            --eval_dataset_name "${EVAL_DATASET_NAME}" \
            --eval_split "${EVAL_SPLIT}" \
            --checkpoint "${ckpt}" \
            --output_dir "${OUTPUT_DIR}" \
            --target_disable_thinking "${TARGET_DISABLE_THINKING}" \
            --cost_per_token "${cost}"
    ) 2>&1 | tee -a "${eval_log}"

    if [[ ! -f "${metrics}" ]]; then
        log "[error] eval ${cost} finished without metrics: ${metrics}"
        exit 1
    fi

    log "[done] eval ${cost}"
}

main() {
    local raw_costs=()
    local cost

    IFS=',' read -r -a raw_costs <<< "${COSTS_CSV}"

    if [[ "${#raw_costs[@]}" -eq 0 ]]; then
        log "[error] no costs assigned to worker ${WORKER_NAME}"
        exit 1
    fi

    log "[info] official AIME worker started"
    log "[info] worker=${WORKER_NAME}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    log "[info] train dataset: ${TRAIN_DATASET_NAME}/${TRAIN_SPLIT}; eval dataset: ${EVAL_DATASET_NAME}/${EVAL_SPLIT}"
    log "[info] train config: num_epochs=${NUM_EPOCHS}, batch_size=${BATCH_SIZE}, val_fraction=${VAL_FRACTION}, val_every=${VAL_EVERY}, eval_every=${EVAL_EVERY}, seed=${SEED}"
    log "[info] models: target=${TARGET_MODEL}, draft=${DRAFT_MODEL}, prm=${PRM_MODEL}"
    log "[info] cost grid: ${raw_costs[*]}"

    for cost in "${raw_costs[@]}"; do
        run_train "${cost}"
        run_eval "${cost}"
    done

    log "[done] official AIME worker complete"
}

main "$@"
