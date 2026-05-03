#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/honghudata/deepseek_VG/maker/anaconda3/envs/routing/bin/python}"

TRAIN_EPISODES="${TRAIN_EPISODES:-${SRC_DIR}/data/episodes/aime_train_episodes.jsonl}"
EVAL_EPISODES="${EVAL_EPISODES:-${SRC_DIR}/data/episodes/aime_test_episodes.jsonl}"

RUN_NAME="${RUN_NAME:-trim_agg_aime_quick_probe_offline}"
CKPT_ROOT="${CKPT_ROOT:-${SRC_DIR}/checkpoints/${RUN_NAME}}"
RESULT_ROOT="${RESULT_ROOT:-${SRC_DIR}/results/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${SRC_DIR}/logs/${RUN_NAME}}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${RESULT_ROOT}/summary.jsonl}"
SUMMARY_MD="${SUMMARY_MD:-${RESULT_ROOT}/summary.md}"
TARGETS_JSON="${TARGETS_JSON:-${RESULT_ROOT}/targets.json}"

LAM_VALUES="${LAM_VALUES:-4e-4 5e-4 6e-4 7e-4 8e-4 9e-4}"
TARGET_CPTS="${TARGET_CPTS:-0.50 0.80 0.95}"
EPOCHS="${EPOCHS:-2}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-128}"
EVAL_LIMIT="${EVAL_LIMIT:-all}"
MAX_STEPS="${MAX_STEPS:-30}"
DEVICE="${DEVICE:-cuda:0}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"

mkdir -p "${CKPT_ROOT}" "${RESULT_ROOT}" "${LOG_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${LOG_DIR}/run.log"
}

lam_tag() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from eval.trim_agg_quick_probe import format_lam_tag
print(format_lam_tag(sys.argv[1]))
PY
}

run_summary() {
    "${PYTHON_BIN}" -m eval.trim_agg_quick_probe summary \
        --episodes_path "${EVAL_EPISODES}" \
        --results_dir "${RESULT_ROOT}" \
        --lam_values "${LAM_VALUES}" \
        --targets "${TARGET_CPTS}" \
        --limit "${EVAL_LIMIT}" \
        --out_jsonl "${SUMMARY_JSONL}" \
        --out_md "${SUMMARY_MD}" \
        --out_targets "${TARGETS_JSON}" \
        2>&1 | tee -a "${LOG_DIR}/summary.log"
}

run_one() {
    local lam="$1"
    local tag save_dir result_dir checkpoint eval_json train_log eval_log

    tag="$(lam_tag "${lam}")"
    save_dir="${CKPT_ROOT}/${tag}"
    result_dir="${RESULT_ROOT}/${tag}"
    checkpoint="${save_dir}/best.pt"
    eval_json="${result_dir}/eval.json"
    train_log="${LOG_DIR}/train_${tag}.log"
    eval_log="${LOG_DIR}/eval_${tag}.log"

    mkdir -p "${save_dir}" "${result_dir}"

    if [[ "${FORCE}" == "1" || ! -f "${checkpoint}" ]]; then
        log "[start] train lambda=${lam} tag=${tag} epochs=${EPOCHS} episodes_per_epoch=${EPISODES_PER_EPOCH}"
        (
            cd "${SRC_DIR}"
            CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m router.train_ppo \
                --episodes_path "${TRAIN_EPISODES}" \
                --lam "${lam}" \
                --num_epochs "${EPOCHS}" \
                --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
                --device "${DEVICE}" \
                --save_dir "${save_dir}" \
                --tag "trim_agg_aime_quick_probe"
        ) 2>&1 | tee -a "${train_log}"
    else
        log "[skip] train lambda=${lam}; checkpoint exists: ${checkpoint}"
    fi

    if [[ ! -f "${checkpoint}" ]]; then
        log "[error] missing checkpoint after train: ${checkpoint}"
        exit 1
    fi

    if [[ "${FORCE}" == "1" || ! -f "${eval_json}" ]]; then
        log "[start] eval lambda=${lam} tag=${tag} checkpoint=${checkpoint}"
        (
            cd "${SRC_DIR}"
            CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m eval.trim_agg_quick_probe eval \
                --episodes_path "${EVAL_EPISODES}" \
                --checkpoint "${checkpoint}" \
                --out "${eval_json}" \
                --lam "${lam}" \
                --device "${DEVICE}" \
                --max_steps "${MAX_STEPS}" \
                --limit "${EVAL_LIMIT}"
        ) 2>&1 | tee -a "${eval_log}"
    else
        log "[skip] eval lambda=${lam}; result exists: ${eval_json}"
    fi

    run_summary
}

main() {
    log "[info] offline TRIM-Agg AIME quick probe started"
    log "[info] run_name=${RUN_NAME}"
    log "[info] train_episodes=${TRAIN_EPISODES}"
    log "[info] eval_episodes=${EVAL_EPISODES}"
    log "[info] lambda_values=${LAM_VALUES}"
    log "[info] target_cpts=${TARGET_CPTS}"
    log "[info] epochs=${EPOCHS}, episodes_per_epoch=${EPISODES_PER_EPOCH}, eval_limit=${EVAL_LIMIT}"
    log "[info] device=${DEVICE}, gpu=${GPU}"

    local lam
    for lam in ${LAM_VALUES}; do
        run_one "${lam}"
    done

    log "[done] offline TRIM-Agg AIME quick probe complete"
    log "[done] summary=${SUMMARY_MD}"
    log "[done] targets=${TARGETS_JSON}"
}

main "$@"
