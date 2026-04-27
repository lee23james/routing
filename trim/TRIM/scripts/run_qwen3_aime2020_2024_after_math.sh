#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
LOG_DIR="${ROOT_DIR}/logs"
WAIT_LOG="${LOG_DIR}/trim_agg_aime2020_2024_after_math.log"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

MATH_RESULTS=(
    "${ROOT_DIR}/rlpolicy_results_qwen3_math1k_sweep/8e-4/eval_metrics.jsonl"
    "${ROOT_DIR}/rlpolicy_results_qwen3_math1k_sweep/5.999999999999999e-4/eval_metrics.jsonl"
    "${ROOT_DIR}/rlpolicy_results_qwen3_math1k_sweep/5e-4/eval_metrics.jsonl"
    "${ROOT_DIR}/rlpolicy_results_qwen3_math1k_sweep/4e-4/eval_metrics.jsonl"
    "${ROOT_DIR}/rlpolicy_results_qwen3_math1k_sweep/3e-4/eval_metrics.jsonl"
)

mkdir -p "${LOG_DIR}"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${WAIT_LOG}"
}

math_complete() {
    local path
    for path in "${MATH_RESULTS[@]}"; do
        if [[ ! -f "${path}" ]]; then
            return 1
        fi
    done
    return 0
}

main() {
    log "[info] waiting for math sweep to finish before starting AIME 2020-2024"
    until math_complete; do
        log "[wait] math sweep incomplete; sleeping ${WAIT_SECONDS}s"
        sleep "${WAIT_SECONDS}"
    done

    log "[start] math sweep complete; launching AIME 2020-2024 sweep"
    exec bash "${ROOT_DIR}/scripts/run_qwen3_aime2020_2024_sweep.sh"
}

main "$@"
