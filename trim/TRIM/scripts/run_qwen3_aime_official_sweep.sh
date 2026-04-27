#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
LOG_DIR="${ROOT_DIR}/logs"
SUPERVISOR_LOG="${LOG_DIR}/trim_agg_aime_official_sweep.log"
WORKER_SCRIPT="${ROOT_DIR}/scripts/run_qwen3_aime_official_sweep_worker.sh"
WORKER_NAMES=("gpu2" "gpu3")
WORKER_GPUS=("2" "3")
WORKER_COSTS=("3e-4,2e-4,1e-4" "2.5e-4,1.5e-4,8e-5")
POLL_SECONDS="${POLL_SECONDS:-5}"

mkdir -p "${LOG_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${SUPERVISOR_LOG}"
}

cleanup_workers() {
    local pid
    for pid in "${WORKER_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

launch_worker() {
    local worker_name="$1"
    local gpu_id="$2"
    local costs_csv="$3"
    local stdout_log="${LOG_DIR}/trim_agg_aime_official_sweep_${worker_name}.stdout.log"
    local pid

    log "[start] worker ${worker_name}: CUDA_VISIBLE_DEVICES=${gpu_id}, costs=${costs_csv}"

    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export WORKER_NAME="${worker_name}"
        export COSTS_CSV="${costs_csv}"
        exec bash "${WORKER_SCRIPT}"
    ) >> "${stdout_log}" 2>&1 &

    pid=$!
    WORKER_PIDS+=("${pid}")
    WORKER_NAMES_ACTIVE+=("${worker_name}")
    log "[info] worker ${worker_name} launched with PID ${pid}"
}

wait_for_workers() {
    local i
    local pid
    local name
    local rc
    local active

    while true; do
        active=0
        for i in "${!WORKER_PIDS[@]}"; do
            pid="${WORKER_PIDS[$i]}"
            name="${WORKER_NAMES_ACTIVE[$i]}"

            if [[ -z "${pid}" ]]; then
                continue
            fi

            if kill -0 "${pid}" 2>/dev/null; then
                active=1
                continue
            fi

            if wait "${pid}"; then
                log "[done] worker ${name} completed"
            else
                rc=$?
                log "[error] worker ${name} failed with exit code ${rc}"
                cleanup_workers
                wait || true
                exit "${rc}"
            fi

            WORKER_PIDS[$i]=""
        done

        if [[ "${active}" -eq 0 ]]; then
            return 0
        fi

        sleep "${POLL_SECONDS}"
    done
}

main() {
    local i

    if [[ ! -x "${WORKER_SCRIPT}" ]]; then
        log "[error] missing executable worker script: ${WORKER_SCRIPT}"
        exit 1
    fi

    log "[info] official AIME sweep coordinator started"
    log "[info] launching 2 workers on GPU 2 and GPU 3"

    WORKER_PIDS=()
    WORKER_NAMES_ACTIVE=()
    trap cleanup_workers INT TERM

    for i in "${!WORKER_NAMES[@]}"; do
        launch_worker "${WORKER_NAMES[$i]}" "${WORKER_GPUS[$i]}" "${WORKER_COSTS[$i]}"
    done

    wait_for_workers
    log "[done] official AIME parallel sweep complete"
}

main "$@"
