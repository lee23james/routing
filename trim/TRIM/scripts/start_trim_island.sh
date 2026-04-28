#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/chencheng/routing/trim/TRIM"
LOG_ROOT="${ROOT_DIR}/logs"

ISLAND_NAME="${ISLAND_NAME:-island}"

VLLM_BIN="${VLLM_BIN:-/home/chencheng/miniconda3/envs/routing/bin/vllm}"

TARGET_MODEL="${TARGET_MODEL:-/mnt/hdd2/chengcheng/qwen3-14b}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/hdd2/chengcheng/qwen3-1.7b}"
PRM_MODEL="${PRM_MODEL:-/mnt/hdd2/chengcheng/qwen2.5-math-prm-7b}"

TARGET_GPU="${TARGET_GPU:-0}"
AUX_GPU="${AUX_GPU:-1}"

TARGET_PORT="${TARGET_PORT:-30000}"
DRAFT_PORT="${DRAFT_PORT:-30001}"
PRM_PORT="${PRM_PORT:-30002}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

TARGET_MEM_UTIL="${TARGET_MEM_UTIL:-0.90}"
DRAFT_MEM_UTIL="${DRAFT_MEM_UTIL:-0.35}"
PRM_MEM_UTIL="${PRM_MEM_UTIL:-0.50}"

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
VLLM_LOG_FLAGS=(--disable-log-requests --disable-log-stats)

LOG_DIR="${LOG_ROOT}/vllm_${ISLAND_NAME}"
PID_FILE="${LOG_DIR}/pids.env"

mkdir -p "${LOG_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

wait_for_server() {
    local port="$1"
    local name="$2"
    local elapsed=0
    local health_url="http://localhost:${port}/health"

    while true; do
        if curl -sf "${health_url}" > /dev/null 2>&1; then
            log "[launch:${ISLAND_NAME}] ${name} ready on ${health_url} (${elapsed}s)"
            return 0
        fi

        if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
            log "[launch:${ISLAND_NAME}] ERROR: ${name} did not become ready within ${WAIT_TIMEOUT_SECONDS}s"
            return 1
        fi

        sleep "${POLL_SECONDS}"
        elapsed=$((elapsed + POLL_SECONDS))
    done
}

check_port_available() {
    local port="$1"
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        log "[launch:${ISLAND_NAME}] ERROR: port ${port} already has a healthy server"
        exit 1
    fi
}

start_server() {
    local gpu_id="$1"
    local name="$2"
    local model="$3"
    local port="$4"
    local mem_util="$5"
    shift 5
    local log_file="${LOG_DIR}/${name}.log"

    check_port_available "${port}"
    log "[launch:${ISLAND_NAME}] starting ${name} on GPU ${gpu_id}, port ${port}"
    nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
        "${VLLM_BIN}" serve "${model}" \
        "$@" \
        --gpu-memory-utilization "${mem_util}" \
        --enable-prefix-caching \
        "${VLLM_LOG_FLAGS[@]}" \
        --port "${port}" \
        > "${log_file}" 2>&1 &
    echo $!
}

TARGET_PID="$(start_server "${TARGET_GPU}" target "${TARGET_MODEL}" "${TARGET_PORT}" "${TARGET_MEM_UTIL}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}")"
wait_for_server "${TARGET_PORT}" "target (${TARGET_MODEL})"

DRAFT_PID="$(start_server "${AUX_GPU}" draft "${DRAFT_MODEL}" "${DRAFT_PORT}" "${DRAFT_MEM_UTIL}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}")"
wait_for_server "${DRAFT_PORT}" "draft (${DRAFT_MODEL})"

PRM_PID="$(start_server "${AUX_GPU}" prm "${PRM_MODEL}" "${PRM_PORT}" "${PRM_MEM_UTIL}" \
    --dtype auto \
    --trust-remote-code)"
wait_for_server "${PRM_PORT}" "prm (${PRM_MODEL})"

cat > "${PID_FILE}" <<EOF
ISLAND_NAME="${ISLAND_NAME}"
TARGET_PID="${TARGET_PID}"
DRAFT_PID="${DRAFT_PID}"
PRM_PID="${PRM_PID}"
TARGET_GPU="${TARGET_GPU}"
AUX_GPU="${AUX_GPU}"
TARGET_PORT="${TARGET_PORT}"
DRAFT_PORT="${DRAFT_PORT}"
PRM_PORT="${PRM_PORT}"
EOF

log "[launch:${ISLAND_NAME}] all servers ready"
log "[launch:${ISLAND_NAME}] target gpu=${TARGET_GPU} port=${TARGET_PORT} pid=${TARGET_PID}"
log "[launch:${ISLAND_NAME}] draft  gpu=${AUX_GPU} port=${DRAFT_PORT} pid=${DRAFT_PID}"
log "[launch:${ISLAND_NAME}] prm    gpu=${AUX_GPU} port=${PRM_PORT} pid=${PRM_PID}"
log "[launch:${ISLAND_NAME}] pid file: ${PID_FILE}"
