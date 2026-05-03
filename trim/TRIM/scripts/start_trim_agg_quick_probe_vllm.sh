#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/deepseek_VG/deepseek_VG/routing/routing/trim/TRIM}"
MODEL_ROOT="${MODEL_ROOT:-/home/deepseek_VG/deepseek_VG/routing/routing/models}"
VLLM_BIN="${VLLM_BIN:-/home/honghudata/deepseek_VG/maker/anaconda3/envs/routing/bin/vllm}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/vllm_trim_agg_quick_probe}"

TARGET_GPU="${TARGET_GPU:-0}"
DRAFT_GPU="${DRAFT_GPU:-0}"
PRM_GPU="${PRM_GPU:-0}"

TARGET_PORT="${TARGET_PORT:-32000}"
DRAFT_PORT="${DRAFT_PORT:-32001}"
PRM_PORT="${PRM_PORT:-32002}"

TARGET_MAX_MODEL_LEN="${TARGET_MAX_MODEL_LEN:-4096}"
DRAFT_MAX_MODEL_LEN="${DRAFT_MAX_MODEL_LEN:-4096}"
PRM_MAX_MODEL_LEN="${PRM_MAX_MODEL_LEN:-4096}"

TARGET_MEM_UTIL="${TARGET_MEM_UTIL:-0.38}"
DRAFT_MEM_UTIL="${DRAFT_MEM_UTIL:-0.12}"
PRM_MEM_UTIL="${PRM_MEM_UTIL:-0.34}"

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"

mkdir -p "${LOG_DIR}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
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
        if curl -sf "${health_url}" >/dev/null 2>&1; then
            log "[vllm:quick_probe] ${name} ready on ${health_url} (${elapsed}s)"
            return 0
        fi
        if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
            log "[vllm:quick_probe] ERROR: ${name} did not become ready within ${WAIT_TIMEOUT_SECONDS}s"
            return 1
        fi
        sleep "${POLL_SECONDS}"
        elapsed=$((elapsed + POLL_SECONDS))
    done
}

start_server() {
    local gpu="$1"
    local name="$2"
    local model="$3"
    local port="$4"
    local mem_util="$5"
    local max_model_len="$6"
    shift 6

    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
        log "[vllm:quick_probe] ${name} already healthy on port ${port}"
        return 0
    fi

    if ss -ltn "sport = :${port}" | grep -q ":${port}"; then
        log "[vllm:quick_probe] ERROR: port ${port} is already bound but not healthy"
        exit 1
    fi

    log "[vllm:quick_probe] starting ${name}: gpu=${gpu} port=${port} mem=${mem_util} max_model_len=${max_model_len}"
    nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${VLLM_BIN}" serve "${model}" \
        --served-model-name "${model}" \
        --dtype auto \
        --max-model-len "${max_model_len}" \
        --gpu-memory-utilization "${mem_util}" \
        --enable-prefix-caching \
        --disable-log-requests \
        --disable-log-stats \
        "$@" \
        --port "${port}" \
        > "${LOG_DIR}/${name}.log" 2>&1 &
    echo $! > "${LOG_DIR}/${name}.pid"
    wait_for_server "${port}" "${name}"
}

start_server "${TARGET_GPU}" target "${MODEL_ROOT}/qwen3-14b" "${TARGET_PORT}" "${TARGET_MEM_UTIL}" "${TARGET_MAX_MODEL_LEN}"
start_server "${DRAFT_GPU}" draft "${MODEL_ROOT}/qwen3-1.7b" "${DRAFT_PORT}" "${DRAFT_MEM_UTIL}" "${DRAFT_MAX_MODEL_LEN}"
start_server "${PRM_GPU}" prm "${MODEL_ROOT}/qwen2.5-math-prm-7b" "${PRM_PORT}" "${PRM_MEM_UTIL}" "${PRM_MAX_MODEL_LEN}" --trust-remote-code

log "[vllm:quick_probe] all services ready"
log "[vllm:quick_probe] target=http://localhost:${TARGET_PORT}/v1"
log "[vllm:quick_probe] draft=http://localhost:${DRAFT_PORT}/v1"
log "[vllm:quick_probe] prm=http://localhost:${PRM_PORT}"
