#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VLLM_BIN="${VLLM_BIN:-/home/honghudata/deepseek_VG/maker/anaconda3/envs/routing/bin/vllm}"
MODEL_ROOT="${MODEL_ROOT:-/home/deepseek_VG/deepseek_VG/routing/routing/models}"
LOG_DIR="${LOG_DIR:-logs/vllm_three_card}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"

mkdir -p "$LOG_DIR"

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

    while true; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            log "[vllm] ${name} ready on port ${port} (${elapsed}s)"
            return 0
        fi
        if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
            log "[vllm] ERROR: ${name} on port ${port} did not become ready"
            return 1
        fi
        sleep "$POLL_SECONDS"
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
        log "[vllm] ${name} already healthy on port ${port}"
        return 0
    fi

    log "[vllm] starting ${name}: gpu=${gpu} port=${port} max_model_len=${max_model_len}"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_BIN" serve "$model" \
        --served-model-name "$model" \
        --dtype auto \
        --max-model-len "$max_model_len" \
        --gpu-memory-utilization "$mem_util" \
        --enable-prefix-caching \
        --disable-log-requests \
        --disable-log-stats \
        "$@" \
        --port "$port" \
        > "${LOG_DIR}/${name}.log" 2>&1 &
    echo $! > "${LOG_DIR}/${name}.pid"
    wait_for_server "$port" "$name"
}

TARGET_MAX_MODEL_LEN="${TARGET_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
DRAFT_MAX_MODEL_LEN="${DRAFT_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
PRM_MAX_MODEL_LEN="${PRM_MAX_MODEL_LEN:-4096}"

start_server 0 math_target "${MODEL_ROOT}/qwen3-14b" 30000 "${MATH_TARGET_MEM_UTIL:-0.90}" "$TARGET_MAX_MODEL_LEN"
start_server 1 math_draft "${MODEL_ROOT}/qwen3-1.7b" 30001 "${MATH_DRAFT_MEM_UTIL:-0.30}" "$DRAFT_MAX_MODEL_LEN"
start_server 1 math_prm "${MODEL_ROOT}/qwen2.5-math-prm-7b" 30002 "${MATH_PRM_MEM_UTIL:-0.45}" "$PRM_MAX_MODEL_LEN" --trust-remote-code
start_server 7 aime_target "${MODEL_ROOT}/qwen3-14b" 31000 "${AIME_TARGET_MEM_UTIL:-0.45}" "$TARGET_MAX_MODEL_LEN"

log "[vllm] all requested TRIM-Rubric services ready"
