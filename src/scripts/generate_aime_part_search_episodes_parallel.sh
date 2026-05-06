#!/usr/bin/env bash
# Generate/resume AIME Part-search episodes with per-problem SRM/LRM parallel calls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_DIR="$SRC_DIR/vllm"

PYTHON_BIN="${PYTHON_BIN:-/home/chencheng/miniconda3/envs/trim/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/chencheng/models}"
SRM_MODEL="${SRM_MODEL:-$MODEL_ROOT/qwen3-1.7b}"
LRM_MODEL="${LRM_MODEL:-$MODEL_ROOT/qwen3-14b}"
PRM_MODEL="${PRM_MODEL:-$MODEL_ROOT/qwen2.5-math-prm-7b}"

SRM_PORT="${SRM_PORT:-4003}"
LRM_PORT="${LRM_PORT:-4001}"
SRM_GPU="${SRM_GPU:-1}"
LRM_GPUS="${LRM_GPUS:-2,3}"
PRM_DEVICE="${PRM_DEVICE:-cuda:0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
GEN_MAX_WORKERS="${GEN_MAX_WORKERS:-1}"

DATASETS=(
  "aime_2010_2024_part1_train"
  "aime_2020_2024_part2_test"
)

cd "$SRC_DIR"
mkdir -p logs/trim_aime_part_search/vllm data/episodes

curl_local() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl "$@"
}

wait_for_port() {
  local port="$1"
  local name="$2"
  for _ in $(seq 1 120); do
    if curl_local -s --max-time 5 "http://localhost:$port/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"probe","messages":[{"role":"user","content":"1+1"}],"max_tokens":1}' |
      grep -q '"choices"'; then
      echo "[$(date '+%F %T')] $name ready on port $port"
      return
    fi
    sleep 5
  done
  echo "$name did not become ready on port $port" >&2
  exit 1
}

stop_services() {
  kill "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
  sleep 3
  kill -9 "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
}

trap stop_services EXIT

echo "[$(date '+%F %T')] starting SRM on GPU $SRM_GPU"
(
  cd "$VLLM_DIR"
  CUDA_VISIBLE_DEVICES="$SRM_GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
    --model "$SRM_MODEL" \
    --port "$SRM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16
) > logs/trim_aime_part_search/vllm/srm.log 2>&1 &
SRM_PID=$!

echo "[$(date '+%F %T')] starting LRM on GPUs $LRM_GPUS"
(
  cd "$VLLM_DIR"
  CUDA_VISIBLE_DEVICES="$LRM_GPUS" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
    --model "$LRM_MODEL" \
    --port "$LRM_PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8
) > logs/trim_aime_part_search/vllm/lrm.log 2>&1 &
LRM_PID=$!

wait_for_port "$SRM_PORT" "SRM"
wait_for_port "$LRM_PORT" "LRM"

for dataset in "${DATASETS[@]}"; do
  echo "[$(date '+%F %T')] generating/resuming $dataset episodes"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m data.generate_episodes \
    --dataset "$dataset" \
    --output_dir data/episodes \
    --srm_server_url "http://localhost:$SRM_PORT/v1" \
    --lrm_server_url "http://localhost:$LRM_PORT/v1" \
    --srm_model_name srm \
    --lrm_model_name lrm \
    --prm_model_name "$PRM_MODEL" \
    --prm_device "$PRM_DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_workers "$GEN_MAX_WORKERS"
done

echo "[$(date '+%F %T')] AIME Part-search episode generation finished"
