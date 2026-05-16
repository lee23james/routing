#!/usr/bin/env bash
# Generate balanced OmniMath 4<=diff<9 test-100 episodes with 20K decoding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_DIR="$SRC_DIR/vllm"

PYTHON_BIN="${PYTHON_BIN:-/mnt/hdd2/chencheng/~/envs/trim/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/hdd2/chencheng/~/models}"
SRM_MODEL="${SRM_MODEL:-$MODEL_ROOT/qwen3-1.7b}"
LRM_MODEL="${LRM_MODEL:-$MODEL_ROOT/qwen3-14b}"
PRM_MODEL="${PRM_MODEL:-$MODEL_ROOT/qwen2.5-math-prm-7b}"

SRM_PORT="${SRM_PORT:-4003}"
LRM_PORT="${LRM_PORT:-4001}"
PRM_DEVICE="${PRM_DEVICE:-cuda:0}"
SRM_GPU="${SRM_GPU:-1}"
LRM_GPUS="${LRM_GPUS:-2,3}"
LRM_TENSOR_PARALLEL_SIZE="${LRM_TENSOR_PARALLEL_SIZE:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GEN_MAX_WORKERS="${GEN_MAX_WORKERS:-4}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-1800}"
OUTPUT_DIR="${OUTPUT_DIR:-data/episodes}"
LOG_DIR="${LOG_DIR:-logs/trim_omnimath49_generation_20k}"
SUMMARY_JSON="${SUMMARY_JSON:-results/trim_omnimath13_to_mixed34_49_search/omnimath49_context_saturation.json}"

cd "$SRC_DIR"

wait_for_port() {
  local port="$1"
  local name="$2"
  local max_tries="${3:-240}"
  for _ in $(seq 1 "$max_tries"); do
    if curl -s --max-time 5 "http://localhost:$port/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"probe","messages":[{"role":"user","content":"A?"}],"max_tokens":1}' |
      grep -q '"choices"'; then
      echo "[$(date '+%F %T')] $name ready on port $port"
      return
    fi
    sleep 5
  done
  echo "$name did not become ready on port $port" >&2
  exit 1
}

stop_generation_services() {
  kill "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
  sleep 3
  kill -9 "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR/vllm" "$(dirname "$SUMMARY_JSON")"

(
  cd "$VLLM_DIR"
  CUDA_VISIBLE_DEVICES="$SRM_GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
    --model "$SRM_MODEL" \
    --port "$SRM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS"
) > "$LOG_DIR/vllm/srm.log" 2>&1 &
SRM_PID=$!

(
  cd "$VLLM_DIR"
  CUDA_VISIBLE_DEVICES="$LRM_GPUS" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
    --model "$LRM_MODEL" \
    --port "$LRM_PORT" \
    --tensor-parallel-size "$LRM_TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS"
) > "$LOG_DIR/vllm/lrm.log" 2>&1 &
LRM_PID=$!

trap stop_generation_services EXIT

wait_for_port "$SRM_PORT" "SRM"
wait_for_port "$LRM_PORT" "LRM"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m data.generate_episodes \
  --dataset omnimath_diff4_9_stratified_test_100 \
  --output_dir "$OUTPUT_DIR" \
  --srm_server_url "http://localhost:$SRM_PORT/v1" \
  --lrm_server_url "http://localhost:$LRM_PORT/v1" \
  --srm_model_name srm \
  --lrm_model_name lrm \
  --prm_model_name "$PRM_MODEL" \
  --prm_device "$PRM_DEVICE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_workers "$GEN_MAX_WORKERS" \
  --generation_workers "$GEN_MAX_WORKERS" \
  --client_timeout "$CLIENT_TIMEOUT" 2>&1 | tee "$LOG_DIR/generation.log"

episodes_path="$OUTPUT_DIR/omnimath_diff4_9_stratified_test_100_episodes.jsonl"
if [ "$(wc -l < "$episodes_path")" -lt 100 ]; then
  echo "OmniMath 4-9 stratified test episodes are incomplete: $episodes_path has $(wc -l < "$episodes_path") rows" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_episode_context.py \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --output_json "$SUMMARY_JSON" \
  "$episodes_path" 2>&1 | tee "$LOG_DIR/context_saturation.log"

stop_generation_services
trap - EXIT

echo "[$(date '+%F %T')] generated $episodes_path"
