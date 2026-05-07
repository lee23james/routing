#!/usr/bin/env bash
# End-to-end GPQA-only TRIM-Agg quick search for local 4-GPU runs.

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
PRM_DEVICE="${PRM_DEVICE:-cuda:0}"
SRM_GPU="${SRM_GPU:-1}"
LRM_GPUS="${LRM_GPUS:-2,3}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
GEN_MAX_WORKERS="${GEN_MAX_WORKERS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-64}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-1}"

TRAIN_EPISODES="$SRC_DIR/data/episodes/gpqa_main_train_200_episodes.jsonl"
TEST_EPISODES="$SRC_DIR/data/episodes/gpqa_diamond_test_100_episodes.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$SRC_DIR/results/trim_agg_gpqa_point_search/final}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-checkpoints/trim_agg_gpqa_main200_point_search_*/*.pt}"

cd "$SRC_DIR"

stop_generation_services() {
  kill "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
  sleep 3
  kill -9 "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
}

wait_for_port() {
  local port="$1"
  local name="$2"
  local max_tries="${3:-120}"
  for _ in $(seq 1 "$max_tries"); do
    if curl -s --max-time 5 "http://localhost:$port/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"probe","messages":[{"role":"user","content":"A?"}],"max_tokens":1}' |
      grep -q '"choices"'; then
      echo "$name ready on port $port"
      return
    fi
    sleep 5
  done
  echo "$name did not become ready on port $port" >&2
  exit 1
}

mkdir -p "$SRC_DIR/data/episodes" "$OUTPUT_DIR" "logs/trim_agg_gpqa_point_search/vllm"

(
  cd "$VLLM_DIR"
  CUDA_VISIBLE_DEVICES="$SRM_GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
    --model "$SRM_MODEL" \
    --port "$SRM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16
) > logs/trim_agg_gpqa_point_search/vllm/srm.log 2>&1 &
SRM_PID=$!

(
  cd "$VLLM_DIR"
  CUDA_VISIBLE_DEVICES="$LRM_GPUS" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
    --model "$LRM_MODEL" \
    --port "$LRM_PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8
) > logs/trim_agg_gpqa_point_search/vllm/lrm.log 2>&1 &
LRM_PID=$!

trap 'kill "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true' EXIT

wait_for_port "$SRM_PORT" "SRM"
wait_for_port "$LRM_PORT" "LRM"

rm -f "$TRAIN_EPISODES" "$TEST_EPISODES"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m data.generate_episodes \
  --dataset gpqa_main_train_200 \
  --output_dir data/episodes \
  --srm_server_url "http://localhost:$SRM_PORT/v1" \
  --lrm_server_url "http://localhost:$LRM_PORT/v1" \
  --srm_model_name srm \
  --lrm_model_name lrm \
  --prm_model_name "$PRM_MODEL" \
  --prm_device "$PRM_DEVICE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_workers "$GEN_MAX_WORKERS" \
  --no_resume

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m data.generate_episodes \
  --dataset gpqa_diamond_test_100 \
  --output_dir data/episodes \
  --srm_server_url "http://localhost:$SRM_PORT/v1" \
  --lrm_server_url "http://localhost:$LRM_PORT/v1" \
  --srm_model_name srm \
  --lrm_model_name lrm \
  --prm_model_name "$PRM_MODEL" \
  --prm_device "$PRM_DEVICE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_workers "$GEN_MAX_WORKERS" \
  --no_resume

if [ "$(wc -l < "$TRAIN_EPISODES")" -lt 200 ]; then
  echo "Training episodes are incomplete: $TRAIN_EPISODES has $(wc -l < "$TRAIN_EPISODES") rows" >&2
  exit 1
fi

if [ "$(wc -l < "$TEST_EPISODES")" -lt 100 ]; then
  echo "Test episodes are incomplete: $TEST_EPISODES has $(wc -l < "$TEST_EPISODES") rows" >&2
  exit 1
fi

stop_generation_services
trap - EXIT

NUM_EPOCHS="$NUM_EPOCHS" \
EPISODES_PER_EPOCH="$EPISODES_PER_EPOCH" \
SAVE_EVERY="$SAVE_EVERY" \
SEED="$SEED" \
PYTHON_BIN="$PYTHON_BIN" \
EPISODES_PATH="$TRAIN_EPISODES" \
bash scripts/search_trim_agg_gpqa_points_4gpu.sh

"$PYTHON_BIN" -u -m eval.plot_trim_agg_baseline \
  --datasets gpqa_diamond_test_100 \
  --gpqa_diamond_episodes "$TEST_EPISODES" \
  --checkpoint_glob "$CHECKPOINT_GLOB" \
  --output_dir "$OUTPUT_DIR" \
  --n_selected_points 8 \
  --device cuda:0

echo "Done. Main outputs:"
echo "  $OUTPUT_DIR/accuracy_vs_flops.png"
echo "  $OUTPUT_DIR/trim_agg_gpqa_diamond_test_100_60_98.json"
echo "  $OUTPUT_DIR/selected_points_gpqa_diamond_test_100.csv"
