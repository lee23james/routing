#!/usr/bin/env bash
# End-to-end MATH-only TRIM-Agg quick search for local 4-GPU runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"
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

MATH500_ARCHIVE="${MATH500_ARCHIVE:-$REPO_ROOT/archive/math500_episodes_fixed.jsonl}"
MATH500_EPISODES="$SRC_DIR/data/episodes/math500_episodes.jsonl"
TRAIN_EPISODES="$SRC_DIR/data/episodes/math_train_200_episodes.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$SRC_DIR/results/trim_agg_math200_point_search/final}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-checkpoints/trim_agg_math200_point_search_*/*.pt}"

SKIP_CLEANUP="${SKIP_CLEANUP:-false}"
SKIP_GENERATE="${SKIP_GENERATE:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

cd "$SRC_DIR"

require_file() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

curl_local() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl "$@"
}

kill_matching_processes() {
  local pattern="$1"
  local matches
  local pids
  matches="$(
    ps -eo pid=,args= | while read -r pid args; do
      if [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ]; then
        continue
      fi
      case "$args" in
        *"$pattern"*)
          case "$args" in
            *"run_trim_agg_math_quick_search.sh"* | *"ps -eo pid=,args="*)
              ;;
            *)
              printf "%s %s\n" "$pid" "$args"
              ;;
          esac
          ;;
      esac
    done
  )"
  if [ -z "$matches" ]; then
    echo "No processes matched: $pattern"
    return
  fi
  pids="$(printf "%s\n" "$matches" | awk '{print $1}')"
  echo "Stopping processes matching: $pattern"
  printf "%s\n" "$matches"
  echo "$pids" | xargs -r kill || true
  sleep 3
  matches="$(
    ps -eo pid=,args= | while read -r pid args; do
      if [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ]; then
        continue
      fi
      case "$args" in
        *"$pattern"*)
          case "$args" in
            *"run_trim_agg_math_quick_search.sh"* | *"ps -eo pid=,args="*)
              ;;
            *)
              printf "%s %s\n" "$pid" "$args"
              ;;
          esac
          ;;
      esac
    done
  )"
  if [ -n "$matches" ]; then
    pids="$(printf "%s\n" "$matches" | awk '{print $1}')"
    echo "Force stopping remaining processes matching: $pattern"
    printf "%s\n" "$matches"
    echo "$pids" | xargs -r kill -9 || true
  fi
}

wait_for_port() {
  local port="$1"
  local name="$2"
  local max_tries="${3:-120}"
  for _ in $(seq 1 "$max_tries"); do
    if curl_local -s --max-time 5 "http://localhost:$port/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"probe","messages":[{"role":"user","content":"1+1"}],"max_tokens":1}' |
      grep -q '"choices"'; then
      echo "$name ready on port $port"
      return
    fi
    sleep 5
  done
  echo "$name did not become ready on port $port" >&2
  exit 1
}

cleanup_services() {
  kill_matching_processes "vllm serve /home/chencheng/models"
  kill_matching_processes "TRIM_Agg.py --mode train"
  kill_matching_processes "TRIM_Agg.py --mode eval"
  kill_matching_processes "run_qwen3_math1k_remaining_sweep"
  kill_matching_processes "server_vllm.py --model $SRM_MODEL"
  kill_matching_processes "server_vllm.py --model $LRM_MODEL"
}

start_generation_services() {
  mkdir -p logs/trim_agg_math200_point_search/vllm
  echo "Starting SRM on GPU $SRM_GPU, port $SRM_PORT"
  (
    cd "$VLLM_DIR"
    CUDA_VISIBLE_DEVICES="$SRM_GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
      --model "$SRM_MODEL" \
      --port "$SRM_PORT" \
      --tensor-parallel-size 1 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 16
  ) > logs/trim_agg_math200_point_search/vllm/srm.log 2>&1 &
  SRM_PID=$!

  echo "Starting LRM on GPUs $LRM_GPUS, port $LRM_PORT"
  (
    cd "$VLLM_DIR"
    CUDA_VISIBLE_DEVICES="$LRM_GPUS" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u server_vllm.py \
      --model "$LRM_MODEL" \
      --port "$LRM_PORT" \
      --tensor-parallel-size 2 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 8
  ) > logs/trim_agg_math200_point_search/vllm/lrm.log 2>&1 &
  LRM_PID=$!

  wait_for_port "$SRM_PORT" "SRM"
  wait_for_port "$LRM_PORT" "LRM"
}

stop_generation_services() {
  kill "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
  sleep 3
  kill -9 "${SRM_PID:-}" "${LRM_PID:-}" 2>/dev/null || true
}

require_file "$PYTHON_BIN"
require_file "$SRM_MODEL"
require_file "$LRM_MODEL"
require_file "$PRM_MODEL"
require_file "$MATH500_ARCHIVE"

mkdir -p "$SRC_DIR/data/episodes" "$OUTPUT_DIR"
cp "$MATH500_ARCHIVE" "$MATH500_EPISODES"
echo "Restored math500 episodes -> $MATH500_EPISODES ($(wc -l < "$MATH500_EPISODES") rows)"

if [ "$SKIP_CLEANUP" != "true" ]; then
  cleanup_services
fi

if [ "$SKIP_GENERATE" != "true" ]; then
  rm -f "$TRAIN_EPISODES"
  start_generation_services
  trap stop_generation_services EXIT
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m data.generate_episodes \
    --dataset math_train_200 \
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
  stop_generation_services
  trap - EXIT

  require_file "$TRAIN_EPISODES"
  if [ "$(wc -l < "$TRAIN_EPISODES")" -lt 200 ]; then
    echo "Training episodes are incomplete: $TRAIN_EPISODES has $(wc -l < "$TRAIN_EPISODES") rows" >&2
    exit 1
  fi
fi

if [ "$SKIP_TRAIN" != "true" ]; then
  require_file "$TRAIN_EPISODES"
  if [ "$(wc -l < "$TRAIN_EPISODES")" -lt 200 ]; then
    echo "Training episodes are incomplete: $TRAIN_EPISODES has $(wc -l < "$TRAIN_EPISODES") rows" >&2
    exit 1
  fi
  NUM_EPOCHS="$NUM_EPOCHS" \
  EPISODES_PER_EPOCH="$EPISODES_PER_EPOCH" \
  SAVE_EVERY="$SAVE_EVERY" \
  SEED="$SEED" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/search_trim_agg_math_points_4gpu.sh
fi

if [ "$SKIP_EVAL" != "true" ]; then
  "$PYTHON_BIN" -u -m eval.plot_trim_agg_baseline \
    --datasets math500 \
    --math500_episodes data/episodes/math500_episodes.jsonl \
    --checkpoint_glob "$CHECKPOINT_GLOB" \
    --output_dir "$OUTPUT_DIR" \
    --n_selected_points 8 \
    --device cuda:0
fi

echo "Done. Main outputs:"
echo "  $OUTPUT_DIR/accuracy_vs_flops.png"
echo "  $OUTPUT_DIR/trim_agg_math500_60_98.json"
echo "  $OUTPUT_DIR/selected_points_math500.csv"
