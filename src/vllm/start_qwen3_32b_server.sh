#!/bin/bash
# 启动 qwen3-4b 模型服务器

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ROOT="${TRIM_MODEL_ROOT:-/mnt/hdd2/chengcheng}"
cd "$SCRIPT_DIR"

# 配置
CUDA_DEVICE=0,1,2,3  # 修改为你的 GPU ID
PORT=4000       # 修改为你想要的端口
MODEL="${MODEL_ROOT}/qwen3-32b"

echo "启动 qwen3-4b 服务器..."
echo "GPU: $CUDA_DEVICE"
echo "端口: $PORT"
echo "模型: $MODEL"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python server_vllm.py \
    --model $MODEL \
    --port $PORT
