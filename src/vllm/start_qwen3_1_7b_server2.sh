#!/bin/bash
# 启动 qwen3-4b 模型服务器

cd /export/shy/pp/pp4/src/vllm

# 配置
CUDA_DEVICE=1  # 修改为你的 GPU ID (GPU 2 空闲)
PORT=4013       # 修改为你想要的端口
MODEL="/export/yuguo/ppyg2/model/qwen3-1.7b"

echo "启动 qwen3-4b 服务器..."
echo "GPU: $CUDA_DEVICE"
echo "端口: $PORT"
echo "模型: $MODEL"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python server_vllm.py \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size 1

