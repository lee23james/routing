#!/bin/bash
# ================================================================
# Step 0: 启动 vLLM 推理服务
# ================================================================
# 目的: 部署 SRM (Qwen3-1.7B) 和 LRM (Qwen3-14B) 的推理服务
# 输入: 模型权重 (MODEL_ROOT)
# 输出: 两个 HTTP 端点 (localhost:4003, localhost:4001)
# 验证: 脚本末尾自动测试两个端点是否正常响应
# 预计时间: ~2min
# ================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="$SCRIPT_DIR/../vllm"
MODEL_ROOT="${TRIM_MODEL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/models}"
CONDA_ENV="${TRIM_CONDA_ENV:-routing}"
CONDA_SH="${TRIM_CONDA_SH:-}"

curl_local() {
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        curl "$@"
}

ACTIVATE_CMD="conda activate $CONDA_ENV"
if [ -n "$CONDA_SH" ]; then
    ACTIVATE_CMD="source $CONDA_SH && conda activate $CONDA_ENV"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 0: 启动 vLLM 服务"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查是否已在运行
SRM_RUNNING=$(curl_local -s --max-time 2 http://localhost:4003/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"1"}],"max_tokens":1}' 2>/dev/null | grep -c "choices" || echo 0)
LRM_RUNNING=$(curl_local -s --max-time 5 http://localhost:4001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"1"}],"max_tokens":1}' 2>/dev/null | grep -c "choices" || echo 0)

if [ "$SRM_RUNNING" -gt 0 ] && [ "$LRM_RUNNING" -gt 0 ]; then
    echo "✓ SRM (port 4003) 已在运行"
    echo "✓ LRM (port 4001) 已在运行"
    echo "无需重启。"
    exit 0
fi

echo "需要启动以下服务:"
[ "$SRM_RUNNING" -eq 0 ] && echo "  - SRM (Qwen3-1.7B) on port 4003, GPU 4"
[ "$LRM_RUNNING" -eq 0 ] && echo "  - LRM (Qwen3-14B)  on port 4001, GPU 5,6"
echo ""
echo "请在对应的 tmux/screen 窗口中运行:"
echo ""
[ "$SRM_RUNNING" -eq 0 ] && echo "  # SRM:"
[ "$SRM_RUNNING" -eq 0 ] && echo "  $ACTIVATE_CMD"
[ "$SRM_RUNNING" -eq 0 ] && echo "  cd $VLLM_DIR && CUDA_VISIBLE_DEVICES=4 python server_vllm.py --model $MODEL_ROOT/qwen3-1.7b --port 4003"
echo ""
[ "$LRM_RUNNING" -eq 0 ] && echo "  # LRM:"
[ "$LRM_RUNNING" -eq 0 ] && echo "  $ACTIVATE_CMD"
[ "$LRM_RUNNING" -eq 0 ] && echo "  cd $VLLM_DIR && CUDA_VISIBLE_DEVICES=5,6 python server_vllm.py --model $MODEL_ROOT/qwen3-14b --port 4001 --tensor-parallel-size 2"
echo ""
echo "启动后再次运行此脚本验证。"
