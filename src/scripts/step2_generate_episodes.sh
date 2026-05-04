#!/bin/bash
# ================================================================
# Step 2: 生成 Episodes (路由训练数据)
# ================================================================
# 目的: 对每个训练问题, SRM和LRM分别生成完整推理, PRM逐步打分
# 输入:
#   - data/omnimath.jsonl (前 N 题)
#   - vLLM 服务 (SRM port 4003, LRM port 4001)
#   - PRM 模型 (Qwen2.5-Math-PRM-7B, GPU 0)
# 输出:
#   - data/episodes/omnimath_episodes.jsonl
#     每行: {id, query, answer, srm_steps, srm_prm_scores, srm_correct,
#            lrm_steps, lrm_prm_scores, lrm_correct, *_token_counts}
# 验证:
#   wc -l data/episodes/omnimath_episodes.jsonl  → 应等于 N
#   python3 -c "上面 PIPELINE.md 中的验证脚本"
# 预计时间: ~2.5min/题 (thinking模式), 200题 ≈ 8h
# 注意: 支持断点续传 (--resume), 中断后重新运行即可继续
# ================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${TRIM_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT/src"

curl_local() {
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        curl "$@"
}

MAX_N="${1:-200}"     # 生成多少题的 episodes (默认200)
PRM_GPU="${2:-cuda:0}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2: 生成 Episodes"
echo "  数据集: OmniMath 前 $MAX_N 题 (difficulty 1-4)"
echo "  PRM: $PRM_GPU"
echo "  预计时间: $(python3 -c "print(f'{$MAX_N * 2.5 / 60:.1f}h')")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查 vLLM 服务
for PORT in 4003 4001; do
    OK=$(curl_local -s --max-time 5 "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"x","messages":[{"role":"user","content":"1"}],"max_tokens":1}' 2>/dev/null | grep -c "choices" || echo 0)
    if [ "$OK" -eq 0 ]; then
        echo "✗ vLLM 端口 $PORT 未响应, 请先运行 step0"
        exit 1
    fi
done
echo "✓ vLLM 服务正常"

# 检查已有进度
EP_FILE="data/episodes/omnimath_episodes.jsonl"
if [ -f "$EP_FILE" ]; then
    DONE=$(wc -l < "$EP_FILE")
    echo "✓ 已有 $DONE 条 episodes, 继续生成 (断点续传)"
else
    echo "  从零开始生成"
fi

echo ""
echo "开始生成 (Ctrl+C 中断后可重新运行继续) ..."
echo ""

PYTHONUNBUFFERED=1 python -u -m data.generate_episodes \
    --dataset omnimath \
    --prm_device "$PRM_GPU" \
    --temperature 0.0 \
    --output_dir "$PROJECT_ROOT/data/episodes"

# 验证
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json
eps = [json.loads(l) for l in open('data/episodes/omnimath_episodes.jsonl')]
srm_acc = sum(e['srm_correct'] for e in eps) / len(eps)
lrm_acc = sum(e['lrm_correct'] for e in eps) / len(eps)
avg_srm_steps = sum(len(e['srm_steps']) for e in eps) / len(eps)
avg_lrm_steps = sum(len(e.get('lrm_steps',[])) for e in eps) / len(eps)
print(f'  Episodes: {len(eps)}')
print(f'  SRM acc: {srm_acc:.2%}  |  LRM acc: {lrm_acc:.2%}')
print(f'  Avg steps: SRM={avg_srm_steps:.1f}  LRM={avg_lrm_steps:.1f}')
print(f'  ✓ 验证通过' if len(eps) >= $MAX_N * 0.9 else f'  ⚠ 只有 {len(eps)} episodes, 预期 ~$MAX_N')
"
