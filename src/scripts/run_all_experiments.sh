#!/bin/bash
# ================================================================
# 一键运行全部实验
# ================================================================
# 按顺序运行:
#   1. Table 1 复现 (MATH-500 + AIME 上的 CPT50/80/95)
#   2. Budgeted Accuracy (LRM 10%/15%/20%/25%/30%)
#
# 如果需要从头训练, 请先运行:
#   bash scripts/step0_start_vllm.sh   # 启动 vLLM 服务
#   bash scripts/step1_baselines.sh    # 运行 baseline
#   bash scripts/step2_generate_episodes.sh  # 生成 episodes
#   bash scripts/step3_rubric_discovery.sh   # 发现 rubric 权重
#   bash scripts/step4_train_router.sh       # 训练 router
#
# 预计时间: ~30 分钟 (CPU, 使用已有 checkpoints)
# ================================================================

set -e
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

CKPT_DIR="${1:-checkpoints}"
DEVICE="${2:-cpu}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          TRIM 实验复现 — 全部评估                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Checkpoints: $CKPT_DIR"
echo "  Device     : $DEVICE"
echo ""

# ---- 前置检查 ----
echo "━━━ 前置检查 ━━━"
MISSING=0
for ep in data/episodes/math500_episodes.jsonl data/episodes/aime2025_episodes.jsonl; do
    if [ -f "$ep" ]; then
        echo "  ✓ $ep  ($(wc -l < "$ep") episodes)"
    else
        echo "  ✗ $ep  — 缺失!"
        MISSING=1
    fi
done

CKPT_COUNT=$(find "$CKPT_DIR" -name "best.pt" 2>/dev/null | wc -l)
echo "  ✓ Checkpoints: $CKPT_COUNT 个 best.pt"
echo ""

if [ "$MISSING" -eq 1 ]; then
    echo "⚠ 缺少 episodes 数据. 请先运行 step2."
    echo "  如果只想评估已有数据, 继续运行."
fi

# ---- 1. Table 1 ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/2] Table 1 复现"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_table1.sh" "$CKPT_DIR" "$DEVICE"

echo ""

# ---- 2. Budgeted Accuracy ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/2] Budgeted Accuracy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_budgeted_accuracy.sh" "$CKPT_DIR" "$DEVICE"

echo ""

# ---- 最终汇总 ----
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    实验完成!                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  输出文件:"
echo "    results/table1/table1_math500.json"
echo "    results/table1/table1_aime2025.json"
echo "    results/budgeted_accuracy/budgeted_accuracy_math500.json"
echo "    results/budgeted_accuracy/budgeted_accuracy_aime2025.json"
echo ""
echo "  运行验证:"
echo "    bash scripts/verify_results.sh"
