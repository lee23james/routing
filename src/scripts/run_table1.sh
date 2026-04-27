#!/bin/bash
# ================================================================
# Table 1 复现: TRIM 论文核心对比表
# ================================================================
# 目的: 在 MATH-500 和 AIME 数据集上评估四种路由策略
#       (Random / TRIM-Thr / TRIM-Agg / TRIM-Rubric)
#       在 CPT50 / CPT80 / CPT95 三个预算点的准确率、IBC 和 PGR
#
# 输入:
#   - data/episodes/math500_episodes.jsonl   (MATH-500 episodes)
#   - data/episodes/aime2025_episodes.jsonl  (AIME 2025 episodes)
#   - checkpoints/v4_agg_*/best.pt           (TRIM-Agg 训练好的模型)
#   - checkpoints/v4_rubric_*/best.pt        (TRIM-Rubric 训练好的模型)
#
# 输出:
#   - results/table1/table1_math500.json
#   - results/table1/table1_aime2025.json
#   - results/table1/table1_combined.json (如果使用 combined)
#
# 验证:
#   - TRIM-Rubric 的 Accuracy 应 >= TRIM-Agg (在大部分 CPT 下)
#   - PGR 值应在合理范围 (0 ~ 1+)
#   - IBC 应 > 0 对于有效策略
#
# 预计时间: ~5-15 分钟 (CPU, 无需 GPU)
# ================================================================

set -e
cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Table 1 复现: TRIM 论文核心对比表"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CKPT_DIR="${1:-checkpoints}"
DEVICE="${2:-cpu}"
TARGET_CPTS="${3:-0.50,0.80,0.95}"
AGG_PREFIX="${4:-v4_agg}"
RUBRIC_PREFIX="${5:-v4_rubric}"

echo "  Checkpoints : $CKPT_DIR"
echo "  Device      : $DEVICE"
echo "  Target CPTs : $TARGET_CPTS"
echo "  Agg prefix  : $AGG_PREFIX"
echo "  Rubric prefix: $RUBRIC_PREFIX"
echo ""

# ---- MATH-500 ----
MATH_EP="data/episodes/math500_episodes.jsonl"
if [ -f "$MATH_EP" ]; then
    echo "▶ MATH-500 评估 ..."
    python -u -m eval.table1_eval \
        --episodes_path "$MATH_EP" \
        --checkpoint_dir "$CKPT_DIR" \
        --target_cpts "$TARGET_CPTS" \
        --device "$DEVICE" \
        --output_dir results/table1 \
        --agg_prefix "$AGG_PREFIX" \
        --rubric_prefix "$RUBRIC_PREFIX"
else
    echo "⚠ $MATH_EP 不存在, 跳过 MATH-500"
fi

echo ""

# ---- AIME 2025 ----
AIME_EP="data/episodes/aime2025_episodes.jsonl"
if [ -f "$AIME_EP" ]; then
    echo "▶ AIME 2025 评估 ..."
    python -u -m eval.table1_eval \
        --episodes_path "$AIME_EP" \
        --checkpoint_dir "$CKPT_DIR" \
        --target_cpts "$TARGET_CPTS" \
        --device "$DEVICE" \
        --output_dir results/table1 \
        --agg_prefix "$AGG_PREFIX" \
        --rubric_prefix "$RUBRIC_PREFIX"
else
    echo "⚠ $AIME_EP 不存在, 跳过 AIME 2025"
fi

echo ""

# ---- 验证: 检查 rubric >= agg ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证: TRIM-Rubric vs TRIM-Agg"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json, glob, os

for f in sorted(glob.glob('results/table1/table1_*.json')):
    data = json.load(open(f))
    ds = data['dataset']
    print(f'  {ds}:')
    rubric_wins = 0
    total = 0
    for cpt_label, methods in data['methods'].items():
        agg_acc = methods['trim_agg'].get('accuracy', 0)
        rub_acc = methods['trim_rubric'].get('accuracy', 0)
        total += 1
        if rub_acc >= agg_acc:
            rubric_wins += 1
        winner = '✓ Rubric' if rub_acc >= agg_acc else '✗ Agg'
        print(f'    {cpt_label}: Agg={agg_acc:.4f}  Rubric={rub_acc:.4f}  {winner}')
    print(f'    Rubric wins: {rubric_wins}/{total}')
" 2>/dev/null || echo "  (结果文件不存在)"

echo ""
echo "  完成! 结果在 results/table1/"
