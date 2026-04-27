#!/bin/bash
# ================================================================
# Budgeted Accuracy: 固定 LRM 预算下的准确率对比
# ================================================================
# 目的: 在 LRM-only 的 10%/15%/20%/25%/30% 预算下, 对比四种策略:
#       Random / TRIM-Thr / TRIM-Agg / TRIM-Rubric
#
# 输入:
#   - data/episodes/math500_episodes.jsonl
#   - data/episodes/aime2025_episodes.jsonl
#   - checkpoints/v4_agg_*/best.pt
#   - checkpoints/v4_rubric_*/best.pt
#
# 输出:
#   - results/budgeted_accuracy/budgeted_accuracy_math500.json
#   - results/budgeted_accuracy/budgeted_accuracy_aime2025.json
#
# 验证:
#   - 在低预算 (10%-15%) 下 TRIM-Rubric 应显著优于 TRIM-Agg
#   - 随预算增加, 所有方法准确率应单调递增
#   - Random 应始终最差 (或接近最差)
#
# 预计时间: ~10-20 分钟 (CPU)
# ================================================================

set -e
cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Budgeted Accuracy: LRM 预算下的准确率对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CKPT_DIR="${1:-checkpoints}"
DEVICE="${2:-cpu}"
BUDGETS="${3:-0.10,0.15,0.20,0.25,0.30}"
AGG_PREFIX="${4:-v4_agg}"
RUBRIC_PREFIX="${5:-v4_rubric}"

echo "  Checkpoints  : $CKPT_DIR"
echo "  Device       : $DEVICE"
echo "  Budget ratios: $BUDGETS"
echo "  Agg prefix   : $AGG_PREFIX"
echo "  Rubric prefix: $RUBRIC_PREFIX"
echo ""

# ---- MATH-500 ----
MATH_EP="data/episodes/math500_episodes.jsonl"
if [ -f "$MATH_EP" ]; then
    echo "▶ MATH-500 Budgeted Accuracy ..."
    python -u -m eval.budgeted_accuracy \
        --episodes_path "$MATH_EP" \
        --checkpoint_dir "$CKPT_DIR" \
        --budget_ratios "$BUDGETS" \
        --device "$DEVICE" \
        --output_dir results/budgeted_accuracy \
        --agg_prefix "$AGG_PREFIX" \
        --rubric_prefix "$RUBRIC_PREFIX"
else
    echo "⚠ $MATH_EP 不存在, 跳过 MATH-500"
fi

echo ""

# ---- AIME 2025 ----
AIME_EP="data/episodes/aime2025_episodes.jsonl"
if [ -f "$AIME_EP" ]; then
    echo "▶ AIME 2025 Budgeted Accuracy ..."
    python -u -m eval.budgeted_accuracy \
        --episodes_path "$AIME_EP" \
        --checkpoint_dir "$CKPT_DIR" \
        --budget_ratios "$BUDGETS" \
        --device "$DEVICE" \
        --output_dir results/budgeted_accuracy \
        --agg_prefix "$AGG_PREFIX" \
        --rubric_prefix "$RUBRIC_PREFIX"
else
    echo "⚠ $AIME_EP 不存在, 跳过 AIME 2025"
fi

echo ""

# ---- 验证 ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证: 结果汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json, glob

for f in sorted(glob.glob('results/budgeted_accuracy/budgeted_accuracy_*.json')):
    data = json.load(open(f))
    ds = data['dataset']
    print(f'\n  {ds} (SRM={data[\"srm_accuracy\"]:.4f}  LRM={data[\"lrm_accuracy\"]:.4f}):')
    print(f'  {\"Budget\":<10} {\"Random\":>8} {\"Thr\":>8} {\"Agg\":>8} {\"Rubric\":>8} {\"Best\":>10}')
    print(f'  {\"-\"*55}')
    for bk, bd in data['budgets'].items():
        accs = {
            'Random': bd['random']['accuracy'],
            'Thr': bd['trim_thr']['accuracy'],
            'Agg': bd['trim_agg']['accuracy'],
            'Rubric': bd['trim_rubric']['accuracy'],
        }
        best = max(accs, key=accs.get)
        print(f'  {bk:<10} {accs[\"Random\"]:>8.4f} {accs[\"Thr\"]:>8.4f} '
              f'{accs[\"Agg\"]:>8.4f} {accs[\"Rubric\"]:>8.4f} {best:>10}')
" 2>/dev/null || echo "  (结果文件不存在)"

echo ""
echo "  完成! 结果在 results/budgeted_accuracy/"
