#!/bin/bash
# ================================================================
# Step 5: 评估与可视化
# ================================================================
# 目的: 在 held-out 测试集上评估所有 router, 生成对比图表
# 输入:
#   - data/episodes/combined_episodes.jsonl  (测试集: MATH-500 + AIME 2025)
#   - checkpoints/*/best.pt                  (训练好的 router)
# 输出:
#   - results/flops_evaluation/flops_comparison.json  (FLOPs 对比)
#   - results/plots/accuracy_vs_flops.png             (Pareto 曲线)
#   - results/final_comparison.json                   (最终对比表)
# 验证: 脚本末尾打印方法对比表, 确认 TRIM-Rubric > TRIM-Agg
# 预计时间: ~10min
# ================================================================

set -e
cd "$(dirname "$0")/.."

TEST_EP="${1:-data/episodes/combined_episodes.jsonl}"
CKPT_DIR="${2:-checkpoints}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 5: 评估与可视化"
echo "  测试集: $TEST_EP"
echo "  Checkpoints: $CKPT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$TEST_EP" ]; then
    echo "✗ 测试集 episodes 不存在: $TEST_EP"
    exit 1
fi

EP_COUNT=$(wc -l < "$TEST_EP")
echo "  测试集: $EP_COUNT 题"

# ---- 5a. FLOPs 评估 ----
echo ""
echo "== 5a. FLOPs 评估 =="
python -u -m eval.flops_eval \
    --episodes_path "$TEST_EP" \
    --checkpoints_dir "$CKPT_DIR" \
    --output_dir results/flops_evaluation 2>&1 | tail -20

# ---- 5b. 离线评估 ----
echo ""
echo "== 5b. 离线评估 (各 checkpoint) =="
python -u -m eval.evaluate \
    --mode offline \
    --episodes_path "$TEST_EP" \
    --checkpoints_dir "$CKPT_DIR" \
    --output_dir results/offline_evaluation 2>&1 | tail -20

# ---- 5c. 可视化 ----
echo ""
echo "== 5c. 可视化 =="
python -u -m eval.plot_results \
    --results_dir results/flops_evaluation \
    --output_dir results/plots 2>/dev/null && echo "  ✓ 图表生成 → results/plots/" || echo "  ⚠ plot_results 失败, 跳过"

# 验证
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  最终方法对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json, glob, os

# 从 offline_evaluation 读取
results = {}
for f in sorted(glob.glob('results/offline_evaluation/*.json')):
    name = os.path.basename(f).replace('.json','')
    try:
        d = json.load(open(f))
        results[name] = d
    except: pass

if not results:
    print('  (无评估结果)')
else:
    print(f'  {\"Method\":40s} | {\"Accuracy\":>8s} | {\"FLOPs Ratio\":>11s} | {\"Regen%\":>6s}')
    print(f'  {\"─\"*70}')
    for name, d in sorted(results.items(), key=lambda x: -x[1].get('accuracy', 0)):
        acc = d.get('accuracy', 0)
        flops = d.get('flops_ratio', d.get('relative_flops', '?'))
        regen = d.get('regen_pct', d.get('avg_regen_ratio', '?'))
        print(f'  {name:40s} | {acc:8.3f} | {flops!s:>11s} | {regen!s:>6s}')
" 2>/dev/null || true
