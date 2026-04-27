#!/bin/bash
# ================================================================
# Step 3: Rubric 发现与验证 (Seed → Explore → Filter)
# ================================================================
# 目的: 从 seed rubric 出发, 探索 derived rubric, 三重统计验证, 输出权重
# 输入:
#   - data/episodes/omnimath_episodes.jsonl (训练集 episodes)
# 输出:
#   - data/rubrics/rubric_weights.json     (通过验证的 rubric 及权重)
#   - data/rubrics/rubric_consistency.json  (一致性检验结果)
#   - data/rubrics/episode_rubric_scores.jsonl (每个 episode 的 rubric 分数)
# 验证: 脚本末尾打印通过验证的 rubric 列表和权重
# 预计时间: ~5min
# ================================================================

set -e
cd "$(dirname "$0")/.."

EP_FILE="${1:-data/episodes/omnimath_episodes.jsonl}"
OUTPUT_DIR="${2:-data/rubrics}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3: Rubric 发现与验证"
echo "  输入: $EP_FILE"
echo "  输出: $OUTPUT_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$EP_FILE" ]; then
    echo "✗ Episodes 文件不存在: $EP_FILE"
    echo "  请先运行 step2"
    exit 1
fi

EP_COUNT=$(wc -l < "$EP_FILE")
echo "  Episodes 数量: $EP_COUNT"
if [ "$EP_COUNT" -lt 20 ]; then
    echo "✗ Episodes 太少 ($EP_COUNT), 至少需要 20 条"
    exit 1
fi

echo ""
python -u -m rubric.generate_rubrics \
    --episodes_path "$EP_FILE" \
    --output_dir "$OUTPUT_DIR"

# 验证
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证: 通过验证的 Rubric"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json
w = json.load(open('$OUTPUT_DIR/rubric_weights.json'))
print(f'  共 {len(w)} 个 rubric 通过三重验证:')
print(f'  {\"─\"*50}')
for name, info in sorted(w.items(), key=lambda x: -x[1]['weight']):
    print(f'  {name:30s} | weight={info[\"weight\"]:.3f} | corr={info.get(\"correlation\",\"?\"):.3f}')
"
