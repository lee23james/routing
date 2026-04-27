#!/bin/bash
# ================================================================
# Step 4: 训练 Router (PPO)
# ================================================================
# 目的: 训练路由策略, 对比:
#   - TRIM-Agg: outcome-only reward (R = 1[correct] - λ·C)
#   - TRIM-Rubric: outcome + rubric (R = 1[correct] + λ_p·R_rubric - λ_c·C)
# 输入:
#   - data/episodes/omnimath_episodes.jsonl  (训练集)
#   - data/rubrics/rubric_weights.json       (rubric 权重, TRIM-Rubric 用)
# 输出:
#   - checkpoints/{tag}/best.pt         (最佳模型)
#   - checkpoints/{tag}/final.pt        (最终模型)
#   - checkpoints/{tag}/train_log.json  (训练日志)
# 验证: 脚本末尾对比所有 checkpoint 的 final_reward
# 预计时间: ~30min (所有配置)
# ================================================================

set -e
cd "$(dirname "$0")/.."

EP_FILE="${1:-data/episodes/omnimath_episodes.jsonl}"
RUBRIC_WEIGHTS="${2:-data/rubrics/rubric_weights.json}"
EPOCHS="${3:-200}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 4: 训练 Router (PPO)"
echo "  训练集: $EP_FILE"
echo "  Epochs: $EPOCHS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$EP_FILE" ]; then
    echo "✗ Episodes 文件不存在. 请先运行 step2."
    exit 1
fi

# ---- 4a. TRIM-Agg (outcome-only) ----
echo ""
echo "== 4a. TRIM-Agg (outcome-only reward) =="
for LAM in 3e-4 2e-4 1e-4 5e-5; do
    TAG="v3_agg_lam${LAM}"
    echo "  Training $TAG ..."
    python -u -m router.train_ppo \
        --episodes_path "$EP_FILE" \
        --lam "$LAM" \
        --num_epochs "$EPOCHS" \
        --tag "$TAG" 2>&1 | tail -2
done

# ---- 4b. TRIM-Rubric (outcome + rubric) ----
echo ""
echo "== 4b. TRIM-Rubric (outcome + rubric process reward) =="
if [ ! -f "$RUBRIC_WEIGHTS" ]; then
    echo "  ⚠ Rubric 权重不存在 ($RUBRIC_WEIGHTS), 跳过. 请先运行 step3."
else
    for LAM in 3e-4 2e-4 1e-4 5e-5; do
        for LAM_R in 0.3 0.5; do
            TAG="v3_rubric_lam${LAM}_rub${LAM_R}"
            echo "  Training $TAG ..."
            python -u -m router.train_ppo \
                --episodes_path "$EP_FILE" \
                --lam "$LAM" \
                --lam_rubric "$LAM_R" \
                --rubric_weights "$RUBRIC_WEIGHTS" \
                --num_epochs "$EPOCHS" \
                --tag "$TAG" 2>&1 | tail -2
        done
    done
fi

# 验证
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  训练结果汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json, glob
logs = sorted(glob.glob('checkpoints/v3_*/train_log.json'))
if not logs:
    logs = sorted(glob.glob('checkpoints/*/train_log.json'))
for f in logs:
    try:
        d = json.load(open(f))
        tag = f.split('/')[-2]
        print(f'  {tag:40s} | reward={d.get(\"final_reward\",\"?\"):.3f} | best_ep={d.get(\"best_epoch\",\"?\")}')
    except: pass
" 2>/dev/null || echo "  (无训练日志)"
