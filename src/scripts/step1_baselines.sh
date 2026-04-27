#!/bin/bash
# ================================================================
# Step 1: 推理 Baselines
# ================================================================
# 目的: 获取所有 baseline 方法在各数据集上的准确率和 FLOPs
# 输入:
#   - data/math500.jsonl, data/aime2025-*.jsonl, data/omnimath.jsonl
#   - vLLM 服务 (localhost:4003, localhost:4001)
# 输出:
#   - results/baselines/{dataset}/{model}/results.jsonl  (逐题结果)
#   - results/baselines/{dataset}/{model}/stats.json     (汇总统计)
#   - results/baselines/summary.json                     (全局汇总)
# 验证: 脚本末尾打印所有 baseline 的准确率表格
# 预计时间:
#   - math500 + aime2025: 已完成 (跳过)
#   - omnimath 前100题: SRM ~30min, LRM ~3h (需用户自己跑)
# ================================================================

set -e
cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1: 推理 Baselines"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DATASET="${1:-all}"   # math500 | aime2025 | omnimath | all
MAX_N="${2:-100}"     # omnimath 最多跑多少题

# ---- 1a. math500 + aime2025 (SRM + LRM) ----
for ds in math500 aime2025; do
    for port in 4003 4001; do
        model=$( [ "$port" -eq 4003 ] && echo "qwen3-1.7b" || echo "qwen3-14b" )
        stats="results/baselines/${ds}/${model}-baseline/stats.json"
        if [ -f "$stats" ] && [ "$DATASET" != "$ds" ] && [ "$DATASET" != "all" ]; then
            echo "  [跳过] $ds/$model — 已有结果"
            continue
        fi
        if [ -f "$stats" ]; then
            acc=$(python3 -c "import json; print(json.load(open('$stats'))['accuracy'])" 2>/dev/null)
            echo "  [已完成] $ds/$model: acc=$acc"
            continue
        fi
        echo "  [运行] $ds/$model ..."
        python -u -m baseline.run_baseline --dataset "$ds" --ports "$port" --think_mode
    done
done

# ---- 1b. omnimath 前 N 题 ----
if [ "$DATASET" = "omnimath" ] || [ "$DATASET" = "all" ]; then
    echo ""
    echo "  omnimath baselines (前 $MAX_N 题):"
    for port in 4003 4001; do
        model=$( [ "$port" -eq 4003 ] && echo "qwen3-1.7b" || echo "qwen3-14b" )
        stats="results/baselines/omnimath/${model}-baseline/stats.json"
        if [ -f "$stats" ]; then
            acc=$(python3 -c "import json; print(json.load(open('$stats'))['accuracy'])" 2>/dev/null)
            echo "  [已完成] omnimath/$model: acc=$acc"
            continue
        fi
        echo "  [运行] omnimath/$model (前${MAX_N}题) ..."
        python -u -m baseline.run_baseline --dataset omnimath --ports "$port" --think_mode --max_n "$MAX_N"
    done
fi

# ---- 1c. Random routing + Threshold baselines (基于已有 episodes) ----
echo ""
echo "  [计算] Random routing + Threshold baselines (基于 episodes) ..."
python3 -u -c "
import json, random, os, sys
sys.path.insert(0, '.')
from data.datasets import load_jsonl

for ep_file in ['data/episodes/combined_episodes.jsonl']:
    if not os.path.exists(ep_file):
        print(f'  跳过 {ep_file} (不存在)')
        continue
    eps = load_jsonl(ep_file)
    n = len(eps)

    # Random routing
    random.seed(42)
    rand_correct = 0
    for ep in eps:
        n_steps = len(ep.get('srm_steps', []))
        actions = [random.randint(0, 1) for _ in range(n_steps)]
        n_regen = sum(actions)
        if n_regen == 0:
            rand_correct += int(ep.get('srm_correct', False))
        elif n_regen == n_steps:
            rand_correct += int(ep.get('lrm_correct', False))
        else:
            rand_correct += int(ep.get('lrm_correct', False) if n_regen >= n_steps/2 else ep.get('srm_correct', False))
    print(f'  Random routing: acc={rand_correct/n:.3f} ({rand_correct}/{n})')

    # Threshold baselines
    for thr in [0.3, 0.5, 0.7]:
        thr_correct = 0
        total_regen = 0
        for ep in eps:
            prm = ep.get('srm_prm_scores', [])
            actions = [1 if s < thr else 0 for s in prm]
            n_regen = sum(actions)
            total_regen += n_regen
            if n_regen == 0:
                thr_correct += int(ep.get('srm_correct', False))
            elif n_regen == len(prm):
                thr_correct += int(ep.get('lrm_correct', False))
            else:
                thr_correct += int(ep.get('lrm_correct', False) if n_regen >= len(prm)/2 else ep.get('srm_correct', False))
        avg_regen = total_regen / n
        print(f'  TRIM-Thr(τ={thr}): acc={thr_correct/n:.3f} ({thr_correct}/{n}), avg_regen={avg_regen:.1f} steps/problem')
" 2>/dev/null || echo "  [跳过] 需要先完成 Step 2 生成 episodes"

# ---- 验证: 打印汇总 ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Baseline 汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json, glob
for f in sorted(glob.glob('results/baselines/*/qwen3-*/stats.json')):
    parts = f.split('/')
    ds, model = parts[-3], parts[-2].replace('-baseline','')
    d = json.load(open(f))
    print(f'  {ds:12s} | {model:15s} | acc={d[\"accuracy\"]:.3f} | total_tokens={d.get(\"total_tokens\",\"?\")}')" 2>/dev/null || true
