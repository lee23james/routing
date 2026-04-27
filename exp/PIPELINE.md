# Experiment Pipeline

> 项目: What Makes a Good Routing Decision? Rubric-Guided Process Reward for Stepwise Model Routing

---

## 总览

```
Step 0: 启动 vLLM 服务             (前置, 手动)
Step 1: 推理 Baselines             (~1-8h, 部分已完成)
Step 2: 生成 Episodes              (~8h, 后台运行)
Step 3: Rubric 发现与验证            (~5min)
Step 4: 训练 Router (PPO)           (~30min)
Step 5: 评估与可视化               (~10min)
```

---

## Step 0: 启动 vLLM 服务

**目的**: 部署 SRM 和 LRM 的推理服务

**脚本**: `src/scripts/step0_start_vllm.sh`

**输入**: 模型权重
| 模型 | 路径 | GPU | 端口 |
|------|------|-----|------|
| SRM (Qwen3-1.7B) | /export/yuguo/ppyg2/model/qwen3-1.7b | GPU 4 | 4003 |
| LRM (Qwen3-14B) | /export/yuguo/ppyg2/model/qwen3-14b | GPU 5,6 | 4001 |

**输出**: 两个 HTTP 服务端点
**验证**: `curl http://localhost:4003/v1/chat/completions` 和 `4001` 均返回 JSON 响应
**预计时间**: ~2min (模型加载)

---

## Step 1: 推理 Baselines

**目的**: 获取 SRM-only, LRM-only, Random-routing, TRIM-threshold 在各数据集上的基准性能

**脚本**: `src/scripts/step1_baselines.sh`

**输入**:
- `data/math500.jsonl` (500题, 实际用169题子集)
- `data/aime2025-I.jsonl` + `data/aime2025-II.jsonl` (30题)
- `data/omnimath.jsonl` (前100题)
- vLLM 服务 (端口 4003, 4001)

**输出**:
- `results/baselines/{dataset}/{model}/results.jsonl` — 每题的推理结果
- `results/baselines/{dataset}/{model}/stats.json` — 准确率、FLOPs统计
- `results/baselines/summary.json` — 汇总表

**验证方式**:
```bash
cat results/baselines/summary.json  # 查看所有baseline准确率
```

**已完成**:
- [x] math500/qwen3-1.7b (acc=66%)
- [x] math500/qwen3-14b (acc=80%)
- [x] aime2025/qwen3-1.7b (acc=10%)
- [x] aime2025/qwen3-14b (acc=20%)
- [ ] omnimath前100题 (SRM + LRM)
- [ ] random-routing (基于已有episodes)
- [ ] TRIM-threshold (PRM阈值路由)

**预计时间**: omnimath 100题 ~3h (LRM think模式); 其余 ~5min

---

## Step 2: 生成 Episodes

**目的**: 为每个问题生成 SRM 和 LRM 的完整推理过程，PRM 逐步打分，构建路由训练数据

**脚本**: `src/scripts/step2_generate_episodes.sh`

**输入**:
- `data/omnimath.jsonl` (前200题, difficulty 1-4)
- vLLM 服务 (SRM + LRM)
- PRM 模型 (Qwen2.5-Math-PRM-7B, GPU 0)

**输出**:
- `data/episodes/omnimath_episodes.jsonl` — 每行一个 episode, 含:
  - `srm_steps`, `srm_prm_scores`, `srm_correct`
  - `lrm_steps`, `lrm_prm_scores`, `lrm_correct`
  - `srm_token_counts`, `lrm_token_counts`

**验证方式**:
```bash
wc -l data/episodes/omnimath_episodes.jsonl   # 应为200行
python3 -c "
import json
eps = [json.loads(l) for l in open('data/episodes/omnimath_episodes.jsonl')]
srm_acc = sum(e['srm_correct'] for e in eps) / len(eps)
lrm_acc = sum(e['lrm_correct'] for e in eps) / len(eps)
print(f'Episodes: {len(eps)}, SRM acc: {srm_acc:.2%}, LRM acc: {lrm_acc:.2%}')
"
```

**预期**: SRM acc ~40-60%, LRM acc ~60-80% (难度1-4的omnimath)

**已完成**: 3/200

**预计时间**: ~8h (每题 ~2.5min, thinking模式)

---

## Step 3: Rubric 发现与验证

**目的**: Seed Rubric → 探索 Derived Rubric → 三重统计验证 → 输出权重

**脚本**: `src/scripts/step3_rubric_discovery.sh`

**输入**:
- `data/episodes/omnimath_episodes.jsonl` (训练集episodes)

**输出**:
- `data/rubrics/rubric_weights.json` — 验证通过的rubric及其权重
- `data/rubrics/rubric_consistency.json` — 一致性检验结果
- `data/rubrics/episode_rubric_scores.jsonl` — 每个episode的各rubric得分

**验证方式**:
```bash
cat data/rubrics/rubric_weights.json   # 查看哪些rubric通过验证及权重
python3 -c "
import json
w = json.load(open('data/rubrics/rubric_weights.json'))
for name, info in w.items():
    print(f'{name}: weight={info[\"weight\"]:.3f}, corr={info[\"correlation\"]:.3f}')
"
```

**预期**: 12个候选rubric中6-8个通过三重验证

**预计时间**: ~5min

---

## Step 4: 训练 Router (PPO)

**目的**: 训练路由策略, 对比 outcome-only (TRIM-Agg) vs outcome+rubric (TRIM-Rubric)

**脚本**: `src/scripts/step4_train_router.sh`

**输入**:
- `data/episodes/omnimath_episodes.jsonl` (训练集)
- `data/rubrics/rubric_weights.json` (rubric权重, 仅TRIM-Rubric需要)

**输出**:
- `checkpoints/trim_agg_lam{X}/best.pt` — 各λ下的TRIM-Agg模型
- `checkpoints/trim_rubric_lam{X}_rub{Y}/best.pt` — 各λ_c, λ_p下的TRIM-Rubric模型
- `checkpoints/*/train_log.json` — 训练日志

**验证方式**:
```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('checkpoints/*/train_log.json')):
    d = json.load(open(f))
    tag = f.split('/')[-2]
    print(f'{tag}: final_reward={d[\"final_reward\"]:.3f}, best_epoch={d[\"best_epoch\"]}')
"
```

**预期**: TRIM-Rubric 的 final_reward > TRIM-Agg

**预计时间**: ~30min (所有配置)

---

## Step 5: 评估与可视化

**目的**: 在 held-out 测试集 (MATH-500 + AIME 2025) 上评估所有方法, 生成对比图表

**脚本**: `src/scripts/step5_evaluate.sh`

**输入**:
- `data/episodes/combined_episodes.jsonl` (测试集episodes, 199题)
- `checkpoints/*/best.pt` (所有训练好的router)

**输出**:
- `results/flops_evaluation/flops_comparison.json` — FLOPs对比
- `results/plots/accuracy_vs_flops.png` — Pareto曲线
- `results/final_comparison.json` — 最终对比表

**验证方式**:
```bash
python3 -c "
import json
d = json.load(open('results/flops_evaluation/flops_comparison.json'))
for method, info in d.items():
    print(f'{method}: acc={info[\"accuracy\"]:.3f}, flops={info[\"flops_ratio\"]:.2f}')
"
```

**预期**: TRIM-Rubric 在相同FLOPs下准确率 > TRIM-Agg, 接近或超过LRM-only

**预计时间**: ~10min

---

## 数据流图

```
omnimath.jsonl ──→ [Step 2] ──→ omnimath_episodes.jsonl ──→ [Step 3] ──→ rubric_weights.json
                                         │                                       │
                                         └──→ [Step 4] ←─────────────────────────┘
                                                  │
                                          checkpoints/*.pt
                                                  │
combined_episodes.jsonl ──→ [Step 5] ←────────────┘
(math500 + aime2025)              │
                           results/ + plots/
```

---

## 关键设计决策

1. **训练/测试分离**: OmniMath (训练) vs MATH-500+AIME (测试), 零数据泄漏
2. **Thinking 模式**: SRM 和 LRM 均开启 thinking, 路由的是 thinking 过程中的步骤
3. **混合正确性估计**: 基于 PRM 轨迹质量插值, 而非直接使用 label (修复泄漏)
4. **Rubric 零成本**: 所有 rubric 评分基于轨迹特征, 无需额外 LLM 调用
