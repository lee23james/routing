# Motivation 分析

论证 outcome-only reward 不足以评价 routing trajectory 过程质量，以及 rubric-based process reward 能够弥补该不足。

## 目录结构

```
motivation/
├── README.md
├── __init__.py
├── construct_trajectory_pairs.py   # 受控轨迹对构建
├── process_quality.py              # 三维过程质量评分
├── llm_judge.py                    # LLM 轨迹对比评估
├── outcome_insufficiency.py        # Sec 2.1 主分析
└── rubric_superiority.py           # Sec 2.2 消融实验
```

## Sec 2.1: Outcome-only reward 不足

**设计思路**: 受控实验，排除 outcome 和 cost 的干扰

1. **筛选 critical episodes**: 只看 SRM 错 + LRM 对 的题（routing 决策关键）
2. **构造受控轨迹对**: 对同一题生成多条 routing trajectories，只保留 **同 outcome + 同 cost (±20%)** 的对
3. **三维评分**: 关键干预命中、切换平稳性、路径简洁性
4. **LLM judge**: 按三维标准判断哪条轨迹更好

**已验证结论**:

| 指标 | MATH-500 | AIME |
|------|----------|------|
| Outcome reward 有意义区分率 | 0.9% | 86.4% |
| Outcome 方向 vs LLM 偏好对齐 | **55.0%** (≈随机) | **40.8%** (<随机) |
| Process quality 区分率 | **83.7%** | **87.7%** |
| LLM-PQ 一致率 | **96.7%** | **95.6%** |

→ Outcome reward 偏好方向与过程质量无关；LLM/PQ 能高一致性区分过程质量

## Sec 2.2: Rubric 有效性消融

四种策略在 CPT 和 Budgeted Accuracy 下对比:

| 策略 | 描述 |
|------|------|
| **Threshold** | PRM 阈值路由 (baseline) |
| **Outcome-only** | PPO + outcome reward (agg checkpoints) |
| **Rubric-only** | 仅用三维 rubric 启发式路由 (无 RL 训练, 无 outcome) |
| **Outcome+Rubric** | PPO + outcome + rubric reward (rubric checkpoints) |

**已验证结论**:

**MATH-500**:
- Rubric-only vs Outcome-only: **7胜1平0负** — rubric 独立路由信号有效
- CPT95: Rubric-only 达最高准确率 0.8639 (超 LRM-only 0.858)

**AIME**:
- Out+Rubric vs Outcome-only: **4胜2平2负** — rubric 增强 RL 有效
- 低预算 (4.4% CPT): Out+Rubric 60% vs Outcome-only 36.7%

## 快速运行

```bash
# 全部分析 (MATH-500)
bash scripts/run_motivation.sh

# 只跑 Sec 2.1 / 2.2
bash scripts/run_motivation.sh sec2_1
bash scripts/run_motivation.sh sec2_2

# AIME 数据集
DATASET=aime bash scripts/run_motivation.sh

# 用真实 LLM API (Sec 2.1)
LLM_MODE=llm bash scripts/run_motivation.sh sec2_1
```

## 三维 Rubric 标准

| 维度 | 含义 | 实现 |
|------|------|------|
| D1: 关键干预命中 | 在 SRM 首次出错附近及时切 LRM | 距 critical_step 越近 → 紧急度越高 |
| D2: 切换平稳性 | 切换模式平稳无震荡 | 选步后填补 1-步间隙，优先连续块 |
| D3: 路径简洁性 | 避免在高 PRM 步浪费 LRM | PRM 越低 → 紧急度越高 |
