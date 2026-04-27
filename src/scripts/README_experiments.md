# TRIM 论文实验复现 — 脚本说明

## 目录结构

```
scripts/
├── step0_start_vllm.sh         # 启动 vLLM 服务 (SRM+LRM)
├── step1_baselines.sh          # SRM-only / LRM-only baseline
├── step2_generate_episodes.sh  # 生成 episodes 数据
├── step3_rubric_discovery.sh   # 发现 rubric 权重
├── step4_train_router.sh       # 训练 PPO router
├── step5_evaluate.sh           # 原有评估脚本
│
├── run_table1.sh               # [新] Table 1 复现
├── run_budgeted_accuracy.sh    # [新] Budgeted Accuracy
├── run_all_experiments.sh      # [新] 一键全部实验
├── verify_results.sh           # [新] 结果验证
└── README_experiments.md       # 本文件
```

## 快速开始

### 前提: 已有训练好的 checkpoints 和 episodes 数据

```bash
# 一键运行全部实验 (Table 1 + Budgeted Accuracy)
cd src && bash scripts/run_all_experiments.sh

# 验证结果
bash scripts/verify_results.sh
```

### 从零开始

```bash
cd src

# 1. 启动 vLLM 服务
bash scripts/step0_start_vllm.sh

# 2. 运行 baseline
bash scripts/step1_baselines.sh

# 3. 生成 episodes (SRM+LRM 并行推理 + PRM 打分)
bash scripts/step2_generate_episodes.sh

# 4. 发现 rubric 权重
bash scripts/step3_rubric_discovery.sh

# 5. 训练 router (TRIM-Agg + TRIM-Rubric)
bash scripts/step4_train_router.sh

# 6. 运行全部评估实验
bash scripts/run_all_experiments.sh

# 7. 验证结果
bash scripts/verify_results.sh
```

## 实验说明

### Table 1: TRIM 论文核心对比表

**脚本**: `run_table1.sh`
**对应代码**: `eval/table1_eval.py`

对比四种路由策略在三个固定预算点下的表现:

| 预算点 | 含义 |
|--------|------|
| CPT50  | 使用 LRM 50% 的 token |
| CPT80  | 使用 LRM 80% 的 token |
| CPT95  | 使用 LRM 95% 的 token |

| 指标 | 计算公式 | 含义 |
|------|----------|------|
| Accuracy | 正确率 | 数学问题正确率 |
| CPT | Σ(lrm_tokens_used) / Σ(lrm_total_tokens) | LRM 预算消耗比 |
| IBC | (Acc - SRM_Acc) / CPT | 单位预算的准确率提升 |
| PGR | (Acc - SRM_Acc) / (LRM_Acc - SRM_Acc) | 性能差距恢复比 |

**参数说明**:
```bash
bash scripts/run_table1.sh [CKPT_DIR] [DEVICE] [TARGET_CPTS] [AGG_PREFIX] [RUBRIC_PREFIX]
# 例如:
bash scripts/run_table1.sh checkpoints cpu "0.50,0.80,0.95" v4_agg v4_rubric
```

### Budgeted Accuracy: 固定预算下的准确率

**脚本**: `run_budgeted_accuracy.sh`
**对应代码**: `eval/budgeted_accuracy.py`

在 LRM-only 的 10%/15%/20%/25%/30% 预算下, 对比四种策略能达到的最高准确率。

**参数说明**:
```bash
bash scripts/run_budgeted_accuracy.sh [CKPT_DIR] [DEVICE] [BUDGETS] [AGG_PREFIX] [RUBRIC_PREFIX]
# 例如:
bash scripts/run_budgeted_accuracy.sh checkpoints cpu "0.10,0.15,0.20,0.25,0.30" v4_agg v4_rubric
```

## 四种路由策略

| 策略 | 描述 | 训练需求 |
|------|------|----------|
| Random | 以概率 p 随机使用 LRM | 无 |
| TRIM-Thr | PRM 分数 < 阈值 τ 时使用 LRM | 无 (扫描 τ) |
| TRIM-Agg | PPO 训练, outcome-only reward | 需训练 |
| TRIM-Rubric | PPO + rubric process reward | 需训练 (ours, SOTA) |

## 输出文件

```
results/
├── table1/
│   ├── table1_math500.json     # MATH-500 Table 1 结果
│   └── table1_aime2025.json    # AIME 2025 Table 1 结果
├── budgeted_accuracy/
│   ├── budgeted_accuracy_math500.json
│   └── budgeted_accuracy_aime2025.json
```

## 验证清单

运行 `verify_results.sh` 会检查:

1. **文件完整性**: 所有结果文件是否存在
2. **SOTA 验证**: TRIM-Rubric 是否 >= TRIM-Agg
3. **单调性**: 预算增加时准确率是否不降
4. **PGR 合理性**: PGR 是否在合理范围内
