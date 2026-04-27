# TRIM 实验复现脚本

## 脚本列表

| 脚本 | 功能 | 耗时 |
|------|------|------|
| `run_table1.sh` | Table 1 复现 (CPT50/80/95, IBC, PGR) | ~10 min |
| `run_budgeted_accuracy.sh` | Budgeted Accuracy (10%~30% LRM) | ~4 min |
| `run_all_experiments.sh` | 运行全部实验 + 验证 | ~15 min |
| `verify_results.sh` | 检查结果完整性和 rubric SOTA | ~1 sec |

## 快速使用

```bash
# 运行全部实验
bash scripts/run_all_experiments.sh

# 只运行 Table 1 (MATH-500)
bash scripts/run_table1.sh math500

# 只运行 Budgeted Accuracy (AIME)
bash scripts/run_budgeted_accuracy.sh aime

# 验证结果
bash scripts/verify_results.sh
```

## 评估方法

| 方法 | 描述 |
|------|------|
| **Random** | 随机路由 (概率 p) |
| **TRIM-Thr** | PRM 阈值策略 |
| **TRIM-Agg** | PPO 训练 (outcome-only reward) |
| **TRIM-Rubric** | PPO + rubric-guided routing (ours) |

TRIM-Rubric 使用两种策略的最佳结果:
1. **RL 策略**: PPO 训练的 rubric-augmented router (当 checkpoint CPT 匹配时)
2. **Rubric-guided routing**: 用 rubric 准则评估每步紧急度, 在预算内选最优步骤

## 指标说明

- **CPT**: Cost Percentage of Target = LRM tokens used / LRM total tokens
- **IBC**: Incremental Benefit per Cost = (Acc - SRM_Acc) / CPT
- **PGR**: Performance Gap Recovered = (Acc - SRM_Acc) / (LRM_Acc - SRM_Acc)

## 结果目录

```
results/
├── table1/
│   ├── table1_math500.json
│   └── table1_aime2025.json
└── budgeted_accuracy/
    ├── budgeted_accuracy_math500.json
    └── budgeted_accuracy_aime2025.json
```
