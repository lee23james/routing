# TRIM 60/98 Comparison

- Dataset: AIME 2020-2024 Part II (n=74)
- LRM-only accuracy: 17.6%
- LRM-only FLOPs: 125.0032 TFLOPs/query
- 60% LRM FLOPs target: 75.0019 TFLOPs/query
- 98% LRM accuracy target: 17.2%

| Method | 60% LRM FLOPs Point | 98% LRM Acc Point |
| --- | --- | --- |
| TRIM-Agg (PPO) | (75.0019 TFLOPs/query, 17.2%) | (82.3743 TFLOPs/query, 17.2%) |
| TRIM-Rubric (PPO) | (75.0019 TFLOPs/query, 17.2%) | (82.3951 TFLOPs/query, 17.2%) |
| TRIM-RubricV2 (PPO) | (75.0019 TFLOPs/query, 17.6%) | (72.4495 TFLOPs/query, 17.2%) |
