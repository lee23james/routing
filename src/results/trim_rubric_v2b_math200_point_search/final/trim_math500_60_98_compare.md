# TRIM 60/98 Comparison

- Dataset: MATH-500 (n=169)
- LRM-only accuracy: 85.8%
- LRM-only FLOPs: 22.8579 TFLOPs/query
- 60% LRM FLOPs target: 13.7148 TFLOPs/query
- 98% LRM accuracy target: 84.1%

| Method | 60% LRM FLOPs Point | 98% LRM Acc Point |
| --- | --- | --- |
| TRIM-Agg (PPO) | (13.7148 TFLOPs/query, 80.0%) | (20.5152 TFLOPs/query, 84.1%) |
| TRIM-Rubric (PPO) | (13.7148 TFLOPs/query, 82.5%) | (19.4093 TFLOPs/query, 84.1%) |
| TRIM-RubricV2 (PPO) | (13.7148 TFLOPs/query, 82.3%) | (20.0878 TFLOPs/query, 84.1%) |
