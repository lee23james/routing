# TRIM 60/98 Comparison

- Dataset: GPQA Diamond-100 (n=100)
- LRM-only accuracy: 42.0%
- LRM-only FLOPs: 113.2681 TFLOPs/query
- 60% LRM FLOPs target: 67.9609 TFLOPs/query
- 98% LRM accuracy target: 41.2%

| Method | 60% LRM FLOPs Point | 98% LRM Acc Point |
| --- | --- | --- |
| TRIM-Agg (PPO) | (67.9609 TFLOPs/query, 39.7%) | (74.7695 TFLOPs/query, 41.2%) |
| TRIM-Rubric (PPO) | (67.9609 TFLOPs/query, 40.2%) | (74.9957 TFLOPs/query, 41.2%) |
| TRIM-RubricV2 (PPO) | (67.9609 TFLOPs/query, 39.9%) | (76.4377 TFLOPs/query, 41.2%) |
