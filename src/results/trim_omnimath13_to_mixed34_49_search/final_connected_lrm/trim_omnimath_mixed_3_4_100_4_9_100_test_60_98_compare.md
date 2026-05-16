# TRIM 60/98 Comparison

- Dataset: OmniMath Mixed 3-4/4-9 Test-200 (n=200)
- LRM-only accuracy: 62.0%
- LRM-only FLOPs: 445.8806 TFLOPs/query
- 60% LRM FLOPs target: 267.5284 TFLOPs/query
- 98% LRM accuracy target: 60.8%

| Method | 60% LRM FLOPs Point | 98% LRM Acc Point |
| --- | --- | --- |
| TRIM-Agg (PPO) | (267.5284 TFLOPs/query, 60.0%) | (346.8840 TFLOPs/query, 60.8%) |
| TRIM-Rubric (PPO) | (267.5284 TFLOPs/query, 61.9%) | (271.9905 TFLOPs/query, 60.8%) |
| TRIM-RubricV2 (PPO) | (267.5284 TFLOPs/query, 62.0%) | (238.4034 TFLOPs/query, 60.8%) |
