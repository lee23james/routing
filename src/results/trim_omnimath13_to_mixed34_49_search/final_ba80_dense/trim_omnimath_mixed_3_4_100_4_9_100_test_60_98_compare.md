# TRIM 60/98 Comparison

- Dataset: OmniMath Mixed 3-4/4-9 Test-200 (n=200)
- LRM-only accuracy: 62.0%
- LRM-only FLOPs: 445.8806 TFLOPs/query
- 60% LRM FLOPs target: 267.5284 TFLOPs/query
- 98% LRM accuracy target: 60.8%

| Method | 60% LRM FLOPs Point | 98% LRM Acc Point |
| --- | --- | --- |
| TRIM-Agg (PPO) | (267.5284 TFLOPs/query, 63.0%) | (189.2601 TFLOPs/query, 60.8%) |
| TRIM-Rubric (PPO) | (267.5284 TFLOPs/query, 64.0%) | (182.0873 TFLOPs/query, 60.8%) |
| TRIM-RubricV2 (PPO) | (267.5284 TFLOPs/query, 63.5%) | (181.6883 TFLOPs/query, 60.8%) |
