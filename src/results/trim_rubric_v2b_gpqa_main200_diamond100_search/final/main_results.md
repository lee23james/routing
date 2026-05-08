Main Results. Acc is measured at 60% of LRM-only FLOPs, and FLOPs denotes the computation required to reach 98% of LRM-only accuracy. The best result is bold, and the second-best result is underlined. Extra Tokens denotes whether the method requires additional token generation during the reasoning process.

| Method | Extra Tokens | GPQA Diamond-100 Acc ↑ | GPQA Diamond-100 FLOPs ↓ |
| --- | --- | --- | --- |
| SRM-Only | No | 28.0% | - |
| LRM-Only | No | - | 100.0% |
| Random Routing | Yes | 35.7% | 94.7% |
| TRIM-Agg (PPO) | Yes | 39.7% | **66.0%** |
| TRIM-Rubric (PPO) | Yes | **40.2%** | <u>66.2%</u> |
| TRIM-RubricV2 (PPO) | Yes | <u>39.9%</u> | 67.5% |
