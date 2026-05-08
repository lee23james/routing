Main Results. Acc is measured at 60% of LRM-only FLOPs, and FLOPs denotes the computation required to reach 98% of LRM-only accuracy. The best result is bold, and the second-best result is underlined. Extra Tokens denotes whether the method requires additional token generation during the reasoning process.

| Method | Extra Tokens | MATH-500 Acc ↑ | MATH-500 FLOPs ↓ |
| --- | --- | --- | --- |
| SRM-Only | No | 68.6% | - |
| LRM-Only | No | - | 100.0% |
| Random Routing | Yes | 78.1% | 91.1% |
| TRIM-Agg (PPO) | Yes | 80.0% | 89.8% |
| TRIM-Rubric (PPO) | Yes | **82.5%** | **84.9%** |
| TRIM-RubricV2 (PPO) | Yes | <u>82.3%</u> | <u>87.9%</u> |
