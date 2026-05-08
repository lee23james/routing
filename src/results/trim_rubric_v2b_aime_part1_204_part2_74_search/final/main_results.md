Main Results. Acc is measured at 60% of LRM-only FLOPs, and FLOPs denotes the computation required to reach 98% of LRM-only accuracy. The best result is bold, and the second-best result is underlined. Extra Tokens denotes whether the method requires additional token generation during the reasoning process.

| Method | Extra Tokens | AIME 2020-2024 Part II Acc ↑ | AIME 2020-2024 Part II FLOPs ↓ |
| --- | --- | --- | --- |
| SRM-Only | No | 9.5% | - |
| LRM-Only | No | - | 100.0% |
| Random Routing | Yes | 13.9% | 96.4% |
| TRIM-Agg (PPO) | Yes | 17.2% | <u>65.9%</u> |
| TRIM-Rubric (PPO) | Yes | <u>17.2%</u> | 65.9% |
| TRIM-RubricV2 (PPO) | Yes | **17.6%** | **58.0%** |
