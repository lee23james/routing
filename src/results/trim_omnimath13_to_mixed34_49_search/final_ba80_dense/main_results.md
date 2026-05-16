Main Results. Acc is measured at 60% of LRM-only FLOPs, and FLOPs denotes the computation required to reach 98% of LRM-only accuracy. The best result is bold, and the second-best result is underlined. Extra Tokens denotes whether the method requires additional token generation during the reasoning process.

| Method | Extra Tokens | OmniMath Mixed 3-4/4-9 Test-200 Acc ↑ | OmniMath Mixed 3-4/4-9 Test-200 FLOPs ↓ |
| --- | --- | --- | --- |
| SRM-Only | No | 51.5% | - |
| LRM-Only | No | - | 100.0% |
| Random Routing | Yes | 57.4% | 90.1% |
| TRIM-Agg (PPO) | Yes | 63.0% | 42.4% |
| TRIM-Rubric (PPO) | Yes | **64.0%** | <u>40.8%</u> |
| TRIM-RubricV2 (PPO) | Yes | <u>63.5%</u> | **40.7%** |
