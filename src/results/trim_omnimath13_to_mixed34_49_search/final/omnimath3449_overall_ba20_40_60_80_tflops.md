# OmniMath 3-4/4-9 Overall BA and FLOPs Summary

Dataset: `OmniMath Mixed 3-4/4-9 Test-200`, n=200; SRM=51.50%, LRM=62.00%, LRM FLOPs=445.88 TFLOPs/query.

| Benchmark | Method | BA@20 ↑ | BA@40 ↑ | BA@60 ↑ | BA@80 ↑ | FLOPs@98 ↓ | FLOPs@98 (% LRM) ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OmniMath 3-4/4-9 Overall | SRM-Only | 51.50 | 51.50 | 51.50 | 51.50 | - | - |
| OmniMath 3-4/4-9 Overall | LRM-Only | 62.00 | 62.00 | 62.00 | 62.00 | 445.88 | 100.00% |
| OmniMath 3-4/4-9 Overall | Random Routing | 52.71 | 55.03 | 57.35 | 59.68 | 401.55 | 90.06% |
| OmniMath 3-4/4-9 Overall | TRIM-Agg | 55.91 | 57.24 | 60.00 | 61.71 | 346.88 | 77.80% |
| OmniMath 3-4/4-9 Overall | TRIM-Rubric | 56.34 | 60.13 | 61.92 | 62.50 | 271.99 | 61.00% |
| OmniMath 3-4/4-9 Overall | TRIM-RubricV2b | 56.13 | 59.98 | 62.00 | 62.00 | 238.40 | 53.47% |
