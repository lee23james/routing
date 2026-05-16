# OmniMath Mixed 3-4/4-9 BA80 Dense Sweep

Dataset: `omnimath_mixed_3_4_100_4_9_100_test`, n=200; LRM FLOPs=445.880613 TFLOPs/query; BA80 target=356.704490 TFLOPs/query.

## Closest Real Dense Point To BA80

| Method | Acc | Correct/N | TFLOPs/query | % LRM FLOPs | Gap to 80% | Threshold | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TRIM-Agg | 61.50 | 123/200 | 356.60 | 79.98% | 0.02 pp | 0.7082 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt` |
| TRIM-Rubric | 59.00 | 118/200 | 356.79 | 80.02% | 0.02 pp | 0.6981 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0020.pt` |
| TRIM-RubricV2b | 61.50 | 123/200 | 356.63 | 79.98% | 0.02 pp | 0.47395 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |

## Best Real Point Under BA80 Budget

| Method | Acc | Correct/N | TFLOPs/query | % LRM FLOPs | Gap Under 80% | Threshold | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TRIM-Agg | 63.00 | 126/200 | 219.49 | 49.23% | 30.77 pp | 0.76555 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt` |
| TRIM-Rubric | 64.00 | 128/200 | 215.34 | 48.30% | 31.70 pp | 0.6743 | `trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
| TRIM-RubricV2b | 63.50 | 127/200 | 230.38 | 51.67% | 28.33 pp | 0.71765 | `trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/best.pt` |

## Augmented Interpolated BA Metrics

These are computed after merging the dense BA80 sweep with the existing coarse raw points and recomputing the Pareto envelope. They are still curve/interpolation metrics, not single threshold measurements.

| Method | BA@20 | BA@40 | BA@60 | BA@80 | FLOPs@98 | FLOPs@98 %LRM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRIM-Agg | 56.49 | 59.85 | 63.00 | 63.00 | 189.26 | 42.45% |
| TRIM-Rubric | 56.19 | 60.44 | 64.00 | 64.00 | 182.09 | 40.84% |
| TRIM-RubricV2b | 56.62 | 60.40 | 63.50 | 63.50 | 181.69 | 40.75% |

## Near-BA80 Real Points Window

The tables below list real evaluated points in the 75%-85% LRM FLOPs window, sorted by distance to 80%.

### TRIM-Agg

| Acc | Correct/N | TFLOPs/query | % LRM | Gap to 80% | Threshold | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 61.50 | 123/200 | 356.60 | 79.98% | 0.02 pp | 0.7082 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt` |
| 61.50 | 123/200 | 357.47 | 80.17% | 0.17 pp | 0.74845 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/final.pt` |
| 61.50 | 123/200 | 357.47 | 80.17% | 0.17 pp | 0.74845 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/epoch_0040.pt` |
| 62.00 | 124/200 | 357.58 | 80.20% | 0.20 pp | 0.75785 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt` |
| 61.50 | 123/200 | 357.87 | 80.26% | 0.26 pp | 0.67255 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0030.pt` |
| 61.50 | 123/200 | 355.18 | 79.66% | 0.34 pp | 0.7488 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/final.pt` |
| 61.50 | 123/200 | 355.18 | 79.66% | 0.34 pp | 0.7488 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 355.10 | 79.64% | 0.36 pp | 0.70855 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt` |
| 61.50 | 123/200 | 354.87 | 79.59% | 0.41 pp | 0.6729 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0030.pt` |
| 57.50 | 115/200 | 354.36 | 79.47% | 0.53 pp | 0.704 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0020.pt` |
| 62.00 | 124/200 | 354.12 | 79.42% | 0.58 pp | 0.7582 | `trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt` |
| 61.50 | 123/200 | 353.46 | 79.27% | 0.73 pp | 0.7089 | `trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt` |

### TRIM-Rubric

| Acc | Correct/N | TFLOPs/query | % LRM | Gap to 80% | Threshold | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 59.00 | 118/200 | 356.79 | 80.02% | 0.02 pp | 0.6981 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0020.pt` |
| 61.50 | 123/200 | 357.06 | 80.08% | 0.08 pp | 0.62675 | `trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 357.12 | 80.09% | 0.09 pp | 0.7411 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 357.12 | 80.09% | 0.09 pp | 0.7411 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/final.pt` |
| 62.00 | 124/200 | 357.22 | 80.12% | 0.12 pp | 0.7222 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/best.pt` |
| 61.50 | 123/200 | 356.16 | 79.88% | 0.12 pp | 0.6568 | `trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
| 61.50 | 123/200 | 356.10 | 79.86% | 0.14 pp | 0.6271 | `trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 357.36 | 80.15% | 0.15 pp | 0.65645 | `trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
| 61.50 | 123/200 | 355.94 | 79.83% | 0.17 pp | 0.59005 | `trim_rubric_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/epoch_0010.pt` |
| 61.50 | 123/200 | 355.84 | 79.81% | 0.19 pp | 0.74145 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 355.84 | 79.81% | 0.19 pp | 0.74145 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/final.pt` |
| 62.00 | 124/200 | 355.83 | 79.80% | 0.20 pp | 0.72255 | `trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/best.pt` |

### TRIM-RubricV2b

| Acc | Correct/N | TFLOPs/query | % LRM | Gap to 80% | Threshold | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 61.50 | 123/200 | 356.63 | 79.98% | 0.02 pp | 0.47395 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
| 61.50 | 123/200 | 356.80 | 80.02% | 0.02 pp | 0.41205 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 356.80 | 80.02% | 0.02 pp | 0.41205 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/final.pt` |
| 61.50 | 123/200 | 356.53 | 79.96% | 0.04 pp | 0.7526 | `trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 356.53 | 79.96% | 0.04 pp | 0.7526 | `trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/final.pt` |
| 61.50 | 123/200 | 356.97 | 80.06% | 0.06 pp | 0.4736 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
| 61.50 | 123/200 | 356.37 | 79.93% | 0.07 pp | 0.4124 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 356.37 | 79.93% | 0.07 pp | 0.4124 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/final.pt` |
| 61.50 | 123/200 | 357.10 | 80.09% | 0.09 pp | 0.4117 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt` |
| 61.50 | 123/200 | 357.10 | 80.09% | 0.09 pp | 0.4117 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/final.pt` |
| 61.50 | 123/200 | 357.37 | 80.15% | 0.15 pp | 0.47325 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
| 61.50 | 123/200 | 356.03 | 79.85% | 0.15 pp | 0.4743 | `trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt` |
