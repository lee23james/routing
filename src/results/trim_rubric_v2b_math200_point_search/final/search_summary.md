# TRIM Point Search Summary

Checkpoint patterns: {"ppo_agg": ["checkpoints/trim_agg_math200_point_search_*/*.pt"], "ppo_rubric": ["checkpoints/trim_rubric_math200_point_search_*/*.pt"], "ppo_rubric_v2": ["checkpoints/trim_rubric_v2b_math200_point_search_*/*.pt"]}

## MATH-500
### TRIM-Agg (PPO)
- Selected points: 8
- Limited by accuracy granularity: False
- target=70.55, actual=71.01, flops=2.51T (11.0% LRM), regen=0.18%, ckpt=trim_agg_math200_point_search_lam1e-4_seed1/epoch_0020.pt, th=0.35
- target=72.45, actual=72.19, flops=2.56T (11.2% LRM), regen=0.31%, ckpt=trim_agg_math200_point_search_lam5e-6_seed1/epoch_0040.pt, th=0.55
- target=74.36, actual=75.15, flops=2.70T (11.8% LRM), regen=0.65%, ckpt=trim_agg_math200_point_search_lam0_seed1/epoch_0040.pt, th=0.5
- target=76.27, actual=76.33, flops=6.67T (29.2% LRM), regen=24.51%, ckpt=trim_agg_math200_point_search_lam2e-5_seed1/epoch_0020.pt, th=0.3
- target=78.17, actual=78.70, flops=12.69T (55.5% LRM), regen=57.28%, ckpt=trim_agg_math200_point_search_lam0_seed1/epoch_0030.pt, th=0.45
- target=80.08, actual=79.88, flops=12.84T (56.2% LRM), regen=60.20%, ckpt=trim_agg_math200_point_search_lam5e-6_seed1/epoch_0030.pt, th=0.5
- target=81.99, actual=82.84, flops=17.19T (75.2% LRM), regen=80.70%, ckpt=trim_agg_math200_point_search_lam1e-4_seed1/epoch_0040.pt, th=0.25
- target=83.89, actual=84.02, flops=20.43T (89.4% LRM), regen=92.17%, ckpt=trim_agg_math200_point_search_lam2e-5_seed1/epoch_0030.pt, th=0.45
### TRIM-Rubric (PPO)
- Selected points: 8
- Limited by accuracy granularity: False
- target=70.55, actual=70.41, flops=2.78T (12.2% LRM), regen=0.52%, ckpt=trim_rubric_math200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.65
- target=72.45, actual=72.78, flops=2.65T (11.6% LRM), regen=0.47%, ckpt=trim_rubric_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0010.pt, th=0.35
- target=74.36, actual=74.56, flops=2.61T (11.4% LRM), regen=0.36%, ckpt=trim_rubric_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.55
- target=76.27, actual=75.74, flops=4.47T (19.6% LRM), regen=3.15%, ckpt=trim_rubric_math200_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.55
- target=78.17, actual=78.11, flops=5.22T (22.9% LRM), regen=12.10%, ckpt=trim_rubric_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.4
- target=80.08, actual=79.88, flops=12.10T (52.9% LRM), regen=43.68%, ckpt=trim_rubric_math200_point_search_lam5e-6_rub0.3_seed1/epoch_0030.pt, th=0.55
- target=81.99, actual=82.25, flops=15.73T (68.8% LRM), regen=64.80%, ckpt=trim_rubric_math200_point_search_lam5e-6_rub0.3_seed1/epoch_0040.pt, th=0.5
- target=83.89, actual=84.02, flops=19.52T (85.4% LRM), regen=87.57%, ckpt=trim_rubric_math200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.3
### TRIM-RubricV2 (PPO)
- Selected points: 8
- Limited by accuracy granularity: False
- target=70.55, actual=70.41, flops=2.51T (11.0% LRM), regen=0.08%, ckpt=trim_rubric_v2b_math200_point_search_lam0_rub0.3_seed1/epoch_0030.pt, th=0.6
- target=72.45, actual=72.19, flops=2.54T (11.1% LRM), regen=0.18%, ckpt=trim_rubric_v2b_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0020.pt, th=0.5
- target=74.36, actual=74.56, flops=2.62T (11.5% LRM), regen=0.36%, ckpt=trim_rubric_v2b_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.6
- target=76.27, actual=76.33, flops=7.79T (34.1% LRM), regen=27.03%, ckpt=trim_rubric_v2b_math200_point_search_lam2e-5_rub0.3_seed1/epoch_0020.pt, th=0.4
- target=78.17, actual=78.11, flops=8.45T (37.0% LRM), regen=25.81%, ckpt=trim_rubric_v2b_math200_point_search_lam0_rub0.3_seed1/epoch_0030.pt, th=0.55
- target=80.08, actual=79.88, flops=4.76T (20.8% LRM), regen=10.30%, ckpt=trim_rubric_v2b_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.45
- target=81.99, actual=82.25, flops=13.61T (59.6% LRM), regen=60.90%, ckpt=trim_rubric_v2b_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.3
- target=83.89, actual=84.02, flops=19.53T (85.5% LRM), regen=89.52%, ckpt=trim_rubric_v2b_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.35
