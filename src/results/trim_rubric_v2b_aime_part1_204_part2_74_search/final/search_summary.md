# TRIM Point Search Summary

Checkpoint patterns: {"ppo_agg": ["checkpoints/trim_agg_aime_part1_204_point_search_*/*.pt"], "ppo_rubric": ["checkpoints/trim_rubric_aime_part1_204_point_search_*/*.pt"], "ppo_rubric_v2": ["checkpoints/trim_rubric_v2b_aime_part1_204_point_search_*/*.pt"]}

## AIME 2020-2024 Part II
### TRIM-Agg (PPO)
- Selected points: 7
- Limited by accuracy granularity: True
- target=10.36, actual=9.46, flops=13.38T (10.7% LRM), regen=0.00%, ckpt=trim_agg_aime_part1_204_point_search_lam0_seed1/best.pt, th=0.98
- target=11.26, actual=10.81, flops=15.98T (12.8% LRM), regen=3.93%, ckpt=trim_agg_aime_part1_204_point_search_lam1e-4_seed1/epoch_0020.pt, th=0.1
- target=12.16, actual=12.16, flops=42.48T (34.0% LRM), regen=31.96%, ckpt=trim_agg_aime_part1_204_point_search_lam4e-5_seed1/epoch_0040.pt, th=0.8
- target=13.06, actual=13.51, flops=51.82T (41.5% LRM), regen=41.87%, ckpt=trim_agg_aime_part1_204_point_search_lam4e-5_seed1/epoch_0030.pt, th=0.75
- target=14.86, actual=14.86, flops=67.86T (54.3% LRM), regen=56.76%, ckpt=trim_agg_aime_part1_204_point_search_lam1e-5_seed1/epoch_0010.pt, th=0.65
- target=15.77, actual=16.22, flops=58.26T (46.6% LRM), regen=47.99%, ckpt=trim_agg_aime_part1_204_point_search_lam4e-5_seed1/epoch_0040.pt, th=0.75
- target=16.67, actual=17.57, flops=82.37T (65.9% LRM), regen=70.27%, ckpt=trim_agg_aime_part1_204_point_search_lam4e-5_seed1/epoch_0020.pt, th=0.6
### TRIM-Rubric (PPO)
- Selected points: 7
- Limited by accuracy granularity: True
- target=10.36, actual=9.46, flops=13.38T (10.7% LRM), regen=0.00%, ckpt=trim_rubric_aime_part1_204_point_search_lam0_rub0.3_seed1/best.pt, th=0.99
- target=11.26, actual=10.81, flops=14.11T (11.3% LRM), regen=1.05%, ckpt=trim_rubric_aime_part1_204_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.65
- target=12.16, actual=12.16, flops=15.14T (12.1% LRM), regen=1.74%, ckpt=trim_rubric_aime_part1_204_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.6
- target=13.06, actual=13.51, flops=45.44T (36.3% LRM), regen=33.61%, ckpt=trim_rubric_aime_part1_204_point_search_lam8e-5_rub0.3_seed1/epoch_0020.pt, th=0.5
- target=14.86, actual=14.86, flops=52.74T (42.2% LRM), regen=41.05%, ckpt=trim_rubric_aime_part1_204_point_search_lam6e-5_rub0.3_seed1/best.pt, th=0.8
- target=15.77, actual=16.22, flops=52.62T (42.1% LRM), regen=43.11%, ckpt=trim_rubric_aime_part1_204_point_search_lam6e-5_rub0.3_seed1/epoch_0020.pt, th=0.8
- target=16.67, actual=17.57, flops=82.40T (65.9% LRM), regen=70.55%, ckpt=trim_rubric_aime_part1_204_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.98
### TRIM-RubricV2 (PPO)
- Selected points: 7
- Limited by accuracy granularity: True
- target=10.36, actual=9.46, flops=13.38T (10.7% LRM), regen=0.00%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam0_rub0.3_seed1/best.pt, th=0.99
- target=11.26, actual=10.81, flops=14.17T (11.3% LRM), regen=1.14%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.6
- target=12.16, actual=12.16, flops=15.16T (12.1% LRM), regen=1.78%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.55
- target=13.06, actual=13.51, flops=41.78T (33.4% LRM), regen=30.41%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam8e-5_rub0.3_seed1/epoch_0020.pt, th=0.45
- target=14.86, actual=14.86, flops=50.44T (40.3% LRM), regen=39.73%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam1e-4_rub0.3_seed1/epoch_0020.pt, th=0.4
- target=15.77, actual=16.22, flops=71.94T (57.5% LRM), regen=60.32%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam6e-5_rub0.3_seed1/epoch_0020.pt, th=0.85
- target=16.67, actual=17.57, flops=72.45T (58.0% LRM), regen=60.96%, ckpt=trim_rubric_v2b_aime_part1_204_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.98
