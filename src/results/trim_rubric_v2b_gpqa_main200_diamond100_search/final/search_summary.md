# TRIM Point Search Summary

Checkpoint patterns: {"ppo_agg": ["checkpoints/trim_agg_gpqa_main200_point_search_*/*.pt"], "ppo_rubric": ["checkpoints/trim_rubric_gpqa_main200_point_search_*/*.pt"], "ppo_rubric_v2": ["checkpoints/trim_rubric_v2b_gpqa_main200_point_search_*/*.pt"]}

## GPQA Diamond-100
### TRIM-Agg (PPO)
- Selected points: 8
- Limited by accuracy granularity: False
- target=29.56, actual=29.00, flops=16.96T (15.0% LRM), regen=4.16%, ckpt=trim_agg_gpqa_main200_point_search_lam0_seed1/epoch_0030.pt, th=0.8
- target=31.11, actual=30.00, flops=13.96T (12.3% LRM), regen=1.53%, ckpt=trim_agg_gpqa_main200_point_search_lam5e-6_seed1/best.pt, th=0.8
- target=32.67, actual=33.00, flops=32.04T (28.3% LRM), regen=22.39%, ckpt=trim_agg_gpqa_main200_point_search_lam0_seed1/epoch_0020.pt, th=0.75
- target=34.22, actual=35.00, flops=30.73T (27.1% LRM), regen=20.90%, ckpt=trim_agg_gpqa_main200_point_search_lam2e-5_seed1/epoch_0040.pt, th=0.75
- target=35.78, actual=36.00, flops=39.55T (34.9% LRM), regen=29.82%, ckpt=trim_agg_gpqa_main200_point_search_lam5e-6_seed1/epoch_0040.pt, th=0.75
- target=37.33, actual=37.00, flops=46.20T (40.8% LRM), regen=37.15%, ckpt=trim_agg_gpqa_main200_point_search_lam0_seed1/epoch_0030.pt, th=0.75
- target=38.89, actual=39.00, flops=59.71T (52.7% LRM), regen=51.30%, ckpt=trim_agg_gpqa_main200_point_search_lam2e-5_seed1/best.pt, th=0.7
- target=40.44, actual=40.00, flops=71.31T (63.0% LRM), regen=62.28%, ckpt=trim_agg_gpqa_main200_point_search_lam0_seed1/epoch_0040.pt, th=0.7
### TRIM-Rubric (PPO)
- Selected points: 8
- Limited by accuracy granularity: False
- target=29.56, actual=30.00, flops=15.77T (13.9% LRM), regen=3.36%, ckpt=trim_rubric_gpqa_main200_point_search_lam5e-6_rub0.3_seed1/epoch_0030.pt, th=0.85
- target=31.11, actual=31.00, flops=14.51T (12.8% LRM), regen=1.53%, ckpt=trim_rubric_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.7
- target=32.67, actual=34.00, flops=26.40T (23.3% LRM), regen=15.33%, ckpt=trim_rubric_gpqa_main200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.8
- target=34.22, actual=35.00, flops=22.30T (19.7% LRM), regen=9.84%, ckpt=trim_rubric_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.7
- target=35.78, actual=36.00, flops=23.96T (21.2% LRM), regen=12.01%, ckpt=trim_rubric_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.6
- target=37.33, actual=37.00, flops=42.91T (37.9% LRM), regen=32.19%, ckpt=trim_rubric_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.65
- target=38.89, actual=39.00, flops=56.88T (50.2% LRM), regen=48.44%, ckpt=trim_rubric_gpqa_main200_point_search_lam0_rub0.3_seed1/epoch_0030.pt, th=0.8
- target=40.44, actual=40.00, flops=66.81T (59.0% LRM), regen=58.16%, ckpt=trim_rubric_gpqa_main200_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.75
### TRIM-RubricV2 (PPO)
- Selected points: 8
- Limited by accuracy granularity: False
- target=29.56, actual=30.00, flops=16.59T (14.6% LRM), regen=3.62%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.8
- target=31.11, actual=32.00, flops=19.72T (17.4% LRM), regen=7.70%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.8
- target=32.67, actual=33.00, flops=19.21T (17.0% LRM), regen=6.33%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.6
- target=34.22, actual=34.00, flops=19.20T (17.0% LRM), regen=6.29%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.65
- target=35.78, actual=36.00, flops=25.46T (22.5% LRM), regen=13.50%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/epoch_0040.pt, th=0.55
- target=37.33, actual=37.00, flops=27.00T (23.8% LRM), regen=14.38%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.65
- target=38.89, actual=39.00, flops=58.51T (51.7% LRM), regen=50.00%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.7
- target=40.44, actual=40.00, flops=68.79T (60.7% LRM), regen=60.14%, ckpt=trim_rubric_v2b_gpqa_main200_point_search_lam5e-6_rub0.3_seed1/epoch_0030.pt, th=0.75
