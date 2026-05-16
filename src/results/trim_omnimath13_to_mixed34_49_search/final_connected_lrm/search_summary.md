# TRIM Point Search Summary

Checkpoint patterns: {"ppo_agg": ["checkpoints/trim_agg_omnimath13_to34_point_search_*/*.pt"], "ppo_rubric": ["checkpoints/trim_rubric_omnimath13_to34_point_search_*/*.pt"], "ppo_rubric_v2": ["checkpoints/trim_rubric_v2b_omnimath13_to34_point_search_*/*.pt"]}
Selected point axis: accuracy

## OmniMath Mixed 3-4/4-9 Test-200
### TRIM-Agg (PPO)
- Selected points: 11
- Selection axis: accuracy
- Limited by accuracy granularity: False
- target_acc=52.38, actual_acc=51.50, flops=42.87T (9.6% LRM), regen=0.00%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt, th=0.8
- target_acc=53.25, actual_acc=53.00, flops=47.83T (10.7% LRM), regen=1.54%, ckpt=trim_agg_omnimath13_to34_point_search_lam5e-6_seed1/best.pt, th=0.75
- target_acc=54.12, actual_acc=54.50, flops=72.04T (16.2% LRM), regen=3.68%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/epoch_0010.pt, th=0.75
- target_acc=55.00, actual_acc=55.00, flops=72.78T (16.3% LRM), regen=5.71%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0030.pt, th=0.7
- target_acc=55.88, actual_acc=56.00, flops=90.72T (20.3% LRM), regen=6.53%, ckpt=trim_agg_omnimath13_to34_point_search_lam1e-4_seed1/epoch_0010.pt, th=0.65
- target_acc=56.75, actual_acc=56.50, flops=290.75T (65.2% LRM), regen=68.44%, ckpt=trim_agg_omnimath13_to34_point_search_lam1e-4_seed1/epoch_0020.pt, th=0.4
- target_acc=57.62, actual_acc=58.00, flops=231.95T (52.0% LRM), regen=37.42%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0040.pt, th=0.65
- target_acc=58.50, actual_acc=60.00, flops=267.35T (60.0% LRM), regen=48.15%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/best.pt, th=0.65
- target_acc=59.38, actual_acc=60.50, flops=438.88T (98.4% LRM), regen=99.33%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/epoch_0020.pt, th=0.7
- target_acc=60.25, actual_acc=61.50, flops=346.88T (77.8% LRM), regen=62.27%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/epoch_0040.pt, th=0.75
- target_acc=61.12, actual_acc=62.00, flops=370.09T (83.0% LRM), regen=71.78%, ckpt=trim_agg_omnimath13_to34_point_search_lam5e-6_seed1/epoch_0030.pt, th=0.75
### TRIM-Rubric (PPO)
- Selected points: 11
- Selection axis: accuracy
- Limited by accuracy granularity: False
- target_acc=52.38, actual_acc=52.00, flops=45.69T (10.2% LRM), regen=0.03%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0010.pt, th=0.75
- target_acc=53.25, actual_acc=53.00, flops=50.18T (11.3% LRM), regen=0.89%, ckpt=trim_rubric_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/epoch_0010.pt, th=0.65
- target_acc=54.12, actual_acc=53.50, flops=54.19T (12.2% LRM), regen=1.14%, ckpt=trim_rubric_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/best.pt, th=0.3
- target_acc=55.00, actual_acc=54.50, flops=54.48T (12.2% LRM), regen=1.38%, ckpt=trim_rubric_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.25
- target_acc=55.88, actual_acc=55.50, flops=87.24T (19.6% LRM), regen=6.20%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/best.pt, th=0.65
- target_acc=56.75, actual_acc=56.00, flops=81.13T (18.2% LRM), regen=5.37%, ckpt=trim_rubric_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/epoch_0010.pt, th=0.75
- target_acc=57.62, actual_acc=57.00, flops=319.40T (71.6% LRM), regen=76.16%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0020.pt, th=0.7
- target_acc=58.50, actual_acc=59.50, flops=241.30T (54.1% LRM), regen=43.32%, ckpt=trim_rubric_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/epoch_0020.pt, th=0.45
- target_acc=59.38, actual_acc=60.50, flops=186.98T (41.9% LRM), regen=27.13%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.65
- target_acc=60.25, actual_acc=61.50, flops=276.17T (61.9% LRM), regen=47.93%, ckpt=trim_rubric_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/epoch_0030.pt, th=0.75
- target_acc=61.12, actual_acc=62.00, flops=271.99T (61.0% LRM), regen=46.21%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0030.pt, th=0.75
### TRIM-RubricV2 (PPO)
- Selected points: 10
- Selection axis: accuracy
- Limited by accuracy granularity: True
- target_acc=52.38, actual_acc=51.50, flops=42.87T (9.6% LRM), regen=0.00%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/best.pt, th=0.8
- target_acc=53.25, actual_acc=52.50, flops=53.43T (12.0% LRM), regen=1.24%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.55
- target_acc=54.12, actual_acc=53.00, flops=47.31T (10.6% LRM), regen=0.35%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/best.pt, th=0.6
- target_acc=55.00, actual_acc=54.00, flops=53.94T (12.1% LRM), regen=1.46%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.6
- target_acc=55.88, actual_acc=54.50, flops=125.35T (28.1% LRM), regen=22.43%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt, th=0.25
- target_acc=56.75, actual_acc=56.50, flops=95.27T (21.4% LRM), regen=7.40%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam1e-4_rub0.3_seed1/epoch_0010.pt, th=0.65
- target_acc=57.62, actual_acc=57.50, flops=152.27T (34.2% LRM), regen=20.25%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/epoch_0010.pt, th=0.75
- target_acc=59.38, actual_acc=60.00, flops=178.53T (40.0% LRM), regen=25.27%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/best.pt, th=0.55
- target_acc=60.25, actual_acc=61.50, flops=238.40T (53.5% LRM), regen=38.55%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0040.pt, th=0.45
- target_acc=61.12, actual_acc=62.00, flops=265.78T (59.6% LRM), regen=46.14%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0020.pt, th=0.65
