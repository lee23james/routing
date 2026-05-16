# TRIM Point Search Summary

Checkpoint patterns: {"ppo_agg": ["checkpoints/trim_agg_omnimath13_to34_point_search_*/*.pt"], "ppo_rubric": ["checkpoints/trim_rubric_omnimath13_to34_point_search_*/*.pt"], "ppo_rubric_v2": ["checkpoints/trim_rubric_v2b_omnimath13_to34_point_search_*/*.pt"]}
Selected point axis: accuracy

## OmniMath Mixed 3-4/4-9 Test-200
### TRIM-Agg (PPO)
- Selected points: 11
- Selection axis: accuracy
- Limited by accuracy granularity: False
- target_acc=52.38, actual_acc=52.50, flops=45.40T (10.2% LRM), regen=1.18%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0030.pt, th=0.7023
- target_acc=53.25, actual_acc=53.00, flops=46.13T (10.3% LRM), regen=0.15%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt, th=0.7411
- target_acc=54.12, actual_acc=54.00, flops=55.68T (12.5% LRM), regen=1.78%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt, th=0.739
- target_acc=55.00, actual_acc=55.00, flops=57.36T (12.9% LRM), regen=3.11%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0030.pt, th=0.70125
- target_acc=55.88, actual_acc=56.00, flops=78.94T (17.7% LRM), regen=4.92%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt, th=0.73655
- target_acc=56.75, actual_acc=57.00, flops=117.76T (26.4% LRM), regen=12.37%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt, th=0.7687
- target_acc=57.62, actual_acc=57.50, flops=109.99T (24.7% LRM), regen=9.82%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt, th=0.76905
- target_acc=58.50, actual_acc=58.50, flops=169.21T (38.0% LRM), regen=22.72%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt, th=0.72675
- target_acc=59.38, actual_acc=59.50, flops=181.08T (40.6% LRM), regen=25.86%, ckpt=trim_agg_omnimath13_to34_point_search_lam2e-5_seed1/epoch_0010.pt, th=0.72535
- target_acc=60.25, actual_acc=60.00, flops=180.14T (40.4% LRM), regen=25.77%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt, th=0.76695
- target_acc=61.12, actual_acc=61.00, flops=189.26T (42.4% LRM), regen=28.34%, ckpt=trim_agg_omnimath13_to34_point_search_lam0_seed1/best.pt, th=0.7666
### TRIM-Rubric (PPO)
- Selected points: 11
- Selection axis: accuracy
- Limited by accuracy granularity: False
- target_acc=52.38, actual_acc=52.50, flops=52.17T (11.7% LRM), regen=1.11%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0010.pt, th=0.7348
- target_acc=53.25, actual_acc=53.00, flops=46.07T (10.3% LRM), regen=0.15%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.6932
- target_acc=54.12, actual_acc=54.00, flops=50.61T (11.3% LRM), regen=0.92%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.69145
- target_acc=55.00, actual_acc=55.00, flops=65.71T (14.7% LRM), regen=3.11%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.68935
- target_acc=55.88, actual_acc=56.00, flops=80.51T (18.1% LRM), regen=6.21%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/best.pt, th=0.7481
- target_acc=56.75, actual_acc=57.00, flops=113.08T (25.4% LRM), regen=10.16%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/best.pt, th=0.74705
- target_acc=57.62, actual_acc=57.50, flops=120.20T (27.0% LRM), regen=12.44%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/best.pt, th=0.74635
- target_acc=58.50, actual_acc=58.50, flops=159.27T (35.7% LRM), regen=20.00%, ckpt=trim_rubric_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt, th=0.7593
- target_acc=59.38, actual_acc=59.50, flops=173.87T (39.0% LRM), regen=23.40%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.67885
- target_acc=60.25, actual_acc=60.00, flops=172.02T (38.6% LRM), regen=22.73%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.6792
- target_acc=61.12, actual_acc=61.00, flops=182.09T (40.8% LRM), regen=25.71%, ckpt=trim_rubric_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt, th=0.6778
### TRIM-RubricV2 (PPO)
- Selected points: 11
- Selection axis: accuracy
- Limited by accuracy granularity: False
- target_acc=52.38, actual_acc=52.50, flops=45.72T (10.3% LRM), regen=0.05%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt, th=0.7736
- target_acc=53.25, actual_acc=53.00, flops=46.68T (10.5% LRM), regen=0.27%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt, th=0.77325
- target_acc=54.12, actual_acc=54.00, flops=52.14T (11.7% LRM), regen=1.63%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/best.pt, th=0.72815
- target_acc=55.00, actual_acc=55.00, flops=62.15T (13.9% LRM), regen=3.09%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/best.pt, th=0.72745
- target_acc=55.88, actual_acc=56.00, flops=84.67T (19.0% LRM), regen=5.71%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam2e-5_rub0.3_seed1/epoch_0010.pt, th=0.7257
- target_acc=56.75, actual_acc=57.00, flops=111.52T (25.0% LRM), regen=10.02%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/best.pt, th=0.725
- target_acc=57.62, actual_acc=57.50, flops=133.53T (29.9% LRM), regen=14.37%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/best.pt, th=0.7236
- target_acc=58.50, actual_acc=58.50, flops=148.53T (33.3% LRM), regen=18.57%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt, th=0.7673
- target_acc=59.38, actual_acc=59.50, flops=174.70T (39.2% LRM), regen=23.93%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam5e-6_rub0.3_seed1/best.pt, th=0.72115
- target_acc=60.25, actual_acc=60.00, flops=173.79T (39.0% LRM), regen=23.94%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt, th=0.7659
- target_acc=61.12, actual_acc=61.00, flops=181.69T (40.7% LRM), regen=25.34%, ckpt=trim_rubric_v2b_omnimath13_to34_point_search_lam0_rub0.3_seed1/epoch_0040.pt, th=0.76555
