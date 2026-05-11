# TRIM-RubricV2 跑点方法

本文档记录当前 `routing/src` 代码库里如何使用 `TRIM-RubricV2` 做跑点、评测和画图。这里的 `RubricV2` 指的是：

1. 先用已有 episodes 生成一套基础 rubric 权重。
2. 先训练一个 `TRIM-Rubric` 的 Router0 checkpoint。
3. 用 Router0 在训练 episodes 上 rollout，计算每条轨迹的 routing utility。
4. 根据 rubric score 与 Router0 utility 的相关性重新调整 rubric weights。
5. 用 evolved rubric weights 再训练 `TRIM-RubricV2` 的 Router1。
6. 最后和 `TRIM-Agg`、`TRIM-Rubric` 一起评测并画图。

核心代码路径：

- `rubric/generate_rubrics.py`
- `rubric/evolve_rubric_weights.py`
- `router/train_ppo.py`
- `eval/plot_trim_agg_baseline.py`
- `scripts/search_trim_rubric_*_points_4gpu.sh`
- `scripts/search_trim_rubric_v2_*_points_4gpu.sh`

## 0. 前置概念

### Episodes

`TRIM-RubricV2` 不直接调用 SRM/LRM 生成答案，它复用已经生成好的 episode 文件。episode 文件里需要包含：

- `srm_steps`
- `lrm_steps`
- `srm_token_counts`
- `lrm_token_counts`
- `srm_prm_scores`
- `lrm_prm_scores`
- `srm_correct`
- `lrm_correct`

也就是说，跑 `RubricV2` 之前，必须先有 train/test episodes。

常见路径：

```bash
data/episodes/math_train_200_episodes.jsonl
data/episodes/math500_episodes.jsonl
data/episodes/gsm8k_train_300_episodes.jsonl
data/episodes/gsm8k_test_200_episodes.jsonl
```

检查行数：

```bash
cd /home/chencheng/routing/src

wc -l data/episodes/math_train_200_episodes.jsonl \
      data/episodes/math500_episodes.jsonl \
      data/episodes/gsm8k_train_300_episodes.jsonl \
      data/episodes/gsm8k_test_200_episodes.jsonl
```

### 三类方法

最终图里通常有三条 learned routing 曲线：

- `TRIM-Agg (PPO)`：只用原始聚合状态训练 router。
- `TRIM-Rubric (PPO)`：用人工/规则 rubric weights 加入训练 reward。
- `TRIM-RubricV2 (PPO)`：先用 Router0 反馈 evolve rubric weights，再训练 Router1。

`RubricV2` 不是替代 `Rubric` 的独立入口，它依赖一个已有 `TRIM-Rubric` checkpoint 作为 Router0。

## 1. RubricV2 的内部流程

以 `scripts/search_trim_rubric_v2_gsm8k_points_4gpu.sh` 为例，流程是：

### 1.1 确认训练 episodes

脚本默认：

```bash
EPISODES_PATH=data/episodes/gsm8k_train_300_episodes.jsonl
```

如果文件不存在，脚本直接退出：

```bash
episodes file not found: ...
```

### 1.2 准备基础 rubric weights

脚本默认：

```bash
BASE_RUBRIC_WEIGHTS=data/rubrics/gsm8k_train300/rubric_weights.json
```

如果这个文件不存在，会自动运行：

```bash
python -u -m rubric.generate_rubrics \
  --episodes_path "$EPISODES_PATH" \
  --output_dir "$(dirname "$BASE_RUBRIC_WEIGHTS")"
```

输出示例：

```bash
data/rubrics/gsm8k_train300/rubric_weights.json
```

这个文件会包含：

- `weights`
- `active_rubrics`
- rubric 诊断信息

当前常见 active rubrics：

```text
timely_escalation
cascading_error_prevention
difficulty_awareness
early_detection
prm_trajectory_quality
critical_step_coverage
```

### 1.3 准备 Router0 checkpoint

脚本默认：

```bash
ROUTER0_CHECKPOINT=checkpoints/trim_rubric_gsm8k_train300_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt
```

这个 checkpoint 必须来自普通 `TRIM-Rubric` 训练。也就是说，第一次跑某个数据集时，应该先跑：

```bash
bash scripts/search_trim_rubric_gsm8k_points_4gpu.sh
```

如果 Router0 checkpoint 不存在，V2 脚本会退出并提示：

```bash
Router0 checkpoint not found: ...
Run scripts/search_trim_rubric_gsm8k_points_4gpu.sh first, or set ROUTER0_CHECKPOINT.
```

### 1.4 Evolve RubricV2 weights

脚本默认输出：

```bash
RUBRIC_DIR=data/rubrics/gsm8k_train300_v2
RUBRIC_WEIGHTS=data/rubrics/gsm8k_train300_v2/rubric_weights_v2.json
```

如果 `rubric_weights_v2.json` 不存在，会运行：

```bash
python -u -m rubric.evolve_rubric_weights \
  --episodes_path "$EPISODES_PATH" \
  --base_rubric_weights "$BASE_RUBRIC_WEIGHTS" \
  --router_checkpoint "$ROUTER0_CHECKPOINT" \
  --output_dir "$RUBRIC_DIR" \
  --output_name "$(basename "$RUBRIC_WEIGHTS")" \
  --lam "$EVOLVE_LAM" \
  --alpha "$ALPHA" \
  --device cpu
```

默认参数：

```bash
EVOLVE_LAM=2e-5
ALPHA=0.3
```

含义：

- `EVOLVE_LAM`：Router0 utility 里的 cost 惩罚系数。
- `ALPHA`：base rubric weights 和 router-feedback weights 的混合比例。
- `ALPHA=0.3` 表示保留 70% base weights，注入 30% router-feedback weights。

evolve 的 utility 定义在 `rubric/evolve_rubric_weights.py`：

```python
utility = correctness - lam * total_lrm_tokens
```

注意：这里故意不把 rubric reward 加进 utility。V2 的目标是让 Router0 告诉我们哪些 rubric score 更能解释真实的 task/cost trade-off。

### 1.5 用 evolved weights 训练 Router1

V2 训练还是调用同一个 PPO 入口：

```bash
python -u -m router.train_ppo \
  --episodes_path "$EPISODES_PATH" \
  --lam "$lam" \
  --lam_rubric "$LAM_RUBRIC" \
  --rubric_weights "$RUBRIC_WEIGHTS" \
  --num_epochs "$NUM_EPOCHS" \
  --episodes_per_epoch "$EPISODES_PER_EPOCH" \
  --device cuda:0 \
  --save_dir "$save_dir" \
  --save_every "$SAVE_EVERY" \
  --save_epoch_checkpoints \
  --seed "$SEED"
```

默认训练参数：

```bash
LAM_RUBRIC=0.3
NUM_EPOCHS=40
EPISODES_PER_EPOCH=64
SAVE_EVERY=10
SEED=1
```

GSM8K 默认 lambda 网格：

```bash
0
5e-6
1e-5
2e-5
5e-5
1e-4
```

MATH 默认 lambda 网格：

```bash
0
5e-6
2e-5
1e-4
```

每个 lambda 会保存：

```text
best.pt
epoch_0010.pt
epoch_0020.pt
epoch_0030.pt
epoch_0040.pt
final.pt
metadata.json
train_log.json
```

## 2. MATH 上跑 RubricV2

MATH 已有一键入口：

```bash
cd /home/chencheng/routing/src

bash scripts/run_trim_rubric_v2_math_quick_search.sh
```

这个脚本会做两件事：

1. 跑 `scripts/search_trim_rubric_v2_math_points_4gpu.sh`
2. 跑 `scripts/eval_math_rubric_v2_final.sh`

默认输入：

```bash
TRAIN_EPISODES=data/episodes/math_train_200_episodes.jsonl
TEST_EPISODES=data/episodes/math500_episodes.jsonl
```

默认 V2 weights：

```bash
data/rubrics/math200_v2/rubric_weights_v2.json
```

默认 Router0：

```bash
checkpoints/trim_rubric_math200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt
```

默认 V2 checkpoints：

```bash
checkpoints/trim_rubric_v2_math200_point_search_lam*_rub0.3_seed1/
```

默认 final 输出：

```bash
results/trim_rubric_v2_math200_point_search/final/
```

如果只想训练 V2，不评测：

```bash
cd /home/chencheng/routing/src

SKIP_EVAL=true \
bash scripts/run_trim_rubric_v2_math_quick_search.sh
```

如果 V2 checkpoints 已经存在，只想重新评测画图：

```bash
cd /home/chencheng/routing/src

SKIP_TRAIN=true \
bash scripts/run_trim_rubric_v2_math_quick_search.sh
```

如果要换 Router0 checkpoint：

```bash
cd /home/chencheng/routing/src

ROUTER0_CHECKPOINT=checkpoints/trim_rubric_math200_point_search_lam1e-4_rub0.3_seed1/epoch_0030.pt \
bash scripts/search_trim_rubric_v2_math_points_4gpu.sh
```

## 3. GSM8K 上跑 RubricV2

GSM8K 的总控脚本是：

```bash
cd /home/chencheng/routing/src

bash scripts/run_trim_gsm8k_rubric_search.sh
```

这个脚本会：

1. 跑普通 `TRIM-Rubric`。
2. 跑 `TRIM-RubricV2`。
3. 用同一套 test episodes 评测 `TRIM-Agg + TRIM-Rubric + TRIM-RubricV2`。
4. 输出主图、主表、60/98 指标和 selected points。

默认输入：

```bash
TRAIN_EPISODES=data/episodes/gsm8k_train_300_episodes.jsonl
TEST_EPISODES=data/episodes/gsm8k_test_200_episodes.jsonl
```

默认 base rubric weights：

```bash
data/rubrics/gsm8k_train300/rubric_weights.json
```

默认 V2 weights：

```bash
data/rubrics/gsm8k_train300_v2/rubric_weights_v2.json
```

默认 Router0：

```bash
checkpoints/trim_rubric_gsm8k_train300_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt
```

默认输出：

```bash
results/trim_gsm8k_train300_test200_rubric_compare/final/
```

后台运行推荐用 tmux：

```bash
cd /home/chencheng/routing/src

mkdir -p logs/trim_gsm8k_rubric_search
LOG=logs/trim_gsm8k_rubric_search/run_$(date +%Y%m%d_%H%M%S).log

tmux new-session -d -s trim_gsm8k_rubric_search \
  "cd /home/chencheng/routing/src && \
   env NUM_EPOCHS=40 EPISODES_PER_EPOCH=64 SAVE_EVERY=10 \
   bash scripts/run_trim_gsm8k_rubric_search.sh 2>&1 | tee '$LOG'"
```

查看进度：

```bash
tmux ls | grep trim_gsm8k_rubric_search
tail -f logs/trim_gsm8k_rubric_search/run_*.log
nvidia-smi
```

只跑 RubricV2，不重新跑普通 Rubric：

```bash
cd /home/chencheng/routing/src

SKIP_RUBRIC_TRAIN=true \
bash scripts/run_trim_gsm8k_rubric_search.sh
```

只重新评测画图，不训练：

```bash
cd /home/chencheng/routing/src

SKIP_RUBRIC_TRAIN=true \
SKIP_RUBRIC_V2_TRAIN=true \
bash scripts/run_trim_gsm8k_rubric_search.sh
```

换 V2 输出目录，避免覆盖旧结果：

```bash
cd /home/chencheng/routing/src

OUTPUT_DIR=results/trim_gsm8k_train300_test200_rubric_compare/redo_v2 \
bash scripts/run_trim_gsm8k_rubric_search.sh
```

## 4. 手动分步跑法

如果不想用总控脚本，可以按下面顺序跑。

### 4.1 先跑普通 TRIM-Rubric

```bash
cd /home/chencheng/routing/src

EPISODES_PATH=data/episodes/gsm8k_train_300_episodes.jsonl \
RUBRIC_DIR=data/rubrics/gsm8k_train300 \
RUBRIC_WEIGHTS=data/rubrics/gsm8k_train300/rubric_weights.json \
SEARCH_NAME=trim_rubric_gsm8k_train300_point_search \
NUM_EPOCHS=40 \
EPISODES_PER_EPOCH=64 \
SAVE_EVERY=10 \
bash scripts/search_trim_rubric_gsm8k_points_4gpu.sh
```

### 4.2 再跑 TRIM-RubricV2

```bash
cd /home/chencheng/routing/src

EPISODES_PATH=data/episodes/gsm8k_train_300_episodes.jsonl \
BASE_RUBRIC_WEIGHTS=data/rubrics/gsm8k_train300/rubric_weights.json \
RUBRIC_DIR=data/rubrics/gsm8k_train300_v2 \
RUBRIC_WEIGHTS=data/rubrics/gsm8k_train300_v2/rubric_weights_v2.json \
ROUTER0_CHECKPOINT=checkpoints/trim_rubric_gsm8k_train300_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt \
SEARCH_NAME=trim_rubric_v2_gsm8k_train300_point_search \
EVOLVE_LAM=2e-5 \
ALPHA=0.3 \
NUM_EPOCHS=40 \
EPISODES_PER_EPOCH=64 \
SAVE_EVERY=10 \
bash scripts/search_trim_rubric_v2_gsm8k_points_4gpu.sh
```

### 4.3 最后评测和画图

```bash
cd /home/chencheng/routing/src

python -u -m eval.plot_trim_agg_baseline \
  --datasets gsm8k_test_200 \
  --gsm8k_episodes data/episodes/gsm8k_test_200_episodes.jsonl \
  --checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/*.pt' \
  --agg_checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/*.pt' \
  --rubric_checkpoint_glob 'checkpoints/trim_rubric_gsm8k_train300_point_search_*/*.pt' \
  --rubric_v2_checkpoint_glob 'checkpoints/trim_rubric_v2_gsm8k_train300_point_search_*/*.pt' \
  --output_dir results/trim_gsm8k_train300_test200_rubric_compare/final \
  --n_selected_points 11 \
  --device cuda:0
```

## 5. 更细的局部 TFLOPs 曲线

如果某个 TFLOPs 区间点太稀，不需要重新训练，只需要用更密 threshold 重新评测。

当前 `eval/plot_trim_agg_baseline.py` 支持：

```bash
--thresholds "0.80,0.79,...,0.20"
```

例子：细化 GSM8K 的 20-60 TFLOPs 段。

```bash
cd /home/chencheng/routing/src

THRESH=$(python - <<'PY'
print(','.join(f'{x/100:.2f}' for x in range(80, 19, -1)))
PY
)

python -u -m eval.plot_trim_agg_baseline \
  --datasets gsm8k_test_200 \
  --gsm8k_episodes data/episodes/gsm8k_test_200_episodes.jsonl \
  --checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/*.pt' \
  --agg_checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/*.pt' \
  --rubric_checkpoint_glob 'checkpoints/trim_rubric_gsm8k_train300_point_search_*/*.pt' \
  --rubric_v2_checkpoint_glob 'checkpoints/trim_rubric_v2_gsm8k_train300_point_search_*/*.pt' \
  --output_dir results/trim_gsm8k_train300_test200_rubric_compare/dense_20_60 \
  --n_selected_points 21 \
  --thresholds "$THRESH" \
  --device cuda:0
```

后台运行：

```bash
cd /home/chencheng/routing/src

mkdir -p logs/trim_gsm8k_dense_20_60
LOG=logs/trim_gsm8k_dense_20_60/run_$(date +%Y%m%d_%H%M%S).log

THRESH=$(python - <<'PY'
print(','.join(f'{x/100:.2f}' for x in range(80, 19, -1)))
PY
)

tmux new-session -d -s trim_gsm8k_dense_20_60 \
  "cd /home/chencheng/routing/src && \
   python -u -m eval.plot_trim_agg_baseline \
     --datasets gsm8k_test_200 \
     --gsm8k_episodes data/episodes/gsm8k_test_200_episodes.jsonl \
     --checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/*.pt' \
     --agg_checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/*.pt' \
     --rubric_checkpoint_glob 'checkpoints/trim_rubric_gsm8k_train300_point_search_*/*.pt' \
     --rubric_v2_checkpoint_glob 'checkpoints/trim_rubric_v2_gsm8k_train300_point_search_*/*.pt' \
     --output_dir results/trim_gsm8k_train300_test200_rubric_compare/dense_20_60 \
     --n_selected_points 21 \
     --thresholds '$THRESH' \
     --device cuda:0 \
   2>&1 | tee '$LOG'"
```

说明：

- 这一步只重评 checkpoint，不重训。
- 输出目录建议单独开，避免覆盖正式结果。
- threshold 越密，评测时间越久。

## 6. 迁移到新数据集的模板

假设新数据集叫 `foo`，训练 episodes 是 `data/episodes/foo_train_200_episodes.jsonl`，测试 episodes 是 `data/episodes/foo_test_100_episodes.jsonl`。

需要准备：

1. `foo` 的 train/test episodes。
2. 一个普通 `TRIM-Rubric` 搜点脚本，可以复制 `scripts/search_trim_rubric_gsm8k_points_4gpu.sh` 改名。
3. 一个 `TRIM-RubricV2` 搜点脚本，可以复制 `scripts/search_trim_rubric_v2_gsm8k_points_4gpu.sh` 改名。
4. eval 入口里要能识别 `foo_test_100`。如果 `eval/plot_trim_agg_baseline.py` 还没有注册该 dataset，需要补：
   - `DEFAULT_EPISODES`
   - `DATASETS`
   - `DS_LABELS`
   - CLI 参数，例如 `--foo_episodes`
   - `episode_paths`

建议命名：

```bash
data/rubrics/foo_train200/rubric_weights.json
data/rubrics/foo_train200_v2/rubric_weights_v2.json
checkpoints/trim_rubric_foo_train200_point_search_lam*_rub0.3_seed1/
checkpoints/trim_rubric_v2_foo_train200_point_search_lam*_rub0.3_seed1/
results/trim_foo_train200_test100_rubric_compare/final/
```

Router0 建议先用普通 `TRIM-Rubric` 的中间 checkpoint：

```bash
checkpoints/trim_rubric_foo_train200_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt
```

如果这个点在该数据集上明显不合适，可以改成普通 Rubric 搜点中表现较稳的 checkpoint。

## 7. 输出文件说明

final eval 会输出：

```text
accuracy_vs_flops.png
accuracy_vs_flops.pdf
main_comparison.png
main_comparison.pdf
plot_data.json
main_results.json
main_results.md
main_results.tex
trim_agg_<dataset>_60_98.md
trim_rubric_<dataset>_60_98.md
trim_rubric_v2_<dataset>_60_98.md
trim_<dataset>_60_98_compare.md
selected_points_trim_agg_<dataset>.csv
selected_points_trim_rubric_<dataset>.csv
selected_points_trim_rubric_v2_<dataset>.csv
search_summary.md
```

最常看的文件：

- `main_results.md`：主表，包含 `Acc@60% LRM FLOPs` 和 `FLOPs@98% LRM Acc`。
- `trim_<dataset>_60_98_compare.md`：三种 TRIM 方法的 60/98 指标对比。
- `accuracy_vs_flops.png`：横轴是 TFLOPs/query。
- `main_comparison.png`：横轴是 `% LRM FLOPs`。
- `search_summary.md`：均匀选点的 checkpoint、threshold、accuracy、FLOPs。
- `plot_data.json`：完整曲线数据，可用于后处理。

## 8. 常见问题

### Router0 checkpoint 不存在

先跑普通 Rubric：

```bash
bash scripts/search_trim_rubric_gsm8k_points_4gpu.sh
```

或者显式指定已有 checkpoint：

```bash
ROUTER0_CHECKPOINT=checkpoints/.../epoch_0030.pt \
bash scripts/search_trim_rubric_v2_gsm8k_points_4gpu.sh
```

### 想重新 evolve weights

删除旧的 V2 weights 后重跑：

```bash
rm -f data/rubrics/gsm8k_train300_v2/rubric_weights_v2.json
bash scripts/search_trim_rubric_v2_gsm8k_points_4gpu.sh
```

也可以换输出目录：

```bash
RUBRIC_DIR=data/rubrics/gsm8k_train300_v2_alpha05 \
RUBRIC_WEIGHTS=data/rubrics/gsm8k_train300_v2_alpha05/rubric_weights_v2.json \
ALPHA=0.5 \
bash scripts/search_trim_rubric_v2_gsm8k_points_4gpu.sh
```

### 想只看 V2 weights，不训练

当前脚本没有单独的 `SKIP_TRAIN` 开关。可以直接调用 evolve 模块：

```bash
cd /home/chencheng/routing/src

python -u -m rubric.evolve_rubric_weights \
  --episodes_path data/episodes/gsm8k_train_300_episodes.jsonl \
  --base_rubric_weights data/rubrics/gsm8k_train300/rubric_weights.json \
  --router_checkpoint checkpoints/trim_rubric_gsm8k_train300_point_search_lam2e-5_rub0.3_seed1/epoch_0030.pt \
  --output_dir data/rubrics/gsm8k_train300_v2_probe \
  --output_name rubric_weights_v2.json \
  --lam 2e-5 \
  --alpha 0.3 \
  --device cpu
```

### GPU 不够

训练脚本通过 `CUDA_VISIBLE_DEVICES="$gpu"` 单卡跑每个 lambda，并行度由 `GPUS=(0 1 2 3)` 控制。

如果只想用两张卡，改脚本里的：

```bash
GPUS=(0 1)
```

或者复制脚本到临时版本后修改，避免影响正式入口。

### 评测太慢

默认会扫所有 checkpoint 和所有 thresholds。可以减少 checkpoint glob，例如只扫 `best.pt` 和 `epoch_0030.pt`：

```bash
python -u -m eval.plot_trim_agg_baseline \
  --datasets gsm8k_test_200 \
  --gsm8k_episodes data/episodes/gsm8k_test_200_episodes.jsonl \
  --agg_checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/epoch_0030.pt' \
  --rubric_checkpoint_glob 'checkpoints/trim_rubric_gsm8k_train300_point_search_*/epoch_0030.pt' \
  --rubric_v2_checkpoint_glob 'checkpoints/trim_rubric_v2_gsm8k_train300_point_search_*/epoch_0030.pt' \
  --checkpoint_glob 'checkpoints/trim_agg_gsm8k_train300_point_search_*/epoch_0030.pt' \
  --output_dir results/tmp_eval \
  --n_selected_points 8 \
  --device cuda:0
```

### 结果看起来 V2 没有超过 Rubric

这是可能的。V2 只是在当前 Router0 utility 下重新加权 rubric，不保证每个数据集都严格优于普通 Rubric。需要重点看两个指标：

- `Acc@60% LRM FLOPs`
- `FLOPs@98% LRM Acc`

如果 V2 在一个指标上更好、另一个指标上更差，要结合论文表格主指标决定是否保留。

## 9. 当前已验证过的 GSM8K 结果口径

GSM8K 使用：

```text
train: gsm8k_train_300_episodes.jsonl
test: gsm8k_test_200_episodes.jsonl
```

主结果目录：

```bash
results/trim_gsm8k_train300_test200_rubric_compare/final
```

这轮结果：

```text
SRM-Only: 84.5%
LRM-Only: 93.5%
Random Routing: Acc@60%=89.5%, FLOPs@98%=82.1%
TRIM-Agg: Acc@60%=94.5%, FLOPs@98%=22.4%
TRIM-Rubric: Acc@60%=92.9%, FLOPs@98%=17.4%
TRIM-RubricV2: Acc@60%=93.5%, FLOPs@98%=18.0%
```

解释：

- `TRIM-Agg` 在 `Acc@60% LRM FLOPs` 上最好。
- `TRIM-Rubric` 在 `FLOPs@98% LRM Acc` 上最省。
- `TRIM-RubricV2` 居中，准确率比 Rubric 高，但省 FLOPs 稍差于 Rubric。
