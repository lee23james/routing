# TRIM: Stepwise Model Routing with Rubric-Guided Process Reward

基于 PRM (Process Reward Model) 的逐步推理路由，在 SRM (Qwen3-1.7B) 和 LRM (Qwen3-14B) 之间动态切换，以在准确率和计算成本之间取得最优平衡。

## 目录结构

```
pp5/
├── data/                        # 数据文件
│   ├── math500.jsonl            # MATH-500 测试集
│   ├── aime2025-I.jsonl         # AIME 2025 Part I
│   ├── aime2025-II.jsonl        # AIME 2025 Part II
│   ├── episodes/                # RL 训练用 episode 数据
│   │   ├── math500_episodes.jsonl
│   │   ├── aime2025_episodes.jsonl
│   │   └── combined_episodes.jsonl
│   └── rubrics/                 # Rubric 权重
├── src/                         # 源代码
│   ├── config.py                # 全局配置 (模型路径、超参数)
│   ├── models.py                # PRM、LLM wrapper、answer extraction
│   ├── vllm_client.py           # vLLM API 客户端
│   ├── baseline/                # Baseline 实现
│   │   ├── run_baseline.py      #   SRM-only / LRM-only
│   │   └── run_untrained_router.py  #   随机路由 (untrained router)
│   ├── data/                    # 数据加载 & episode 生成
│   │   ├── datasets.py
│   │   └── generate_episodes.py
│   ├── router/                  # 路由器
│   │   ├── policy.py            #   MLP 策略网络
│   │   ├── env.py               #   TRIM 环境
│   │   └── train_ppo.py         #   PPO 训练
│   ├── rubric/                  # Rubric process reward
│   │   ├── rubric_scorer.py
│   │   └── generate_rubrics.py
│   ├── eval/                    # 评估
│   │   ├── evaluate.py          #   在线/离线评估
│   │   ├── flops_eval.py        #   FLOPs 计算 & 准确率估计
│   │   ├── plot_clean.py        #   Accuracy-vs-FLOPs 图
│   │   ├── print_results.py     #   统一数字输出
│   │   └── verify_results.py    #   结果验证
│   ├── scripts/                 # 运行脚本 (见下方)
│   └── vllm/                    # vLLM 启动脚本
├── checkpoints/                 # 训练好的路由器
├── results/                     # 结果输出
│   ├── baselines/               #   Baseline 结果
│   ├── plots/                   #   图表 (png/pdf)
│   ├── summary_table.json       #   统一数字对比表
│   └── ...
└── archive/                     # 归档文件 (论文、备份等)
```

## 快速开始

### 前提条件
- Python 3.10+, PyTorch, transformers, vllm, openai, tqdm, matplotlib
- GPU: 至少 2 张 (SRM 1张, LRM 2张 TP)
- 模型权重: `/export/yuguo/ppyg2/model/{qwen3-1.7b, qwen3-14b, qwen2.5-math-prm-7b}`

### 虚拟环境与依赖

当前 `trim/TRIM` 子项目已经提供了可复现环境文件，推荐优先使用：

```bash
cd /home/chencheng/routing/trim/TRIM
conda env create -f environment.yml
conda activate trim
```

如果环境已经存在，更新方式为：

```bash
cd /home/chencheng/routing/trim/TRIM
conda env update -f environment.yml --prune
conda activate trim
```

如果你更想手动创建虚拟环境，最小安装步骤是：

```bash
conda create -n trim python=3.11 -y
conda activate trim
pip install -r /home/chencheng/routing/trim/TRIM/requirements.txt
```

实验当前依赖的核心 Python 包如下：

- `torch>=2.4.0,<2.7`
- `vllm>=0.8.0`
- `transformers>=4.46.0,<5.0.0`
- `datasets>=3.0.0`
- `tokenizers>=0.20.0`
- `openai>=1.50.0`
- `httpx>=0.27.0`
- `sympy>=1.13`
- `latex2sympy2>=1.9.1`
- `word2number>=1.1`
- `regex>=2024.7.24`
- `wandb>=0.18.0`
- `numpy>=1.26`
- `tqdm>=4.66`

配套文件位置：

- Conda 环境文件：`trim/TRIM/environment.yml`
- Conda 锁定文件：`trim/TRIM/environment.lock.yml`
- pip 依赖：`trim/TRIM/requirements.txt`
- pip 锁定文件：`trim/TRIM/requirements.lock.txt`

系统侧建议与当前实验保持一致：

- CUDA 12.x 驱动环境
- NVIDIA GPU，推荐至少 2 张卡；TRIM-Agg 双 island 并行建议 4 张卡
- 已下载本地模型权重：
  - `qwen3-14b`
  - `qwen3-1.7b`
  - `qwen2.5-math-prm-7b`

### 运行全部流程
```bash
# 1. 启动 vLLM 服务
bash src/scripts/00_start_vllm.sh

# 2. 一键运行所有步骤
bash src/scripts/run_all.sh

# 或者分步运行:
bash src/scripts/01_eval_baselines.sh      # Baseline 评估
bash src/scripts/02_generate_episodes.sh   # 生成 episode 数据
bash src/scripts/03_train_router.sh        # 训练路由器
bash src/scripts/04_eval_and_plot.sh       # 评估 + 图表
```

### 只查看数字结果
```bash
cd src && python -m eval.print_results
```

### 只重新生成图表
```bash
cd src && python -m eval.plot_clean
```

## 方法概述

| 方法 | 说明 |
|------|------|
| SRM-Only | 所有步骤用 Qwen3-1.7B |
| LRM-Only | 所有步骤用 Qwen3-14B |
| Random Routing | 随机决定用 SRM 还是 LRM |
| TRIM-Agg | PPO 训练的路由器，仅 outcome reward |
| TRIM-Rubric | PPO 路由器 + rubric process reward |
