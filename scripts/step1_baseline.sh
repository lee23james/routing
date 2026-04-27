#!/bin/bash
# Step 1: Baseline — SRM-only / LRM-only 评测
# 数据集: MATH-500, AIME 2020-2024
# 模型: qwen3-1.7b (SRM, port 4003), qwen3-14b (LRM, port 4001)

set -e
cd /export/shy/pp/pp5/src

echo "=============================="
echo " Step 1: Baseline Evaluation"
echo "=============================="

# 1) SRM on MATH-500
echo "[1/4] SRM on MATH-500 ..."
python -m baseline.run_baseline --model srm --dataset math500 &
PID_1=$!

# 2) SRM on AIME 2020-2024
echo "[2/4] SRM on AIME 2020-2024 ..."
python -m baseline.run_baseline --model srm --dataset aime &
PID_2=$!

# 3) LRM on MATH-500
echo "[3/4] LRM on MATH-500 ..."
python -m baseline.run_baseline --model lrm --dataset math500 &
PID_3=$!

# 4) LRM on AIME 2020-2024
echo "[4/4] LRM on AIME 2020-2024 ..."
python -m baseline.run_baseline --model lrm --dataset aime &
PID_4=$!

echo "All 4 baseline jobs started. PIDs: $PID_1 $PID_2 $PID_3 $PID_4"
echo "Waiting for completion..."
wait $PID_1 $PID_2 $PID_3 $PID_4
echo "All baseline evaluations done."
