#!/bin/bash
# 一键运行所有实验
# 注意：Step 1 (baseline) 在后台并行运行，其他顺序执行
# 需要确保两组 vLLM 服务已启动:
#   - Baseline: SRM port 4003, LRM port 4001
#   - Routing:  SRM port 4013, LRM port 4011

set -e

echo "============================================"
echo "  TRIM 全流程实验"
echo "  Baseline ports: 4003 (SRM), 4001 (LRM)"
echo "  Routing  ports: 4013 (SRM), 4011 (LRM)"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo ">>> Step 1: Baseline (background) ..."
bash "$SCRIPT_DIR/step1_baseline.sh" &
BASELINE_PID=$!

echo ""
echo ">>> Step 2: Random Routing ..."
bash "$SCRIPT_DIR/step2_random_routing.sh"

echo ""
echo ">>> Step 3: TRIM Routing ..."
bash "$SCRIPT_DIR/step3_trim.sh"

echo ""
echo ">>> Step 4: Rubric Routing ..."
bash "$SCRIPT_DIR/step4_rubric.sh"

echo ""
echo ">>> Step 5: Motivation Analysis ..."
bash "$SCRIPT_DIR/step5_motivation.sh"

echo ""
echo "Waiting for baseline to finish..."
wait $BASELINE_PID

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results in: /export/shy/pp/pp5/results/"
echo "============================================"
