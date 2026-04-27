#!/usr/bin/env bash
# ============================================================
# 运行全部实验: Table 1 + Budgeted Accuracy
#
# 用法:
#   bash scripts/run_all_experiments.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "============================================"
echo "  TRIM 实验复现 — 全部评估"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "============================================"

echo ""
echo "===== [1/3] Table 1 评估 ====="
bash "${SCRIPT_DIR}/run_table1.sh" all

echo ""
echo "===== [2/3] Budgeted Accuracy 评估 ====="
bash "${SCRIPT_DIR}/run_budgeted_accuracy.sh" all

echo ""
echo "===== [3/3] 结果验证 ====="
bash "${SCRIPT_DIR}/verify_results.sh"

echo ""
echo "============================================"
echo "  全部实验完成!"
echo "  结果目录: ${PROJECT_ROOT}/results/"
echo "============================================"
