#!/usr/bin/env bash
# ============================================================
# Budgeted Accuracy 评估: LRM 预算约束下各策略的准确率
#
# 预算:  LRM-only 的 10% / 15% / 20% / 25% / 30%
# 方法:  Random, TRIM-Thr, TRIM-Agg, TRIM-Rubric
#
# 用法:
#   bash scripts/run_budgeted_accuracy.sh              # 默认全部
#   bash scripts/run_budgeted_accuracy.sh math500      # 只跑 MATH-500
#   bash scripts/run_budgeted_accuracy.sh aime         # 只跑 AIME
# ============================================================
set -euo pipefail

PROJECT_ROOT="/export/shy/pp/pp5"
SRC_DIR="${PROJECT_ROOT}/src"
DATA_DIR="${PROJECT_ROOT}/data/episodes"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"
OUT_DIR="${PROJECT_ROOT}/results/budgeted_accuracy"
DEVICE="cpu"

DATASET="${1:-all}"

mkdir -p "${OUT_DIR}"

run_eval() {
    local name="$1"
    local episodes="$2"
    echo ""
    echo "========================================"
    echo "  Budgeted Accuracy 评估: ${name}"
    echo "========================================"
    cd "${SRC_DIR}"
    python3 -m eval.budgeted_accuracy \
        --episodes_path "${episodes}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --budget_ratios "0.10,0.15,0.20,0.25,0.30" \
        --device "${DEVICE}" \
        --output_dir "${OUT_DIR}" \
        --n_random_trials 30
}

if [[ "${DATASET}" == "all" || "${DATASET}" == "math500" ]]; then
    run_eval "MATH-500" "${DATA_DIR}/math500_episodes.jsonl"
fi

if [[ "${DATASET}" == "all" || "${DATASET}" == "aime" ]]; then
    run_eval "AIME-2025" "${DATA_DIR}/aime2025_episodes.jsonl"
fi

echo ""
echo "========================================"
echo "  Budgeted Accuracy 完成! 结果保存在: ${OUT_DIR}"
echo "========================================"
