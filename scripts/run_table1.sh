#!/usr/bin/env bash
# ============================================================
# Table 1 复现: MATH-500 + AIME 上四种路由策略的评估
#
# 评估指标: Accuracy @ CPT50/CPT80/CPT95, IBC, PGR
# 方法:     Random, TRIM-Thr, TRIM-Agg, TRIM-Rubric
#
# 用法:
#   bash scripts/run_table1.sh              # 默认全部
#   bash scripts/run_table1.sh math500      # 只跑 MATH-500
#   bash scripts/run_table1.sh aime         # 只跑 AIME
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRIM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SRC_DIR="${PROJECT_ROOT}/src"
DATA_DIR="${PROJECT_ROOT}/data/episodes"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"
OUT_DIR="${PROJECT_ROOT}/results/table1"
DEVICE="cpu"

DATASET="${1:-all}"

mkdir -p "${OUT_DIR}"

run_eval() {
    local name="$1"
    local episodes="$2"
    echo ""
    echo "========================================"
    echo "  Table 1 评估: ${name}"
    echo "========================================"
    cd "${SRC_DIR}"
    python3 -m eval.table1_eval \
        --episodes_path "${episodes}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --target_cpts "0.50,0.80,0.95" \
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
echo "  Table 1 完成! 结果保存在: ${OUT_DIR}"
echo "========================================"
