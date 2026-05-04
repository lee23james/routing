#!/usr/bin/env bash
# ============================================================
# Motivation 分析: Sec 2.1 + Sec 2.2
#
# Sec 2.1: Outcome-only reward 不足 — 受控轨迹对 + LLM judge
# Sec 2.2: Rubric reward 提升 — 对齐度 + policy 性能对比
#
# 用法:
#   bash scripts/run_motivation.sh             # 全部 (mock LLM)
#   bash scripts/run_motivation.sh sec2_1      # 只跑 Sec 2.1
#   bash scripts/run_motivation.sh sec2_2      # 只跑 Sec 2.2
#   LLM_MODE=llm bash scripts/run_motivation.sh  # 用真实 LLM
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRIM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SRC_DIR="${PROJECT_ROOT}/src"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"
OUT_BASE="${PROJECT_ROOT}/results/motivation"
LLM_MODE="${LLM_MODE:-mock}"
DATASET="${DATASET:-math500}"

SECTION="${1:-all}"

case "${DATASET}" in
    math500) EPISODES="${PROJECT_ROOT}/data/episodes/math500_episodes.jsonl" ;;
    aime)    EPISODES="${PROJECT_ROOT}/data/episodes/aime2025_episodes.jsonl" ;;
    *)       echo "Unknown DATASET=${DATASET}"; exit 1 ;;
esac

mkdir -p "${OUT_BASE}"

if [[ "${SECTION}" == "all" || "${SECTION}" == "sec2_1" ]]; then
    echo "===== Sec 2.1: Outcome-only reward 不足 (${DATASET}) ====="
    cd "${SRC_DIR}"
    python3 -m motivation.outcome_insufficiency \
        --episodes_path "${EPISODES}" \
        --output_dir "${OUT_BASE}/sec2_1_${DATASET}" \
        --llm_mode "${LLM_MODE}"
fi

if [[ "${SECTION}" == "all" || "${SECTION}" == "sec2_2" ]]; then
    echo ""
    echo "===== Sec 2.2: Rubric 有效性消融 (${DATASET}) ====="
    cd "${SRC_DIR}"
    python3 -m motivation.rubric_superiority \
        --episodes_path "${EPISODES}" \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${OUT_BASE}/sec2_2_${DATASET}" \
        --device cpu
fi

echo ""
echo "===== Motivation 分析完成! ====="
echo "结果目录: ${OUT_BASE}"
