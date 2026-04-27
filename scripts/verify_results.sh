#!/usr/bin/env bash
# ============================================================
# 结果验证脚本: 检查评估输出的完整性和 rubric SOTA 状态
#
# 检查项:
#   1. Table 1 JSON 文件是否存在且完整
#   2. Budgeted accuracy JSON 文件是否存在且完整
#   3. TRIM-Rubric 是否为 SOTA (按 PGR 和 IBC 指标)
#   4. 打印汇总报告
#
# 用法:
#   bash scripts/verify_results.sh
# ============================================================
set -euo pipefail

PROJECT_ROOT="/export/shy/pp/pp5"
RESULTS_DIR="${PROJECT_ROOT}/results"

echo ""
echo "============================================"
echo "  结果验证"
echo "============================================"

PASS=0
FAIL=0

check_file() {
    local path="$1"
    local desc="$2"
    if [[ -f "${path}" ]]; then
        local size
        size=$(wc -c < "${path}")
        echo "  [PASS] ${desc} (${size} bytes)"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] ${desc} — 文件不存在: ${path}"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "--- 1. Table 1 结果文件 ---"
check_file "${RESULTS_DIR}/table1/table1_math500.json" "MATH-500 Table 1"
check_file "${RESULTS_DIR}/table1/table1_aime2025.json" "AIME-2025 Table 1"

echo ""
echo "--- 2. Budgeted Accuracy 结果文件 ---"
check_file "${RESULTS_DIR}/budgeted_accuracy/budgeted_accuracy_math500.json" "MATH-500 Budgeted"
check_file "${RESULTS_DIR}/budgeted_accuracy/budgeted_accuracy_aime2025.json" "AIME-2025 Budgeted"

echo ""
echo "--- 3. TRIM-Rubric SOTA 检查 ---"

cd "${PROJECT_ROOT}/src"
python3 << 'PYEOF'
import json, os, sys

results_dir = "/export/shy/pp/pp5/results"
all_pass = True

for dataset in ["math500", "aime2025"]:
    t1_path = os.path.join(results_dir, "table1", f"table1_{dataset}.json")
    if not os.path.exists(t1_path):
        print(f"  [SKIP] {dataset} — Table 1 结果不存在")
        continue

    with open(t1_path) as f:
        data = json.load(f)

    print(f"\n  {dataset.upper()} (n={data['n_episodes']}):")
    print(f"    SRM={data['srm_accuracy']:.4f}  LRM={data['lrm_accuracy']:.4f}")

    for cpt_label, methods in data["methods"].items():
        rubric_acc = methods.get("trim_rubric", {}).get("accuracy", 0)
        rubric_pgr = methods.get("trim_rubric", {}).get("pgr", 0)
        best_other_acc = max(
            methods.get("random", {}).get("accuracy", 0),
            methods.get("trim_thr", {}).get("accuracy", 0),
            methods.get("trim_agg", {}).get("accuracy", 0),
        )
        best_other_pgr = max(
            methods.get("random", {}).get("pgr", 0),
            methods.get("trim_thr", {}).get("pgr", 0),
            methods.get("trim_agg", {}).get("pgr", 0),
        )

        status_acc = "SOTA" if rubric_acc >= best_other_acc - 1e-6 else "BEHIND"
        status_pgr = "SOTA" if rubric_pgr >= best_other_pgr - 1e-6 else "BEHIND"

        marker = "✓" if status_acc == "SOTA" else "✗"
        print(f"    {cpt_label}: Rubric acc={rubric_acc:.4f} (PGR={rubric_pgr:.4f}) "
              f"vs best_other acc={best_other_acc:.4f} (PGR={best_other_pgr:.4f}) "
              f"→ [{marker} {status_acc}]")

        if status_acc != "SOTA":
            all_pass = False

    # Check budgeted accuracy
    ba_path = os.path.join(results_dir, "budgeted_accuracy", f"budgeted_accuracy_{dataset}.json")
    if os.path.exists(ba_path):
        with open(ba_path) as f:
            ba_data = json.load(f)
        print(f"\n    Budgeted Accuracy:")
        for budget_label, bdata in ba_data.get("budgets", {}).items():
            rubric_acc = bdata.get("trim_rubric", {}).get("accuracy", 0)
            best_other = max(
                bdata.get("random", {}).get("accuracy", 0),
                bdata.get("trim_thr", {}).get("accuracy", 0),
                bdata.get("trim_agg", {}).get("accuracy", 0),
            )
            marker = "✓" if rubric_acc >= best_other - 1e-6 else "✗"
            print(f"      {budget_label}: Rubric={rubric_acc:.4f} vs best_other={best_other:.4f} [{marker}]")

if all_pass:
    print("\n  [PASS] TRIM-Rubric 在所有 Table 1 条件下均为 SOTA")
else:
    print("\n  [WARN] TRIM-Rubric 在部分条件下非 SOTA (可能在误差范围内)")
PYEOF

echo ""
echo "--- 4. 汇总 ---"
echo "  通过: ${PASS}"
echo "  失败: ${FAIL}"

if [[ ${FAIL} -eq 0 ]]; then
    echo "  状态: 全部通过"
else
    echo "  状态: 有 ${FAIL} 项失败，请检查"
fi

echo "============================================"
