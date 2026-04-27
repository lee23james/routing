#!/bin/bash
# ================================================================
# 结果验证脚本
# ================================================================
# 验证内容:
#   1. Table 1 结果: 检查文件存在性、数据完整性、SOTA 验证
#   2. Budgeted Accuracy: 检查单调性、SOTA 验证
#   3. 汇总所有数据集的关键指标
#
# 用法:
#   bash scripts/verify_results.sh
# ================================================================

set -e
cd "$(dirname "$0")/.."

echo "╔══════════════════════════════════════════════════════════╗"
echo "║              实验结果验证                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0
WARN=0

check_pass() { echo "  ✓ $1"; PASS=$((PASS+1)); }
check_fail() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
check_warn() { echo "  ⚠ $1"; WARN=$((WARN+1)); }

# ============================================================
# 1. Table 1 验证
# ============================================================
echo "━━━ [1/3] Table 1 结果验证 ━━━"

python3 -c "
import json, glob, sys

files = sorted(glob.glob('results/table1/table1_*.json'))
if not files:
    print('  ✗ 无 Table 1 结果文件')
    sys.exit(1)

all_ok = True
for f in files:
    data = json.load(open(f))
    ds = data['dataset']
    n = data['n_episodes']
    srm = data['srm_accuracy']
    lrm = data['lrm_accuracy']
    print(f'  {ds} ({n} episodes, SRM={srm:.4f}, LRM={lrm:.4f}):')

    for cpt_label, methods in data['methods'].items():
        # 检查每个方法的数据完整性
        for m_name in ['random', 'trim_thr', 'trim_agg', 'trim_rubric']:
            m = methods[m_name]
            if 'accuracy' not in m or 'cpt' not in m:
                print(f'    ✗ {cpt_label}/{m_name}: 缺少 accuracy 或 cpt')
                all_ok = False
                continue

        # 检查 TRIM-Rubric >= TRIM-Agg
        agg_acc = methods['trim_agg'].get('accuracy', 0)
        rub_acc = methods['trim_rubric'].get('accuracy', 0)
        if rub_acc >= agg_acc - 0.001:
            print(f'    ✓ {cpt_label}: Rubric({rub_acc:.4f}) >= Agg({agg_acc:.4f})')
        else:
            print(f'    ⚠ {cpt_label}: Rubric({rub_acc:.4f}) < Agg({agg_acc:.4f}), 差距={agg_acc-rub_acc:.4f}')

        # 检查 PGR 合理性
        for m_name in ['trim_agg', 'trim_rubric']:
            pgr = methods[m_name].get('pgr', 0)
            if pgr < -0.5 or pgr > 2.0:
                print(f'    ⚠ {cpt_label}/{m_name}: PGR={pgr:.4f} 超出预期范围 [-0.5, 2.0]')

if all_ok:
    print('  ✓ Table 1 数据完整性通过')
" 2>/dev/null && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""

# ============================================================
# 2. Budgeted Accuracy 验证
# ============================================================
echo "━━━ [2/3] Budgeted Accuracy 验证 ━━━"

python3 -c "
import json, glob, sys

files = sorted(glob.glob('results/budgeted_accuracy/budgeted_accuracy_*.json'))
if not files:
    print('  ✗ 无 Budgeted Accuracy 结果文件')
    sys.exit(1)

for f in files:
    data = json.load(open(f))
    ds = data['dataset']
    print(f'  {ds}:')

    # 检查单调性 (预算增加 → 准确率不降低)
    for method in ['random', 'trim_thr', 'trim_agg', 'trim_rubric']:
        prev_acc = -1
        monotonic = True
        for bk in sorted(data['budgets'].keys()):
            acc = data['budgets'][bk][method]['accuracy']
            if acc < prev_acc - 0.02:
                monotonic = False
            prev_acc = acc
        status = '✓' if monotonic else '⚠ 非单调'
        print(f'    {status} {method}: 单调性检查')

    # 检查 TRIM-Rubric 是否在多数预算下 >= TRIM-Agg
    rub_wins = 0
    total = 0
    for bk in data['budgets']:
        agg = data['budgets'][bk]['trim_agg']['accuracy']
        rub = data['budgets'][bk]['trim_rubric']['accuracy']
        total += 1
        if rub >= agg - 0.001:
            rub_wins += 1
    pct = rub_wins / total * 100 if total > 0 else 0
    if pct >= 60:
        print(f'    ✓ Rubric >= Agg: {rub_wins}/{total} ({pct:.0f}%)')
    else:
        print(f'    ⚠ Rubric >= Agg: {rub_wins}/{total} ({pct:.0f}%) — 不足 60%')
" 2>/dev/null && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""

# ============================================================
# 3. 综合汇总
# ============================================================
echo "━━━ [3/3] 综合汇总 ━━━"

python3 -c "
import json, glob

# Table 1
for f in sorted(glob.glob('results/table1/table1_*.json')):
    data = json.load(open(f))
    ds = data['dataset']
    print(f'  Table 1 — {ds}:')
    print(f'    {\"CPT\":<8} {\"Random\":>8} {\"Thr\":>8} {\"Agg\":>8} {\"Rubric\":>8} {\"PGR(R)\":>8}')
    print(f'    {\"-\"*50}')
    for cpt_l, ms in data['methods'].items():
        r = ms['random']['accuracy']
        t = ms['trim_thr']['accuracy']
        a = ms['trim_agg']['accuracy']
        rb = ms['trim_rubric']['accuracy']
        pgr = ms['trim_rubric'].get('pgr', 0)
        print(f'    {cpt_l:<8} {r:>8.4f} {t:>8.4f} {a:>8.4f} {rb:>8.4f} {pgr:>8.4f}')
    print()

# Budgeted
for f in sorted(glob.glob('results/budgeted_accuracy/budgeted_accuracy_*.json')):
    data = json.load(open(f))
    ds = data['dataset']
    print(f'  Budgeted — {ds}:')
    print(f'    {\"Budget\":<8} {\"Random\":>8} {\"Thr\":>8} {\"Agg\":>8} {\"Rubric\":>8}')
    print(f'    {\"-\"*45}')
    for bk in sorted(data['budgets'].keys()):
        bd = data['budgets'][bk]
        r = bd['random']['accuracy']
        t = bd['trim_thr']['accuracy']
        a = bd['trim_agg']['accuracy']
        rb = bd['trim_rubric']['accuracy']
        print(f'    {bk:<8} {r:>8.4f} {t:>8.4f} {a:>8.4f} {rb:>8.4f}')
    print()
" 2>/dev/null || echo "  (无结果文件)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证完成: PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
