#!/bin/bash
# Step 3: TRIM-Thr 和 TRIM-Agg 评测
# TRIM-Thr: 阈值路由 (PRM score < threshold → use LRM)
# TRIM-Agg: PPO 策略路由 (outcome-only reward)

set -e
cd /export/shy/pp/pp5/src

echo "=============================="
echo " Step 3: TRIM Routing"
echo "=============================="

# --- TRIM-Thr ---
for thr in 0.3 0.5 0.7; do
    echo "[MATH-500] TRIM-Thr threshold=$thr ..."
    python -m trim_agg.trim_thr --dataset math500 --threshold $thr

    echo "[AIME] TRIM-Thr threshold=$thr ..."
    python -m trim_agg.trim_thr --dataset aime --threshold $thr
done

# --- TRIM-Agg (outcome-only PPO policies) ---
for ckpt in trim_agg/best.pt trim_agg_lam3e-04/best.pt trim_agg_lam5e-04/best.pt; do
    if [ -f "/export/shy/pp/pp5/checkpoints/$ckpt" ]; then
        echo "[MATH-500] TRIM-Agg $ckpt ..."
        python -m trim_agg.trim_agg --dataset math500 --checkpoint "$ckpt"

        echo "[AIME] TRIM-Agg $ckpt ..."
        python -m trim_agg.trim_agg --dataset aime --checkpoint "$ckpt"
    else
        echo "Skipping $ckpt (not found)"
    fi
done

echo "TRIM routing done."
