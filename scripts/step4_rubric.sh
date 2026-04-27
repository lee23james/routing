#!/bin/bash
# Step 4: Rubric 路由评测
# - Rubric heuristic (无训练的规则路由)
# - TRIM-Rubric (PPO with outcome + rubric reward)

set -e
cd /export/shy/pp/pp5/src

echo "=============================="
echo " Step 4: Rubric Routing"
echo "=============================="

# --- Rubric Heuristic ---
echo "[MATH-500] Rubric heuristic ..."
python -m rubric.rubric_router --dataset math500 --mode heuristic

echo "[AIME] Rubric heuristic ..."
python -m rubric.rubric_router --dataset aime --mode heuristic

# --- TRIM-Rubric (PPO policy) ---
for ckpt in trim_rubric/best.pt trim_rubric_lam3e-04/best.pt trim_rubric_lam5e-04/best.pt; do
    if [ -f "/export/shy/pp/pp5/checkpoints/$ckpt" ]; then
        echo "[MATH-500] TRIM-Rubric $ckpt ..."
        python -m rubric.rubric_router --dataset math500 --mode policy --checkpoint "$ckpt"

        echo "[AIME] TRIM-Rubric $ckpt ..."
        python -m rubric.rubric_router --dataset aime --mode policy --checkpoint "$ckpt"
    else
        echo "Skipping $ckpt (not found)"
    fi
done

echo "Rubric routing done."
