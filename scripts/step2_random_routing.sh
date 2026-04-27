#!/bin/bash
# Step 2: Random Routing 评测
# 测试不同 p(LRM) 比例的 random routing

set -e
cd /export/shy/pp/pp5/src

echo "=============================="
echo " Step 2: Random Routing"
echo "=============================="

for p in 0.10 0.20 0.30 0.50; do
    echo "[MATH-500] Random p=$p ..."
    python -m router.random_router --dataset math500 --regen_ratio $p

    echo "[AIME] Random p=$p ..."
    python -m router.random_router --dataset aime --regen_ratio $p
done

echo "Random routing done."
