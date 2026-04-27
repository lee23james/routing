#!/bin/bash
# Step 5: Motivation Analysis
# Sec 2.1: Outcome-only reward insufficiency
# Sec 2.2: Rubric superiority ablation

set -e
cd /export/shy/pp/pp5/src

echo "=============================="
echo " Step 5: Motivation Analysis"
echo "=============================="

# --- Sec 2.1: Outcome insufficiency ---
echo "[MATH-500] Outcome insufficiency analysis ..."
python -m motivation.outcome_insufficiency --dataset math500 --limit 50

echo "[AIME] Outcome insufficiency analysis ..."
python -m motivation.outcome_insufficiency --dataset aime --limit 30

# --- Sec 2.2: Rubric superiority ablation ---
echo "[MATH-500] Rubric ablation ..."
python -m motivation.rubric_superiority --dataset math500 --limit 50

echo "[AIME] Rubric ablation ..."
python -m motivation.rubric_superiority --dataset aime --limit 30

echo "Motivation analysis done."
