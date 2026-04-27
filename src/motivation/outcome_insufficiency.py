"""Sec 2.1 — Outcome-only reward is insufficient for evaluating routing
trajectory process quality.

Controlled analysis:
1. Filter problems where SRM fails and LRM succeeds.
2. For each such problem, construct diverse routing trajectories with similar
   outcomes and costs via different routing strategies.
3. Score trajectories with three process quality rubrics:
   - Timely intervention (did routing switch at the critical error?)
   - Switch smoothness (no frequent SRM↔LRM oscillation)
   - Path simplicity (no excessive backtracking / repetition)
4. Show that outcome-only reward cannot distinguish quality differences
   captured by these process criteria.

Usage:
    python -m motivation.outcome_insufficiency --dataset math500 --limit 30
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import RESULTS_DIR, MAX_STEPS
from common.datasets import load_math500, load_aime
from common.prm import score_steps
from router.core import online_route

BACKTRACK_MARKERS = [
    "wait", "actually", "no,", "let me reconsider", "I made a mistake",
    "that's wrong", "correction:", "hmm, that doesn't",
    "let me redo", "I think I was wrong",
]


# ==================== Rubric scoring ====================

def score_timely_intervention(steps: List[Dict], prm_scores: List[float]) -> float:
    """Higher score if LRM was used right when PRM score first dipped below 0.5."""
    if not prm_scores:
        return 0.5
    first_bad = None
    for i, s in enumerate(prm_scores):
        if s < 0.5:
            first_bad = i
            break
    if first_bad is None:
        return 1.0
    if first_bad < len(steps) and steps[first_bad]["model"] == "LRM":
        return 1.0
    window = 2
    for j in range(max(0, first_bad - 1), min(len(steps), first_bad + window + 1)):
        if j < len(steps) and steps[j]["model"] == "LRM":
            return 0.7
    return 0.0


def score_switch_smoothness(steps: List[Dict]) -> float:
    """Fewer model switches → higher score. Normalized to [0,1]."""
    if len(steps) <= 1:
        return 1.0
    switches = sum(1 for i in range(1, len(steps)) if steps[i]["model"] != steps[i-1]["model"])
    max_switches = len(steps) - 1
    return 1.0 - (switches / max_switches)


def score_path_simplicity(steps: List[Dict]) -> float:
    """Fewer backtracking markers → simpler path. Normalized to [0,1]."""
    if not steps:
        return 1.0
    bt_count = sum(1 for s in steps
                   if any(m in s["text"].lower() for m in BACKTRACK_MARKERS))
    return 1.0 - (bt_count / len(steps))


def compute_process_score(steps, prm_scores):
    ti = score_timely_intervention(steps, prm_scores)
    ss = score_switch_smoothness(steps)
    ps = score_path_simplicity(steps)
    return {"timely_intervention": ti, "switch_smoothness": ss,
            "path_simplicity": ps, "mean": (ti + ss + ps) / 3}


# ==================== Trajectory generation ====================

def _always_srm(step_idx, srm_step, history):
    return False

def _always_lrm(step_idx, srm_step, history):
    return True

def _random_50(step_idx, srm_step, history):
    return random.random() < 0.5

def _random_30(step_idx, srm_step, history):
    return random.random() < 0.3

def _random_70(step_idx, srm_step, history):
    return random.random() < 0.7


STRATEGIES = [
    ("random_30", _random_30),
    ("random_50", _random_50),
    ("random_70", _random_70),
]


def analyze_problem(problem_text, answer, problem_id):
    """Generate multiple trajectories for one problem and compare."""
    trajectories = []
    for name, router_fn in STRATEGIES:
        try:
            result = online_route(
                problem=problem_text,
                ground_truth=answer,
                router_fn=router_fn,
                router_name=name,
            )
            # PRM score the trajectory steps
            step_texts = [s["text"] for s in result["steps"]]
            prm_scores = score_steps(problem_text, step_texts) if step_texts else []
            proc = compute_process_score(result["steps"], prm_scores)
            result["prm_scores"] = prm_scores
            result["process_quality"] = proc
            trajectories.append(result)
        except Exception as e:
            print(f"    Strategy {name} failed: {e}")

    return trajectories


def run_analysis(dataset_name, limit=None, output_dir=None):
    if dataset_name == "math500":
        items = load_math500()
    else:
        items = load_aime(2020, 2024)

    if limit:
        items = items[:limit]

    tag = f"outcome_insufficiency_{dataset_name}"
    out_dir = os.path.join(output_dir or os.path.join(RESULTS_DIR, "motivation"), tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Motivation Sec 2.1: Outcome Insufficiency")
    print(f"  Dataset: {dataset_name} ({len(items)} problems)")
    print(f"{'='*60}\n")

    all_results = []
    pairs_outcome_same = 0
    pairs_process_diff = 0
    total_pairs = 0

    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}] Problem {item['id']}...")
        trajs = analyze_problem(item["problem"], item["answer"], item["id"])

        correct_trajs = [t for t in trajs if t.get("is_correct")]
        if len(correct_trajs) < 2:
            print(f"    Only {len(correct_trajs)} correct trajectories, skipping pair analysis")
            all_results.append({"id": item["id"], "trajectories": len(trajs),
                                "correct_trajs": len(correct_trajs)})
            continue

        # Compare pairs of trajectories with same outcome
        for i in range(len(correct_trajs)):
            for j in range(i+1, len(correct_trajs)):
                t1, t2 = correct_trajs[i], correct_trajs[j]
                total_pairs += 1
                # Same outcome (both correct)
                pairs_outcome_same += 1
                # Process quality difference
                pq1 = t1["process_quality"]["mean"]
                pq2 = t2["process_quality"]["mean"]
                if abs(pq1 - pq2) > 0.05:
                    pairs_process_diff += 1

        all_results.append({
            "id": item["id"],
            "trajectories": len(trajs),
            "correct_trajs": len(correct_trajs),
            "process_scores": [t["process_quality"] for t in correct_trajs],
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  Analysis Complete")
    print(f"  Total pairs (same outcome): {pairs_outcome_same}")
    print(f"  Pairs with process quality diff > 0.05: {pairs_process_diff}")
    if pairs_outcome_same > 0:
        pct = pairs_process_diff / pairs_outcome_same * 100
        print(f"  → {pct:.1f}% of same-outcome pairs show process quality differences")
        print(f"  → Outcome-only reward fails to distinguish them")
    print(f"{'='*60}\n")

    # Save
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "total_problems": len(items),
            "total_pairs": total_pairs,
            "pairs_outcome_same": pairs_outcome_same,
            "pairs_process_diff": pairs_process_diff,
            "pct_undistinguished": round(pairs_process_diff / max(1, pairs_outcome_same) * 100, 2),
            "details": all_results,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["math500", "aime"], default="math500")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_analysis(args.dataset, args.limit)
