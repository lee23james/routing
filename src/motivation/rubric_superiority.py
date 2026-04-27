"""Sec 2.2 — Rubric-based reward enhances routing process quality.

Ablation study comparing four routing strategies online:
1. TRIM-Thr (threshold-based routing)
2. Outcome-only (TRIM-Agg: PPO trained with outcome reward only)
3. Rubric-only (heuristic rubric routing, no outcome signal)
4. Rubric+Outcome (TRIM-Rubric: PPO trained with outcome + rubric reward)

Evaluation: accuracy, CPT, and process quality metrics.

Usage:
    python -m motivation.rubric_superiority --dataset math500 --limit 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import RESULTS_DIR, CHECKPOINTS_DIR
from common.datasets import load_math500, load_aime
from common.prm import score_steps
from router.core import online_route
from trim_agg.trim_thr import make_thr_router
from trim_agg.trim_agg import load_policy, make_agg_router
from rubric.rubric_router import make_rubric_heuristic_router, make_rubric_policy_router
from motivation.outcome_insufficiency import compute_process_score


def _find_best_checkpoint(prefix: str) -> str:
    """Find best.pt under checkpoints/{prefix}/."""
    path = os.path.join(CHECKPOINTS_DIR, prefix, "best.pt")
    if os.path.exists(path):
        return path
    path = os.path.join(CHECKPOINTS_DIR, prefix, "final.pt")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No checkpoint found for {prefix}")


def run_ablation(dataset_name, limit=None, output_dir=None,
                 agg_ckpt=None, rubric_ckpt=None, threshold=0.5):
    if dataset_name == "math500":
        items = load_math500()
    else:
        items = load_aime(2020, 2024)

    if limit:
        items = items[:limit]

    # Load policies
    if agg_ckpt is None:
        agg_ckpt = _find_best_checkpoint("trim_agg")
    if rubric_ckpt is None:
        rubric_ckpt = _find_best_checkpoint("trim_rubric")

    agg_policy = load_policy(agg_ckpt)
    agg_sdim = next(p for p in agg_policy.encoder.parameters()).shape[1]
    rubric_policy = load_policy(rubric_ckpt)
    rubric_sdim = next(p for p in rubric_policy.encoder.parameters()).shape[1]

    tag = f"rubric_ablation_{dataset_name}"
    out_dir = os.path.join(output_dir or os.path.join(RESULTS_DIR, "motivation"), tag)
    os.makedirs(out_dir, exist_ok=True)

    strategies = {
        "trim_thr": lambda prob: make_thr_router(threshold, prob),
        "trim_agg": lambda prob: make_agg_router(agg_policy, prob, agg_sdim),
        "rubric_heuristic": lambda prob: make_rubric_heuristic_router(prob),
        "trim_rubric": lambda prob: make_rubric_policy_router(rubric_policy, prob, rubric_sdim),
    }

    print(f"\n{'='*60}")
    print(f"  Motivation Sec 2.2: Rubric Superiority Ablation")
    print(f"  Dataset: {dataset_name} ({len(items)} problems)")
    print(f"  Agg ckpt: {agg_ckpt}")
    print(f"  Rubric ckpt: {rubric_ckpt}")
    print(f"  Strategies: {list(strategies.keys())}")
    print(f"{'='*60}\n")

    all_results = {s: [] for s in strategies}

    for idx, item in enumerate(items):
        print(f"\n[{idx+1}/{len(items)}] Problem {item['id']}")
        for sname, make_fn in strategies.items():
            try:
                rfn = make_fn(item["problem"])
                result = online_route(
                    problem=item["problem"],
                    ground_truth=item["answer"],
                    router_fn=rfn,
                    router_name=sname,
                )
                result["id"] = item["id"]
                # Process quality
                step_texts = [s["text"] for s in result["steps"]]
                prm_scores = score_steps(item["problem"], step_texts) if step_texts else []
                proc = compute_process_score(result["steps"], prm_scores)
                result["process_quality"] = proc

                all_results[sname].append(result)
                mark = "✓" if result["is_correct"] else "✗"
                print(f"  {sname:20s} {mark} acc={result['is_correct']} "
                      f"cpt={result['cpt']:.1f}% rr={result['regen_ratio']:.2f} "
                      f"pq={proc['mean']:.3f}")
            except Exception as e:
                print(f"  {sname:20s} ERROR: {e}")
                all_results[sname].append({"id": item["id"], "error": str(e), "is_correct": False})

        # Periodic save
        if (idx + 1) % 5 == 0:
            _save_all(all_results, out_dir)

    _save_all(all_results, out_dir)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Strategy':20s} {'Acc%':>8s} {'AvgCPT':>8s} {'AvgRR':>8s} {'AvgPQ':>8s}")
    print(f"{'-'*70}")
    for sname, rl in all_results.items():
        valid = [r for r in rl if "error" not in r]
        n = len(valid)
        if n == 0:
            print(f"{sname:20s} {'N/A':>8s}")
            continue
        acc = sum(1 for r in valid if r["is_correct"]) / n * 100
        cpt = sum(r.get("cpt", 0) for r in valid) / n
        rr = sum(r.get("regen_ratio", 0) for r in valid) / n
        pq = sum(r.get("process_quality", {}).get("mean", 0) for r in valid) / n
        print(f"{sname:20s} {acc:>7.2f}% {cpt:>7.1f}% {rr:>7.3f} {pq:>7.3f}")
    print(f"{'='*70}\n")


def _save_all(all_results, out_dir):
    for sname, rl in all_results.items():
        path = os.path.join(out_dir, f"{sname}.jsonl")
        with open(path, "w") as f:
            for r in rl:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["math500", "aime"], default="math500")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agg_ckpt", type=str, default=None)
    parser.add_argument("--rubric_ckpt", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_ablation(args.dataset, args.limit, agg_ckpt=args.agg_ckpt, rubric_ckpt=args.rubric_ckpt)
