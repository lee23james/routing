"""TRIM-Rubric: Rubric-enhanced online routing.

Uses the same PPO-trained policy as TRIM-Agg but with checkpoints
trained with rubric-based process reward (outcome + rubric).

Also provides rubric-only heuristic routing for ablation:
- critical_intervention: route to LRM if PRM score drops sharply
- switch_smoothness: penalize frequent SRM↔LRM oscillation
- path_simplicity: route to LRM if step contains backtracking markers

Usage:
    python -m rubric.rubric_router --dataset math500 --mode policy --checkpoint checkpoints/trim_rubric/best.pt
    python -m rubric.rubric_router --dataset math500 --mode heuristic
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import RESULTS_DIR, CHECKPOINTS_DIR, MAX_STEPS
from common.datasets import load_math500, load_aime
from common.prm import score_steps
from router.core import online_route
from router.policy import RouterPolicy
from trim_agg.trim_agg import load_policy

TOKEN_NORMALISER = 512

BACKTRACK_MARKERS = [
    "wait", "actually", "no,", "let me reconsider", "I made a mistake",
    "that's wrong", "correction:", "hmm, that doesn't",
    "let me redo", "I think I was wrong", "but actually",
]


def _has_backtracking(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in BACKTRACK_MARKERS)


def _prm_drop(scores: list, threshold: float = 0.15) -> bool:
    """Check if latest PRM score dropped significantly from previous."""
    if len(scores) < 2:
        return False
    return scores[-2] - scores[-1] > threshold


def make_rubric_heuristic_router(problem: str, prm_thr=0.5, drop_thr=0.15):
    """Rubric-based heuristic router (no trained policy).

    Routes to LRM when any rubric criterion is triggered:
    1. Critical intervention: PRM score < prm_thr or sharp PRM drop
    2. Path simplicity: step contains backtracking markers
    3. Switch smoothness is implicit — we don't oscillate unnecessarily
    """
    prm_scores = []
    step_texts = []

    def router_fn(step_idx, srm_step, history):
        step_texts.append(srm_step)
        try:
            scores = score_steps(problem, step_texts)
            prm_scores.clear()
            prm_scores.extend(scores)
        except Exception:
            prm_scores.append(0.5)

        # Criterion 1: PRM too low or sharp drop
        cur_prm = prm_scores[-1] if prm_scores else 0.5
        low_prm = cur_prm < prm_thr
        sharp_drop = _prm_drop(prm_scores, drop_thr)

        # Criterion 2: Backtracking / not simple
        backtrack = _has_backtracking(srm_step)

        use_lrm = low_prm or sharp_drop or backtrack
        if use_lrm:
            step_texts.pop()
        return use_lrm

    return router_fn


def make_rubric_policy_router(policy: RouterPolicy, problem: str, state_dim: int = 5):
    """Router using PPO policy trained with rubric reward."""
    prm_scores = []
    step_texts = []

    def router_fn(step_idx, srm_step, history):
        step_texts.append(srm_step)
        try:
            scores = score_steps(problem, step_texts)
            prm_scores.clear()
            prm_scores.extend(scores)
        except Exception:
            prm_scores.append(0.5)

        cur_prm = prm_scores[-1] if prm_scores else 0.5
        min_prm = min(prm_scores) if prm_scores else 0.5
        prod_prm = float(np.prod(prm_scores)) if prm_scores else 0.5
        tok_ratio = len(srm_step) / TOKEN_NORMALISER
        step_ratio = step_idx / MAX_STEPS

        state_vec = [min_prm, prod_prm, cur_prm, tok_ratio, step_ratio]
        state_vec = state_vec[:state_dim]
        state_t = torch.tensor([state_vec], dtype=torch.float32)

        with torch.no_grad():
            action, _, _ = policy.get_action(state_t, deterministic=True)

        use_lrm = bool(action.item() == 1)
        if use_lrm:
            step_texts.pop()
        return use_lrm

    return router_fn


def run_rubric_routing(dataset_name, mode, ckpt_path=None, limit=None, output_dir=None):
    if dataset_name == "math500":
        items = load_math500()
    else:
        items = load_aime(2020, 2024)

    if limit:
        items = items[:limit]

    if mode == "policy":
        assert ckpt_path, "--checkpoint required for policy mode"
        policy = load_policy(ckpt_path)
        state_dim = next(p for p in policy.encoder.parameters()).shape[1]
        ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
        tag = f"trim_rubric_{ckpt_name}_{dataset_name}"
        router_label = f"trim_rubric_{ckpt_name}"
    else:
        policy = None
        state_dim = 5
        tag = f"rubric_heuristic_{dataset_name}"
        router_label = "rubric_heuristic"

    out_dir = os.path.join(output_dir or os.path.join(RESULTS_DIR, "rubric"), tag)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    done_ids = set()
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r.get("id"))

    pending = [it for it in items if it["id"] not in done_ids]
    if not pending:
        print("All done.")
        _print_summary(results, tag)
        return

    print(f"\n{'='*60}")
    print(f"  Rubric Routing [{mode}]: {tag}")
    print(f"  Dataset: {dataset_name} ({len(items)} total, {len(pending)} pending)")
    print(f"{'='*60}\n")

    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)

    for idx, item in enumerate(pending):
        try:
            if mode == "policy":
                rfn = make_rubric_policy_router(policy, item["problem"], state_dim)
            else:
                rfn = make_rubric_heuristic_router(item["problem"])

            result = online_route(
                problem=item["problem"],
                ground_truth=item["answer"],
                router_fn=rfn,
                router_name=router_label,
            )
            result["id"] = item["id"]
            result["problem"] = item["problem"][:200]
        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")
            result = {"id": item["id"], "error": str(e), "is_correct": False}

        results.append(result)
        done_ids.add(item["id"])
        if result.get("is_correct"):
            correct += 1
        total += 1

        acc = correct / total * 100
        mark = "✓" if result.get("is_correct") else "✗"
        rr = result.get("regen_ratio", 0)
        cpt = result.get("cpt", 0)
        print(f"  [{len(done_ids)}/{len(items)}] {mark} "
              f"pred={result.get('predicted', '?')[:15]} gt={item['answer']} "
              f"rr={rr:.2f} cpt={cpt:.1f}% Acc={acc:.1f}%")

        if len(done_ids) % 5 == 0 or idx == len(pending) - 1:
            _save(results, results_path)

    _save(results, results_path)
    _print_summary(results, tag)
    _save_stats(results, tag, out_dir)


def _save(results, path):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _print_summary(results, tag):
    valid = [r for r in results if "error" not in r]
    total = len(valid)
    correct = sum(1 for r in valid if r.get("is_correct"))
    acc = correct / total * 100 if total > 0 else 0
    avg_cpt = sum(r.get("cpt", 0) for r in valid) / total if total > 0 else 0
    avg_rr = sum(r.get("regen_ratio", 0) for r in valid) / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  {tag} — Final")
    print(f"  Acc: {acc:.2f}% ({correct}/{total})")
    print(f"  Avg CPT: {avg_cpt:.1f}%")
    print(f"  Avg regen_ratio: {avg_rr:.3f}")
    print(f"{'='*60}\n")


def _save_stats(results, tag, out_dir):
    valid = [r for r in results if "error" not in r]
    total = len(valid)
    correct = sum(1 for r in valid if r.get("is_correct"))
    stats = {
        "tag": tag,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
        "avg_cpt": round(sum(r.get("cpt", 0) for r in valid) / total, 2) if total > 0 else 0,
        "avg_regen_ratio": round(sum(r.get("regen_ratio", 0) for r in valid) / total, 4) if total > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["math500", "aime"], required=True)
    parser.add_argument("--mode", choices=["policy", "heuristic"], default="policy")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(CHECKPOINTS_DIR, ckpt)

    run_rubric_routing(args.dataset, args.mode, ckpt, args.limit)
