"""TRIM-Agg: PPO-trained routing policy.

Loads a trained RouterPolicy checkpoint, and at each step builds the
5-dim state vector (min_prm, prod_prm, cur_prm, tok_ratio, step_ratio)
to decide SRM vs LRM.

Usage:
    python -m trim_agg.trim_agg --dataset math500 --checkpoint checkpoints/trim_agg/best.pt
"""

import argparse
import json
import os
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

TOKEN_NORMALISER = 512


def load_policy(ckpt_path: str, device: str = "cpu") -> RouterPolicy:
    """Load a trained RouterPolicy, auto-detecting architecture."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Detect state_dim from first layer
    for key in ["encoder.0.weight", "shared.0.weight"]:
        if key in sd:
            state_dim = sd[key].shape[1]
            break
    else:
        state_dim = 5

    # Detect hidden_dim
    for key in ["encoder.0.weight", "shared.0.weight"]:
        if key in sd:
            hidden_dim = sd[key].shape[0]
            break
    else:
        hidden_dim = 64

    policy = RouterPolicy(state_dim=state_dim, hidden_dim=hidden_dim)

    # Handle old checkpoint format (shared.X.weight -> encoder.X.weight)
    if "shared.0.weight" in sd:
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("shared.", "encoder.")
            new_sd[new_k] = v
        sd = new_sd

    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy


def make_agg_router(policy: RouterPolicy, problem: str, state_dim: int = 5):
    """Create a PPO-policy-based router function."""
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


def run_trim_agg(dataset_name, ckpt_path, limit=None, output_dir=None, tag_suffix=""):
    if dataset_name == "math500":
        items = load_math500()
    else:
        items = load_aime(2020, 2024)

    if limit:
        items = items[:limit]

    ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
    tag = f"trim_agg_{ckpt_name}{tag_suffix}_{dataset_name}"
    out_dir = os.path.join(output_dir or os.path.join(RESULTS_DIR, "trim_agg"), tag)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    policy = load_policy(ckpt_path)
    state_dim = next(p for p in policy.encoder.parameters()).shape[1]

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
    print(f"  TRIM-Agg: {ckpt_path}")
    print(f"  Dataset: {dataset_name} ({len(items)} total, {len(pending)} pending)")
    print(f"  Policy state_dim={state_dim}")
    print(f"{'='*60}\n")

    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)

    for idx, item in enumerate(pending):
        try:
            router_fn = make_agg_router(policy, item["problem"], state_dim)
            result = online_route(
                problem=item["problem"],
                ground_truth=item["answer"],
                router_fn=router_fn,
                router_name=f"trim_agg_{ckpt_name}",
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
        "checkpoint": os.path.basename(os.path.dirname(results[0].get("router", ""))) if results else "",
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["math500", "aime"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(CHECKPOINTS_DIR, ckpt)

    run_trim_agg(args.dataset, ckpt, args.limit)
