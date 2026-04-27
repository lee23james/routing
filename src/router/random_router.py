"""Random routing — randomly choose SRM or LRM for each step.

Usage:
    python -m router.random_router --dataset math500 --regen_ratio 0.5
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import RESULTS_DIR
from common.datasets import load_math500, load_aime
from router.core import online_route


def make_random_router(p_lrm: float = 0.5):
    """Returns a router function that uses LRM with probability p_lrm."""
    def router_fn(step_idx, srm_step, history):
        return random.random() < p_lrm
    return router_fn


def run_random_routing(dataset_name, regen_ratio=0.5, limit=None, output_dir=None, seed=42):
    random.seed(seed)

    if dataset_name == "math500":
        items = load_math500()
    else:
        items = load_aime(2020, 2024)

    if limit:
        items = items[:limit]

    tag = f"random_p{regen_ratio:.2f}_{dataset_name}"
    out_dir = os.path.join(output_dir or os.path.join(RESULTS_DIR, "router"), tag)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    # Resume
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

    router_fn = make_random_router(regen_ratio)
    print(f"\n{'='*60}")
    print(f"  Random Routing: {tag}")
    print(f"  p(LRM) = {regen_ratio}")
    print(f"  Dataset: {dataset_name} ({len(items)} total, {len(pending)} remaining)")
    print(f"{'='*60}\n")

    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)

    for idx, item in enumerate(pending):
        try:
            result = online_route(
                problem=item["problem"],
                ground_truth=item["answer"],
                router_fn=router_fn,
                router_name=f"random_p{regen_ratio:.2f}",
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
    parser.add_argument("--regen_ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_random_routing(args.dataset, args.regen_ratio, args.limit, seed=args.seed)
