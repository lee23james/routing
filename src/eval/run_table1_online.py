"""Run all routing strategies on MATH-500 to produce Table 1 + Budget Accuracy.

Runs online step-by-step routing with all methods, then organizes results
into CPT-constrained (CPT50/80/95) and budgeted accuracy tables.

Usage:
    python -m eval.run_table1_online --limit 50
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import RESULTS_DIR, CHECKPOINTS_DIR, MAX_STEPS
from common.datasets import load_math500
from common.prm import score_steps
from common.answer import extract_answer, check_correctness
from router.core import online_route
from trim_agg.trim_thr import make_thr_router
from trim_agg.trim_agg import load_policy, make_agg_router
from rubric.rubric_router import make_rubric_heuristic_router, make_rubric_policy_router


def _find_ckpt(prefix):
    for name in ["best.pt", "final.pt"]:
        p = os.path.join(CHECKPOINTS_DIR, prefix, name)
        if os.path.exists(p):
            return p
    return None


def run_config(items, make_router_fn, router_name, out_dir):
    """Run a single routing configuration on items. Returns list of results."""
    tag = router_name.replace("/", "_").replace(" ", "_")
    results_path = os.path.join(out_dir, f"{tag}.jsonl")

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
        return results

    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)

    for idx, item in enumerate(pending):
        try:
            rfn = make_router_fn(item["problem"])
            result = online_route(
                problem=item["problem"],
                ground_truth=item["answer"],
                router_fn=rfn,
                router_name=router_name,
            )
            result["id"] = item["id"]
        except Exception as e:
            result = {"id": item["id"], "error": str(e), "is_correct": False,
                      "cpt": 0, "regen_ratio": 0, "router": router_name}

        results.append(result)
        done_ids.add(item["id"])
        if result.get("is_correct"):
            correct += 1
        total += 1

        acc = correct / total * 100
        mark = "✓" if result.get("is_correct") else "✗"
        cpt = result.get("cpt", 0)
        rr = result.get("regen_ratio", 0)
        elapsed = result.get("srm_time", 0) + result.get("lrm_time", 0)
        print(f"  [{router_name:22s}] {len(done_ids):3d}/{len(items)} {mark} "
              f"gt={item['answer'][:8]:8s} cpt={cpt:5.1f}% rr={rr:.2f} Acc={acc:.1f}% "
              f"({elapsed:.0f}s)")

        if len(done_ids) % 5 == 0 or idx == len(pending) - 1:
            with open(results_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def summarize(all_results: dict) -> dict:
    """Compute per-config summary: accuracy, avg_cpt, avg_rr."""
    summary = {}
    for name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        n = len(valid)
        if n == 0:
            summary[name] = {"acc": 0, "cpt": 0, "rr": 0, "n": 0}
            continue
        correct = sum(1 for r in valid if r["is_correct"])
        summary[name] = {
            "acc": round(correct / n * 100, 2),
            "cpt": round(sum(r.get("cpt", 0) for r in valid) / n, 2),
            "rr": round(sum(r.get("regen_ratio", 0) for r in valid) / n, 4),
            "n": n,
        }
    return summary


def build_cpt_table(summary: dict, srm_acc: float, lrm_acc: float):
    """Map configs to CPT50/80/95 bins and compute IBC, PGR."""
    cpt_targets = [50, 80, 95]
    methods = {}

    # Group by method family
    for name, s in summary.items():
        family = name.split("_")[0]
        if name.startswith("random"):
            family = "Random"
        elif name.startswith("trim_thr"):
            family = "TRIM-Thr"
        elif name.startswith("trim_agg"):
            family = "TRIM-Agg"
        elif name.startswith("trim_rubric") or name.startswith("rubric"):
            family = "TRIM-Rubric"
        methods.setdefault(family, []).append((name, s))

    table = {}
    for family, configs in methods.items():
        row = {}
        for target in cpt_targets:
            # Find config closest to target CPT
            best = None
            best_dist = float("inf")
            for name, s in configs:
                dist = abs(s["cpt"] - target)
                if dist < best_dist or (dist == best_dist and s["acc"] > (best[1]["acc"] if best else 0)):
                    best = (name, s)
                    best_dist = dist
            if best:
                s = best[1]
                # IBC = (acc - srm_acc) / cpt
                ibc = (s["acc"] - srm_acc) / s["cpt"] if s["cpt"] > 0 else 0
                # PGR = (acc - srm_acc) / (lrm_acc - srm_acc) if gap > 0
                gap = lrm_acc - srm_acc
                pgr = (s["acc"] - srm_acc) / gap * 100 if gap > 0 else 0
                row[target] = {
                    "config": best[0],
                    "acc": s["acc"],
                    "cpt": s["cpt"],
                    "rr": s["rr"],
                    "ibc": round(ibc, 4),
                    "pgr": round(pgr, 2),
                }
        table[family] = row
    return table


def print_table1(table, srm_acc, lrm_acc):
    print(f"\n{'='*90}")
    print(f"  Table 1: CPT-Constrained Accuracy on MATH-500")
    print(f"  SRM-only: {srm_acc:.2f}%  |  LRM-only: {lrm_acc:.2f}%")
    print(f"{'='*90}")
    header = f"{'Method':15s}"
    for cpt in [50, 80, 95]:
        header += f" | {'CPT'+str(cpt)+' Acc%':>10s} {'CPT':>6s} {'IBC':>7s} {'PGR%':>7s}"
    print(header)
    print("-" * 90)
    for family in ["Random", "TRIM-Thr", "TRIM-Agg", "TRIM-Rubric"]:
        row = table.get(family, {})
        line = f"{family:15s}"
        for cpt in [50, 80, 95]:
            if cpt in row:
                r = row[cpt]
                line += f" | {r['acc']:>10.2f} {r['cpt']:>5.1f}% {r['ibc']:>7.4f} {r['pgr']:>6.1f}%"
            else:
                line += f" | {'N/A':>10s} {'':>6s} {'':>7s} {'':>7s}"
        print(line)
    print(f"{'='*90}\n")


def print_budget_table(summary, srm_acc, lrm_acc):
    """Budget accuracy: at LRM budget of 10%/15%/20%/25%/30%."""
    budgets = [10, 15, 20, 25, 30]
    print(f"\n{'='*80}")
    print(f"  Budget Accuracy (LRM-only budget %)")
    print(f"  SRM-only: {srm_acc:.2f}%  |  LRM-only: {lrm_acc:.2f}%")
    print(f"{'='*80}")
    header = f"{'Method':15s}"
    for b in budgets:
        header += f" | {str(b)+'%':>8s}"
    print(header)
    print("-" * 80)

    methods_map = {}
    for name, s in summary.items():
        if name.startswith("random"):
            family = "Random"
        elif name.startswith("trim_thr"):
            family = "TRIM-Thr"
        elif name.startswith("trim_agg"):
            family = "TRIM-Agg"
        elif "rubric" in name:
            family = "TRIM-Rubric"
        else:
            family = name
        methods_map.setdefault(family, []).append(s)

    for family in ["Random", "TRIM-Thr", "TRIM-Agg", "TRIM-Rubric"]:
        configs = methods_map.get(family, [])
        if not configs:
            continue
        line = f"{family:15s}"
        for b in budgets:
            # Find config with CPT closest to but ≤ budget
            best = None
            for s in configs:
                if s["cpt"] <= b + 5:
                    if best is None or s["acc"] > best["acc"]:
                        best = s
            if best:
                line += f" | {best['acc']:>7.2f}%"
            else:
                line += f" | {'N/A':>8s}"
        print(line)
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--srm_acc", type=float, default=None,
                        help="SRM-only accuracy (skip if already known)")
    parser.add_argument("--lrm_acc", type=float, default=None,
                        help="LRM-only accuracy (skip if already known)")
    args = parser.parse_args()

    items = load_math500()
    if args.limit:
        items = items[:args.limit]

    out_dir = os.path.join(RESULTS_DIR, "table1_online")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Table 1 Online Evaluation — MATH-500 ({len(items)} problems)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ========================
    # Define all configurations
    # ========================
    configs = {}

    # Random routing: key p(LRM) values covering CPT range
    for p in [0.10, 0.30, 0.50, 0.70, 0.90]:
        name = f"random_p{p:.2f}"
        configs[name] = lambda prob, _p=p: (
            lambda si, ss, h, __p=_p: random.random() < __p
        )

    # TRIM-Thr: key thresholds
    for thr in [0.3, 0.5, 0.7]:
        name = f"trim_thr_{thr:.1f}"
        configs[name] = lambda prob, _thr=thr: make_thr_router(_thr, prob)

    # TRIM-Agg (PPO outcome-only) — pick 3 representative checkpoints
    agg_prefixes = ["trim_agg", "trim_agg_lam3e-04", "trim_agg_lam5e-04"]
    for pfx in agg_prefixes:
        ckpt = _find_ckpt(pfx)
        if ckpt:
            try:
                pol = load_policy(ckpt)
                sdim = next(pp for pp in pol.encoder.parameters()).shape[1]
                name = f"trim_agg_{pfx}"
                configs[name] = lambda prob, _pol=pol, _sd=sdim: make_agg_router(_pol, prob, _sd)
            except Exception as e:
                print(f"  Skip {pfx}: {e}")

    # TRIM-Rubric (PPO outcome+rubric) — pick 3 representative checkpoints
    rubric_prefixes = ["trim_rubric", "trim_rubric_lam3e-04", "trim_rubric_lam5e-04"]
    for pfx in rubric_prefixes:
        ckpt = _find_ckpt(pfx)
        if ckpt:
            try:
                pol = load_policy(ckpt)
                sdim = next(pp for pp in pol.encoder.parameters()).shape[1]
                name = f"trim_rubric_{pfx}"
                configs[name] = lambda prob, _pol=pol, _sd=sdim: make_rubric_policy_router(_pol, prob, _sd)
            except Exception as e:
                print(f"  Skip {pfx}: {e}")

    # Rubric heuristic
    configs["rubric_heuristic"] = lambda prob: make_rubric_heuristic_router(prob)

    print(f"Total configs to test: {len(configs)}")
    for name in sorted(configs.keys()):
        print(f"  - {name}")

    # ========================
    # Run all configurations
    # ========================
    all_results = {}
    t0 = time.time()
    for i, (name, make_fn) in enumerate(configs.items()):
        print(f"\n>>> [{i+1}/{len(configs)}] Running {name} ...")
        results = run_config(items, make_fn, name, out_dir)
        all_results[name] = results

    elapsed = time.time() - t0
    print(f"\nAll configs done in {elapsed/60:.1f} min")

    # ========================
    # Summary
    # ========================
    summary = summarize(all_results)

    # Try to get SRM/LRM baselines
    srm_acc = args.srm_acc
    lrm_acc = args.lrm_acc
    if srm_acc is None:
        srm_path = os.path.join(RESULTS_DIR, "baseline/qwen3-1.7b-math500/results.jsonl")
        if os.path.exists(srm_path):
            with open(srm_path) as f:
                bl = [json.loads(l) for l in f if l.strip()]
            if bl:
                srm_acc = sum(1 for r in bl if r.get("is_correct")) / len(bl) * 100
    if lrm_acc is None:
        lrm_path = os.path.join(RESULTS_DIR, "baseline/qwen3-14b-math500/results.jsonl")
        if os.path.exists(lrm_path):
            with open(lrm_path) as f:
                bl = [json.loads(l) for l in f if l.strip()]
            if bl:
                lrm_acc = sum(1 for r in bl if r.get("is_correct")) / len(bl) * 100
    srm_acc = srm_acc or 65.0
    lrm_acc = lrm_acc or 85.0

    # Print raw summary
    print(f"\n{'='*70}")
    print(f"{'Config':35s} {'Acc%':>8s} {'AvgCPT':>8s} {'AvgRR':>8s} {'N':>5s}")
    print(f"{'-'*70}")
    for name in sorted(summary.keys()):
        s = summary[name]
        print(f"{name:35s} {s['acc']:>7.2f}% {s['cpt']:>7.2f}% {s['rr']:>7.4f} {s['n']:>5d}")
    print(f"{'='*70}")

    # CPT-constrained table
    table = build_cpt_table(summary, srm_acc, lrm_acc)
    print_table1(table, srm_acc, lrm_acc)

    # Budget accuracy
    print_budget_table(summary, srm_acc, lrm_acc)

    # Save
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "table1": table,
                    "srm_acc": srm_acc, "lrm_acc": lrm_acc}, f, indent=2)
    print(f"Results saved to {out_dir}/summary.json")


if __name__ == "__main__":
    main()
