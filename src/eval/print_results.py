"""Unified numerical results output.

Reads episode data + all trained checkpoints, prints a clean comparison
table with accuracy, FLOPs, and improvement numbers.

Usage:
    python -m eval.print_results
    python -m eval.print_results --episodes_path data/episodes/combined_episodes.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STATE_DIM, HIDDEN_DIM, ACTION_DIM, CHECKPOINTS_DIR, RESULTS_DIR
from router.policy import RouterPolicy
from router.env import TRIMEnv
from data.datasets import load_jsonl
from eval.flops_eval import (
    SRM_FLOPS_PER_TOKEN, LRM_FLOPS_PER_TOKEN,
    compute_episode_flops, compute_lrm_only_flops, compute_srm_only_flops,
    estimate_mixed_correctness,
)


def evaluate_checkpoint(episodes, episodes_path, checkpoint, device="cpu"):
    env = TRIMEnv(episodes_path)
    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()

    results_by_ds = defaultdict(lambda: {"correct": 0, "total": 0,
                                          "flops": 0.0, "regens": 0, "steps": 0})
    overall = {"correct": 0, "total": 0, "flops": 0.0, "regens": 0, "steps": 0}

    for i in range(len(episodes)):
        ep = episodes[i]
        state = env.reset(i)
        done = False
        while not done:
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = policy.get_action(st, deterministic=True)
            state, _, done, _ = env.step(action.item())

        info = env.get_episode_info()
        actions = info["actions"]
        is_correct = estimate_mixed_correctness(ep, actions)
        flops = compute_episode_flops(ep, actions)
        n_regens = sum(actions)
        n_steps = len(actions)
        dataset = ep.get("dataset", "unknown")

        for target in [overall, results_by_ds[dataset]]:
            target["correct"] += int(is_correct)
            target["total"] += 1
            target["flops"] += flops
            target["regens"] += n_regens
            target["steps"] += n_steps

    def _summarize(d):
        n = max(d["total"], 1)
        return {
            "accuracy": d["correct"] / n,
            "avg_flops_tflops": d["flops"] / n / 1e12,
            "regen_ratio": d["regens"] / max(d["steps"], 1),
            "correct": d["correct"],
            "total": n,
        }

    return {
        "overall": _summarize(overall),
        "by_dataset": {ds: _summarize(v) for ds, v in results_by_ds.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_path", type=str,
                        default="data/episodes/combined_episodes.jsonl")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    episodes = load_jsonl(args.episodes_path)
    n = len(episodes)
    print(f"Loaded {n} episodes from {args.episodes_path}")

    # ---- Baselines ----
    ds_stats = defaultdict(lambda: {"srm": 0, "lrm": 0, "total": 0,
                                     "srm_flops": 0.0, "lrm_flops": 0.0})
    for ep in episodes:
        ds = ep.get("dataset", "unknown")
        ds_stats[ds]["srm"] += int(ep.get("srm_correct", False))
        ds_stats[ds]["lrm"] += int(ep.get("lrm_correct", False))
        ds_stats[ds]["total"] += 1
        ds_stats[ds]["srm_flops"] += compute_srm_only_flops(ep)
        ds_stats[ds]["lrm_flops"] += compute_lrm_only_flops(ep)
    ds_stats["all"] = {
        "srm": sum(v["srm"] for v in ds_stats.values()),
        "lrm": sum(v["lrm"] for v in ds_stats.values()),
        "total": n,
        "srm_flops": sum(v["srm_flops"] for v in ds_stats.values()),
        "lrm_flops": sum(v["lrm_flops"] for v in ds_stats.values()),
    }

    print(f"\n{'='*90}")
    print(f"  BASELINE ACCURACY (from episode data)")
    print(f"{'='*90}")
    print(f"  {'Dataset':<15} {'N':>5} {'SRM-Only':>12} {'LRM-Only':>12} {'Gap':>10} {'SRM TF':>10} {'LRM TF':>10}")
    print(f"  {'-'*75}")
    for ds in ["math500", "aime2025", "all"]:
        s = ds_stats.get(ds)
        if not s:
            continue
        t = s["total"]
        srm_acc = s["srm"] / t * 100
        lrm_acc = s["lrm"] / t * 100
        gap = lrm_acc - srm_acc
        srm_tf = s["srm_flops"] / t / 1e12
        lrm_tf = s["lrm_flops"] / t / 1e12
        print(f"  {ds:<15} {t:>5} {srm_acc:>11.2f}% {lrm_acc:>11.2f}% {gap:>+9.2f}% {srm_tf:>9.2f} {lrm_tf:>9.2f}")

    # ---- Evaluate all checkpoints ----
    print(f"\n{'='*90}")
    print(f"  TRAINED ROUTER RESULTS")
    print(f"{'='*90}")

    import glob
    ckpt_dirs = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "*")))
    all_rows = []

    for ckpt_dir in ckpt_dirs:
        best_path = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(best_path):
            continue
        name = os.path.basename(ckpt_dir)
        try:
            result = evaluate_checkpoint(episodes, args.episodes_path, best_path)
            ov = result["overall"]
            all_rows.append({"name": name, **ov, "by_dataset": result["by_dataset"]})
        except Exception as e:
            print(f"  {name}: ERROR — {e}")

    # Sort by accuracy descending
    all_rows.sort(key=lambda r: r["accuracy"], reverse=True)

    lrm_acc_all = ds_stats["all"]["lrm"] / n
    srm_acc_all = ds_stats["all"]["srm"] / n
    lrm_tf_all = ds_stats["all"]["lrm_flops"] / n / 1e12

    print(f"\n  {'Method':<45} {'Acc%':>7} {'vs SRM':>8} {'vs LRM':>8} "
          f"{'TFLOPs':>8} {'%LRM':>7} {'Regen%':>7}")
    print(f"  {'-'*95}")

    print(f"  {'SRM-Only (Qwen3-1.7B)':<45} {srm_acc_all*100:>7.2f} {'—':>8} "
          f"{(srm_acc_all-lrm_acc_all)*100:>+7.2f}% {'—':>8} {'—':>7} {'0.0%':>7}")
    print(f"  {'LRM-Only (Qwen3-14B)':<45} {lrm_acc_all*100:>7.2f} "
          f"{(lrm_acc_all-srm_acc_all)*100:>+7.2f}% {'—':>8} {lrm_tf_all:>8.2f} "
          f"{'100%':>7} {'100%':>7}")
    print(f"  {'-'*95}")

    for row in all_rows:
        acc = row["accuracy"]
        vs_srm = (acc - srm_acc_all) * 100
        vs_lrm = (acc - lrm_acc_all) * 100
        tf = row["avg_flops_tflops"]
        pct_lrm = tf / lrm_tf_all * 100
        regen = row["regen_ratio"] * 100
        print(f"  {row['name']:<45} {acc*100:>7.2f} {vs_srm:>+7.2f}% "
              f"{vs_lrm:>+7.2f}% {tf:>8.2f} {pct_lrm:>6.1f}% {regen:>6.1f}%")

    # ---- Per-dataset breakdown for top configs ----
    print(f"\n{'='*90}")
    print(f"  PER-DATASET BREAKDOWN (top 10 by overall accuracy)")
    print(f"{'='*90}")
    for row in all_rows[:10]:
        print(f"\n  {row['name']}:")
        for ds in ["math500", "aime2025"]:
            ds_res = row.get("by_dataset", {}).get(ds)
            if not ds_res:
                continue
            ds_base = ds_stats.get(ds, {})
            ds_lrm = ds_base["lrm"] / ds_base["total"] * 100
            ds_srm = ds_base["srm"] / ds_base["total"] * 100
            acc = ds_res["accuracy"] * 100
            print(f"    {ds:<12}: {acc:>6.2f}% "
                  f"(SRM={ds_srm:.1f}%, LRM={ds_lrm:.1f}%, "
                  f"regen={ds_res['regen_ratio']:.1%}, "
                  f"TF={ds_res['avg_flops_tflops']:.2f})")

    # ---- Save plot_data.json (for plot_clean.py) ----
    plot_data = {"baselines": {}, "ppo_agg": [], "ppo_rubric": []}
    for ds in ["math500", "aime2025", "all"]:
        s = ds_stats.get(ds)
        if not s:
            continue
        t = s["total"]
        plot_data["baselines"][ds] = {
            "srm_acc": s["srm"] / t * 100,
            "lrm_acc": s["lrm"] / t * 100,
            "srm_flops": s["srm_flops"] / t / 1e12,
            "lrm_flops": s["lrm_flops"] / t / 1e12,
            "n": t,
        }

    for row in all_rows:
        name = row["name"]
        is_rubric = "rubric" in name.lower()
        for ds in ["math500", "aime2025", "all"]:
            ds_res = row.get("by_dataset", {}).get(ds)
            if ds == "all":
                ds_res = {"accuracy": row["accuracy"],
                          "avg_flops_tflops": row["avg_flops_tflops"]}
            if not ds_res:
                continue
            entry = {
                "acc": ds_res["accuracy"] * 100,
                "flops_tflops": ds_res["avg_flops_tflops"],
                "name": name,
                "dataset": ds,
            }
            if is_rubric:
                plot_data["ppo_rubric"].append(entry)
            else:
                plot_data["ppo_agg"].append(entry)

    plot_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "plot_data.json")
    with open(plot_path, "w") as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    print(f"\nPlot data saved → {plot_path}")

    # ---- Save summary JSON ----
    summary = {
        "baselines": {ds: {"srm_acc": v["srm"]/v["total"],
                           "lrm_acc": v["lrm"]/v["total"],
                           "n": v["total"]}
                      for ds, v in ds_stats.items()},
        "router_results": [{k: v for k, v in r.items() if k != "by_dataset"}
                           for r in all_rows],
    }
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(RESULTS_DIR, "summary_table.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary table saved → {out_path}")


if __name__ == "__main__":
    main()
