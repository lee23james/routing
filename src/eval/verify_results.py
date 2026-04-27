"""Verification script for TRIM-Agg and TRIM-Rubric results.

This script:
1. Verifies the correctness of the evaluation pipeline
2. Produces a clean Pareto-optimal comparison table
3. Generates per-dataset breakdowns
4. Saves verification results to results/verification/
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

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


def evaluate_policy_detailed(episodes: List[Dict], checkpoint: str,
                              episodes_path: str = "../data/episodes/all_episodes.jsonl") -> Dict:
    """Detailed evaluation of a single policy checkpoint."""
    env = TRIMEnv(episodes_path)

    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    policy.eval()

    results_by_dataset = defaultdict(lambda: {"correct": 0, "total": 0, "flops": 0.0,
                                               "regens": 0, "steps": 0})
    overall = {"correct": 0, "total": 0, "flops": 0.0, "regens": 0, "steps": 0}
    per_problem = []

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

        overall["correct"] += int(is_correct)
        overall["total"] += 1
        overall["flops"] += flops
        overall["regens"] += n_regens
        overall["steps"] += n_steps

        results_by_dataset[dataset]["correct"] += int(is_correct)
        results_by_dataset[dataset]["total"] += 1
        results_by_dataset[dataset]["flops"] += flops
        results_by_dataset[dataset]["regens"] += n_regens
        results_by_dataset[dataset]["steps"] += n_steps

        per_problem.append({
            "id": ep.get("id", i),
            "dataset": dataset,
            "correct": is_correct,
            "srm_correct": ep.get("srm_correct", False),
            "lrm_correct": ep.get("lrm_correct", False),
            "actions": actions,
            "n_regens": n_regens,
            "flops": flops,
        })

    n = overall["total"]
    result = {
        "overall": {
            "accuracy": overall["correct"] / n,
            "avg_flops": overall["flops"] / n,
            "regen_ratio": overall["regens"] / max(overall["steps"], 1),
            "correct": overall["correct"],
            "total": n,
        },
        "by_dataset": {},
        "per_problem": per_problem,
    }

    for ds, stats in results_by_dataset.items():
        dn = stats["total"]
        result["by_dataset"][ds] = {
            "accuracy": stats["correct"] / dn,
            "avg_flops": stats["flops"] / dn,
            "regen_ratio": stats["regens"] / max(stats["steps"], 1),
            "correct": stats["correct"],
            "total": dn,
        }

    return result


def find_pareto_optimal(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """Find Pareto-optimal points (maximize accuracy, minimize FLOPs)."""
    sorted_pts = sorted(points, key=lambda x: x[2])  # sort by FLOPs
    pareto = []
    best_acc = -1
    for name, acc, flops in sorted_pts:
        if acc > best_acc:
            pareto.append((name, acc, flops))
            best_acc = acc
    return pareto


def main():
    episodes_path = "../data/episodes/all_episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    pass  # episodes loaded

    print(f"Loaded {len(episodes)} episodes")

    output_dir = os.path.join(RESULTS_DIR, "verification")
    os.makedirs(output_dir, exist_ok=True)

    n = len(episodes)
    srm_acc = sum(1 for ep in episodes if ep.get("srm_correct", False)) / n
    lrm_acc = sum(1 for ep in episodes if ep.get("lrm_correct", False)) / n
    srm_flops = np.mean([compute_srm_only_flops(ep) for ep in episodes])
    lrm_flops = np.mean([compute_lrm_only_flops(ep) for ep in episodes])

    print(f"\n{'='*80}")
    print(f"  BASELINE VERIFICATION")
    print(f"{'='*80}")
    print(f"  SRM-only: acc={srm_acc:.4f}  avg_flops={srm_flops:.2e}")
    print(f"  LRM-only: acc={lrm_acc:.4f}  avg_flops={lrm_flops:.2e}")
    print(f"  FLOPs ratio: SRM/LRM = {srm_flops/lrm_flops:.1%}")
    print(f"  Performance gap: LRM-SRM = {lrm_acc-srm_acc:.4f}")

    ds_stats = defaultdict(lambda: {"srm_corr": 0, "lrm_corr": 0, "total": 0})
    for ep in episodes:
        ds = ep.get("dataset", "unknown")
        ds_stats[ds]["srm_corr"] += int(ep.get("srm_correct", False))
        ds_stats[ds]["lrm_corr"] += int(ep.get("lrm_correct", False))
        ds_stats[ds]["total"] += 1

    print(f"\n  Per-dataset baselines:")
    for ds, stats in sorted(ds_stats.items()):
        t = stats["total"]
        print(f"    {ds:15s}: SRM={stats['srm_corr']}/{t} ({stats['srm_corr']/t:.1%}), "
              f"LRM={stats['lrm_corr']}/{t} ({stats['lrm_corr']/t:.1%})")

    # Evaluate all checkpoints
    print(f"\n{'='*80}")
    print(f"  POLICY EVALUATION")
    print(f"{'='*80}")

    all_points = [
        ("SRM-only", srm_acc, srm_flops),
        ("LRM-only", lrm_acc, lrm_flops),
    ]
    all_results = {}
    import glob
    for ckpt_dir in sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "*"))):
        best_path = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(best_path):
            continue
        name = os.path.basename(ckpt_dir)
        try:
            result = evaluate_policy_detailed(episodes, best_path, episodes_path)
            ov = result["overall"]
            flops_pct = ov["avg_flops"] / lrm_flops
            all_points.append((name, ov["accuracy"], ov["avg_flops"]))
            all_results[name] = result

            print(f"\n  {name}:")
            print(f"    Overall: acc={ov['accuracy']:.4f}  FLOPs={ov['avg_flops']:.2e} "
                  f"({flops_pct:.1%} LRM)  regen={ov['regen_ratio']:.2%}")
            for ds, ds_res in sorted(result["by_dataset"].items()):
                print(f"    {ds:15s}: acc={ds_res['accuracy']:.4f}  "
                      f"regen={ds_res['regen_ratio']:.2%}  "
                      f"({ds_res['correct']}/{ds_res['total']})")
        except Exception as e:
            print(f"  {name}: ERROR — {e}")

    # Pareto analysis
    agg_points = [(n, a, f) for n, a, f in all_points if "rubric" not in n.lower()]
    rub_points = [(n, a, f) for n, a, f in all_points if "rubric" in n.lower()]

    agg_pareto = find_pareto_optimal(agg_points)
    rub_pareto = find_pareto_optimal(rub_points)

    print(f"\n{'='*80}")
    print(f"  PARETO-OPTIMAL CONFIGURATIONS")
    print(f"{'='*80}")

    print(f"\n  TRIM-Agg Pareto front:")
    print(f"    {'Name':<45} {'Acc':>7} {'FLOPs':>12} {'%LRM':>7}")
    print(f"    {'-'*75}")
    for name, acc, flops in agg_pareto:
        print(f"    {name:<45} {acc:>7.4f} {flops:>12.2e} {flops/lrm_flops:>7.1%}")

    print(f"\n  TRIM-Rubric Pareto front:")
    print(f"    {'Name':<45} {'Acc':>7} {'FLOPs':>12} {'%LRM':>7}")
    print(f"    {'-'*75}")
    for name, acc, flops in rub_pareto:
        print(f"    {name:<45} {acc:>7.4f} {flops:>12.2e} {flops/lrm_flops:>7.1%}")

    # Key comparisons at matched accuracy levels
    print(f"\n{'='*80}")
    print(f"  KEY COMPARISONS (Matched Accuracy)")
    print(f"{'='*80}")

    acc_levels = sorted(set(a for _, a, _ in all_points if a > srm_acc))
    for target_acc in acc_levels:
        agg_at = [(n, f) for n, a, f in all_points if a == target_acc and "rubric" not in n.lower()]
        rub_at = [(n, f) for n, a, f in all_points if a == target_acc and "rubric" in n.lower()]
        if agg_at and rub_at:
            best_agg = min(agg_at, key=lambda x: x[1])
            best_rub = min(rub_at, key=lambda x: x[1])
            savings = (best_agg[1] - best_rub[1]) / best_agg[1] * 100
            print(f"  At acc={target_acc:.4f}:")
            print(f"    Best TRIM-Agg:    {best_agg[0]:45s}  FLOPs={best_agg[1]:.2e} ({best_agg[1]/lrm_flops:.1%} LRM)")
            print(f"    Best TRIM-Rubric: {best_rub[0]:45s}  FLOPs={best_rub[1]:.2e} ({best_rub[1]/lrm_flops:.1%} LRM)")
            print(f"    → TRIM-Rubric saves {savings:.1f}% FLOPs")

    # FLOPs metrics (EcoTab-style)
    target_flops_60 = 0.6 * lrm_flops
    target_acc_98 = 0.98 * lrm_acc

    print(f"\n{'='*80}")
    print(f"  ECOTAB-STYLE METRICS")
    print(f"{'='*80}")
    print(f"  60% LRM FLOPs budget = {target_flops_60:.2e}")
    print(f"  98% LRM accuracy target = {target_acc_98:.4f}")

    for category, points in [("TRIM-Agg", agg_points), ("TRIM-Rubric", rub_points)]:
        best_under_budget = [(n, a, f) for n, a, f in points if f <= target_flops_60]
        if best_under_budget:
            best = max(best_under_budget, key=lambda x: x[0] if x[1] == max(b[1] for b in best_under_budget) else "")
            best = max(best_under_budget, key=lambda x: (x[1], -x[2]))
            print(f"\n  {category}:")
            print(f"    Acc@60%FLOPs: {best[1]:.4f} ({best[0]})")
        above_98 = [(n, a, f) for n, a, f in points if a >= target_acc_98]
        if above_98:
            cheapest = min(above_98, key=lambda x: x[2])
            print(f"    FLOPs@98%Acc: {cheapest[2]:.2e} ({cheapest[2]/lrm_flops:.1%} LRM) ({cheapest[0]})")

    # Save verification data
    save_data = {
        "baselines": {
            "srm_only": {"accuracy": srm_acc, "avg_flops": srm_flops},
            "lrm_only": {"accuracy": lrm_acc, "avg_flops": lrm_flops},
        },
        "per_dataset_baselines": {ds: {"srm_acc": s["srm_corr"]/s["total"],
                                        "lrm_acc": s["lrm_corr"]/s["total"],
                                        "total": s["total"]}
                                   for ds, s in ds_stats.items()},
        "all_points": [{"name": n, "accuracy": a, "avg_flops": f, "flops_pct_lrm": f/lrm_flops}
                       for n, a, f in all_points],
        "pareto_agg": [{"name": n, "accuracy": a, "avg_flops": f} for n, a, f in agg_pareto],
        "pareto_rubric": [{"name": n, "accuracy": a, "avg_flops": f} for n, a, f in rub_pareto],
        "ecotab_metrics": {
            "target_flops_60pct": target_flops_60,
            "target_acc_98pct": target_acc_98,
            "lrm_only_flops": lrm_flops,
        },
    }

    save_path = os.path.join(output_dir, "verification_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
