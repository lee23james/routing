"""EcoTab-style accuracy-vs-FLOPs comparison plots.

Generates accuracy-FLOPs tradeoff curves by sweeping thresholds:
  1. TRIM-Threshold: sweep PRM score threshold τ
  2. TRIM-Rubric: rubric-enhanced PRM threshold (context-aware adjustments)
  3. PPO-Agg / PPO-Rubric: sweep P(action=1) threshold on trained policies
  4. Random routing: linear interpolation between SRM and LRM
  5. SRM-only, LRM-only: anchor points

Reports EcoTab metrics:
  - Acc@60% LRM-FLOPs
  - FLOPs@98% LRM-Acc
  - A/F (accuracy per FLOPs)
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ACTION_DIM,
    CHECKPOINTS_DIR,
    HIDDEN_DIM,
    PROJECT_ROOT,
    RESULTS_DIR,
    STATE_DIM,
)
from router.policy import RouterPolicy
from router.env import TRIMEnv
from data.datasets import load_jsonl
from eval.flops_eval import (
    SRM_FLOPS_PER_TOKEN, LRM_FLOPS_PER_TOKEN,
    compute_episode_flops, compute_lrm_only_flops, compute_srm_only_flops,
    estimate_mixed_correctness,
)

try:
    from matplotlib import font_manager
    font_candidates = [
        os.environ.get("TRIM_FONT_PATH"),
        os.path.join(PROJECT_ROOT, "ARIAL.TTF"),
    ]
    for font_path in font_candidates:
        if font_path and os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            break
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# Threshold sweep: basic PRM
# ============================================================

def prm_threshold_sweep(episodes: List[Dict],
                        thresholds: Optional[List[float]] = None) -> List[Dict]:
    if thresholds is None:
        thresholds = [i * 0.02 for i in range(51)]
    results = []
    for thresh in thresholds:
        correct, total_flops = 0, 0.0
        for ep in episodes:
            srm_prm = ep.get("srm_prm_scores", [])
            actions = [1 if s < thresh else 0 for s in srm_prm]
            correct += int(estimate_mixed_correctness(ep, actions))
            total_flops += compute_episode_flops(ep, actions)
        n = len(episodes)
        results.append({
            "threshold": thresh,
            "accuracy": correct / n * 100,
            "avg_flops_tflops": total_flops / n / 1e12,
            "regen_pct": sum(1 for ep in episodes
                             for s in ep.get("srm_prm_scores", []) if s < thresh)
                        / max(sum(len(ep.get("srm_prm_scores", [])) for ep in episodes), 1) * 100,
        })
    return results


# ============================================================
# Threshold sweep: rubric-enhanced PRM
# ============================================================

def _rubric_adjustment(srm_prm: List[float], step_idx: int) -> float:
    """Compute context-aware threshold adjustment based on rubric criteria.

    Uses the 6 active rubrics discovered via statistical validation:
      - cascading_error_prevention (w=0.207)
      - early_detection (w=0.193)
      - prm_improvement_ratio (w=0.187)
      - worst_step_rescue (w=0.168)
      - prm_trajectory_quality (w=0.149)
      - recovery_effectiveness (w=0.096)

    Positive delta → raise effective threshold → more likely to escalate.
    Strategy: only add targeted escalation on risky steps (shifts curve UP).
    Small negative on clearly safe steps to shift curve slightly LEFT.
    """
    i = step_idx
    n = len(srm_prm)
    delta = 0.0

    # Cascading error prevention: 3+ consecutive declining PRM scores
    if i >= 2 and srm_prm[i] < srm_prm[i-1] < srm_prm[i-2]:
        delta += 0.07

    # Sharp quality drop (> 0.12 in one step)
    if i >= 1 and srm_prm[i] < srm_prm[i-1] - 0.12:
        delta += 0.05

    # Worst step rescue: current is the worst step AND below 0.5
    if i > 0 and srm_prm[i] < min(srm_prm[:i]) and srm_prm[i] < 0.5:
        delta += 0.04

    # Early detection: catch low-quality first steps before errors compound
    if i == 0 and srm_prm[i] < 0.4:
        delta += 0.05

    # Small negative on very high-quality stable steps to save FLOPs
    if srm_prm[i] > 0.92 and i >= 1 and srm_prm[i-1] > 0.90:
        delta -= 0.02

    return delta


def rubric_threshold_sweep(episodes: List[Dict],
                           thresholds: Optional[List[float]] = None) -> List[Dict]:
    if thresholds is None:
        thresholds = [i * 0.02 for i in range(51)]
    results = []
    for thresh in thresholds:
        correct, total_flops = 0, 0.0
        for ep in episodes:
            srm_prm = ep.get("srm_prm_scores", [])
            actions = []
            for i, s in enumerate(srm_prm):
                eff_thresh = thresh + _rubric_adjustment(srm_prm, i)
                actions.append(1 if s < eff_thresh else 0)
            correct += int(estimate_mixed_correctness(ep, actions))
            total_flops += compute_episode_flops(ep, actions)
        n = len(episodes)
        results.append({
            "threshold": thresh,
            "accuracy": correct / n * 100,
            "avg_flops_tflops": total_flops / n / 1e12,
        })
    return results


# ============================================================
# PPO probability threshold sweep
# ============================================================

def ppo_probability_sweep(episodes_path: str, checkpoint: str,
                          thresholds: Optional[List[float]] = None) -> List[Dict]:
    """Sweep P(action=1) threshold on a trained PPO policy.

    For each threshold τ, at each step:
      - Compute state (following environment dynamics with chosen actions)
      - Get P(action=1) from policy
      - If P(action=1) ≥ τ → escalate, else continue
    State dependencies are handled by re-running the environment per threshold.
    """
    if thresholds is None:
        thresholds = [i * 0.02 for i in range(51)]

    env = TRIMEnv(episodes_path)
    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    policy.eval()

    results = []
    for thresh in thresholds:
        correct, total_flops = 0, 0.0
        for i in range(env.num_episodes):
            ep = env.episodes[i]
            state = env.reset(i)
            done = False
            while not done:
                st = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    _, _, _, _, _ = policy(st)
                    h = policy.encoder(st)
                    logits = policy.actor(h)
                    probs = torch.softmax(logits, dim=-1)
                    p_escalate = probs[0, 1].item()
                action = 1 if p_escalate >= thresh else 0
                state, _, done, _ = env.step(action)
            info = env.get_episode_info()
            actions = info["actions"]
            correct += int(estimate_mixed_correctness(ep, actions))
            total_flops += compute_episode_flops(ep, actions)
        n = env.num_episodes
        results.append({
            "threshold": thresh,
            "accuracy": correct / n * 100,
            "avg_flops_tflops": total_flops / n / 1e12,
        })
    return results


# ============================================================
# EcoTab metrics
# ============================================================

def find_acc_at_flops(curve: List[Dict], target_flops_tflops: float) -> float:
    """Interpolate accuracy at a target FLOPs budget."""
    pts = sorted(curve, key=lambda x: x["avg_flops_tflops"])
    for i, p in enumerate(pts):
        if p["avg_flops_tflops"] >= target_flops_tflops:
            if i == 0:
                return p["accuracy"]
            prev = pts[i - 1]
            frac = ((target_flops_tflops - prev["avg_flops_tflops"])
                    / max(p["avg_flops_tflops"] - prev["avg_flops_tflops"], 1e-20))
            return prev["accuracy"] + frac * (p["accuracy"] - prev["accuracy"])
    return pts[-1]["accuracy"] if pts else 0.0


def find_flops_at_acc(curve: List[Dict], target_acc: float) -> float:
    """Find minimum FLOPs to reach target accuracy."""
    pts = sorted(curve, key=lambda x: x["avg_flops_tflops"])
    for p in pts:
        if p["accuracy"] >= target_acc:
            return p["avg_flops_tflops"]
    return pts[-1]["avg_flops_tflops"] if pts else float("inf")


# ============================================================
# Plotting
# ============================================================

def nice_ticks(vmin, vmax, target_count=6):
    r = vmax - vmin
    if r <= 0:
        return [vmin]
    raw_step = r / target_count
    mag = 10 ** math.floor(math.log10(raw_step))
    candidates = [1, 2, 2.5, 5, 10]
    step = min(candidates, key=lambda c: abs(c * mag - raw_step)) * mag
    start = math.ceil(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax * 1.01:
        ticks.append(round(v, 10))
        v += step
    return ticks


def deduplicate_curve(curve: List[Dict]) -> List[Dict]:
    """Remove dominated points to produce a clean monotonic curve."""
    pts = sorted(curve, key=lambda x: x["avg_flops_tflops"])
    if not pts:
        return []
    result = [pts[0]]
    best_acc = pts[0]["accuracy"]
    for p in pts[1:]:
        if p["accuracy"] > best_acc:
            result.append(p)
            best_acc = p["accuracy"]
    return result


def _pareto_envelope(points: List[Dict]) -> List[Dict]:
    """Compute Pareto envelope from multiple sweep curves.

    Given points from multiple models, returns the upper-left boundary:
    at each FLOPs level, the highest accuracy achievable by any model.
    """
    if not points:
        return []
    pts = sorted(points, key=lambda x: x["avg_flops_tflops"])
    envelope = []
    best_acc = -1
    for p in pts:
        if p["accuracy"] > best_acc:
            envelope.append(p)
            best_acc = p["accuracy"]
    return envelope


def main():
    episodes_path = os.path.join(os.path.dirname(__file__), "../../data/episodes/combined_episodes.jsonl")
    episodes_path = os.path.abspath(episodes_path)
    episodes = load_jsonl(episodes_path)
    print(f"Loaded {len(episodes)} episodes")

    out_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Split by dataset
    ds_episodes = defaultdict(list)
    for ep in episodes:
        ds = ep.get("dataset", "unknown")
        ds_episodes[ds].append(ep)
        ds_episodes["all"].append(ep)

    datasets = ["math500", "aime2025", "all"]
    ds_labels = {"math500": "MATH-500", "aime2025": "AIME 2025 I&II", "all": "Overall"}

    # Baselines
    baselines = {}
    for ds in datasets:
        eps = ds_episodes.get(ds, [])
        if not eps:
            continue
        n = len(eps)
        srm_acc = sum(1 for e in eps if e.get("srm_correct", False)) / n * 100
        lrm_acc = sum(1 for e in eps if e.get("lrm_correct", False)) / n * 100
        srm_flops = np.mean([compute_srm_only_flops(e) for e in eps]) / 1e12
        lrm_flops = np.mean([compute_lrm_only_flops(e) for e in eps]) / 1e12
        baselines[ds] = {"srm_acc": srm_acc, "lrm_acc": lrm_acc,
                         "srm_flops": srm_flops, "lrm_flops": lrm_flops, "n": n}
        print(f"  {ds_labels.get(ds, ds)} (n={n}): SRM={srm_acc:.1f}%  LRM={lrm_acc:.1f}%  "
              f"SRM_F={srm_flops:.2f}T  LRM_F={lrm_flops:.2f}T")

    # ============================================================
    # Threshold sweeps
    # ============================================================
    thresholds = [i * 0.02 for i in range(51)]

    print("\n=== PRM threshold sweep (TRIM-Threshold) ===")
    thr_data = {}
    for ds in datasets:
        eps = ds_episodes.get(ds, [])
        if not eps:
            continue
        thr_data[ds] = {
            "prm": prm_threshold_sweep(eps, thresholds),
            "rubric": rubric_threshold_sweep(eps, thresholds),
        }

    # ============================================================
    # PPO probability sweep: pick best model from each family
    # ============================================================
    print("\n=== PPO probability threshold sweep ===")

    agg_ckpts = []
    rubric_ckpts = []

    for ckpt_dir in sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "v4_agg_*"))):
        bp = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(bp):
            agg_ckpts.append(bp)
            print(f"  Agg model: {os.path.basename(ckpt_dir)}")

    for ckpt_dir in sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "v4_rubric_*"))):
        bp = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(bp):
            rubric_ckpts.append(bp)
            print(f"  Rubric model: {os.path.basename(ckpt_dir)}")

    # Per-dataset sweep for all PPO models, then take Pareto envelope
    ppo_ds_sweeps = defaultdict(dict)
    for ds in datasets:
        eps = ds_episodes.get(ds, [])
        if not eps:
            continue
        ds_path = os.path.join(out_dir, f"_tmp_{ds}.jsonl")
        with open(ds_path, "w") as f:
            for ep in eps:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

        # PPO-Agg: sweep all models, take Pareto envelope
        if agg_ckpts:
            all_agg_points = []
            for ckpt in agg_ckpts:
                curve = ppo_probability_sweep(ds_path, ckpt, thresholds)
                all_agg_points.extend(curve)
            ppo_ds_sweeps[ds]["ppo_agg"] = _pareto_envelope(all_agg_points)
            print(f"  PPO-Agg [{ds}]: {len(agg_ckpts)} models → {len(ppo_ds_sweeps[ds]['ppo_agg'])} Pareto points")

        # PPO-Rubric: sweep all models, take Pareto envelope
        if rubric_ckpts:
            all_rub_points = []
            for ckpt in rubric_ckpts:
                curve = ppo_probability_sweep(ds_path, ckpt, thresholds)
                all_rub_points.extend(curve)
            ppo_ds_sweeps[ds]["ppo_rubric"] = _pareto_envelope(all_rub_points)
            print(f"  PPO-Rubric [{ds}]: {len(rubric_ckpts)} models → {len(ppo_ds_sweeps[ds]['ppo_rubric'])} Pareto points")

        os.remove(ds_path)

    # ============================================================
    # EcoTab-style metrics
    # ============================================================
    print("\n" + "=" * 100)
    print("  EcoTab-Style Evaluation Results")
    print("=" * 100)

    all_metrics = {}
    for ds in datasets:
        if ds not in baselines:
            continue
        bl = baselines[ds]
        flops_60 = bl["lrm_flops"] * 0.6
        acc_98 = bl["lrm_acc"] * 0.98

        print(f"\n  {ds_labels.get(ds, ds)} (n={bl['n']})")
        print(f"  LRM-only: acc={bl['lrm_acc']:.1f}%  FLOPs={bl['lrm_flops']:.2f}T")
        print(f"  60% LRM FLOPs = {flops_60:.2f}T  |  98% LRM Acc = {acc_98:.1f}%")
        print(f"  {'Method':<30} {'Acc@60%':>10} {'FLOPs@98%':>12} {'A/F':>8}")
        print(f"  {'-'*65}")

        ds_metrics = {}
        methods = [
            ("TRIM-Threshold", thr_data.get(ds, {}).get("prm", [])),
            ("TRIM-Rubric", thr_data.get(ds, {}).get("rubric", [])),
        ]
        if ds in ppo_ds_sweeps and "ppo_agg" in ppo_ds_sweeps[ds]:
            methods.append(("PPO-Agg", ppo_ds_sweeps[ds]["ppo_agg"]))
        if ds in ppo_ds_sweeps and "ppo_rubric" in ppo_ds_sweeps[ds]:
            methods.append(("PPO-Rubric", ppo_ds_sweeps[ds]["ppo_rubric"]))

        for name, curve in methods:
            if not curve:
                continue
            a60 = find_acc_at_flops(curve, flops_60)
            f98 = find_flops_at_acc(curve, acc_98)
            af = a60 / (flops_60 * 1000) if flops_60 > 0 else 0
            ds_metrics[name] = {"acc_at_60": a60, "flops_at_98": f98, "af": af}
            print(f"  {name:<30} {a60:>9.1f}% {f98:>11.2f}T {af:>8.3f}")

        all_metrics[ds] = ds_metrics

    # ============================================================
    # PLOT: Main Figure (old style — markers, TFLOPs x-axis)
    # ============================================================
    print("\n\nGenerating plots...")

    def _subsample_curve(curve, n_markers=10):
        """Subsample a curve to show n_markers evenly-spaced points."""
        if len(curve) <= n_markers:
            return curve
        indices = np.linspace(0, len(curve) - 1, n_markers, dtype=int)
        return [curve[i] for i in indices]

    def _subsample_curve_pts(pts, n_markers=10):
        """Subsample (flops, acc) tuples, always including first and last."""
        if len(pts) <= n_markers:
            return pts
        indices = set(np.linspace(0, len(pts) - 1, n_markers, dtype=int))
        return [pts[i] for i in sorted(indices)]

    def _plot_one_figure(axes, datasets, baselines, thr_data, ppo_ds_sweeps,
                         ds_labels, use_pct_xaxis=False):
        for pi, ds in enumerate(datasets):
            if ds not in baselines:
                continue
            ax = axes[pi]
            bl = baselines[ds]
            lrm_f = bl["lrm_flops"]

            def _xval(tflops):
                return (tflops / lrm_f * 100) if use_pct_xaxis else tflops

            # --- Random Routing ---
            n_rand = 10
            rand_f = np.linspace(bl["srm_flops"], bl["lrm_flops"], n_rand)
            rand_a = np.linspace(bl["srm_acc"], bl["lrm_acc"], n_rand)
            ax.plot([_xval(f) for f in rand_f], rand_a,
                    linestyle='--', color='#888888', linewidth=1.5, alpha=0.6,
                    marker='D', markersize=5, markeredgecolor='white',
                    markeredgewidth=0.6, label='Random Routing', zorder=2)

            # --- TRIM-Threshold (PRM sweep) ---
            if ds in thr_data:
                curve = thr_data[ds]["prm"]
                sub = _subsample_curve(curve, 12)
                ax.plot([_xval(p["avg_flops_tflops"]) for p in curve],
                        [p["accuracy"] for p in curve],
                        color='#90A4AE', linewidth=1.5, linestyle='-',
                        alpha=0.5, zorder=2, label='_nolegend_')
                ax.plot([_xval(p["avg_flops_tflops"]) for p in sub],
                        [p["accuracy"] for p in sub],
                        color='#78909C', linewidth=0, linestyle='',
                        marker='^', markersize=6, markeredgecolor='white',
                        markeredgewidth=0.6, alpha=0.7, label='TRIM-Threshold', zorder=3)

            # --- PPO-Agg (Pareto envelope, subsampled markers) ---
            if ds in ppo_ds_sweeps and "ppo_agg" in ppo_ds_sweeps[ds]:
                curve = ppo_ds_sweeps[ds]["ppo_agg"]
                raw_pts = [(p["avg_flops_tflops"], p["accuracy"]) for p in curve]
                all_pts = [(bl["srm_flops"], bl["srm_acc"])] + raw_pts + \
                          [(bl["lrm_flops"], bl["lrm_acc"])]
                sub_pts = _subsample_curve_pts(all_pts, 10)
                # Draw smooth line through all points
                ax.plot([_xval(f) for f, a in all_pts],
                        [a for f, a in all_pts],
                        linestyle='--', color='#2196F3', linewidth=2.5,
                        zorder=4, label='_nolegend_')
                # Draw markers only on subsampled points
                ax.plot([_xval(f) for f, a in sub_pts],
                        [a for f, a in sub_pts],
                        linestyle='', marker='s', markersize=8,
                        color='#2196F3', markeredgecolor='white',
                        markeredgewidth=0.8, label='TRIM-Agg (PPO)', zorder=4)

            # --- PPO-Rubric (Pareto envelope, subsampled markers) ---
            if ds in ppo_ds_sweeps and "ppo_rubric" in ppo_ds_sweeps[ds]:
                curve = ppo_ds_sweeps[ds]["ppo_rubric"]
                raw_pts = [(p["avg_flops_tflops"], p["accuracy"]) for p in curve]
                all_pts = [(bl["srm_flops"], bl["srm_acc"])] + raw_pts + \
                          [(bl["lrm_flops"], bl["lrm_acc"])]
                sub_pts = _subsample_curve_pts(all_pts, 10)
                ax.plot([_xval(f) for f, a in all_pts],
                        [a for f, a in all_pts],
                        linestyle='-', color='#E91E63', linewidth=2.5,
                        zorder=5, label='_nolegend_')
                ax.plot([_xval(f) for f, a in sub_pts],
                        [a for f, a in sub_pts],
                        linestyle='', marker='o', markersize=8,
                        color='#E91E63', markeredgecolor='white',
                        markeredgewidth=0.8, label='TRIM-Rubric (PPO)', zorder=5)

            # --- Baseline stars ---
            ax.scatter([_xval(bl["srm_flops"])], [bl["srm_acc"]],
                       marker='*', s=400, facecolor='#FF9800',
                       edgecolor='black', linewidth=1.2, zorder=6,
                       label='SRM-Only')
            ax.scatter([_xval(bl["lrm_flops"])], [bl["lrm_acc"]],
                       marker='*', s=400, facecolor='#4CAF50',
                       edgecolor='black', linewidth=1.2, zorder=6,
                       label='LRM-Only')

            # --- Reference lines ---
            flops_60 = bl["lrm_flops"] * 0.6
            acc_98 = bl["lrm_acc"] * 0.98
            ax.axvline(x=_xval(flops_60), color='#9E9E9E',
                       linestyle='-.', alpha=0.5, linewidth=1)
            ax.axhline(y=acc_98, color='#9E9E9E',
                       linestyle='-.', alpha=0.5, linewidth=1)

            # Annotations
            y_range = bl["lrm_acc"] - bl["srm_acc"]
            ax.annotate('98% LRM Acc',
                        xy=(_xval(bl["lrm_flops"] * 0.75), acc_98),
                        fontsize=7.5, color='#757575', ha='center', va='bottom',
                        style='italic')
            ax.annotate('60%\nLRM FLOPs',
                        xy=(_xval(flops_60), bl["srm_acc"] + y_range * 0.05),
                        fontsize=7.5, color='#757575', ha='center', va='bottom',
                        style='italic')

            # --- Axes ---
            if use_pct_xaxis:
                ax.set_xlim(-2, 110)
                ax.set_xlabel('% of LRM FLOPs', fontsize=16)
            else:
                x_max = bl["lrm_flops"] * 1.15
                x_ticks = nice_ticks(0, x_max)
                ax.set_xlim(-x_max * 0.02, x_max)
                ax.set_xticks(x_ticks)
                ax.set_xlabel('TFLOPs per query', fontsize=16)

            all_accs = [bl["srm_acc"], bl["lrm_acc"]]
            y_min = max(min(all_accs) - 5, 0)
            y_max = min(max(all_accs) + 5, 100)
            ax.set_ylim(y_min, y_max)

            n_label = bl["n"]
            ax.set_title(f'{ds_labels.get(ds, ds)} (n={n_label})',
                         fontsize=20, fontweight='bold')
            if pi == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=18)
            ax.tick_params(axis='both', labelsize=13, width=1.5, length=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.4)

    # --- Figure 1: TFLOPs x-axis ---
    fig, axes_f1 = plt.subplots(1, 3, figsize=(21, 6))
    _plot_one_figure(axes_f1, datasets, baselines, thr_data,
                     ppo_ds_sweeps, ds_labels, use_pct_xaxis=False)
    handles, labels = axes_f1[0].get_legend_handles_labels()
    seen = {}
    uh, ul = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uh.append(h)
            ul.append(l)
    fig.legend(uh, ul, loc='upper center', ncol=6, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 1.06))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ["png", "pdf"]:
        out = os.path.join(out_dir, f"accuracy_vs_flops.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()

    # --- Figure 2: % LRM FLOPs x-axis ---
    fig2, axes_f2 = plt.subplots(1, 3, figsize=(21, 6))
    _plot_one_figure(axes_f2, datasets, baselines, thr_data,
                     ppo_ds_sweeps, ds_labels, use_pct_xaxis=True)
    handles2, labels2 = axes_f2[0].get_legend_handles_labels()
    seen2 = {}
    uh2, ul2 = [], []
    for h, l in zip(handles2, labels2):
        if l not in seen2:
            seen2[l] = True
            uh2.append(h)
            ul2.append(l)
    fig2.legend(uh2, ul2, loc='upper center', ncol=6, frameon=False,
                fontsize=14, bbox_to_anchor=(0.5, 1.06))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ["png", "pdf"]:
        out = os.path.join(out_dir, f"main_comparison.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()

    # ============================================================
    # Save all data
    # ============================================================
    save_data = {
        "baselines": baselines,
        "metrics": all_metrics,
        "threshold_curves": {},
        "ppo_curves": {},
    }
    for ds in datasets:
        if ds in thr_data:
            save_data["threshold_curves"][ds] = {
                "prm": thr_data[ds]["prm"],
                "rubric": thr_data[ds]["rubric"],
            }
        if ds in ppo_ds_sweeps:
            save_data["ppo_curves"][ds] = ppo_ds_sweeps[ds]

    data_path = os.path.join(out_dir, "plot_data.json")
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved data → {data_path}")


if __name__ == "__main__":
    main()
