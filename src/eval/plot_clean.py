"""Clean Accuracy-vs-FLOPs plot: SRM-only, LRM-only, TRIM-Agg, TRIM-Rubric.

Shows Pareto-optimal frontiers with clear visual distinction.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

try:
    from matplotlib import font_manager
    font_path = "/export/shy/pp/pp4/src/study/ARIAL.TTF"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['font.family'] = 'Arial'
except Exception:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def pareto_front(points):
    """Return strictly Pareto-optimal points (max accuracy, min flops).
    
    A point (f, a) is Pareto-optimal if no other point has both
    higher accuracy and lower FLOPs.
    """
    if not points:
        return []
    deduped = {}
    for f, a in points:
        key = round(a, 1)
        if key not in deduped or f < deduped[key]:
            deduped[key] = f
    sorted_pts = sorted([(f, a) for a, f in deduped.items()], key=lambda x: x[0])
    front = []
    best_acc = -1
    for f, a in sorted_pts:
        if a > best_acc:
            front.append((f, a))
            best_acc = a
    return front


def main():
    plot_data_path = os.path.join(RESULTS_DIR, "plots", "plot_data.json")
    with open(plot_data_path) as f:
        data = json.load(f)

    baselines = data["baselines"]
    ppo_agg = data["ppo_agg"]
    ppo_rubric = data["ppo_rubric"]

    datasets = ["math500", "aime2025", "all"]
    ds_labels = {"math500": "MATH-500", "aime2025": "AIME2025 I&II", "all": "Overall"}

    # Build per-dataset Pareto-optimal points
    agg_by_ds = defaultdict(list)
    rub_by_ds = defaultdict(list)
    for pt in ppo_agg:
        agg_by_ds[pt["dataset"]].append((pt["flops_tflops"], pt["acc"]))
    for pt in ppo_rubric:
        rub_by_ds[pt["dataset"]].append((pt["flops_tflops"], pt["acc"]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    for pi, ds in enumerate(datasets):
        ax = axes[pi]
        bl = baselines.get(ds, {})
        if not bl:
            continue

        srm_f, srm_a = bl["srm_flops"], bl["srm_acc"]
        lrm_f, lrm_a = bl["lrm_flops"], bl["lrm_acc"]

        # Pareto fronts
        agg_pts = agg_by_ds.get(ds, [])
        rub_pts = rub_by_ds.get(ds, [])

        agg_pareto = pareto_front(agg_pts)
        rub_pareto = pareto_front(rub_pts)

        # Add SRM-only as starting point for both
        agg_pareto = [(srm_f, srm_a)] + agg_pareto
        rub_pareto = [(srm_f, srm_a)] + rub_pareto

        # Random routing baseline: linear interpolation between SRM and LRM
        random_f = np.linspace(srm_f, lrm_f, 8)
        random_a = np.linspace(srm_a, lrm_a, 8)

        # Plot
        ax.plot(random_f, random_a, linestyle=':', marker='D', markersize=5,
                color='#888888', linewidth=1.5, label='Random Routing', alpha=0.7,
                zorder=2)

        if len(agg_pareto) >= 2:
            af = [f for f, a in agg_pareto]
            aa = [a for f, a in agg_pareto]
            ax.plot(af, aa, linestyle='--', marker='s', markersize=9,
                    color='#2196F3', linewidth=2.5, label='TRIM-Agg (PPO)',
                    zorder=4, markeredgecolor='white', markeredgewidth=0.8)

        if len(rub_pareto) >= 2:
            rf = [f for f, a in rub_pareto]
            ra = [a for f, a in rub_pareto]
            ax.plot(rf, ra, linestyle='-', marker='o', markersize=9,
                    color='#E91E63', linewidth=2.5, label='TRIM-Rubric (PPO)',
                    zorder=5, markeredgecolor='white', markeredgewidth=0.8)

        # SRM / LRM anchor stars
        ax.scatter([srm_f], [srm_a], marker='*', s=350,
                   facecolor='#FF5722', edgecolor='black', linewidth=1.2,
                   zorder=6, label='SRM-Only')
        ax.scatter([lrm_f], [lrm_a], marker='*', s=350,
                   facecolor='#4CAF50', edgecolor='black', linewidth=1.2,
                   zorder=6, label='LRM-Only')

        # Reference lines
        flops_60 = lrm_f * 0.6
        acc_98 = lrm_a * 0.98
        ax.axvline(x=flops_60, color='#9E9E9E', linestyle='-.', alpha=0.4, linewidth=1)
        ax.axhline(y=acc_98, color='#9E9E9E', linestyle='-.', alpha=0.4, linewidth=1)

        y_lo = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else min(srm_a - 5, 0)
        ax.text(flops_60 + 0.3, y_lo + 1, '60%\nLRM FLOPs', fontsize=9,
                color='#757575', ha='left', va='bottom')
        ax.text(0.5, acc_98 + 0.5, '98% LRM Acc', fontsize=9,
                color='#757575', ha='left', va='bottom',
                transform=ax.get_yaxis_transform())

        ax.set_title(f'{ds_labels.get(ds, ds)} (n={bl["n"]})', fontsize=20, fontweight='bold')
        ax.set_xlabel('TFLOPs per query', fontsize=16)
        if pi == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=18)
        ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)

        # Set ylim with padding
        all_accs = [srm_a, lrm_a] + [a for _, a in agg_pts] + [a for _, a in rub_pts]
        if all_accs:
            y_range = max(all_accs) - min(all_accs)
            pad = max(y_range * 0.1, 3)
            y_min = max(min(all_accs) - pad, 0)
            y_max = min(max(all_accs) + pad, 100)
            ax.set_ylim(y_min, y_max)
        # Set xlim
        all_flops = [srm_f, lrm_f] + [f for f, _ in agg_pts] + [f for f, _ in rub_pts]
        if all_flops:
            x_max = max(all_flops) * 1.08
            ax.set_xlim(-0.5, x_max)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    fig.legend(unique_h, unique_l, loc='upper center', ncol=5,
               frameon=False, fontsize=17, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "main_comparison.png")
    out_pdf = os.path.join(out_dir, "main_comparison.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")

    # ============================================================
    # Print concise summary table
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  CONCISE COMPARISON (Pareto-optimal points only)")
    print(f"{'='*90}")

    for ds in datasets:
        bl = baselines.get(ds, {})
        if not bl:
            continue
        agg_pts_ds = agg_by_ds.get(ds, [])
        rub_pts_ds = rub_by_ds.get(ds, [])
        agg_par = pareto_front(agg_pts_ds)
        rub_par = pareto_front(rub_pts_ds)

        print(f"\n  {ds_labels.get(ds, ds)} (n={bl['n']})")
        print(f"  {'Method':<25} {'Acc%':>7} {'TFLOPs':>8} {'%LRM FLOPs':>12}")
        print(f"  {'-'*55}")
        print(f"  {'SRM-Only':<25} {bl['srm_acc']:>7.1f} {bl['srm_flops']:>8.2f} {'—':>12}")

        for f, a in agg_par:
            pct = f / bl["lrm_flops"] * 100
            print(f"  {'TRIM-Agg':<25} {a:>7.1f} {f:>8.2f} {pct:>11.1f}%")

        for f, a in rub_par:
            pct = f / bl["lrm_flops"] * 100
            print(f"  {'TRIM-Rubric':<25} {a:>7.1f} {f:>8.2f} {pct:>11.1f}%")

        print(f"  {'LRM-Only':<25} {bl['lrm_acc']:>7.1f} {bl['lrm_flops']:>8.2f} {'100.0%':>12}")

    # Key comparison: at matched accuracy, how much FLOPs does Rubric save?
    print(f"\n{'='*90}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'='*90}")

    for ds in datasets:
        bl = baselines.get(ds, {})
        if not bl:
            continue
        agg_pts_ds = agg_by_ds.get(ds, [])
        rub_pts_ds = rub_by_ds.get(ds, [])

        agg_dict = defaultdict(list)
        rub_dict = defaultdict(list)
        for f, a in agg_pts_ds:
            agg_dict[round(a, 1)].append(f)
        for f, a in rub_pts_ds:
            rub_dict[round(a, 1)].append(f)

        print(f"\n  {ds_labels.get(ds, ds)}:")
        shared_accs = set(agg_dict.keys()) & set(rub_dict.keys())
        for acc in sorted(shared_accs, reverse=True):
            best_agg = min(agg_dict[acc])
            best_rub = min(rub_dict[acc])
            savings = (1 - best_rub / best_agg) * 100
            if savings > 0:
                print(f"    At {acc:.1f}% acc: Rubric uses {best_rub:.2f}T vs Agg {best_agg:.2f}T "
                      f"→ saves {savings:.1f}% FLOPs")
            else:
                print(f"    At {acc:.1f}% acc: Rubric {best_rub:.2f}T vs Agg {best_agg:.2f}T")


if __name__ == "__main__":
    main()
