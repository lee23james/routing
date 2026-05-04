"""Local TRIM-Agg baseline plots and main-results table.

This script reproduces a compact baseline figure from the current ``src``
workspace: SRM-only, LRM-only, Random Routing, and TRIM-Agg (PPO).
It intentionally avoids the historical ``v4_agg_*`` paths used by
``plot_results.py`` and reads the local ``trim_agg_baseline_lam*/best.pt``
checkpoints instead.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ACTION_DIM, HIDDEN_DIM, STATE_DIM
from data.datasets import load_jsonl
from eval.flops_eval import (
    compute_episode_flops,
    estimate_mixed_correctness,
)
from router.env import TRIMEnv
from router.policy import RouterPolicy

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - exercised by CLI environment.
    raise RuntimeError("matplotlib is required to generate baseline plots") from exc


SRC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = {
    "math500": SRC_ROOT / "data/episodes/math500_episodes.jsonl",
    "aime2025": SRC_ROOT / "data/episodes/aime2025_episodes.jsonl",
}
DEFAULT_CHECKPOINT_GLOB = str(SRC_ROOT / "checkpoints/trim_agg_baseline_lam*/best.pt")
DEFAULT_OUTPUT_DIR = SRC_ROOT / "results/trim_agg_baseline/plots_copy_style"
DATASETS = ["math500", "aime2025", "all"]
DS_LABELS = {"math500": "MATH-500", "aime2025": "AIME 2025 I&II", "all": "Overall"}


def probability_thresholds() -> List[float]:
    """Return a dense sweep grid for P(action=1) routing thresholds."""
    values = set(np.linspace(0.0, 1.0, 21).round(6).tolist())
    values.update([0.995, 0.99, 0.98, 0.95, 0.90])
    return sorted(values, reverse=True)


def _parse_float_token(token: str) -> Optional[float]:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def parse_checkpoint_metadata(checkpoint_path: str) -> Dict:
    """Parse lambda/seed/kind metadata from point-search checkpoint paths."""
    path = Path(checkpoint_path)
    directory = path.parent.name
    filename = path.name

    lam = None
    seed = None
    lam_match = re.search(r"_lam([^_/]+)", directory)
    if lam_match:
        lam = _parse_float_token(lam_match.group(1))
    seed_match = re.search(r"_seed(\d+)", directory)
    if seed_match:
        seed = int(seed_match.group(1))

    checkpoint_kind = "unknown"
    epoch = None
    if filename == "best.pt":
        checkpoint_kind = "best"
    elif filename == "final.pt":
        checkpoint_kind = "final"
    else:
        epoch_match = re.match(r"epoch_(\d+)\.pt$", filename)
        if epoch_match:
            checkpoint_kind = "epoch"
            epoch = int(epoch_match.group(1))

    return {
        "checkpoint_file": str(path),
        "checkpoint_dir": directory,
        "checkpoint_name": f"{directory}/{filename}",
        "lambda": lam,
        "seed": seed,
        "checkpoint_kind": checkpoint_kind,
        "epoch": epoch,
    }


def expand_checkpoint_globs(patterns: List[str]) -> List[Path]:
    paths = []
    seen = set()
    for pattern in patterns:
        for item in glob.glob(pattern, recursive=True):
            path = Path(item)
            if path.is_file() and str(path) not in seen:
                seen.add(str(path))
                paths.append(path)
    return sorted(paths, key=lambda p: str(p))


def parse_dataset_names(value: str) -> List[str]:
    """Parse requested plot datasets and add Overall when both component sets exist."""
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one dataset must be requested")

    invalid = [item for item in requested if item not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown dataset(s): {', '.join(invalid)}")

    result = []
    for dataset in requested:
        if dataset not in result:
            result.append(dataset)
    if "math500" in result and "aime2025" in result and "all" not in result:
        result.append("all")
    return result


def _base_datasets_for(requested_datasets: Iterable[str]) -> List[str]:
    base = []
    for dataset in requested_datasets:
        if dataset == "all":
            for component in ["math500", "aime2025"]:
                if component not in base:
                    base.append(component)
        elif dataset not in base:
            base.append(dataset)
    return base


def load_episode_groups(paths: Dict[str, Path], requested_datasets: Iterable[str] = DATASETS) -> Dict[str, List[Dict]]:
    groups = {}
    requested = list(requested_datasets)
    for dataset in _base_datasets_for(requested):
        path = paths[dataset]
        if not path.exists():
            raise FileNotFoundError(f"Episode file not found: {path}")
        groups[dataset] = load_jsonl(str(path))
    if "all" in requested:
        groups["all"] = [ep for dataset in ["math500", "aime2025"] for ep in groups[dataset]]
    return groups


def _routing_flops(ep: Dict, actions: List[int]) -> float:
    """Compute routing FLOPs for an explicit action sequence.

    Scheme A keeps baselines and policy curves on the same routing-cost
    surface: SRM-only is all action=0, and LRM-only is all action=1.
    """
    return compute_episode_flops(ep, actions)


def _num_routing_steps(ep: Dict) -> int:
    return min(
        len(ep.get("srm_steps", [])),
        len(ep.get("lrm_steps", [])),
        len(ep.get("srm_token_counts", [])),
        len(ep.get("lrm_token_counts", [])),
    )


def _srm_only_routing_flops(ep: Dict) -> float:
    return _routing_flops(ep, [0] * _num_routing_steps(ep))


def _lrm_only_routing_flops(ep: Dict) -> float:
    return _routing_flops(ep, [1] * _num_routing_steps(ep))


def compute_baselines(groups: Dict[str, List[Dict]], datasets: Iterable[str] = DATASETS) -> Dict[str, Dict]:
    baselines = {}
    for dataset in datasets:
        episodes = groups[dataset]
        n = len(episodes)
        srm_correct = sum(1 for ep in episodes if ep.get("srm_correct", False))
        lrm_correct = sum(1 for ep in episodes if ep.get("lrm_correct", False))
        srm_flops = np.mean([_srm_only_routing_flops(ep) for ep in episodes]) / 1e12
        lrm_flops = np.mean([_lrm_only_routing_flops(ep) for ep in episodes]) / 1e12
        baselines[dataset] = {
            "srm_acc": srm_correct / max(n, 1) * 100,
            "lrm_acc": lrm_correct / max(n, 1) * 100,
            "srm_flops": float(srm_flops),
            "lrm_flops": float(lrm_flops),
            "n": n,
        }
    return baselines


def random_curve(baseline: Dict, n_points: int = 101) -> List[Dict]:
    flops = np.linspace(baseline["srm_flops"], baseline["lrm_flops"], n_points)
    accs = np.linspace(baseline["srm_acc"], baseline["lrm_acc"], n_points)
    return [
        {
            "p": float(i / (n_points - 1)),
            "accuracy": float(acc),
            "avg_flops_tflops": float(fl),
            "regen_ratio": float(i / (n_points - 1)),
        }
        for i, (fl, acc) in enumerate(zip(flops, accs))
    ]


def evaluate_policy_threshold_curve(
    episodes_path: Path,
    checkpoint: Path,
    thresholds: Iterable[float],
    device: str,
) -> List[Dict]:
    env = TRIMEnv(str(episodes_path))
    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    policy.load_state_dict(torch.load(str(checkpoint), map_location=device, weights_only=True))
    policy.eval()

    # The TRIM state depends on previous chosen PRM values, so probabilities must be
    # computed inside each threshold rollout. Caching the model still avoids reloads.
    results = []
    metadata = parse_checkpoint_metadata(str(checkpoint))
    for threshold in thresholds:
        correct = 0
        total_flops = 0.0
        total_regens = 0
        total_steps = 0

        for idx in range(env.num_episodes):
            ep = env.episodes[idx]
            state = env.reset(idx)
            done = False
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    h = policy.encoder(state_t)
                    logits = policy.actor(h)
                    probs = torch.softmax(logits, dim=-1)
                    p_escalate = probs[0, 1].item()
                action = 1 if p_escalate >= threshold else 0
                state, _, done, _ = env.step(action)

            actions = env.get_episode_info()["actions"]
            correct += int(estimate_mixed_correctness(ep, actions))
            total_flops += _routing_flops(ep, actions)
            total_regens += sum(actions)
            total_steps += len(actions)

        n = max(env.num_episodes, 1)
        results.append(
            {
                "checkpoint": metadata["checkpoint_name"],
                **metadata,
                "threshold": float(threshold),
                "accuracy": correct / n * 100,
                "avg_flops_tflops": total_flops / n / 1e12,
                "regen_ratio": total_regens / max(total_steps, 1),
                "correct": correct,
                "n": env.num_episodes,
            }
        )

    return results


def aggregate_all_dataset_points(per_dataset_points: Dict[str, List[Dict]]) -> List[Dict]:
    if "math500" not in per_dataset_points or "aime2025" not in per_dataset_points:
        return []
    by_key = defaultdict(lambda: {"correct": 0, "n": 0, "flops_sum_t": 0.0, "regens": 0.0, "template": None})
    for dataset in ["math500", "aime2025"]:
        for point in per_dataset_points[dataset]:
            key = (point["checkpoint_file"], point["threshold"])
            row = by_key[key]
            row["correct"] += point["correct"]
            row["n"] += point["n"]
            row["flops_sum_t"] += point["avg_flops_tflops"] * point["n"]
            row["regens"] += point["regen_ratio"] * point["n"]
            row["template"] = point

    results = []
    for (_checkpoint_file, threshold), row in by_key.items():
        n = max(row["n"], 1)
        template = row["template"] or {}
        results.append(
            {
                "checkpoint": template.get("checkpoint", ""),
                "checkpoint_file": template.get("checkpoint_file", ""),
                "checkpoint_dir": template.get("checkpoint_dir", ""),
                "checkpoint_name": template.get("checkpoint_name", ""),
                "lambda": template.get("lambda"),
                "seed": template.get("seed"),
                "checkpoint_kind": template.get("checkpoint_kind"),
                "epoch": template.get("epoch"),
                "threshold": threshold,
                "accuracy": row["correct"] / n * 100,
                "avg_flops_tflops": row["flops_sum_t"] / n,
                "regen_ratio": row["regens"] / n,
                "correct": row["correct"],
                "n": row["n"],
            }
        )
    return results


def _best_point_by_accuracy(points: List[Dict], srm_acc: float, lrm_acc: float) -> List[Dict]:
    """Deduplicate accuracy levels and keep points inside the SRM/LRM range."""
    low = min(srm_acc, lrm_acc) - 1e-9
    high = max(srm_acc, lrm_acc) + 1e-9
    best_by_acc = {}
    for point in points:
        accuracy = point["accuracy"]
        if accuracy < low or accuracy > high:
            continue
        key = round(accuracy, 6)
        previous = best_by_acc.get(key)
        if previous is None or (
            point["avg_flops_tflops"],
            point.get("regen_ratio", 0.0),
            point.get("checkpoint", ""),
            point.get("threshold", 0.0),
        ) < (
            previous["avg_flops_tflops"],
            previous.get("regen_ratio", 0.0),
            previous.get("checkpoint", ""),
            previous.get("threshold", 0.0),
        ):
            best_by_acc[key] = point
    return sorted(best_by_acc.values(), key=lambda p: p["accuracy"])


def _match_candidates_to_targets(candidates: List[Dict], targets: List[float]) -> List:
    """Match sorted candidates to sorted targets without crossing assignments."""
    if not candidates or not targets:
        return []

    choose_candidates = len(candidates) >= len(targets)
    left = targets if choose_candidates else candidates
    right = candidates if choose_candidates else targets
    n_left = len(left)
    n_right = len(right)
    inf = (float("inf"), float("inf"), float("inf"), float("inf"))
    dp = [[inf for _ in range(n_right + 1)] for _ in range(n_left + 1)]
    take = [[False for _ in range(n_right + 1)] for _ in range(n_left + 1)]
    dp[0] = [(0.0, 0.0, 0.0, 0.0) for _ in range(n_right + 1)]

    for i in range(1, n_left + 1):
        for j in range(1, n_right + 1):
            skip_cost = dp[i][j - 1]
            target = left[i - 1] if choose_candidates else right[j - 1]
            point = right[j - 1] if choose_candidates else left[i - 1]
            gap = abs(point["accuracy"] - target)
            prev = dp[i - 1][j - 1]
            use_cost = (
                max(prev[0], gap),
                prev[1] + gap,
                prev[2] + point["avg_flops_tflops"],
                prev[3] + point.get("regen_ratio", 0.0),
            )
            if use_cost < skip_cost:
                dp[i][j] = use_cost
                take[i][j] = True
            else:
                dp[i][j] = skip_cost

    pairs = []
    i, j = n_left, n_right
    while i > 0 and j > 0:
        if take[i][j]:
            target = left[i - 1] if choose_candidates else right[j - 1]
            point = right[j - 1] if choose_candidates else left[i - 1]
            pairs.append((float(target), point))
            i -= 1
            j -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def select_even_accuracy_points(dataset: str, baseline: Dict, points: List[Dict], n_targets: int = 8) -> Dict:
    """Select points closest to evenly spaced targets between SRM and LRM accuracy."""
    srm_acc = baseline["srm_acc"]
    lrm_acc = baseline["lrm_acc"]
    if not points or lrm_acc <= srm_acc:
        return {
            "dataset": dataset,
            "targets": [],
            "points": [],
            "limited_by_accuracy_granularity": True,
        }

    targets = np.linspace(srm_acc, lrm_acc, n_targets + 2)[1:-1]
    candidates = _best_point_by_accuracy(points, srm_acc, lrm_acc)
    matched = _match_candidates_to_targets(candidates, [float(x) for x in targets])
    selected = []

    for target, chosen in matched:
        selected.append(
            {
                "dataset": dataset,
                "target_acc": float(target),
                "actual_acc": float(chosen["accuracy"]),
                "acc_gap": float(abs(chosen["accuracy"] - target)),
                "avg_flops_tflops": float(chosen["avg_flops_tflops"]),
                "pct_lrm_flops": float(chosen["avg_flops_tflops"] / baseline["lrm_flops"] * 100),
                "regen_ratio": float(chosen.get("regen_ratio", 0.0)),
                "checkpoint": chosen.get("checkpoint", ""),
                "checkpoint_file": chosen.get("checkpoint_file", ""),
                "checkpoint_dir": chosen.get("checkpoint_dir", ""),
                "checkpoint_kind": chosen.get("checkpoint_kind"),
                "lambda": chosen.get("lambda"),
                "seed": chosen.get("seed"),
                "epoch": chosen.get("epoch"),
                "threshold": chosen.get("threshold"),
                "correct": chosen.get("correct"),
                "n": chosen.get("n"),
            }
        )

    return {
        "dataset": dataset,
        "targets": [float(x) for x in targets],
        "points": selected,
        "limited_by_accuracy_granularity": len(selected) < n_targets,
    }


def pareto_envelope(points: List[Dict]) -> List[Dict]:
    """Return the accuracy/FLOPs upper-left frontier."""
    if not points:
        return []

    best_by_flops = {}
    for point in points:
        key = round(point["avg_flops_tflops"], 10)
        previous = best_by_flops.get(key)
        if previous is None or point["accuracy"] > previous["accuracy"]:
            best_by_flops[key] = point

    frontier = []
    best_acc = -float("inf")
    for point in sorted(best_by_flops.values(), key=lambda p: p["avg_flops_tflops"]):
        if point["accuracy"] > best_acc + 1e-9:
            frontier.append(point)
            best_acc = point["accuracy"]
    return frontier


def find_acc_at_flops(curve: List[Dict], target_flops_tflops: float) -> Optional[float]:
    if not curve:
        return None
    pts = sorted(curve, key=lambda p: p["avg_flops_tflops"])
    for idx, point in enumerate(pts):
        if point["avg_flops_tflops"] >= target_flops_tflops:
            if idx == 0:
                return point["accuracy"]
            prev = pts[idx - 1]
            denom = point["avg_flops_tflops"] - prev["avg_flops_tflops"]
            frac = (target_flops_tflops - prev["avg_flops_tflops"]) / max(denom, 1e-20)
            return prev["accuracy"] + frac * (point["accuracy"] - prev["accuracy"])
    return pts[-1]["accuracy"]


def find_flops_at_acc(curve: List[Dict], target_acc: float) -> Optional[float]:
    if not curve:
        return None
    pts = sorted(curve, key=lambda p: p["avg_flops_tflops"])
    for point in pts:
        if point["accuracy"] >= target_acc:
            return point["avg_flops_tflops"]
    return None


def _metric(method: str, dataset: str, plot_data: Dict) -> Dict:
    baseline = plot_data["baselines"][dataset]
    flops_60 = baseline["lrm_flops"] * 0.6
    acc_98 = baseline["lrm_acc"] * 0.98

    if method == "SRM-Only":
        acc_at_60 = baseline["srm_acc"] if baseline["srm_flops"] <= flops_60 else None
        flops_at_98 = baseline["srm_flops"] if baseline["srm_acc"] >= acc_98 else None
    elif method == "LRM-Only":
        acc_at_60 = baseline["lrm_acc"] if baseline["lrm_flops"] <= flops_60 else None
        flops_at_98 = baseline["lrm_flops"] if baseline["lrm_acc"] >= acc_98 else None
    elif method == "Random Routing":
        curve = plot_data["random_curves"][dataset]
        acc_at_60 = find_acc_at_flops(curve, flops_60)
        flops_at_98 = find_flops_at_acc(curve, acc_98)
    elif method == "TRIM-Agg (PPO)":
        curve = plot_data["ppo_curves"][dataset]["ppo_agg"]
        acc_at_60 = find_acc_at_flops(curve, flops_60)
        flops_at_98 = find_flops_at_acc(curve, acc_98)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "acc_at_60": None if acc_at_60 is None else float(acc_at_60),
        "flops_at_98_tflops": None if flops_at_98 is None else float(flops_at_98),
        "flops_at_98_pct": None if flops_at_98 is None else float(flops_at_98 / baseline["lrm_flops"] * 100),
    }


def _assign_ranks(rows: List[Dict], datasets: List[str]) -> None:
    for dataset in datasets:
        acc_values = [
            (idx, row["metrics"][dataset]["acc_at_60"])
            for idx, row in enumerate(rows)
            if row["metrics"][dataset]["acc_at_60"] is not None
        ]
        flops_values = [
            (idx, row["metrics"][dataset]["flops_at_98_pct"])
            for idx, row in enumerate(rows)
            if row["metrics"][dataset]["flops_at_98_pct"] is not None
        ]

        for rank, (idx, _value) in enumerate(sorted(acc_values, key=lambda x: (-x[1], rows[x[0]]["method"]))[:2]):
            rows[idx]["metrics"][dataset]["acc_rank"] = "best" if rank == 0 else "second"
        for rank, (idx, _value) in enumerate(sorted(flops_values, key=lambda x: (x[1], rows[x[0]]["method"]))[:2]):
            rows[idx]["metrics"][dataset]["flops_rank"] = "best" if rank == 0 else "second"


def build_main_results(plot_data: Dict) -> Dict:
    datasets = plot_data["datasets"]
    method_extra_tokens = {
        "SRM-Only": "No",
        "LRM-Only": "No",
        "Random Routing": "Yes",
        "TRIM-Agg (PPO)": "Yes",
    }
    rows = []
    for method, extra_tokens in method_extra_tokens.items():
        rows.append(
            {
                "method": method,
                "extra_tokens": extra_tokens,
                "metrics": {dataset: _metric(method, dataset, plot_data) for dataset in datasets},
            }
        )
    _assign_ranks(rows, datasets)
    return {
        "caption": (
            "Main Results. Acc is measured at 60% of LRM-only FLOPs, and "
            "FLOPs denotes the computation required to reach 98% of LRM-only accuracy. "
            "The best result is bold, and the second-best result is underlined. "
            "Extra Tokens denotes whether the method requires additional token generation "
            "during the reasoning process."
        ),
        "datasets": datasets,
        "rows": rows,
    }


def _format_value(value: Optional[float], suffix: str, precision: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}{suffix}"


def _format_markdown(value: Optional[float], rank: Optional[str], suffix: str) -> str:
    text = _format_value(value, suffix)
    if text == "-":
        return text
    if rank == "best":
        return f"**{text}**"
    if rank == "second":
        return f"<u>{text}</u>"
    return text


def _format_latex(value: Optional[float], rank: Optional[str], suffix: str) -> str:
    text = _format_value(value, suffix)
    if text == "-":
        return text
    if rank == "best":
        return f"\\textbf{{{text}}}"
    if rank == "second":
        return f"\\underline{{{text}}}"
    return text


def render_main_results_markdown(main_results: Dict) -> str:
    headers = ["Method", "Extra Tokens"]
    for dataset in main_results["datasets"]:
        label = DS_LABELS.get(dataset, dataset)
        headers.extend([f"{label} Acc ↑", f"{label} FLOPs ↓"])

    lines = [main_results["caption"], "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in main_results["rows"]:
        cells = [row["method"], row["extra_tokens"]]
        for dataset in main_results["datasets"]:
            metric = row["metrics"][dataset]
            cells.append(_format_markdown(metric["acc_at_60"], metric.get("acc_rank"), "%"))
            cells.append(_format_markdown(metric["flops_at_98_pct"], metric.get("flops_rank"), "%"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_main_results_latex(main_results: Dict) -> str:
    col_spec = "ll" + "cc" * len(main_results["datasets"])
    header_1 = ["Method", "Extra Tokens"]
    header_2 = ["", ""]
    for dataset in main_results["datasets"]:
        label = DS_LABELS.get(dataset, dataset)
        header_1.append(f"\\multicolumn{{2}}{{c}}{{{label}}}")
        header_1.append("")
        header_2.extend(["Acc $\\uparrow$", "FLOPs $\\downarrow$"])

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_1).replace(" &  &", " &") + " \\\\",
        " & ".join(header_2) + " \\\\",
        "\\midrule",
    ]
    for row in main_results["rows"]:
        cells = [row["method"], row["extra_tokens"]]
        for dataset in main_results["datasets"]:
            metric = row["metrics"][dataset]
            cells.append(_format_latex(metric["acc_at_60"], metric.get("acc_rank"), "\\%"))
            cells.append(_format_latex(metric["flops_at_98_pct"], metric.get("flops_rank"), "\\%"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Main Results. Acc is measured at 60\\% of LRM-only FLOPs, and FLOPs denotes the computation required to reach 98\\% of LRM-only accuracy. The best result is bold, and the second-best result is underlined. Extra Tokens denotes whether the method requires additional token generation during the reasoning process.}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _subsample(points: List[Dict], n_markers: int) -> List[Dict]:
    if len(points) <= n_markers:
        return points
    indices = np.linspace(0, len(points) - 1, n_markers, dtype=int)
    return [points[i] for i in indices]


def _plot_figures(plot_data: Dict, output_dir: Path) -> None:
    for use_pct_axis, filename in [(False, "accuracy_vs_flops"), (True, "main_comparison")]:
        datasets = plot_data["datasets"]
        fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6))
        axes = np.atleast_1d(axes)
        for panel_idx, dataset in enumerate(plot_data["datasets"]):
            ax = axes[panel_idx]
            baseline = plot_data["baselines"][dataset]
            lrm_flops = baseline["lrm_flops"]

            def x_value(flops_tflops: float) -> float:
                return flops_tflops / lrm_flops * 100 if use_pct_axis else flops_tflops

            random = plot_data["random_curves"][dataset]
            ax.plot(
                [x_value(p["avg_flops_tflops"]) for p in random[:: max(len(random) // 10, 1)]],
                [p["accuracy"] for p in random[:: max(len(random) // 10, 1)]],
                linestyle=":",
                marker="D",
                markersize=5,
                color="#888888",
                linewidth=1.5,
                label="Random Routing",
                alpha=0.7,
                zorder=2,
            )

            ppo_curve = plot_data["ppo_curves"][dataset]["ppo_agg"]
            if ppo_curve:
                marker_points = _subsample(ppo_curve, 10)
                ax.plot(
                    [x_value(p["avg_flops_tflops"]) for p in ppo_curve],
                    [p["accuracy"] for p in ppo_curve],
                    linestyle="--",
                    color="#2196F3",
                    linewidth=2.5,
                    label="_nolegend_",
                    zorder=4,
                )
                ax.plot(
                    [x_value(p["avg_flops_tflops"]) for p in marker_points],
                    [p["accuracy"] for p in marker_points],
                    linestyle="",
                    marker="s",
                    markersize=8,
                    color="#2196F3",
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label="TRIM-Agg (PPO)",
                    zorder=5,
                )

            ax.scatter(
                [x_value(baseline["srm_flops"])],
                [baseline["srm_acc"]],
                marker="*",
                s=350,
                facecolor="#FF5722",
                edgecolor="black",
                linewidth=1.2,
                zorder=6,
                label="SRM-Only",
            )
            ax.scatter(
                [x_value(baseline["lrm_flops"])],
                [baseline["lrm_acc"]],
                marker="*",
                s=350,
                facecolor="#4CAF50",
                edgecolor="black",
                linewidth=1.2,
                zorder=6,
                label="LRM-Only",
            )

            flops_60 = baseline["lrm_flops"] * 0.6
            acc_98 = baseline["lrm_acc"] * 0.98
            ax.axvline(x=x_value(flops_60), color="#9E9E9E", linestyle="-.", alpha=0.45, linewidth=1)
            ax.axhline(y=acc_98, color="#9E9E9E", linestyle="-.", alpha=0.45, linewidth=1)

            y_values = [baseline["srm_acc"], baseline["lrm_acc"]]
            y_values.extend([p["accuracy"] for p in ppo_curve])
            y_pad = max((max(y_values) - min(y_values)) * 0.12, 3)
            y_min = max(min(y_values) - y_pad, 0)
            y_max = min(max(y_values) + y_pad, 100)
            ax.set_ylim(y_min, y_max)

            x_values = [baseline["srm_flops"], baseline["lrm_flops"]]
            x_values.extend([p["avg_flops_tflops"] for p in ppo_curve])
            if use_pct_axis:
                ax.set_xlim(-2, max(110, max(x_value(x) for x in x_values) * 1.05))
                ax.set_xlabel("% of LRM FLOPs", fontsize=16)
            else:
                x_max = max(x_values) * 1.08
                ax.set_xlim(-x_max * 0.02, x_max)
                ax.set_xlabel("TFLOPs per query", fontsize=16)

            ax.text(
                x_value(flops_60),
                y_min + (y_max - y_min) * 0.05,
                "60%\nLRM FLOPs",
                fontsize=8,
                color="#757575",
                ha="center",
                va="bottom",
                style="italic",
            )
            ax.text(
                x_value(baseline["lrm_flops"] * 0.72),
                acc_98,
                "98% LRM Acc",
                fontsize=8,
                color="#757575",
                ha="center",
                va="bottom",
                style="italic",
            )

            ax.set_title(f"{DS_LABELS.get(dataset, dataset)} (n={baseline['n']})", fontsize=20, fontweight="bold")
            if panel_idx == 0:
                ax.set_ylabel("Accuracy (%)", fontsize=18)
            ax.tick_params(axis="both", labelsize=13, width=1.5, length=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.45)

        handles, labels = axes[0].get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label != "_nolegend_" and label not in unique:
                unique[label] = handle
        fig.legend(
            list(unique.values()),
            list(unique.keys()),
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=16,
            bbox_to_anchor=(0.5, 1.05),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        for ext in ["png", "pdf"]:
            out = output_dir / f"{filename}.{ext}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved -> {out}")
        plt.close(fig)


def build_plot_data(args: argparse.Namespace) -> Dict:
    datasets = parse_dataset_names(args.datasets)
    episode_paths = {
        "math500": Path(args.math500_episodes),
        "aime2025": Path(args.aime2025_episodes),
    }
    groups = load_episode_groups(episode_paths, datasets)
    baselines = compute_baselines(groups, datasets)
    thresholds = probability_thresholds()
    checkpoint_patterns = [p.strip() for p in args.checkpoint_glob.split(",") if p.strip()]
    checkpoints = expand_checkpoint_globs(checkpoint_patterns)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {checkpoint_patterns}")

    print(f"Found {len(checkpoints)} checkpoints", flush=True)
    for checkpoint in checkpoints:
        print(f"  {checkpoint.parent.name}", flush=True)

    raw_curves = {dataset: [] for dataset in _base_datasets_for(datasets)}
    for dataset in _base_datasets_for(datasets):
        path = episode_paths[dataset]
        print(f"\n=== PPO threshold sweep: {dataset} ===", flush=True)
        for ckpt_idx, checkpoint in enumerate(checkpoints, 1):
            print(f"  [{ckpt_idx}/{len(checkpoints)}] {checkpoint.parent.name}", flush=True)
            curve = evaluate_policy_threshold_curve(path, checkpoint, thresholds, args.device)
            raw_curves[dataset].extend(curve)

    if "all" in datasets:
        raw_curves["all"] = aggregate_all_dataset_points(raw_curves)
    ppo_curves = {
        dataset: {"ppo_agg": pareto_envelope(raw_curves[dataset])}
        for dataset in datasets
    }
    random_curves = {dataset: random_curve(baselines[dataset]) for dataset in datasets}

    plot_data = {
        "datasets": datasets,
        "baselines": baselines,
        "random_curves": random_curves,
        "ppo_curves": ppo_curves,
        "raw_ppo_points": raw_curves,
        "source_files": {k: str(v) for k, v in episode_paths.items()},
        "checkpoint_patterns": checkpoint_patterns,
        "thresholds": thresholds,
    }
    plot_data["main_results"] = build_main_results(plot_data)
    plot_data["selected_points"] = {
        dataset: select_even_accuracy_points(
            dataset,
            baselines[dataset],
            raw_curves[dataset],
            n_targets=args.n_selected_points,
        )
        for dataset in datasets
    }
    return plot_data


def write_outputs(plot_data: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_figures(plot_data, output_dir)

    plot_data_path = output_dir / "plot_data.json"
    with open(plot_data_path, "w") as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {plot_data_path}")

    main_results = plot_data["main_results"]
    main_results_json = output_dir / "main_results.json"
    with open(main_results_json, "w") as f:
        json.dump(main_results, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {main_results_json}")

    main_results_md = output_dir / "main_results.md"
    main_results_md.write_text(render_main_results_markdown(main_results), encoding="utf-8")
    print(f"Saved -> {main_results_md}")

    main_results_tex = output_dir / "main_results.tex"
    main_results_tex.write_text(render_main_results_latex(main_results), encoding="utf-8")
    print(f"Saved -> {main_results_tex}")

    for dataset, selected in plot_data.get("selected_points", {}).items():
        json_path = output_dir / f"selected_points_{dataset}.json"
        with open(json_path, "w") as f:
            json.dump(selected, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {json_path}")

        csv_path = output_dir / f"selected_points_{dataset}.csv"
        fieldnames = [
            "dataset", "target_acc", "actual_acc", "acc_gap",
            "avg_flops_tflops", "pct_lrm_flops", "regen_ratio",
            "checkpoint", "checkpoint_file", "checkpoint_dir", "checkpoint_kind",
            "lambda", "seed", "epoch", "threshold", "correct", "n",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected.get("points", []):
                writer.writerow({key: row.get(key) for key in fieldnames})
        print(f"Saved -> {csv_path}")

    summary_lines = [
        "# TRIM-Agg Point Search Summary",
        "",
        f"Checkpoint patterns: {', '.join(plot_data.get('checkpoint_patterns', []))}",
        "",
    ]
    for dataset in plot_data["datasets"]:
        selected = plot_data.get("selected_points", {}).get(dataset, {})
        points = selected.get("points", [])
        summary_lines.append(f"## {DS_LABELS.get(dataset, dataset)}")
        summary_lines.append(f"- Selected points: {len(points)}")
        summary_lines.append(f"- Limited by accuracy granularity: {selected.get('limited_by_accuracy_granularity')}")
        for row in points:
            summary_lines.append(
                "- target={target:.2f}, actual={actual:.2f}, flops={flops:.2f}T "
                "({pct:.1f}% LRM), regen={regen:.2%}, ckpt={ckpt}, th={th}".format(
                    target=row["target_acc"],
                    actual=row["actual_acc"],
                    flops=row["avg_flops_tflops"],
                    pct=row["pct_lrm_flops"],
                    regen=row["regen_ratio"],
                    ckpt=row["checkpoint"],
                    th=row["threshold"],
                )
            )
        summary_lines.append("")

    summary_path = output_dir / "search_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved -> {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local TRIM-Agg baseline plots")
    parser.add_argument(
        "--datasets",
        default="math500,aime2025",
        help="Comma-separated datasets to evaluate: math500, aime2025, all. "
        "If math500 and aime2025 are both requested, all is added automatically.",
    )
    parser.add_argument("--math500_episodes", default=str(DEFAULT_EPISODES["math500"]))
    parser.add_argument("--aime2025_episodes", default=str(DEFAULT_EPISODES["aime2025"]))
    parser.add_argument("--checkpoint_glob", default=DEFAULT_CHECKPOINT_GLOB)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_selected_points", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_data = build_plot_data(args)
    write_outputs(plot_data, Path(args.output_dir))


if __name__ == "__main__":
    main()
