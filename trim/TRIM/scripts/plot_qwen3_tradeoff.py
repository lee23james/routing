#!/usr/bin/env python3
"""
Plot trade-off curves for the current Qwen3 TRIM-Agg sweeps.

Outputs:
  1. Accuracy vs avg target tokens per question
  2. Accuracy vs target-token fraction
  3. A CSV summary of all sweep points plus Pareto-front membership

The script reads the finalized eval metrics from:
  - rlpolicy_results_qwen3_math1k_sweep
  - rlpolicy_results_qwen3_aime_official_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepSpec:
    dataset_key: str
    dataset_label: str
    results_dir: Path
    color: str


SPECS = (
    SweepSpec(
        dataset_key="MATH",
        dataset_label="math500/test_100",
        results_dir=ROOT_DIR / "rlpolicy_results_qwen3_math1k_sweep",
        color="#0f766e",
    ),
    SweepSpec(
        dataset_key="AIME",
        dataset_label="aime/test",
        results_dir=ROOT_DIR / "rlpolicy_results_qwen3_aime_official_sweep",
        color="#b45309",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen3 TRIM-Agg trade-off curves.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "plots" / "qwen3_tradeoff",
        help="Directory for generated figures and CSV.",
    )
    return parser.parse_args()


def load_points(spec: SweepSpec) -> list[dict]:
    points: list[dict] = []
    for metrics_path in sorted(spec.results_dir.glob("*/eval_metrics.jsonl")):
        rows = [line for line in metrics_path.read_text().splitlines() if line.strip()]
        if not rows:
            continue

        metrics = json.loads(rows[-1])
        cost_tag = metrics_path.parent.name
        point = {
            "dataset_key": spec.dataset_key,
            "dataset_label": spec.dataset_label,
            "cost_per_token_label": cost_tag,
            "cost_per_token_value": float(cost_tag),
            "accuracy": float(metrics["accuracy"]),
            "accuracy_pct": float(metrics["accuracy"]) * 100.0,
            "avg_target_tokens_per_question": float(metrics["avg_target_tokens_per_question"]),
            "target_token_frac": float(metrics["target_token_frac"]),
            "target_token_pct": float(metrics["target_token_frac"]) * 100.0,
            "avg_target_calls_per_question": float(metrics["avg_target_calls_per_question"]),
        }
        points.append(point)

    if not points:
        raise FileNotFoundError(f"No eval metrics found under {spec.results_dir}")

    return points


def compute_pareto_front(points: Iterable[dict], x_key: str, y_key: str) -> list[str]:
    sorted_points = sorted(points, key=lambda p: (p[x_key], -p[y_key], p["cost_per_token_value"]))
    frontier_ids: list[str] = []
    best_y = float("-inf")
    for point in sorted_points:
        if point[y_key] > best_y:
            frontier_ids.append(point["cost_per_token_label"])
            best_y = point[y_key]
    return frontier_ids


def write_summary_csv(points_by_dataset: dict[str, list[dict]], output_path: Path) -> None:
    fieldnames = [
        "dataset_key",
        "dataset_label",
        "cost_per_token_label",
        "cost_per_token_value",
        "accuracy",
        "accuracy_pct",
        "avg_target_tokens_per_question",
        "target_token_frac",
        "target_token_pct",
        "avg_target_calls_per_question",
        "pareto_by_target_tokens",
        "pareto_by_target_frac",
    ]

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_key in ("MATH", "AIME"):
            for point in sorted(points_by_dataset[dataset_key], key=lambda p: p["cost_per_token_value"]):
                writer.writerow(point)


def _annotation_offsets(n: int) -> list[tuple[int, int]]:
    base = [
        (8, 8),
        (-10, 10),
        (8, -12),
        (-12, -12),
        (10, 16),
        (-14, 16),
        (12, -18),
        (-16, -18),
    ]
    return [base[i % len(base)] for i in range(n)]


def plot_accuracy_vs_target_tokens(points_by_dataset: dict[str, list[dict]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, spec in zip(axes, SPECS):
        points = sorted(points_by_dataset[spec.dataset_key], key=lambda p: p["avg_target_tokens_per_question"])
        frontier = [p for p in points if p["pareto_by_target_tokens"] == "yes"]

        ax.scatter(
            [p["avg_target_tokens_per_question"] for p in points],
            [p["accuracy_pct"] for p in points],
            s=80,
            color=spec.color,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.plot(
            [p["avg_target_tokens_per_question"] for p in frontier],
            [p["accuracy_pct"] for p in frontier],
            color=spec.color,
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )

        for point, offset in zip(points, _annotation_offsets(len(points))):
            ax.annotate(
                point["cost_per_token_label"],
                (point["avg_target_tokens_per_question"], point["accuracy_pct"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                color="#111827",
            )

        ax.set_title(f"{spec.dataset_key} ({spec.dataset_label})")
        ax.set_xlabel("Avg target tokens / question")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    fig.suptitle("TRIM-Agg Trade-off: Accuracy vs Avg Target Tokens", fontsize=14)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_target_frac(points_by_dataset: dict[str, list[dict]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, spec in zip(axes, SPECS):
        points = sorted(points_by_dataset[spec.dataset_key], key=lambda p: p["target_token_pct"])
        frontier = [p for p in points if p["pareto_by_target_frac"] == "yes"]

        ax.scatter(
            [p["target_token_pct"] for p in points],
            [p["accuracy_pct"] for p in points],
            s=80,
            color=spec.color,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.plot(
            [p["target_token_pct"] for p in frontier],
            [p["accuracy_pct"] for p in frontier],
            color=spec.color,
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )

        for point, offset in zip(points, _annotation_offsets(len(points))):
            ax.annotate(
                point["cost_per_token_label"],
                (point["target_token_pct"], point["accuracy_pct"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                color="#111827",
            )

        xmax = max(p["target_token_pct"] for p in points)
        ax.set_xlim(left=-0.5, right=max(5.0, xmax * 1.15))
        ax.set_title(f"{spec.dataset_key} ({spec.dataset_label})")
        ax.set_xlabel("Target token fraction (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(True, alpha=0.25, linewidth=0.6)

    fig.suptitle("TRIM-Agg Trade-off: Accuracy vs Target Token Fraction", fontsize=14)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points_by_dataset = {spec.dataset_key: load_points(spec) for spec in SPECS}

    for spec in SPECS:
        points = points_by_dataset[spec.dataset_key]
        pareto_tokens = set(compute_pareto_front(points, "avg_target_tokens_per_question", "accuracy"))
        pareto_frac = set(compute_pareto_front(points, "target_token_pct", "accuracy"))

        for point in points:
            point["pareto_by_target_tokens"] = "yes" if point["cost_per_token_label"] in pareto_tokens else "no"
            point["pareto_by_target_frac"] = "yes" if point["cost_per_token_label"] in pareto_frac else "no"

    summary_csv = args.output_dir / "tradeoff_points.csv"
    tokens_plot = args.output_dir / "accuracy_vs_avg_target_tokens.png"
    frac_plot = args.output_dir / "accuracy_vs_target_token_frac.png"

    write_summary_csv(points_by_dataset, summary_csv)
    plot_accuracy_vs_target_tokens(points_by_dataset, tokens_plot)
    plot_accuracy_vs_target_frac(points_by_dataset, frac_plot)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {tokens_plot}")
    print(f"Wrote {frac_plot}")


if __name__ == "__main__":
    main()
