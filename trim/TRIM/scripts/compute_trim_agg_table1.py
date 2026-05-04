#!/usr/bin/env python3
"""
Compute a Table-1-style summary for TRIM-Agg only.

For each dataset, the script expects:
  - Mw baseline metrics (draft-only)
  - Ms baseline metrics (target-only)
  - A TRIM-Agg sweep directory containing */eval_metrics.jsonl

It outputs one summary row per dataset with:
  - CPT(50%), CPT(80%), CPT(95%)
  - avg ΔIBC over 100 equally spaced PGR targets

By default CPT and ΔIBC are computed on the Pareto frontier using linear
interpolation in cost-accuracy space. This matches the spirit of the paper
better than choosing only the raw observed points when the sweep is sparse.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Point:
    sweep_id: str
    accuracy: float
    avg_target_tokens_per_question: float
    target_token_frac: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Table-1-style TRIM-Agg summary.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest describing datasets, baselines, and sweep dirs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to <manifest_dir>/table1_trim_agg",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=100,
        help="Number of equally spaced PGR targets used for average ΔIBC.",
    )
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    if path.suffix == ".json":
        return json.loads(text)

    rows = [line for line in text.splitlines() if line.strip()]
    return json.loads(rows[-1])


def load_baseline(spec: Any) -> dict[str, float]:
    if isinstance(spec, dict):
        data = spec
    elif isinstance(spec, str):
        data = load_json_or_jsonl(Path(spec))
    else:
        raise TypeError(f"Unsupported baseline spec: {spec!r}")

    return {
        "accuracy": float(data["accuracy"]),
        "avg_target_tokens_per_question": float(data["avg_target_tokens_per_question"]),
        "target_token_frac": float(data.get("target_token_frac", 0.0)),
    }


def load_sweep_points(sweep_dir: Path) -> list[Point]:
    points: list[Point] = []
    for metrics_path in sorted(sweep_dir.glob("*/eval_metrics.jsonl")):
        data = load_json_or_jsonl(metrics_path)
        points.append(
            Point(
                sweep_id=metrics_path.parent.name,
                accuracy=float(data["accuracy"]),
                avg_target_tokens_per_question=float(data["avg_target_tokens_per_question"]),
                target_token_frac=float(data.get("target_token_frac", 0.0)),
            )
        )

    if not points:
        raise FileNotFoundError(f"No eval_metrics.jsonl found under {sweep_dir}")

    return points


def pareto_front(points: list[Point]) -> list[Point]:
    sorted_points = sorted(
        points,
        key=lambda p: (p.avg_target_tokens_per_question, -p.accuracy, p.sweep_id),
    )
    frontier: list[Point] = []
    best_acc = float("-inf")
    for point in sorted_points:
        if point.accuracy > best_acc:
            frontier.append(point)
            best_acc = point.accuracy
    return frontier


def interpolate_cost_for_accuracy(frontier: list[Point], target_accuracy: float) -> float | None:
    if not frontier:
        return None

    if target_accuracy <= frontier[0].accuracy:
        return frontier[0].avg_target_tokens_per_question

    for left, right in zip(frontier, frontier[1:]):
        if left.accuracy <= target_accuracy <= right.accuracy:
            if right.accuracy == left.accuracy:
                return min(
                    left.avg_target_tokens_per_question,
                    right.avg_target_tokens_per_question,
                )
            frac = (target_accuracy - left.accuracy) / (right.accuracy - left.accuracy)
            return left.avg_target_tokens_per_question + frac * (
                right.avg_target_tokens_per_question - left.avg_target_tokens_per_question
            )

    if target_accuracy <= frontier[-1].accuracy:
        return frontier[-1].avg_target_tokens_per_question
    return None


def format_cpt(cost: float | None, ms_cost: float) -> str:
    if cost is None:
        return "unreachable"
    if ms_cost <= 0:
        return f"{cost:.2f} (n/a)"
    return f"{cost:.2f} ({100.0 * cost / ms_cost:.2f}%)"


def compute_avg_delta_ibc(
    frontier: list[Point],
    mw_acc: float,
    ms_acc: float,
    ms_cost: float,
    num_regions: int,
) -> float | None:
    if ms_cost <= 0 or ms_acc <= mw_acc:
        return None

    ibc_base = (ms_acc - mw_acc) / ms_cost
    if ibc_base <= 0:
        return None

    deltas: list[float] = []
    for idx in range(1, num_regions + 1):
        pgr_target = idx / num_regions
        target_accuracy = mw_acc + pgr_target * (ms_acc - mw_acc)
        cost = interpolate_cost_for_accuracy(frontier, target_accuracy)
        if cost is None or cost <= 0:
            continue
        ibc = (target_accuracy - mw_acc) / cost
        delta = (ibc - ibc_base) / ibc_base
        deltas.append(delta)

    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def compute_dataset_row(dataset_cfg: dict[str, Any], num_regions: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_key = dataset_cfg["dataset_key"]
    dataset_label = dataset_cfg.get("dataset_label", dataset_key)
    method_name = dataset_cfg.get("method", "TRIM-Agg")

    mw = load_baseline(dataset_cfg["mw_baseline"])
    ms = load_baseline(dataset_cfg["ms_baseline"])
    sweep_dir = Path(dataset_cfg["sweep_dir"])
    sweep_points = load_sweep_points(sweep_dir)
    frontier = pareto_front(sweep_points)

    mw_acc = mw["accuracy"]
    ms_acc = ms["accuracy"]
    ms_cost = ms["avg_target_tokens_per_question"]
    gap = ms_acc - mw_acc

    long_rows: list[dict[str, Any]] = []
    for point in sweep_points:
        pgr = None
        if gap > 0:
            pgr = (point.accuracy - mw_acc) / gap
        ibc = None
        if point.avg_target_tokens_per_question > 0:
            ibc = (point.accuracy - mw_acc) / point.avg_target_tokens_per_question
        delta_ibc = None
        if ibc is not None and ms_cost > 0 and gap > 0:
            ibc_base = gap / ms_cost
            if ibc_base > 0:
                delta_ibc = (ibc - ibc_base) / ibc_base

        long_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": dataset_label,
                "method": method_name,
                "sweep_id": point.sweep_id,
                "accuracy": point.accuracy,
                "avg_target_tokens_per_question": point.avg_target_tokens_per_question,
                "target_token_frac": point.target_token_frac,
                "normalized_cost_vs_ms": (
                    point.avg_target_tokens_per_question / ms_cost if ms_cost > 0 else None
                ),
                "pgr": pgr,
                "ibc": ibc,
                "delta_ibc": delta_ibc,
                "on_pareto_front": any(point.sweep_id == x.sweep_id for x in frontier),
            }
        )

    cpt_costs: dict[int, float | None] = {}
    for pct in (50, 80, 95):
        if gap <= 0:
            cpt_costs[pct] = None
            continue
        target_accuracy = mw_acc + (pct / 100.0) * gap
        cpt_costs[pct] = interpolate_cost_for_accuracy(frontier, target_accuracy)

    avg_delta_ibc = compute_avg_delta_ibc(frontier, mw_acc, ms_acc, ms_cost, num_regions)

    summary_row = {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "method": method_name,
        "mw_accuracy": mw_acc,
        "ms_accuracy": ms_acc,
        "ms_avg_target_tokens_per_question": ms_cost,
        "num_sweep_points": len(sweep_points),
        "num_pareto_points": len(frontier),
        "cpt50_cost": cpt_costs[50],
        "cpt80_cost": cpt_costs[80],
        "cpt95_cost": cpt_costs[95],
        "cpt50_display": format_cpt(cpt_costs[50], ms_cost),
        "cpt80_display": format_cpt(cpt_costs[80], ms_cost),
        "cpt95_display": format_cpt(cpt_costs[95], ms_cost),
        "avg_delta_ibc": avg_delta_ibc,
    }
    return summary_row, long_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Dataset | Method | CPT(50%) | CPT(80%) | CPT(95%) | avg ΔIBC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        avg_delta_ibc = "n/a" if row["avg_delta_ibc"] is None else f'{row["avg_delta_ibc"]:.2f}'
        lines.append(
            "| "
            f'{row["dataset_label"]} | {row["method"]} | '
            f'{row["cpt50_display"]} | {row["cpt80_display"]} | {row["cpt95_display"]} | {avg_delta_ibc} |'
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())

    output_dir = args.output_dir or (args.manifest.parent / "table1_trim_agg")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for dataset_cfg in manifest["datasets"]:
        summary_row, dataset_long_rows = compute_dataset_row(dataset_cfg, args.num_regions)
        summary_rows.append(summary_row)
        long_rows.extend(dataset_long_rows)

    write_csv(
        output_dir / "trim_agg_table1_summary.csv",
        summary_rows,
        [
            "dataset_key",
            "dataset_label",
            "method",
            "mw_accuracy",
            "ms_accuracy",
            "ms_avg_target_tokens_per_question",
            "num_sweep_points",
            "num_pareto_points",
            "cpt50_cost",
            "cpt80_cost",
            "cpt95_cost",
            "cpt50_display",
            "cpt80_display",
            "cpt95_display",
            "avg_delta_ibc",
        ],
    )
    write_csv(
        output_dir / "trim_agg_table1_long.csv",
        long_rows,
        [
            "dataset_key",
            "dataset_label",
            "method",
            "sweep_id",
            "accuracy",
            "avg_target_tokens_per_question",
            "target_token_frac",
            "normalized_cost_vs_ms",
            "pgr",
            "ibc",
            "delta_ibc",
            "on_pareto_front",
        ],
    )
    write_markdown(output_dir / "trim_agg_table1_summary.md", summary_rows)

    print(f"Wrote {output_dir / 'trim_agg_table1_summary.csv'}")
    print(f"Wrote {output_dir / 'trim_agg_table1_long.csv'}")
    print(f"Wrote {output_dir / 'trim_agg_table1_summary.md'}")


if __name__ == "__main__":
    main()
