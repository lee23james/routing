#!/usr/bin/env python3
"""Utilities for fast TRIM-Agg AIME lambda probes.

This module intentionally avoids importing TRIM_Agg.py so it can be used for
cheap orchestration, tests, and summaries without initializing torch/vLLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Iterable


def format_cost_tag(cost: float) -> str:
    dec = Decimal(str(cost)).normalize()
    mantissa, exp = f"{dec:.15E}".split("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def official_cost_tag(cost: str) -> str:
    """Return TRIM_Agg.py's float-based tag for compatibility."""
    value = float(cost)
    mantissa, exp = f"{value:.15e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def parse_cost_grid(costs_csv: str) -> list[str]:
    costs: list[str] = []
    for raw in costs_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        costs.append(format_cost_tag(raw))
    if not costs:
        raise ValueError("empty cost grid")
    return costs


def metric_path(result_root: Path, cost: str) -> Path:
    return result_root / format_cost_tag(cost) / "eval_metrics.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_episode_baseline(path: Path, limit: int | None = None) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    n = len(rows)
    if n == 0:
        raise ValueError(f"no episode rows found in {path}")
    return {
        "n": n,
        "srm_acc": sum(bool(row.get("srm_correct")) for row in rows) / n,
        "lrm_acc": sum(bool(row.get("lrm_correct")) for row in rows) / n,
        "lrm_avg_tokens": sum(float(row.get("lrm_total_tokens", 0.0)) for row in rows) / n,
    }


def _write_limited_jsonl(src: Path, dst: Path, limit: int | None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                break
            out.write(line)
            count += 1
    return count


def prepare_data(source_root: Path, output_root: Path, train_limit: int | None, eval_limit: int | None) -> dict:
    """Create a TRIM-compatible AIME data_dir, optionally limited for speed."""
    src_train = source_root / "aime" / "train.jsonl"
    src_test = source_root / "aime" / "test.jsonl"
    dst_train = output_root / "aime" / "train.jsonl"
    dst_test = output_root / "aime" / "test.jsonl"

    if train_limit is None and eval_limit is None:
        dst_train.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_train, dst_train)
        shutil.copy2(src_test, dst_test)
        return {"train": sum(1 for _ in dst_train.open()), "test": sum(1 for _ in dst_test.open())}

    train_n = _write_limited_jsonl(src_train, dst_train, train_limit)
    test_n = _write_limited_jsonl(src_test, dst_test, eval_limit)
    return {"train": train_n, "test": test_n}


def _load_last_metric(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    if not rows:
        return None
    return rows[-1]


def _load_metric_with_compat(result_root: Path, cost: str) -> dict | None:
    canonical = metric_path(result_root, cost)
    metrics = _load_last_metric(canonical)
    if metrics is not None:
        return metrics
    compat = result_root / official_cost_tag(cost) / "eval_metrics.jsonl"
    if compat == canonical:
        return None
    return _load_last_metric(compat)


def summarize_metrics(
    result_root: Path,
    costs: Iterable[str],
    srm_acc: float,
    lrm_acc: float,
    lrm_avg_tokens: float,
) -> list[dict]:
    rows: list[dict] = []
    gap = lrm_acc - srm_acc
    for cost in costs:
        tag = format_cost_tag(cost)
        metrics = _load_metric_with_compat(result_root, tag)
        if metrics is None:
            rows.append({"cost": tag, "status": "missing"})
            continue

        acc = float(metrics.get("accuracy", 0.0))
        target_tokens = float(metrics.get("avg_target_tokens_per_question", 0.0))
        cpt = target_tokens / lrm_avg_tokens if lrm_avg_tokens > 0 else 0.0
        pgr = (acc - srm_acc) / gap if gap > 0 else 0.0
        row = {
            "cost": tag,
            "status": "done",
            "accuracy": acc,
            "num_correct": metrics.get("num_correct"),
            "avg_target_tokens_per_question": target_tokens,
            "cpt": cpt,
            "pgr": pgr,
            "avg_target_calls_per_question": metrics.get("avg_target_calls_per_question"),
            "target_token_frac_raw": metrics.get("target_token_frac"),
        }
        rows.append(row)
    return rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _optional_limit(raw: str) -> int | None:
    if raw.lower() in {"all", "none", "full"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("limit must be positive, or 'all'")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIM-Agg quick-probe helpers")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    prep = subparsers.add_parser("prepare-data")
    prep.add_argument("--source_root", type=Path, required=True)
    prep.add_argument("--output_root", type=Path, required=True)
    prep.add_argument("--train_limit", default="all")
    prep.add_argument("--eval_limit", default="all")

    summary = subparsers.add_parser("summary")
    summary.add_argument("--result_root", type=Path, required=True)
    summary.add_argument("--costs", required=True)
    summary.add_argument("--srm_acc", type=float, required=True)
    summary.add_argument("--lrm_acc", type=float, required=True)
    summary.add_argument("--lrm_avg_tokens", type=float, required=True)
    summary.add_argument("--episode_baseline", type=Path)
    summary.add_argument("--eval_limit", default="all")
    summary.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "prepare-data":
        result = prepare_data(
            args.source_root,
            args.output_root,
            _optional_limit(args.train_limit),
            _optional_limit(args.eval_limit),
        )
        print(json.dumps(result, ensure_ascii=False))
    elif args.cmd == "summary":
        srm_acc = args.srm_acc
        lrm_acc = args.lrm_acc
        lrm_avg_tokens = args.lrm_avg_tokens
        if args.episode_baseline is not None:
            baseline = compute_episode_baseline(args.episode_baseline, _optional_limit(args.eval_limit))
            srm_acc = baseline["srm_acc"]
            lrm_acc = baseline["lrm_acc"]
            lrm_avg_tokens = baseline["lrm_avg_tokens"]
        rows = summarize_metrics(
            args.result_root,
            parse_cost_grid(args.costs),
            srm_acc,
            lrm_acc,
            lrm_avg_tokens,
        )
        write_summary(rows, args.out)
        for row in rows:
            if row["status"] != "done":
                print(f"{row['cost']:>7}  missing")
                continue
            print(
                f"{row['cost']:>7}  acc={row['accuracy']:.4f}  "
                f"target_tok/q={row['avg_target_tokens_per_question']:.1f}  "
                f"CPT~{row['cpt']:.3f}  PGR~{row['pgr']:.3f}"
            )


if __name__ == "__main__":
    main()
