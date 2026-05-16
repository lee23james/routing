#!/usr/bin/env python3
"""Summarize episode token lengths and context saturation."""

import argparse
import json
import statistics
from pathlib import Path


def _percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def summarize(path: Path, cap: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {
        "path": str(path),
        "n": len(rows),
        "max_new_tokens": cap,
        "models": {},
    }
    for side in ("srm", "lrm"):
        toks = [int(row.get(f"{side}_total_tokens", 0) or 0) for row in rows]
        if not toks:
            continue
        near_95 = sum(t >= 0.95 * cap for t in toks)
        near_strict = sum(t >= cap - 64 for t in toks)
        result["models"][side] = {
            "avg_tokens": statistics.mean(toks),
            "median_tokens": statistics.median(toks),
            "p90_tokens": _percentile(toks, 0.90),
            "max_tokens": max(toks),
            "near_95pct_cap_count": near_95,
            "near_95pct_cap_rate": near_95 / len(toks),
            "near_cap_minus_64_count": near_strict,
            "near_cap_minus_64_rate": near_strict / len(toks),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("episodes", nargs="+")
    args = parser.parse_args()

    summaries = [summarize(Path(path), args.max_new_tokens) for path in args.episodes]
    for item in summaries:
        print(f"\n# {item['path']} n={item['n']} cap={item['max_new_tokens']}")
        for side, stats in item["models"].items():
            print(
                f"{side.upper()}: avg={stats['avg_tokens']/1000:.2f}K "
                f"median={stats['median_tokens']/1000:.2f}K "
                f"p90={stats['p90_tokens']/1000:.2f}K "
                f"max={stats['max_tokens']/1000:.2f}K "
                f">=95%cap={stats['near_95pct_cap_count']}/{item['n']} "
                f"({stats['near_95pct_cap_rate']:.1%}) "
                f">=cap-64={stats['near_cap_minus_64_count']}/{item['n']} "
                f"({stats['near_cap_minus_64_rate']:.1%})"
            )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
