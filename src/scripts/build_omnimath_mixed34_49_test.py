#!/usr/bin/env python3
"""Build a mixed OmniMath 3-4/4-9 episode test file."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import load_jsonl


def _difficulty_bucket(row: Dict) -> str:
    diff = float(row["difficulty"])
    bucket = int(diff)
    return f"[{bucket},{bucket + 1})"


def _selection_record(row: Dict) -> Dict:
    return {
        "id": row.get("id"),
        "source_id": row.get("source_id", row.get("id")),
        "source_index": row.get("source_index"),
        "difficulty": row.get("difficulty"),
    }


def build_mixed_episodes(
    *,
    omnimath34_episodes: Path,
    omnimath49_episodes: Path,
    output: Path,
    manifest: Path,
    seed: int,
    n_34: int,
    n_49: int,
) -> Dict:
    rng = random.Random(seed)
    rows_34 = load_jsonl(str(omnimath34_episodes))
    rows_49 = load_jsonl(str(omnimath49_episodes))

    if len(rows_34) < n_34:
        raise ValueError(f"{omnimath34_episodes} has {len(rows_34)} rows; need {n_34}")
    if len(rows_49) < n_49:
        raise ValueError(f"{omnimath49_episodes} has {len(rows_49)} rows; need {n_49}")

    selected_34 = rng.sample(rows_34, n_34)
    selected_49 = list(rows_49[:n_49])

    mixed = []
    for row in selected_34:
        item = dict(row)
        item["mixed_group"] = "diff3_4"
        mixed.append(item)
    for row in selected_49:
        item = dict(row)
        item["mixed_group"] = "diff4_9"
        mixed.append(item)

    rng.shuffle(mixed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_data = {
        "seed": seed,
        "inputs": {
            "omnimath34_episodes": str(omnimath34_episodes),
            "omnimath49_episodes": str(omnimath49_episodes),
        },
        "output": str(output),
        "counts": {
            "total": len(mixed),
            "diff3_4": len(selected_34),
            "diff4_9": len(selected_49),
        },
        "difficulty_bucket_counts": {
            "diff3_4": dict(sorted(Counter(_difficulty_bucket(row) for row in selected_34).items())),
            "diff4_9": dict(sorted(Counter(_difficulty_bucket(row) for row in selected_49).items())),
        },
        "selected": {
            "diff3_4": [_selection_record(row) for row in selected_34],
            "diff4_9": [_selection_record(row) for row in selected_49],
        },
        "shuffle_order": [
            {
                "mixed_group": row.get("mixed_group"),
                "id": row.get("id"),
                "source_id": row.get("source_id", row.get("id")),
                "source_index": row.get("source_index"),
                "difficulty": row.get("difficulty"),
            }
            for row in mixed
        ],
    }

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed OmniMath 3-4/4-9 test episodes")
    parser.add_argument(
        "--omnimath34_episodes",
        default="data/episodes/omnimath_diff3_4_test_200_episodes.jsonl",
    )
    parser.add_argument(
        "--omnimath49_episodes",
        default="data/episodes/omnimath_diff4_9_stratified_test_100_episodes.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/episodes/omnimath_mixed_3_4_100_4_9_100_test_episodes.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="results/trim_omnimath13_to_mixed34_49_search/mixed_test_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n34", type=int, default=100)
    parser.add_argument("--n49", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mixed_episodes(
        omnimath34_episodes=Path(args.omnimath34_episodes),
        omnimath49_episodes=Path(args.omnimath49_episodes),
        output=Path(args.output),
        manifest=Path(args.manifest),
        seed=args.seed,
        n_34=args.n34,
        n_49=args.n49,
    )
    print(
        "Wrote {output} with total={total}, diff3_4={n34}, diff4_9={n49}; manifest={manifest}".format(
            output=manifest["output"],
            total=manifest["counts"]["total"],
            n34=manifest["counts"]["diff3_4"],
            n49=manifest["counts"]["diff4_9"],
            manifest=args.manifest,
        )
    )


if __name__ == "__main__":
    main()
