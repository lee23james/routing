"""Prepare an AIME 2020-2024 subset for the official TRIM pipeline.

Source data:
    /home/chencheng/routing/data/aime_1983_2024.jsonl

Split strategy:
    - Train: 2020-2024 AIME Part I
    - Test:  2020-2024 AIME Part II

The official TRIM code expects JSONL files under:
    math_eval/data/<dataset_name>/{train,test}.jsonl

This script writes:
    math_eval/data/aime2020_2024/train.jsonl
    math_eval/data/aime2020_2024/test.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path


SOURCE_PATH = Path("/home/chencheng/routing/data/aime_1983_2024.jsonl")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "math_eval" / "data" / "aime2020_2024"


def normalize_part(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace("AIME", "").strip()
    if text in {"I", "1"}:
        return "I"
    if text in {"II", "2"}:
        return "II"
    return None


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with SOURCE_PATH.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def convert_row(row: dict) -> dict:
    answer = str(row["answer"]).strip()
    return {
        "problem": row["question"],
        # The official parser extracts the final answer from `solution`.
        # A plain answer string is sufficient for ground-truth extraction.
        "solution": answer,
        "answer": answer,
        "year": row["year"],
        "part": normalize_part(row.get("part")),
        "source_id": row["id"],
        "problem_number": row.get("problem_number"),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_PATH}")

    rows = load_rows()
    subset = [row for row in rows if 2020 <= int(row.get("year", 0)) <= 2024]

    train_rows = [convert_row(row) for row in subset if normalize_part(row.get("part")) == "I"]
    test_rows = [convert_row(row) for row in subset if normalize_part(row.get("part")) == "II"]

    train_path = OUTPUT_DIR / "train.jsonl"
    test_path = OUTPUT_DIR / "test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    print(f"Source rows (2020-2024): {len(subset)}")
    print(f"Train rows (Part I):     {len(train_rows)}")
    print(f"Test rows (Part II):     {len(test_rows)}")
    print(f"Wrote: {train_path}")
    print(f"Wrote: {test_path}")


if __name__ == "__main__":
    main()
