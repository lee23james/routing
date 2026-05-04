"""Dataset loading utilities."""

import json
import os
import re
from pathlib import Path
from typing import List, Dict

TRIM_DATA_DIR = os.environ.get(
    "TRIM_DATA_DIR",
    str(Path(__file__).resolve().parents[2] / "trim" / "TRIM" / "math_eval" / "data"),
)


def load_math500() -> List[Dict]:
    return _load_trim_dataset("math500", "test_100")


def load_aime(year_from: int = 2020, year_to: int = 2024) -> List[Dict]:
    items = _load_trim_dataset("aime", "test")
    return [
        item
        for item in items
        if year_from <= item.get("year", 0) <= year_to
    ]


def _load_trim_dataset(dataset_name: str, split: str) -> List[Dict]:
    path = os.path.join(TRIM_DATA_DIR, dataset_name, f"{split}.jsonl")
    items = []
    for i, row in enumerate(_load_jsonl(path)):
        problem = row.get("problem") or row.get("question")
        if not problem:
            continue
        answer = str(row.get("answer") or "").strip()
        if not answer:
            answer = _extract_boxed(row.get("solution", "")) or str(row.get("solution", "")).strip()
        if "or" in answer.lower():
            answer = re.split(r"\s+or\s+", answer)[0].strip()
        raw_id = row.get("unique_id") or row.get("ID") or row.get("id")
        items.append({
            "id": str(raw_id or f"{dataset_name}_{split}_{i:05d}").replace("/", "_"),
            "problem": problem,
            "answer": answer,
            "dataset": dataset_name,
            "year": row.get("Year", row.get("year", 0)),
        })
    return items


def _load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _extract_boxed(solution: str) -> str:
    i = 0
    last = ""
    while True:
        idx = solution.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(solution)):
            if solution[j] == "{":
                depth += 1
            elif solution[j] == "}":
                if depth == 0:
                    last = solution[start:j]
                    break
                depth -= 1
        i = idx + 1
    return last.strip()
