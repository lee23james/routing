"""Dataset loading utilities — reads local and TRIM JSONL files."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

from config import DATA_DIR, TRIM_DATA_DIR


REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", str(REPO_DATA_DIR))


def load_math500() -> List[Dict]:
    """Load the Math eval split used by the TRIM-Agg experiment sweep."""
    return load_trim_dataset("math500", "test_100")


def load_aime2025() -> List[Dict]:
    """Load the official AIME test split used by the experiment sweep."""
    return load_trim_dataset("aime", "test")


def load_math_train() -> List[Dict]:
    """Load the full local MATH train split for MATH-only point search."""
    local_path = os.path.join(TRIM_DATA_DIR, "math", "train.jsonl")
    if os.path.exists(local_path):
        items = _load_math_rows(local_path, dataset="math", split="train", id_prefix="math_train")
        print(f"Loaded {len(items)} MATH training problems from {local_path}")
        return items

    return load_trim_dataset("math", "train_1k")


def load_trim_dataset(dataset_name: str, split: str) -> List[Dict]:
    """Load the exact JSONL dataset used by trim/TRIM/TRIM_Agg.py."""
    path = os.path.join(TRIM_DATA_DIR, dataset_name, f"{split}.jsonl")
    rows = load_jsonl(path)
    items = []
    for i, row in enumerate(rows):
        query = row.get("problem") or row.get("question")
        if not query:
            continue

        answer = row.get("answer")
        if answer is None or str(answer).strip() == "":
            answer = _extract_boxed(row.get("solution", "")) or row.get("solution", "")

        raw_id = row.get("unique_id") or row.get("ID") or row.get("id")
        item_id = raw_id if raw_id else f"{dataset_name}_{split}_{i:05d}"
        item_id = str(item_id).replace("/", "_")

        items.append({
            "id": item_id,
            "query": query,
            "answer": str(answer).strip(),
            "dataset": dataset_name,
            "split": split,
            "source_path": path,
            "subject": row.get("subject", ""),
            "level": row.get("level", row.get("Level", 0)),
            "year": row.get("Year", row.get("year", 0)),
        })
    print(f"Loaded {len(items)} TRIM problems from {path}")
    return items


TRIM_DATASET_ALIASES = {
    "trim_math_train_1k": ("math", "train_1k"),
    "trim_math500_test_100": ("math500", "test_100"),
    "trim_aime_train": ("aime", "train"),
    "trim_aime_test": ("aime", "test"),
}


def load_trim_dataset_alias(alias: str) -> List[Dict]:
    if alias not in TRIM_DATASET_ALIASES:
        raise ValueError(f"Unknown TRIM dataset alias: {alias}")
    dataset_name, split = TRIM_DATASET_ALIASES[alias]
    return load_trim_dataset(dataset_name, split)


def _load_math_rows(path: str, dataset: str, split: str, id_prefix: str) -> List[Dict]:
    items = []
    for i, row in enumerate(load_jsonl(path)):
        query = row.get("problem") or row.get("question")
        if not query:
            continue
        answer = row.get("answer", _extract_boxed(row.get("solution", "")))
        item_id = row.get("unique_id") or row.get("ID") or row.get("id") or f"{id_prefix}_{i:05d}"
        items.append({
            "id": str(item_id).replace("/", "_"),
            "query": query,
            "answer": str(answer).strip(),
            "full_solution": row.get("solution", ""),
            "source_path": path,
            "subject": row.get("subject", ""),
            "level": row.get("level", row.get("Level", 0)),
            "year": row.get("Year", row.get("year", 0)),
            "dataset": dataset,
            "split": split,
        })
    return items


def load_omnimath(max_items: int = 0, min_diff: float = 1.0, max_diff: float = 10.0) -> List[Dict]:
    """Load OmniMath from local repo data; fall back to MATH train if absent."""
    path = os.path.join(LOCAL_DATA_DIR, "omnimath.jsonl")
    if not os.path.exists(path):
        items = load_math_train()
        return items[:max_items] if max_items > 0 else items

    items = []
    for i, row in enumerate(load_jsonl(path)):
        diff = row.get("difficulty", 5.0)
        if diff < min_diff or diff > max_diff:
            continue
        answer = row.get("answer", "")
        if not answer or answer.strip() == "":
            answer = _extract_boxed(row.get("solution", ""))
        if not answer.strip():
            continue
        items.append({
            "id": f"omnimath_{i:05d}",
            "query": row["problem"],
            "answer": answer,
            "difficulty": diff,
            "source": row.get("source", ""),
            "domain": row.get("domain", []),
            "dataset": "omnimath",
            "split": "train",
        })
    if max_items > 0:
        items = items[:max_items]
    print(f"Loaded {len(items)} OmniMath problems (diff {min_diff}-{max_diff})")
    return items


def load_aime_1983_2024() -> List[Dict]:
    """Load the official AIME train split used by the experiment sweep."""
    return load_trim_dataset("aime", "train")


def load_aime_train() -> List[Dict]:
    """Load the official AIME train split used by the experiment sweep."""
    return load_trim_dataset("aime", "train")


def _extract_boxed(solution: str) -> str:
    """Extract answer from \\boxed{} in solution text."""
    i = 0
    last_match = ""
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
                    last_match = solution[start:j]
                    break
                depth -= 1
        i = idx + 1
    return last_match.strip()


def save_jsonl(items: List[Dict], path: str):
    """Save items to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    """Load items from JSONL file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items
