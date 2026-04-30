"""Dataset loading utilities — reads local and TRIM JSONL files."""

import json
import os
from typing import Dict, List

from config import DATA_DIR, TRIM_DATA_DIR

# Local data directory
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", DATA_DIR)


def load_math500() -> List[Dict]:
    """Load the Math eval split used by the TRIM-Agg experiment sweep."""
    return load_trim_dataset("math500", "test_100")


def load_aime2025() -> List[Dict]:
    """Load the official AIME test split used by the experiment sweep."""
    return load_trim_dataset("aime", "test")


def load_math_train() -> List[Dict]:
    """Load the Math train split used by the TRIM-Agg experiment sweep."""
    return load_trim_dataset("math", "train_1k")


def load_trim_dataset(dataset_name: str, split: str) -> List[Dict]:
    """Load the exact JSONL dataset used by trim/TRIM/TRIM_Agg.py.

    Examples:
      - dataset_name="math", split="train_1k"
      - dataset_name="math500", split="test_100"
      - dataset_name="aime", split="train" or "test"
    """
    path = os.path.join(TRIM_DATA_DIR, dataset_name, f"{split}.jsonl")
    rows = load_jsonl(path)
    items = []
    for i, row in enumerate(rows):
        query = row.get("problem") or row.get("question")
        if not query:
            continue

        answer = row.get("answer")
        if answer is None or str(answer).strip() == "":
            # Official AIME files store the numeric answer in "solution".
            answer = _extract_boxed(row.get("solution", "")) or row.get("solution", "")

        raw_id = row.get("unique_id") or row.get("ID") or row.get("id")
        item_id = raw_id if raw_id else f"{dataset_name}_{split}_{i:05d}"
        item_id = str(item_id).replace("/", "_")

        items.append(
            {
                "id": item_id,
                "query": query,
                "answer": str(answer).strip(),
                "dataset": dataset_name,
                "split": split,
                "source_path": path,
                "subject": row.get("subject", ""),
                "level": row.get("level", row.get("Level", 0)),
                "year": row.get("Year", row.get("year", 0)),
            }
        )
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


def load_omnimath(max_items: int = 0, min_diff: float = 1.0, max_diff: float = 10.0) -> List[Dict]:
    """Compatibility wrapper: use the experiment Math train_1k split."""
    items = load_math_train()
    if max_items > 0:
        items = items[:max_items]
    return items


def load_aime_1983_2024() -> List[Dict]:
    """Compatibility wrapper: use the official AIME train split."""
    return load_trim_dataset("aime", "train")


def load_aime_train() -> List[Dict]:
    """Load the official AIME train split used by the experiment sweep."""
    return load_trim_dataset("aime", "train")


def _extract_boxed(solution: str) -> str:
    """Extract answer from \\boxed{} in solution text."""
    # Handle nested braces
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
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    """Load items from JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items
