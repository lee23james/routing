"""Dataset loading utilities — reads local JSONL files."""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

from config import TRIM_DATA_DIR

# Local data directory
LOCAL_DATA_DIR = os.environ.get(
    "LOCAL_DATA_DIR",
    str(Path(__file__).resolve().parents[2] / "data"),
)


def load_math500() -> List[Dict]:
    """Load MATH-500 test set from local JSONL."""
    path = os.path.join(LOCAL_DATA_DIR, "math500.jsonl")
    items = []
    for i, row in enumerate(load_jsonl(path)):
        items.append({
            "id": f"math500_{i:05d}",
            "query": row["problem"],
            "answer": row.get("answer", _extract_boxed(row.get("solution", ""))),
            "subject": row.get("subject", ""),
            "level": row.get("level", 0),
            "dataset": "math500",
            "split": "test",
        })
    print(f"Loaded {len(items)} problems from {path}")
    return items


def load_aime2025() -> List[Dict]:
    """Load AIME 2025 (I + II) from local JSONL files."""
    items = []
    for fname in ["aime2025-I.jsonl", "aime2025-II.jsonl"]:
        path = os.path.join(LOCAL_DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        tag = fname.replace(".jsonl", "")
        for i, row in enumerate(load_jsonl(path)):
            items.append({
                "id": f"{tag}_{i:05d}",
                "query": row["question"],
                "answer": str(row["answer"]),
                "dataset": "aime2025",
                "split": "test",
            })
    print(f"Loaded {len(items)} AIME 2025 problems")
    return items


def load_math_train() -> List[Dict]:
    """Load MATH training set for RL training data generation."""
    local_path = os.path.join(TRIM_DATA_DIR, "math", "train.jsonl")
    if os.path.exists(local_path):
        items = _load_math_rows(local_path, dataset="math", split="train", id_prefix="math_train")
        print(f"Loaded {len(items)} MATH training problems from {local_path}")
        return items

    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
        items = []
        for i, row in enumerate(ds):
            answer = _extract_boxed(row["solution"])
            items.append({
                "id": f"math_train_{i:05d}",
                "query": row["problem"],
                "answer": answer,
                "full_solution": row["solution"],
                "dataset": "math",
                "split": "train",
            })
        print(f"Loaded {len(items)} MATH training problems from HuggingFace")
        return items
    except Exception as e:
        print(f"Failed to load MATH from HuggingFace: {e}")
        print("Falling back to local math500.jsonl as training data")
        return load_math500()


def _load_math_rows(path: str, dataset: str, split: str, id_prefix: str) -> List[Dict]:
    items = []
    for i, row in enumerate(load_jsonl(path)):
        query = row.get("problem") or row.get("question")
        if not query:
            continue
        answer = row.get("answer", _extract_boxed(row.get("solution", "")))
        item_id = row.get("unique_id") or row.get("id") or f"{id_prefix}_{i:05d}"
        items.append({
            "id": str(item_id).replace("/", "_"),
            "query": query,
            "answer": str(answer).strip(),
            "full_solution": row.get("solution", ""),
            "subject": row.get("subject", ""),
            "level": row.get("level", 0),
            "dataset": dataset,
            "split": split,
        })
    return items


def load_omnimath(max_items: int = 0, min_diff: float = 1.0, max_diff: float = 10.0) -> List[Dict]:
    """Load OmniMath dataset from local JSONL for RL training (held-out from test)."""
    path = os.path.join(LOCAL_DATA_DIR, "omnimath.jsonl")
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
    """Load AIME 1983-2024 from local JSONL (converted from CSV)."""
    path = os.path.join(LOCAL_DATA_DIR, "aime_1983_2024.jsonl")
    items = []
    for row in load_jsonl(path):
        answer = str(row["answer"]).strip()
        if "both" in answer.lower() or "or" in answer.lower():
            answer = re.split(r'\s+or\s+', answer)[0].strip()
        items.append({
            "id": row["id"],
            "query": row["question"],
            "answer": answer,
            "year": row.get("year", 0),
            "dataset": "aime",
            "split": "test",
        })
    print(f"Loaded {len(items)} AIME 1983-2024 problems from {path}")
    return items


def load_aime_train() -> List[Dict]:
    """Load AIME training data from local JSONL or HuggingFace."""
    try:
        return load_aime_1983_2024()
    except Exception as e:
        print(f"Failed to load local AIME: {e}")
        return load_aime2025()


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
