"""Dataset loading utilities — reads local, raw, and TRIM JSONL files."""

import csv
import json
import os
import re
import random
from zipfile import ZipFile
from pathlib import Path
from typing import Dict, List

from config import DATA_DIR, TRIM_DATA_DIR


REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", str(REPO_DATA_DIR))
SRC_DATA_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", str(SRC_DATA_DIR / "raw")))
WIKITQ_RAW_DIR = RAW_DATA_DIR / "wikitablequestions"
TABLEBENCH_RAW_DIR = RAW_DATA_DIR / "tablebench"
GPQA_CACHE_ZIP = Path(os.environ.get(
    "GPQA_CACHE_ZIP",
    str(Path.home() / ".cache/vscode-tmp/gpqa_dataset.zip"),
))
GPQA_CACHE_PASSWORD = os.environ.get("GPQA_CACHE_PASSWORD", "deserted-untie-orchid")


def load_math500() -> List[Dict]:
    """Load the Math eval split used by the TRIM-Agg experiment sweep."""
    return load_trim_dataset("math500", "test_100")


def load_aime2025() -> List[Dict]:
    """Load the official AIME test split used by the experiment sweep."""
    return load_trim_dataset("aime", "test")


def load_wikitq(split: str = "train", raw_dir: str | None = None) -> List[Dict]:
    """Load WikiTableQuestions examples with inline CSV tables.

    Supported split aliases:
    - train / training
    - test / pristine-unseen-tables
    - dev / pristine-seen-tables
    """
    split_map = {
        "train": ("training.tsv", "train"),
        "training": ("training.tsv", "train"),
        "test": ("pristine-unseen-tables.tsv", "test"),
        "pristine-unseen-tables": ("pristine-unseen-tables.tsv", "test"),
        "dev": ("pristine-seen-tables.tsv", "dev"),
        "pristine-seen-tables": ("pristine-seen-tables.tsv", "dev"),
    }
    if split not in split_map:
        raise ValueError(f"Unsupported WikiTQ split: {split}")

    dataset_file, normalized_split = split_map[split]
    root = _resolve_wikitq_root(Path(raw_dir) if raw_dir else WIKITQ_RAW_DIR)
    tsv_path = root / "data" / dataset_file
    if not tsv_path.exists():
        raise FileNotFoundError(f"WikiTQ split not found: {tsv_path}")

    items = []
    with open(tsv_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            context_rel = row["context"]
            table_path = root / context_rel
            table = _load_csv_table(table_path)
            items.append({
                "id": row["id"],
                "query": row["utterance"],
                "answer": row["targetValue"],
                "table": table,
                "dataset": "wikitq",
                "split": normalized_split,
                "source_path": str(tsv_path),
                "context_path": context_rel,
                "qtype": "TableQA",
                "qsubtype": "WikiTableQuestions",
            })
    print(f"Loaded {len(items)} WikiTQ {normalized_split} examples from {tsv_path}")
    return items


def load_tablebench(
    raw_dir: str | None = None,
    include_visualization: bool = False,
) -> List[Dict]:
    """Load TableBench test items from the official JSONL file."""
    root = Path(raw_dir) if raw_dir else TABLEBENCH_RAW_DIR
    jsonl_path = root / "TableBench.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"TableBench file not found: {jsonl_path}")

    items = []
    for row in load_jsonl(str(jsonl_path)):
        if not include_visualization and row.get("qtype") == "Visualization":
            continue
        items.append({
            "id": row["id"],
            "query": row["question"],
            "answer": row["answer"],
            "table": row["table"],
            "dataset": "tablebench",
            "split": "test",
            "source_path": str(jsonl_path),
            "qtype": row.get("qtype", ""),
            "qsubtype": row.get("qsubtype", ""),
        })
    print(
        f"Loaded {len(items)} TableBench items from {jsonl_path} "
        f"(include_visualization={include_visualization})"
    )
    return items


def load_aime_2010_2024_part1_train() -> List[Dict]:
    """Load 2010-2024 AIME Part I problems for AIME quick-search training."""
    path = os.path.join(LOCAL_DATA_DIR, "aime_1983_2024.jsonl")
    items = []
    for row in load_jsonl(path):
        year = int(row.get("year", 0) or 0)
        part = str(row.get("part", "")).strip()
        if not (2010 <= year <= 2024 and part == "I"):
            continue
        item = _normalise_aime_row(
            row,
            dataset="aime_2010_2024_part1_train",
            split="train",
            source_path=path,
            fallback_id=f"aime_part1_train_{len(items):05d}",
        )
        items.append(item)
    print(f"Loaded {len(items)} AIME 2010-2024 Part I train problems from {path}")
    return items


def load_aime_2020_2024_part2_test() -> List[Dict]:
    """Load the TRIM/Agg AIME 2020-2024 Part II test split."""
    path = os.path.join(TRIM_DATA_DIR, "aime2020_2024", "test.jsonl")
    items = []
    for row in load_jsonl(path):
        year = int(row.get("year", 0) or 0)
        part = str(row.get("part", "")).strip()
        if not (2020 <= year <= 2024 and part == "II"):
            continue
        item = _normalise_aime_row(
            row,
            dataset="aime_2020_2024_part2_test",
            split="test",
            source_path=path,
            fallback_id=f"aime_part2_test_{len(items):05d}",
        )
        items.append(item)
    print(f"Loaded {len(items)} AIME 2020-2024 Part II test problems from {path}")
    return items


def _load_gpqa_csv_rows(csv_name: str) -> List[Dict]:
    if not GPQA_CACHE_ZIP.exists():
        raise FileNotFoundError(f"GPQA archive not found: {GPQA_CACHE_ZIP}")
    with ZipFile(GPQA_CACHE_ZIP) as zf:
        zf.setpassword(GPQA_CACHE_PASSWORD.encode("utf-8"))
        with zf.open(f"dataset/{csv_name}") as handle:
            text = (line.decode("utf-8") for line in handle)
            return list(csv.DictReader(text))


def _normalize_gpqa_row(row: Dict, dataset: str, split: str, fallback_id: str, seed: int) -> Dict:
    choices = [
        str(row.get("Correct Answer", "")).strip(),
        str(row.get("Incorrect Answer 1", "")).strip(),
        str(row.get("Incorrect Answer 2", "")).strip(),
        str(row.get("Incorrect Answer 3", "")).strip(),
    ]
    indexed = list(enumerate(choices))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    shuffled_choices = [choice for _, choice in indexed]
    answer_idx = next(idx for idx, (orig_idx, _) in enumerate(indexed) if orig_idx == 0)
    answer_letter = "ABCD"[answer_idx]
    query = f"{row['Question'].strip()}\nAnswer Choices: " + " ".join(
        f"({lab}) {choice}" for lab, choice in zip("ABCD", shuffled_choices)
    )
    return {
        "id": fallback_id,
        "query": query,
        "answer": answer_letter,
        "choices": shuffled_choices,
        "dataset": dataset,
        "split": split,
        "source_path": str(GPQA_CACHE_ZIP),
        "subject": row.get("Subdomain", ""),
        "task_type": "multiple_choice",
    }


def load_gpqa_main_train_200(seed: int = 1) -> List[Dict]:
    rows = _load_gpqa_csv_rows("gpqa_main.csv")
    items = [
        _normalize_gpqa_row(
            row,
            dataset="gpqa_main_train_200",
            split="train",
            fallback_id=f"gpqa_main_{idx:05d}",
            seed=seed + idx,
        )
        for idx, row in enumerate(rows)
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:200]
    print(f"Loaded {len(items)} GPQA main train items from {GPQA_CACHE_ZIP}")
    return items


def load_gpqa_diamond_test_100(seed: int = 1) -> List[Dict]:
    rows = _load_gpqa_csv_rows("gpqa_diamond.csv")
    items = [
        _normalize_gpqa_row(
            row,
            dataset="gpqa_diamond_test_100",
            split="test",
            fallback_id=f"gpqa_diamond_{idx:05d}",
            seed=seed + idx,
        )
        for idx, row in enumerate(rows)
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:100]
    print(f"Loaded {len(items)} GPQA diamond test items from {GPQA_CACHE_ZIP}")
    return items


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


def _normalise_aime_row(row: Dict, dataset: str, split: str, source_path: str, fallback_id: str) -> Dict:
    query = row.get("problem") or row.get("question")
    answer = row.get("answer")
    if answer is None or str(answer).strip() == "":
        answer = _extract_boxed(row.get("solution", "")) or row.get("solution", "")
    raw_id = row.get("source_id") or row.get("unique_id") or row.get("ID") or row.get("id") or fallback_id
    return {
        "id": str(raw_id).replace("/", "_"),
        "query": query,
        "answer": str(answer).strip(),
        "full_solution": row.get("solution", ""),
        "source_path": source_path,
        "source_id": row.get("source_id", raw_id),
        "problem_number": row.get("problem_number", 0),
        "part": str(row.get("part", "")).strip(),
        "year": int(row.get("year", row.get("Year", 0)) or 0),
        "dataset": dataset,
        "split": split,
    }


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


def _resolve_wikitq_root(root: Path) -> Path:
    if (root / "WikiTableQuestions").exists():
        return root / "WikiTableQuestions"
    return root


def _load_csv_table(path: Path) -> Dict:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"columns": [], "data": []}
    return {
        "columns": rows[0],
        "data": rows[1:],
    }


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
