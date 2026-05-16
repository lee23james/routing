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
GSM8K_DATA_DIR = Path(os.environ.get(
    "GSM8K_DATA_DIR",
    "/home/chencheng/RSD/external/qwen25_math_evaluation/data/gsm8k",
))
MMLU_STEM_DATA_DIR = Path(os.environ.get(
    "MMLU_STEM_DATA_DIR",
    "/home/chencheng/RSD/external/qwen25_math_evaluation/data/mmlu_stem",
))


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


def _extract_gsm8k_final_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        return str(answer).strip()
    return answer.rsplit(marker, 1)[1].strip()


def _load_gsm8k_split_sample(
    *,
    split: str,
    dataset: str,
    max_items: int,
    seed: int = 1,
) -> List[Dict]:
    path = GSM8K_DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GSM8K split not found: {path}")

    items = []
    for i, row in enumerate(load_jsonl(str(path))):
        query = row.get("question")
        answer = _extract_gsm8k_final_answer(str(row.get("answer", "")))
        if not query or not answer:
            continue
        source_index = int(row.get("idx", i))
        source_id = f"gsm8k_{split}_{source_index:05d}"
        items.append({
            "id": source_id,
            "query": query,
            "answer": answer,
            "full_solution": str(row.get("answer", "")).split("####", 1)[0].strip(),
            "dataset": dataset,
            "split": split,
            "source_path": str(path),
            "source_index": source_index,
            "source_id": source_id,
        })

    if len(items) < max_items:
        raise ValueError(
            f"Only found {len(items)} GSM8K {split} items for {dataset}; need {max_items}"
        )
    rng = random.Random(seed)
    rng.shuffle(items)
    selected = items[:max_items]
    print(f"Loaded {len(selected)} GSM8K {dataset} items from {path} (seed={seed})")
    return selected


def load_gsm8k_train_300(seed: int = 1) -> List[Dict]:
    """Load a fixed random GSM8K train-300 split from the local RSD data."""
    return _load_gsm8k_split_sample(
        split="train",
        dataset="gsm8k_train_300",
        max_items=300,
        seed=seed,
    )


def load_gsm8k_test_189(seed: int = 1) -> List[Dict]:
    """Load a fixed random GSM8K test-189 split from the local RSD data."""
    return _load_gsm8k_split_sample(
        split="test",
        dataset="gsm8k_test_189",
        max_items=189,
        seed=seed,
    )


def _normalize_mmlu_stem_row(row: Dict, dataset: str, split: str, source_index: int) -> Dict:
    choices = [str(choice).strip() for choice in row.get("choices", [])]
    if len(choices) != 4:
        raise ValueError(f"MMLU-STEM row {source_index} has {len(choices)} choices, expected 4")

    answer_idx = int(row["answer"])
    if not 0 <= answer_idx < 4:
        raise ValueError(f"MMLU-STEM row {source_index} has invalid answer index: {answer_idx}")

    query = f"{str(row['question']).strip()}\nAnswer Choices: " + " ".join(
        f"({label}) {choice}" for label, choice in zip("ABCD", choices)
    )
    source_id = f"mmlu_stem_{source_index:05d}"
    return {
        "id": f"{dataset}_{source_index:05d}",
        "query": query,
        "answer": "ABCD"[answer_idx],
        "choices": choices,
        "dataset": dataset,
        "split": split,
        "source_path": str(MMLU_STEM_DATA_DIR / "test.jsonl"),
        "source_index": source_index,
        "source_id": source_id,
        "subject": row.get("type", ""),
        "task_type": "multiple_choice",
    }


def _load_mmlu_stem_sample(seed: int = 1) -> List[Dict]:
    path = MMLU_STEM_DATA_DIR / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"MMLU-STEM split not found: {path}")

    rows = load_jsonl(str(path))
    if len(rows) < 389:
        raise ValueError(f"Only found {len(rows)} MMLU-STEM rows in {path}; need 389")

    indexed_rows = list(enumerate(rows))
    rng = random.Random(seed)
    rng.shuffle(indexed_rows)
    return indexed_rows[:389]


def load_mmlu_stem_train_200(seed: int = 1) -> List[Dict]:
    """Load a fixed random MMLU-STEM train-200 split from the local RSD pool."""
    sampled = _load_mmlu_stem_sample(seed)[:200]
    items = [
        _normalize_mmlu_stem_row(row, "mmlu_stem_train_200", "train", source_index)
        for source_index, row in sampled
    ]
    print(f"Loaded {len(items)} MMLU-STEM train items from {MMLU_STEM_DATA_DIR / 'test.jsonl'} (seed={seed})")
    return items


def load_mmlu_stem_test_189(seed: int = 1) -> List[Dict]:
    """Load the held-out 189 MMLU-STEM items from the fixed random 389 sample."""
    sampled = _load_mmlu_stem_sample(seed)[200:389]
    items = [
        _normalize_mmlu_stem_row(row, "mmlu_stem_test_189", "test", source_index)
        for source_index, row in sampled
    ]
    print(f"Loaded {len(items)} MMLU-STEM test items from {MMLU_STEM_DATA_DIR / 'test.jsonl'} (seed={seed})")
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


def load_omnimath7_9_test_100(max_items: int = 100) -> List[Dict]:
    """Load the first 100 valid OmniMath difficulty 7-9 problems as a test split."""
    path = os.path.join(LOCAL_DATA_DIR, "omnimath.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OmniMath file not found: {path}")

    items = []
    for i, row in enumerate(load_jsonl(path)):
        try:
            diff = float(row.get("difficulty", 5.0))
        except (TypeError, ValueError):
            continue
        if diff < 7.0 or diff > 9.0:
            continue

        answer = str(row.get("answer") or "").strip()
        if not answer:
            answer = _extract_boxed(row.get("solution", ""))
        if not answer.strip():
            continue

        query = row.get("problem") or row.get("question")
        if not query:
            continue

        source_id = f"omnimath_{i:05d}"
        items.append({
            "id": source_id,
            "query": query,
            "answer": answer.strip(),
            "full_solution": row.get("solution", ""),
            "difficulty": diff,
            "source": row.get("source", ""),
            "domain": row.get("domain", []),
            "source_path": path,
            "source_index": i,
            "source_id": source_id,
            "dataset": "omnimath7_9_test_100",
            "split": "test",
        })
        if max_items > 0 and len(items) >= max_items:
            break

    print(f"Loaded {len(items)} OmniMath difficulty 7-9 test problems from {path}")
    return items


def load_omnimath_diff_range_sample(
    *,
    dataset: str,
    split: str,
    min_diff: float,
    max_diff: float,
    max_items: int,
    seed: int = 1,
) -> List[Dict]:
    """Load a fixed random OmniMath sample for a half-open difficulty range."""
    path = os.path.join(LOCAL_DATA_DIR, "omnimath.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OmniMath file not found: {path}")

    items = []
    for i, row in enumerate(load_jsonl(path)):
        try:
            diff = float(row.get("difficulty", 5.0))
        except (TypeError, ValueError):
            continue
        if not (min_diff <= diff < max_diff):
            continue

        answer = str(row.get("answer") or "").strip()
        if not answer:
            answer = _extract_boxed(row.get("solution", ""))
        if not answer.strip():
            continue

        query = row.get("problem") or row.get("question")
        if not query:
            continue

        source_id = f"omnimath_{i:05d}"
        items.append({
            "id": source_id,
            "query": query,
            "answer": answer.strip(),
            "full_solution": row.get("solution", ""),
            "difficulty": diff,
            "source": row.get("source", ""),
            "domain": row.get("domain", []),
            "source_path": path,
            "source_index": i,
            "source_id": source_id,
            "dataset": dataset,
            "split": split,
        })

    if max_items > 0:
        if len(items) < max_items:
            raise ValueError(
                f"Only found {len(items)} OmniMath problems for {dataset} "
                f"(difficulty {min_diff} <= d < {max_diff}); need {max_items}"
            )
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:max_items]

    print(
        f"Loaded {len(items)} OmniMath {dataset} problems "
        f"(diff {min_diff} <= d < {max_diff}, seed={seed}) from {path}"
    )
    return items


def load_omnimath_diff_range_stratified_sample(
    *,
    dataset: str,
    split: str,
    min_bucket: int,
    max_bucket: int,
    items_per_bucket: int,
    seed: int = 1,
) -> List[Dict]:
    """Load a fixed random balanced OmniMath sample over integer difficulty buckets."""
    path = os.path.join(LOCAL_DATA_DIR, "omnimath.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OmniMath file not found: {path}")

    buckets: Dict[int, List[Dict]] = {bucket: [] for bucket in range(min_bucket, max_bucket)}
    for i, row in enumerate(load_jsonl(path)):
        try:
            diff = float(row.get("difficulty", 5.0))
        except (TypeError, ValueError):
            continue
        if not (float(min_bucket) <= diff < float(max_bucket)):
            continue

        bucket = int(diff)
        if bucket not in buckets:
            continue

        answer = str(row.get("answer") or "").strip()
        if not answer:
            answer = _extract_boxed(row.get("solution", ""))
        if not answer.strip():
            continue

        query = row.get("problem") or row.get("question")
        if not query:
            continue

        source_id = f"omnimath_{i:05d}"
        buckets[bucket].append({
            "id": source_id,
            "query": query,
            "answer": answer.strip(),
            "full_solution": row.get("solution", ""),
            "difficulty": diff,
            "source": row.get("source", ""),
            "domain": row.get("domain", []),
            "source_path": path,
            "source_index": i,
            "source_id": source_id,
            "difficulty_bucket": bucket,
            "dataset": dataset,
            "split": split,
        })

    rng = random.Random(seed)
    selected = []
    for bucket in range(min_bucket, max_bucket):
        bucket_items = buckets[bucket]
        if len(bucket_items) < items_per_bucket:
            raise ValueError(
                f"Only found {len(bucket_items)} OmniMath problems for {dataset} "
                f"in difficulty bucket [{bucket}, {bucket + 1}); need {items_per_bucket}"
            )
        rng.shuffle(bucket_items)
        selected.extend(bucket_items[:items_per_bucket])

    rng.shuffle(selected)
    print(
        f"Loaded {len(selected)} OmniMath {dataset} problems "
        f"(balanced buckets {min_bucket}-{max_bucket - 1}, "
        f"{items_per_bucket}/bucket, seed={seed}) from {path}"
    )
    return selected


def load_omnimath_diff1_3_train_200(seed: int = 1) -> List[Dict]:
    """Load fixed random OmniMath 1<=difficulty<3 train sample."""
    return load_omnimath_diff_range_sample(
        dataset="omnimath_diff1_3_train_200",
        split="train",
        min_diff=1.0,
        max_diff=3.0,
        max_items=200,
        seed=seed,
    )


def load_omnimath_diff3_4_test_200(seed: int = 1) -> List[Dict]:
    """Load fixed random OmniMath 3<=difficulty<4 test sample."""
    return load_omnimath_diff_range_sample(
        dataset="omnimath_diff3_4_test_200",
        split="test",
        min_diff=3.0,
        max_diff=4.0,
        max_items=200,
        seed=seed,
    )


def load_omnimath_diff4_9_stratified_test_100(seed: int = 1) -> List[Dict]:
    """Load fixed random OmniMath 4<=difficulty<9 test sample, 20 per bucket."""
    return load_omnimath_diff_range_stratified_sample(
        dataset="omnimath_diff4_9_stratified_test_100",
        split="test",
        min_bucket=4,
        max_bucket=9,
        items_per_bucket=20,
        seed=seed,
    )


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
