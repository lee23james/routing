"""Dataset loading utilities."""

import json
import os
import re
from typing import List, Dict

DATA_DIR = "/export/shy/pp/pp5/data"


def load_math500() -> List[Dict]:
    path = os.path.join(DATA_DIR, "math500.jsonl")
    items = []
    for i, row in enumerate(_load_jsonl(path)):
        answer = row.get("answer", "")
        if not answer:
            answer = _extract_boxed(row.get("solution", ""))
        items.append({
            "id": f"math500_{i:04d}",
            "problem": row["problem"],
            "answer": answer,
            "dataset": "math500",
        })
    return items


def load_aime(year_from: int = 2020, year_to: int = 2024) -> List[Dict]:
    path = os.path.join(DATA_DIR, "aime_1983_2024.jsonl")
    items = []
    for row in _load_jsonl(path):
        y = row.get("year", 0)
        if y < year_from or y > year_to:
            continue
        answer = str(row["answer"]).strip()
        if "or" in answer.lower():
            answer = re.split(r"\s+or\s+", answer)[0].strip()
        items.append({
            "id": row["id"],
            "problem": row["question"],
            "answer": answer,
            "dataset": "aime",
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
