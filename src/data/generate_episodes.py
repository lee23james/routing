"""Generate episode data for TRIM-Agg RL training.

For each problem:
1. Call SRM (vLLM API) → full solution → split into steps
2. Call LRM (vLLM API) → full solution → split into steps
3. Score each step with PRM (local model)
4. Record token counts, correctness, PRM scores
5. Write one JSONL line per problem

Usage:
    python -m data.generate_episodes --dataset all --prm_device cuda:0
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VLLM_SRM_PORT, VLLM_LRM_PORT, PRM_MODEL,
    EPISODES_DIR, MAX_STEPS, MAX_NEW_TOKENS, PRM_DEVICE, THINK_MODE,
)
from vllm_client import VLLMClient
from models import PRMScorer, split_steps, extract_answer, check_correctness
from data.datasets import (
    load_math500, load_aime2025, load_aime_1983_2024, load_omnimath,
    save_jsonl, load_jsonl,
)


def _distribute_tokens(steps: List[str], total_tokens: int) -> List[int]:
    """Distribute total token count proportionally across steps by char length."""
    if not steps or total_tokens <= 0:
        return [len(s.split()) for s in steps]
    char_lens = [max(len(s), 1) for s in steps]
    total_chars = sum(char_lens)
    per_step = [max(1, round(c / total_chars * total_tokens)) for c in char_lens]
    return per_step


def generate_episodes(
    dataset_name: str,
    output_dir: str = None,
    prm_device: str = "cuda:0",
    n_solutions: int = 1,
    temperature: float = 0.0,
    max_workers: int = 4,
    resume: bool = True,
):
    if dataset_name == "math500":
        items = load_math500()
    elif dataset_name == "aime2025":
        items = load_aime2025()
    elif dataset_name == "aime":
        items = load_aime_1983_2024()
    elif dataset_name.startswith("aime_"):
        all_aime = load_aime_1983_2024()
        try:
            parts = dataset_name.split("_")[1:]
            if len(parts) == 2:
                y_from, y_to = int(parts[0]), int(parts[1])
            else:
                y_from = y_to = int(parts[0])
            items = [it for it in all_aime if y_from <= it.get("year", 0) <= y_to]
            print(f"Filtered AIME {y_from}-{y_to}: {len(items)} problems")
        except ValueError:
            items = all_aime
    elif dataset_name == "all":
        items = load_math500() + load_aime2025()
    elif dataset_name == "omnimath":
        items = load_omnimath(max_items=200, min_diff=1.0, max_diff=4.0)
    elif dataset_name == "omnimath_full":
        items = load_omnimath(max_items=500)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {len(items)} problems ({dataset_name})")

    if output_dir is None:
        output_dir = EPISODES_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_episodes.jsonl")

    # Resume support
    done_ids = set()
    if resume and os.path.exists(output_path):
        for row in load_jsonl(output_path):
            done_ids.add(row["id"])
        print(f"Resuming: {len(done_ids)} already done")

    items = [it for it in items if it["id"] not in done_ids]
    if not items:
        print("All episodes already generated.")
        return

    srm = VLLMClient(VLLM_SRM_PORT, model_name="srm")
    lrm = VLLMClient(VLLM_LRM_PORT, model_name="lrm")

    print(f"Loading PRM on {prm_device} ...")
    prm = PRMScorer(PRM_MODEL, device=prm_device)
    print("PRM loaded.")

    total = len(items)
    elapsed_times = []
    srm_correct_cnt, lrm_correct_cnt = 0, 0
    print(f"\n{'='*70}")
    print(f"  Episode Generation: {total} problems | thinking={THINK_MODE}")
    print(f"  Output: {output_path}")
    print(f"{'='*70}\n")

    for idx, item in enumerate(items):
        qid = item["id"]
        query = item["query"]
        answer = item["answer"]
        t0 = time.time()

        # ---- SRM solution ----
        t_srm = time.time()
        srm_text, srm_tok = srm.generate_solution(
            query, max_tokens=MAX_NEW_TOKENS, temperature=temperature,
            think_mode=THINK_MODE
        )
        srm_time = time.time() - t_srm
        srm_steps = split_steps(srm_text)[:MAX_STEPS]
        srm_prm = prm.score_trace(query, srm_steps) if srm_steps else []
        srm_tokens = _distribute_tokens(srm_steps, srm_tok)
        srm_answer = extract_answer(srm_text)
        srm_correct = check_correctness(srm_answer, answer)

        # ---- LRM solution ----
        t_lrm = time.time()
        lrm_text, lrm_tok = lrm.generate_solution(
            query, max_tokens=MAX_NEW_TOKENS, temperature=temperature,
            think_mode=THINK_MODE
        )
        lrm_time = time.time() - t_lrm
        lrm_steps = split_steps(lrm_text)[:MAX_STEPS]
        lrm_prm = prm.score_trace(query, lrm_steps) if lrm_steps else []
        lrm_tokens = _distribute_tokens(lrm_steps, lrm_tok)
        lrm_answer = extract_answer(lrm_text)
        lrm_correct = check_correctness(lrm_answer, answer)

        # ---- Build LRM alternatives for each SRM step position ----
        lrm_alt_steps, lrm_alt_prm, lrm_alt_tokens = [], [], []
        for si in range(len(srm_steps)):
            if si < len(lrm_steps):
                lrm_alt_steps.append(lrm_steps[si])
                lrm_alt_prm.append(lrm_prm[si] if si < len(lrm_prm) else 0.5)
                lrm_alt_tokens.append(lrm_tokens[si] if si < len(lrm_tokens) else 0)
            else:
                lrm_alt_steps.append(lrm_steps[-1] if lrm_steps else "")
                lrm_alt_prm.append(lrm_prm[-1] if lrm_prm else 0.5)
                lrm_alt_tokens.append(lrm_tokens[-1] if lrm_tokens else 0)

        episode = {
            "id": qid,
            "query": query,
            "answer": answer,
            "dataset": item.get("dataset", dataset_name),
            "srm_solution": srm_text,
            "srm_steps": srm_steps,
            "srm_prm_scores": srm_prm,
            "srm_token_counts": srm_tokens,
            "srm_total_tokens": srm_tok,
            "srm_correct": srm_correct,
            "srm_answer": srm_answer,
            "lrm_solution": lrm_text,
            "lrm_steps": lrm_alt_steps,
            "lrm_prm_scores": lrm_alt_prm,
            "lrm_token_counts": lrm_alt_tokens,
            "lrm_total_tokens": lrm_tok,
            "lrm_correct": lrm_correct,
            "lrm_answer": lrm_answer,
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

        dt = time.time() - t0
        elapsed_times.append(dt)
        srm_correct_cnt += int(srm_correct)
        lrm_correct_cnt += int(lrm_correct)
        avg_time = sum(elapsed_times) / len(elapsed_times)
        remaining = (total - idx - 1) * avg_time
        eta_min = remaining / 60
        eta_h = remaining / 3600

        pct = (idx + 1) / total * 100
        bar_len = 30
        filled = int(bar_len * (idx + 1) / total)
        bar = '█' * filled + '░' * (bar_len - filled)

        print(f"[{bar}] {idx+1}/{total} ({pct:.0f}%) | "
              f"SRM={'✓' if srm_correct else '✗'} LRM={'✓' if lrm_correct else '✗'} | "
              f"tok={srm_tok}/{lrm_tok} | "
              f"SRM {srm_time:.0f}s LRM {lrm_time:.0f}s Total {dt:.0f}s | "
              f"ETA: {eta_h:.1f}h ({eta_min:.0f}min)")
        if (idx + 1) % 10 == 0:
            print(f"  >> Running acc: SRM={srm_correct_cnt}/{idx+1} "
                  f"({srm_correct_cnt/(idx+1)*100:.1f}%) | "
                  f"LRM={lrm_correct_cnt}/{idx+1} "
                  f"({lrm_correct_cnt/(idx+1)*100:.1f}%)")

    # Print summary
    all_eps = load_jsonl(output_path)
    n = len(all_eps)
    s_acc = sum(1 for e in all_eps if e["srm_correct"]) / max(n, 1)
    l_acc = sum(1 for e in all_eps if e["lrm_correct"]) / max(n, 1)
    print(f"\nDone. {n} episodes → {output_path}")
    print(f"SRM accuracy: {s_acc:.3f}  |  LRM accuracy: {l_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prm_device", type=str, default=PRM_DEVICE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_think", action="store_true",
                        help="Disable think mode for faster generation")
    args = parser.parse_args()

    if args.no_think:
        import config
        config.THINK_MODE = False

    generate_episodes(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        prm_device=args.prm_device,
        temperature=args.temperature,
        max_workers=args.max_workers,
        resume=not args.no_resume,
    )
