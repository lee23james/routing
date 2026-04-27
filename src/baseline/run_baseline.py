"""Baseline evaluation: SRM-only and LRM-only on MATH-500 / AIME.

Calls models with thinking mode enabled via vLLM endpoints.
Output format: <think>reasoning</think>\\boxed{answer}

Usage:
    python -m baseline.run_baseline --model srm --dataset math500
    python -m baseline.run_baseline --model lrm --dataset aime
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import (
    BASELINE_SRM_URL, BASELINE_LRM_URL, SRM_PARAMS_B, LRM_PARAMS_B,
    SYSTEM_PROMPT, RESULTS_DIR, MAX_TOTAL_TOKENS,
)
from common.llm import generate_full_solution
from common.answer import extract_answer, check_correctness
from common.datasets import load_math500, load_aime


MODEL_MAP = {
    "srm": {"url": BASELINE_SRM_URL, "name": "qwen3-1.7b", "params_b": SRM_PARAMS_B},
    "lrm": {"url": BASELINE_LRM_URL, "name": "qwen3-14b",  "params_b": LRM_PARAMS_B},
}


def run_baseline(
    model_key: str,
    dataset_name: str,
    output_dir: str,
    limit: int = None,
    save_interval: int = 5,
):
    model_cfg = MODEL_MAP[model_key]
    url = model_cfg["url"]
    model_name = model_cfg["name"]
    params_b = model_cfg["params_b"]

    if dataset_name == "math500":
        items = load_math500()
    elif dataset_name == "aime":
        items = load_aime(2020, 2024)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if limit:
        items = items[:limit]

    tag = f"{model_name}-{dataset_name}"
    out_dir = os.path.join(output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    # Resume support
    done_ids = set()
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r["id"])
        print(f"Resuming: {len(done_ids)} already done")

    pending = [it for it in items if it["id"] not in done_ids]
    if not pending:
        print("All done.")
        _print_summary(results, tag, params_b)
        return

    print(f"\n{'='*60}")
    print(f"  Baseline: {tag}")
    print(f"  Model: {model_name} ({params_b}B) → {url}")
    print(f"  Dataset: {dataset_name} ({len(items)} total, {len(pending)} remaining)")
    print(f"  Think mode: True")
    print(f"{'='*60}\n")

    correct = sum(1 for r in results if r.get("is_correct"))
    total = len(results)
    total_tokens = sum(r.get("completion_tokens", 0) for r in results)
    total_time = sum(r.get("inference_time", 0) for r in results)

    for idx, item in enumerate(pending):
        t0 = time.time()
        try:
            resp = generate_full_solution(
                url, item["problem"], SYSTEM_PROMPT,
                max_tokens=MAX_TOTAL_TOKENS,
                think_mode=True,
                timeout=600,
            )
            content = resp["content"]
            comp_tokens = resp["tokens"]
            elapsed = resp["elapsed"]
        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")
            content = ""
            comp_tokens = 0
            elapsed = time.time() - t0

        pred = extract_answer(content)
        is_correct = check_correctness(pred, item["answer"])

        if is_correct:
            correct += 1
        total += 1
        total_tokens += comp_tokens
        total_time += elapsed

        result = {
            "id": item["id"],
            "problem": item["problem"][:200],
            "ground_truth": item["answer"],
            "predicted": pred,
            "is_correct": is_correct,
            "completion_tokens": comp_tokens,
            "inference_time": round(elapsed, 2),
            "response": content,
        }
        results.append(result)
        done_ids.add(item["id"])

        acc = correct / total * 100
        avg_tok = total_tokens / total
        flops_g = 2 * params_b * avg_tok
        af = acc / (flops_g / 1000) if flops_g > 0 else 0

        mark = "✓" if is_correct else "✗"
        bar_len = 30
        pct = (len(done_ids)) / len(items)
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        eta_s = (len(items) - len(done_ids)) * (total_time / total) if total > 0 else 0
        print(
            f"[{bar}] {len(done_ids)}/{len(items)} "
            f"| {mark} pred={pred[:20]} gt={item['answer']} "
            f"| Acc={acc:.1f}% tok={comp_tokens} {elapsed:.1f}s "
            f"| ETA {eta_s/60:.0f}m"
        )

        if len(done_ids) % save_interval == 0 or idx == len(pending) - 1:
            with open(results_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final save
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _print_summary(results, tag, params_b)
    _save_stats(results, tag, params_b, out_dir)


def _print_summary(results, tag, params_b):
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    tokens = sum(r.get("completion_tokens", 0) for r in results)
    ttime = sum(r.get("inference_time", 0) for r in results)

    acc = correct / total * 100 if total > 0 else 0
    avg_tok = tokens / total if total > 0 else 0
    avg_flops_g = 2 * params_b * avg_tok
    af = acc / (avg_flops_g / 1000) if avg_flops_g > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {tag} — Final Results")
    print(f"{'='*60}")
    print(f"  Samples:   {total}")
    print(f"  Correct:   {correct}")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Avg tokens:{avg_tok:.0f}")
    print(f"  Avg GFLOPs:{avg_flops_g:.1f}")
    print(f"  A/F ratio: {af:.4f}")
    print(f"  Total time:{ttime:.0f}s ({ttime/3600:.1f}h)")
    print(f"{'='*60}\n")


def _save_stats(results, tag, params_b, out_dir):
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    tokens = sum(r.get("completion_tokens", 0) for r in results)
    ttime = sum(r.get("inference_time", 0) for r in results)

    acc = correct / total * 100 if total > 0 else 0
    avg_tok = tokens / total if total > 0 else 0
    avg_flops_g = 2 * params_b * avg_tok

    stats = {
        "tag": tag,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "avg_tokens": round(avg_tok, 1),
        "avg_gflops": round(avg_flops_g, 1),
        "af_ratio": round(acc / (avg_flops_g / 1000), 4) if avg_flops_g > 0 else 0,
        "total_time_s": round(ttime, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["srm", "lrm"], required=True)
    parser.add_argument("--dataset", choices=["math500", "aime"], required=True)
    parser.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "baseline"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_baseline(args.model, args.dataset, args.output_dir, args.limit)
