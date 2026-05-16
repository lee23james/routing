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
import queue
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VLLM_SRM_PORT, VLLM_LRM_PORT, PRM_MODEL,
    EPISODES_DIR, MAX_STEPS, MAX_NEW_TOKENS, PRM_DEVICE, THINK_MODE,
    SYSTEM_PROMPT, MCQ_SYSTEM_PROMPT,
)
from vllm_client import VLLMClient
from models import PRMScorer, ServerPRMScorer, split_steps, extract_answer, check_correctness
from data.datasets import (
    load_math500, load_aime2025, load_aime_1983_2024, load_omnimath,
    load_omnimath7_9_test_100, load_omnimath_diff1_3_train_200,
    load_omnimath_diff3_4_test_200, load_omnimath_diff4_9_stratified_test_100,
    load_math_train, load_aime_2010_2024_part1_train,
    load_aime_2020_2024_part2_test, load_gpqa_main_train_200,
    load_gpqa_diamond_test_100, load_gsm8k_train_300,
    load_gsm8k_test_189, load_mmlu_stem_train_200,
    load_mmlu_stem_test_189, save_jsonl, load_jsonl,
)


def _is_multiple_choice_item(item: Dict) -> bool:
    dataset = item.get("dataset", "")
    return item.get("task_type") == "multiple_choice" or dataset.startswith(("gpqa", "mmlu_stem"))


def _distribute_tokens(steps: List[str], total_tokens: int) -> List[int]:
    """Distribute total token count proportionally across steps by char length."""
    if not steps or total_tokens <= 0:
        return [len(s.split()) for s in steps]
    char_lens = [max(len(s), 1) for s in steps]
    total_chars = sum(char_lens)
    per_step = [max(1, round(c / total_chars * total_tokens)) for c in char_lens]
    return per_step


def load_items_for_dataset(dataset_name: str) -> List[Dict]:
    if dataset_name == "math500":
        return load_math500()
    elif dataset_name == "math_train_200":
        return load_math_train()[:200]
    elif dataset_name in ("math_train_1k", "trim_math_train_1k"):
        return load_math_train()
    elif dataset_name in ("math500_test_100", "trim_math500_test_100"):
        return load_math500()
    elif dataset_name == "aime2025":
        return load_aime2025()
    elif dataset_name == "aime_2010_2024_part1_train":
        return load_aime_2010_2024_part1_train()
    elif dataset_name == "aime_2020_2024_part2_test":
        return load_aime_2020_2024_part2_test()
    elif dataset_name in ("aime_train", "trim_aime_train"):
        return load_aime_1983_2024()
    elif dataset_name in ("aime_test", "trim_aime_test"):
        return load_aime2025()
    elif dataset_name == "aime":
        return load_aime_1983_2024()
    elif dataset_name == "gpqa_main_train_200":
        return load_gpqa_main_train_200()
    elif dataset_name == "gpqa_diamond_test_100":
        return load_gpqa_diamond_test_100()
    elif dataset_name == "gsm8k_train_300":
        return load_gsm8k_train_300()
    elif dataset_name == "gsm8k_test_189":
        return load_gsm8k_test_189()
    elif dataset_name == "mmlu_stem_train_200":
        return load_mmlu_stem_train_200()
    elif dataset_name == "mmlu_stem_test_189":
        return load_mmlu_stem_test_189()
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
        return items
    elif dataset_name == "all":
        return load_math500() + load_aime2025()
    elif dataset_name == "omnimath":
        return load_omnimath(max_items=200, min_diff=1.0, max_diff=4.0)
    elif dataset_name == "omnimath_full":
        return load_omnimath(max_items=500)
    elif dataset_name == "omnimath7_9_test_100":
        return load_omnimath7_9_test_100()
    elif dataset_name == "omnimath_diff1_3_train_200":
        return load_omnimath_diff1_3_train_200()
    elif dataset_name == "omnimath_diff3_4_test_200":
        return load_omnimath_diff3_4_test_200()
    elif dataset_name == "omnimath_diff4_9_stratified_test_100":
        return load_omnimath_diff4_9_stratified_test_100()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_model_solutions_parallel(
    srm: VLLMClient,
    lrm: VLLMClient,
    query: str,
    max_new_tokens: int,
    temperature: float,
    think_mode: bool,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, tuple]:
    """Generate SRM and LRM solutions for one problem concurrently."""

    def _generate(client: VLLMClient):
        start = time.time()
        try:
            text, tokens = client.generate_solution(
                query,
                max_tokens=max_new_tokens,
                temperature=temperature,
                think_mode=think_mode,
                system_prompt=system_prompt,
            )
        except TypeError:
            text, tokens = client.generate_solution(
                query,
                max_tokens=max_new_tokens,
                temperature=temperature,
                think_mode=think_mode,
            )
        return text, tokens, time.time() - start

    with ThreadPoolExecutor(max_workers=2) as pool:
        srm_future = pool.submit(_generate, srm)
        lrm_future = pool.submit(_generate, lrm)
        return {
            "srm": srm_future.result(),
            "lrm": lrm_future.result(),
        }


def _generate_model_outputs_for_item(
    item: Dict,
    srm: VLLMClient,
    lrm: VLLMClient,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Dict, Dict[str, tuple], float]:
    is_mcq = _is_multiple_choice_item(item)
    system_prompt = MCQ_SYSTEM_PROMPT if is_mcq else SYSTEM_PROMPT
    t0 = time.time()
    model_outputs = generate_model_solutions_parallel(
        srm=srm,
        lrm=lrm,
        query=item["query"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        think_mode=THINK_MODE,
        system_prompt=system_prompt,
    )
    return item, model_outputs, time.time() - t0


def _build_episode_from_outputs(
    item: Dict,
    model_outputs: Dict[str, tuple],
    prm,
) -> Dict:
    qid = item["id"]
    query = item["query"]
    answer = item["answer"]
    is_mcq = _is_multiple_choice_item(item)

    srm_text, srm_tok, srm_time = model_outputs["srm"]
    lrm_text, lrm_tok, lrm_time = model_outputs["lrm"]

    srm_steps = split_steps(srm_text)[:MAX_STEPS]
    srm_prm = prm.score_trace(query, srm_steps) if srm_steps else []
    srm_tokens = _distribute_tokens(srm_steps, srm_tok)
    srm_answer = extract_answer(srm_text, mode="multiple_choice" if is_mcq else "math")
    srm_correct = check_correctness(srm_answer, answer, mode="multiple_choice" if is_mcq else "math")

    lrm_steps = split_steps(lrm_text)[:MAX_STEPS]
    lrm_prm = prm.score_trace(query, lrm_steps) if lrm_steps else []
    lrm_tokens = _distribute_tokens(lrm_steps, lrm_tok)
    lrm_answer = extract_answer(lrm_text, mode="multiple_choice" if is_mcq else "math")
    lrm_correct = check_correctness(lrm_answer, answer, mode="multiple_choice" if is_mcq else "math")

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
        "dataset": item.get("dataset", ""),
        "difficulty": item.get("difficulty"),
        "source": item.get("source", ""),
        "domain": item.get("domain", []),
        "source_path": item.get("source_path", ""),
        "source_index": item.get("source_index"),
        "source_id": item.get("source_id", qid),
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
        "_timing": {
            "srm_time": srm_time,
            "lrm_time": lrm_time,
        },
    }
    return episode


def _append_episode(output_path: str, episode: Dict) -> None:
    row = dict(episode)
    row.pop("_timing", None)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _print_generation_progress(
    *,
    done_count: int,
    total_count: int,
    episode: Dict,
    elapsed_times: List[float],
    srm_correct_cnt: int,
    lrm_correct_cnt: int,
    dt: float,
) -> None:
    avg_time = sum(elapsed_times) / len(elapsed_times)
    remaining = (total_count - done_count) * avg_time
    eta_min = remaining / 60
    eta_h = remaining / 3600

    pct = done_count / total_count * 100
    bar_len = 30
    filled = int(bar_len * done_count / total_count)
    bar = '█' * filled + '░' * (bar_len - filled)
    timing = episode.get("_timing", {})

    print(f"[{bar}] {done_count}/{total_count} ({pct:.0f}%) | "
          f"id={episode['id']} | "
          f"SRM={'✓' if episode['srm_correct'] else '✗'} "
          f"LRM={'✓' if episode['lrm_correct'] else '✗'} | "
          f"tok={episode['srm_total_tokens']}/{episode['lrm_total_tokens']} | "
          f"SRM {timing.get('srm_time', 0):.0f}s "
          f"LRM {timing.get('lrm_time', 0):.0f}s Total {dt:.0f}s | "
          f"ETA: {eta_h:.1f}h ({eta_min:.0f}min)")
    if done_count % 10 == 0:
        print(f"  >> Running acc: SRM={srm_correct_cnt}/{done_count} "
              f"({srm_correct_cnt/done_count*100:.1f}%) | "
              f"LRM={lrm_correct_cnt}/{done_count} "
              f"({lrm_correct_cnt/done_count*100:.1f}%)")


def _generate_episodes_sequential(
    items: List[Dict],
    output_path: str,
    prm,
    srm: VLLMClient,
    lrm: VLLMClient,
    max_new_tokens: int,
    temperature: float,
    done_offset: int,
    total_target: int,
) -> None:
    elapsed_times = []
    srm_correct_cnt, lrm_correct_cnt = 0, 0

    for idx, item in enumerate(items):
        t0 = time.time()
        item, model_outputs, _ = _generate_model_outputs_for_item(
            item=item,
            srm=srm,
            lrm=lrm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        episode = _build_episode_from_outputs(item, model_outputs, prm)
        _append_episode(output_path, episode)

        dt = time.time() - t0
        elapsed_times.append(dt)
        srm_correct_cnt += int(episode["srm_correct"])
        lrm_correct_cnt += int(episode["lrm_correct"])
        _print_generation_progress(
            done_count=done_offset + idx + 1,
            total_count=total_target,
            episode=episode,
            elapsed_times=elapsed_times,
            srm_correct_cnt=srm_correct_cnt,
            lrm_correct_cnt=lrm_correct_cnt,
            dt=dt,
        )


def _generate_episodes_pipelined(
    items: List[Dict],
    output_path: str,
    prm,
    srm: VLLMClient,
    lrm: VLLMClient,
    max_new_tokens: int,
    temperature: float,
    generation_workers: int,
    done_offset: int,
    total_target: int,
) -> None:
    result_queue: queue.Queue = queue.Queue(maxsize=max(1, generation_workers * 2))
    stop_token = object()
    errors = []

    def producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=generation_workers) as pool:
                futures = [
                    pool.submit(
                        _generate_model_outputs_for_item,
                        item,
                        srm,
                        lrm,
                        max_new_tokens,
                        temperature,
                    )
                    for item in items
                ]
                for fut in as_completed(futures):
                    result_queue.put(fut.result())
        except BaseException as exc:
            errors.append(exc)
        finally:
            result_queue.put(stop_token)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    elapsed_times = []
    srm_correct_cnt, lrm_correct_cnt = 0, 0
    completed = 0
    while True:
        payload = result_queue.get()
        if payload is stop_token:
            break

        item, model_outputs, generation_dt = payload
        t0 = time.time()
        episode = _build_episode_from_outputs(item, model_outputs, prm)
        _append_episode(output_path, episode)
        dt = time.time() - t0 + generation_dt

        completed += 1
        elapsed_times.append(dt)
        srm_correct_cnt += int(episode["srm_correct"])
        lrm_correct_cnt += int(episode["lrm_correct"])
        _print_generation_progress(
            done_count=done_offset + completed,
            total_count=total_target,
            episode=episode,
            elapsed_times=elapsed_times,
            srm_correct_cnt=srm_correct_cnt,
            lrm_correct_cnt=lrm_correct_cnt,
            dt=dt,
        )

    producer_thread.join()
    if errors:
        raise errors[0]


def generate_episodes(
    dataset_name: str,
    output_dir: str = None,
    prm_device: str = "cuda:0",
    srm_port: int = VLLM_SRM_PORT,
    lrm_port: int = VLLM_LRM_PORT,
    srm_server_url: str = None,
    lrm_server_url: str = None,
    srm_model_name: str = "srm",
    lrm_model_name: str = "lrm",
    prm_server_url: str = None,
    prm_model_name: str = PRM_MODEL,
    max_new_tokens: int = MAX_NEW_TOKENS,
    n_solutions: int = 1,
    temperature: float = 0.0,
    max_workers: int = 4,
    generation_workers: int = 1,
    client_timeout: int = 600,
    resume: bool = True,
):
    items = load_items_for_dataset(dataset_name)

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

    srm = VLLMClient(
        srm_port,
        model_name=srm_model_name,
        server_url=srm_server_url,
        timeout=client_timeout,
    )
    lrm = VLLMClient(
        lrm_port,
        model_name=lrm_model_name,
        server_url=lrm_server_url,
        timeout=client_timeout,
    )

    if prm_server_url:
        print(f"Using PRM server: {prm_server_url}")
        prm = ServerPRMScorer(
            prm_server_url,
            model_name=prm_model_name,
            max_workers=max_workers,
        )
    else:
        print(f"Loading PRM on {prm_device} ...")
        prm = PRMScorer(prm_model_name, device=prm_device)
    print("PRM ready.")

    total = len(items)
    elapsed_times = []
    srm_correct_cnt, lrm_correct_cnt = 0, 0
    print(f"\n{'='*70}")
    print(f"  Episode Generation: {total} remaining problems | thinking={THINK_MODE}")
    print(f"  Output: {output_path}")
    print(f"  Generation workers: {generation_workers}")
    print(f"  Client timeout: {client_timeout}s")
    print(f"{'='*70}\n")

    total_target = len(done_ids) + len(items)
    if generation_workers <= 1:
        _generate_episodes_sequential(
            items=items,
            output_path=output_path,
            prm=prm,
            srm=srm,
            lrm=lrm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            done_offset=len(done_ids),
            total_target=total_target,
        )
    else:
        _generate_episodes_pipelined(
            items=items,
            output_path=output_path,
            prm=prm,
            srm=srm,
            lrm=lrm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            generation_workers=generation_workers,
            done_offset=len(done_ids),
            total_target=total_target,
        )

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
    parser.add_argument("--srm_port", type=int, default=VLLM_SRM_PORT)
    parser.add_argument("--lrm_port", type=int, default=VLLM_LRM_PORT)
    parser.add_argument("--srm_server_url", type=str, default=None)
    parser.add_argument("--lrm_server_url", type=str, default=None)
    parser.add_argument("--srm_model_name", type=str, default="srm")
    parser.add_argument("--lrm_model_name", type=str, default="lrm")
    parser.add_argument("--prm_server_url", type=str, default=None,
                        help="Root PRM server URL, e.g. http://localhost:30002")
    parser.add_argument("--prm_model_name", type=str, default=PRM_MODEL)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--generation_workers",
        type=int,
        default=1,
        help="Number of problems to generate concurrently; each problem still calls SRM/LRM in parallel.",
    )
    parser.add_argument("--client_timeout", type=int, default=600)
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
        srm_port=args.srm_port,
        lrm_port=args.lrm_port,
        srm_server_url=args.srm_server_url,
        lrm_server_url=args.lrm_server_url,
        srm_model_name=args.srm_model_name,
        lrm_model_name=args.lrm_model_name,
        prm_server_url=args.prm_server_url,
        prm_model_name=args.prm_model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_workers=args.max_workers,
        generation_workers=args.generation_workers,
        client_timeout=args.client_timeout,
        resume=not args.no_resume,
    )
