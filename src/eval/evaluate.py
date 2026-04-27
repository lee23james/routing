"""Evaluation for stepwise routing methods via vLLM API.

Saves ALL outputs to /export/shy/pp/pp5/results/:
  - Per-problem inference outputs (solution text, predicted answer, actions)
  - Accuracy, cost metrics, PGR
  - Summary comparison tables

Supports:
  1. SRM-only / LRM-only baselines
  2. TRIM-Agg trained router (online stepwise routing)
  3. TRIM-Rubric trained router
  4. Offline evaluation from pre-generated episodes

Usage:
    python -m eval.evaluate --dataset math500 --mode all
    python -m eval.evaluate --mode offline --episodes_path data/episodes/all_episodes.jsonl \
        --checkpoint checkpoints/trim_agg/best.pt
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

from config import (
    VLLM_SRM_PORT, VLLM_LRM_PORT, PRM_MODEL,
    STATE_DIM, HIDDEN_DIM, ACTION_DIM,
    MAX_STEPS, RESULTS_DIR, PRM_DEVICE, SYSTEM_PROMPT,
    TOKEN_NORMALISER,
)
from vllm_client import VLLMClient
from models import PRMScorer, split_steps, extract_answer, check_correctness
from data.datasets import load_math500, load_aime2025
from router.policy import RouterPolicy


def _save_results(output_dir: str, name: str, data: Dict):
    """Save full results (with per-problem details) to results dir."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {path}")
    return path


def evaluate_model_only(client: VLLMClient, items: List[Dict],
                        model_label: str, output_dir: str) -> Dict:
    """Baseline: all steps from one model. Saves full outputs."""
    correct = 0
    total_tokens = 0
    per_problem = []

    for idx, item in enumerate(items):
        text, tokens = client.generate_solution(item["query"])
        pred = extract_answer(text)
        is_correct = check_correctness(pred, item["answer"])
        correct += int(is_correct)
        total_tokens += tokens
        per_problem.append({
            "id": item["id"],
            "query": item["query"][:200],
            "ground_truth": item["answer"],
            "predicted": pred,
            "correct": is_correct,
            "tokens": tokens,
            "solution": text,
        })
        if (idx + 1) % 50 == 0:
            print(f"    [{idx+1}/{len(items)}] acc so far: {correct/(idx+1):.3f}")

    n = max(len(items), 1)
    accuracy = correct / n
    avg_tokens = total_tokens / n
    result = {
        "method": model_label,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(items),
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
        "lrm_tokens_avg": avg_tokens if "lrm" in model_label else 0,
        "srm_tokens_avg": avg_tokens if "srm" in model_label else 0,
        "per_problem": per_problem,
    }
    _save_results(output_dir, model_label, result)
    return result


def evaluate_router(
    srm: VLLMClient,
    lrm: VLLMClient,
    prm: PRMScorer,
    policy: RouterPolicy,
    items: List[Dict],
    device: str = "cpu",
    deterministic: bool = True,
    label: str = "router",
    output_dir: str = ".",
) -> Dict:
    """Online stepwise routing evaluation with full output logging."""
    correct = 0
    total_lrm_tokens = 0
    total_srm_tokens = 0
    per_problem = []

    for idx, item in enumerate(items):
        query = item["query"]
        answer = item["answer"]

        steps_taken = []
        prm_scores = []
        chosen_prm_scores = []
        actions = []
        lrm_toks = 0
        srm_toks = 0

        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        assistant_text = ""

        for t in range(MAX_STEPS):
            if assistant_text:
                messages = prefix_messages + [
                    {"role": "assistant", "content": assistant_text},
                ]
                data = srm._call(messages, max_tokens=512, temperature=0.0,
                                 stop=["\n\n"])
                srm_step = data["choices"][0]["message"]["content"].strip()
                s_tok = data.get("usage", {}).get("completion_tokens", 0)
            else:
                srm_step_text, s_tok = srm.generate_step(prefix_messages)
                srm_step = srm_step_text

            if not srm_step.strip():
                break

            all_steps = steps_taken + [srm_step]
            prm_list = prm.score_trace(query, all_steps)
            r_t = prm_list[-1] if prm_list else 0.5

            min_prev = min(chosen_prm_scores) if chosen_prm_scores else 1.0
            prod_prev = math.prod(chosen_prm_scores) if chosen_prm_scores else 1.0
            c_t = len(srm_step.split()) / TOKEN_NORMALISER
            t_norm = t / MAX_STEPS
            state = torch.FloatTensor([min_prev, prod_prev, r_t, c_t, t_norm]).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _ = policy.get_action(state, deterministic=deterministic)
            action_int = action.item()
            actions.append(action_int)

            if action_int == 1:
                if assistant_text:
                    lrm_msgs = prefix_messages + [
                        {"role": "assistant", "content": assistant_text},
                    ]
                else:
                    lrm_msgs = prefix_messages
                try:
                    data = lrm._call(lrm_msgs, max_tokens=512, temperature=0.0,
                                     stop=["\n\n"])
                    chosen = data["choices"][0]["message"]["content"].strip()
                    l_tok = data.get("usage", {}).get("completion_tokens", 0)
                except Exception:
                    chosen = srm_step
                    l_tok = 0
                lrm_toks += l_tok
                srm_toks += s_tok
            else:
                chosen = srm_step
                srm_toks += s_tok

            steps_taken.append(chosen)
            prm_scores.append(r_t)
            if action_int == 1:
                chosen_scores = prm.score_trace(query, steps_taken)
                chosen_prm_scores.append(chosen_scores[-1] if chosen_scores else r_t)
            else:
                chosen_prm_scores.append(r_t)
            assistant_text = (assistant_text + "\n\n" + chosen).strip() if assistant_text else chosen

            if "\\boxed{" in chosen or "answer is" in chosen.lower():
                break

        full_solution = "\n\n".join(steps_taken)
        pred = extract_answer(full_solution)
        is_correct = check_correctness(pred, answer)
        correct += int(is_correct)
        total_lrm_tokens += lrm_toks
        total_srm_tokens += srm_toks

        per_problem.append({
            "id": item["id"],
            "query": item["query"][:200],
            "ground_truth": answer,
            "predicted": pred,
            "correct": is_correct,
            "num_steps": len(steps_taken),
            "num_regens": sum(actions),
            "actions": actions,
            "prm_scores": [round(s, 4) for s in prm_scores],
            "srm_tokens": srm_toks,
            "lrm_tokens": lrm_toks,
            "solution": full_solution,
        })

        if (idx + 1) % 20 == 0:
            print(f"    [{idx+1}/{len(items)}] acc={correct/(idx+1):.3f}  "
                  f"lrm_tok_avg={total_lrm_tokens/(idx+1):.0f}")

    n = max(len(items), 1)
    accuracy = correct / n
    result = {
        "method": label,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(items),
        "srm_tokens_avg": total_srm_tokens / n,
        "lrm_tokens_avg": total_lrm_tokens / n,
        "total_lrm_tokens": total_lrm_tokens,
        "total_srm_tokens": total_srm_tokens,
        "avg_regens": sum(p["num_regens"] for p in per_problem) / n,
        "avg_steps": sum(p["num_steps"] for p in per_problem) / n,
        "regen_ratio": (sum(p["num_regens"] for p in per_problem)
                        / max(sum(p["num_steps"] for p in per_problem), 1)),
        "per_problem": per_problem,
    }
    _save_results(output_dir, label, result)
    return result


def evaluate_offline(episodes_path: str, checkpoint: str,
                     device: str = "cpu", label: str = "offline",
                     lam_values: Optional[list] = None,
                     output_dir: str = ".") -> Dict:
    """Offline evaluation using pre-generated episodes.

    Saves full per-problem routing decisions and metrics.
    """
    from router.env import TRIMEnv

    env = TRIMEnv(episodes_path)
    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()

    if lam_values is None:
        lam_values = [3e-4]

    all_results = {}
    for lam in lam_values:
        correct = 0
        total_lrm = 0
        n_regens_total = 0
        n_steps_total = 0
        per_problem = []

        for i in range(env.num_episodes):
            state = env.reset(i)
            done = False
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _ = policy.get_action(state_t, deterministic=True)
                state, _, done, _ = env.step(action.item())

            reward = env.compute_episode_reward(lam)
            info = env.get_episode_info()
            ep = env.current_ep

            is_correct = (reward > -lam * info["total_lrm_tokens"])
            correct += int(is_correct)
            total_lrm += info["total_lrm_tokens"]
            n_regens_total += info["num_regens"]
            n_steps_total += info["num_steps"]

            per_problem.append({
                "id": info["query_id"],
                "query": ep.get("query", "")[:200],
                "ground_truth": ep.get("answer", ""),
                "srm_answer": ep.get("srm_answer", ""),
                "lrm_answer": ep.get("lrm_answer", ""),
                "srm_correct": ep.get("srm_correct", False),
                "lrm_correct": ep.get("lrm_correct", False),
                "router_correct": is_correct,
                "actions": info["actions"],
                "num_regens": info["num_regens"],
                "num_steps": info["num_steps"],
                "total_lrm_tokens": info["total_lrm_tokens"],
                "prm_scores": [round(s, 4) for s in info["prm_scores"]],
                "reward": round(reward, 6),
            })

        n = max(env.num_episodes, 1)
        srm_acc = sum(1 for p in per_problem if p["srm_correct"]) / n
        lrm_acc = sum(1 for p in per_problem if p["lrm_correct"]) / n
        router_acc = correct / n
        pgr = (router_acc - srm_acc) / max(lrm_acc - srm_acc, 1e-8) if lrm_acc > srm_acc else 0.0

        lam_label = f"{label}_lam{lam:.0e}"
        res = {
            "method": lam_label,
            "lam": lam,
            "accuracy": router_acc,
            "pgr": round(pgr, 4),
            "correct": correct,
            "total": n,
            "srm_accuracy": srm_acc,
            "lrm_accuracy": lrm_acc,
            "avg_lrm_tokens": total_lrm / n,
            "avg_regens": n_regens_total / n,
            "avg_steps": n_steps_total / n,
            "regen_ratio": n_regens_total / max(n_steps_total, 1),
            "lrm_cost_ratio": n_regens_total / max(n_steps_total, 1),
            "per_problem": per_problem,
        }
        all_results[lam_label] = res

        _save_results(output_dir, lam_label, res)

        print(f"  λ={lam:.0e}  acc={router_acc:.4f}  PGR={pgr:.2%}  "
              f"avg_lrm_tok={res['avg_lrm_tokens']:.0f}  "
              f"regen_ratio={res['regen_ratio']:.2%}")

    return all_results


def print_summary(all_results: Dict):
    print("\n" + "=" * 100)
    print(f"{'Method':<35} {'Acc':>8} {'PGR':>8} {'LRM Tok':>10} "
          f"{'Regen%':>8} {'Steps':>7}")
    print("-" * 100)
    for name, r in all_results.items():
        acc = f"{r['accuracy']:.4f}"
        pgr = f"{r.get('pgr', 'N/A')}"
        if isinstance(r.get('pgr'), (int, float)):
            pgr = f"{r['pgr']:.2%}"
        lrm = f"{r.get('lrm_tokens_avg', r.get('avg_lrm_tokens', r.get('avg_tokens', 0))):.0f}"
        regen = "N/A"
        steps = "N/A"
        per = r.get("per_problem", [])
        if per and "num_regens" in per[0]:
            regen_tot = sum(p.get("num_regens", 0) for p in per)
            steps_tot = sum(p.get("num_steps", 1) for p in per)
            regen = f"{regen_tot / max(steps_tot, 1):.2%}"
            steps = f"{steps_tot / len(per):.1f}"
        print(f"{name:<35} {acc:>8} {pgr:>8} {lrm:>10} {regen:>8} {steps:>7}")
    print("=" * 100)


def save_comparison_summary(all_results: Dict, output_dir: str):
    """Save a clean comparison summary (without per-problem details)."""
    summary = {}
    for name, r in all_results.items():
        summary[name] = {k: v for k, v in r.items() if k != "per_problem"}
    path = os.path.join(output_dir, "comparison_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nComparison summary → {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math500",
                        choices=["math500", "aime2025", "all"])
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "srm_only", "lrm_only", "router",
                                 "offline", "baselines"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_rubric", type=str, default=None)
    parser.add_argument("--episodes_path", type=str, default=None)
    parser.add_argument("--prm_device", type=str, default=PRM_DEVICE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lam_values", type=str, default="8e-4,5e-4,3e-4,1e-4")
    args = parser.parse_args()

    lam_values = [float(x) for x in args.lam_values.split(",")]

    if args.dataset == "all":
        items = load_math500() + load_aime2025()
    elif args.dataset == "math500":
        items = load_math500()
    else:
        items = load_aime2025()
    print(f"Evaluating on {args.dataset}: {len(items)} problems")

    if args.output_dir is None:
        args.output_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # ---- Offline mode ----
    if args.mode == "offline" and args.episodes_path:
        print("\n=== Offline Evaluation ===")

        if args.checkpoint:
            print(f"\n--- TRIM-Agg ---")
            r = evaluate_offline(
                args.episodes_path, args.checkpoint,
                device=args.device, label="trim_agg",
                lam_values=lam_values,
                output_dir=args.output_dir,
            )
            all_results.update(r)

        if args.checkpoint_rubric:
            print(f"\n--- TRIM-Rubric ---")
            r = evaluate_offline(
                args.episodes_path, args.checkpoint_rubric,
                device=args.device, label="trim_rubric",
                lam_values=lam_values,
                output_dir=args.output_dir,
            )
            all_results.update(r)

        print_summary(all_results)
        save_comparison_summary(all_results, args.output_dir)
        return

    # ---- Online modes ----
    if args.mode in ("all", "srm_only", "baselines"):
        print("\n=== SRM-only ===")
        srm = VLLMClient(VLLM_SRM_PORT)
        r = evaluate_model_only(srm, items, "srm_only", args.output_dir)
        all_results["srm_only"] = r
        print(f"SRM accuracy: {r['accuracy']:.4f}  avg_tokens: {r['avg_tokens']:.0f}")

    if args.mode in ("all", "lrm_only", "baselines"):
        print("\n=== LRM-only ===")
        lrm_client = VLLMClient(VLLM_LRM_PORT)
        r = evaluate_model_only(lrm_client, items, "lrm_only", args.output_dir)
        all_results["lrm_only"] = r
        print(f"LRM accuracy: {r['accuracy']:.4f}  avg_tokens: {r['avg_tokens']:.0f}")

    if args.mode in ("all", "router"):
        if not args.checkpoint:
            print("Skipping router: no --checkpoint")
        else:
            print(f"\nLoading PRM on {args.prm_device} ...")
            prm = PRMScorer(PRM_MODEL, device=args.prm_device)
            srm = VLLMClient(VLLM_SRM_PORT)
            lrm_client = VLLMClient(VLLM_LRM_PORT)

            print("\n=== TRIM-Agg Router ===")
            policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(args.device)
            policy.load_state_dict(
                torch.load(args.checkpoint, map_location=args.device))
            policy.eval()
            r = evaluate_router(
                srm, lrm_client, prm, policy, items,
                device=args.device, label="trim_agg",
                output_dir=args.output_dir)
            all_results["trim_agg"] = r
            print(f"Accuracy: {r['accuracy']:.4f}  LRM tokens: {r['lrm_tokens_avg']:.0f}")

            if args.checkpoint_rubric:
                print("\n=== TRIM-Rubric Router ===")
                policy_r = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(args.device)
                policy_r.load_state_dict(
                    torch.load(args.checkpoint_rubric, map_location=args.device))
                policy_r.eval()
                r = evaluate_router(
                    srm, lrm_client, prm, policy_r, items,
                    device=args.device, label="trim_rubric",
                    output_dir=args.output_dir)
                all_results["trim_rubric"] = r
                print(f"Accuracy: {r['accuracy']:.4f}  LRM tokens: {r['lrm_tokens_avg']:.0f}")

    print_summary(all_results)
    save_comparison_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
