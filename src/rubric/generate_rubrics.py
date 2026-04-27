"""RLCER-style rubric generation, filtering, and weight learning.

Pipeline:
  Phase 1: Score all episodes with seed + derived rubrics
  Phase 2: Learn rubric weights via correlation with correctness
  Phase 3: (Optional) Use LLM to generate extra rubric descriptions
  Phase 4: Filter rubrics by consistency and discriminability
  Phase 5: Save final rubric weights + diagnostics

Usage:
    python -m rubric.generate_rubrics --episodes_path data/episodes/all_episodes.jsonl
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VLLM_LRM_PORT, RUBRIC_DIR, RESULTS_DIR
from rubric.rubric_scorer import (
    SEED_RUBRICS, DERIVED_RUBRICS, ALL_RUBRICS,
    learn_rubric_weights, score_trajectory_rubrics,
)
from data.datasets import load_jsonl, save_jsonl


RUBRIC_GEN_PROMPT = """You are an expert at evaluating routing strategies for multi-step mathematical reasoning.

In a stepwise routing system, a weak model (SRM) generates each reasoning step. A router decides whether to accept the SRM step (action=continue) or regenerate it with a strong model (LRM, action=regenerate). The goal is to maximise answer correctness while minimising LRM usage.

Below are two routing trajectories for the same problem:

**Problem**: {query}
**Correct Answer**: {answer}

**Trajectory A** (result: {result_a}, LRM calls: {regen_a}/{steps_a} steps):
{traj_a_text}

**Trajectory B** (result: {result_b}, LRM calls: {regen_b}/{steps_b} steps):
{traj_b_text}

Please:
1. Analyse which routing strategy is better and why.
2. From your analysis, extract 3-5 general routing quality criteria (rubrics). Each rubric should:
   - Distinguish good routing decisions from bad ones
   - Be applicable to any math problem (not specific to this problem)
   - Be phrased as a clear yes/no evaluable statement

Output format (JSON list):
```json
[
  {{"name": "short_name", "description": "Clear description of the criterion", "importance": 0.0-1.0}},
  ...
]
```"""


def format_trajectory(prm_scores: List[float], actions: List[int],
                      num_steps: int) -> str:
    lines = []
    for i in range(min(num_steps, len(prm_scores))):
        action_str = "REGEN(LRM)" if (i < len(actions) and actions[i] == 1) else "continue(SRM)"
        prm_str = f"{prm_scores[i]:.3f}" if i < len(prm_scores) else "N/A"
        lines.append(f"  Step {i+1}: PRM={prm_str}, action={action_str}")
    return "\n".join(lines)


# ============================================================
# Phase 1 + 2: Learn weights for ALL rubrics (seed + derived)
# ============================================================

def compute_and_save_weights(
    episodes: List[Dict],
    output_dir: str,
    n_trajectories: int = 20,
    corr_threshold: float = 0.05,
    std_threshold: float = 0.02,
) -> Tuple[Dict[str, float], Dict]:
    """Learn rubric weights via RLCER-style correlation filtering.

    Uses diverse trajectory sampling and lower thresholds to retain
    more rubrics compared to v1.
    """
    print(f"Learning rubric weights from {len(episodes)} episodes ...")
    print(f"  Rubric pool: {len(SEED_RUBRICS)} seed + {len(DERIVED_RUBRICS)} derived"
          f" = {len(ALL_RUBRICS)} total")

    weights, diagnostics = learn_rubric_weights(
        episodes,
        rubric_set=ALL_RUBRICS,
        n_trajectories=n_trajectories,
        corr_threshold=corr_threshold,
        std_threshold=std_threshold,
    )

    print("\n  Learned rubric weights:")
    active_count = 0
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        diag = diagnostics.get(name, {})
        status = diag.get("status", "unknown")
        corr = diag.get("corr", 0)
        std = diag.get("std", 0)
        tag = "✓ ACTIVE" if w > 0 else "✗ filtered"
        if w > 0:
            active_count += 1
        print(f"    {name:35s}  w={w:.4f}  corr={corr:.4f}  std={std:.4f}  [{tag}]")

    print(f"\n  Active rubrics: {active_count}/{len(ALL_RUBRICS)}")

    result = {
        "weights": weights,
        "diagnostics": {
            k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in diagnostics.items()
        },
        "n_episodes": len(episodes),
        "n_trajectories_per_episode": n_trajectories,
        "corr_threshold": corr_threshold,
        "std_threshold": std_threshold,
        "active_rubrics": [k for k, v in weights.items() if v > 0],
        "seed_rubrics": list(SEED_RUBRICS.keys()),
        "derived_rubrics": list(DERIVED_RUBRICS.keys()),
    }

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "rubric_weights.json")
    with open(weights_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved weights → {weights_path}")

    return weights, result


# ============================================================
# Phase 3: Pre-score all episodes with rubrics
# ============================================================

def score_all_episodes(
    episodes: List[Dict],
    weights: Dict[str, float],
    output_dir: str,
) -> List[Dict]:
    """Score every episode under multiple routing strategies."""
    print("\n  Pre-scoring all episodes with rubrics ...")
    scored = []

    for ep in episodes:
        srm_prm = ep.get("srm_prm_scores", [])
        lrm_prm = ep.get("lrm_prm_scores", [])
        n = len(srm_prm)
        if n == 0:
            continue

        all_srm = [0] * n
        srm_scores = score_trajectory_rubrics(srm_prm, all_srm, lrm_prm, weights)

        all_lrm = [1] * n
        lrm_chosen_prm = lrm_prm if lrm_prm else srm_prm
        lrm_scores = score_trajectory_rubrics(lrm_chosen_prm, all_lrm, lrm_prm, weights)

        smart_actions = [1 if (i < len(srm_prm) and srm_prm[i] < 0.5) else 0
                         for i in range(n)]
        smart_prm = [
            lrm_prm[i] if smart_actions[i] == 1 and i < len(lrm_prm)
            else srm_prm[i] if i < len(srm_prm) else 0.5
            for i in range(n)
        ]
        smart_scores = score_trajectory_rubrics(smart_prm, smart_actions, lrm_prm, weights)

        scored.append({
            "id": ep["id"],
            "srm_correct": ep.get("srm_correct", False),
            "lrm_correct": ep.get("lrm_correct", False),
            "srm_rubric_scores": srm_scores,
            "lrm_rubric_scores": lrm_scores,
            "smart_rubric_scores": smart_scores,
        })

    save_jsonl(scored, os.path.join(output_dir, "episode_rubric_scores.jsonl"))
    print(f"  Saved rubric scores for {len(scored)} episodes")
    return scored


# ============================================================
# Phase 4 (Optional): LLM-generated rubric descriptions
# ============================================================

def build_contrastive_pairs(episodes: List[Dict],
                            max_pairs: int = 100) -> List[Tuple]:
    pairs = []
    for ep in episodes[:max_pairs]:
        srm_prm = ep.get("srm_prm_scores", [])
        lrm_prm = ep.get("lrm_prm_scores", [])
        n = len(srm_prm)
        if n == 0:
            continue

        traj_a = {"actions": [0] * n, "prm": srm_prm,
                  "correct": ep.get("srm_correct", False), "n_regens": 0}
        smart_actions = [1 if (i < len(srm_prm) and srm_prm[i] < 0.5) else 0
                         for i in range(n)]
        smart_prm = [lrm_prm[i] if smart_actions[i] == 1 and i < len(lrm_prm)
                      else srm_prm[i] for i in range(n)]
        traj_b = {"actions": smart_actions, "prm": smart_prm,
                  "correct": ep.get("lrm_correct", False),
                  "n_regens": sum(smart_actions)}

        pairs.append((ep, traj_a, traj_b))
        if len(pairs) >= max_pairs:
            break
    return pairs


def generate_llm_rubrics(episodes: List[Dict],
                         lrm_port: int,
                         max_pairs: int = 30,
                         output_path: str = None) -> List[Dict]:
    """Use LRM to generate rubric candidates from trajectory pairs."""
    from vllm_client import VLLMClient
    lrm_client = VLLMClient(lrm_port, model_name="lrm")
    pairs = build_contrastive_pairs(episodes, max_pairs)
    all_rubrics = []

    for idx, (ep, traj_a, traj_b) in enumerate(pairs):
        prompt = RUBRIC_GEN_PROMPT.format(
            query=ep["query"][:500],
            answer=ep["answer"],
            result_a="correct" if traj_a["correct"] else "incorrect",
            regen_a=traj_a["n_regens"],
            steps_a=len(traj_a["prm"]),
            traj_a_text=format_trajectory(traj_a["prm"], traj_a["actions"],
                                          len(traj_a["prm"])),
            result_b="correct" if traj_b["correct"] else "incorrect",
            regen_b=traj_b["n_regens"],
            steps_b=len(traj_b["prm"]),
            traj_b_text=format_trajectory(traj_b["prm"], traj_b["actions"],
                                          len(traj_b["prm"])),
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            data = lrm_client._call(messages, max_tokens=1024, temperature=0.3)
            response = data["choices"][0]["message"]["content"]
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                rubrics = json.loads(json_match.group())
                for r in rubrics:
                    r["source_query_id"] = ep["id"]
                    r["pair_idx"] = idx
                all_rubrics.extend(rubrics)
                print(f"    Pair {idx+1}/{len(pairs)}: extracted {len(rubrics)} rubrics")
        except Exception as e:
            print(f"    Pair {idx+1}/{len(pairs)}: error — {e}")
            continue

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_jsonl(all_rubrics, output_path)
        print(f"  Saved {len(all_rubrics)} LLM-generated rubric candidates → {output_path}")

    return all_rubrics


# ============================================================
# Phase 5: Consistency-based verification (OpenRubric-inspired)
# ============================================================

def verify_rubric_consistency(
    episodes: List[Dict],
    weights: Dict[str, float],
    n_verify: int = 50,
) -> Dict[str, float]:
    """Verify rubrics by checking if high-rubric-score trajectories are
    more often correct than low-rubric-score ones (preference consistency).

    Returns per-rubric consistency scores.
    """
    print("\n  Verifying rubric consistency (OpenRubric-style) ...")
    consistency = {name: [] for name in weights if weights[name] > 0}

    rng = np.random.RandomState(42)
    for ep in episodes[:n_verify]:
        srm_prm = ep.get("srm_prm_scores", [])
        lrm_prm = ep.get("lrm_prm_scores", [])
        n = len(srm_prm)
        if n == 0:
            continue

        pairs = []
        for _ in range(5):
            a1 = [int(rng.random() < 0.3) for _ in range(n)]
            a2 = [int(rng.random() < 0.3) for _ in range(n)]
            c1 = _estimate_correctness(a1, ep, n, srm_prm, lrm_prm)
            c2 = _estimate_correctness(a2, ep, n, srm_prm, lrm_prm)
            if c1 != c2:
                pairs.append((a1, a2, c1, c2))

        for a1, a2, c1, c2 in pairs:
            p1 = _make_chosen_prm(a1, srm_prm, lrm_prm, n)
            p2 = _make_chosen_prm(a2, srm_prm, lrm_prm, n)
            s1 = score_trajectory_rubrics(p1, a1, lrm_prm, weights)
            s2 = score_trajectory_rubrics(p2, a2, lrm_prm, weights)
            for name in consistency:
                r1, r2 = s1.get(name, 0.5), s2.get(name, 0.5)
                if abs(r1 - r2) > 0.01:
                    correct_is_better = (r1 > r2) == c1
                    consistency[name].append(1.0 if correct_is_better else 0.0)

    result = {}
    for name, vals in consistency.items():
        if vals:
            result[name] = float(np.mean(vals))
            print(f"    {name:35s}  consistency={result[name]:.4f}  (n={len(vals)})")
        else:
            result[name] = 0.5
    return result


def _estimate_correctness(actions, ep, n, srm_prm, lrm_prm):
    n_regens = sum(actions)
    if n_regens == 0:
        return ep.get("srm_correct", False)
    if n_regens == n:
        return ep.get("lrm_correct", False)
    return ep.get("lrm_correct", False) if n_regens > n * 0.3 else ep.get("srm_correct", False)


def _make_chosen_prm(actions, srm_prm, lrm_prm, n):
    chosen = []
    for i in range(n):
        if actions[i] == 1 and i < len(lrm_prm):
            chosen.append(lrm_prm[i])
        elif i < len(srm_prm):
            chosen.append(srm_prm[i])
        else:
            chosen.append(0.5)
    return chosen


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=RUBRIC_DIR)
    parser.add_argument("--gen_llm_rubrics", action="store_true",
                        help="Also generate LLM-based rubrics (slower)")
    parser.add_argument("--max_pairs", type=int, default=30)
    parser.add_argument("--n_trajectories", type=int, default=20)
    parser.add_argument("--corr_threshold", type=float, default=0.05)
    parser.add_argument("--std_threshold", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    episodes = load_jsonl(args.episodes_path)
    print(f"Loaded {len(episodes)} episodes")

    print("\n" + "=" * 60)
    print("  Phase 1-2: Learn rubric weights (RLCER-style)")
    print("=" * 60)
    weights, result_data = compute_and_save_weights(
        episodes,
        output_dir=args.output_dir,
        n_trajectories=args.n_trajectories,
        corr_threshold=args.corr_threshold,
        std_threshold=args.std_threshold,
    )

    print("\n" + "=" * 60)
    print("  Phase 3: Pre-score episodes")
    print("=" * 60)
    score_all_episodes(episodes, weights, args.output_dir)

    print("\n" + "=" * 60)
    print("  Phase 4: Consistency verification")
    print("=" * 60)
    consistency = verify_rubric_consistency(episodes, weights)

    consistency_path = os.path.join(args.output_dir, "rubric_consistency.json")
    with open(consistency_path, "w") as f:
        json.dump(consistency, f, indent=2)
    print(f"  Saved consistency scores → {consistency_path}")

    if args.gen_llm_rubrics:
        print("\n" + "=" * 60)
        print("  Phase 5: LLM-based rubric generation")
        print("=" * 60)
        generate_llm_rubrics(
            episodes, VLLM_LRM_PORT, max_pairs=args.max_pairs,
            output_path=os.path.join(args.output_dir, "llm_rubric_candidates.jsonl"),
        )

    print("\n" + "=" * 60)
    print("  Rubric pipeline complete!")
    print("=" * 60)
    active = result_data["active_rubrics"]
    print(f"  Active rubrics ({len(active)}/{len(ALL_RUBRICS)}): {active}")


if __name__ == "__main__":
    main()
