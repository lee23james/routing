"""Router-feedback evolution for TRIM-RubricV2 rubric weights.

V2 keeps the existing Python rubric functions fixed and evolves only their
weights using a trained router's real routing utility:

    utility = correctness - lam * total_lrm_tokens

The rubric process reward is intentionally excluded from the feedback signal.
This lets Router0 tell us which rubric scores correlate with the task/cost
trade-off that the final Router1 should optimise.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ACTION_DIM, DROPOUT, HIDDEN_DIM, STATE_DIM
from router.env import TRIMEnv
from router.policy import RouterPolicy
from rubric.rubric_scorer import ALL_RUBRICS, score_trajectory_rubrics


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize positive weights while preserving every input key."""
    clean = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        if not clean:
            return {}
        equal = 1.0 / len(clean)
        return {name: equal for name in clean}
    return {name: value / total for name, value in clean.items()}


def smooth_weights(
    base_weights: Dict[str, float],
    router_weights: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """Blend base and router-feedback weights and renormalize."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    keys = list(dict.fromkeys([*base_weights.keys(), *router_weights.keys()]))
    base = normalize_weights({key: base_weights.get(key, 0.0) for key in keys})
    router = normalize_weights({key: router_weights.get(key, 0.0) for key in keys})
    mixed = {
        key: (1.0 - alpha) * base.get(key, 0.0) + alpha * router.get(key, 0.0)
        for key in keys
    }
    return normalize_weights(mixed)


def compute_router_utility(correct: bool, total_lrm_tokens: int, lam: float) -> float:
    """Router-feedback target. Rubric reward is deliberately not included."""
    return (1.0 if correct else 0.0) - float(lam) * float(total_lrm_tokens)


def load_rubric_weight_file(path: str) -> Tuple[Dict[str, float], Dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"rubric weight file has no non-empty weights dict: {path}")
    return normalize_weights({str(k): float(v) for k, v in weights.items()}), payload


def _checkpoint_state_dict(checkpoint_path: str, device: str) -> Dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _policy_action(policy: RouterPolicy, state: np.ndarray, device: str, deterministic: bool) -> int:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            h = policy.encoder(state_t)
            logits = policy.actor(h)
            return int(torch.argmax(logits, dim=-1).item())
        action, _log_prob, _value = policy.get_action(state_t)
        return int(action.item())


def rollout_policy_on_episodes(
    episodes_path: str,
    checkpoint_path: str,
    device: str = "cpu",
    max_steps: int = 30,
    deterministic: bool = True,
    limit: Optional[int] = None,
    lam: float = 0.0,
    rubric_names: Optional[Iterable[str]] = None,
) -> List[Dict]:
    """Run Router0 on episodes and collect utilities plus raw rubric scores."""
    env = TRIMEnv(episodes_path, max_steps=max_steps, rubric_weights=None)
    if env.num_episodes == 0:
        raise ValueError(f"no usable episodes loaded from {episodes_path}")

    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM, DROPOUT).to(device)
    policy.load_state_dict(_checkpoint_state_dict(checkpoint_path, device))
    policy.eval()

    names = list(rubric_names or ALL_RUBRICS.keys())
    rubric_set = {name: ALL_RUBRICS[name] for name in names if name in ALL_RUBRICS}
    if not rubric_set:
        raise ValueError("no requested rubric names exist in ALL_RUBRICS")

    n = env.num_episodes if limit is None else min(int(limit), env.num_episodes)
    rollouts = []
    for idx in range(n):
        ep = env.episodes[idx]
        state = env.reset(idx)
        done = False
        while not done:
            action = _policy_action(policy, state, device, deterministic)
            state, _reward, done, _info = env.step(action)

        info = env.get_episode_info()
        correct = env._is_correct()
        utility = compute_router_utility(correct, info["total_lrm_tokens"], lam)
        scores = score_trajectory_rubrics(
            env.prm_scores,
            env.actions,
            ep.get("lrm_prm_scores", []),
            rubric_set=rubric_set,
        )
        rollouts.append(
            {
                "episode_id": ep.get("id", str(idx)),
                "correct": bool(correct),
                "total_lrm_tokens": int(info["total_lrm_tokens"]),
                "utility": float(utility),
                "actions": list(info["actions"]),
                "rubric_scores": {name: float(scores.get(name, 0.5)) for name in rubric_set},
            }
        )

    return rollouts


def _fallback_router_weights(
    rubric_names: List[str],
    fallback_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if fallback_weights:
        return normalize_weights({name: fallback_weights.get(name, 0.0) for name in rubric_names})
    equal = 1.0 / max(len(rubric_names), 1)
    return {name: equal for name in rubric_names}


def learn_router_feedback_weights(
    rollouts: List[Dict],
    rubric_names: Iterable[str],
    corr_threshold: float = 0.0,
    std_threshold: float = 0.02,
    fallback_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Learn rubric weights by positive correlation with router utility."""
    names = list(rubric_names)
    diagnostics = {}
    if not rollouts:
        weights = _fallback_router_weights(names, fallback_weights)
        diagnostics = {
            name: {"corr": 0.0, "std": 0.0, "status": "fallback_empty"}
            for name in names
        }
        return weights, diagnostics

    utilities = np.array([float(row.get("utility", 0.0)) for row in rollouts], dtype=np.float64)
    utility_std = float(np.std(utilities))
    if utility_std < 1e-12:
        weights = _fallback_router_weights(names, fallback_weights)
        diagnostics = {
            name: {"corr": 0.0, "std": 0.0, "utility_std": utility_std, "status": "fallback_constant_utility"}
            for name in names
        }
        return weights, diagnostics

    raw = {}
    for name in names:
        arr = np.array(
            [float(row.get("rubric_scores", {}).get(name, 0.5)) for row in rollouts],
            dtype=np.float64,
        )
        arr_std = float(np.std(arr))
        if arr_std < std_threshold:
            raw[name] = 0.0
            diagnostics[name] = {
                "corr": 0.0,
                "std": arr_std,
                "utility_std": utility_std,
                "status": "low_std",
            }
            continue

        corr, pval = stats.pearsonr(arr, utilities)
        if math.isnan(corr):
            raw[name] = 0.0
            diagnostics[name] = {
                "corr": 0.0,
                "pval": 1.0,
                "std": arr_std,
                "utility_std": utility_std,
                "mean": float(np.mean(arr)),
                "status": "nan_corr",
            }
            continue

        status = "active" if corr > corr_threshold else "negative_corr" if corr < 0 else "low_corr"
        raw[name] = float(corr) if status == "active" else 0.0
        diagnostics[name] = {
            "corr": float(corr),
            "pval": float(pval),
            "std": arr_std,
            "utility_std": utility_std,
            "mean": float(np.mean(arr)),
            "status": status,
        }

    if sum(raw.values()) <= 0:
        weights = _fallback_router_weights(names, fallback_weights)
        for name in names:
            diagnostics.setdefault(name, {"corr": 0.0, "std": 0.0})
            diagnostics[name]["status"] = f"fallback_{diagnostics[name].get('status', 'inactive')}"
        return weights, diagnostics

    return normalize_weights(raw), diagnostics


def build_output_payload(
    weights: Dict[str, float],
    base_weights: Dict[str, float],
    router_feedback_weights: Dict[str, float],
    diagnostics: Dict[str, Dict],
    alpha: float,
    lam: float,
    router_checkpoint: str,
    n_rollouts: int,
    extra_metadata: Optional[Dict] = None,
) -> Dict:
    normalized = normalize_weights(weights)
    payload = {
        "method": "trim_rubric_v2_router_feedback",
        "weights": normalized,
        "active_rubrics": [name for name, value in normalized.items() if value > 0],
        "diagnostics": diagnostics,
        "base_weights": normalize_weights(base_weights),
        "router_feedback_weights": normalize_weights(router_feedback_weights),
        "alpha": float(alpha),
        "utility_lam": float(lam),
        "router_checkpoint": router_checkpoint,
        "n_rollouts": int(n_rollouts),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def evolve_rubric_weights(
    episodes_path: str,
    base_rubric_weights: str,
    router_checkpoint: str,
    output_dir: str,
    output_name: str = "rubric_weights_v2.json",
    lam: float = 2e-5,
    alpha: float = 0.3,
    device: str = "cpu",
    max_steps: int = 30,
    limit: Optional[int] = None,
    corr_threshold: float = 0.0,
    std_threshold: float = 0.02,
    deterministic: bool = True,
    save_rollouts: bool = True,
) -> Dict:
    base_weights, base_payload = load_rubric_weight_file(base_rubric_weights)
    rubric_names = [name for name in base_weights if name in ALL_RUBRICS]
    if not rubric_names:
        raise ValueError(f"no base rubric keys match ALL_RUBRICS in {base_rubric_weights}")

    rollouts = rollout_policy_on_episodes(
        episodes_path=episodes_path,
        checkpoint_path=router_checkpoint,
        device=device,
        max_steps=max_steps,
        deterministic=deterministic,
        limit=limit,
        lam=lam,
        rubric_names=rubric_names,
    )
    router_weights, diagnostics = learn_router_feedback_weights(
        rollouts,
        rubric_names=rubric_names,
        corr_threshold=corr_threshold,
        std_threshold=std_threshold,
        fallback_weights=base_weights,
    )
    evolved = smooth_weights(base_weights, router_weights, alpha=alpha)
    payload = build_output_payload(
        weights=evolved,
        base_weights=base_weights,
        router_feedback_weights=router_weights,
        diagnostics=diagnostics,
        alpha=alpha,
        lam=lam,
        router_checkpoint=router_checkpoint,
        n_rollouts=len(rollouts),
        extra_metadata={
            "episodes_path": episodes_path,
            "base_rubric_weights": base_rubric_weights,
            "base_method": base_payload.get("method", "trim_rubric_static"),
            "corr_threshold": float(corr_threshold),
            "std_threshold": float(std_threshold),
            "deterministic": bool(deterministic),
            "max_steps": int(max_steps),
        },
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / output_name
    weights_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if save_rollouts:
        rollouts_path = out_dir / "router_feedback_rollouts.jsonl"
        with rollouts_path.open("w", encoding="utf-8") as f:
            for row in rollouts:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved TRIM-RubricV2 weights -> {weights_path}", flush=True)
    print(f"Active rubrics: {len(payload['active_rubrics'])}/{len(payload['weights'])}", flush=True)
    for name, value in sorted(payload["weights"].items(), key=lambda item: -item[1]):
        if value > 0:
            diag = diagnostics.get(name, {})
            print(
                f"  {name}: w={value:.4f}, router_w={router_weights.get(name, 0.0):.4f}, "
                f"corr={diag.get('corr', 0.0):.4f}, status={diag.get('status', 'unknown')}",
                flush=True,
            )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve TRIM-Rubric weights with router feedback")
    parser.add_argument("--episodes_path", required=True)
    parser.add_argument("--base_rubric_weights", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="rubric_weights_v2.json")
    parser.add_argument("--lam", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corr_threshold", type=float, default=0.0)
    parser.add_argument("--std_threshold", type=float, default=0.02)
    parser.add_argument("--sample_actions", action="store_true")
    parser.add_argument("--no_save_rollouts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evolve_rubric_weights(
        episodes_path=args.episodes_path,
        base_rubric_weights=args.base_rubric_weights,
        router_checkpoint=args.router_checkpoint,
        output_dir=args.output_dir,
        output_name=args.output_name,
        lam=args.lam,
        alpha=args.alpha,
        device=args.device,
        max_steps=args.max_steps,
        limit=args.limit,
        corr_threshold=args.corr_threshold,
        std_threshold=args.std_threshold,
        deterministic=not args.sample_actions,
        save_rollouts=not args.no_save_rollouts,
    )


if __name__ == "__main__":
    main()
