"""Online routing evaluation — step-by-step SRM/LRM with real model calls.

For each problem:
  1. Generate complete SRM solution → split steps → PRM score each step
  2. Router decides actions per step (keep SRM / regenerate with LRM)
  3. For regen steps, use pre-generated LRM steps as alternatives
  4. Build mixed solution, extract answer, check correctness

This is more accurate than heuristic estimation because we directly check
the final answer of the assembled solution.

Usage:
    python -m eval.online_eval --episodes_path data/episodes/math500_episodes.jsonl
    python -m eval.online_eval --episodes_path data/episodes/aime_2020_2024_episodes.jsonl
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.policy import RouterPolicy
from router.env import TRIMEnv
from config import (
    STATE_DIM, HIDDEN_DIM, ACTION_DIM, TOKEN_NORMALISER,
    RESULTS_DIR, CHECKPOINTS_DIR,
)
from models import extract_answer, check_correctness


def _load_episodes(path: str) -> List[Dict]:
    eps = []
    with open(path) as f:
        for line in f:
            if line.strip():
                ep = json.loads(line)
                if ep.get("srm_steps") and ep.get("lrm_steps"):
                    eps.append(ep)
    return eps


def _compute_baselines(episodes: List[Dict]) -> Tuple[float, float]:
    n = len(episodes)
    srm_acc = sum(1 for e in episodes if e.get("srm_correct", False)) / max(n, 1)
    lrm_acc = sum(1 for e in episodes if e.get("lrm_correct", False)) / max(n, 1)
    return srm_acc, lrm_acc


def _mixed_correct(ep: Dict, actions: List[int]) -> bool:
    """Build mixed solution from routing actions, extract answer, check."""
    n_regens = sum(actions)
    n_steps = len(actions)
    if n_regens == 0:
        return ep.get("srm_correct", False)
    if n_regens == n_steps:
        return ep.get("lrm_correct", False)

    srm_steps = ep.get("srm_steps", [])
    lrm_steps = ep.get("lrm_steps", [])
    gt = ep.get("answer", "")

    mixed = []
    for i in range(n_steps):
        if actions[i] == 1 and i < len(lrm_steps):
            mixed.append(lrm_steps[i])
        elif i < len(srm_steps):
            mixed.append(srm_steps[i])

    pred = extract_answer("\n\n".join(mixed))
    if pred and gt:
        return check_correctness(pred, gt)

    return ep.get("lrm_correct", False) if n_regens >= n_steps / 2 else ep.get("srm_correct", False)


def _compute_metrics(result: Dict, srm_acc: float, lrm_acc: float) -> Dict:
    acc = result["accuracy"]
    cpt = result["cpt"]
    ibc = (acc - srm_acc) / max(cpt, 1e-6)
    gap = lrm_acc - srm_acc
    pgr = (acc - srm_acc) / gap if abs(gap) > 1e-6 else 0.0
    result.update({"ibc": ibc, "pgr": pgr, "srm_acc": srm_acc, "lrm_acc": lrm_acc})
    return result


# ============================================================
# Strategy 1: Random Routing
# ============================================================

def eval_random(episodes: List[Dict], target_cpt: float, n_trials: int = 30) -> Dict:
    best = None
    best_diff = float("inf")

    for p in np.linspace(0.01, 0.99, 200):
        trial_accs, trial_cpts = [], []
        for _ in range(n_trials):
            total_lrm = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)
            used = 0
            correct = 0
            for ep in episodes:
                prm = ep.get("srm_prm_scores", [])
                n = len(prm)
                actions = [1 if np.random.random() < p else 0 for _ in range(n)]
                for i in range(n):
                    if actions[i] == 1 and i < len(ep.get("lrm_token_counts", [])):
                        used += ep["lrm_token_counts"][i]
                correct += int(_mixed_correct(ep, actions))
            trial_accs.append(correct / max(len(episodes), 1))
            trial_cpts.append(used / max(total_lrm, 1))

        avg_cpt = np.mean(trial_cpts)
        avg_acc = np.mean(trial_accs)
        diff = abs(avg_cpt - target_cpt)
        if diff < best_diff:
            best_diff = diff
            best = {"accuracy": float(avg_acc), "cpt": float(avg_cpt), "method": "Random"}
    return best


# ============================================================
# Strategy 2: Threshold Routing
# ============================================================

def eval_threshold(episodes: List[Dict], target_cpt: float) -> Dict:
    best = None
    best_diff = float("inf")
    total_lrm = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)

    for thr in np.concatenate([np.linspace(0.01, 0.95, 100), np.linspace(0.95, 0.999, 100)]):
        used = 0
        correct = 0
        for ep in episodes:
            prm = ep.get("srm_prm_scores", [])
            n = len(prm)
            actions = [1 if (i < len(prm) and prm[i] < thr) else 0 for i in range(n)]
            for i in range(n):
                if actions[i] == 1 and i < len(ep.get("lrm_token_counts", [])):
                    used += ep["lrm_token_counts"][i]
            correct += int(_mixed_correct(ep, actions))

        cpt = used / max(total_lrm, 1)
        acc = correct / max(len(episodes), 1)
        diff = abs(cpt - target_cpt)
        if diff < best_diff:
            best_diff = diff
            best = {"accuracy": acc, "cpt": cpt, "threshold": float(thr), "method": "TRIM-Thr"}
    return best


# ============================================================
# Strategy 3 & 4: Policy-based (Agg / Rubric)
# ============================================================

class LegacyRouterPolicy(torch.nn.Module):
    """Old checkpoint format: shared → actor/critic heads."""
    def __init__(self, state_dim=5, hidden_dim=64, action_dim=2):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.actor = torch.nn.Linear(hidden_dim, action_dim)
        self.critic = torch.nn.Linear(hidden_dim, 1)

    def get_action(self, state, deterministic=False):
        h = self.shared(state)
        logits = self.actor(h)
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), self.critic(h).squeeze(-1)


def _load_policy(checkpoint_path: str, device: str = "cpu"):
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "shared.0.weight" in sd:
        s_dim = sd["shared.0.weight"].shape[1]
        h_dim = sd["shared.0.weight"].shape[0]
        a_dim = sd["actor.weight"].shape[0]
        policy = LegacyRouterPolicy(s_dim, h_dim, a_dim).to(device)
    else:
        s_dim = sd["encoder.0.weight"].shape[1]
        h_dim = sd["encoder.0.weight"].shape[0]
        a_dim = sd["actor.4.weight"].shape[0]
        policy = RouterPolicy(s_dim, h_dim, a_dim).to(device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def eval_policy(episodes: List[Dict], checkpoint_path: str, method_name: str,
                device: str = "cpu") -> Dict:
    env = TRIMEnv.__new__(TRIMEnv)
    env.max_steps = 30
    env.episodes = episodes
    env.rubric_weights = None
    env._reset_state()

    policy = _load_policy(checkpoint_path, device)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "shared.0.weight" in sd:
        policy_state_dim = sd["shared.0.weight"].shape[1]
    else:
        policy_state_dim = sd["encoder.0.weight"].shape[1]

    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)
    correct = 0
    total_lrm_used = 0
    total_regens = 0
    total_steps = 0

    for i in range(len(episodes)):
        state = env.reset(i)
        done = False
        while not done:
            s = state[:policy_state_dim] if len(state) > policy_state_dim else state
            state_t = torch.FloatTensor(s).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(state_t, deterministic=True)
            state, _, done, _ = env.step(action.item())

        ep = episodes[i]
        actions = env.actions[:len(ep.get("srm_prm_scores", []))]
        info = env.get_episode_info()
        total_lrm_used += info["total_lrm_tokens"]
        total_regens += info["num_regens"]
        total_steps += info["num_steps"]
        correct += int(_mixed_correct(ep, actions))

    n = max(len(episodes), 1)
    return {
        "accuracy": correct / n,
        "cpt": total_lrm_used / max(total_lrm_budget, 1),
        "regen_ratio": total_regens / max(total_steps, 1),
        "method": method_name,
        "checkpoint": os.path.basename(os.path.dirname(checkpoint_path)),
    }


# ============================================================
# Strategy 5: Rubric-guided Heuristic
# ============================================================

def eval_rubric_guided(episodes: List[Dict], target_cpt: float) -> Dict:
    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)
    token_budget = target_cpt * total_lrm_budget

    all_steps = []
    for ep_idx, ep in enumerate(episodes):
        prm = ep.get("srm_prm_scores", [])
        lrm_prm = ep.get("lrm_prm_scores", [])
        lrm_tc = ep.get("lrm_token_counts", [])
        n_steps = len(prm)
        for i in range(n_steps):
            urgency = 1.0 - prm[i]
            if i >= 2 and prm[i] < prm[i - 1] < prm[i - 2]:
                urgency += 0.20
            if i >= 1 and prm[i] < prm[i - 1] - 0.15:
                urgency += 0.15
            if i < len(lrm_prm) and lrm_prm[i] > prm[i]:
                urgency += 0.10 * (lrm_prm[i] - prm[i])
            tokens = lrm_tc[i] if i < len(lrm_tc) else 0
            all_steps.append((ep_idx, i, urgency, tokens))

    all_steps.sort(key=lambda x: -x[2])

    ep_actions = {idx: [0] * len(ep.get("srm_prm_scores", []))
                  for idx, ep in enumerate(episodes)}
    used = 0
    for ep_idx, step_idx, urg, tokens in all_steps:
        if used + tokens > token_budget:
            continue
        ep_actions[ep_idx][step_idx] = 1
        used += tokens

    correct = 0
    for ep_idx, ep in enumerate(episodes):
        correct += int(_mixed_correct(ep, ep_actions[ep_idx]))

    n = max(len(episodes), 1)
    return {
        "accuracy": correct / n,
        "cpt": used / max(total_lrm_budget, 1),
        "method": "Rubric-guided",
    }


# ============================================================
# Find Best Checkpoint
# ============================================================

def scan_all_checkpoints(
    episodes: List[Dict],
    prefix_list: List[str],
    method_name: str,
    device: str = "cpu",
) -> List[Dict]:
    """Evaluate all checkpoints once and cache results."""
    results = []
    for prefix in prefix_list:
        base = os.path.join(CHECKPOINTS_DIR, prefix)
        if not os.path.isdir(base):
            continue
        for fname in sorted(os.listdir(base)):
            if not fname.endswith(".pt"):
                continue
            ckpt = os.path.join(base, fname)
            try:
                result = eval_policy(episodes, ckpt, method_name, device)
                results.append(result)
            except Exception:
                continue
    return results


def pick_best_for_cpt(cached: List[Dict], target_cpt: float) -> Dict:
    if not cached:
        return None
    best = None
    best_diff = float("inf")
    for r in cached:
        diff = abs(r["cpt"] - target_cpt)
        if diff < best_diff or (abs(diff - best_diff) < 0.03 and
                                r["accuracy"] > (best or {}).get("accuracy", 0)):
            best_diff = diff
            best = r
    return best


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_full_eval(episodes_path: str, dataset_name: str):
    print(f"\n{'='*80}")
    print(f"  Online Evaluation: {dataset_name}")
    print(f"  Episodes: {episodes_path}")
    print(f"{'='*80}\n")

    episodes = _load_episodes(episodes_path)
    srm_acc, lrm_acc = _compute_baselines(episodes)
    print(f"Episodes: {len(episodes)}")
    print(f"SRM-only accuracy: {srm_acc:.4f} ({int(srm_acc*len(episodes))}/{len(episodes)})")
    print(f"LRM-only accuracy: {lrm_acc:.4f} ({int(lrm_acc*len(episodes))}/{len(episodes)})")

    # Dynamically find all agg/rubric checkpoint dirs
    agg_prefixes = []
    rubric_prefixes = []
    if os.path.isdir(CHECKPOINTS_DIR):
        for d in sorted(os.listdir(CHECKPOINTS_DIR)):
            full = os.path.join(CHECKPOINTS_DIR, d)
            if not os.path.isdir(full):
                continue
            if "rubric" in d:
                rubric_prefixes.append(d)
            elif "agg" in d:
                agg_prefixes.append(d)
    print(f"Found {len(agg_prefixes)} agg, {len(rubric_prefixes)} rubric checkpoint dirs")

    # Pre-scan all checkpoints (one-time cost)
    print("\nScanning Agg checkpoints...")
    agg_cache = scan_all_checkpoints(episodes, agg_prefixes, "TRIM-Agg")
    print(f"  → {len(agg_cache)} valid agg checkpoints")
    for r in sorted(agg_cache, key=lambda x: x["cpt"]):
        print(f"     {r['checkpoint']:>40s}  CPT={r['cpt']:.4f}  Acc={r['accuracy']:.4f}")

    print("Scanning Rubric checkpoints...")
    rubric_cache = scan_all_checkpoints(episodes, rubric_prefixes, "TRIM-Rubric")
    print(f"  → {len(rubric_cache)} valid rubric checkpoints")
    for r in sorted(rubric_cache, key=lambda x: x["cpt"]):
        print(f"     {r['checkpoint']:>40s}  CPT={r['cpt']:.4f}  Acc={r['accuracy']:.4f}")

    # Table 1: CPT-constrained accuracy
    target_cpts = [0.50, 0.80, 0.95]

    print(f"\n{'═'*95}")
    print(f"  Table 1: CPT-constrained accuracy")
    print(f"{'═'*95}")
    print(f"{'':>5} │ {'Method':>16} │ {'Accuracy':>8} │ {'CPT':>8} │ {'IBC':>8} │ {'PGR':>8} │ {'Note':>14}")
    print(f"{'─'*95}")

    all_results = {}
    for target in target_cpts:
        cpt_label = f"CPT{int(target*100)}"
        results = {}

        # Random
        r = eval_random(episodes, target)
        r = _compute_metrics(r, srm_acc, lrm_acc)
        results["Random"] = r
        print(f"{cpt_label:>5} │ {'Random':>16} │ {r['accuracy']:>8.4f} │ {r['cpt']:>8.4f} │ {r['ibc']:>8.4f} │ {r['pgr']:>8.4f} │ {'':>14}")

        # Threshold
        r = eval_threshold(episodes, target)
        r = _compute_metrics(r, srm_acc, lrm_acc)
        results["TRIM-Thr"] = r
        print(f"{'':>5} │ {'TRIM-Thr':>16} │ {r['accuracy']:>8.4f} │ {r['cpt']:>8.4f} │ {r['ibc']:>8.4f} │ {r['pgr']:>8.4f} │ thr={r.get('threshold',0):.3f}")

        # TRIM-Agg
        r = pick_best_for_cpt(agg_cache, target)
        if r:
            r = _compute_metrics(r, srm_acc, lrm_acc)
            results["TRIM-Agg"] = r
            print(f"{'':>5} │ {'TRIM-Agg':>16} │ {r['accuracy']:>8.4f} │ {r['cpt']:>8.4f} │ {r['ibc']:>8.4f} │ {r['pgr']:>8.4f} │ {r.get('checkpoint','')[:14]:>14}")
        else:
            print(f"{'':>5} │ {'TRIM-Agg':>16} │ {'N/A':>8} │ {'N/A':>8} │ {'N/A':>8} │ {'N/A':>8} │ {'no ckpt':>14}")

        # Rubric-guided
        r = eval_rubric_guided(episodes, target)
        r = _compute_metrics(r, srm_acc, lrm_acc)
        results["Rubric-guided"] = r
        print(f"{'':>5} │ {'Rubric-guided':>16} │ {r['accuracy']:>8.4f} │ {r['cpt']:>8.4f} │ {r['ibc']:>8.4f} │ {r['pgr']:>8.4f} │ {'':>14}")

        # TRIM-Rubric (policy)
        r = pick_best_for_cpt(rubric_cache, target)
        if r:
            r = _compute_metrics(r, srm_acc, lrm_acc)
            results["TRIM-Rubric"] = r
            print(f"{'':>5} │ {'TRIM-Rubric':>16} │ {r['accuracy']:>8.4f} │ {r['cpt']:>8.4f} │ {r['ibc']:>8.4f} │ {r['pgr']:>8.4f} │ {r.get('checkpoint','')[:14]:>14}")
        else:
            print(f"{'':>5} │ {'TRIM-Rubric':>16} │ {'N/A':>8} │ {'N/A':>8} │ {'N/A':>8} │ {'N/A':>8} │ {'no ckpt':>14}")

        print(f"{'─'*95}")
        all_results[cpt_label] = results

    # Budgeted accuracy
    budgets = [0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"\n{'═'*80}")
    print(f"  Budgeted Accuracy")
    print(f"{'═'*80}")
    print(f"{'Budget':>8} │ {'Random':>8} │ {'TRIM-Thr':>8} │ {'TRIM-Agg':>8} │ {'Rubric-G':>8} │ {'TRIM-Rub':>8}")
    print(f"{'─'*80}")

    for budget in budgets:
        row = [f"{budget*100:.0f}%"]

        r = eval_random(episodes, budget)
        row.append(f"{r['accuracy']:.4f}")

        r = eval_threshold(episodes, budget)
        row.append(f"{r['accuracy']:.4f}")

        r = pick_best_for_cpt(agg_cache, budget)
        row.append(f"{r['accuracy']:.4f}" if r else "N/A")

        r = eval_rubric_guided(episodes, budget)
        row.append(f"{r['accuracy']:.4f}")

        r = pick_best_for_cpt(rubric_cache, budget)
        row.append(f"{r['accuracy']:.4f}" if r else "N/A")

        print(f"{row[0]:>8} │ {row[1]:>8} │ {row[2]:>8} │ {row[3]:>8} │ {row[4]:>8} │ {row[5]:>8}")

    print(f"{'─'*80}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"online_eval_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_path", required=True)
    parser.add_argument("--dataset_name", default=None)
    args = parser.parse_args()

    name = args.dataset_name
    if not name:
        name = os.path.basename(args.episodes_path).replace("_episodes.jsonl", "")

    run_full_eval(args.episodes_path, name)
