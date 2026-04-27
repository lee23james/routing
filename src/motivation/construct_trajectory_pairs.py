"""受控轨迹对构建: 为 motivation 分析生成配对数据.

筛选条件 (受控实验):
  1. 只看 SRM-wrong + LRM-right 的题 (routing 决策真正关键的子集)
  2. 对每题用不同策略生成多条 routing trajectories
  3. 只保留 最终对错相同 且 LRM token 成本差 < 20% 的轨迹对
  → outcome-level reward 几乎无法区分, 差异只来自过程本身

用法:
    python -m motivation.construct_trajectory_pairs \
        --episodes_path data/episodes/math500_episodes.jsonl \
        --output_path results/motivation/trajectory_pairs.jsonl
"""

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_critical_step(ep: Dict) -> int:
    """找到 SRM 首次明显出错的关键步骤 (PRM 首次大幅下降)."""
    prm = ep.get("srm_prm_scores", [])
    if len(prm) < 2:
        return 0
    for i in range(1, len(prm)):
        if prm[i] < prm[i - 1] - 0.10:
            return i
    # fallback: PRM 最低点
    return int(np.argmin(prm))


def generate_trajectories(ep: Dict, n_per_strategy: int = 5,
                          seed: int = 42) -> List[Dict]:
    """对一个 episode 用多种策略生成 routing trajectories.

    策略:
      - threshold_sweep: 不同阈值的 PRM 阈值路由
      - random_budget: 固定预算随机选步
      - critical_first: 优先关键步骤
      - uniform: 均匀选步
      - anti_critical: 避开关键步骤 (对照)
    """
    rng = np.random.RandomState(seed)
    prm = ep.get("srm_prm_scores", [])
    lrm_prm = ep.get("lrm_prm_scores", [])
    lrm_tc = ep.get("lrm_token_counts", [])
    n = len(prm)
    if n == 0:
        return []

    critical = find_critical_step(ep)
    trajectories = []

    # threshold sweep
    for thr in np.linspace(0.5, 0.999, n_per_strategy):
        actions = [1 if prm[i] < thr else 0 for i in range(n)]
        trajectories.append(_make_traj(ep, actions, f"threshold_{thr:.3f}"))

    # random budget: 随机选 k 步用 LRM
    for trial in range(n_per_strategy):
        k = rng.randint(1, max(2, n // 2))
        chosen = set(rng.choice(n, size=min(k, n), replace=False))
        actions = [1 if i in chosen else 0 for i in range(n)]
        trajectories.append(_make_traj(ep, actions, f"random_{trial}"))

    # critical_first: 以关键步骤为中心辐射选步
    for radius in range(1, n_per_strategy + 1):
        actions = [0] * n
        for i in range(max(0, critical - radius), min(n, critical + radius + 1)):
            actions[i] = 1
        trajectories.append(_make_traj(ep, actions, f"critical_r{radius}"))

    # anti_critical: 避开关键步骤, 在远处选步 (对照)
    for trial in range(n_per_strategy):
        actions = [0] * n
        candidates = [i for i in range(n) if abs(i - critical) > 2]
        if candidates:
            k = rng.randint(1, max(2, len(candidates) // 2))
            chosen = set(rng.choice(candidates, size=min(k, len(candidates)),
                                    replace=False))
            for i in chosen:
                actions[i] = 1
        trajectories.append(_make_traj(ep, actions, f"anti_critical_{trial}"))

    return trajectories


def _make_traj(ep: Dict, actions: List[int], strategy: str) -> Dict:
    """从 episode 和 actions 构建一条轨迹记录."""
    prm = ep.get("srm_prm_scores", [])
    lrm_prm = ep.get("lrm_prm_scores", [])
    lrm_tc = ep.get("lrm_token_counts", [])
    n = len(actions)

    lrm_tokens = sum(lrm_tc[i] for i in range(n)
                     if actions[i] == 1 and i < len(lrm_tc))
    n_regens = sum(actions)

    correct = _estimate_correct(ep, actions)

    chosen_prm = []
    for i in range(min(n, len(prm))):
        if actions[i] == 1 and i < len(lrm_prm):
            chosen_prm.append(lrm_prm[i])
        elif i < len(prm):
            chosen_prm.append(prm[i])

    return {
        "strategy": strategy,
        "actions": actions,
        "n_regens": n_regens,
        "lrm_tokens": lrm_tokens,
        "correct": correct,
        "chosen_prm_mean": float(np.mean(chosen_prm)) if chosen_prm else 0.5,
    }


def _estimate_correct(ep: Dict, actions: List[int]) -> bool:
    n_regens = sum(actions)
    n_steps = len(actions)
    if n_regens == 0:
        return ep.get("srm_correct", False)
    if n_regens == n_steps:
        return ep.get("lrm_correct", False)

    srm_c = ep.get("srm_correct", False)
    lrm_c = ep.get("lrm_correct", False)
    if srm_c == lrm_c:
        return srm_c

    srm_prm = ep.get("srm_prm_scores", [])
    lrm_prm = ep.get("lrm_prm_scores", [])
    chosen = []
    for i in range(min(n_steps, len(srm_prm))):
        if actions[i] == 1 and i < len(lrm_prm):
            chosen.append(lrm_prm[i])
        else:
            chosen.append(srm_prm[i] if i < len(srm_prm) else 0.5)
    if not chosen:
        return lrm_c if n_regens >= n_steps / 2 else srm_c

    mean_c = np.mean(chosen)
    srm_mean = np.mean(srm_prm) if srm_prm else 0.5
    lrm_mean = np.mean(lrm_prm) if lrm_prm else 0.5
    if abs(lrm_mean - srm_mean) < 1e-6:
        return lrm_c if n_regens >= n_steps / 2 else srm_c
    progress = (mean_c - srm_mean) / (lrm_mean - srm_mean)
    return lrm_c if progress >= 0.4 else srm_c


def build_controlled_pairs(
    ep: Dict,
    trajectories: List[Dict],
    cost_tolerance: float = 0.20,
) -> List[Dict]:
    """从同一题的轨迹中构建受控配对.

    只保留: 最终对错相同 且 LRM token 差 < cost_tolerance (相对) 的对.
    """
    pairs = []
    for i, j in combinations(range(len(trajectories)), 2):
        a, b = trajectories[i], trajectories[j]
        if a["correct"] != b["correct"]:
            continue
        if a["actions"] == b["actions"]:
            continue
        max_tok = max(a["lrm_tokens"], b["lrm_tokens"], 1)
        if abs(a["lrm_tokens"] - b["lrm_tokens"]) / max_tok > cost_tolerance:
            continue
        pairs.append({
            "traj_a": a,
            "traj_b": b,
            "same_outcome": True,
            "cost_diff_pct": abs(a["lrm_tokens"] - b["lrm_tokens"]) / max_tok,
        })
    return pairs


def run(episodes_path: str, output_path: str, cost_tolerance: float = 0.20,
        seed: int = 42) -> Dict:
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    critical_eps = [e for e in episodes
                    if not e.get("srm_correct", True) and e.get("lrm_correct", False)]

    print(f"Total episodes: {len(episodes)}")
    print(f"Critical (SRM-wrong + LRM-right): {len(critical_eps)}")

    all_pairs = []
    for ep in critical_eps:
        trajs = generate_trajectories(ep, seed=seed)
        pairs = build_controlled_pairs(ep, trajs, cost_tolerance)
        for p in pairs:
            p["episode_id"] = ep.get("id", "")
            p["query"] = ep.get("query", "")[:200]
            p["critical_step"] = find_critical_step(ep)
            p["srm_prm_scores"] = ep.get("srm_prm_scores", [])
            p["lrm_prm_scores"] = ep.get("lrm_prm_scores", [])
        all_pairs.extend(pairs)

    print(f"Controlled pairs generated: {len(all_pairs)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved → {output_path}")

    return {"n_critical": len(critical_eps), "n_pairs": len(all_pairs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_path", required=True)
    parser.add_argument("--output_path", default="results/motivation/trajectory_pairs.jsonl")
    parser.add_argument("--cost_tolerance", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.episodes_path, args.output_path, args.cost_tolerance, args.seed)


if __name__ == "__main__":
    main()
