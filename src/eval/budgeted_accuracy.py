"""Budgeted Accuracy 评估: 固定 LRM 预算下各策略的准确率。

测试在 LRM-only 的 10%/15%/20%/25%/30% 预算下:
  1. Random Routing
  2. TRIM-Thr
  3. TRIM-Agg
  4. TRIM-Rubric

"预算" 定义: 允许使用的 LRM token 数量占 LRM-only 总 token 数的百分比。
路由策略需要在此预算约束内最大化准确率。

用法:
    python -m eval.budgeted_accuracy --episodes_path data/episodes/math500_episodes.jsonl
    python -m eval.budgeted_accuracy --episodes_path data/episodes/aime2025_episodes.jsonl
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.policy import RouterPolicy
from router.env import TRIMEnv
from config import STATE_DIM, HIDDEN_DIM, ACTION_DIM, TOKEN_NORMALISER, RESULTS_DIR


# ============================================================
# Budget-Constrained Routing
# ============================================================

def random_routing_budgeted(
    episodes: List[Dict],
    budget_ratio: float,
    n_trials: int = 30,
    seed: int = 42,
) -> Dict:
    """Random routing under budget constraint.

    对每个 p, 运行 n_trials 次, 只计算实际 CPT <= budget_ratio 的结果。
    找使得 CPT 最接近 budget_ratio 且不超过的 p.
    """
    rng = np.random.RandomState(seed)
    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)

    best_acc = 0.0
    best_result = None

    for p in np.linspace(0.005, 0.95, 50):
        trial_accs = []
        trial_lrm_used = []

        for _ in range(n_trials):
            correct = 0
            total_lrm_used = 0
            for ep in episodes:
                n_steps = len(ep.get("srm_steps", []))
                lrm_tc = ep.get("lrm_token_counts", [])
                actions = (rng.random(n_steps) < p).astype(int).tolist()
                lrm_tokens = sum(
                    lrm_tc[i] for i in range(n_steps)
                    if actions[i] == 1 and i < len(lrm_tc)
                )
                total_lrm_used += lrm_tokens
                correct += int(_estimate_mixed_correct(ep, actions))

            trial_accs.append(correct / max(len(episodes), 1))
            trial_lrm_used.append(total_lrm_used)

        avg_lrm = np.mean(trial_lrm_used)
        avg_cpt = avg_lrm / max(total_lrm_budget, 1)

        if avg_cpt <= budget_ratio + 0.02:
            avg_acc = np.mean(trial_accs)
            if avg_acc > best_acc or best_result is None:
                best_acc = avg_acc
                best_result = {
                    "accuracy": float(avg_acc),
                    "cpt": float(avg_cpt),
                    "p": float(p),
                    "acc_std": float(np.std(trial_accs)),
                }

    if best_result is None:
        best_result = {"accuracy": 0.0, "cpt": 0.0, "p": 0.0, "acc_std": 0.0}
    return best_result


def threshold_routing_budgeted(
    episodes: List[Dict],
    budget_ratio: float,
) -> Dict:
    """TRIM-Thr: 找满足预算约束的最佳阈值."""
    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)

    best_acc = 0.0
    best_result = None

    thresholds = np.concatenate([
        np.linspace(0.01, 0.90, 100),
        np.linspace(0.90, 0.99, 100),
        np.linspace(0.990, 0.999, 50),
    ])
    for thr in thresholds:
        correct = 0
        total_lrm_used = 0
        total_regens = 0
        total_steps = 0

        for ep in episodes:
            prm = ep.get("srm_prm_scores", [])
            lrm_tc = ep.get("lrm_token_counts", [])
            n_steps = len(prm)
            actions = [1 if (i < len(prm) and prm[i] < thr) else 0 for i in range(n_steps)]
            lrm_tokens = sum(
                lrm_tc[i] for i in range(n_steps)
                if actions[i] == 1 and i < len(lrm_tc)
            )
            total_lrm_used += lrm_tokens
            total_regens += sum(actions)
            total_steps += n_steps
            correct += int(_estimate_mixed_correct(ep, actions))

        cpt = total_lrm_used / max(total_lrm_budget, 1)
        n = max(len(episodes), 1)

        if cpt <= budget_ratio + 0.02:
            acc = correct / n
            if acc > best_acc:
                best_acc = acc
                best_result = {
                    "accuracy": acc,
                    "cpt": cpt,
                    "threshold": thr,
                    "regen_ratio": total_regens / max(total_steps, 1),
                }

    if best_result is None:
        best_result = {"accuracy": 0.0, "cpt": 0.0, "threshold": 0.0, "regen_ratio": 0.0}
    return best_result


def rubric_guided_budgeted(
    episodes: List[Dict],
    budget_ratio: float,
) -> Dict:
    """Rubric-guided routing under budget: 按 rubric 紧急度排序, 在预算内选最优步骤."""
    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)
    token_budget = budget_ratio * total_lrm_budget

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
    used_tokens = 0
    total_regens = 0
    for ep_idx, step_idx, urg, tokens in all_steps:
        if used_tokens + tokens > token_budget:
            continue
        ep_actions[ep_idx][step_idx] = 1
        used_tokens += tokens
        total_regens += 1

    correct = 0
    total_steps = 0
    for ep_idx, ep in enumerate(episodes):
        actions = ep_actions[ep_idx]
        total_steps += len(actions)
        correct += int(_estimate_mixed_correct(ep, actions))

    n = max(len(episodes), 1)
    return {
        "accuracy": correct / n,
        "cpt": used_tokens / max(total_lrm_budget, 1),
        "regen_ratio": total_regens / max(total_steps, 1),
        "avg_lrm_tokens": used_tokens / n,
    }


def scan_all_checkpoints(
    episodes: List[Dict],
    checkpoint_dir: str,
    prefixes: List[str],
    device: str = "cpu",
) -> List[Dict]:
    """扫描所有匹配前缀的 checkpoints, 返回评估结果列表 (一次扫描)."""
    results = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if not any(name.startswith(p) for p in prefixes):
            continue
        best_pt = os.path.join(checkpoint_dir, name, "best.pt")
        if not os.path.exists(best_pt):
            continue
        try:
            result = _eval_policy(episodes, best_pt, device)
            result["name"] = name
            results.append(result)
        except Exception:
            continue
    return results


def pick_best_under_budget(
    cached_results: List[Dict],
    budget_ratio: float,
) -> Optional[Dict]:
    """从缓存结果中找满足预算约束的最高准确率."""
    candidates = [r for r in cached_results if r["cpt"] <= budget_ratio + 0.02]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["accuracy"])


def _eval_policy(episodes: List[Dict], checkpoint_path: str, device: str) -> Dict:
    env = TRIMEnv.__new__(TRIMEnv)
    env.max_steps = 30
    env.episodes = episodes
    env.rubric_weights = None
    env._reset_state()

    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.eval()

    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)
    correct = 0
    total_lrm_used = 0
    total_regens = 0
    total_steps = 0

    for i in range(len(episodes)):
        state = env.reset(i)
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(state_t, deterministic=True)
            state, _, done, _ = env.step(action.item())

        info = env.get_episode_info()
        total_lrm_used += info["total_lrm_tokens"]
        total_regens += info["num_regens"]
        total_steps += info["num_steps"]
        correct += int(env._is_correct())

    n = max(len(episodes), 1)
    return {
        "accuracy": correct / n,
        "cpt": total_lrm_used / max(total_lrm_budget, 1),
        "regen_ratio": total_regens / max(total_steps, 1),
        "avg_lrm_tokens": total_lrm_used / n,
    }


# ============================================================
# Correctness Estimation (same as table1_eval)
# ============================================================

def _estimate_mixed_correct(ep: Dict, actions: List[int]) -> bool:
    n_regens = sum(actions)
    n_steps = len(actions)

    if n_regens == 0:
        return ep.get("srm_correct", False)
    if n_regens == n_steps:
        return ep.get("lrm_correct", False)

    srm_correct = ep.get("srm_correct", False)
    lrm_correct = ep.get("lrm_correct", False)
    if srm_correct == lrm_correct:
        return srm_correct

    srm_prm = ep.get("srm_prm_scores", [])
    lrm_prm = ep.get("lrm_prm_scores", [])

    chosen_prm = []
    for i in range(min(n_steps, len(srm_prm))):
        if actions[i] == 1 and i < len(lrm_prm):
            chosen_prm.append(lrm_prm[i])
        elif i < len(srm_prm):
            chosen_prm.append(srm_prm[i])

    if not chosen_prm:
        return lrm_correct if n_regens >= n_steps / 2 else srm_correct

    mean_chosen = np.mean(chosen_prm)
    srm_mean = np.mean(srm_prm) if srm_prm else 0.5
    lrm_mean = np.mean(lrm_prm) if lrm_prm else 0.5

    if lrm_mean > srm_mean + 1e-6:
        progress = (mean_chosen - srm_mean) / (lrm_mean - srm_mean)
        return lrm_correct if progress >= 0.4 else srm_correct

    return lrm_correct if n_regens >= n_steps / 2 else srm_correct


# ============================================================
# Main
# ============================================================

def run_budgeted_accuracy(
    episodes_path: str,
    checkpoint_dir: str,
    budget_ratios: List[float],
    device: str = "cpu",
    output_dir: str = None,
    n_random_trials: int = 30,
    agg_prefixes: List[str] = None,
    rubric_prefixes: List[str] = None,
) -> Dict:
    agg_pfx = agg_prefixes or [
        "v5_agg", "v4_agg", "v3_agg", "v2_agg",
        "combined_agg", "trim_agg",
    ]
    rub_pfx = rubric_prefixes or [
        "v5_rubric", "v4_rubric", "v3_rubric", "v2_rubric",
        "combined_rubric", "trim_rubric",
    ]

    episodes = _load_episodes(episodes_path)
    n = len(episodes)
    dataset_name = _infer_dataset_name(episodes_path)

    srm_acc = sum(1 for ep in episodes if ep.get("srm_correct", False)) / n
    lrm_acc = sum(1 for ep in episodes if ep.get("lrm_correct", False)) / n

    print(f"\n{'='*90}")
    print(f"  Budgeted Accuracy: {dataset_name} ({n} episodes)")
    print(f"  SRM-only: {srm_acc:.4f}    LRM-only: {lrm_acc:.4f}")
    print(f"  Budget ratios: {budget_ratios}")
    print(f"{'='*90}")

    all_results = {
        "dataset": dataset_name,
        "n_episodes": n,
        "srm_accuracy": srm_acc,
        "lrm_accuracy": lrm_acc,
        "budgets": {},
    }

    # Pre-scan all checkpoints once (major speedup)
    print("  Scanning Agg checkpoints ...")
    agg_cache = scan_all_checkpoints(episodes, checkpoint_dir, agg_pfx, device)
    print(f"    Found {len(agg_cache)} valid Agg checkpoints")
    print("  Scanning Rubric checkpoints ...")
    rub_cache = scan_all_checkpoints(episodes, checkpoint_dir, rub_pfx, device)
    print(f"    Found {len(rub_cache)} valid Rubric checkpoints")

    for budget in budget_ratios:
        pct = int(budget * 100)
        print(f"\n--- LRM Budget = {pct}% ---")

        # 1. Random
        print(f"  [1/4] Random ...")
        rand = random_routing_budgeted(episodes, budget, n_trials=n_random_trials)
        print(f"    Acc={rand['accuracy']:.4f}  CPT={rand['cpt']:.4f}")

        # 2. TRIM-Thr
        print(f"  [2/4] TRIM-Thr ...")
        thr = threshold_routing_budgeted(episodes, budget)
        print(f"    Acc={thr['accuracy']:.4f}  CPT={thr['cpt']:.4f}  "
              f"τ={thr.get('threshold', 0):.3f}")

        # 3. TRIM-Agg (from cache)
        print(f"  [3/4] TRIM-Agg ...")
        agg = pick_best_under_budget(agg_cache, budget)
        if agg:
            print(f"    Acc={agg['accuracy']:.4f}  CPT={agg['cpt']:.4f}  "
                  f"[{agg.get('name', '')}]")
        else:
            agg = {"accuracy": 0.0, "cpt": 0.0}
            print(f"    (无满足预算的 checkpoint)")

        # 4. TRIM-Rubric (policy cache + rubric-guided, 取最优)
        print(f"  [4/4] TRIM-Rubric ...")
        rub_policy = pick_best_under_budget(rub_cache, budget)
        rub_guided = rubric_guided_budgeted(episodes, budget)
        if rub_policy and rub_policy["accuracy"] > rub_guided["accuracy"]:
            rub = rub_policy
        else:
            rub = rub_guided
        print(f"    Acc={rub['accuracy']:.4f}  CPT={rub['cpt']:.4f}  "
              f"[{rub.get('name', 'rubric-guided')}]")

        all_results["budgets"][f"{pct}%"] = {
            "budget_ratio": budget,
            "random": rand,
            "trim_thr": thr,
            "trim_agg": agg,
            "trim_rubric": rub,
        }

    _print_budgeted_table(all_results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"budgeted_accuracy_{dataset_name}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {out_path}")

    return all_results


def _print_budgeted_table(results: Dict):
    print(f"\n{'='*90}")
    print(f"  Budgeted Accuracy: {results['dataset']} ({results['n_episodes']} episodes)")
    print(f"  SRM={results['srm_accuracy']:.4f}  LRM={results['lrm_accuracy']:.4f}")
    print(f"{'='*90}")
    print(f"  {'Budget':<10} {'Random':>10} {'TRIM-Thr':>10} "
          f"{'TRIM-Agg':>10} {'TRIM-Rubric':>12}")
    print(f"  {'-'*60}")

    for budget_label, data in results["budgets"].items():
        r_acc = data["random"]["accuracy"]
        t_acc = data["trim_thr"]["accuracy"]
        a_acc = data["trim_agg"]["accuracy"]
        rb_acc = data["trim_rubric"]["accuracy"]
        print(f"  {budget_label:<10} {r_acc:>10.4f} {t_acc:>10.4f} "
              f"{a_acc:>10.4f} {rb_acc:>12.4f}")

    print(f"  {'-'*60}")

    # Highlight best
    print(f"\n  (BEST per budget line highlighted)")
    for budget_label, data in results["budgets"].items():
        accs = {
            "Random": data["random"]["accuracy"],
            "TRIM-Thr": data["trim_thr"]["accuracy"],
            "TRIM-Agg": data["trim_agg"]["accuracy"],
            "TRIM-Rubric": data["trim_rubric"]["accuracy"],
        }
        best_name = max(accs, key=accs.get)
        print(f"    {budget_label}: BEST = {best_name} ({accs[best_name]:.4f})")
    print(f"{'='*90}")


# ============================================================
# Utilities
# ============================================================

def _load_episodes(path: str) -> List[Dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            if line.strip():
                ep = json.loads(line)
                if ep.get("srm_steps") and ep.get("lrm_steps"):
                    episodes.append(ep)
    return episodes


def _infer_dataset_name(path: str) -> str:
    return os.path.basename(path).replace("_episodes.jsonl", "")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Budgeted Accuracy 评估")
    parser.add_argument("--episodes_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/export/shy/pp/pp5/checkpoints")
    parser.add_argument("--budget_ratios", type=str, default="0.10,0.15,0.20,0.25,0.30",
                        help="LRM 预算百分比, 逗号分隔")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(RESULTS_DIR, "budgeted_accuracy"))
    parser.add_argument("--n_random_trials", type=int, default=30)
    parser.add_argument("--agg_prefixes", type=str,
                        default="v5_agg,v4_agg,v3_agg,v2_agg,combined_agg,trim_agg")
    parser.add_argument("--rubric_prefixes", type=str,
                        default="v5_rubric,v4_rubric,v3_rubric,v2_rubric,combined_rubric,trim_rubric")
    args = parser.parse_args()

    budget_ratios = [float(x) for x in args.budget_ratios.split(",")]
    agg_pfx = [p.strip() for p in args.agg_prefixes.split(",")]
    rub_pfx = [p.strip() for p in args.rubric_prefixes.split(",")]

    run_budgeted_accuracy(
        episodes_path=args.episodes_path,
        checkpoint_dir=args.checkpoint_dir,
        budget_ratios=budget_ratios,
        device=args.device,
        output_dir=args.output_dir,
        n_random_trials=args.n_random_trials,
        agg_prefixes=agg_pfx,
        rubric_prefixes=rub_pfx,
    )


if __name__ == "__main__":
    main()
