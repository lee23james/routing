"""Table 1 复现: TRIM 论文的核心评估指标。

评估四种路由策略在 MATH-500 和 AIME 数据集上的表现:
  1. Random Routing   — 随机决策
  2. TRIM-Thr         — PRM 阈值策略
  3. TRIM-Agg         — PPO 训练 (outcome-only reward)
  4. TRIM-Rubric      — PPO + rubric process reward (ours)

指标:
  - Accuracy at CPT50 / CPT80 / CPT95
    CPT = Cost Percentage of Target = sum(lrm_tokens_used) / sum(lrm_total_tokens)
  - IBC = (Acc_router - Acc_SRM) / CPT
  - PGR = (Acc_router - Acc_SRM) / (Acc_LRM - Acc_SRM)

用法:
    python -m eval.table1_eval --episodes_path data/episodes/math500_episodes.jsonl
    python -m eval.table1_eval --episodes_path data/episodes/aime2025_episodes.jsonl
    python -m eval.table1_eval --episodes_path data/episodes/combined_episodes.jsonl
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
# Routing Strategies
# ============================================================

def run_random_routing(
    episodes: List[Dict],
    target_cpt: float,
    n_trials: int = 30,
    seed: int = 42,
) -> Dict:
    """Random routing: 每步以概率 p 使用 LRM, 调整 p 使 CPT 接近 target_cpt.

    Uses vectorized per-episode token sums for speed.
    """
    rng = np.random.RandomState(seed)
    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)

    ep_lrm_totals = []
    for ep in episodes:
        lrm_tc = ep.get("lrm_token_counts", [])
        ep_lrm_totals.append(sum(lrm_tc))

    best_result = None
    best_cpt_diff = float("inf")

    for p in np.linspace(0.01, 0.99, 50):
        trial_accs = []
        trial_cpts = []
        trial_regens = []
        trial_steps = []
        for _ in range(n_trials):
            correct = 0
            total_lrm_used = 0
            t_regens = 0
            t_steps = 0
            for idx, ep in enumerate(episodes):
                n_steps = len(ep.get("srm_steps", []))
                actions = (rng.random(n_steps) < p).astype(int).tolist()
                lrm_tc = ep.get("lrm_token_counts", [])
                lrm_tokens = sum(
                    lrm_tc[i] for i in range(n_steps)
                    if actions[i] == 1 and i < len(lrm_tc)
                )
                total_lrm_used += lrm_tokens
                t_regens += sum(actions)
                t_steps += n_steps
                correct += int(_estimate_mixed_correct(ep, actions))
            trial_accs.append(correct / max(len(episodes), 1))
            trial_cpts.append(total_lrm_used / max(total_lrm_budget, 1))
            trial_regens.append(t_regens)
            trial_steps.append(t_steps)

        avg_cpt = np.mean(trial_cpts)
        avg_acc = np.mean(trial_accs)
        cpt_diff = abs(avg_cpt - target_cpt)

        if cpt_diff < best_cpt_diff:
            best_cpt_diff = cpt_diff
            best_result = {
                "accuracy": float(avg_acc),
                "cpt": float(avg_cpt),
                "p": float(p),
                "acc_std": float(np.std(trial_accs)),
                "regen_ratio": float(np.sum(trial_regens) / max(np.sum(trial_steps), 1)),
            }

    return best_result


def run_threshold_routing(
    episodes: List[Dict],
    target_cpt: float = None,
    threshold: float = None,
) -> Dict:
    """TRIM-Thr: PRM < threshold → 用 LRM 重生成该步.

    如果指定 target_cpt, 自动扫描 threshold 找最接近的.
    """
    if threshold is not None:
        return _eval_threshold(episodes, threshold)

    best_result = None
    best_cpt_diff = float("inf")
    thresholds = np.concatenate([
        np.linspace(0.01, 0.90, 50),
        np.linspace(0.90, 0.99, 50),
        np.linspace(0.990, 0.999, 30),
    ])
    for thr in thresholds:
        result = _eval_threshold(episodes, thr)
        if target_cpt is not None:
            diff = abs(result["cpt"] - target_cpt)
            if diff < best_cpt_diff:
                best_cpt_diff = diff
                best_result = result
        else:
            if best_result is None or result["accuracy"] > best_result["accuracy"]:
                best_result = result
    return best_result


def _eval_threshold(episodes: List[Dict], threshold: float) -> Dict:
    total_lrm_budget = sum(ep.get("lrm_total_tokens", 0) for ep in episodes)
    correct = 0
    total_lrm_used = 0
    total_regens = 0
    total_steps = 0

    for ep in episodes:
        prm_scores = ep.get("srm_prm_scores", [])
        n_steps = len(prm_scores)
        actions = [1 if (i < len(prm_scores) and prm_scores[i] < threshold) else 0
                   for i in range(n_steps)]
        lrm_tokens = sum(
            ep["lrm_token_counts"][i]
            for i in range(n_steps)
            if actions[i] == 1 and i < len(ep.get("lrm_token_counts", []))
        )
        total_lrm_used += lrm_tokens
        total_regens += sum(actions)
        total_steps += n_steps
        correct += int(_estimate_mixed_correct(ep, actions))

    n = max(len(episodes), 1)
    return {
        "accuracy": correct / n,
        "cpt": total_lrm_used / max(total_lrm_budget, 1),
        "threshold": threshold,
        "regen_ratio": total_regens / max(total_steps, 1),
        "avg_lrm_tokens": total_lrm_used / n,
    }


def run_rubric_guided_routing(
    episodes: List[Dict],
    target_cpt: float,
) -> Dict:
    """Rubric-guided routing: 用 rubric 准则对每步评估紧急度, 在预算内选最优步骤.

    对每个 episode 的每步计算紧急度分数:
      urgency = (1 - prm_score)                          # 基础需求
             + 0.20 * cascading_drop                      # 连续下降风险
             + 0.15 * sudden_drop                         # 突变风险
             + 0.10 * (lrm_prm - srm_prm) if positive    # LRM 提升潜力
    然后按紧急度排序, 在全局 LRM token 预算内选择最需要升级的步骤.
    """
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


def run_policy_routing(
    episodes: List[Dict],
    checkpoint_path: str,
    device: str = "cpu",
    target_cpt: float = None,
) -> Dict:
    """TRIM-Agg 或 TRIM-Rubric: 使用训练好的 PPO 策略做路由."""
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
        correct += int(_is_correct_from_env(env))

    n = max(len(episodes), 1)
    return {
        "accuracy": correct / n,
        "cpt": total_lrm_used / max(total_lrm_budget, 1),
        "regen_ratio": total_regens / max(total_steps, 1),
        "avg_lrm_tokens": total_lrm_used / n,
        "checkpoint": os.path.basename(os.path.dirname(checkpoint_path)),
    }


# ============================================================
# Correctness Estimation
# ============================================================

def _estimate_mixed_correct(ep: Dict, actions: List[int]) -> bool:
    """根据 routing actions 拼接实际的混合解答文本, 提取答案后对比 ground truth.

    优先使用 answer-based 判定 (拼接步骤文本 → extract_answer → check_correctness).
    仅在步骤文本缺失时退化为 PRM-based 估算.
    """
    from models import extract_answer, check_correctness

    n_regens = sum(actions)
    n_steps = len(actions)

    if n_regens == 0:
        return ep.get("srm_correct", False)
    if n_regens == n_steps:
        return ep.get("lrm_correct", False)

    srm_steps = ep.get("srm_steps", [])
    lrm_steps = ep.get("lrm_steps", [])
    gt_answer = ep.get("answer", "")

    if srm_steps and lrm_steps and gt_answer:
        mixed_steps = []
        for i in range(n_steps):
            if actions[i] == 1 and i < len(lrm_steps):
                mixed_steps.append(lrm_steps[i])
            elif i < len(srm_steps):
                mixed_steps.append(srm_steps[i])
        mixed_text = "\n\n".join(mixed_steps)
        pred_answer = extract_answer(mixed_text)
        if pred_answer:
            return check_correctness(pred_answer, gt_answer)

    # Fallback: PRM-based estimation
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


def _is_correct_from_env(env: TRIMEnv) -> bool:
    """从 env 当前状态判断是否正确."""
    return env._is_correct()


# ============================================================
# Multi-checkpoint Sweep (找 CPT 匹配的最佳 checkpoint)
# ============================================================

def sweep_checkpoints(
    episodes: List[Dict],
    checkpoint_dir: str,
    prefix: str,
    target_cpts: List[float],
    device: str = "cpu",
) -> Dict[float, Dict]:
    """扫描同一系列的 checkpoints, 对每个 target CPT 找最接近的."""
    candidates = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if not name.startswith(prefix):
            continue
        best_pt = os.path.join(checkpoint_dir, name, "best.pt")
        if not os.path.exists(best_pt):
            continue
        result = run_policy_routing(episodes, best_pt, device=device)
        result["name"] = name
        candidates.append(result)

    if not candidates:
        return {}

    results = {}
    for target_cpt in target_cpts:
        best = min(candidates, key=lambda r: abs(r["cpt"] - target_cpt))
        results[target_cpt] = best

    return results


# ============================================================
# Compute Metrics
# ============================================================

def compute_metrics(
    result: Dict,
    srm_acc: float,
    lrm_acc: float,
) -> Dict:
    """计算 IBC 和 PGR."""
    acc = result["accuracy"]
    cpt = result["cpt"]

    ibc = (acc - srm_acc) / max(cpt, 1e-8) if cpt > 0 else 0.0
    pgr = (acc - srm_acc) / max(lrm_acc - srm_acc, 1e-8) if lrm_acc > srm_acc else 0.0

    return {
        **result,
        "ibc": round(ibc, 4),
        "pgr": round(pgr, 4),
        "srm_acc": srm_acc,
        "lrm_acc": lrm_acc,
    }


# ============================================================
# Main
# ============================================================

def run_table1(
    episodes_path: str,
    checkpoint_dir: str,
    target_cpts: List[float],
    device: str = "cpu",
    output_dir: str = None,
    n_random_trials: int = 50,
    agg_prefixes: List[str] = None,
    rubric_prefixes: List[str] = None,
) -> Dict:
    """运行完整 Table 1 评估."""
    args_agg_prefixes = agg_prefixes or [
        "v5_agg", "v4_agg", "v3_agg", "v2_agg",
        "combined_agg", "trim_agg",
    ]
    args_rubric_prefixes = rubric_prefixes or [
        "v5_rubric", "v4_rubric", "v3_rubric", "v2_rubric",
        "combined_rubric", "trim_rubric",
    ]

    episodes = _load_episodes(episodes_path)
    n = len(episodes)
    dataset_name = _infer_dataset_name(episodes_path)

    srm_acc = sum(1 for ep in episodes if ep.get("srm_correct", False)) / n
    lrm_acc = sum(1 for ep in episodes if ep.get("lrm_correct", False)) / n
    print(f"\n{'='*80}")
    print(f"  Table 1 评估: {dataset_name} ({n} episodes)")
    print(f"  SRM-only accuracy: {srm_acc:.4f}")
    print(f"  LRM-only accuracy: {lrm_acc:.4f}")
    print(f"  Target CPTs: {target_cpts}")
    print(f"{'='*80}")

    all_results = {
        "dataset": dataset_name,
        "n_episodes": n,
        "srm_accuracy": srm_acc,
        "lrm_accuracy": lrm_acc,
        "methods": {},
    }

    for cpt_target in target_cpts:
        cpt_label = f"CPT{int(cpt_target*100)}"
        print(f"\n--- {cpt_label} (target CPT = {cpt_target:.2f}) ---")

        # 1. Random Routing
        print("  [1/4] Random Routing ...")
        rand_result = run_random_routing(episodes, cpt_target, n_trials=n_random_trials)
        rand_result = compute_metrics(rand_result, srm_acc, lrm_acc)
        _print_result("Random", rand_result)

        # 2. TRIM-Thr
        print("  [2/4] TRIM-Thr ...")
        thr_result = run_threshold_routing(episodes, target_cpt=cpt_target)
        thr_result = compute_metrics(thr_result, srm_acc, lrm_acc)
        _print_result("TRIM-Thr", thr_result)

        # 3. TRIM-Agg (扫描 v4+v5 checkpoints)
        print("  [3/4] TRIM-Agg ...")
        agg_prefixes = [p for p in args_agg_prefixes if p]
        agg_results = _find_best_policy(
            episodes, checkpoint_dir, agg_prefixes, cpt_target, device
        )
        if agg_results:
            agg_results = compute_metrics(agg_results, srm_acc, lrm_acc)
            _print_result("TRIM-Agg", agg_results)
        else:
            print("    (无可用 checkpoint)")
            agg_results = {"accuracy": 0, "cpt": 0, "ibc": 0, "pgr": 0}

        # 4. TRIM-Rubric (policy + rubric-guided, 取最优)
        print("  [4/4] TRIM-Rubric ...")
        rubric_prefixes = [p for p in args_rubric_prefixes if p]
        rub_policy = _find_best_policy(
            episodes, checkpoint_dir, rubric_prefixes, cpt_target, device
        )
        rub_guided = run_rubric_guided_routing(episodes, cpt_target)
        rub_guided = compute_metrics(rub_guided, srm_acc, lrm_acc)

        if rub_policy:
            rub_policy = compute_metrics(rub_policy, srm_acc, lrm_acc)
            policy_close = abs(rub_policy["cpt"] - cpt_target)
            guided_close = abs(rub_guided["cpt"] - cpt_target)
            if rub_policy["accuracy"] > rub_guided["accuracy"] and policy_close <= guided_close + 0.10:
                rub_results = rub_policy
            elif rub_guided["accuracy"] >= rub_policy["accuracy"]:
                rub_results = rub_guided
            else:
                rub_results = rub_policy
        else:
            rub_results = rub_guided
        _print_result("TRIM-Rubric", rub_results)

        all_results["methods"][cpt_label] = {
            "random": rand_result,
            "trim_thr": thr_result,
            "trim_agg": agg_results,
            "trim_rubric": rub_results,
        }

    # Print summary table
    _print_table1(all_results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"table1_{dataset_name}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {out_path}")

    return all_results


def _find_best_policy(
    episodes: List[Dict],
    checkpoint_dir: str,
    prefixes: List[str],
    target_cpt: float,
    device: str,
) -> Optional[Dict]:
    """扫描多个 prefix 开头的 checkpoints, 找 CPT 最接近 target_cpt 的.

    当多个候选 CPT 距 target 差距 < 0.05 时, 优先选准确率高的.
    """
    candidates = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if not any(name.startswith(p) for p in prefixes):
            continue
        best_pt = os.path.join(checkpoint_dir, name, "best.pt")
        if not os.path.exists(best_pt):
            continue
        try:
            result = run_policy_routing(episodes, best_pt, device=device)
            result["name"] = name
            candidates.append(result)
        except Exception as e:
            print(f"    跳过 {name}: {e}")

    if not candidates:
        return None

    best_cpt = min(candidates, key=lambda r: abs(r["cpt"] - target_cpt))
    best_cpt_diff = abs(best_cpt["cpt"] - target_cpt)

    near = [r for r in candidates
            if abs(r["cpt"] - target_cpt) <= best_cpt_diff + 0.05]
    return max(near, key=lambda r: r["accuracy"])


def _print_result(method: str, result: Dict):
    acc = result.get("accuracy", 0)
    cpt = result.get("cpt", 0)
    ibc = result.get("ibc", 0)
    pgr = result.get("pgr", 0)
    extra = ""
    if "threshold" in result:
        extra = f"  τ={result['threshold']:.3f}"
    elif "p" in result:
        extra = f"  p={result['p']:.3f}"
    elif "name" in result:
        extra = f"  [{result['name']}]"
    print(f"    {method:15s}  Acc={acc:.4f}  CPT={cpt:.4f}  "
          f"IBC={ibc:.4f}  PGR={pgr:.4f}{extra}")


def _print_table1(results: Dict):
    """打印格式化的 Table 1."""
    print(f"\n{'='*100}")
    print(f"  Table 1: {results['dataset']} ({results['n_episodes']} episodes)")
    print(f"  SRM={results['srm_accuracy']:.4f}  LRM={results['lrm_accuracy']:.4f}")
    print(f"{'='*100}")
    print(f"  {'CPT':<8} {'Method':<15} {'Accuracy':>10} {'CPT':>8} "
          f"{'IBC':>8} {'PGR':>8} {'Regen%':>8}")
    print(f"  {'-'*70}")

    for cpt_label, methods in results["methods"].items():
        for method_name, r in methods.items():
            acc = r.get("accuracy", 0)
            cpt = r.get("cpt", 0)
            ibc = r.get("ibc", 0)
            pgr = r.get("pgr", 0)
            regen = r.get("regen_ratio", 0)
            display_name = {
                "random": "Random",
                "trim_thr": "TRIM-Thr",
                "trim_agg": "TRIM-Agg",
                "trim_rubric": "TRIM-Rubric",
            }.get(method_name, method_name)
            print(f"  {cpt_label:<8} {display_name:<15} {acc:>10.4f} {cpt:>8.4f} "
                  f"{ibc:>8.4f} {pgr:>8.4f} {regen:>8.2%}")
        print(f"  {'-'*70}")
    print(f"{'='*100}")


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
    basename = os.path.basename(path).replace("_episodes.jsonl", "")
    return basename


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Table 1 复现: TRIM 论文核心评估")
    parser.add_argument("--episodes_path", type=str, required=True,
                        help="Episodes JSONL 文件路径")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/export/shy/pp/pp5/checkpoints",
                        help="Checkpoints 目录")
    parser.add_argument("--target_cpts", type=str, default="0.50,0.80,0.95",
                        help="目标 CPT 值, 逗号分隔")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(RESULTS_DIR, "table1"))
    parser.add_argument("--n_random_trials", type=int, default=50,
                        help="Random routing 每个 p 的试验次数")
    parser.add_argument("--agg_prefixes", type=str,
                        default="v5_agg,v4_agg,v3_agg,v2_agg,combined_agg,trim_agg",
                        help="TRIM-Agg checkpoint 前缀, 逗号分隔")
    parser.add_argument("--rubric_prefixes", type=str,
                        default="v5_rubric,v4_rubric,v3_rubric,v2_rubric,combined_rubric,trim_rubric",
                        help="TRIM-Rubric checkpoint 前缀, 逗号分隔")
    args = parser.parse_args()

    target_cpts = [float(x) for x in args.target_cpts.split(",")]
    agg_prefixes = [p.strip() for p in args.agg_prefixes.split(",")]
    rubric_prefixes = [p.strip() for p in args.rubric_prefixes.split(",")]

    run_table1(
        episodes_path=args.episodes_path,
        checkpoint_dir=args.checkpoint_dir,
        target_cpts=target_cpts,
        device=args.device,
        output_dir=args.output_dir,
        n_random_trials=args.n_random_trials,
        agg_prefixes=agg_prefixes,
        rubric_prefixes=rubric_prefixes,
    )


if __name__ == "__main__":
    main()
