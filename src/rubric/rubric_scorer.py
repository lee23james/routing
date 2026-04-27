"""Rubric-based process reward scoring for stepwise routing trajectories.

Two-tier rubric system inspired by RLCER + OpenRubric:
  Tier-1  "Seed rubrics"   — fixed rule-based criteria (always available)
  Tier-2  "Derived rubrics" — automatically mined from episode statistics

All rubrics are functions of trajectory features (PRM scores, actions, token
counts, episode metadata), so scoring is zero-cost during training.

Filtering follows RLCER: keep rubrics whose satisfaction positively correlates
with final answer correctness and that are discriminative across trajectories.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ============================================================
# Tier-1: Seed Rubrics (Fixed Rule-Based)
# ============================================================

def rubric_timely_escalation(prm_scores: List[float],
                             actions: List[int],
                             threshold: float = 0.5,
                             **kw) -> float:
    """R1: Escalate when PRM score drops below threshold.

    Score = fraction of low-PRM steps that were escalated.
    """
    low_steps = [i for i, s in enumerate(prm_scores) if s < threshold]
    if not low_steps:
        return 1.0
    escalated = sum(1 for i in low_steps if i < len(actions) and actions[i] == 1)
    return escalated / len(low_steps)


def rubric_cost_efficiency(prm_scores: List[float],
                           actions: List[int],
                           threshold: float = 0.7,
                           **kw) -> float:
    """R2: Avoid unnecessary escalations when SRM is doing well.

    Score = 1 - (fraction of high-PRM steps that were needlessly escalated).
    """
    high_steps = [i for i, s in enumerate(prm_scores) if s >= threshold]
    if not high_steps:
        return 1.0
    wasted = sum(1 for i in high_steps if i < len(actions) and actions[i] == 1)
    return 1.0 - wasted / len(high_steps)


def rubric_cascading_error_prevention(prm_scores: List[float],
                                      actions: List[int],
                                      window: int = 2,
                                      **kw) -> float:
    """R3: Prevent cascading errors by escalating before consecutive drops.

    Score = fraction of declining windows where escalation happened.
    """
    if len(prm_scores) < window + 1:
        return 1.0
    n_declining = 0
    n_intervened = 0
    for i in range(len(prm_scores) - window):
        segment = prm_scores[i:i + window + 1]
        if all(segment[j] > segment[j + 1] for j in range(len(segment) - 1)):
            n_declining += 1
            w_actions = actions[i:i + window + 1] if i + window + 1 <= len(actions) else actions[i:]
            if any(a == 1 for a in w_actions):
                n_intervened += 1
    if n_declining == 0:
        return 1.0
    return n_intervened / n_declining


def rubric_recovery_effectiveness(prm_scores: List[float],
                                  actions: List[int],
                                  lrm_prm_scores: Optional[List[float]] = None,
                                  **kw) -> float:
    """R4: When the router escalates, the chosen step's PRM should be better.

    Score = fraction of escalations that yielded PRM improvement.
    """
    improvements = []
    for i in range(len(actions)):
        if actions[i] == 1 and i < len(prm_scores):
            srm_score = prm_scores[i]
            if lrm_prm_scores and i < len(lrm_prm_scores):
                lrm_score = lrm_prm_scores[i]
                improvements.append(1.0 if lrm_score > srm_score else 0.0)
    if not improvements:
        return 0.5
    return float(np.mean(improvements))


def rubric_budget_allocation(prm_scores: List[float],
                             actions: List[int],
                             **kw) -> float:
    """R5: Allocate LRM budget to steps with the lowest PRM scores.

    Score = Spearman correlation between (1 - PRM_score) and action,
    mapped to [0, 1]. Higher = better budget targeting.
    """
    if not actions or sum(actions) == 0 or len(prm_scores) == 0:
        return 0.5
    n = min(len(prm_scores), len(actions))
    need = [1.0 - prm_scores[i] for i in range(n)]
    acts = [float(actions[i]) for i in range(n)]
    if np.std(need) < 1e-8 or np.std(acts) < 1e-8:
        return 0.5
    corr, _ = stats.spearmanr(need, acts)
    if math.isnan(corr):
        return 0.5
    return float(np.clip((corr + 1.0) / 2.0, 0, 1))


# ============================================================
# Tier-2: Derived Rubrics (Mined from Episode Statistics)
# ============================================================

def rubric_difficulty_awareness(prm_scores: List[float],
                                actions: List[int],
                                ep_meta: Optional[Dict] = None,
                                **kw) -> float:
    """R6: Harder problems (lower avg SRM PRM) should get more LRM budget.

    Score = 1 if escalation rate is inversely proportional to avg PRM.
    """
    if not prm_scores or not actions:
        return 0.5
    avg_prm = np.mean(prm_scores)
    regen_rate = sum(actions) / max(len(actions), 1)
    difficulty = 1.0 - avg_prm
    if difficulty < 0.1:
        return 1.0 if regen_rate < 0.2 else max(0.0, 1.0 - regen_rate)
    return float(np.clip(min(regen_rate / difficulty, 1.0), 0, 1))


def rubric_confidence_calibration(prm_scores: List[float],
                                  actions: List[int],
                                  **kw) -> float:
    """R7: Escalation probability should increase as PRM score decreases.

    Measures monotonicity of (low PRM → high escalation rate) across bins.
    """
    if len(prm_scores) < 4 or not actions:
        return 0.5
    n = min(len(prm_scores), len(actions))
    bins = {"low": [], "mid": [], "high": []}
    for i in range(n):
        s = prm_scores[i]
        a = actions[i]
        if s < 0.4:
            bins["low"].append(a)
        elif s < 0.7:
            bins["mid"].append(a)
        else:
            bins["high"].append(a)

    rates = {}
    for k, v in bins.items():
        rates[k] = np.mean(v) if v else None

    score = 0.5
    comparisons = 0
    correct = 0
    if rates["low"] is not None and rates["mid"] is not None:
        comparisons += 1
        if rates["low"] >= rates["mid"]:
            correct += 1
    if rates["mid"] is not None and rates["high"] is not None:
        comparisons += 1
        if rates["mid"] >= rates["high"]:
            correct += 1
    if rates["low"] is not None and rates["high"] is not None:
        comparisons += 1
        if rates["low"] >= rates["high"]:
            correct += 1
    if comparisons > 0:
        score = correct / comparisons
    return float(score)


def rubric_action_consistency(prm_scores: List[float],
                              actions: List[int],
                              **kw) -> float:
    """R8: Similar PRM scores should lead to similar routing decisions.

    Measures consistency: for steps with similar PRM, actions should agree.
    """
    if len(prm_scores) < 3 or not actions:
        return 0.5
    n = min(len(prm_scores), len(actions))
    pairs_agree = 0
    pairs_total = 0
    threshold = 0.1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(prm_scores[i] - prm_scores[j]) < threshold:
                pairs_total += 1
                if actions[i] == actions[j]:
                    pairs_agree += 1
    if pairs_total == 0:
        return 1.0
    return pairs_agree / pairs_total


def rubric_marginal_gain(prm_scores: List[float],
                         actions: List[int],
                         lrm_prm_scores: Optional[List[float]] = None,
                         **kw) -> float:
    """R9: Escalate only when LRM offers significant PRM improvement.

    Score = avg(LRM_PRM - SRM_PRM) at escalation points, normalised.
    Only meaningful when lrm_prm_scores are available.
    """
    if not lrm_prm_scores or not actions:
        return 0.5
    gains = []
    for i in range(min(len(actions), len(prm_scores))):
        if actions[i] == 1 and i < len(lrm_prm_scores):
            gains.append(lrm_prm_scores[i] - prm_scores[i])
    if not gains:
        return 0.5
    avg_gain = np.mean(gains)
    return float(np.clip(avg_gain + 0.5, 0, 1))


def rubric_no_oscillation(prm_scores: List[float],
                          actions: List[int],
                          **kw) -> float:
    """R10: Avoid rapid oscillation between SRM and LRM.

    Score = 1 - (fraction of action switches / total actions).
    Smooth routing is generally better than oscillating.
    """
    if len(actions) < 2:
        return 1.0
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    return 1.0 - switches / (len(actions) - 1)


def rubric_early_detection(prm_scores: List[float],
                           actions: List[int],
                           **kw) -> float:
    """R11: Detect the first significant PRM drop and intervene.

    Score = 1 if escalation happens at or before the first large drop.
    """
    if len(prm_scores) < 2 or not actions:
        return 0.5
    first_drop_idx = None
    for i in range(1, len(prm_scores)):
        if prm_scores[i] < prm_scores[i - 1] - 0.15:
            first_drop_idx = i
            break
    if first_drop_idx is None:
        return 1.0
    window = actions[max(0, first_drop_idx - 1):first_drop_idx + 1]
    if any(a == 1 for a in window):
        return 1.0
    return 0.0


def rubric_prm_trajectory_quality(prm_scores: List[float],
                                  actions: List[int],
                                  lrm_prm_scores: Optional[List[float]] = None,
                                  **kw) -> float:
    """R12: Overall quality of the chosen PRM trajectory.

    Score = mean of the chosen (SRM or LRM) PRM scores across steps.
    Higher chosen PRM trajectory = better routing.
    """
    if not prm_scores or not actions:
        return 0.5
    chosen = []
    n = min(len(prm_scores), len(actions))
    for i in range(n):
        if actions[i] == 1 and lrm_prm_scores and i < len(lrm_prm_scores):
            chosen.append(lrm_prm_scores[i])
        elif i < len(prm_scores):
            chosen.append(prm_scores[i])
    if not chosen:
        return 0.5
    return float(np.mean(chosen))


def rubric_critical_step_coverage(prm_scores: List[float],
                                  actions: List[int],
                                  lrm_prm_scores: Optional[List[float]] = None,
                                  **kw) -> float:
    """R13: Escalate at steps where LRM significantly outperforms SRM.

    Score = fraction of "critical" steps (LRM PRM >> SRM PRM) that were escalated.
    """
    if not lrm_prm_scores or not actions or not prm_scores:
        return 0.5
    n = min(len(prm_scores), len(actions), len(lrm_prm_scores))
    critical = []
    for i in range(n):
        gap = lrm_prm_scores[i] - prm_scores[i]
        if gap > 0.1:
            critical.append(i)
    if not critical:
        return 1.0
    escalated = sum(1 for i in critical if actions[i] == 1)
    return escalated / len(critical)


def rubric_prm_improvement_ratio(prm_scores: List[float],
                                 actions: List[int],
                                 lrm_prm_scores: Optional[List[float]] = None,
                                 **kw) -> float:
    """R14: How much routing improved the trajectory vs all-SRM baseline.

    Score = (chosen_avg_prm - srm_avg_prm) / (lrm_avg_prm - srm_avg_prm),
    clipped to [0, 1]. Higher = better capture of available improvement.
    """
    if not prm_scores or not lrm_prm_scores or not actions:
        return 0.5
    n = min(len(prm_scores), len(actions), len(lrm_prm_scores))
    chosen_prm = []
    for i in range(n):
        if actions[i] == 1:
            chosen_prm.append(lrm_prm_scores[i])
        else:
            chosen_prm.append(prm_scores[i])

    srm_avg = np.mean(prm_scores[:n])
    lrm_avg = np.mean(lrm_prm_scores[:n])
    chosen_avg = np.mean(chosen_prm)

    denom = lrm_avg - srm_avg
    if abs(denom) < 1e-6:
        return 0.5
    ratio = (chosen_avg - srm_avg) / denom
    return float(np.clip(ratio, 0, 1))


def rubric_worst_step_rescue(prm_scores: List[float],
                             actions: List[int],
                             lrm_prm_scores: Optional[List[float]] = None,
                             **kw) -> float:
    """R15: Rescue the worst steps by escalating them.

    Score = fraction of the bottom-K lowest SRM PRM steps that were escalated.
    """
    if not prm_scores or not actions:
        return 0.5
    n = min(len(prm_scores), len(actions))
    if n < 2:
        return 0.5
    k = max(1, n // 3)
    indexed = sorted(range(n), key=lambda i: prm_scores[i])
    worst_k = indexed[:k]
    escalated = sum(1 for i in worst_k if actions[i] == 1)
    return escalated / len(worst_k)


# ============================================================
# Registry
# ============================================================

SEED_RUBRICS = {
    "timely_escalation": rubric_timely_escalation,
    "cost_efficiency": rubric_cost_efficiency,
    "cascading_error_prevention": rubric_cascading_error_prevention,
    "recovery_effectiveness": rubric_recovery_effectiveness,
    "budget_allocation": rubric_budget_allocation,
}

DERIVED_RUBRICS = {
    "difficulty_awareness": rubric_difficulty_awareness,
    "confidence_calibration": rubric_confidence_calibration,
    "action_consistency": rubric_action_consistency,
    "marginal_gain": rubric_marginal_gain,
    "no_oscillation": rubric_no_oscillation,
    "early_detection": rubric_early_detection,
    "prm_trajectory_quality": rubric_prm_trajectory_quality,
    "critical_step_coverage": rubric_critical_step_coverage,
    "prm_improvement_ratio": rubric_prm_improvement_ratio,
    "worst_step_rescue": rubric_worst_step_rescue,
}

ALL_RUBRICS = {**SEED_RUBRICS, **DERIVED_RUBRICS}


# ============================================================
# Aggregate Rubric Scoring
# ============================================================

def score_trajectory_rubrics(
    prm_scores: List[float],
    actions: List[int],
    lrm_prm_scores: Optional[List[float]] = None,
    weights: Optional[Dict[str, float]] = None,
    ep_meta: Optional[Dict] = None,
    rubric_set: Optional[Dict] = None,
) -> Dict[str, float]:
    """Score a routing trajectory against rubric criteria.

    Returns dict with per-rubric scores and the weighted aggregate.
    """
    rubrics = rubric_set or ALL_RUBRICS
    w = weights or {k: 1.0 for k in rubrics}
    scores = {}

    for name, fn in rubrics.items():
        scores[name] = fn(
            prm_scores=prm_scores,
            actions=actions,
            lrm_prm_scores=lrm_prm_scores,
            ep_meta=ep_meta,
        )

    active_w = {k: w.get(k, 0.0) for k in scores if w.get(k, 0.0) > 0}
    total_w = sum(active_w.values())
    if total_w > 0:
        scores["aggregate"] = sum(
            scores[k] * active_w[k] for k in active_w
        ) / total_w
    else:
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        scores["aggregate"] = float(np.mean(vals)) if vals else 0.5

    return scores


# ============================================================
# Rubric Weight Learning (RLCER-inspired correlation filtering)
# ============================================================

def learn_rubric_weights(
    episodes: List[Dict],
    rubric_set: Optional[Dict] = None,
    n_trajectories: int = 20,
    corr_threshold: float = 0.05,
    std_threshold: float = 0.02,
    sampling_strategies: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Learn rubric weights from episode data using correlation with correctness.

    Improvements over v1:
    - Uses diverse sampling strategies (random, PRM-guided, threshold-based)
    - Lower filtering thresholds to keep more rubrics active
    - Returns per-rubric diagnostics for analysis
    """
    rubrics = rubric_set or ALL_RUBRICS
    if sampling_strategies is None:
        sampling_strategies = ["random", "prm_guided", "threshold_sweep"]

    rubric_scores_all = {name: [] for name in rubrics}
    correctness_all = []

    for ep in episodes:
        srm_prm = ep.get("srm_prm_scores", [])
        lrm_prm = ep.get("lrm_prm_scores", [])
        num_steps = len(srm_prm)
        if num_steps == 0:
            continue

        trajectories = _sample_trajectories(
            srm_prm, lrm_prm, ep, num_steps, n_trajectories, sampling_strategies
        )

        for actions, correct in trajectories:
            scores = score_trajectory_rubrics(
                srm_prm, actions, lrm_prm,
                rubric_set=rubrics,
            )
            for name in rubrics:
                rubric_scores_all[name].append(scores.get(name, 0.5))
            correctness_all.append(float(correct))

    correctness_arr = np.array(correctness_all)
    if len(correctness_arr) == 0 or np.std(correctness_arr) < 1e-8:
        equal_w = 1.0 / len(rubrics)
        weights = {k: equal_w for k in rubrics}
        diagnostics = {k: {"corr": 0, "std": 0, "status": "fallback"} for k in rubrics}
        return weights, diagnostics

    weights = {}
    diagnostics = {}
    for name in rubrics:
        arr = np.array(rubric_scores_all[name])
        arr_std = float(np.std(arr))
        if arr_std < std_threshold:
            weights[name] = 0.0
            diagnostics[name] = {"corr": 0.0, "std": arr_std, "status": "low_std"}
            continue
        corr, pval = stats.pearsonr(arr, correctness_arr)
        if math.isnan(corr):
            weights[name] = 0.0
            diagnostics[name] = {"corr": 0.0, "std": arr_std, "status": "nan_corr"}
            continue

        diagnostics[name] = {
            "corr": float(corr), "pval": float(pval),
            "std": arr_std, "mean": float(np.mean(arr)),
        }

        if corr > corr_threshold:
            weights[name] = corr
            diagnostics[name]["status"] = "active"
        elif abs(corr) < corr_threshold:
            weights[name] = 0.0
            diagnostics[name]["status"] = "low_corr"
        else:
            weights[name] = 0.0
            diagnostics[name]["status"] = "negative_corr"

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        equal_w = 1.0 / len(rubrics)
        weights = {k: equal_w for k in rubrics}
        for k in diagnostics:
            diagnostics[k]["status"] = "fallback_equal"

    return weights, diagnostics


def _sample_trajectories(
    srm_prm: List[float],
    lrm_prm: List[float],
    ep: Dict,
    num_steps: int,
    n_total: int,
    strategies: List[str],
) -> List[Tuple[List[int], bool]]:
    """Sample diverse routing trajectories for rubric weight learning.

    Three strategies to ensure rubric scores have variance:
    1. random: uniform random actions
    2. prm_guided: probabilistic escalation based on PRM score
    3. threshold_sweep: deterministic threshold-based policies
    """
    trajectories = []
    n_per_strategy = max(1, n_total // len(strategies))

    for strat in strategies:
        for trial in range(n_per_strategy):
            if strat == "random":
                p = np.random.uniform(0.1, 0.5)
                actions = [int(np.random.random() < p) for _ in range(num_steps)]

            elif strat == "prm_guided":
                actions = []
                for i in range(num_steps):
                    s = srm_prm[i] if i < len(srm_prm) else 0.5
                    p_escalate = max(0.0, 1.0 - s) * np.random.uniform(0.5, 1.5)
                    actions.append(int(np.random.random() < min(p_escalate, 0.9)))

            elif strat == "threshold_sweep":
                thresh = 0.2 + trial * (0.6 / max(n_per_strategy - 1, 1))
                actions = [
                    int(srm_prm[i] < thresh) if i < len(srm_prm) else 0
                    for i in range(num_steps)
                ]
            else:
                actions = [0] * num_steps

            n_regens = sum(actions)
            if n_regens == 0:
                correct = ep.get("srm_correct", False)
            elif n_regens == num_steps:
                correct = ep.get("lrm_correct", False)
            else:
                min_srm_chosen = min(
                    (srm_prm[i] for i in range(num_steps) if actions[i] == 0 and i < len(srm_prm)),
                    default=1.0,
                )
                avg_lrm_chosen = np.mean([
                    lrm_prm[i] for i in range(num_steps) if actions[i] == 1 and i < len(lrm_prm)
                ]) if any(actions[i] == 1 and i < len(lrm_prm) for i in range(num_steps)) else 0.5

                srm_corr = ep.get("srm_correct", False)
                lrm_corr = ep.get("lrm_correct", False)
                if lrm_corr and not srm_corr:
                    correct = (n_regens / num_steps) > 0.3 or avg_lrm_chosen > 0.6
                elif srm_corr and not lrm_corr:
                    correct = (n_regens / num_steps) < 0.5
                elif srm_corr and lrm_corr:
                    correct = True
                else:
                    correct = avg_lrm_chosen > 0.7 and n_regens > num_steps * 0.4

            trajectories.append((actions, correct))

    all_srm = ([0] * num_steps, ep.get("srm_correct", False))
    all_lrm = ([1] * num_steps, ep.get("lrm_correct", False))
    trajectories.extend([all_srm, all_lrm])

    return trajectories
