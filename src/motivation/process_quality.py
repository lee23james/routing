"""三维过程质量评分: 用于量化 routing trajectory 的过程差异.

三个维度 (对应用户直觉):
  D1. 关键干预命中 (Critical Intervention Hit)
      — 是否在 SRM 首次明显出错附近及时切到 LRM
  D2. 切换平稳性 (Switching Smoothness)
      — 切换次数少, 无频繁 SRM↔LRM 震荡
  D3. 路径简洁性 (Path Conciseness)
      — 无冗余 LRM 调用, 总 regen 率与需求匹配

每维度输出 [0, 1] 分数, 越高越好.
综合分 = 加权和 (默认等权).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def score_critical_hit(actions: List[int], critical_step: int,
                       srm_prm: List[float], window: int = 2) -> float:
    """D1: 关键干预命中.

    检查在 critical_step ± window 范围内是否有 action=1 (使用 LRM).
    同时考虑是否命中了所有 PRM 低谷步骤.
    """
    n = len(actions)
    if n == 0:
        return 0.5

    lo = max(0, critical_step - window)
    hi = min(n, critical_step + window + 1)
    hit = any(actions[i] == 1 for i in range(lo, hi))

    low_steps = [i for i, s in enumerate(srm_prm) if s < np.percentile(srm_prm, 25)]
    if low_steps:
        low_hit = sum(1 for i in low_steps if i < n and actions[i] == 1)
        low_ratio = low_hit / len(low_steps)
    else:
        low_ratio = 1.0

    return 0.6 * float(hit) + 0.4 * low_ratio


def score_switching_smoothness(actions: List[int]) -> float:
    """D2: 切换平稳性.

    计算 action 序列中 0→1 和 1→0 的切换次数, 归一化后取反.
    完全平稳 (全 0 或全 1 或一次连续切换) → 1.0
    频繁震荡 → 接近 0.0
    """
    if len(actions) < 2:
        return 1.0
    switches = sum(1 for i in range(1, len(actions))
                   if actions[i] != actions[i - 1])
    max_switches = len(actions) - 1
    return 1.0 - switches / max_switches


def score_path_conciseness(actions: List[int], srm_prm: List[float],
                           lrm_prm: Optional[List[float]] = None) -> float:
    """D3: 路径简洁性.

    惩罚: 在 SRM PRM 已经很高的步骤仍然调用 LRM (浪费).
    奖励: LRM 调用集中在真正需要的步骤.
    """
    n = len(actions)
    if n == 0 or sum(actions) == 0:
        return 1.0

    wasted = 0
    useful = 0
    for i in range(n):
        if actions[i] == 1:
            if i < len(srm_prm) and srm_prm[i] >= 0.95:
                wasted += 1
            else:
                useful += 1
                if lrm_prm and i < len(lrm_prm) and lrm_prm[i] > srm_prm[i]:
                    useful += 0.5

    total_regens = sum(actions)
    waste_ratio = wasted / total_regens if total_regens > 0 else 0
    return float(np.clip(1.0 - waste_ratio, 0, 1))


def score_trajectory(
    actions: List[int],
    srm_prm: List[float],
    critical_step: int,
    lrm_prm: Optional[List[float]] = None,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> Dict[str, float]:
    """计算三维过程质量分和综合分."""
    d1 = score_critical_hit(actions, critical_step, srm_prm)
    d2 = score_switching_smoothness(actions)
    d3 = score_path_conciseness(actions, srm_prm, lrm_prm)

    w1, w2, w3 = weights
    aggregate = w1 * d1 + w2 * d2 + w3 * d3

    return {
        "critical_hit": round(d1, 4),
        "smoothness": round(d2, 4),
        "conciseness": round(d3, 4),
        "process_quality": round(aggregate, 4),
    }


def compare_pair(pair: Dict, weights=(0.4, 0.3, 0.3)) -> Dict:
    """对一个轨迹对计算过程质量分并确定偏好.

    返回 'prefer': 'A' | 'B' | 'tie' 以及详细分数.
    """
    srm_prm = pair.get("srm_prm_scores", [])
    lrm_prm = pair.get("lrm_prm_scores", [])
    critical = pair.get("critical_step", 0)

    sa = score_trajectory(pair["traj_a"]["actions"], srm_prm, critical, lrm_prm, weights)
    sb = score_trajectory(pair["traj_b"]["actions"], srm_prm, critical, lrm_prm, weights)

    diff = sa["process_quality"] - sb["process_quality"]
    if diff > 0.02:
        prefer = "A"
    elif diff < -0.02:
        prefer = "B"
    else:
        prefer = "tie"

    return {
        "scores_a": sa,
        "scores_b": sb,
        "diff": round(diff, 4),
        "prefer": prefer,
    }
