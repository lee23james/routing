"""Stepwise routing environment for TRIM-Agg (+ rubric process reward).

Simulates the online routing process using pre-generated episode data.
At each step t, the router sees a state and chooses continue/regenerate.

State features (aligned with TRIM source, dim=5):
    0. min PRM score up to t-1
    1. product of PRM scores up to t-1
    2. PRM score at step t (draft / SRM)
    3. step token length / TOKEN_NORMALISER
    4. step_num / max_steps

Reward structure (per-step, aligned with TRIM source):
    - action=0 (accept SRM): reward = 0
    - action=1 (use LRM):    reward = -cost_per_token * lrm_tokens
    - final step:             reward += task_reward (1.0 if correct)
"""

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import TOKEN_NORMALISER, TASK_REWARD


class TRIMEnv:
    """TRIM stepwise routing environment with optional rubric reward."""

    def __init__(self, episodes_path: str, max_steps: int = 30,
                 rubric_weights: Optional[Dict[str, float]] = None):
        self.max_steps = max_steps
        self.episodes = self._load_episodes(episodes_path)
        self.rubric_weights = rubric_weights
        self._reset_state()

    def _load_episodes(self, path: str) -> List[Dict]:
        episodes = []
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            return episodes
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if ep.get("srm_steps") and ep.get("lrm_steps"):
                    episodes.append(ep)
        print(f"Loaded {len(episodes)} episodes from {path}")
        return episodes

    def _reset_state(self):
        self.current_ep = None
        self.step_idx = 0
        self.actions: List[int] = []
        self.prm_scores: List[float] = []
        self.lrm_token_costs: List[int] = []
        self.chosen_prm: List[float] = []

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def reset(self, episode_idx: Optional[int] = None) -> np.ndarray:
        self._reset_state()
        if episode_idx is not None:
            self.current_ep = self.episodes[episode_idx % len(self.episodes)]
        else:
            self.current_ep = random.choice(self.episodes)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Compute TRIM-Agg state (aligned with TRIM build_observations).

        Features: (min_prev, prod_prev, r_t, c_t, t_norm)
        """
        ep = self.current_ep
        t = self.step_idx
        num_steps = min(len(ep["srm_steps"]), self.max_steps)

        if t >= num_steps:
            return np.zeros(5, dtype=np.float32)

        # Previous steps' PRM stats (from chosen path)
        if self.chosen_prm:
            min_prev = min(self.chosen_prm)
            prod_prev = math.prod(self.chosen_prm)
        else:
            min_prev = 1.0
            prod_prev = 1.0

        r_t = ep["srm_prm_scores"][t] if t < len(ep["srm_prm_scores"]) else 0.5
        c_t = (ep["srm_token_counts"][t] / TOKEN_NORMALISER
               if t < len(ep["srm_token_counts"]) else 0.0)
        t_norm = t / self.max_steps

        return np.array([min_prev, prod_prev, r_t, c_t, t_norm], dtype=np.float32)

    def step(self, action: int, cost_per_token: float = 0.0
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one routing decision.

        action: 0 = continue (accept SRM), 1 = regenerate (use LRM)

        Per-step reward (aligned with TRIM):
            - action=0: 0.0
            - action=1: -cost_per_token * lrm_tokens_at_step
        """
        ep = self.current_ep
        t = self.step_idx
        num_steps = min(len(ep["srm_steps"]), self.max_steps)
        self.actions.append(action)

        if action == 1:
            prm = ep["lrm_prm_scores"][t] if t < len(ep["lrm_prm_scores"]) else 0.5
            lrm_tokens = ep["lrm_token_counts"][t] if t < len(ep["lrm_token_counts"]) else 0
        else:
            prm = ep["srm_prm_scores"][t] if t < len(ep["srm_prm_scores"]) else 0.5
            lrm_tokens = 0

        self.prm_scores.append(
            ep["srm_prm_scores"][t] if t < len(ep["srm_prm_scores"]) else 0.5
        )
        self.chosen_prm.append(prm)
        self.lrm_token_costs.append(lrm_tokens)
        self.step_idx += 1

        # Per-step cost reward (TRIM-aligned)
        if action == 1:
            step_reward = -cost_per_token * lrm_tokens
        else:
            step_reward = 0.0

        done = self.step_idx >= num_steps
        next_state = self._get_state() if not done else np.zeros(5, dtype=np.float32)
        info = {"step": t, "action": action, "prm_score": prm, "lrm_tokens": lrm_tokens}
        return next_state, step_reward, done, info

    def compute_outcome_reward(self, lam_rubric: float = 0.0) -> float:
        """Compute the task outcome reward (added to the final step).

        Returns task_reward (1.0 if correct) + optional rubric reward.
        """
        correct = self._is_correct()
        outcome = TASK_REWARD if correct else 0.0

        rubric_reward = 0.0
        if lam_rubric > 0 and self.rubric_weights:
            rubric_reward = lam_rubric * self._compute_rubric_reward()

        return outcome + rubric_reward

    def _is_correct(self) -> bool:
        ep = self.current_ep
        if sum(self.actions) == 0:
            return ep.get("srm_correct", False)
        return self._estimate_mixed_correct(ep, self.actions, self.chosen_prm)

    def compute_episode_reward(self, lam: float,
                               lam_rubric: float = 0.0) -> float:
        """Backward-compatible: total episode reward."""
        outcome = self.compute_outcome_reward(lam_rubric)
        cost = lam * sum(self.lrm_token_costs)
        return outcome - cost

    def _compute_rubric_reward(self) -> float:
        """Compute normalised rubric process reward for the current trajectory."""
        from rubric.rubric_scorer import score_trajectory_rubrics, ALL_RUBRICS

        ep = self.current_ep
        lrm_prm = ep.get("lrm_prm_scores", [])
        active_rubrics = {k: v for k, v in ALL_RUBRICS.items()
                          if self.rubric_weights.get(k, 0) > 0}
        if not active_rubrics:
            active_rubrics = ALL_RUBRICS
        scores = score_trajectory_rubrics(
            self.prm_scores, self.actions, lrm_prm,
            weights=self.rubric_weights,
            rubric_set=active_rubrics,
        )
        return scores["aggregate"]

    @staticmethod
    def _estimate_mixed_correct(ep: Dict, actions: list, chosen_prm: list) -> bool:
        """拼接混合步骤文本, 提取答案对比 ground truth."""
        from models import extract_answer, check_correctness

        n_regens = sum(actions)
        n_steps = len(actions)
        if n_regens == 0:
            return ep.get("srm_correct", False)
        if n_regens == n_steps:
            return ep.get("lrm_correct", False)

        srm_steps = ep.get("srm_steps", [])
        lrm_steps = ep.get("lrm_steps", [])
        gt = ep.get("answer", "")
        if srm_steps and lrm_steps and gt:
            mixed = []
            for i in range(n_steps):
                if actions[i] == 1 and i < len(lrm_steps):
                    mixed.append(lrm_steps[i])
                elif i < len(srm_steps):
                    mixed.append(srm_steps[i])
            pred = extract_answer("\n\n".join(mixed))
            if pred:
                return check_correctness(pred, gt)

        srm_correct = ep.get("srm_correct", False)
        lrm_correct = ep.get("lrm_correct", False)
        if srm_correct == lrm_correct:
            return srm_correct
        return lrm_correct if n_regens >= n_steps / 2 else srm_correct

    def get_episode_info(self) -> Dict:
        return {
            "query_id": self.current_ep.get("id", ""),
            "num_steps": self.step_idx,
            "actions": list(self.actions),
            "num_regens": sum(self.actions),
            "total_lrm_tokens": sum(self.lrm_token_costs),
            "prm_scores": list(self.prm_scores),
        }
