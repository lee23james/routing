"""TRIM-Agg Router Policy: actor-critic with LayerNorm, Dropout, and orthogonal init.

Architecture aligned with TRIM source (RoutingPolicy).

State space (dim=5):
    0. min PRM score up to t-1
    1. product of PRM scores up to t-1
    2. PRM score at step t
    3. current step token length / TOKEN_NORMALISER
    4. step_num / max_steps

Actions:
    0 = accept SRM step (draft)
    1 = reject and use LRM step (target)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class RouterPolicy(nn.Module):

    def __init__(self, state_dim: int = 5, hidden_dim: int = 64,
                 action_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (PPO best practice from CleanRL/OpenAI)."""
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, action=None):
        h = self.encoder(state)
        logits = self.actor(h)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(h).squeeze(-1),
            logits.argmax(dim=-1),
        )

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy.

        Returns: (action, log_prob, value)
        """
        action, log_prob, _entropy, value, action_greedy = self.forward(state)
        if deterministic:
            action = action_greedy
            log_prob = Categorical(logits=self.actor(self.encoder(state))).log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update.

        Returns: (log_probs, values, entropy)
        """
        _action, log_probs, entropy, values, _greedy = self.forward(states, actions)
        return log_probs, values, entropy
