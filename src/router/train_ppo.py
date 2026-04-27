"""PPO training for TRIM-Agg router (aligned with TRIM source).

Key changes from naive PPO (aligned with TRIM TRIM_Agg.py):
  - Per-step cost reward (not end-of-episode only)
  - Clipped value loss
  - Entropy coefficient annealing (0.01 → 0.001)
  - LinearLR scheduler
  - Orthogonal-init actor-critic with LayerNorm + Dropout

Usage:
    # TRIM-Agg (outcome-only reward)
    python -m router.train_ppo --episodes_path data/episodes/all_episodes.jsonl --lam 3e-4

    # TRIM-Agg + Rubric Process Reward
    python -m router.train_ppo --episodes_path data/episodes/all_episodes.jsonl \
        --lam 3e-4 --lam_rubric 0.3 --rubric_weights data/rubrics/rubric_weights.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.policy import RouterPolicy
from router.env import TRIMEnv
from config import (
    STATE_DIM, HIDDEN_DIM, ACTION_DIM, DROPOUT,
    LR, CLIP_COEF, ENTROPY_COEF, ENTROPY_COEF_FINAL,
    VALUE_LOSS_COEF, GAE_LAMBDA, GAMMA,
    MAX_GRAD_NORM, PPO_EPOCHS,
    NORMALIZE_ADVANTAGES, TASK_REWARD, CHECKPOINTS_DIR,
)


@dataclass
class RolloutBuffer:
    states: List = field(default_factory=list)
    actions: List = field(default_factory=list)
    log_probs: List = field(default_factory=list)
    rewards: List = field(default_factory=list)
    values: List = field(default_factory=list)
    dones: List = field(default_factory=list)

    def clear(self):
        for attr in [self.states, self.actions, self.log_probs,
                     self.rewards, self.values, self.dones]:
            attr.clear()


def compute_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_val = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train_ppo(
    env: TRIMEnv,
    lam: float,
    lam_rubric: float = 0.0,
    num_epochs: int = 200,
    episodes_per_epoch: int = 64,
    ppo_epochs: int = PPO_EPOCHS,
    mini_batch_size: int = 64,
    device: str = "cpu",
    save_dir: str = None,
    log_interval: int = 10,
    tag: str = "",
):
    policy = RouterPolicy(STATE_DIM, HIDDEN_DIM, ACTION_DIM, DROPOUT).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5, weight_decay=1e-5)

    total_updates = num_epochs
    lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_updates,
    )

    label = f"lam{lam:.0e}"
    if lam_rubric > 0:
        label += f"_rubric{lam_rubric}"
    if tag:
        label = f"{tag}_{label}"

    if save_dir is None:
        save_dir = os.path.join(CHECKPOINTS_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    best_reward = -float("inf")
    log_data = []
    update_count = 0

    for epoch in range(num_epochs):
        policy.train()
        buffer = RolloutBuffer()
        epoch_rewards, epoch_regens, epoch_corrects = [], [], []

        for _ in range(episodes_per_epoch):
            state = env.reset()
            ep_states, ep_actions, ep_log_probs, ep_values, ep_dones = [], [], [], [], []
            ep_rewards = []

            done = False
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, log_prob, value = policy.get_action(state_t)

                action_int = action.item()
                next_state, _, done, info = env.step(action_int)

                ep_states.append(state)
                ep_actions.append(action_int)
                ep_log_probs.append(log_prob.item())
                ep_values.append(value.item())
                ep_dones.append(float(done))
                ep_rewards.append(0.0)
                state = next_state

            # End-of-episode reward: outcome - cost (aggregated)
            ep_reward = env.compute_episode_reward(lam, lam_rubric)
            if ep_rewards:
                ep_rewards[-1] = ep_reward

            ep_info = env.get_episode_info()
            total_ep_reward = ep_reward

            buffer.states.extend(ep_states)
            buffer.actions.extend(ep_actions)
            buffer.log_probs.extend(ep_log_probs)
            buffer.rewards.extend(ep_rewards)
            buffer.values.extend(ep_values)
            buffer.dones.extend(ep_dones)

            epoch_rewards.append(total_ep_reward)
            epoch_regens.append(ep_info["num_regens"])
            cost = lam * ep_info["total_lrm_tokens"]
            epoch_corrects.append(1.0 if ep_reward > -cost else 0.0)

        # ---- PPO Update (aligned with TRIM source) ----
        if not buffer.states:
            continue

        policy.train()
        states_t = torch.FloatTensor(np.array(buffer.states)).to(device)
        actions_t = torch.LongTensor(buffer.actions).to(device)
        old_log_probs_t = torch.FloatTensor(buffer.log_probs).to(device)
        old_values_t = torch.FloatTensor(buffer.values).to(device)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        )
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)

        # Linearly anneal entropy coefficient
        frac = min(update_count / max(total_updates, 1), 1.0)
        ent_coef = ENTROPY_COEF + frac * (ENTROPY_COEF_FINAL - ENTROPY_COEF)

        if NORMALIZE_ADVANTAGES and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(buffer.states)
        for _ in range(ppo_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, mini_batch_size):
                end = min(start + mini_batch_size, n_samples)
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_old_val = old_values_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]

                new_lp, new_v, entropy = policy.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_lp - mb_old_lp)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg1, pg2).mean()

                # Clipped value loss (aligned with TRIM)
                v_unclipped = F.mse_loss(new_v, mb_ret, reduction="none")
                v_clipped_val = mb_old_val + torch.clamp(
                    new_v - mb_old_val,
                    -CLIP_COEF * TASK_REWARD,
                    CLIP_COEF * TASK_REWARD,
                )
                v_clipped = F.mse_loss(v_clipped_val, mb_ret, reduction="none")
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + VALUE_LOSS_COEF * v_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        lr_scheduler.step()
        update_count += 1

        # ---- Logging ----
        mean_reward = float(np.mean(epoch_rewards))
        mean_regens = float(np.mean(epoch_regens))
        mean_correct = float(np.mean(epoch_corrects))
        current_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "mean_reward": mean_reward,
            "mean_regens": mean_regens,
            "mean_correct": mean_correct,
            "lr": current_lr,
            "entropy_coef": ent_coef,
        }
        log_data.append(log_entry)

        if epoch % log_interval == 0:
            print(f"Epoch {epoch:4d} | reward={mean_reward:.4f} | "
                  f"regens={mean_regens:.1f} | correct={mean_correct:.3f} | "
                  f"lr={current_lr:.2e} | ent_coef={ent_coef:.4f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(policy.state_dict(), os.path.join(save_dir, "best.pt"))

    torch.save(policy.state_dict(), os.path.join(save_dir, "final.pt"))
    with open(os.path.join(save_dir, "train_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\nTraining done. Best reward: {best_reward:.4f}")
    print(f"Checkpoints → {save_dir}")
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_path", type=str, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--lam_rubric", type=float, default=0.0,
                        help="Rubric process reward weight (0 = TRIM-Agg only)")
    parser.add_argument("--rubric_weights", type=str, default=None,
                        help="Path to rubric_weights.json")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--episodes_per_epoch", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    rubric_w = None
    if args.rubric_weights and os.path.exists(args.rubric_weights):
        with open(args.rubric_weights) as f:
            data = json.load(f)
        rubric_w = data.get("weights", None)
        active = data.get("active_rubrics", [])
        print(f"Loaded rubric weights ({len(active)} active): {active}")
        for k, v in sorted((rubric_w or {}).items(), key=lambda x: -x[1]):
            if v > 0:
                print(f"  {k}: {v:.4f}")

    env = TRIMEnv(args.episodes_path, rubric_weights=rubric_w)
    train_ppo(
        env=env,
        lam=args.lam,
        lam_rubric=args.lam_rubric,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        device=args.device,
        save_dir=args.save_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
