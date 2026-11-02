# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization (PPO) Training Engine for COG-EW Gymnasium Environment.
Trains a continuous Actor-Critic neural policy on CogEWEnv (gymnasium.Env) with:
- State Space: Box(28,) continuous physical observations
- Action Space: Box(4,) continuous actuators in [-1.0, 1.0]
- Algorithm: Clipped PPO with Generalized Advantage Estimation (GAE)
- Hardware: Accelerated with PyTorch 2.5 on CUDA (NVIDIA RTX GPU)
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.env import CogEWEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 28, action_dim: int = 4):
        super().__init__()
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        # Actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            # Enforce continuous action bounds [-1.0, 1.0]
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value


def train_ppo(
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    num_envs: int = 4,
    num_steps: int = 512,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 10,
    num_minibatches: int = 4,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    device_name: str | None = None,
):
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("=" * 72)
    print("COG-EW GYMNASIUM PPO REINFORCEMENT LEARNING TRAINER")
    print("=" * 72)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Total Timesteps Target: {total_timesteps:,}")
    print(f"Parallel Envs: {num_envs} | Rollout Horizon: {num_steps} | Batch: {num_envs * num_steps}")
    print(f"Update Epochs: {update_epochs} | Minibatches: {num_minibatches} | LR: {learning_rate}")
    print("=" * 72)

    # Initialize environments
    envs = [CogEWEnv() for _ in range(num_envs)]
    state_dim = envs[0].observation_space.shape[0]  # 28
    action_dim = envs[0].action_space.shape[0]      # 4

    agent = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Rollout storage buffers
    obs_buffer = torch.zeros((num_steps, num_envs, state_dim), dtype=torch.float32).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32).to(device)
    logprobs_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
    dones_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
    values_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)

    # Initial environment reset
    next_obs = torch.zeros((num_envs, state_dim), dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs, dtype=torch.float32).to(device)

    for i in range(num_envs):
        o, _ = envs[i].reset(seed=seed + i * 100)
        next_obs[i] = torch.tensor(o, dtype=torch.float32).to(device)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size

    history = {
        "timesteps": [],
        "mean_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "fps": [],
        "success_rate": [],
    }

    global_step = 0
    start_time = time.time()
    recent_ep_returns = []
    recent_ep_lens = []
    episode_returns_acc = np.zeros(num_envs)
    episode_lens_acc = np.zeros(num_envs)
    best_mean_return = -float("inf")

    models_dir = PROJECT_ROOT / "models" / "rl"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "ppo_cogew_policy.pt"

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # Rollout collection loop
        for step in range(num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buffer[step] = value.flatten()

            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # Step all environments
            action_np = action.cpu().numpy()
            for env_idx in range(num_envs):
                o, r, terminated, truncated, info = envs[env_idx].step(action_np[env_idx])
                rewards_buffer[step, env_idx] = r
                done = terminated or truncated

                episode_returns_acc[env_idx] += r
                episode_lens_acc[env_idx] += 1

                if done:
                    recent_ep_returns.append(float(episode_returns_acc[env_idx]))
                    recent_ep_lens.append(int(episode_lens_acc[env_idx]))
                    episode_returns_acc[env_idx] = 0.0
                    episode_lens_acc[env_idx] = 0.0
                    o, _ = envs[env_idx].reset()

                next_obs[env_idx] = torch.tensor(o, dtype=torch.float32).to(device)
                next_done[env_idx] = 1.0 if done else 0.0

        # Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        # Flatten the rollout buffer
        b_obs = obs_buffer.reshape(-1, state_dim)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        pg_losses = []
        v_losses = []
        ent_losses = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        fps = int(global_step / (time.time() - start_time))
        avg_ret = float(np.mean(recent_ep_returns[-20:])) if recent_ep_returns else 0.0
        avg_len = float(np.mean(recent_ep_lens[-20:])) if recent_ep_lens else 0.0

        history["timesteps"].append(global_step)
        history["mean_rewards"].append(avg_ret)
        history["policy_losses"].append(float(np.mean(pg_losses)))
        history["value_losses"].append(float(np.mean(v_losses)))
        history["entropies"].append(float(np.mean(ent_losses)))
        history["fps"].append(fps)

        print(
            f"Update {update:03d}/{num_updates:03d} | "
            f"Step: {global_step:06d}/{total_timesteps} | "
            f"Mean Return: {avg_ret:8.2f} | "
            f"Ep Len: {avg_len:5.1f} | "
            f"Pi Loss: {np.mean(pg_losses):+.4f} | "
            f"V Loss: {np.mean(v_losses):6.3f} | "
            f"FPS: {fps:4d}"
        )

        # Save best model
        if avg_ret > best_mean_return and len(recent_ep_returns) >= 5:
            best_mean_return = avg_ret
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "mean_return": best_mean_return,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                },
                best_model_path,
            )

    # Save final model checkpoint
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "mean_return": avg_ret,
            "state_dim": state_dim,
            "action_dim": action_dim,
        },
        models_dir / "ppo_cogew_policy_final.pt",
    )

    # Save training history JSON
    history_path = models_dir / "ppo_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("=" * 72)
    print(f"PPO TRAINING COMPLETE IN {total_time:.1f}s!")
    print(f"Best Mean Return: {best_mean_return:.2f}")
    print(f"Model Checkpoint Saved to: {best_model_path}")
    print(f"Training History Saved to: {history_path}")
    print("=" * 72)
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Trainer for COG-EW Gymnasium Environment")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total environment steps to train")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel rollout environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.envs,
    )

