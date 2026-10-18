"""Train PPO in a closed synthetic 2D beacon-navigation environment.

This is a standalone simulation benchmark. It does not use the
hypersonic/EW environment or control any real-world equipment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SyntheticNavigationEnv:
    """Small continuous-position world with discrete movement actions."""

    WIDTH = 1000.0
    HEIGHT = 600.0
    SPEED = 8.5
    VEHICLE_RADIUS = 10.0
    BEACON_RADIUS = 22.0
    MAX_STEPS = 250

    OBSTACLES = (
        (350.0, 280.0, 60.0),
        (540.0, 300.0, 65.0),
        (700.0, 390.0, 50.0),
    )

    # Eight directions, including diagonals.
    DIRECTIONS = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=np.float32)
        self.target = np.asarray([900.0, 300.0], dtype=np.float32)
        self.steps = 0
        self.distance_travelled = 0.0
        self.status = "ready"

    def reset(self):
        # Small start-position variation improves generalization.
        self.position = np.asarray(
            [
                80.0,
                float(self.rng.uniform(285.0, 315.0)),
            ],
            dtype=np.float32,
        )
        self.steps = 0
        self.distance_travelled = 0.0
        self.status = "running"
        return self._observation(), {}

    def _observation(self):
        """Ten normalized features: position, target vector, 3 obstacle vectors."""
        pos = self.position
        target_delta = self.target - pos

        features = [
            pos[0] / self.WIDTH,
            pos[1] / self.HEIGHT,
            target_delta[0] / self.WIDTH,
            target_delta[1] / self.HEIGHT,
        ]

        obstacles = sorted(
            self.OBSTACLES,
            key=lambda o: math.hypot(o[0] - pos[0], o[1] - pos[1]),
        )

        for ox, oy, _radius in obstacles:
            features.extend(
                [
                    (ox - pos[0]) / self.WIDTH,
                    (oy - pos[1]) / self.HEIGHT,
                ]
            )

        return np.asarray(features, dtype=np.float32)

    def step(self, action: int):
        if self.status != "running":
            raise RuntimeError("Call reset() before stepping a finished episode.")

        self.steps += 1

        previous = self.position.copy()
        old_distance = float(np.linalg.norm(self.target - previous))

        direction = self.DIRECTIONS[int(action)]
        candidate = previous + direction * self.SPEED

        # Keep the virtual vehicle inside the world.
        candidate[0] = np.clip(
            candidate[0], self.VEHICLE_RADIUS, self.WIDTH - self.VEHICLE_RADIUS
        )
        candidate[1] = np.clip(
            candidate[1], self.VEHICLE_RADIUS, self.HEIGHT - self.VEHICLE_RADIUS
        )

        collision = False
        for ox, oy, radius in self.OBSTACLES:
            obstacle_distance = math.hypot(
                float(candidate[0] - ox), float(candidate[1] - oy)
            )
            if obstacle_distance <= radius + self.VEHICLE_RADIUS:
                collision = True
                break

        if not collision:
            self.position = candidate

        moved = float(np.linalg.norm(self.position - previous))
        self.distance_travelled += moved

        new_distance = float(np.linalg.norm(self.target - self.position))

        # Reward progress toward the virtual beacon, not a real-world target.
        reward = (old_distance - new_distance) / self.SPEED
        reward -= 0.02

        terminated = False
        truncated = False

        if collision:
            reward -= 10.0
            terminated = True
            self.status = "collision"

        elif new_distance <= self.BEACON_RADIUS:
            reward += 20.0
            terminated = True
            self.status = "completed"

        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.status = "time_limit"

        info = {
            "status": self.status if terminated or truncated else "running",
            "distance_to_beacon": new_distance,
            "distance_travelled": self.distance_travelled,
            "collision": collision,
            "success": self.status == "completed",
        }

        return self._observation(), float(reward), terminated, truncated, info


class ActorCritic(nn.Module):
    """Discrete-action actor and scalar value critic."""

    def __init__(self, observation_dim: int = 10, action_dim: int = 8):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations):
        features = self.backbone(observations)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    def act(self, observations, deterministic=False):
        logits, values = self(observations)
        distribution = Categorical(logits=logits)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(self, observations, actions):
        logits, values = self(observations)
        distribution = Categorical(logits=logits)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values


def evaluate_policy(model, device, episodes=20, seed=10000):
    """Evaluate deterministically without using training rollouts."""
    env = SyntheticNavigationEnv(seed=seed)
    rewards = []
    successes = 0
    collisions = 0
    episode_lengths = []

    model.eval()

    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            tensor_obs = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                action, _, _ = model.act(tensor_obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        successes += int(info["success"])
        collisions += int(info["collision"])
        episode_lengths.append(env.steps)

    model.train()

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PPO training for synthetic 2D beacon navigation."
    )
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--checkpoint",
        default="models/synthetic_navigation_ppo.pt",
    )
    parser.add_argument(
        "--report",
        default="reports/synthetic_navigation_ppo_evaluation.json",
    )
    args = parser.parse_args()

    if args.timesteps < 1 or args.rollout_steps < 1:
        parser.error("Timesteps and rollout steps must be positive.")
    if args.epochs < 1 or args.minibatch_size < 1:
        parser.error("Epochs and minibatch size must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested, but torch.cuda.is_available() is false.")

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = SyntheticNavigationEnv(seed=args.seed)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    observations, _ = env.reset()
    total_steps = 0
    update = 0
    recent_rewards = []
    episode_reward = 0.0
    completed_episodes = 0
    collision_episodes = 0

    print("=" * 68)
    print("COG-EW — SYNTHETIC NAVIGATION PPO TRAINING")
    print("=" * 68)
    print(f"Device: {device}")
    print(f"Training budget: {args.timesteps:,} environment steps")
    print(f"Observation dimensions: 10 | Discrete actions: 8")
    print("Environment: closed synthetic 2D world")
    print("No RF model, RF test set, or operational EW environment is used.")

    model.train()

    while total_steps < args.timesteps:
        count = min(args.rollout_steps, args.timesteps - total_steps)

        obs_buf = np.zeros((count, 10), dtype=np.float32)
        action_buf = np.zeros(count, dtype=np.int64)
        logprob_buf = np.zeros(count, dtype=np.float32)
        reward_buf = np.zeros(count, dtype=np.float32)
        done_buf = np.zeros(count, dtype=np.float32)
        value_buf = np.zeros(count, dtype=np.float32)

        for t in range(count):
            obs_buf[t] = observations

            obs_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = model.act(obs_tensor)

            action_int = int(action.item())
            action_buf[t] = action_int
            logprob_buf[t] = float(log_prob.item())
            value_buf[t] = float(value.item())

            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            reward_buf[t] = reward
            done_buf[t] = float(done)
            episode_reward += reward
            observations = next_obs
            total_steps += 1

            if done:
                recent_rewards.append(episode_reward)
                recent_rewards = recent_rewards[-100:]
                completed_episodes += int(info["success"])
                collision_episodes += int(info["collision"])
                episode_reward = 0.0
                observations, _ = env.reset()

        with torch.no_grad():
            last_obs = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            ).unsqueeze(0)
            _, last_value_tensor = model(last_obs)
            last_value = float(last_value_tensor.item())

        # Generalized Advantage Estimation (GAE).
        advantages = np.zeros(count, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(count)):
            if t == count - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - done_buf[t]
                next_value = value_buf[t + 1]

            delta = reward_buf[t] + gamma * next_value * next_nonterminal - value_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + value_buf

        obs_t = torch.as_tensor(obs_buf, device=device)
        actions_t = torch.as_tensor(action_buf, device=device)
        old_logprobs_t = torch.as_tensor(logprob_buf, device=device)
        advantages_t = torch.as_tensor(advantages, device=device)
        returns_t = torch.as_tensor(returns, device=device)

        # PPO clipped surrogate optimization.
        batch_indices = np.arange(count)
        policy_losses = []
        value_losses = []
        entropy_values = []

        for _ in range(args.epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, count, args.minibatch_size):
                indices = batch_indices[start : start + args.minibatch_size]
                index_t = torch.as_tensor(indices, device=device)

                new_logprobs, entropy, new_values = model.evaluate_actions(
                    obs_t[index_t], actions_t[index_t]
                )

                log_ratio = new_logprobs - old_logprobs_t[index_t]
                ratio = torch.exp(log_ratio)

                batch_advantages = advantages_t[index_t]
                if len(indices) > 1:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std(unbiased=False) + 1e-8
                    )

                unclipped = ratio * batch_advantages
                clipped = (
                    torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    * batch_advantages
                )

                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (new_values - returns_t[index_t]).pow(2).mean()
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropy_values.append(float(entropy_mean.detach().item()))

        update += 1
        mean_recent = float(np.mean(recent_rewards)) if recent_rewards else float("nan")

        print(
            f"Update {update:04d} | Steps {total_steps:7d}/{args.timesteps} | "
            f"Recent reward {mean_recent:8.2f} | "
            f"Policy loss {np.mean(policy_losses):8.4f} | "
            f"Value loss {np.mean(value_losses):8.4f}"
        )

    checkpoint_path = ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_dim": 10,
            "action_dim": 8,
            "timesteps": total_steps,
            "seed": args.seed,
            "environment": "synthetic_2d_beacon_navigation",
        },
        checkpoint_path,
    )

    evaluation = evaluate_policy(model, device=device, episodes=20)
    evaluation.update(
        {
            "environment": "synthetic_2d_beacon_navigation",
            "training_timesteps": total_steps,
            "updates": update,
            "device": device,
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "completed_training_episodes": completed_episodes,
            "collision_training_episodes": collision_episodes,
            "final_recent_training_reward": (
                float(np.mean(recent_rewards)) if recent_rewards else None
            ),
        }
    )

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(evaluation, indent=2),
        encoding="utf-8",
    )

    print("\n=== HELD-OUT SIMULATION EVALUATION ===")
    print(json.dumps(evaluation, indent=2))
    print(f"\nCheckpoint saved: {checkpoint_path}")
    print(f"Evaluation report: {report_path}")
    print("Training and evaluation refer only to the synthetic navigation task.")


if __name__ == "__main__":
    main()
