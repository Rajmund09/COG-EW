"""
PPO Training Loop with Curriculum Learning (rl/training_loop.py)
Manages PPO policy updates, rollout storage, and progressive curriculum scaling.
"""

import numpy as np
from sim.environment import HypersonicEWEnv
from rl.rl_agent import PPOAgent

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PPOTrainer:
    """
    Trainer for executing PPO policy optimization on HypersonicEWEnv.
    """

    def __init__(self, env: HypersonicEWEnv, agent: PPOAgent, config: dict = None):
        self.env = env
        self.agent = agent
        self.config = config or {}

    def train_epoch(self, num_episodes: int = 10):
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, log_prob, val = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated
                obs = next_obs

            total_rewards.append(ep_reward)

        mean_reward = float(np.mean(total_rewards))
        return mean_reward
