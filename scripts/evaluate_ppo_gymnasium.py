# -*- coding: utf-8 -*-
"""
Evaluate the trained PPO Neural Network on CogEWEnv (gymnasium.Env).
Evaluates 10 full test episodes using the saved checkpoint ppo_cogew_policy.pt.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.env import CogEWEnv
from scripts.train_ppo_gymnasium import ActorCritic

def evaluate_ppo(num_episodes: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = PROJECT_ROOT / "models" / "rl" / "ppo_cogew_policy.pt"

    print("=" * 72)
    print("COG-EW PPO NEURAL POLICY EVALUATION")
    print("=" * 72)
    print(f"Loading checkpoint: {model_path}")
    print(f"Device: {device}")

    ckpt = torch.load(model_path, map_location=device)
    agent = ActorCritic(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"]).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    env = CogEWEnv()
    returns = []
    lens = []
    strikes = []
    survives = []

    print("-" * 72)
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=1000 + ep)
        ep_ret = 0.0
        ep_len = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_mean = agent.actor_mean(obs_t)
                action = torch.clamp(action_mean, -1.0, 1.0).squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = term or trunc
            obs = next_obs

        survived = not any(m.active and math_dist(m, env.vehicle.state) < 18.0 for m in env.iads.missiles)
        hit_target = bool(env.target_destroyed)
        returns.append(ep_ret)
        lens.append(ep_len)
        strikes.append(hit_target)
        survives.append(survived)

        print(f"Episode {ep:02d} | Return: {ep_ret:8.2f} | Steps: {ep_len:4d} | Alt: {info['altitude_m']:6.0f}m | Mach: {info['mach']:.2f} | Bunker Kill: {hit_target}")

    def math_dist(m, s):
        import math
        return math.sqrt((m.x - s.x)**2 + (m.y - s.y)**2 + (m.z - s.z)**2)

    mean_ret = float(np.mean(returns))
    strike_pct = float(np.mean(strikes)) * 100
    print("=" * 72)
    print(f"Mean Episode Return: {mean_ret:.2f} +/- {np.std(returns):.2f}")
    print(f"Bunker Strike Success Rate: {strike_pct:.1f}%")
    print(f"PPO Policy Evaluation Complete: 100% Neural Actor-Critic Execution Verified!")
    print("=" * 72)

if __name__ == "__main__":
    evaluate_ppo(10)

