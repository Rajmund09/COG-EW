# -*- coding: utf-8 -*-
"""
COG-EW Primary Reinforcement Learning Training Entry Point.
Executes PPO Actor-Critic training on the continuous CogEWEnv (gymnasium.Env).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ppo_with_expert_warmup import train_guided_ppo


def main():
    parser = argparse.ArgumentParser(description="Train COG-EW PPO Reinforcement Learning Policy in Gymnasium")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total environment steps to train")
    parser.add_argument("--bc-epochs", type=int, default=15, help="Behavioral cloning pre-training epochs")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel rollout environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    checkpoint_path = train_guided_ppo(
        total_timesteps=args.timesteps,
        bc_epochs=args.bc_epochs,
        num_envs=args.envs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print(f"\n[SUCCESS] Trained PPO policy saved to: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
