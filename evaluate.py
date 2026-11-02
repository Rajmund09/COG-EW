# -*- coding: utf-8 -*-
"""
COG-EW Primary Reinforcement Learning Policy Evaluation Entry Point.
Evaluates the trained PPO Actor-Critic neural checkpoint on CogEWEnv (gymnasium.Env).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ppo_gymnasium import evaluate_ppo


def main():
    parser = argparse.ArgumentParser(description="Evaluate COG-EW PPO Reinforcement Learning Policy")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes to evaluate")
    args = parser.parse_args()

    evaluate_ppo(num_episodes=args.episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
