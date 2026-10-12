"""
Main Training Execution Script (train.py)
Entry point for training the joint Cognitive EW-Guidance Co-Designer reinforcement learning policy.
"""

import argparse
import yaml
import os
from sim.environment import HypersonicEWEnv
from rl.rl_agent import PPOAgent
from rl.training_loop import PPOTrainer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def main():
    parser = argparse.ArgumentParser(description="Train Cognitive EW-Guidance Agent")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    print("==========================================================================")
    print("      COGNITIVE MULTI-DOMAIN EW–GUIDANCE CO-DESIGNER (TRAINING)          ")
    print("==========================================================================")

    if not HAS_TORCH:
        print("[!] PyTorch is not installed in the current environment.")
        print("[!] Install dependencies using: pip install -r requirements.txt")

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        print(f"[+] Loaded training configuration from {args.config}")
    else:
        config = {}
        print(f"[!] Config file {args.config} not found. Using defaults.")

    # Initialize Environment & Agent
    env = HypersonicEWEnv()
    agent = PPOAgent(state_dim=16)
    trainer = PPOTrainer(env=env, agent=agent, config=config)

    print("\n[+] Starting PPO Policy Optimization Loop...")
    for epoch in range(1, args.epochs + 1):
        mean_reward = trainer.train_epoch(num_episodes=5)
        print(f"    Epoch {epoch}/{args.epochs} | Mean Episode Reward: {mean_reward:.2f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_model.pt"
    agent.save_checkpoint(checkpoint_path)
    print(f"\n[+] Saved trained model checkpoint to {checkpoint_path}")
    print("[+] Training completed successfully.")


if __name__ == "__main__":
    main()
