"""
COG-EW training entry point.

The current PPOTrainer collects episode rewards but does not optimize
policy parameters. This script supports explicitly requested rollout
diagnostics and prevents them from being mistaken for PPO training.
"""

import argparse
import sys
from pathlib import Path

import yaml

from sim.environment import HypersonicEWEnv
from rl.rl_agent import PPOAgent
from rl.training_loop import PPOTrainer

try:
    import torch
except ImportError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="COG-EW offline rollout diagnostic.")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Configuration path, relative to the project root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of diagnostic rounds.",
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=2,
        help="Episodes collected per diagnostic round.",
    )
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="Explicitly run diagnostics without PPO optimization.",
    )
    args = parser.parse_args()

    # Do not claim to train PPO until the optimizer loop is implemented.
    if not args.rollout_only:
        print("[STOP] PPO policy optimization has not been implemented.")
        print("The current trainer collects rewards but does not update weights.")
        print("No checkpoint was saved.")
        print("For an environment-only diagnostic, use " "--rollout-only explicitly.")
        return 2

    if args.epochs < 1 or args.episodes_per_epoch < 1:
        parser.error("--epochs and --episodes-per-epoch must both be at least 1.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        print(f"[ERROR] Configuration not found: {config_path}")
        return 2

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[ERROR] Could not read configuration: {exc}")
        return 2

    env = HypersonicEWEnv()
    actual_state_dim = int(env.observation_space.shape[0])
    actual_action_dim = int(env.action_space.shape[0])

    network_config = config.get("network", {})
    configured_state_dim = int(network_config.get("state_dim", actual_state_dim))
    configured_action_dim = int(
        network_config.get(
            "total_action_dim",
            network_config.get("guidance_action_dim", 2)
            + network_config.get("ew_action_dim", 2),
        )
    )

    if configured_state_dim != actual_state_dim:
        print(
            f"[ERROR] State dimension mismatch: "
            f"config={configured_state_dim}, environment={actual_state_dim}"
        )
        return 2

    if configured_action_dim != actual_action_dim:
        print(
            f"[ERROR] Action dimension mismatch: "
            f"config={configured_action_dim}, environment={actual_action_dim}"
        )
        return 2

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    agent = PPOAgent(state_dim=actual_state_dim, device=device)
    trainer = PPOTrainer(env=env, agent=agent, config=config)

    print("=" * 68)
    print("COG-EW -- ROLLOUT DIAGNOSTIC ONLY")
    print("=" * 68)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Observation dimension: {actual_state_dim}")
    print(f"Action dimension: {actual_action_dim}")
    print("Policy optimization: NOT RUN")
    print("Checkpoint saving: DISABLED")

    for epoch in range(1, args.epochs + 1):
        try:
            mean_reward = trainer.train_epoch(num_episodes=args.episodes_per_epoch)
        except Exception as exc:
            print(f"[ERROR] Diagnostic failed: {exc}")
            return 1

        print(
            f"Diagnostic round {epoch}/{args.epochs} | "
            f"Mean episode reward: {mean_reward:.4f}"
        )

    print("\n[PASS] Rollout diagnostic completed.")
    print("No PPO learning was performed.")
    print("No trained-model checkpoint was saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
