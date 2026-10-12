"""
COG-EW training entry point.

The current PPOTrainer collects episode rewards but does not optimize
policy parameters. This file therefore supports rollout diagnostics only
and deliberately refuses to create a "trained" checkpoint.
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
        help="YAML configuration path, relative to the project root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of rollout diagnostic rounds.",
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=2,
        help="Episodes to collect in each diagnostic round.",
    )
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        default=True,
        help="Run diagnostics without policy weight updates.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.episodes_per_epoch < 1:
        parser.error("--epochs and --episodes-per-epoch must both be at least 1.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        print(f"[ERROR] Configuration file not found: {config_path}")
        return 2

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[ERROR] Could not read configuration: {exc}")
        return 2

    network_config = config.get("network", {})
    env = HypersonicEWEnv()

    actual_state_dim = int(env.observation_space.shape[0])
    actual_action_dim = int(env.action_space.shape[0])
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
            "[ERROR] State dimension mismatch: "
            f"config={configured_state_dim}, environment={actual_state_dim}"
        )
        return 2

    if configured_action_dim != actual_action_dim:
        print(
            "[ERROR] Action dimension mismatch: "
            f"config={configured_action_dim}, environment={actual_action_dim}"
        )
        return 2

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    agent = PPOAgent(
        state_dim=actual_state_dim,
        device=device,
    )
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=config,
    )

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
            print(f"\n[ERROR] Rollout diagnostic failed: {exc}")
            return 1

        print(
            f"Diagnostic round {epoch}/{args.epochs} | "
            f"Mean episode reward: {mean_reward:.4f}"
        )

    print("\n[PASS] Rollout diagnostic completed.")
    print(
        "This run did not establish PPO learning and did not save "
        "a trained-model checkpoint."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
