"""COG-EW read-only RL configuration preflight.

Checks configuration consistency without training a model, running a
simulation episode, or reading/writing model checkpoints.
"""

import sys
from pathlib import Path

try:
    import yaml
    import torch
except ImportError as exc:
    print(f"[ERROR] Required dependency is unavailable: {exc}")
    raise SystemExit(2)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"

REQUIRED_PPO_KEYS = (
    "learning_rate_actor",
    "learning_rate_critic",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "ppo_epochs",
    "batch_size",
    "entropy_coef",
    "value_loss_coef",
    "max_grad_norm",
)

REQUIRED_TRAINING_KEYS = (
    "total_timesteps",
    "checkpoint_interval",
    "curriculum_stages",
)


def fail(message):
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def require_positive_number(section, key, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{section}.{key} must be numeric.")
    if not value > 0:
        fail(f"{section}.{key} must be greater than zero.")


def main():
    print("=" * 68)
    print("COG-EW — READ-ONLY CONFIGURATION PREFLIGHT")
    print("=" * 68)

    if not CONFIG_PATH.is_file():
        fail(f"Configuration file not found: {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        fail(f"Could not parse configuration: {exc}")

    if not isinstance(config, dict):
        fail("Configuration root must be a YAML mapping.")

    for section_name in ("network", "ppo", "training"):
        if not isinstance(config.get(section_name), dict):
            fail(f"Missing or invalid configuration section: {section_name}")

    network = config["network"]
    ppo = config["ppo"]
    training = config["training"]

    try:
        from sim.environment import HypersonicEWEnv
        from rl.policy_network import MultiHeadPolicyNet
    except Exception as exc:
        fail(f"Could not import environment or policy: {exc}")

    # Verify configuration dimensions against the declared environment.
    env = HypersonicEWEnv()
    actual_state_dim = int(env.observation_space.shape[0])
    actual_action_dim = int(env.action_space.shape[0])

    configured_state_dim = network.get("state_dim")
    configured_action_dim = network.get("total_action_dim")
    guidance_dim = network.get("guidance_action_dim")
    ew_dim = network.get("ew_action_dim")
    hidden_dims = network.get("hidden_dims")

    for key, value in (
        ("state_dim", configured_state_dim),
        ("total_action_dim", configured_action_dim),
        ("guidance_action_dim", guidance_dim),
        ("ew_action_dim", ew_dim),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            fail(f"network.{key} must be an integer.")

    if configured_state_dim != actual_state_dim:
        fail(
            f"State dimension mismatch: config={configured_state_dim}, "
            f"environment={actual_state_dim}"
        )

    if configured_action_dim != actual_action_dim:
        fail(
            f"Action dimension mismatch: config={configured_action_dim}, "
            f"environment={actual_action_dim}"
        )

    if guidance_dim < 1 or ew_dim < 1:
        fail("Guidance and EW dimensions must both be positive.")

    if guidance_dim + ew_dim != configured_action_dim:
        fail("Guidance and EW dimensions do not sum to total_action_dim.")

    if (
        not isinstance(hidden_dims, list)
        or not hidden_dims
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in hidden_dims
        )
    ):
        fail("network.hidden_dims must be a non-empty list of positive integers.")

    # Inspect the model's actual backbone without training it.
    try:
        model = MultiHeadPolicyNet(
            state_dim=configured_state_dim,
            hidden_dims=hidden_dims,
        )
    except Exception as exc:
        fail(f"Could not construct policy from configuration: {exc}")

    actual_hidden_dims = [
        layer.out_features
        for layer in model.backbone
        if isinstance(layer, torch.nn.Linear)
    ]

    if actual_hidden_dims != hidden_dims:
        fail(
            f"Policy hidden dimensions {actual_hidden_dims} do not match "
            f"configuration {hidden_dims}."
        )

    # Validate required PPO settings.
    for key in REQUIRED_PPO_KEYS:
        if key not in ppo:
            fail(f"Missing PPO configuration key: {key}")

    for key in (
        "learning_rate_actor",
        "learning_rate_critic",
        "clip_eps",
        "max_grad_norm",
    ):
        require_positive_number("ppo", key, ppo[key])

    for key in ("gamma", "gae_lambda"):
        value = ppo[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            fail(f"ppo.{key} must be numeric and between 0 and 1.")

    for key in ("entropy_coef", "value_loss_coef"):
        value = ppo[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            fail(f"ppo.{key} must be a non-negative number.")

    for key in ("ppo_epochs", "batch_size"):
        value = ppo[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            fail(f"ppo.{key} must be a positive integer.")

    # Validate training settings without launching training.
    for key in REQUIRED_TRAINING_KEYS:
        if key not in training:
            fail(f"Missing training configuration key: {key}")
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            fail(f"training.{key} must be a positive integer.")

    print(f"[PASS] Configuration parsed: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[PASS] State dimension: {actual_state_dim}")
    print(f"[PASS] Action dimension: {actual_action_dim}")
    print(f"[PASS] Hidden dimensions: {actual_hidden_dims}")
    print(f"[PASS] PPO parameters: {len(REQUIRED_PPO_KEYS)} checked")
    print(f"[PASS] Training settings: {len(REQUIRED_TRAINING_KEYS)} checked")

    print("\n[WARNING] This is a configuration check, not a training run.")
    print("[WARNING] The current production trainer collects rewards but does")
    print("          not perform PPO policy optimization.")
    print("[WARNING] Passing this preflight does not prove a checkpoint is trained.")
    print("\nNo simulation episode was run. No checkpoint was read or written.")
    print("[PASS] Configuration preflight completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
