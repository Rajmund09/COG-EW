"""
COG-EW Phase 2: Synthetic PPO Smoke Test

Purpose:
- Validate PPO clipped-policy optimization in a toy environment.
- Confirm policy parameters actually change.
- Check policy/value losses and gradients for numerical stability.
- Evaluate using a fixed, reproducible synthetic dataset.
- Verify checkpoint save/load integrity.
- Exercise the synthetic environment's actual step() interface.

This test is intentionally independent of the operational simulation.
It does not modify or train the existing COG-EW guidance/EW policy.
"""

from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "synthetic_ppo_smoke_test.pt"

SEED = 42
STATE_DIM = 4
ACTION_DIM = 2

ROLLOUT_STEPS = 512
EPISODE_LENGTH = 8
PPO_UPDATES = 60
PPO_EPOCHS = 4
MINIBATCH_SIZE = 128

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
MIN_FINAL_ACCURACY = 0.75


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticPolicyEnvironment:
    """
    Bounded synthetic two-action environment.

    The correct action is determined by the sign of the first
    observation feature. Rewards are +1 for a correct action
    and -1 otherwise. Episodes have a fixed horizon.
    """

    def __init__(self, episode_length: int = EPISODE_LENGTH):
        if episode_length < 1:
            raise ValueError("episode_length must be at least 1")

        self.episode_length = episode_length
        self.step_count = 0
        self.current_state = None

    @staticmethod
    def _sample_state() -> np.ndarray:
        signal = np.random.uniform(-1.0, 1.0)
        noise = np.random.normal(
            0.0,
            0.5,
            size=STATE_DIM - 1,
        )

        return np.asarray(
            [signal, *noise],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        """Start a new episode and return its initial observation."""
        self.step_count = 0
        self.current_state = self._sample_state()
        return self.current_state.copy()

    def step(self, action: int):
        """
        Apply an action to the current observation.

        Returns:
            next_state, reward, terminated
        """
        if self.current_state is None:
            raise RuntimeError(
                "Environment must be reset before stepping, "
                "including after a terminal transition."
            )

        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")

        current_state = self.current_state
        correct_action = 1 if current_state[0] >= 0 else 0
        reward = 1.0 if action == correct_action else -1.0

        self.step_count += 1
        terminated = self.step_count >= self.episode_length

        next_state = self._sample_state()

        # The next observation from a terminal transition is not
        # used for value bootstrapping. A reset starts the next episode.
        self.current_state = None if terminated else next_state

        return next_state, reward, terminated


class ActorCritic(nn.Module):
    """Small categorical actor and scalar value critic."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )

        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, states: torch.Tensor):
        features = self.backbone(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        logits, values = self(states)
        distribution = Categorical(logits=logits)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_probs, entropy


def create_evaluation_dataset(
    count: int = 2048,
    seed: int = SEED + 1000,
):
    """
    Create a fixed synthetic evaluation dataset.

    A separate NumPy generator prevents evaluation from consuming
    the global random-number stream used by training rollouts.
    """
    if count < 1:
        raise ValueError("count must be positive")

    rng = np.random.default_rng(seed)

    signal = rng.uniform(
        -1.0,
        1.0,
        size=count,
    ).astype(np.float32)

    noise = rng.normal(
        0.0,
        0.5,
        size=(count, STATE_DIM - 1),
    ).astype(np.float32)

    features = np.column_stack([signal, noise]).astype(np.float32)

    labels = (signal >= 0).astype(np.int64)

    return features, labels


def evaluate_accuracy(
    model: ActorCritic,
    device: torch.device,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Evaluate on the same fixed synthetic examples each time."""
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                features,
                dtype=torch.float32,
                device=device,
            )

            logits, _ = model(state_tensor)
            predictions = logits.argmax(dim=-1).cpu().numpy()

        return float(np.mean(predictions == labels))

    finally:
        model.train(was_training)


def collect_rollout(
    model: ActorCritic,
    device: torch.device,
):
    """Collect aligned transitions and fixed old-policy log probabilities."""
    env = SyntheticPolicyEnvironment()

    observations = []
    actions = []
    rewards = []
    dones = []
    old_log_probs = []
    values = []
    next_observations = []

    state = env.reset()

    for _ in range(ROLLOUT_STEPS):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits, value = model(state_tensor)
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)

        action = int(action_tensor.item())

        next_state, reward, terminated = env.step(action)

        observations.append(state.copy())
        actions.append(action)
        rewards.append(reward)
        dones.append(float(terminated))
        old_log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        next_observations.append(next_state.copy())

        if terminated:
            state = env.reset()
        else:
            state = next_state

    rollout = {
        "states": torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=device,
        ),
        "actions": torch.as_tensor(
            actions,
            dtype=torch.long,
            device=device,
        ),
        "rewards": torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        ),
        "dones": torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device,
        ),
        "old_log_probs": torch.as_tensor(
            old_log_probs,
            dtype=torch.float32,
            device=device,
        ),
        "values": torch.as_tensor(
            values,
            dtype=torch.float32,
            device=device,
        ),
        "next_states": torch.as_tensor(
            np.asarray(next_observations),
            dtype=torch.float32,
            device=device,
        ),
    }

    # Basic rollout integrity checks.
    expected_shape = (ROLLOUT_STEPS,)

    for key in (
        "actions",
        "rewards",
        "dones",
        "old_log_probs",
        "values",
    ):
        if rollout[key].shape != expected_shape:
            raise RuntimeError(
                f"Unexpected rollout shape for {key}: " f"{rollout[key].shape}"
            )

    if rollout["states"].shape != (ROLLOUT_STEPS, STATE_DIM):
        raise RuntimeError("Unexpected rollout state shape.")

    if rollout["next_states"].shape != (ROLLOUT_STEPS, STATE_DIM):
        raise RuntimeError("Unexpected next-state shape.")

    if not torch.isfinite(rollout["old_log_probs"]).all().item():
        raise RuntimeError("Rollout contains non-finite old log probabilities.")

    return rollout


def calculate_gae(
    rollout,
    model: ActorCritic,
):
    """
    Calculate generalized advantage estimates and value targets.

    Terminal transitions do not bootstrap from the next observation.
    """
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    next_states = rollout["next_states"]

    with torch.no_grad():
        _, next_values = model(next_states)

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(
        (),
        dtype=torch.float32,
        device=rewards.device,
    )

    for index in reversed(range(len(rewards))):
        mask = 1.0 - dones[index]

        delta = rewards[index] + GAMMA * next_values[index] * mask - values[index]

        gae = delta + GAMMA * GAE_LAMBDA * mask * gae

        advantages[index] = gae

    returns = advantages + values

    if not torch.isfinite(advantages).all().item():
        raise RuntimeError("GAE produced non-finite advantages.")

    if not torch.isfinite(returns).all().item():
        raise RuntimeError("GAE produced non-finite returns.")

    return advantages, returns


def train_ppo(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Perform synthetic PPO clipped-policy and critic updates."""
    rollout = collect_rollout(model, device)

    advantages, returns = calculate_gae(rollout, model)

    # Normalize advantages for the policy objective.
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    batch_size = rollout["states"].shape[0]

    last_policy_loss = None
    last_value_loss = None
    last_entropy = None

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(
            batch_size,
            device=device,
        )

        for start in range(0, batch_size, MINIBATCH_SIZE):
            batch_indices = indices[start : start + MINIBATCH_SIZE]

            states = rollout["states"][batch_indices]
            actions = rollout["actions"][batch_indices]
            old_log_probs = rollout["old_log_probs"][batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            new_values, new_log_probs, entropy = model.evaluate_actions(states, actions)

            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            unclipped_objective = ratio * batch_advantages

            clipped_objective = (
                torch.clamp(
                    ratio,
                    1.0 - CLIP_EPS,
                    1.0 + CLIP_EPS,
                )
                * batch_advantages
            )

            policy_loss = -torch.minimum(
                unclipped_objective,
                clipped_objective,
            ).mean()

            value_loss = F.mse_loss(
                new_values,
                batch_returns,
            )

            mean_entropy = entropy.mean()

            total_loss = (
                policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * mean_entropy
            )

            if not torch.isfinite(total_loss).item():
                raise RuntimeError("PPO loss became non-finite.")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]

            if not gradients:
                raise RuntimeError("No gradients were produced.")

            if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                raise RuntimeError("PPO produced non-finite gradients.")

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
                error_if_nonfinite=True,
            )

            optimizer.step()

            last_policy_loss = float(policy_loss.detach().item())
            last_value_loss = float(value_loss.detach().item())
            last_entropy = float(mean_entropy.detach().item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }


def main() -> int:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed examples are generated once and reused for every evaluation.
    eval_features, eval_labels = create_evaluation_dataset()

    print("=" * 68)
    print("COG-EW PHASE 2 - SYNTHETIC PPO VALIDATION")
    print("=" * 68)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Rollout steps per update: {ROLLOUT_STEPS}")
    print(f"PPO updates: {PPO_UPDATES}")
    print(f"Fixed evaluation examples: {len(eval_labels)}")

    model = ActorCritic(
        STATE_DIM,
        ACTION_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    initial_accuracy = evaluate_accuracy(
        model,
        device,
        eval_features,
        eval_labels,
    )

    print(f"\nInitial held-out accuracy: {initial_accuracy:.3f}")

    latest_metrics = {}

    for update in range(1, PPO_UPDATES + 1):
        latest_metrics = train_ppo(
            model,
            optimizer,
            device,
        )

        if update % 10 == 0 or update == 1:
            accuracy = evaluate_accuracy(
                model,
                device,
                eval_features,
                eval_labels,
            )

            print(
                f"Update {update:03d}/{PPO_UPDATES} | "
                f"Policy loss: {latest_metrics['policy_loss']:.4f} | "
                f"Value loss: {latest_metrics['value_loss']:.4f} | "
                f"Entropy: {latest_metrics['entropy']:.4f} | "
                f"Held-out accuracy: {accuracy:.3f}"
            )

    final_accuracy = evaluate_accuracy(
        model,
        device,
        eval_features,
        eval_labels,
    )

    parameter_change = sum(
        (parameter.detach() - initial_parameters[name]).abs().sum().item()
        for name, parameter in model.named_parameters()
    )

    if not np.isfinite(parameter_change) or parameter_change <= 0:
        raise RuntimeError("Policy parameters did not change.")

    if not all(np.isfinite(value) for value in latest_metrics.values()):
        raise RuntimeError("Final PPO metrics are non-finite.")

    if final_accuracy < MIN_FINAL_ACCURACY:
        raise RuntimeError(
            "Synthetic PPO accuracy did not reach the minimum "
            f"{MIN_FINAL_ACCURACY:.2f} threshold; "
            f"observed {final_accuracy:.3f}."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "policy_state_dict": model.state_dict(),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "seed": SEED,
            "test_type": "synthetic_ppo_smoke_test",
            "final_accuracy": final_accuracy,
        },
        CHECKPOINT_PATH,
    )

    # Reload into a fresh model and compare outputs.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    reloaded_model = ActorCritic(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
    ).to(device)

    reloaded_model.load_state_dict(
        checkpoint["policy_state_dict"],
        strict=True,
    )

    reloaded_model.eval()
    model.eval()

    probe = torch.tensor(
        [
            [-0.8, 0.1, 0.0, -0.2],
            [0.8, -0.1, 0.2, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        original_logits, original_values = model(probe)
        loaded_logits, loaded_values = reloaded_model(probe)

    if not torch.allclose(
        original_logits,
        loaded_logits,
        atol=1e-6,
    ):
        raise RuntimeError("Checkpoint round-trip logits differ.")

    if not torch.allclose(
        original_values,
        loaded_values,
        atol=1e-6,
    ):
        raise RuntimeError("Checkpoint round-trip values differ.")

    print("\n" + "=" * 68)
    print("SYNTHETIC PPO VALIDATION RESULTS")
    print("=" * 68)
    print(f"Initial held-out accuracy: {initial_accuracy:.3f}")
    print(f"Final held-out accuracy:   {final_accuracy:.3f}")
    print(f"Absolute parameter change: {parameter_change:.6f}")
    print(f"Final policy loss:         {latest_metrics['policy_loss']:.6f}")
    print(f"Final value loss:          {latest_metrics['value_loss']:.6f}")
    print(f"Final entropy:             {latest_metrics['entropy']:.6f}")
    print("[PASS] Policy parameters changed.")
    print("[PASS] Final PPO metrics are finite.")
    print("[PASS] Fixed evaluation accuracy passed the threshold.")
    print("[PASS] Checkpoint save/load predictions match.")
    print(f"Test checkpoint: {CHECKPOINT_PATH}")
    print("=" * 68)
    print(
        "This validates only the isolated synthetic PPO test. "
        "It does not validate the existing COG-EW operational policy."
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n[FAIL] Synthetic PPO validation failed: {exc}")
        sys.exit(1)
