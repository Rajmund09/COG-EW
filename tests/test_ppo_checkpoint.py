"""
COG-EW Phase 2D: Synthetic PPO checkpoint integrity tests.

Checks:
1. Model weights can be saved and reloaded.
2. Reloaded model produces identical outputs.
3. Optimizer state survives checkpoint round-trip.
4. Missing checkpoint paths are handled explicitly.

This test uses only a small synthetic network.
It does not access or modify production COG-EW checkpoints.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


class TinyActorCritic(nn.Module):
    """Minimal synthetic actor-critic used only for testing."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
        )
        self.actor = nn.Linear(16, 2)
        self.critic = nn.Linear(16, 1)

    def forward(self, states):
        features = self.shared(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values


def save_synthetic_checkpoint(path, model, optimizer):
    """Save synthetic model and optimizer state."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "format_version": 1,
                "purpose": "synthetic_test_only",
            },
        },
        path,
    )


def load_synthetic_checkpoint(path, model, optimizer=None):
    """Load a checkpoint or fail clearly if it does not exist."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Synthetic checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "metadata",
    }

    if not required_keys.issubset(checkpoint):
        raise ValueError("Checkpoint is missing required fields.")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["metadata"]


class TestSyntheticCheckpoint(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.states = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [-0.4, 0.3, -0.2, 0.1],
                [0.5, -0.1, 0.2, -0.3],
            ],
            dtype=torch.float32,
        )

    def test_model_outputs_match_after_reload(self):
        original = TinyActorCritic()
        original_optimizer = torch.optim.Adam(
            original.parameters(),
            lr=1e-3,
        )

        with torch.no_grad():
            expected_logits, expected_values = original(self.states)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "synthetic.pt"

            save_synthetic_checkpoint(
                checkpoint_path,
                original,
                original_optimizer,
            )

            restored = TinyActorCritic()
            restored_optimizer = torch.optim.Adam(
                restored.parameters(),
                lr=1e-3,
            )

            metadata = load_synthetic_checkpoint(
                checkpoint_path,
                restored,
                restored_optimizer,
            )

            with torch.no_grad():
                actual_logits, actual_values = restored(self.states)

            self.assertTrue(torch.equal(expected_logits, actual_logits))
            self.assertTrue(torch.equal(expected_values, actual_values))
            self.assertEqual(metadata["format_version"], 1)
            self.assertEqual(
                metadata["purpose"],
                "synthetic_test_only",
            )

    def test_optimizer_state_survives_reload(self):
        model = TinyActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logits, values = model(self.states)
        loss = logits.square().mean() + values.square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "synthetic.pt"
            save_synthetic_checkpoint(
                checkpoint_path,
                model,
                optimizer,
            )

            restored = TinyActorCritic()
            restored_optimizer = torch.optim.Adam(
                restored.parameters(),
                lr=1e-3,
            )

            load_synthetic_checkpoint(
                checkpoint_path,
                restored,
                restored_optimizer,
            )

            self.assertEqual(
                len(optimizer.state),
                len(restored_optimizer.state),
            )

            for parameter in restored.parameters():
                state = restored_optimizer.state.get(parameter, {})
                if "step" in state:
                    self.assertTrue(
                        torch.isfinite(
                            torch.as_tensor(state["step"], dtype=torch.float32)
                        )
                        .all()
                        .item()
                    )

    def test_missing_checkpoint_fails_explicitly(self):
        model = TinyActorCritic()

        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "does_not_exist.pt"

            with self.assertRaises(FileNotFoundError):
                load_synthetic_checkpoint(
                    missing_path,
                    model,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
