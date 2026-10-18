"""
COG-EW Phase 2B: PPO mathematics unit tests.

Tests the actual calculate_gae() function from the synthetic PPO
smoke test, using hand-computed expected values.

Run:
    python -m unittest tests.test_ppo_math -v
"""

import sys
from pathlib import Path
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.synthetic_ppo_smoke_test import calculate_gae, GAMMA, GAE_LAMBDA


class FixedValueModel:
    """Returns a known value estimate from the first state feature."""

    def __call__(self, states):
        values = states[:, 0]
        logits = torch.zeros(
            (states.shape[0], 2),
            dtype=torch.float32,
            device=states.device,
        )
        return logits, values


def make_rollout(rewards, dones, values, next_values):
    """Build a minimal rollout compatible with calculate_gae()."""
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
        "next_states": torch.tensor(
            [[value, 0.0, 0.0, 0.0] for value in next_values],
            dtype=torch.float32,
        ),
    }


class TestPPOMathematics(unittest.TestCase):

    def setUp(self):
        self.model = FixedValueModel()

    def test_gae_matches_hand_calculated_values(self):
        """
        Check a three-transition rollout with a terminal boundary
        between transitions 1 and 2.

        Expected values use:
        delta_t = reward_t + gamma * next_value_t * (1-done_t)
                  - value_t
        """
        rollout = make_rollout(
            rewards=[1.0, 2.0, 3.0],
            dones=[0.0, 1.0, 0.0],
            values=[0.5, 0.25, 0.75],
            next_values=[0.25, 0.0, 0.1],
        )

        advantages, returns = calculate_gae(
            rollout,
            self.model,
        )

        expected_advantages = torch.tensor(
            [
                0.7475 + GAMMA * GAE_LAMBDA * 1.75,
                1.75,
                3.0 + GAMMA * 0.1 - 0.75,
            ],
            dtype=torch.float32,
        )

        expected_returns = expected_advantages + rollout["values"]

        torch.testing.assert_close(
            advantages,
            expected_advantages,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            returns,
            expected_returns,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_terminal_state_blocks_future_advantage(self):
        """
        A terminal transition must not inherit advantages from the
        following transition, even if its reward is very large.
        """
        rollout = make_rollout(
            rewards=[1.0, 100.0],
            dones=[1.0, 0.0],
            values=[0.0, 0.0],
            next_values=[0.0, 0.0],
        )

        advantages, _ = calculate_gae(
            rollout,
            self.model,
        )

        self.assertAlmostEqual(
            advantages[0].item(),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            advantages[1].item(),
            100.0,
            places=5,
        )

    def test_gae_outputs_are_finite(self):
        rollout = make_rollout(
            rewards=[0.5, -0.5, 1.0, 0.25],
            dones=[0.0, 0.0, 1.0, 0.0],
            values=[0.1, 0.2, -0.1, 0.3],
            next_values=[0.2, -0.1, 0.0, 0.1],
        )

        advantages, returns = calculate_gae(
            rollout,
            self.model,
        )

        self.assertTrue(torch.isfinite(advantages).all().item())
        self.assertTrue(torch.isfinite(returns).all().item())

    def test_returns_equal_advantages_plus_values(self):
        rollout = make_rollout(
            rewards=[1.0, -1.0, 0.5],
            dones=[0.0, 1.0, 0.0],
            values=[0.25, 0.5, -0.25],
            next_values=[0.5, 0.0, 0.1],
        )

        advantages, returns = calculate_gae(
            rollout,
            self.model,
        )

        torch.testing.assert_close(
            returns,
            advantages + rollout["values"],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
