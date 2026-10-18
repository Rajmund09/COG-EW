"""
COG-EW Phase 2C: PPO clipping and gradient tests.

Tests:
1. PPO probability-ratio clipping.
2. Clipped surrogate objective for positive and negative advantages.
3. Finite, nonzero gradients.
4. Actual parameter updates after an optimizer step.
5. Finite losses before and after the update.

This test is isolated from the operational COG-EW simulation.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


CLIP_EPS = 0.2


def ppo_clipped_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = CLIP_EPS,
):
    """Return PPO surrogate objective and probability ratios."""
    ratios = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratios * advantages
    clipped = (
        torch.clamp(
            ratios,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * advantages
    )

    objective = torch.minimum(unclipped, clipped)
    return objective, ratios


class TinyActorCritic(nn.Module):
    """Small network for testing gradient flow and optimization."""

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


class TestPPOClipping(unittest.TestCase):

    def test_probability_ratio_is_clipped(self):
        # exp(log(1.5)) = 1.5, which exceeds the upper bound 1.2.
        new_log_probs = torch.tensor([torch.log(torch.tensor(1.5))])
        old_log_probs = torch.tensor([0.0])
        advantages = torch.tensor([1.0])

        objective, ratios = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            advantages,
        )

        self.assertAlmostEqual(ratios.item(), 1.5, places=5)
        self.assertAlmostEqual(objective.item(), 1.2, places=5)

    def test_positive_advantage_uses_upper_clip(self):
        # Positive advantage: ratios above 1 + epsilon are capped.
        new_log_probs = torch.log(torch.tensor([1.8]))
        old_log_probs = torch.zeros(1)
        advantages = torch.tensor([2.0])

        objective, _ = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            advantages,
        )

        # min(1.8 * 2, 1.2 * 2) = 2.4
        self.assertAlmostEqual(objective.item(), 2.4, places=5)

    def test_negative_advantage_uses_lower_clip(self):
        # Negative advantage: a ratio below 1 - epsilon selects
        # the clipped surrogate objective.
        new_log_probs = torch.log(torch.tensor([0.5]))
        old_log_probs = torch.zeros(1)
        advantages = torch.tensor([-1.0])

        objective, _ = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            advantages,
        )

        # min(0.5 * -1, 0.8 * -1) = -0.8
        self.assertAlmostEqual(objective.item(), -0.8, places=5)

    def test_ratio_inside_clip_range_is_unchanged(self):
        new_log_probs = torch.log(torch.tensor([1.1]))
        old_log_probs = torch.zeros(1)
        advantages = torch.tensor([2.0])

        objective, ratios = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            advantages,
        )

        self.assertAlmostEqual(ratios.item(), 1.1, places=5)
        self.assertAlmostEqual(objective.item(), 2.2, places=5)

    def test_gradients_are_finite_and_nonzero(self):
        torch.manual_seed(123)

        model = TinyActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        states = torch.tensor(
            [
                [0.8, 0.1, -0.2, 0.3],
                [-0.7, 0.4, 0.1, -0.1],
                [0.5, -0.3, 0.2, 0.6],
                [-0.4, 0.2, -0.5, 0.1],
            ],
            dtype=torch.float32,
        )
        actions = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        advantages = torch.tensor([1.0, -1.0, 0.8, -0.7])
        returns = torch.tensor([0.8, -0.6, 0.5, -0.4])

        # Create old log probabilities from a fixed reference policy.
        with torch.no_grad():
            reference_logits, _ = model(states)
            old_log_probs = Categorical(logits=reference_logits).log_prob(actions)

        # A small perturbation makes the current policy differ from
        # the reference policy while keeping the test reproducible.
        with torch.no_grad():
            model.actor.bias.add_(torch.tensor([0.05, -0.05]))

        logits, values = model(states)
        distribution = Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        surrogate, _ = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            advantages,
        )

        policy_loss = -surrogate.mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.assertTrue(torch.isfinite(loss).item())

        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gradient_total = 0.0

        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                gradient_total += parameter.grad.abs().sum().item()

        self.assertGreater(gradient_total, 0.0)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.5,
        )
        optimizer.step()

        parameter_change = sum(
            (parameter.detach() - before[name]).abs().sum().item()
            for name, parameter in model.named_parameters()
        )

        self.assertGreater(parameter_change, 0.0)

        # Recompute loss after the update and check numerical stability.
        updated_logits, updated_values = model(states)
        updated_distribution = Categorical(logits=updated_logits)
        updated_log_probs = updated_distribution.log_prob(actions)

        updated_surrogate, _ = ppo_clipped_objective(
            updated_log_probs,
            old_log_probs,
            advantages,
        )

        updated_policy_loss = -updated_surrogate.mean()
        updated_value_loss = F.mse_loss(updated_values, returns)
        updated_entropy = updated_distribution.entropy().mean()

        updated_loss = (
            updated_policy_loss + 0.5 * updated_value_loss - 0.01 * updated_entropy
        )

        self.assertTrue(torch.isfinite(updated_loss).item())

    def test_invalid_probability_ratio_is_detected(self):
        # Extremely large log-probability differences can overflow.
        new_log_probs = torch.tensor([1000.0])
        old_log_probs = torch.tensor([0.0])

        # Negative advantage: unclipped branch is -inf, so objective is non-finite.
        neg_advantages = torch.tensor([-1.0])
        neg_objective, neg_ratios = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            neg_advantages,
        )
        self.assertFalse(torch.isfinite(neg_ratios).all().item())
        self.assertFalse(torch.isfinite(neg_objective).all().item())

        # Positive advantage: ratio overflows to inf, but clipping bounds the objective.
        # Direct ratio finite checks are needed because clipping masks overflow in objective.
        pos_advantages = torch.tensor([1.0])
        pos_objective, pos_ratios = ppo_clipped_objective(
            new_log_probs,
            old_log_probs,
            pos_advantages,
        )
        self.assertFalse(torch.isfinite(pos_ratios).all().item())
        self.assertTrue(torch.isfinite(pos_objective).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
