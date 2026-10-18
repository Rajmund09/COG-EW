import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from rl.policy_network import MultiHeadPolicyNet
from sim.environment import HypersonicEWEnv

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"


class TestConfigContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            cls.config = yaml.safe_load(file)

        cls.network_config = cls.config["network"]
        cls.ppo_config = cls.config["ppo"]
        cls.training_config = cls.config["training"]
        cls.env = HypersonicEWEnv()

    def test_config_dimensions_match_environment(self):
        self.assertEqual(
            self.network_config["state_dim"],
            self.env.observation_space.shape[0],
        )
        self.assertEqual(
            self.network_config["total_action_dim"],
            self.env.action_space.shape[0],
        )
        self.assertEqual(
            self.network_config["guidance_action_dim"]
            + self.network_config["ew_action_dim"],
            self.network_config["total_action_dim"],
        )

    def test_config_hidden_dimensions_match_policy(self):
        """Catch drift between YAML and the policy's default architecture."""
        model = MultiHeadPolicyNet(state_dim=self.network_config["state_dim"])

        actual_hidden_dims = [
            layer.out_features
            for layer in model.backbone
            if isinstance(layer, torch.nn.Linear)
        ]

        self.assertEqual(
            actual_hidden_dims,
            self.network_config["hidden_dims"],
            "Policy architecture differs from train_config.yaml. "
            "The current agent does not pass hidden_dims to the model.",
        )

    def test_ppo_hyperparameters_are_valid(self):
        ppo = self.ppo_config

        for key in (
            "learning_rate_actor",
            "learning_rate_critic",
            "clip_eps",
            "max_grad_norm",
        ):
            self.assertGreater(ppo[key], 0, f"{key} must be positive")

        for key in ("gamma", "gae_lambda"):
            self.assertGreaterEqual(ppo[key], 0, f"{key} must be >= 0")
            self.assertLessEqual(ppo[key], 1, f"{key} must be <= 1")

        self.assertGreaterEqual(ppo["entropy_coef"], 0)
        self.assertGreaterEqual(ppo["value_loss_coef"], 0)
        self.assertGreaterEqual(ppo["ppo_epochs"], 1)
        self.assertGreaterEqual(ppo["batch_size"], 1)

    def test_training_settings_are_positive(self):
        training = self.training_config

        for key in (
            "total_timesteps",
            "checkpoint_interval",
            "curriculum_stages",
        ):
            self.assertGreater(training[key], 0, f"{key} must be positive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
