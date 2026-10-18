import sys
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from rl.rl_agent import PPOAgent


class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.agent = PPOAgent(state_dim=16, device="cpu")

        # A fixed, valid observation for repeatable inference tests.
        self.state = np.linspace(-0.5, 0.5, 16, dtype=np.float32)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_policy_initializes(self):
        self.assertIsNotNone(self.agent.policy)
        self.assertIsNotNone(self.agent.optimizer)

    def test_inference_returns_finite_outputs(self):
        action, log_prob, value = self.agent.select_action(
            self.state, deterministic=True
        )

        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.isfinite(np.asarray(log_prob)).all())
        self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_rejects_wrong_state_shape(self):
        invalid_state = np.zeros(15, dtype=np.float32)

        with self.assertRaises(ValueError):
            self.agent.select_action(invalid_state)

    def test_rejects_non_finite_state(self):
        invalid_state = self.state.copy()
        invalid_state[3] = np.nan

        with self.assertRaises(ValueError):
            self.agent.select_action(invalid_state)

    def test_missing_checkpoint_is_rejected(self):
        missing_path = self.temp_path / "missing_checkpoint.pt"

        with self.assertRaises(FileNotFoundError):
            self.agent.load_checkpoint(str(missing_path))

    def test_save_and_reload_preserves_inference(self):
        checkpoint_path = self.temp_path / "test_policy.pt"

        expected = self.agent.select_action(self.state, deterministic=True)
        self.agent.save_checkpoint(str(checkpoint_path))

        self.assertTrue(checkpoint_path.is_file())

        restored_agent = PPOAgent(state_dim=16, device="cpu")
        restored_agent.load_checkpoint(str(checkpoint_path))

        actual = restored_agent.select_action(self.state, deterministic=True)

        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(
                np.asarray(expected_array),
                np.asarray(actual_array),
                rtol=1e-6,
                atol=1e-7,
            )

    def test_malformed_checkpoint_is_rejected(self):
        checkpoint_path = self.temp_path / "malformed.pt"

        # Safe tensor loading should succeed, but the content is invalid.
        torch.save({"not_a_tensor": "invalid"}, checkpoint_path)

        with self.assertRaises(ValueError):
            self.agent.load_checkpoint(str(checkpoint_path))

    def test_unexpected_checkpoint_keys_are_rejected(self):
        checkpoint_path = self.temp_path / "wrong_keys.pt"
        torch.save(
            {"unexpected.policy.weight": torch.zeros(1)},
            checkpoint_path,
        )

        with self.assertRaises(ValueError):
            self.agent.load_checkpoint(str(checkpoint_path))

    def test_incompatible_tensor_shapes_are_rejected(self):
        checkpoint_path = self.temp_path / "wrong_shapes.pt"
        state_dict = self.agent.policy.state_dict()

        first_key = next(iter(state_dict))
        original_tensor = state_dict[first_key]

        # Append a dimension to guarantee a shape mismatch.
        state_dict[first_key] = torch.zeros(
            tuple(original_tensor.shape) + (1,),
            dtype=original_tensor.dtype,
        )

        torch.save(state_dict, checkpoint_path)

        with self.assertRaises(ValueError):
            self.agent.load_checkpoint(str(checkpoint_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
