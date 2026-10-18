"""Regression tests for the COG-EW training entry-point safeguards."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
PYTHON = sys.executable


class TestTrainingGuard(unittest.TestCase):
    def run_train(self, *args):
        return subprocess.run(
            [PYTHON, str(TRAIN_SCRIPT), *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_normal_training_request_is_blocked(self):
        result = self.run_train()

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "PPO policy optimization has not been implemented",
            result.stdout,
        )
        self.assertIn("No checkpoint was saved.", result.stdout)
        self.assertNotIn("Diagnostic round", result.stdout)

    def test_zero_epochs_are_rejected(self):
        result = self.run_train(
            "--rollout-only",
            "--epochs",
            "0",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--epochs", result.stderr)

    def test_missing_configuration_is_rejected(self):
        result = self.run_train(
            "--rollout-only",
            "--config",
            str(PROJECT_ROOT / "configs" / "missing_test_config.yaml"),
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("[ERROR] Configuration not found:", result.stdout)

    def test_state_dimension_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_dimensions.yaml"
            config_path.write_text(
                "network:\n" "  state_dim: 999\n" "  total_action_dim: 4\n",
                encoding="utf-8",
            )

            result = self.run_train(
                "--rollout-only",
                "--config",
                str(config_path),
                "--epochs",
                "1",
                "--episodes-per-epoch",
                "1",
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("State dimension mismatch", result.stdout)
        self.assertNotIn("Diagnostic round", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
