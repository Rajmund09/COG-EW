"""Regression tests for COG-EW policy evaluation and benchmark metrics.

Validates that:
1. Missing or invalid checkpoints fail closed (no false successful evaluation).
2. Evaluator produces all required benchmark metrics with finite, valid values.
3. Evaluation never writes to or overwrites checkpoints.
"""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.evaluation import Evaluator
from rl.rl_agent import PPOAgent
from sim.environment import HypersonicEWEnv


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_benchmark_metrics_are_present_finite_and_valid(self):
        env = HypersonicEWEnv()
        agent = PPOAgent(state_dim=int(env.observation_space.shape[0]), device="cpu")
        evaluator = Evaluator(env=env, agent=agent)

        metrics = evaluator.run_benchmark(num_episodes=2)

        required_keys = (
            "mission_success_rate",
            "survivability_index",
            "mean_inference_latency_ms",
            "max_inference_latency_ms",
            "latency_constraint_met",
        )
        for key in required_keys:
            self.assertIn(key, metrics)

        # Numerical validity and finite checks
        self.assertTrue(math.isfinite(metrics["mission_success_rate"]))
        self.assertGreaterEqual(metrics["mission_success_rate"], 0.0)
        self.assertLessEqual(metrics["mission_success_rate"], 100.0)

        self.assertTrue(math.isfinite(metrics["survivability_index"]))
        self.assertGreaterEqual(metrics["survivability_index"], 0.0)
        self.assertLessEqual(metrics["survivability_index"], 1.0)

        self.assertTrue(math.isfinite(metrics["mean_inference_latency_ms"]))
        self.assertGreaterEqual(metrics["mean_inference_latency_ms"], 0.0)

        self.assertTrue(math.isfinite(metrics["max_inference_latency_ms"]))
        self.assertGreaterEqual(
            metrics["max_inference_latency_ms"],
            metrics["mean_inference_latency_ms"],
        )

        self.assertIsInstance(metrics["latency_constraint_met"], bool)

    def test_missing_checkpoint_fails_closed(self):
        missing_path = self.temp_path / "definitely_missing.pt"

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "evaluate.py"),
                "--model_path",
                str(missing_path),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("[ERROR] Checkpoint not found.", result.stdout)
        self.assertNotIn("Evaluation results", result.stdout)

    def test_invalid_checkpoint_fails_closed(self):
        invalid_path = self.temp_path / "invalid.pt"
        torch.save({"corrupt_key": torch.zeros(1)}, invalid_path)

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "evaluate.py"),
                "--model_path",
                str(invalid_path),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertTrue(
            "[ERROR] Checkpoint or agent initialization failed:" in result.stdout
            or "[ERROR] Checkpoint could not be loaded:" in result.stdout
        )
        self.assertNotIn("Evaluation results", result.stdout)

    def test_evaluation_does_not_modify_or_create_checkpoints(self):
        checkpoint_path = self.temp_path / "reference.pt"
        agent = PPOAgent(state_dim=16, device="cpu")
        agent.save_checkpoint(str(checkpoint_path))

        stat_before = checkpoint_path.stat()

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "evaluate.py"),
                "--model_path",
                str(checkpoint_path),
                "--episodes",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Evaluation completed.", result.stdout)

        stat_after = checkpoint_path.stat()
        self.assertEqual(stat_before.st_size, stat_after.st_size)
        self.assertEqual(stat_before.st_mtime, stat_after.st_mtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
