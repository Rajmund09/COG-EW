import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rl.evaluation import Evaluator, validate_metrics


VALID_METRICS = {
    "mission_success_rate": 75.0,
    "survivability_index": 0.8,
    "mean_inference_latency_ms": 10.0,
    "max_inference_latency_ms": 20.0,
    "latency_constraint_met": True,
}


class FakeAgent:
    def select_action(self, state, deterministic=True):
        return np.zeros(4, dtype=np.float32), 0.0, 0.0


class FakeEnv:
    def __init__(self, detection_probability=0.2):
        self.detection_probability = detection_probability

    def reset(self):
        return np.zeros(16, dtype=np.float32), {}

    def step(self, action):
        info = {"max_p_detection": self.detection_probability}
        return np.zeros(16, dtype=np.float32), 0.0, True, False, info


class TestEvaluationValidation(unittest.TestCase):
    def test_valid_metrics_are_normalized(self):
        result = validate_metrics(VALID_METRICS)
        self.assertEqual(result["mission_success_rate"], 75.0)
        self.assertIs(result["latency_constraint_met"], True)

    def test_missing_metric_is_rejected(self):
        metrics = VALID_METRICS.copy()
        del metrics["survivability_index"]

        with self.assertRaises(ValueError):
            validate_metrics(metrics)

    def test_non_finite_metric_is_rejected(self):
        metrics = VALID_METRICS.copy()
        metrics["mean_inference_latency_ms"] = float("nan")

        with self.assertRaises(ValueError):
            validate_metrics(metrics)

    def test_out_of_range_success_rate_is_rejected(self):
        metrics = VALID_METRICS.copy()
        metrics["mission_success_rate"] = 101.0

        with self.assertRaises(ValueError):
            validate_metrics(metrics)

    def test_inconsistent_latency_flag_is_rejected(self):
        metrics = VALID_METRICS.copy()
        metrics["max_inference_latency_ms"] = 60.0
        metrics["latency_constraint_met"] = True

        with self.assertRaises(ValueError):
            validate_metrics(metrics)

    def test_zero_episodes_is_rejected(self):
        evaluator = Evaluator(FakeEnv(), FakeAgent())

        with self.assertRaises(ValueError):
            evaluator.run_benchmark(num_episodes=0)

    def test_invalid_detection_probability_is_rejected(self):
        evaluator = Evaluator(
            FakeEnv(detection_probability=1.5),
            FakeAgent(),
        )

        with self.assertRaises(ValueError):
            evaluator.run_benchmark(num_episodes=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
