import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from fastapi import HTTPException

from scripts import dashboard_api as api


class FixedLogitModel(torch.nn.Module):
    """Deterministic model for API tests; no real checkpoint is loaded."""

    def __init__(self, logits=None):
        super().__init__()
        self.logits = (
            torch.tensor([logits], dtype=torch.float32)
            if logits is not None
            else torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
        )

    def forward(self, inputs):
        return self.logits.to(inputs.device)


class TestDashboardPrediction(unittest.TestCase):

    def setUp(self):
        patcher = patch(
            "scripts.dashboard_api.load_model",
            return_value=FixedLogitModel(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_valid_prediction_has_five_probabilities(self):
        result = api.predict_signal(np.zeros((128, 2), dtype=np.float32))

        self.assertEqual(len(result["probabilities"]), 5)
        self.assertIn(result["predicted_class"], api.CLASS_NAMES)
        self.assertAlmostEqual(
            sum(result["probabilities"].values()),
            1.0,
            places=5,
        )
        self.assertTrue(all(np.isfinite(v) for v in result["probabilities"].values()))
        self.assertGreaterEqual(result["confidence_prob"], 0.0)
        self.assertLessEqual(result["confidence_prob"], 1.0)

    def test_wrong_signal_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected signal shape"):
            api.predict_signal(np.zeros((64, 2), dtype=np.float32))

    def test_non_finite_signal_is_rejected(self):
        signal = np.zeros((128, 2), dtype=np.float32)
        signal[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            api.predict_signal(signal)

    def test_infinite_signal_is_rejected(self):
        signal = np.zeros((128, 2), dtype=np.float32)
        signal[0, 1] = np.inf

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            api.predict_signal(signal)

    def test_unexpected_model_output_shape_is_rejected(self):
        with patch(
            "scripts.dashboard_api.load_model",
            return_value=FixedLogitModel([1.0, 2.0]),
        ):
            with self.assertRaisesRegex(RuntimeError, "output shape"):
                api.predict_signal(np.zeros((128, 2), dtype=np.float32))

    def test_non_finite_model_output_is_rejected(self):
        bad_model = FixedLogitModel([float("nan"), 1.0, 0.0, -1.0, -2.0])

        with patch("scripts.dashboard_api.load_model", return_value=bad_model):
            with self.assertRaisesRegex(RuntimeError, "non-finite logits"):
                api.predict_signal(np.zeros((128, 2), dtype=np.float32))


class TestDashboardTestSampleEndpoint(unittest.TestCase):

    def make_sample(self, source_id):
        return {
            "x": np.zeros((128, 2), dtype=np.float32),
            "y": 1,
            "snr": 10,
            "original_index": 1234,
            "source": source_id,
        }

    def make_prediction(self):
        return {
            "predicted_class": "BPSK",
            "predicted_index": 1,
            "confidence_prob": 0.8,
            "confidence_percent": 80.0,
            "inference_ms": 1.2,
            "probabilities": {
                "8PSK": 0.05,
                "BPSK": 0.80,
                "QPSK": 0.05,
                "16QAM": 0.05,
                "64QAM": 0.05,
            },
            "probability_percentages": {
                "8PSK": 5.0,
                "BPSK": 80.0,
                "QPSK": 5.0,
                "16QAM": 5.0,
                "64QAM": 5.0,
            },
        }

    def test_source_zero_maps_to_radioml2016(self):
        with (
            patch(
                "scripts.dashboard_api.load_test_sample",
                return_value=self.make_sample(0),
            ),
            patch(
                "scripts.dashboard_api.predict_signal",
                return_value=self.make_prediction(),
            ),
            patch(
                "scripts.dashboard_api.analyze_signal_quality",
                return_value={"signal_quality": "MODERATE"},
            ),
        ):
            response = api.predict_test(api.SampleRequest(index=5))

        self.assertEqual(response["source"], "RadioML2016")
        self.assertEqual(response["source_name"], "RadioML2016")
        self.assertEqual(response["source_id"], 0)
        self.assertEqual(response["true_class"], "BPSK")

    def test_source_one_maps_to_radioml2018(self):
        with (
            patch(
                "scripts.dashboard_api.load_test_sample",
                return_value=self.make_sample(1),
            ),
            patch(
                "scripts.dashboard_api.predict_signal",
                return_value=self.make_prediction(),
            ),
            patch(
                "scripts.dashboard_api.analyze_signal_quality",
                return_value={"signal_quality": "MODERATE"},
            ),
        ):
            response = api.predict_test(api.SampleRequest(test_index=6))

        self.assertEqual(response["source"], "RadioML2018")
        self.assertEqual(response["source_name"], "RadioML2018")
        self.assertEqual(response["source_id"], 1)

    def test_unknown_source_is_rejected(self):
        with patch(
            "scripts.dashboard_api.load_test_sample",
            return_value=self.make_sample(9),
        ):
            with self.assertRaises(HTTPException) as context:
                api.predict_test(api.SampleRequest(index=5))

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn(
            "Unsupported test dataset source ID", str(context.exception.detail)
        )


class TestDashboardHealth(unittest.TestCase):

    def setUp(self):
        # Prevent cache state from leaking between tests.
        api.load_model.cache_clear()
        self.addCleanup(api.load_model.cache_clear)

    def test_existing_checkpoint_is_not_available_until_loaded(self):
        mock_model = MagicMock()
        mock_model.is_file.return_value = True
        mock_dataset = MagicMock()
        mock_dataset.is_file.return_value = True
        with (
            patch.object(api, "MODEL_PATH", mock_model),
            patch.object(api, "TEST_DATASET", mock_dataset),
        ):
            result = api.health()

        self.assertTrue(result["online"])
        self.assertTrue(result["model_exists"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["model_available"])
        self.assertTrue(result["test_dataset_available"])

    def test_missing_checkpoint_is_not_available(self):
        mock_model = MagicMock()
        mock_model.is_file.return_value = False
        mock_dataset = MagicMock()
        mock_dataset.is_file.return_value = True
        with (
            patch.object(api, "MODEL_PATH", mock_model),
            patch.object(api, "TEST_DATASET", mock_dataset),
        ):
            result = api.health()

        self.assertTrue(result["online"])
        self.assertFalse(result["model_exists"])
        self.assertFalse(result["model_available"])


class TestDashboardPages(unittest.TestCase):

    def test_simulation_page_is_served(self):
        response = api.dashboard_page("simulation.html")
        self.assertTrue(Path(response.path).is_file())
        self.assertEqual(Path(response.path).name, "simulation.html")

    def test_unknown_page_raises_404(self):
        with self.assertRaises(HTTPException) as context:
            api.dashboard_page("non_existent_page.html")
        self.assertEqual(context.exception.status_code, 404)


class TestDashboardSimulationEndpoints(unittest.TestCase):

    def setUp(self):
        api.simulation_engine.reset()

    def test_simulation_state_and_lifecycle(self):
        state = api.simulation_state()
        self.assertEqual(state["status"], "ready")
        self.assertIn("vehicle", state)
        self.assertIn("obstacles", state)

        started = api.simulation_start()
        self.assertEqual(started["status"], "running")

        ticked = api.simulation_tick(api.SimulationTickRequest(dt=0.1))
        self.assertGreater(ticked["telemetry"]["elapsed_seconds"], 0.0)

        paused = api.simulation_pause()
        self.assertEqual(paused["status"], "paused")

        reset = api.simulation_reset()
        self.assertEqual(reset["status"], "ready")
        self.assertEqual(reset["telemetry"]["elapsed_seconds"], 0.0)

    def test_invalid_dt_raises_400(self):
        with self.assertRaises(HTTPException) as context:
            api.simulation_tick(api.SimulationTickRequest(dt=0.5))
        self.assertEqual(context.exception.status_code, 400)

    def test_invalid_interference_raises_400(self):
        with self.assertRaises(HTTPException) as context:
            api.simulation_interference(api.SimulationInterferenceRequest(level=2.0))
        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
