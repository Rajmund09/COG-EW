"""Contract tests for the COG-EW defensive decision engine.

Run from the project root:
    python -m unittest discover -s tests -p "test_*.py" -v

These tests validate report interpretation and safety boundaries.
They do not perform RF transmission or control external equipment.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cogew_decision_engine import generate_decision


class TestDecisionEngine(unittest.TestCase):

    def make_report(
        self,
        modulation="16QAM",
        confidence=0.90,
        signal_quality="GOOD",
        reliability="HIGH",
    ):
        """Build a minimal signal-quality report using the existing schema."""
        return {
            "classification": {
                "predicted_class": modulation,
                "confidence": confidence,
            },
            "quality_assessment": {
                "signal_quality": signal_quality,
                "classification_reliability": reliability,
            },
        }

    def test_high_confidence_good_signal(self):
        result = generate_decision(self.make_report())

        self.assertEqual(
            result["decision"]["confidence_state"],
            "CLASSIFICATION_ACCEPTED",
        )
        self.assertEqual(
            result["decision"]["simulation_state"],
            "HIGH_CONFIDENCE_RF_TRACK",
        )
        self.assertEqual(
            result["classification"]["modulation_family"],
            "QAM",
        )

    def test_medium_reliability_requires_review(self):
        result = generate_decision(
            self.make_report(
                confidence=0.65,
                signal_quality="MODERATE",
                reliability="MEDIUM",
            )
        )

        self.assertEqual(
            result["decision"]["confidence_state"],
            "CLASSIFICATION_REQUIRES_REVIEW",
        )
        self.assertEqual(
            result["decision"]["simulation_state"],
            "MEDIUM_CONFIDENCE_RF_TRACK",
        )
        self.assertIn(
            "additional RF evidence",
            result["decision"]["operator_action"],
        )

    def test_low_confidence_is_unreliable(self):
        result = generate_decision(
            self.make_report(
                confidence=0.30,
                signal_quality="POOR",
                reliability="LOW",
            )
        )

        self.assertEqual(
            result["decision"]["confidence_state"],
            "CLASSIFICATION_UNCERTAIN",
        )
        self.assertEqual(
            result["decision"]["simulation_state"],
            "UNRELIABLE_RF_TRACK",
        )
        self.assertIn(
            "Do not rely",
            result["decision"]["operator_action"],
        )

    def test_unknown_modulation_has_unknown_family(self):
        result = generate_decision(self.make_report(modulation="UNKNOWN_MODULATION"))

        self.assertEqual(
            result["classification"]["modulation_family"],
            "UNKNOWN",
        )

    def test_safety_boundaries_remain_disabled(self):
        result = generate_decision(self.make_report())
        boundary = result["safety_boundary"]

        self.assertIs(boundary["real_world_transmission"], False)
        self.assertIs(boundary["real_world_jamming"], False)
        self.assertIs(boundary["weapon_control"], False)
        self.assertEqual(
            boundary["purpose"],
            "Defensive RF monitoring and simulation",
        )

    def test_confidence_percent_is_consistent(self):
        result = generate_decision(self.make_report(confidence=0.73))

        self.assertAlmostEqual(
            result["classification"]["confidence_percent"],
            73.0,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
