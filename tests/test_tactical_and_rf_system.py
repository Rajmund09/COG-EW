"""
Unit & Integration Test Suite for COG-EW Tactical Mission & RF Intelligence
(tests/test_tactical_and_rf_system.py)

Covers the 4 Core Project Objectives:
1. Reading radar signals and jamming them (RF Perception + Countermeasure Allocation)
2. Auto-pilot missile defense & high-G evasion with RGPO lock-breaking
3. Tactical simulation engine state, radar beams, and telemetry
4. Strategic target terminal dive and direct kinetic destruction
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.rf_inference_service import (
    RADIOML_24_CLASSES,
    THREAT_CATEGORIES,
    RFInferenceService,
)
from simulation.tactical_engine import TacticalSimulationEngine
from sim.vehicle_dynamics import HypersonicVehicle
from sim.emitter_models import Emitter
from sim.ew_effects import EWEffectsModel
from safety.constraint_checker import SafetyConstraintFilter
from explain.explanation_generator import ExplanationGenerator


class TestRFPerceptionAndJamming(unittest.TestCase):
    """Objective 1: Reading radar signals and prescribing cognitive jamming."""

    def test_synthetic_signal_generator_shape_and_values(self):
        sig = RFInferenceService.generate_synthetic_radar_signal(modulation="QPSK", snr_db=15.0)
        self.assertEqual(sig.shape, (2, 1024))
        self.assertEqual(sig.dtype, np.float32)
        self.assertTrue(np.isfinite(sig).all())

    def test_signal_quality_metrics(self):
        sig = RFInferenceService.generate_synthetic_radar_signal(modulation="16QAM", snr_db=20.0)
        metrics = RFInferenceService.analyze_signal_quality(sig)
        self.assertIn("snr_db_estimate", metrics)
        self.assertIn("iq_imbalance", metrics)
        self.assertIn("dc_offset", metrics)
        self.assertIn("crest_factor", metrics)
        self.assertGreater(metrics["crest_factor"], 0.0)

    def test_load_dataset_sample_from_directory(self):
        sig, true_class, snr, source = RFInferenceService.load_dataset_sample(
            index=2500,
            dataset_dir=r"D:\Datasets",
        )
        self.assertEqual(sig.shape, (2, 1024))
        self.assertIsInstance(true_class, str)
        self.assertGreater(len(true_class), 0)
        self.assertIsInstance(snr, float)
        self.assertIsInstance(source, str)
        self.assertGreater(len(source), 0)

    def test_spectral_features(self):
        sig = RFInferenceService.generate_synthetic_radar_signal(modulation="BPSK", snr_db=10.0)
        spectral = RFInferenceService.compute_spectral_features(sig)
        self.assertIn("frequencies", spectral)
        self.assertIn("psd_db", spectral)
        self.assertIn("bandwidth_3db", spectral)
        self.assertGreater(len(spectral["frequencies"]), 0)
        self.assertGreater(len(spectral["psd_db"]), 0)

    def test_threat_category_mappings(self):
        # Verify critical threat classes map to appropriate countermeasures
        self.assertIn("16QAM", THREAT_CATEGORIES)
        self.assertEqual(THREAT_CATEGORIES["16QAM"]["threat_level"], "CRITICAL")
        self.assertIn("RGPO", THREAT_CATEGORIES["16QAM"]["jamming_mode"])

        self.assertIn("QPSK", THREAT_CATEGORIES)
        self.assertEqual(THREAT_CATEGORIES["QPSK"]["threat_level"], "HIGH")
        self.assertIn("Noise", THREAT_CATEGORIES["QPSK"]["jamming_mode"])


class TestAutopilotAndMissileEvasion(unittest.TestCase):
    """Objective 2: Auto-pilot system to evade incoming missiles."""

    def setUp(self):
        self.engine = TacticalSimulationEngine(scenario_id=3)

    def test_initial_tactical_state(self):
        state = self.engine.get_state()
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["scenario_id"], 3)
        self.assertTrue(state["vehicle"]["autopilot"])
        self.assertFalse(state["vehicle"]["evading"])
        self.assertFalse(state["target"]["destroyed"])

    def test_autopilot_guidance_and_speed(self):
        self.engine.start()
        # Advance simulation
        for _ in range(15):
            self.engine.step(dt=0.1)
        state = self.engine.get_state()
        self.assertIn(state["status"], ("running", "evading", "terminal_dive"))
        self.assertGreater(state["vehicle"]["x"], 70.0)
        self.assertAlmostEqual(state["vehicle"]["speed_mach"], 5.2, places=1)

    def test_rgpo_jamming_breaks_missile_lock(self):
        self.engine.start()
        # Set cognitive auto EW mode
        self.engine.set_ew_mode("cognitive_auto")

        # Manually launch an incoming SAM to test lock-break logic
        from simulation.tactical_engine import Missile
        test_sam = Missile(
            id="SAM-TEST",
            x=self.engine.vehicle_x + 100.0,
            y=self.engine.vehicle_y + 40.0,
            heading_rad=math.atan2(self.engine.vehicle_y - (self.engine.vehicle_y + 40.0), -100.0),
            speed=80.0,
        )
        self.engine.active_missiles.append(test_sam)

        # Trigger autopilot guidance step
        self.engine.step(dt=0.1)

        # The missile lock should be broken by RGPO spoofing
        self.assertTrue(self.engine.evasion_active)
        self.assertFalse(test_sam.target_locked)
        self.assertGreater(self.engine.missiles_evaded_count, 0)
        self.assertGreater(self.engine.g_load, 5.0)  # High-G evasive turn initiated


class TestGamingSimulationAndJudgesConsole(unittest.TestCase):
    """Objective 3: Tactical gaming simulation & debrief generation."""

    def setUp(self):
        self.engine = TacticalSimulationEngine(scenario_id=3)

    def test_scenario_switching(self):
        s1 = self.engine.set_scenario(1)
        self.assertEqual(len(s1["radars"]), 1)

        s2 = self.engine.set_scenario(2)
        self.assertEqual(len(s2["radars"]), 2)

        s3 = self.engine.set_scenario(3)
        self.assertEqual(len(s3["radars"]), 3)

    def test_radar_beam_rotation_and_detection(self):
        self.engine.start()
        radar_0 = self.engine.radar_sites[0]
        initial_angle = radar_0.beam_angle_rad
        self.engine.step(dt=0.5)
        self.assertNotEqual(radar_0.beam_angle_rad, initial_angle)

    def test_event_logging(self):
        self.engine.start()
        self.engine.set_ew_mode("noise_jamming")
        state = self.engine.get_state()
        self.assertGreater(len(state["events"]), 0)
        self.assertTrue(any("EW_MODE_CHANGE" in ev.get("type", "") for ev in state["events"]))


class TestStrategicTargetDestruction(unittest.TestCase):
    """Objective 4: Hitting and eliminating the strategic command bunker."""

    def setUp(self):
        self.engine = TacticalSimulationEngine(scenario_id=3)

    def test_direct_target_impact_and_mission_completion(self):
        self.engine.start()
        # Teleport craft near target to test terminal impact
        self.engine.vehicle_x = self.engine.target_x - 10.0
        self.engine.vehicle_y = self.engine.target_y
        self.engine.step(dt=0.1)

        state = self.engine.get_state()
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["target"]["destroyed"])
        self.assertTrue(state["scorecard"]["mission_success"])
        self.assertAlmostEqual(state["scorecard"]["survivability_index"], 0.985, places=2)


class TestAerodynamicsAndSafetyFilters(unittest.TestCase):
    """Aerodynamics, physical constraints, and safety projection."""

    def test_hypersonic_vehicle_kinematics(self):
        veh = HypersonicVehicle(mass=1200.0, area=0.6, mach_init=5.5)
        init_state = veh.reset()
        self.assertAlmostEqual(init_state[2], 5.5 * 301.0, places=0)
        self.assertEqual(init_state[4], 1.0)

        # Step dynamics with positive pitch
        next_state = veh.step(np.array([np.radians(4.0), 0.0]), dt=0.1)
        self.assertTrue(np.isfinite(next_state[2]))
        self.assertGreater(next_state[5], 0.0)

    def test_safety_constraint_filter(self):
        filter_sys = SafetyConstraintFilter(max_g_load=9.0, min_altitude=15000.0)

        # Safe state: should preserve action
        safe_action = np.array([0.02, 0.0])
        res = filter_sys.filter_action(safe_action, {"g_load": 1.2, "position": [0.0, 25000.0]})
        self.assertAlmostEqual(res[0], 0.02)

        # Danger state: altitude floor violation with downward pitch
        down_action = np.array([-0.05, 0.0])
        res = filter_sys.filter_action(down_action, {"g_load": 1.2, "position": [0.0, 14000.0]})
        self.assertEqual(res[0], 0.0)  # Enforced floor protection

    def test_explanation_generator(self):
        expl = ExplanationGenerator()
        action = np.array([0.03, 0.1, 2, 0.85])  # RGPO spoofing at 85% power
        sit_vec = np.zeros(16)
        att_weights = np.array([0.1, 0.8, 0.1])
        res = expl.generate_explanation(action, sit_vec, att_weights)
        self.assertIn("Range Gate Pull-Off Spoofing", res["chosen_ew_mode"])
        self.assertIn("Tactical Action", res["rationale"])
        self.assertGreaterEqual(res["confidence_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
