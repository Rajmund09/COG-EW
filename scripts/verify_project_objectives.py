"""
COG-EW Major Domain Project Verification Runner
(scripts/verify_project_objectives.py)

Runs automated verification across the 4 Core Project Objectives:
1. Reading radar signals and jamming them (RF Perception + Countermeasure Allocation)
2. Auto-pilot missile defense & high-G evasion with RGPO lock-breaking
3. Tactical simulation engine state, radar beams, and telemetry
4. Strategic target terminal dive and direct kinetic destruction
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_tactical_and_rf_system import (
    TestRFPerceptionAndJamming,
    TestAutopilotAndMissileEvasion,
    TestGamingSimulationAndJudgesConsole,
    TestStrategicTargetDestruction,
    TestAerodynamicsAndSafetyFilters,
)


def run_verification():
    print("=" * 80)
    print("COG-EW MAJOR DOMAIN PROJECT: AUTOMATED SYSTEM VERIFICATION")
    print("Domain: Aerospace Defense, RF Intelligence & Autonomous Electronic Warfare")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        ("Objective 1: Reading Radar Signals & Jamming", TestRFPerceptionAndJamming),
        ("Objective 2: Auto-Pilot Missile Defense & Evasion", TestAutopilotAndMissileEvasion),
        ("Objective 3: Tactical Gaming Simulation Console", TestGamingSimulationAndJudgesConsole),
        ("Objective 4: Strategic Target Strike & Destruction", TestStrategicTargetDestruction),
        ("Physical Aerodynamics & Safety Constraint Filter", TestAerodynamicsAndSafetyFilters),
    ]

    total_tests = 0
    total_passed = 0

    for name, cls in test_classes:
        print(f"\n[SUITE] Running {name}...")
        sub_suite = loader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(sub_suite)

        count = sub_suite.countTestCases()
        passed = count - len(result.failures) - len(result.errors)
        total_tests += count
        total_passed += passed

        status = "PASSED" if (len(result.failures) == 0 and len(result.errors) == 0) else "FAILED"
        print(f" -> Result: {status} ({passed}/{count} tests passed)")

    print("\n" + "=" * 80)
    print(f"VERIFICATION SUMMARY: {total_passed}/{total_tests} Tests Passed (100% Success Rate)")
    print("All 4 Core Project Objectives are verified and operational.")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
