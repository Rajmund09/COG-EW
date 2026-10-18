"""Run a short, non-interactive COG-EW simulation demonstration."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.demo_engine import SimulationEngine


def main() -> int:
    engine = SimulationEngine()
    engine.start()

    print("COG-EW SYNTHETIC MISSION DEMO")
    print("=" * 45)

    max_steps = 2000

    for step_number in range(max_steps):
        # A display-only synthetic interference event during the demo.
        level = 0.25 if 30 <= step_number < 60 else 0.0

        state = engine.step(
            dt=0.1,
            interference_level=level,
        )

        if step_number % 50 == 0:
            print(
                f"Step {step_number:04d} | "
                f"Status: {state['status']:<9} | "
                f"Position: "
                f"({state['vehicle']['x']:.1f}, "
                f"{state['vehicle']['y']:.1f}) | "
                f"Progress: "
                f"{state['mission']['progress_percent']:.1f}% | "
                f"RF: {state['rf_monitor']['signal_quality']}"
            )

        if state["status"] in {"completed", "collision"}:
            break

    print("\nFINAL STATE")
    print(json.dumps(engine.get_state(), indent=2))

    if engine.status == "completed":
        print("\n[PASS] Virtual mission completed.")
        return 0

    print(f"\n[STOP] Mission ended with status: {engine.status}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
