"""
COG-EW DECISION ENGINE

Converts RF classification + signal quality information into
a defensive simulation decision.

This module is intentionally limited to:
- RF monitoring
- classification assessment
- confidence assessment
- simulation state
- operator review
- defensive decision support

It does NOT:
- transmit RF energy
- generate jamming waveforms
- provide real-world jamming parameters
- control weapons
- provide missile defeat instructions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "reports" / "rf_signal_quality_analysis.json"

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "cogew_decision.json"


# ============================================================
# THREAT / SIGNAL CATEGORIES
# ============================================================

MODULATION_FAMILIES = {
    "BPSK": "LOW_ORDER_PSK",
    "QPSK": "LOW_ORDER_PSK",
    "8PSK": "HIGHER_ORDER_PSK",
    "16QAM": "QAM",
    "64QAM": "HIGH_ORDER_QAM",
}


# ============================================================
# DECISION ENGINE
# ============================================================


def generate_decision(data):

    classification = data["classification"]

    quality = data["quality_assessment"]

    predicted_class = classification["predicted_class"]

    confidence = float(classification["confidence"])

    signal_quality = quality["signal_quality"]

    reliability = quality["classification_reliability"]

    family = MODULATION_FAMILIES.get(predicted_class, "UNKNOWN")

    # --------------------------------------------------------
    # Confidence gate
    # --------------------------------------------------------

    if reliability == "HIGH":

        confidence_state = "CLASSIFICATION_ACCEPTED"

    elif reliability == "MEDIUM":

        confidence_state = "CLASSIFICATION_REQUIRES_REVIEW"

    else:

        confidence_state = "CLASSIFICATION_UNCERTAIN"

    # --------------------------------------------------------
    # Simulation state
    # --------------------------------------------------------

    if (
        signal_quality
        in (
            "EXCELLENT",
            "GOOD",
        )
        and confidence >= 0.80
    ):

        simulation_state = "HIGH_CONFIDENCE_RF_TRACK"

    elif confidence >= 0.60:

        simulation_state = "MEDIUM_CONFIDENCE_RF_TRACK"

    elif confidence >= 0.40:

        simulation_state = "LOW_CONFIDENCE_RF_TRACK"

    else:

        simulation_state = "UNRELIABLE_RF_TRACK"

    # --------------------------------------------------------
    # Operator recommendation
    # --------------------------------------------------------

    if reliability == "HIGH":

        operator_action = "Continue defensive RF monitoring " "and simulation tracking."

    elif reliability == "MEDIUM":

        operator_action = (
            "Continue monitoring and request "
            "additional RF evidence before "
            "high-impact automated decisions."
        )

    else:

        operator_action = (
            "Do not rely on this classification "
            "for automated high-impact decisions. "
            "Acquire additional observations."
        )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = {
        "project": "COG-EW",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_engine": "RF Defensive Decision Support",
        "classification": {
            "modulation": predicted_class,
            "modulation_family": family,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
        },
        "signal_assessment": {
            "signal_quality": signal_quality,
            "classification_reliability": reliability,
        },
        "decision": {
            "confidence_state": confidence_state,
            "simulation_state": simulation_state,
            "operator_action": operator_action,
        },
        "safety_boundary": {
            "real_world_transmission": False,
            "real_world_jamming": False,
            "weapon_control": False,
            "purpose": "Defensive RF monitoring and simulation",
        },
    }

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_path = Path(args.output)

    print("=" * 70)
    print("COG-EW RF DECISION ENGINE")
    print("=" * 70)

    print()
    print("Input:")
    print(input_path)

    if not input_path.exists():

        raise FileNotFoundError(f"Input analysis file not found:\n" f"{input_path}")

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    print()
    print("=" * 70)
    print("GENERATING DECISION")
    print("=" * 70)

    result = generate_decision(data)

    print()
    print(
        "Detected modulation :",
        result["classification"]["modulation"],
    )

    print(
        "Confidence           :",
        result["classification"]["confidence_percent"],
        "%",
    )

    print(
        "Signal quality       :",
        result["signal_assessment"]["signal_quality"],
    )

    print(
        "Reliability          :",
        result["signal_assessment"]["classification_reliability"],
    )

    print()
    print(
        "Confidence state     :",
        result["decision"]["confidence_state"],
    )

    print(
        "Simulation state     :",
        result["decision"]["simulation_state"],
    )

    print()
    print("Operator recommendation:")

    print(result["decision"]["operator_action"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("DECISION ENGINE COMPLETE")
    print("=" * 70)

    print()
    print("[OK] Decision report:")
    print(output_path)


if __name__ == "__main__":

    main()
