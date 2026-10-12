"""
COG-EW RF SIGNAL QUALITY + CLASSIFICATION ANALYZER

Purpose
-------
Analyzes an RF IQ sample using the trained COG-EW RF CNN and
produces:

1. RF signal power
2. IQ amplitude statistics
3. Estimated signal quality
4. SNR
5. Predicted modulation
6. Prediction confidence
7. Confidence category
8. Classification reliability
9. Safe simulation recommendation

This module is intended for:
- RF classification
- defensive RF monitoring
- simulation
- decision-support dashboards

It does NOT perform real-world RF transmission or jamming.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "cogew_rf_cnn_best.pt"

TEST_DATASET = PROJECT_ROOT / "data" / "splits" / "cogew_test.h5"

REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_OUTPUT = REPORT_DIR / "rf_signal_quality_analysis.json"

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

EXPECTED_IQ_LENGTH = 128


# ============================================================
# MODEL
# ============================================================


class COGEWRFCNN(nn.Module):
    """
    Must match the architecture used during training.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            # Block 4
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# SHA256
# ============================================================


def calculate_sha256(path: Path) -> str:

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# MODEL LOADING
# ============================================================


def load_model(device):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    model = COGEWRFCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ============================================================
# IQ VALIDATION
# ============================================================


def validate_iq(iq: np.ndarray):

    if not isinstance(iq, np.ndarray):

        raise TypeError("IQ sample must be a NumPy array.")

    if iq.shape != (EXPECTED_IQ_LENGTH, 2):

        raise ValueError(
            f"Expected IQ shape " f"({EXPECTED_IQ_LENGTH}, 2), " f"received {iq.shape}"
        )

    if not np.isfinite(iq).all():

        raise ValueError("IQ sample contains NaN or Inf.")


# ============================================================
# RF SIGNAL METRICS
# ============================================================


def calculate_signal_metrics(iq: np.ndarray) -> Dict[str, Any]:

    i = iq[:, 0].astype(np.float64)
    q = iq[:, 1].astype(np.float64)

    complex_signal = i + 1j * q

    amplitude = np.abs(complex_signal)

    power = amplitude**2

    phase = np.angle(complex_signal)

    signal_power = float(np.mean(power))

    amplitude_mean = float(np.mean(amplitude))
    amplitude_std = float(np.std(amplitude))

    amplitude_min = float(np.min(amplitude))
    amplitude_max = float(np.max(amplitude))

    phase_mean = float(np.mean(phase))
    phase_std = float(np.std(phase))

    i_power = float(np.mean(i**2))
    q_power = float(np.mean(q**2))

    iq_power_imbalance = abs(i_power - q_power) / max(i_power + q_power, 1e-12)

    dc_i = float(np.mean(i))
    dc_q = float(np.mean(q))

    dc_offset = math.sqrt(dc_i**2 + dc_q**2)

    # Robust noise estimate using first differences.
    differences = np.diff(complex_signal)

    noise_power_estimate = float(np.mean(np.abs(differences) ** 2)) / 2.0

    noise_power_estimate = max(noise_power_estimate, 1e-12)

    estimated_snr_db = 10.0 * math.log10(
        max(signal_power, 1e-12) / noise_power_estimate
    )

    return {
        "signal_power": signal_power,
        "iq_amplitude": {
            "mean": amplitude_mean,
            "std": amplitude_std,
            "min": amplitude_min,
            "max": amplitude_max,
        },
        "iq_phase": {
            "mean": phase_mean,
            "std": phase_std,
        },
        "i_power": i_power,
        "q_power": q_power,
        "iq_power_imbalance": float(iq_power_imbalance),
        "dc_offset": float(dc_offset),
        "estimated_noise_power": float(noise_power_estimate),
        "estimated_snr_db": float(estimated_snr_db),
    }


# ============================================================
# SIGNAL QUALITY
# ============================================================


def determine_signal_quality(
    snr_db: float,
    confidence: float,
) -> Dict[str, Any]:

    # Confidence thresholds
    if confidence >= 0.85:

        confidence_level = "HIGH"

    elif confidence >= 0.65:

        confidence_level = "MEDIUM"

    elif confidence >= 0.45:

        confidence_level = "LOW"

    else:

        confidence_level = "VERY_LOW"

    # Signal quality based primarily on SNR.
    if snr_db >= 10:

        signal_quality = "EXCELLENT"

    elif snr_db >= 5:

        signal_quality = "GOOD"

    elif snr_db >= 0:

        signal_quality = "MODERATE"

    elif snr_db >= -5:

        signal_quality = "POOR"

    else:

        signal_quality = "VERY_POOR"

    # Combined reliability.
    if snr_db >= 5 and confidence >= 0.80:

        reliability = "HIGH"

    elif snr_db >= 0 and confidence >= 0.60:

        reliability = "MEDIUM"

    elif snr_db >= -5 and confidence >= 0.40:

        reliability = "LOW"

    else:

        reliability = "VERY_LOW"

    return {
        "signal_quality": signal_quality,
        "confidence_level": confidence_level,
        "classification_reliability": reliability,
    }


# ============================================================
# SAFE DECISION SUPPORT
# ============================================================


def generate_simulation_recommendation(
    predicted_class: str,
    confidence: float,
    snr_db: float,
    reliability: str,
) -> Dict[str, Any]:
    """
    Produces a defensive simulation recommendation.

    It does NOT generate real-world jamming commands,
    transmission parameters, targeting information,
    or weapon-control instructions.
    """

    if reliability == "HIGH":

        action = "CLASSIFICATION_CONFIDENT"

        explanation = (
            "Signal classification is sufficiently "
            "confident for defensive simulation analysis."
        )

    elif reliability == "MEDIUM":

        action = "CLASSIFICATION_REVIEW"

        explanation = (
            "Classification can be used for simulation, "
            "but additional signal evidence is recommended."
        )

    elif reliability == "LOW":

        action = "LOW_CONFIDENCE_REVIEW"

        explanation = (
            "Classification confidence is limited. "
            "Do not make an automated high-impact decision."
        )

    else:

        action = "INSUFFICIENT_SIGNAL_CONFIDENCE"

        explanation = (
            "Signal quality or classifier confidence is "
            "too low for reliable automated classification."
        )

    return {
        "simulation_action": action,
        "predicted_modulation": predicted_class,
        "confidence": float(confidence),
        "estimated_snr_db": float(snr_db),
        "classification_reliability": reliability,
        "explanation": explanation,
    }


# ============================================================
# MODEL PREDICTION
# ============================================================


@torch.no_grad()
def predict_signal(
    model,
    iq: np.ndarray,
    device,
):

    validate_iq(iq)

    # Current IQ shape:
    # (128, 2)

    # CNN expects:
    # (batch, channels, time)
    #
    # Convert:
    # (128, 2)
    # ->
    # (2, 128)

    x = iq.astype(np.float32).T

    x = torch.from_numpy(x).unsqueeze(0)

    x = x.to(device)

    logits = model(x)

    probabilities = torch.softmax(logits, dim=1)[0]

    confidence, prediction = torch.max(probabilities, dim=0)

    predicted_index = int(prediction.item())

    confidence_value = float(confidence.item())

    probabilities_np = probabilities.detach().cpu().numpy()

    class_probabilities = {
        CLASS_NAMES[i]: float(probabilities_np[i]) for i in range(len(CLASS_NAMES))
    }

    return (
        predicted_index,
        confidence_value,
        class_probabilities,
    )


# ============================================================
# LOAD TEST SAMPLE
# ============================================================


def load_test_sample(index: int):

    if not TEST_DATASET.exists():

        raise FileNotFoundError(f"Test dataset not found:\n" f"{TEST_DATASET}")

    with h5py.File(TEST_DATASET, "r") as h5:

        total = h5["X"].shape[0]

        if index < 0 or index >= total:

            raise IndexError(f"Index {index} is outside " f"range 0-{total - 1}")

        iq = h5["X"][index]

        label = int(h5["Y"][index])

        snr = int(h5["Z"][index])

        original_index = None
        source = None

        if "original_index" in h5:

            original_index = int(h5["original_index"][index])

        if "source" in h5:

            source = int(h5["source"][index])

    return (
        iq,
        label,
        snr,
        original_index,
        source,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================


def analyze_sample(
    index: int,
    output_path: Path,
):

    print("=" * 70)
    print("COG-EW RF SIGNAL QUALITY ANALYZER")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("Device :", device)

    if device.type == "cuda":

        print("GPU    :", torch.cuda.get_device_name(0))

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Test dataset:")
    print(TEST_DATASET)

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = load_model(device)

    model_sha256 = calculate_sha256(MODEL_PATH)

    print()
    print("[PASS] Model loaded")
    print("SHA256:", model_sha256)

    print()
    print("=" * 70)
    print("LOADING RF SAMPLE")
    print("=" * 70)

    (
        iq,
        true_label,
        dataset_snr,
        original_index,
        source,
    ) = load_test_sample(index)

    print()
    print("Sample index :", index)
    print("IQ shape     :", iq.shape)
    print("Dataset SNR  :", dataset_snr, "dB")

    if original_index is not None:

        print("Original index:", original_index)

    if source is not None:

        source_name = {
            0: "RadioML2016",
            1: "RadioML2018",
        }.get(source, f"Unknown({source})")

        print("Source       :", source_name)

    print()
    print("=" * 70)
    print("SIGNAL QUALITY ANALYSIS")
    print("=" * 70)

    metrics = calculate_signal_metrics(iq)

    print()
    print(f"Signal power : " f"{metrics['signal_power']:.6f}")

    print(f"Estimated SNR: " f"{metrics['estimated_snr_db']:.2f} dB")

    print(f"IQ imbalance : " f"{metrics['iq_power_imbalance']:.4f}")

    print(f"DC offset    : " f"{metrics['dc_offset']:.6f}")

    print()
    print("=" * 70)
    print("CNN CLASSIFICATION")
    print("=" * 70)

    (
        predicted_index,
        confidence,
        probabilities,
    ) = predict_signal(
        model,
        iq,
        device,
    )

    predicted_class = CLASS_NAMES[predicted_index]

    true_class = CLASS_NAMES[true_label]

    quality = determine_signal_quality(
        metrics["estimated_snr_db"],
        confidence,
    )

    recommendation = generate_simulation_recommendation(
        predicted_class,
        confidence,
        metrics["estimated_snr_db"],
        quality["classification_reliability"],
    )

    print()
    print("Predicted class :", predicted_class)

    print("Confidence      :", f"{confidence * 100:.2f}%")

    print("True class      :", true_class)

    print("Correct         :", predicted_class == true_class)

    print()
    print("CLASS PROBABILITIES")
    print("-" * 50)

    sorted_probs = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for name, probability in sorted_probs:

        print(f"{name:<10}" f"{probability * 100:>8.2f}%")

    print()
    print("SIGNAL QUALITY")
    print("-" * 50)

    print("Signal quality:", quality["signal_quality"])

    print("Confidence    :", quality["confidence_level"])

    print("Reliability   :", quality["classification_reliability"])

    print()
    print("SIMULATION DECISION SUPPORT")
    print("-" * 50)

    print("Action:", recommendation["simulation_action"])

    print("Reason:", recommendation["explanation"])

    result = {
        "project": "COG-EW",
        "module": "RF Signal Quality Analyzer",
        "sample": {
            "test_index": index,
            "original_index": original_index,
            "source_id": source,
            "dataset_snr_db": dataset_snr,
            "true_class": true_class,
        },
        "model": {
            "path": str(MODEL_PATH),
            "sha256": model_sha256,
        },
        "signal_metrics": metrics,
        "classification": {
            "predicted_class": predicted_class,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "correct": predicted_class == true_class,
            "class_probabilities": probabilities,
        },
        "quality_assessment": quality,
        "simulation_decision_support": recommendation,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("[OK] Report saved:")
    print(output_path)


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze COG-EW RF signal quality " "and CNN classification confidence."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=5000,
        help=("Index from the untouched " "COG-EW test dataset."),
    )

    parser.add_argument(
        "--save",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=("Output JSON report path."),
    )

    args = parser.parse_args()

    analyze_sample(
        index=args.index,
        output_path=Path(args.save),
    )


if __name__ == "__main__":

    main()
