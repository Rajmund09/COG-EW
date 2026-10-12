"""
COG-EW RF VISUALIZATION GENERATOR

Purpose:
    Generate analytical RF visualizations from the completed
    COG-EW model evaluation artifacts.

This module is for:
    - RF signal analysis
    - modulation-classification visualization
    - model-performance visualization
    - simulation decision support

It does NOT:
    - control hardware
    - transmit RF signals
    - implement real-world jamming
    - control aircraft
    - control weapons
"""

from pathlib import Path
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\COG-EW")

TEST_DATASET = PROJECT_ROOT / "data" / "splits" / "cogew_test.h5"

EVALUATION_JSON = PROJECT_ROOT / "reports" / "cogew_rf_cnn_test_evaluation.json"

SUMMARY_JSON = PROJECT_ROOT / "reports" / "cogew_rf_model_summary.json"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]


# ============================================================
# LOAD JSON
# ============================================================


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD TEST DATA
# ============================================================


def load_test_data():

    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    with h5py.File(TEST_DATASET, "r") as h5:

        X = h5["X"][:]
        Y = h5["Y"][:]
        Z = h5["Z"][:]

    print(f"X shape : {X.shape}")
    print(f"Y shape : {Y.shape}")
    print(f"Z shape : {Z.shape}")

    return X, Y, Z


# ============================================================
# IQ WAVEFORM
# ============================================================


def plot_iq_waveform(X, index=5000):

    sample = X[index]

    I = sample[:, 0]
    Q = sample[:, 1]

    time = np.arange(len(I))

    plt.figure(figsize=(12, 6))

    plt.plot(time, I, label="In-phase (I)")

    plt.plot(time, Q, label="Quadrature (Q)")

    plt.title(f"COG-EW RF IQ Waveform - Sample {index}")

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.grid(True, alpha=0.3)
    plt.legend()

    path = OUTPUT_DIR / "rf_iq_waveform.png"

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] {path}")


# ============================================================
# CONSTELLATION
# ============================================================


def plot_constellation(X, index=5000):

    sample = X[index]

    I = sample[:, 0]
    Q = sample[:, 1]

    plt.figure(figsize=(7, 7))

    plt.scatter(I, Q, s=20, alpha=0.7)

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.title(f"RF IQ Constellation - Sample {index}")

    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")

    plt.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "rf_constellation.png"

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] {path}")


# ============================================================
# AMPLITUDE
# ============================================================


def plot_amplitude(X, index=5000):

    sample = X[index]

    I = sample[:, 0]
    Q = sample[:, 1]

    amplitude = np.sqrt(I**2 + Q**2)

    plt.figure(figsize=(12, 5))

    plt.plot(amplitude)

    plt.title(f"RF Signal Amplitude - Sample {index}")

    plt.xlabel("Sample")
    plt.ylabel("Magnitude")

    plt.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "rf_amplitude.png"

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] {path}")


# ============================================================
# PHASE
# ============================================================


def plot_phase(X, index=5000):

    sample = X[index]

    I = sample[:, 0]
    Q = sample[:, 1]

    phase = np.unwrap(np.arctan2(Q, I))

    plt.figure(figsize=(12, 5))

    plt.plot(phase)

    plt.title(f"RF Signal Phase - Sample {index}")

    plt.xlabel("Sample")
    plt.ylabel("Phase (radians)")

    plt.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "rf_phase.png"

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] {path}")


# ============================================================
# SNR PERFORMANCE
# ============================================================


def plot_snr_performance():

    evaluation = load_json(EVALUATION_JSON)

    snr_data = evaluation.get("snr_accuracy", evaluation.get("accuracy_by_snr", {}))

    if not snr_data:

        print("[WARN] snr_accuracy not found.")

        return

    snrs = []
    accuracies = []

    for snr, val in snr_data.items():

        acc = val.get("accuracy", val.get("acc")) if isinstance(val, dict) else val

        if acc is not None:
            snrs.append(float(snr))
            accuracies.append(float(acc))

    order = np.argsort(snrs)

    snrs = np.array(snrs)[order]
    accuracies = np.array(accuracies)[order]

    plt.figure(figsize=(12, 6))

    plt.plot(snrs, accuracies, marker="o")

    plt.axhline(0.5, linestyle="--", linewidth=1)

    plt.title("COG-EW RF CNN Accuracy vs SNR")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1)

    plt.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "accuracy_by_snr.png"

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] {path}")


# ============================================================
# CLASS PERFORMANCE
# ============================================================


def plot_class_performance():

    evaluation = load_json(EVALUATION_JSON)

    metrics = evaluation.get("class_metrics", evaluation.get("classification_metrics", {}))

    if not metrics:

        print("[WARN] class_metrics not found.")

        return

    classes = []
    f1_scores = []

    for class_name in CLASS_NAMES:

        if class_name not in metrics:
            continue

        m = metrics[class_name]
        f1_val = m.get("f1_score", m.get("f1"))

        if f1_val is not None:
            classes.append(class_name)
            f1_scores.append(float(f1_val))

    plt.figure(figsize=(10, 6))

    plt.bar(classes, f1_scores)

    plt.title("COG-EW RF CNN Class-wise F1 Score")

    plt.xlabel("Modulation")
    plt.ylabel("F1 Score")

    plt.ylim(0, 1)

    plt.grid(axis="y", alpha=0.3)

    path = OUTPUT_DIR / "class_f1_scores.png"

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] {path}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("COG-EW RF VISUALIZATION GENERATOR")
    print("=" * 70)

    X, Y, Z = load_test_data()

    probe_index = 5000

    plot_iq_waveform(X, probe_index)

    plot_constellation(X, probe_index)

    plot_amplitude(X, probe_index)

    plot_phase(X, probe_index)

    plot_snr_performance()

    plot_class_performance()

    print()
    print("=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)

    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
