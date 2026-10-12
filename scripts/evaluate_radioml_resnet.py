from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import DATA_ROOT

from data.loaders.baseline_loaders import (
    create_baseline_loaders,
)

from perception.models.radioml_resnet import (
    RadioMLResNet,
)


# ============================================================
# PATHS
# ============================================================

def resolve_hdf5_path() -> Path:
    data_root = Path(DATA_ROOT)
    for p in data_root.glob("*RadioML 2018.01A*"):
        hdf5_candidate = p / "GOLD_XYZ_OSC.0001_1024.hdf5"
        if hdf5_candidate.exists():
            return hdf5_candidate
    
    # Check rglob if nested
    for hdf5_candidate in data_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"):
        if hdf5_candidate.exists():
            return hdf5_candidate

    return data_root / "RadioML 2018.01A ⭐ Main RF benchmark" / "GOLD_XYZ_OSC.0001_1024.hdf5"


HDF5_PATH = resolve_hdf5_path()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "radioml_resnet_best.pt"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 24
BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# CLASS NAMES
# ============================================================

# RadioML 2018.01A modulation classes.

CLASS_NAMES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


# ============================================================
# HELPERS
# ============================================================


def print_header(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:

    if len(labels) == 0:
        return 0.0

    return float(np.mean(predictions == labels))


# ============================================================
# MAIN EVALUATION
# ============================================================


def main():

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    print_header("COG-EW RADIOML RESNET EVALUATION")

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking required files...")

    if not HDF5_PATH.exists():

        raise FileNotFoundError(f"RadioML HDF5 dataset not found:\n" f"{HDF5_PATH}")

    print("[OK] RadioML dataset found.")

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(f"ResNet checkpoint not found:\n" f"{CHECKPOINT_PATH}")

    print("[OK] ResNet checkpoint found.")

    # --------------------------------------------------------
    # Explicit split verification
    # --------------------------------------------------------

    print("\nVerifying benchmark split...")

    train_split = np.load(PROCESSED_DIR / "baseline_train_split.npy")

    val_split = np.load(PROCESSED_DIR / "baseline_val_split.npy")

    test_split = np.load(PROCESSED_DIR / "baseline_test_split.npy")

    print(f"Baseline train samples: {len(train_split):,}")

    print(f"Baseline validation samples: {len(val_split):,}")

    print(f"Baseline test samples: {len(test_split):,}")

    print("\n[IMPORTANT] Evaluation uses " "baseline_test_split.npy")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print("\nCreating DataLoaders...")

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_baseline_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train samples: " f"{len(train_loader.dataset):,}")

    print(f"Validation samples: " f"{len(val_loader.dataset):,}")

    print(f"Test samples: " f"{len(test_loader.dataset):,}")

    print(f"Test batches: " f"{len(test_loader):,}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: " f"{parameter_count:,}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        str(CHECKPOINT_PATH),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("[OK] Best ResNet checkpoint loaded.")

    if "epoch" in checkpoint:

        print(f"Checkpoint epoch: " f"{checkpoint['epoch']}")

    if "validation_accuracy" in checkpoint:

        print(
            "Checkpoint validation accuracy:",
            f"{checkpoint['validation_accuracy'] * 100:.2f}%",
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    all_predictions = []
    all_labels = []
    all_snrs = []
    all_confidences = []

    total_batches = len(test_loader)

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            test_loader,
            start=1,
        ):

            signals, labels, snrs = batch

            signals = signals.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(signals)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predictions = torch.max(
                probabilities,
                dim=1,
            )

            all_predictions.append(predictions.cpu().numpy())

            all_labels.append(labels.numpy())

            all_snrs.append(snrs.numpy())

            all_confidences.append(confidence.cpu().numpy())

            if batch_idx % 25 == 0:

                print(f"  Processed " f"{batch_idx:,}/" f"{total_batches:,} batches")

    # --------------------------------------------------------
    # Concatenate results
    # --------------------------------------------------------

    predictions = np.concatenate(all_predictions)

    labels = np.concatenate(all_labels)

    snrs = np.concatenate(all_snrs)

    confidences = np.concatenate(all_confidences)

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    overall_accuracy = calculate_accuracy(
        predictions,
        labels,
    )

    average_confidence = float(np.mean(confidences))

    print_header("OVERALL PERFORMANCE")

    print(f"Test samples: " f"{len(labels):,}")

    print(f"Overall accuracy: " f"{overall_accuracy * 100:.2f}%")

    print(f"Average confidence: " f"{average_confidence * 100:.2f}%")

    # --------------------------------------------------------
    # SNR performance
    # --------------------------------------------------------

    print_header("SNR PERFORMANCE")

    unique_snrs = sorted(np.unique(snrs))

    snr_results = {}

    for snr in unique_snrs:

        mask = snrs == snr

        snr_accuracy = calculate_accuracy(
            predictions[mask],
            labels[mask],
        )

        sample_count = int(np.sum(mask))

        snr_results[str(int(snr))] = {
            "accuracy": snr_accuracy,
            "accuracy_percent": snr_accuracy * 100,
            "samples": sample_count,
        }

        print(
            f"SNR {int(snr):>3} dB: "
            f"{snr_accuracy * 100:6.2f}% "
            f"({sample_count:,} samples)"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nGenerating confusion matrix...")

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )

    cm_path = FIGURE_DIR / "radioml_resnet_confusion_matrix.png"

    plt.figure(figsize=(14, 12))

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title("COG-EW RadioML ResNet Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=90,
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES,
    )

    plt.xlabel("Predicted Class")

    plt.ylabel("True Class")

    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # SNR accuracy plot
    # --------------------------------------------------------

    print("Generating SNR accuracy plot...")

    snr_x = [int(x) for x in unique_snrs]

    snr_y = [snr_results[str(x)]["accuracy_percent"] for x in snr_x]

    snr_plot_path = FIGURE_DIR / "radioml_resnet_accuracy_vs_snr.png"

    plt.figure(figsize=(12, 6))

    plt.plot(
        snr_x,
        snr_y,
        marker="o",
    )

    plt.title("RadioML ResNet Accuracy vs SNR")

    plt.xlabel("SNR (dB)")

    plt.ylabel("Accuracy (%)")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.xticks(
        snr_x,
        rotation=45,
    )

    plt.tight_layout()

    plt.savefig(
        snr_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nGenerating classification report...")

    classification_report_dict = classification_report(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    classification_report_text = classification_report(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print("\n")
    print(classification_report_text)

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    evaluation_report = {
        "project": "COG-EW",
        "experiment": "RadioML ResNet",
        "dataset": "RadioML 2018.01A",
        "model": "RadioMLResNet",
        "checkpoint": str(CHECKPOINT_PATH),
        "evaluation_split": "baseline_test_split.npy",
        "train_samples": int(len(train_split)),
        "validation_samples": int(len(val_split)),
        "test_samples": int(len(labels)),
        "num_classes": NUM_CLASSES,
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "average_confidence": average_confidence,
        "average_confidence_percent": average_confidence * 100,
        "snr_performance": snr_results,
        "classification_report": classification_report_dict,
        "confusion_matrix": cm.tolist(),
    }

    evaluation_path = REPORT_DIR / "radioml_resnet_evaluation.json"

    with evaluation_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation_report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print_header("FILES CREATED")

    print("Evaluation report:")

    print(evaluation_path)

    print("\nConfusion matrix:")

    print(cm_path)

    print("\nSNR accuracy plot:")

    print(snr_plot_path)

    print_header("RADIO ML RESNET EVALUATION COMPLETE")

    print(f"Final test accuracy: " f"{overall_accuracy * 100:.2f}%")

    print(f"Evaluated samples: " f"{len(labels):,}")

    print("\n[PASS] Evaluation used the same " "baseline test split as training.")


if __name__ == "__main__":
    main()
