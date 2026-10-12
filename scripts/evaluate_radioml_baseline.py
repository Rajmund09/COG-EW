from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from data.loaders.baseline_loaders import (
    create_baseline_loaders,
)

from perception.models.radioml_cnn import (
    RadioMLCNN,
)

HDF5_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "radioml_cnn_baseline_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 64

NUM_WORKERS = 0

NUM_CLASSES = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW RADIOML BASELINE EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    print("\nLoading test dataset...")

    _, _, test_loader = create_baseline_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Test samples: " f"{len(test_loader.dataset):,}")

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = RadioMLCNN(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(
        str(CHECKPOINT_PATH),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("\n[OK] Best checkpoint loaded.")

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    all_predictions = []

    all_labels = []

    all_snrs = []

    print("\nGenerating predictions...")

    with torch.no_grad():

        for batch_idx, (
            signals,
            labels,
            snr,
        ) in enumerate(test_loader):

            signals = signals.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(signals)

            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            all_predictions.append(predictions)

            all_labels.append(labels.numpy())

            all_snrs.append(snr.numpy())

            if (batch_idx + 1) % 25 == 0:

                print(f"  Processed " f"{batch_idx + 1}/" f"{len(test_loader)} batches")

    predictions = np.concatenate(all_predictions)

    labels = np.concatenate(all_labels)

    snrs = np.concatenate(all_snrs)

    # ------------------------------------------------------
    # Overall accuracy
    # ------------------------------------------------------

    accuracy = (predictions == labels).mean()

    print("\nOverall accuracy:")

    print(f"{accuracy * 100:.2f}%")

    # ------------------------------------------------------
    # Classification report
    # ------------------------------------------------------

    class_names = [f"Class_{i:02d}" for i in range(NUM_CLASSES)]

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # ------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )

    # ------------------------------------------------------
    # SNR analysis
    # ------------------------------------------------------

    snr_results = {}

    unique_snrs = sorted(np.unique(snrs))

    for snr_value in unique_snrs:

        mask = snrs == snr_value

        snr_accuracy = (predictions[mask] == labels[mask]).mean()

        snr_results[str(int(snr_value))] = {
            "samples": int(mask.sum()),
            "accuracy": float(snr_accuracy),
            "accuracy_percent": float(snr_accuracy * 100),
        }

    # ------------------------------------------------------
    # Per-class accuracy
    # ------------------------------------------------------

    class_results = {}

    for class_id in range(NUM_CLASSES):

        mask = labels == class_id

        class_accuracy = (predictions[mask] == labels[mask]).mean()

        class_results[str(class_id)] = {
            "samples": int(mask.sum()),
            "accuracy": float(class_accuracy),
            "accuracy_percent": float(class_accuracy * 100),
        }

    # ------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------

    evaluation = {
        "checkpoint": str(CHECKPOINT_PATH),
        "test_samples": int(len(labels)),
        "overall_accuracy": float(accuracy),
        "overall_accuracy_percent": float(accuracy * 100),
        "classification_report": report,
        "snr_analysis": snr_results,
        "class_analysis": class_results,
        "confusion_matrix": cm.tolist(),
    }

    report_path = REPORT_DIR / "radioml_baseline_evaluation.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=2,
        )

    # ------------------------------------------------------
    # Confusion matrix figure
    # ------------------------------------------------------

    plt.figure(figsize=(12, 10))

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title("COG-EW RadioML Baseline Confusion Matrix")

    plt.xlabel("Predicted Class")

    plt.ylabel("True Class")

    plt.colorbar()

    plt.xticks(np.arange(NUM_CLASSES))

    plt.yticks(np.arange(NUM_CLASSES))

    plt.tight_layout()

    cm_path = FIGURE_DIR / "radioml_baseline_confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # SNR accuracy figure
    # ------------------------------------------------------

    snr_values = np.array(unique_snrs)

    snr_accuracy_values = np.array(
        [snr_results[str(int(s))]["accuracy_percent"] for s in snr_values]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        snr_values,
        snr_accuracy_values,
        marker="o",
    )

    plt.title("RadioML Classification Accuracy vs SNR")

    plt.xlabel("SNR (dB)")

    plt.ylabel("Accuracy (%)")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    snr_path = FIGURE_DIR / "radioml_accuracy_vs_snr.png"

    plt.savefig(
        snr_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print("SNR PERFORMANCE")

    print("=" * 70)

    for snr_value in unique_snrs:

        result = snr_results[str(int(snr_value))]

        print(
            f"SNR {int(snr_value):>3} dB:"
            f" "
            f"{result['accuracy_percent']:.2f}%"
            f" "
            f"({result['samples']:,} samples)"
        )

    print("\n" + "=" * 70)

    print("FILES CREATED")

    print("=" * 70)

    print("Evaluation:")

    print(report_path)

    print("\nConfusion matrix:")

    print(cm_path)

    print("\nSNR accuracy:")

    print(snr_path)

    print("\n[PASS] Evaluation completed.")


if __name__ == "__main__":
    main()
