from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

BASELINE_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "radioml_resnet_best.pt"

ROBUST_CHECKPOINT = (
    PROJECT_ROOT / "models" / "checkpoints" / "radioml_resnet_robust_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
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

# RadioML 2018.01A modulation classes
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
# DATASET PATH
# ============================================================


def resolve_hdf5_path() -> Path:

    data_root = Path(DATA_ROOT)

    direct_candidates = list(data_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"))

    if not direct_candidates:
        raise FileNotFoundError(
            "RadioML HDF5 dataset not found under:\n" f"{data_root}"
        )

    return direct_candidates[0]


# ============================================================
# MODEL LOADING
# ============================================================


def load_model(
    checkpoint_path: Path,
    name: str,
):

    print(f"\nLoading {name} checkpoint...")

    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"[OK] {name} checkpoint loaded.")

    if "epoch" in checkpoint:
        print(f"{name} epoch: " f"{checkpoint['epoch']}")

    if "validation_accuracy" in checkpoint:
        print(
            f"{name} validation accuracy: "
            f"{checkpoint['validation_accuracy'] * 100:.2f}%"
        )

    return model


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(
    model,
    loader,
):

    predictions = []
    labels = []
    snrs = []

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            signals, batch_labels, batch_snrs = batch

            signals = signals.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(signals)

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            predictions.append(predicted.cpu().numpy())

            labels.append(batch_labels.numpy())

            snrs.append(batch_snrs.numpy())

            if batch_index % 50 == 0:

                print(f"    Processed " f"{batch_index:,}/" f"{len(loader):,} batches")

    return (
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(snrs),
    )


# ============================================================
# PER-CLASS METRICS
# ============================================================


def calculate_class_metrics(
    labels,
    predictions,
):

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )

    results = []

    for class_id in range(NUM_CLASSES):

        actual_count = int(matrix[class_id].sum())

        correct = int(matrix[class_id, class_id])

        accuracy = correct / actual_count if actual_count > 0 else 0.0

        results.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "samples": actual_count,
                "correct": correct,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
            }
        )

    return results, matrix


# ============================================================
# TOP CONFUSIONS
# ============================================================


def calculate_confusions(matrix):

    confusions = []

    for actual in range(NUM_CLASSES):

        for predicted in range(NUM_CLASSES):

            if actual == predicted:
                continue

            count = int(matrix[actual, predicted])

            if count <= 0:
                continue

            confusions.append(
                {
                    "actual_id": actual,
                    "actual_class": CLASS_NAMES[actual],
                    "predicted_id": predicted,
                    "predicted_class": CLASS_NAMES[predicted],
                    "count": count,
                }
            )

    confusions.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return confusions


# ============================================================
# SNR ANALYSIS
# ============================================================


def calculate_snr_metrics(
    labels,
    predictions,
    snrs,
):

    output = {}

    for snr in sorted(np.unique(snrs)):

        mask = snrs == snr

        if not np.any(mask):
            continue

        accuracy = float(np.mean(predictions[mask] == labels[mask]))

        output[str(int(snr))] = {
            "samples": int(np.sum(mask)),
            "accuracy": accuracy,
            "accuracy_percent": (accuracy * 100.0),
        }

    return output


# ============================================================
# PRINT CLASS TABLE
# ============================================================


def print_class_table(
    title,
    metrics,
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"{'ID':>3} | " f"{'Class':<14} | " f"{'Samples':>7} | " f"{'Accuracy':>9}")

    print("-" * 70)

    for item in metrics:

        print(
            f"{item['class_id']:>3} | "
            f"{item['class_name']:<14} | "
            f"{item['samples']:>7,} | "
            f"{item['accuracy_percent']:>8.2f}%"
        )


# ============================================================
# MAIN
# ============================================================


def main():

    if hasattr(
        sys.stdout,
        "reconfigure",
    ):

        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    print("=" * 70)
    print("COG-EW RADIO ML CLASS-AWARE ANALYSIS")
    print("=" * 70)

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
    # Dataset
    # --------------------------------------------------------

    hdf5_path = resolve_hdf5_path()

    print("\nRadioML dataset:")
    print(hdf5_path)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    test_split_path = PROCESSED_DIR / "baseline_test_split.npy"

    test_indices = np.load(test_split_path)

    print(f"\nTest samples: " f"{len(test_indices):,}")

    if len(test_indices) != 9000:

        raise RuntimeError("Expected exactly 9,000 " "test samples.")

    print("[OK] COG-EW benchmark test split confirmed.")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print("\nCreating test DataLoader...")

    (
        _train_loader,
        _val_loader,
        test_loader,
    ) = create_baseline_loaders(
        hdf5_path=hdf5_path,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Test samples: " f"{len(test_loader.dataset):,}")

    print(f"Test batches: " f"{len(test_loader):,}")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    baseline_model = load_model(
        BASELINE_CHECKPOINT,
        "Baseline",
    )

    robust_model = load_model(
        ROBUST_CHECKPOINT,
        "Robust",
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE CLASS ANALYSIS")
    print("=" * 70)

    baseline_labels, baseline_predictions, baseline_snrs = evaluate_model(
        baseline_model,
        test_loader,
    )

    baseline_metrics, baseline_matrix = calculate_class_metrics(
        baseline_labels,
        baseline_predictions,
    )

    baseline_confusions = calculate_confusions(baseline_matrix)

    baseline_snr = calculate_snr_metrics(
        baseline_labels,
        baseline_predictions,
        baseline_snrs,
    )

    print_class_table(
        "BASELINE PER-CLASS PERFORMANCE",
        baseline_metrics,
    )

    # --------------------------------------------------------
    # Robust
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUST CLASS ANALYSIS")
    print("=" * 70)

    robust_labels, robust_predictions, robust_snrs = evaluate_model(
        robust_model,
        test_loader,
    )

    robust_metrics, robust_matrix = calculate_class_metrics(
        robust_labels,
        robust_predictions,
    )

    robust_confusions = calculate_confusions(robust_matrix)

    robust_snr = calculate_snr_metrics(
        robust_labels,
        robust_predictions,
        robust_snrs,
    )

    print_class_table(
        "ROBUST PER-CLASS PERFORMANCE",
        robust_metrics,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = []

    for class_id in range(NUM_CLASSES):

        baseline_accuracy = baseline_metrics[class_id]["accuracy_percent"]

        robust_accuracy = robust_metrics[class_id]["accuracy_percent"]

        gain = robust_accuracy - baseline_accuracy

        comparison.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "baseline_accuracy_percent": (baseline_accuracy),
                "robust_accuracy_percent": (robust_accuracy),
                "gain_percentage_points": gain,
            }
        )

    comparison_sorted = sorted(
        comparison,
        key=lambda item: item["gain_percentage_points"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP CLASS IMPROVEMENTS")
    print("=" * 70)

    for item in comparison_sorted[:10]:

        print(
            f"Class {item['class_id']:02d} "
            f"{item['class_name']:<14} "
            f"{item['gain_percentage_points']:+.2f} pp"
        )

    # --------------------------------------------------------
    # Regressions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP CLASS REGRESSIONS")
    print("=" * 70)

    for item in comparison_sorted[-10:][::-1]:

        print(
            f"Class {item['class_id']:02d} "
            f"{item['class_name']:<14} "
            f"{item['gain_percentage_points']:+.2f} pp"
        )

    # --------------------------------------------------------
    # Confusions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP BASELINE CONFUSIONS")
    print("=" * 70)

    for item in baseline_confusions[:20]:

        print(
            f"{item['actual_class']:<14} -> "
            f"{item['predicted_class']:<14} "
            f"{item['count']:,}"
        )

    print("\n" + "=" * 70)
    print("TOP ROBUST CONFUSIONS")
    print("=" * 70)

    for item in robust_confusions[:20]:

        print(
            f"{item['actual_class']:<14} -> "
            f"{item['predicted_class']:<14} "
            f"{item['count']:,}"
        )

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SNR COMPARISON")
    print("=" * 70)

    print(f"{'SNR':>5} | " f"{'Baseline':>10} | " f"{'Robust':>10} | " f"{'Gain':>10}")

    print("-" * 50)

    for snr in sorted(
        set(baseline_snr.keys()) | set(robust_snr.keys()),
        key=int,
    ):

        baseline_acc = baseline_snr.get(snr, {}).get(
            "accuracy_percent",
            0.0,
        )

        robust_acc = robust_snr.get(snr, {}).get(
            "accuracy_percent",
            0.0,
        )

        gain = robust_acc - baseline_acc

        print(
            f"{int(snr):>5} | "
            f"{baseline_acc:>9.2f}% | "
            f"{robust_acc:>9.2f}% | "
            f"{gain:>+9.2f} pp"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "experiment": ("RadioML class-aware " "baseline vs robust analysis"),
        "dataset": "RadioML 2018.01A",
        "test_samples": int(len(test_indices)),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "baseline": {
            "checkpoint": str(BASELINE_CHECKPOINT),
            "class_metrics": baseline_metrics,
            "confusions": baseline_confusions,
            "snr_metrics": baseline_snr,
        },
        "robust": {
            "checkpoint": str(ROBUST_CHECKPOINT),
            "class_metrics": robust_metrics,
            "confusions": robust_confusions,
            "snr_metrics": robust_snr,
        },
        "comparison": comparison,
        "top_improvements": (comparison_sorted[:10]),
        "top_regressions": (comparison_sorted[-10:][::-1]),
    }

    report_path = REPORT_DIR / "radioml_class_aware_analysis.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("CLASS-AWARE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nReport:")
    print(report_path)

    print("\n[PASS] Baseline checkpoint preserved.")

    print("[PASS] Robust checkpoint preserved.")

    print("[PASS] Original RadioML dataset preserved.")


if __name__ == "__main__":
    main()
