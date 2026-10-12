from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

BASELINE_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "radioml_resnet_best.pt"

ROBUST_CHECKPOINT = (
    PROJECT_ROOT / "models" / "checkpoints" / "radioml_resnet_robust_best.pt"
)

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

SEED = 42


# ============================================================
# IMPORTANT:
# DO NOT ASSUME SEMANTIC CLASS NAMES
# ============================================================
#
# The current COG-EW RadioML pipeline uses class IDs.
# Until the exact Y mapping has been independently verified,
# we deliberately use generic names.
#
# This prevents accidentally assigning the wrong modulation
# name to a class ID.
# ============================================================

CLASS_NAMES = [f"Class {index:02d}" for index in range(NUM_CLASSES)]


# ============================================================
# HDF5 RESOLUTION
# ============================================================


def resolve_hdf5_path() -> Path:

    data_root = Path(DATA_ROOT)

    direct_candidates = list(data_root.glob("*RadioML*/GOLD_XYZ_OSC.0001_1024.hdf5"))

    for candidate in direct_candidates:

        if candidate.exists():

            return candidate

    recursive_candidates = list(data_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"))

    if recursive_candidates:

        return recursive_candidates[0]

    raise FileNotFoundError(
        "Could not locate RadioML HDF5 dataset.\n" f"DATA_ROOT:\n{data_root}"
    )


# ============================================================
# HEADER
# ============================================================


def print_header(title: str):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


# ============================================================
# CHECKPOINT LOADING
# ============================================================


def load_model(
    checkpoint_path: Path,
    model_name: str,
):

    print(f"\nLoading {model_name} checkpoint...")

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"{model_name} checkpoint not found:\n" f"{checkpoint_path}"
        )

    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(checkpoint["model_state_dict"])

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    print(f"[OK] {model_name} model loaded.")

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:

            print(f"{model_name} checkpoint epoch: " f"{checkpoint['epoch']}")

        if "validation_accuracy" in checkpoint:

            print(
                f"{model_name} validation accuracy: "
                f"{checkpoint['validation_accuracy'] * 100:.2f}%"
            )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================


def evaluate_model(
    model,
    test_loader,
    model_name: str,
):

    print_header(f"{model_name.upper()} DIAGNOSTIC EVALUATION")

    all_labels = []

    all_predictions = []

    all_snrs = []

    all_confidences = []

    total_batches = len(test_loader)

    with torch.no_grad():

        for batch_index, batch in enumerate(
            test_loader,
            start=1,
        ):

            signals, labels, snrs = batch

            signals = signals.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
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

            all_labels.append(labels.cpu().numpy())

            all_predictions.append(predictions.cpu().numpy())

            all_snrs.append(snrs.numpy())

            all_confidences.append(confidence.cpu().numpy())

            if batch_index % 50 == 0:

                print(f"  Processed " f"{batch_index:,}/" f"{total_batches:,} batches")

    y_true = np.concatenate(all_labels)

    y_pred = np.concatenate(all_predictions)

    snr_values = np.concatenate(all_snrs)

    confidences = np.concatenate(all_confidences)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr_results = {}

    for snr in sorted(np.unique(snr_values)):

        mask = snr_values == snr

        if not np.any(mask):

            continue

        snr_accuracy = float(np.mean(y_pred[mask] == y_true[mask]))

        snr_results[str(int(snr))] = {
            "accuracy": snr_accuracy,
            "accuracy_percent": (snr_accuracy * 100.0),
            "samples": int(np.sum(mask)),
        }

    # --------------------------------------------------------
    # PER CLASS
    # --------------------------------------------------------

    per_class = {}

    for class_id in range(NUM_CLASSES):

        mask = y_true == class_id

        samples = int(np.sum(mask))

        if samples == 0:

            class_accuracy = 0.0

        else:

            class_accuracy = float(np.mean(y_pred[mask] == class_id))

        per_class[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "accuracy": class_accuracy,
            "accuracy_percent": (class_accuracy * 100.0),
            "samples": samples,
        }

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )

    cm_normalized = cm.astype(np.float64) / np.maximum(
        cm.sum(axis=1, keepdims=True),
        1,
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    classification = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "model": model_name,
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100.0),
        "average_confidence": float(np.mean(confidences)),
        "average_confidence_percent": float(np.mean(confidences) * 100.0),
        "samples": int(len(y_true)),
        "snr_results": snr_results,
        "per_class": per_class,
        "classification_report": (classification),
        "confusion_matrix": (cm.tolist()),
        "confusion_matrix_normalized": (cm_normalized.tolist()),
        # Keep raw arrays internally.
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_snr_values": snr_values,
    }

    print(f"\n{model_name} accuracy: " f"{accuracy * 100:.2f}%")

    print(f"{model_name} confidence: " f"{np.mean(confidences) * 100:.2f}%")

    return result


# ============================================================
# PER-CLASS COMPARISON
# ============================================================


def compare_classes(
    baseline,
    robust,
):

    rows = []

    for class_id in range(NUM_CLASSES):

        key = str(class_id)

        baseline_accuracy = baseline["per_class"][key]["accuracy_percent"]

        robust_accuracy = robust["per_class"][key]["accuracy_percent"]

        gain = robust_accuracy - baseline_accuracy

        samples = baseline["per_class"][key]["samples"]

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "samples": samples,
                "baseline_accuracy_percent": (baseline_accuracy),
                "robust_accuracy_percent": (robust_accuracy),
                "gain_percentage_points": gain,
            }
        )

    return rows


# ============================================================
# SNR COMPARISON
# ============================================================


def compare_snr(
    baseline,
    robust,
):

    rows = []

    snrs = sorted(
        set(baseline["snr_results"].keys()) | set(robust["snr_results"].keys()),
        key=lambda value: int(value),
    )

    for snr in snrs:

        baseline_value = baseline["snr_results"].get(
            snr,
            {"accuracy_percent": 0.0},
        )["accuracy_percent"]

        robust_value = robust["snr_results"].get(
            snr,
            {"accuracy_percent": 0.0},
        )["accuracy_percent"]

        rows.append(
            {
                "snr_db": int(snr),
                "baseline_accuracy_percent": (baseline_value),
                "robust_accuracy_percent": (robust_value),
                "gain_percentage_points": (robust_value - baseline_value),
            }
        )

    return rows


# ============================================================
# FIGURE: PER-CLASS COMPARISON
# ============================================================


def create_class_comparison_plot(
    class_rows,
):

    class_ids = [row["class_id"] for row in class_rows]

    baseline_values = [row["baseline_accuracy_percent"] for row in class_rows]

    robust_values = [row["robust_accuracy_percent"] for row in class_rows]

    x = np.arange(len(class_ids))

    width = 0.38

    plt.figure(figsize=(14, 7))

    plt.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline",
    )

    plt.bar(
        x + width / 2,
        robust_values,
        width,
        label="Robust",
    )

    plt.xlabel("RadioML class ID")

    plt.ylabel("Accuracy (%)")

    plt.title("COG-EW Baseline vs Robust " "Per-Class Accuracy")

    plt.xticks(
        x,
        [f"{value:02d}" for value in class_ids],
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "radioml_baseline_vs_robust_per_class.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================
# FIGURE: SNR COMPARISON
# ============================================================


def create_snr_comparison_plot(
    snr_rows,
):

    x = [row["snr_db"] for row in snr_rows]

    baseline_values = [row["baseline_accuracy_percent"] for row in snr_rows]

    robust_values = [row["robust_accuracy_percent"] for row in snr_rows]

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        baseline_values,
        marker="o",
        linewidth=2,
        label="Baseline",
    )

    plt.plot(
        x,
        robust_values,
        marker="s",
        linewidth=2,
        label="Robust",
    )

    plt.xlabel("SNR (dB)")

    plt.ylabel("Classification accuracy (%)")

    plt.title("COG-EW RadioML Accuracy vs SNR")

    plt.xticks(x)

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "radioml_baseline_vs_robust_snr.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================
# FIGURE: CONFUSION MATRICES
# ============================================================


def create_confusion_plot(
    matrix,
    title,
    filename,
):

    plt.figure(figsize=(10, 9))

    plt.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar(label="Normalized frequency")

    plt.xlabel("Predicted class")

    plt.ylabel("True class")

    plt.title(title)

    ticks = np.arange(NUM_CLASSES)

    plt.xticks(
        ticks,
        [f"{i:02d}" for i in ticks],
        rotation=90,
    )

    plt.yticks(
        ticks,
        [f"{i:02d}" for i in ticks],
    )

    plt.tight_layout()

    path = FIGURE_DIR / filename

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


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

    print_header("COG-EW RADIO ML DIAGNOSTIC ANALYSIS")

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
    # Test split
    # --------------------------------------------------------

    test_split_path = PROCESSED_DIR / "baseline_test_split.npy"

    if not test_split_path.exists():

        raise FileNotFoundError(f"Test split not found:\n" f"{test_split_path}")

    test_indices = np.load(test_split_path)

    print(
        "\nTest samples:",
        f"{len(test_indices):,}",
    )

    if len(test_indices) != 9000:

        raise RuntimeError("Expected exactly 9,000 " "test samples.")

    print("[OK] COG-EW benchmark test split confirmed.")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print_header("CREATING TEST DATALOADER")

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

    print(
        "Test samples:",
        f"{len(test_loader.dataset):,}",
    )

    print(
        "Test batches:",
        f"{len(test_loader):,}",
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print_header("LOADING MODELS")

    baseline_model = load_model(
        BASELINE_CHECKPOINT,
        "Baseline",
    )

    robust_model = load_model(
        ROBUST_CHECKPOINT,
        "Robust",
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    baseline = evaluate_model(
        baseline_model,
        test_loader,
        "Baseline",
    )

    robust = evaluate_model(
        robust_model,
        test_loader,
        "Robust",
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    baseline_accuracy = baseline["accuracy_percent"]

    robust_accuracy = robust["accuracy_percent"]

    overall_gain = robust_accuracy - baseline_accuracy

    print_header("OVERALL DIAGNOSTIC RESULT")

    print(f"Baseline accuracy: " f"{baseline_accuracy:.2f}%")

    print(f"Robust accuracy:   " f"{robust_accuracy:.2f}%")

    print(f"Robust gain:       " f"{overall_gain:+.2f} pp")

    print()

    print(f"Baseline confidence: " f"{baseline['average_confidence_percent']:.2f}%")

    print(f"Robust confidence:   " f"{robust['average_confidence_percent']:.2f}%")

    # --------------------------------------------------------
    # Per-class
    # --------------------------------------------------------

    class_rows = compare_classes(
        baseline,
        robust,
    )

    print_header("PER-CLASS ROBUSTNESS ANALYSIS")

    print("Class | Baseline | Robust | Gain")

    print("-" * 55)

    for row in class_rows:

        print(
            f"{row['class_id']:>5} | "
            f"{row['baseline_accuracy_percent']:>8.2f}% | "
            f"{row['robust_accuracy_percent']:>6.2f}% | "
            f"{row['gain_percentage_points']:>+6.2f} pp"
        )

    # Sort to find strongest improvements
    sorted_gains = sorted(
        class_rows,
        key=lambda row: row["gain_percentage_points"],
        reverse=True,
    )

    print_header("TOP ROBUST-MODEL IMPROVEMENTS")

    for row in sorted_gains[:5]:

        print(
            f"Class {row['class_id']:02d}: " f"{row['gain_percentage_points']:+.2f} pp"
        )

    print_header("LARGEST ROBUST-MODEL REGRESSIONS")

    for row in sorted_gains[-5:][::-1]:

        print(
            f"Class {row['class_id']:02d}: " f"{row['gain_percentage_points']:+.2f} pp"
        )

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr_rows = compare_snr(
        baseline,
        robust,
    )

    print_header("SNR DIAGNOSTIC COMPARISON")

    print("SNR | Baseline | Robust | Gain")

    print("-" * 55)

    for row in snr_rows:

        print(
            f"{row['snr_db']:>3} | "
            f"{row['baseline_accuracy_percent']:>8.2f}% | "
            f"{row['robust_accuracy_percent']:>6.2f}% | "
            f"{row['gain_percentage_points']:>+6.2f} pp"
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print_header("GENERATING DIAGNOSTIC FIGURES")

    class_plot = create_class_comparison_plot(class_rows)

    print("[OK] Per-class figure:")

    print(class_plot)

    snr_plot = create_snr_comparison_plot(snr_rows)

    print("[OK] SNR figure:")

    print(snr_plot)

    baseline_cm_plot = create_confusion_plot(
        np.asarray(baseline["confusion_matrix_normalized"]),
        "COG-EW Baseline " "RadioML Confusion Matrix",
        "radioml_baseline_confusion_matrix.png",
    )

    print("[OK] Baseline confusion matrix:")

    print(baseline_cm_plot)

    robust_cm_plot = create_confusion_plot(
        np.asarray(robust["confusion_matrix_normalized"]),
        "COG-EW Robust " "RadioML Confusion Matrix",
        "radioml_robust_confusion_matrix.png",
    )

    print("[OK] Robust confusion matrix:")

    print(robust_cm_plot)

    # --------------------------------------------------------
    # Determine difficult classes
    # --------------------------------------------------------

    difficult_classes = sorted(
        class_rows,
        key=lambda row: row["robust_accuracy_percent"],
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "experiment": ("Baseline vs Robust " "RadioML Diagnostic Analysis"),
        "dataset": ("RadioML 2018.01A"),
        "test_samples": int(len(test_indices)),
        "num_classes": NUM_CLASSES,
        "baseline": {
            "checkpoint": str(BASELINE_CHECKPOINT),
            "accuracy_percent": (baseline_accuracy),
            "average_confidence_percent": (baseline["average_confidence_percent"]),
            "snr_results": (baseline["snr_results"]),
            "per_class": (baseline["per_class"]),
        },
        "robust": {
            "checkpoint": str(ROBUST_CHECKPOINT),
            "accuracy_percent": (robust_accuracy),
            "average_confidence_percent": (robust["average_confidence_percent"]),
            "snr_results": (robust["snr_results"]),
            "per_class": (robust["per_class"]),
        },
        "overall_gain_percentage_points": (overall_gain),
        "per_class_comparison": (class_rows),
        "snr_comparison": (snr_rows),
        "most_improved_classes": (sorted_gains[:5]),
        "largest_regressions": (sorted_gains[-5:][::-1]),
        "lowest_accuracy_classes": (difficult_classes[:5]),
        "figures": {
            "per_class": str(class_plot),
            "snr": str(snr_plot),
            "baseline_confusion_matrix": str(baseline_cm_plot),
            "robust_confusion_matrix": str(robust_cm_plot),
        },
    }

    report_path = REPORT_DIR / "radioml_baseline_vs_robust_diagnostic.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_header("DIAGNOSTIC ANALYSIS COMPLETE")

    print(f"Baseline: " f"{baseline_accuracy:.2f}%")

    print(f"Robust:   " f"{robust_accuracy:.2f}%")

    print(f"Gain:     " f"{overall_gain:+.2f} pp")

    print()

    print("Diagnostic report:")

    print(report_path)

    print()

    print("[PASS] Existing checkpoints " "were not modified.")

    print("[PASS] Original RadioML dataset " "was not modified.")


if __name__ == "__main__":
    main()
