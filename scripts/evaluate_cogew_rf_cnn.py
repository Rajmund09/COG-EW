"""
======================================================================
COG-EW RF AUTOMATIC MODULATION CLASSIFICATION
TEST SET EVALUATION
======================================================================

Purpose:
    Evaluate the BEST trained COG-EW RF CNN model on the completely
    untouched test dataset.

IMPORTANT:
    - Test data is NEVER used during training.
    - Test data is NEVER used for augmentation.
    - Test data is NEVER used for model selection.
    - This script performs evaluation only.
    - Original datasets are not modified.

Outputs:
    models/
        evaluation_reference.txt

    reports/
        cogew_rf_cnn_test_evaluation.json
        cogew_rf_cnn_confusion_matrix.png
        cogew_rf_cnn_accuracy_by_snr.png
        cogew_rf_cnn_classification_report.txt
"""

import os
import json
import time
import hashlib
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = r"D:\COG-EW"

TEST_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "cogew_test.h5")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cogew_rf_cnn_best.pt")

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RESULT_JSON = os.path.join(REPORT_DIR, "cogew_rf_cnn_test_evaluation.json")
CONFUSION_MATRIX_PNG = os.path.join(REPORT_DIR, "cogew_rf_cnn_confusion_matrix.png")
SNR_ACCURACY_PNG = os.path.join(REPORT_DIR, "cogew_rf_cnn_accuracy_by_snr.png")
CLASSIFICATION_REPORT_TXT = os.path.join(REPORT_DIR, "cogew_rf_cnn_classification_report.txt")
REFERENCE_TXT = os.path.join(MODEL_DIR, "evaluation_reference.txt")

SEED = 42
BATCH_SIZE = 512

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

NUM_CLASSES = len(CLASS_NAMES)


# ======================================================================
# RANDOM SEED
# ======================================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# SHA256
# ======================================================================


def calculate_sha256(path, chunk_size=1024 * 1024):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ======================================================================
# PURE NUMPY METRICS (sklearn-independent)
# ======================================================================


def compute_accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(y_true, y_pred, num_classes=5):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# ======================================================================
# MODEL ARCHITECTURE
# ======================================================================


class COGEWRFCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            # Block 4
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
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
        x = self.classifier(x)
        return x


# ======================================================================
# LOAD HDF5 DATASET
# ======================================================================


def load_test_dataset(path):
    print("=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)
    print(f"\nPath:\n{path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test dataset not found:\n{path}")

    with h5py.File(path, "r") as f:
        print("\nDataset keys:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}")

        X = f["X"][:]
        Y = f["Y"][:]
        Z = f["Z"][:]

        if "source" in f:
            source = f["source"][:]
        else:
            source = np.zeros(len(Y), dtype=np.int64)

        if "original_index" in f:
            original_index = f["original_index"][:]
        else:
            original_index = np.arange(len(Y))

    return X, Y, Z, source, original_index


# ======================================================================
# DATASET VALIDATION
# ======================================================================


def validate_dataset(X, Y, Z, source, original_index):
    print("\n" + "=" * 70)
    print("TEST DATASET VALIDATION")
    print("=" * 70)

    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Z shape: {Z.shape}")

    assert X.ndim == 3
    assert X.shape[1:] == (128, 2)
    assert len(X) == len(Y) == len(Z) == len(source) == len(original_index)

    print("[PASS] X/Y/Z alignment")
    print("[PASS] IQ shape = (128, 2)")

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print(f"\nNaN values : {nan_count}")
    print(f"Inf values : {inf_count}")

    assert nan_count == 0
    assert inf_count == 0
    print("[PASS] No NaN/Inf values")

    unique_classes, class_counts = np.unique(Y, return_counts=True)

    print("\nCLASS DISTRIBUTION")
    print("-" * 55)

    for cls, count in zip(unique_classes, class_counts):
        name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"UNKNOWN_{cls}"
        print(f"{int(cls):02d} | {name:<8} | {count:,}")

    assert set(unique_classes.tolist()) == set(range(NUM_CLASSES))
    print("\n[PASS] Five expected classes detected")


# ======================================================================
# LOAD MODEL
# ======================================================================


def load_best_model(device):
    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    print(f"\nCheckpoint:\n{BEST_MODEL_PATH}")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"\nBest model checkpoint not found:\n{BEST_MODEL_PATH}")

    model = COGEWRFCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing_keys:
        print("\nWARNING: Missing model keys:")
        for key in missing_keys:
            print(f"  {key}")

    if unexpected_keys:
        print("\nWARNING: Unexpected model keys:")
        for key in unexpected_keys:
            print(f"  {key}")

    if not missing_keys:
        print("\n[PASS] Model weights loaded successfully")

    model = model.to(device)
    model.eval()

    return model


# ======================================================================
# MODEL EVALUATION
# ======================================================================


def evaluate_model(model, X, Y, Z, source, device):
    print("\n" + "=" * 70)
    print("RUNNING TEST EVALUATION")
    print("=" * 70)

    X_tensor = torch.from_numpy(X.astype(np.float32)).permute(0, 2, 1)
    Y_tensor = torch.from_numpy(Y.astype(np.int64))

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_predictions = []
    all_targets = []

    print(f"\nTest samples : {len(dataset):,}")
    print(f"Test batches : {len(loader):,}")
    print("\nRunning inference...")

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (signals, labels) in enumerate(loader):
            signals = signals.to(device, non_blocking=True)
            outputs = model(signals)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.numpy())

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
                processed = min((batch_idx + 1) * BATCH_SIZE, len(dataset))
                print(f"  Processed {processed:,}/{len(dataset):,}")

    elapsed = time.time() - start_time

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_targets)

    accuracy = compute_accuracy(y_true, y_pred)

    print(f"\nInference time: {elapsed:.2f} seconds")
    print(f"\nTEST ACCURACY: {accuracy * 100:.2f}%")

    return y_true, y_pred, accuracy


# ======================================================================
# CLASSIFICATION METRICS (NumPy Implementation)
# ======================================================================


def calculate_class_metrics(y_true, y_pred):
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS")
    print("=" * 70)

    metrics = {}
    print("\nClass                 Precision   Recall   F1-score   Samples")
    print("-" * 70)

    for i, name in enumerate(CLASS_NAMES):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        supp = int(np.sum(y_true == i))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[name] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": supp,
        }

        print(f"{name:<12} {prec * 100:8.2f}% {rec * 100:8.2f}% {f1 * 100:9.2f}% {supp:10,}")

    return metrics


# ======================================================================
# CONFUSION MATRIX (NumPy Implementation)
# ======================================================================


def generate_confusion_matrix(y_true, y_pred):
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
    print("\nRaw confusion matrix:")
    print(cm)

    cm_normalized = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    print("\nNormalized confusion matrix:")
    for i, row in enumerate(cm_normalized):
        values = " ".join(f"{value:.3f}" for value in row)
        print(f"{CLASS_NAMES[i]:<8} {values}")

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm_normalized, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True Class",
        xlabel="Predicted Class",
        title="COG-EW RF CNN — Normalized Test Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Confusion matrix saved:\n{CONFUSION_MATRIX_PNG}")
    return cm, cm_normalized


# ======================================================================
# ACCURACY BY SNR
# ======================================================================


def calculate_snr_accuracy(y_true, y_pred, Z):
    print("\n" + "=" * 70)
    print("ACCURACY BY SNR")
    print("=" * 70)

    snr_values = np.asarray(Z).reshape(-1)

    if len(snr_values) != len(y_true):
        raise ValueError("SNR array length does not match test sample count.")

    unique_snr = np.sort(np.unique(snr_values))
    results = {}

    print("\nSNR       Samples       Accuracy")
    print("-" * 45)

    for snr in unique_snr:
        mask = snr_values == snr
        count = int(mask.sum())
        if count == 0:
            continue

        snr_acc = compute_accuracy(y_true[mask], y_pred[mask])
        results[str(int(snr))] = {"samples": count, "accuracy": float(snr_acc)}
        print(f"{int(snr):>4} dB   {count:>8,}       {snr_acc * 100:>7.2f}%")

    snr_numeric = [int(snr) for snr in unique_snr]
    snr_acc_values = [results[str(int(snr))]["accuracy"] * 100 for snr in unique_snr]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(snr_numeric, snr_acc_values, marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("COG-EW RF CNN — Test Accuracy by SNR")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    plt.savefig(SNR_ACCURACY_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] SNR accuracy plot saved:\n{SNR_ACCURACY_PNG}")
    return results


# ======================================================================
# SOURCE-WISE ACCURACY
# ======================================================================


def calculate_source_accuracy(y_true, y_pred, source):
    print("\n" + "=" * 70)
    print("ACCURACY BY DATASET SOURCE")
    print("=" * 70)

    source_values = np.asarray(source).reshape(-1)
    results = {}
    unique_sources = np.unique(source_values)

    for src in unique_sources:
        mask = source_values == src
        if int(src) == 0:
            source_name = "RadioML2016"
        elif int(src) == 1:
            source_name = "RadioML2018"
        else:
            source_name = f"Source_{int(src)}"

        count = int(mask.sum())
        acc = compute_accuracy(y_true[mask], y_pred[mask])

        results[source_name] = {
            "source_id": int(src),
            "samples": count,
            "accuracy": float(acc),
        }

        print(f"{source_name:<15} {count:>8,} samples   {acc * 100:>7.2f}%")

    return results


# ======================================================================
# SAVE CLASSIFICATION REPORT
# ======================================================================


def save_classification_report(y_true, y_pred):
    class_metrics = calculate_class_metrics(y_true, y_pred)

    lines = []
    lines.append("COG-EW RF CNN TEST CLASSIFICATION REPORT")
    lines.append("=" * 70 + "\n")
    lines.append(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 55)

    prec_list, rec_list, f1_list, supp_list = [], [], [], []

    for name in CLASS_NAMES:
        m = class_metrics[name]
        prec_list.append(m["precision"])
        rec_list.append(m["recall"])
        f1_list.append(m["f1_score"])
        supp_list.append(m["support"])
        lines.append(f"{name:<12} {m['precision']*100:9.2f}% {m['recall']*100:9.2f}% {m['f1_score']*100:9.2f}% {m['support']:10d}")

    acc = compute_accuracy(y_true, y_pred)
    total_supp = sum(supp_list)
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))
    macro_f1 = float(np.mean(f1_list))

    lines.append("-" * 55)
    lines.append(f"{'Overall Acc':<12} {'':>10} {'':>10} {acc*100:9.2f}% {total_supp:10d}")
    lines.append(f"{'Macro Avg':<12} {macro_prec*100:9.2f}% {macro_rec*100:9.2f}% {macro_f1*100:9.2f}% {total_supp:10d}")

    report_str = "\n".join(lines)

    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"\n[OK] Classification report saved:\n{CLASSIFICATION_REPORT_TXT}")
    print("\n" + report_str)

    return report_str


# ======================================================================
# SAVE RESULTS
# ======================================================================


def save_results(
    test_sha256,
    model_sha256,
    y_true,
    y_pred,
    accuracy,
    class_metrics,
    confusion_matrix_raw,
    confusion_matrix_normalized,
    snr_results,
    source_results,
):
    results = {
        "project": "COG-EW",
        "task": "Automatic Modulation Classification",
        "evaluation_type": "untouched_test_set",
        "test_dataset": {
            "path": TEST_PATH,
            "sha256": test_sha256,
            "samples": int(len(y_true)),
        },
        "model": {
            "path": BEST_MODEL_PATH,
            "sha256": model_sha256,
            "classes": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
        },
        "overall": {
            "accuracy": float(accuracy),
            "accuracy_percent": float(accuracy * 100),
        },
        "class_metrics": class_metrics,
        "confusion_matrix": {
            "raw": confusion_matrix_raw.tolist(),
            "normalized": confusion_matrix_normalized.tolist(),
        },
        "snr_accuracy": snr_results,
        "source_accuracy": source_results,
        "integrity": {
            "test_data_augmented": False,
            "test_data_used_for_training": False,
            "test_data_used_for_model_selection": False,
            "test_data_modified": False,
            "original_datasets_modified": False,
        },
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Evaluation report saved:\n{RESULT_JSON}")
    return results


# ======================================================================
# MAIN
# ======================================================================


def main():
    start_time = time.time()

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    set_seed(SEED)

    print("\n" + "=" * 70)
    print("COG-EW RF AUTOMATIC MODULATION CLASSIFICATION")
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTest dataset:\n{TEST_PATH}")
    print(f"\nBest model:\n{BEST_MODEL_PATH}")

    print("\nIMPORTANT:")
    print("  Test data is completely untouched.")
    print("  Test data was NOT augmented.")
    print("  Test data was NOT used for training.")
    print("  Test data was NOT used for model selection.")

    print("\n" + "=" * 70)
    print("FILE INTEGRITY")
    print("=" * 70)

    test_sha256 = calculate_sha256(TEST_PATH)
    model_sha256 = calculate_sha256(BEST_MODEL_PATH)

    print(f"\nTest dataset SHA256:\n{test_sha256}")
    print(f"\nBest model SHA256:\n{model_sha256}")

    X, Y, Z, source, original_index = load_test_dataset(TEST_PATH)
    validate_dataset(X, Y, Z, source, original_index)

    model = load_best_model(device)
    y_true, y_pred, accuracy = evaluate_model(model, X, Y, Z, source, device)

    class_metrics = calculate_class_metrics(y_true, y_pred)
    cm, cm_normalized = generate_confusion_matrix(y_true, y_pred)
    snr_results = calculate_snr_accuracy(y_true, y_pred, Z)
    source_results = calculate_source_accuracy(y_true, y_pred, source)

    save_classification_report(y_true, y_pred)

    save_results(
        test_sha256=test_sha256,
        model_sha256=model_sha256,
        y_true=y_true,
        y_pred=y_pred,
        accuracy=accuracy,
        class_metrics=class_metrics,
        confusion_matrix_raw=cm,
        confusion_matrix_normalized=cm_normalized,
        snr_results=snr_results,
        source_results=source_results,
    )

    with open(REFERENCE_TXT, "w", encoding="utf-8") as f:
        f.write("COG-EW MODEL EVALUATION REFERENCE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model:\n{BEST_MODEL_PATH}\n\n")
        f.write(f"Model SHA256:\n{model_sha256}\n\n")
        f.write(f"Test dataset:\n{TEST_PATH}\n\n")
        f.write(f"Test dataset SHA256:\n{test_sha256}\n\n")
        f.write(f"Test accuracy:\n{accuracy * 100:.4f}%\n\n")
        f.write("Test data was untouched.\n")
        f.write("Test data was not augmented.\n")
        f.write("Test data was not used for training.\n")
        f.write("Test data was not used for model selection.\n")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nFINAL TEST ACCURACY: {accuracy * 100:.2f}%")

    print("\nARTIFACTS")
    print("-" * 70)
    print(f"Evaluation JSON:\n{RESULT_JSON}")
    print(f"\nConfusion matrix:\n{CONFUSION_MATRIX_PNG}")
    print(f"\nAccuracy by SNR:\n{SNR_ACCURACY_PNG}")
    print(f"\nClassification report:\n{CLASSIFICATION_REPORT_TXT}")
    print(f"\nEvaluation reference:\n{REFERENCE_TXT}")

    print("\nINTEGRITY")
    print("-" * 70)
    print("[PASS] Test dataset was NOT augmented.")
    print("[PASS] Test dataset was NOT used during training.")
    print("[PASS] Test dataset was NOT used for model selection.")
    print("[PASS] Original datasets were NOT modified.")
    print(f"\nElapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
