"""
======================================================================
COG-EW RF AUTOMATIC MODULATION CLASSIFICATION
1D CNN TRAINING PIPELINE (PyTorch)
======================================================================

Purpose:
    Train a 5-class RF Automatic Modulation Classification (AMC) model
    using the leakage-safe, augmented COG-EW training dataset in PyTorch.

Classes:
    0 -> 8PSK
    1 -> BPSK
    2 -> QPSK
    3 -> 16QAM
    4 -> 64QAM

Training data:
    data/augmented/cogew_train_augmented.h5

Validation data:
    data/splits/cogew_validation.h5

Test data:
    NOT USED during training.

Important:
    - Original datasets are never modified.
    - Validation dataset is never augmented.
    - Test dataset is never loaded.
    - No test accuracy is calculated here.
    - Best model is selected using validation accuracy/loss.
    - Training history and audit information are saved.
======================================================================
"""

from __future__ import annotations

import json
import hashlib
import random
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(r"D:\COG-EW")

TRAIN_PATH = PROJECT_ROOT / "data" / "augmented" / "cogew_train_augmented.h5"
VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "cogew_validation.h5"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "cogew_rf_cnn_best.pt"
FINAL_MODEL_PATH = MODEL_DIR / "cogew_rf_cnn_final.pt"

HISTORY_PATH = REPORT_DIR / "cogew_rf_cnn_training_history.json"
AUDIT_PATH = REPORT_DIR / "cogew_rf_cnn_training_audit.json"

CURVES_ACCURACY_PATH = OUTPUT_DIR / "cogew_rf_cnn_accuracy.png"
CURVES_LOSS_PATH = OUTPUT_DIR / "cogew_rf_cnn_loss.png"


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

NUM_CLASSES = 5
IQ_LENGTH = 128
IQ_CHANNELS = 2

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

BATCH_SIZE = 512
EPOCHS = 60

LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 4

MIN_LR = 1e-6

WEIGHT_DECAY = 1e-4

LOAD_BATCH_SIZE = 4096


# ======================================================================
# REPRODUCIBILITY & DEVICE
# ======================================================================


def set_reproducibility(seed: int = 42) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ======================================================================
# DATASET INSPECTION
# ======================================================================


def inspect_h5_dataset(path: Path, dataset_name: str) -> dict:
    print()
    print("=" * 70)
    print(f"VALIDATING {dataset_name}")
    print("=" * 70)
    print(f"Path: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Required dataset does not exist:\n{path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    with h5py.File(path, "r") as h5:
        print("\nDataset keys:")
        for key in h5.keys():
            print(f"  {key}: {h5[key].shape}")

        for key in ["X", "Y", "Z"]:
            if key not in h5:
                raise RuntimeError(f"Required dataset '{key}' is missing from {path}")

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]

        print(f"\nDataset structure:\nX: {X.shape}\nY: {Y.shape}\nZ: {Z.shape}")

        if len(X.shape) != 3 or X.shape[1] != IQ_LENGTH or X.shape[2] != IQ_CHANNELS:
            raise RuntimeError(f"Expected X shape (*, {IQ_LENGTH}, {IQ_CHANNELS}), got {X.shape}")

        if Y.shape[0] != X.shape[0] or Z.shape[0] != X.shape[0]:
            raise RuntimeError("X/Y/Z sample count mismatch.")

        print("\n[PASS] X/Y/Z alignment")
        print("[PASS] IQ shape = (128, 2)")

        sample_count = X.shape[0]
        labels = np.asarray(Y[:], dtype=np.int64)
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        print("\nClass distribution:")
        class_distribution = {}
        for label, count in zip(unique_labels, label_counts):
            label_int = int(label)
            if label_int < 0 or label_int >= NUM_CLASSES:
                raise RuntimeError(f"Invalid class label: {label_int}")
            class_name = CLASS_NAMES[label_int]
            class_distribution[class_name] = int(count)
            print(f"  {label_int:02d} | {class_name:<8} | {count:,}")

        if set(unique_labels.tolist()) != set(range(NUM_CLASSES)):
            raise RuntimeError("Dataset does not contain exactly 5 expected classes.")

        print("\n[PASS] Five expected classes detected")

        print("\nChecking finite values...")
        nan_count, inf_count = 0, 0
        for start in range(0, sample_count, LOAD_BATCH_SIZE):
            end = min(start + LOAD_BATCH_SIZE, sample_count)
            batch = np.asarray(X[start:end], dtype=np.float32)
            nan_count += int(np.isnan(batch).sum())
            inf_count += int(np.isinf(batch).sum())

        print(f"NaN values : {nan_count:,}")
        print(f"Inf values : {inf_count:,}")
        if nan_count != 0 or inf_count != 0:
            raise RuntimeError("Dataset contains NaN or Inf values.")
        print("[PASS] No NaN/Inf values")

        probe_size = min(20_000, sample_count)
        probe = np.asarray(X[:probe_size], dtype=np.float32)
        signal_mean, signal_std = float(np.mean(probe)), float(np.std(probe))
        signal_min, signal_max = float(np.min(probe)), float(np.max(probe))

        print("\nSignal statistics probe:")
        print(f"  Samples: {probe_size:,}")
        print(f"  Mean   : {signal_mean:.8f}")
        print(f"  Std    : {signal_std:.8f}")
        print(f"  Min    : {signal_min:.8f}")
        print(f"  Max    : {signal_max:.8f}")

    return {
        "path": str(path),
        "sample_count": int(sample_count),
        "x_shape": list(X.shape),
        "y_shape": list(Y.shape),
        "z_shape": list(Z.shape),
        "class_distribution": class_distribution,
        "signal_statistics_probe": {
            "samples": probe_size,
            "mean": signal_mean,
            "std": signal_std,
            "min": signal_min,
            "max": signal_max,
        },
        "sha256": calculate_sha256(path),
    }


# ======================================================================
# PYTORCH HDF5 DATASET & LOADERS
# ======================================================================


class HDF5Dataset(Dataset):
    """
    Fast in-memory PyTorch Dataset for COG-EW HDF5 signals.
    Loads dataset tensors into RAM (~286 MB for 280k samples),
    converting IQ shape from (N, 128, 2) to (N, 2, 128).
    """

    def __init__(self, path: Path):
        self.path = path
        print(f"Loading {path.name} into RAM...")
        with h5py.File(self.path, "r") as h5:
            X = np.asarray(h5["X"][:], dtype=np.float32)
            Y = np.asarray(h5["Y"][:], dtype=np.int64)

        # Transpose from (N, 128, 2) to (N, 2, 128)
        X_transposed = np.transpose(X, (0, 2, 1))

        self.x_data = torch.from_numpy(X_transposed).float()
        self.y_data = torch.from_numpy(Y).long()
        print(f"Loaded {len(self.y_data):,} samples into RAM memory successfully.")

    def __len__(self) -> int:
        return len(self.y_data)

    def __getitem__(self, idx: int):
        return self.x_data[idx], self.y_data[idx]


# ======================================================================
# PYTORCH MODEL DEFINITION
# ======================================================================


class COGEWRFCNN(nn.Module):
    """
    1D CNN Architecture for RF Modulation Classification.

    Input:
        (batch, 2, 128)

    Output:
        Logits of shape (batch, 5)
    """

    def __init__(self, num_classes: int = 5):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================================================================
# TRAINING UTILITIES
# ======================================================================


def save_training_curves(history_data: dict) -> dict:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history_data["loss"]) + 1)

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history_data["accuracy"]], label="Training Accuracy")
    plt.plot(epochs, [acc * 100 for acc in history_data["val_accuracy"]], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("COG-EW RF CNN - Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_ACCURACY_PATH, dpi=160)
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_data["loss"], label="Training Loss")
    plt.plot(epochs, history_data["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("COG-EW RF CNN - Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_LOSS_PATH, dpi=160)
    plt.close()

    return {
        "accuracy_curve": str(CURVES_ACCURACY_PATH),
        "loss_curve": str(CURVES_LOSS_PATH),
    }


# ======================================================================
# MAIN TRAINING LOOP
# ======================================================================


def main():
    start_time = time.time()
    device = set_reproducibility(SEED)

    print()
    print("=" * 70)
    print("COG-EW RF AUTOMATIC MODULATION CLASSIFICATION")
    print("1D CNN TRAINING (PyTorch)")
    print("=" * 70)

    print(f"\nDevice                 : {device}")
    if device.type == "cuda":
        print(f"GPU Model              : {torch.cuda.get_device_name(0)}")
    print(f"Random seed            : {SEED}")
    print(f"Input shape            : (2, {IQ_LENGTH}) [Channels, Time]")
    print(f"Number of classes      : {NUM_CLASSES}")
    print(f"Classes                : {', '.join(CLASS_NAMES)}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Maximum epochs         : {EPOCHS}")
    print(f"Learning rate          : {LEARNING_RATE}")
    print(f"Early stopping         : {EARLY_STOPPING_PATIENCE} epochs")
    print(f"LR reduction patience  : {LR_REDUCTION_PATIENCE} epochs")

    print("\nIMPORTANT:")
    print("  Validation data is NOT augmented.")
    print("  Test data is NOT loaded.")
    print("  Original datasets are NOT modified.\n")

    train_info = inspect_h5_dataset(TRAIN_PATH, "AUGMENTED TRAINING DATA")
    validation_info = inspect_h5_dataset(VALIDATION_PATH, "VALIDATION DATA")

    print("\n" + "=" * 70)
    print("CREATING PYTORCH DATA LOADERS")
    print("=" * 70)

    train_dataset = HDF5Dataset(TRAIN_PATH)
    val_dataset = HDF5Dataset(VALIDATION_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training samples   : {len(train_dataset):,}")
    print(f"Validation samples : {len(val_dataset):,}")
    print(f"Training batches   : {len(train_loader):,}")
    print(f"Validation batches : {len(val_loader):,}")

    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = COGEWRFCNN(num_classes=NUM_CLASSES).to(device)

    # Print summary
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_REDUCTION_PATIENCE, min_lr=MIN_LR, verbose=True
    )

    print("\n" + "=" * 70)
    print("STARTING MODEL TRAINING")
    print("=" * 70)

    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += x_batch.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += x_batch.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        scheduler.step(epoch_val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} [{epoch_time:.1f}s] - "
            f"loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc*100:.2f}% - "
            f"val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc*100:.2f}% - lr: {current_lr:.1e}"
        )

        # Checkpoint best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved best model checkpoint to {BEST_MODEL_PATH.name} (val_acc: {epoch_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[EARLY STOP] No validation accuracy improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

    # Save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n[OK] Final model saved to {FINAL_MODEL_PATH}")

    # Save training history
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    curve_paths = save_training_curves(history)

    best_epoch_index = int(np.argmax(history["val_accuracy"]))
    best_epoch = best_epoch_index + 1
    best_val_accuracy = float(history["val_accuracy"][best_epoch_index])
    best_val_loss = float(history["val_loss"][best_epoch_index])

    elapsed_seconds = time.time() - start_time

    audit = {
        "project": "COG-EW",
        "experiment": {
            "name": "RF Automatic Modulation Classification CNN (PyTorch)",
            "model": "COGEWRFCNN",
            "task": "5-class Automatic Modulation Classification",
            "framework": f"PyTorch {torch.__version__}",
        },
        "timestamp": datetime.now().isoformat(),
        "random_seed": SEED,
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "input": {
            "iq_length": IQ_LENGTH,
            "iq_channels": IQ_CHANNELS,
            "shape": [IQ_CHANNELS, IQ_LENGTH],
        },
        "training": {
            "dataset": str(TRAIN_PATH),
            "samples": len(train_dataset),
            "batch_size": BATCH_SIZE,
            "maximum_epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "lr_reduction_patience": LR_REDUCTION_PATIENCE,
            "weight_decay": WEIGHT_DECAY,
        },
        "validation": {
            "dataset": str(VALIDATION_PATH),
            "samples": len(val_dataset),
        },
        "test_usage": {
            "used_during_training": False,
            "used_during_validation": False,
            "test_dataset_loaded": False,
        },
        "source_dataset_integrity": {
            "training_sha256": train_info["sha256"],
            "validation_sha256": validation_info["sha256"],
        },
        "model": {
            "name": model.__class__.__name__,
            "parameter_count": param_count,
        },
        "results": {
            "epochs_completed": len(history["loss"]),
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_val_accuracy,
            "best_validation_loss": best_val_loss,
            "final_training_accuracy": float(history["accuracy"][-1]),
            "final_validation_accuracy": float(history["val_accuracy"][-1]),
            "final_training_loss": float(history["loss"][-1]),
            "final_validation_loss": float(history["val_loss"][-1]),
        },
        "artifacts": {
            "best_model": str(BEST_MODEL_PATH),
            "final_model": str(FINAL_MODEL_PATH),
            "history": str(HISTORY_PATH),
            "accuracy_curve": curve_paths["accuracy_curve"],
            "loss_curve": curve_paths["loss_curve"],
        },
        "integrity": {
            "source_datasets_modified": False,
            "validation_augmented": False,
            "test_augmented": False,
            "test_used_for_training": False,
            "test_used_for_model_selection": False,
        },
        "elapsed_seconds": elapsed_seconds,
    }

    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nBEST VALIDATION RESULT")
    print("-" * 70)
    print(f"Best epoch            : {best_epoch}")
    print(f"Best validation loss  : {best_val_loss:.6f}")
    print(f"Best validation acc.  : {best_val_accuracy * 100:.2f}%")

    print("\nFINAL EPOCH RESULT")
    print("-" * 70)
    print(f"Training loss         : {history['loss'][-1]:.6f}")
    print(f"Training accuracy     : {history['accuracy'][-1] * 100:.2f}%")
    print(f"Validation loss       : {history['val_loss'][-1]:.6f}")
    print(f"Validation accuracy   : {history['val_accuracy'][-1] * 100:.2f}%")

    print("\nMODEL ARTIFACTS")
    print("-" * 70)
    print(f"Best model       : {BEST_MODEL_PATH}")
    print(f"Final model      : {FINAL_MODEL_PATH}")
    print(f"Training history : {HISTORY_PATH}")
    print(f"Training audit   : {AUDIT_PATH}")

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)
    print("[PASS] Training dataset was used.")
    print("[PASS] Validation dataset was used.")
    print("[PASS] Test dataset was NOT loaded.")
    print("[PASS] Test accuracy was NOT calculated.")
    print("[PASS] Original datasets were NOT modified.")

    print("\nNEXT STAGE:")
    print("Evaluate the selected best model on the untouched COG-EW test dataset.")
    print(f"\nElapsed time: {elapsed_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    main()
