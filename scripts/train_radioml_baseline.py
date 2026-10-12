from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

REPORT_DIR = PROJECT_ROOT / "reports"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

EPOCHS = 10

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PATIENCE = 3


def set_seed(
    seed: int,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(
    logits,
    labels,
):

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (predictions == labels).sum().item()

    total = labels.size(0)

    return correct, total


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    start_time = time.perf_counter()

    for batch_idx, (signals, labels, _) in enumerate(loader):

        signals = signals.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(signals)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        correct, count = calculate_accuracy(
            logits,
            labels,
        )

        total_correct += correct

        total_samples += count

        if (batch_idx + 1) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1:,}/"
                f"{len(loader):,} | "
                f"Loss: "
                f"{loss.item():.4f}"
            )

    elapsed = time.perf_counter() - start_time

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return (
        average_loss,
        accuracy,
        elapsed,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for signals, labels, _ in loader:

        signals = signals.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(signals)

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        correct, count = calculate_accuracy(
            logits,
            labels,
        )

        total_correct += correct

        total_samples += count

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return (
        average_loss,
        accuracy,
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW RADIOML BASELINE TRAINING")
    print("=" * 70)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    print("\nCreating DataLoaders...")

    train_loader, val_loader, test_loader = create_baseline_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train batches: " f"{len(train_loader):,}")

    print(f"Validation batches: " f"{len(val_loader):,}")

    print(f"Test batches: " f"{len(test_loader):,}")

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = RadioMLCNN(num_classes=24).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(f"\nModel parameters: " f"{sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    history = []

    best_val_loss = float("inf")

    patience_counter = 0

    best_checkpoint = CHECKPOINT_DIR / "radioml_cnn_baseline_best.pt"

    print("\nStarting training...")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n" + "-" * 70)

        print(f"Epoch " f"{epoch}/{EPOCHS}")

        train_loss, train_acc, train_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "learning_rate": float(current_lr),
            "train_time_seconds": float(train_time),
        }

        history.append(epoch_result)

        print(f"\nTrain loss: " f"{train_loss:.4f}")

        print(f"Train accuracy: " f"{train_acc * 100:.2f}%")

        print(f"Validation loss: " f"{val_loss:.4f}")

        print(f"Validation accuracy: " f"{val_acc * 100:.2f}%")

        print(f"Learning rate: " f"{current_lr:.6f}")

        # --------------------------------------------------
        # Save best checkpoint
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                    "class_count": 24,
                    "seed": SEED,
                },
                str(best_checkpoint),
            )

            print("\n[CHECKPOINT] Best model saved.")

        else:

            patience_counter += 1

            print(
                f"\nNo improvement. Patience {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            print("\nEarly stopping.")

            break

    # ------------------------------------------------------
    # Save training history
    # ------------------------------------------------------

    history_path = REPORT_DIR / "radioml_baseline_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # ------------------------------------------------------
    # Load best model
    # ------------------------------------------------------

    checkpoint = torch.load(
        str(best_checkpoint),
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n" + "=" * 70)

    print("FINAL BASELINE RESULT")

    print("=" * 70)

    print(f"Test loss: " f"{test_loss:.4f}")

    print(f"Test accuracy: " f"{test_acc * 100:.2f}%")

    print(f"\nBest checkpoint:")

    print(best_checkpoint)

    print(f"\nTraining history:")

    print(history_path)

    print("\n[PASS] Baseline training completed.")


if __name__ == "__main__":
    main()
