from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.models.radioml_resnet import RadioMLResNet
from data.loaders.baseline_loaders import create_baseline_loaders


HDF5_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 24
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 15
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
PATIENCE = 4
SEED = 42


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        signals, labels, _ = batch
        signals = signals.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(signals)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        if training and (batch_idx + 1) % 100 == 0:
            print(
                f"    Batch {batch_idx + 1}/{len(loader)} | Loss: {loss.item():.4f}"
            )

    return total_loss / total, correct / total


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW RADIOML RESNET TRAINING")
    print("=" * 70)

    seed_everything(SEED)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_baseline_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    best_val_accuracy = 0.0
    patience_counter = 0
    history = []
    checkpoint_path = CHECKPOINT_DIR / "radioml_resnet_best.pt"

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        print(f"\nTrain loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc * 100:.2f}%")
        print(f"Learning rate: {current_lr:.7f}")
        print(f"Epoch time: {epoch_time:.1f} sec")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_time,
            }
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": val_acc,
                    "validation_loss": val_loss,
                    "num_classes": NUM_CLASSES,
                    "model_name": "RadioMLResNet",
                },
                str(checkpoint_path),
            )
            print("\n[CHECKPOINT] Best model saved.")

        else:
            patience_counter += 1
            print(f"\nNo improvement. Patience {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("\nEarly stopping.")
                break

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------
    history_path = REPORT_DIR / "radioml_resnet_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # --------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL RESNET RESULT")
    print("=" * 70)

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"\nBest validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print("\n[PASS] RadioML ResNet training completed.")


if __name__ == "__main__":
    main()
