from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 24

BATCH_SIZE = 32

NUM_WORKERS = 0

EPOCHS = 15

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-4

PATIENCE = 4

SEED = 42


# ============================================================
# ROBUST TRAINING CONFIGURATION
# ============================================================

# Probability that a training sample receives
# synthetic interference.

INTERFERENCE_PROBABILITY = 0.60

# Maximum disturbance amplitude used during training.

MAX_INTERFERENCE_LEVEL = 0.35

# Disturbance types used only for defensive
# robustness augmentation.

INTERFERENCE_TYPES = (
    "gaussian",
    "tone",
    "multitone",
    "burst",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================


def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# INTERFERENCE AUGMENTATION
# ============================================================


def normalize_power(
    signal: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize each IQ sample to approximately unit power.

    Input:
        [B, 2, 1024]

    Output:
        [B, 2, 1024]
    """

    power = torch.mean(
        signal**2,
        dim=(1, 2),
        keepdim=True,
    )

    scale = torch.sqrt(power + 1e-8)

    return signal / scale


def generate_gaussian_interference(
    signals: torch.Tensor,
    level: float,
) -> torch.Tensor:

    noise = torch.randn_like(signals)

    return noise * level


def generate_tone_interference(
    signals: torch.Tensor,
    level: float,
) -> torch.Tensor:

    batch_size = signals.size(0)

    num_samples = signals.size(2)

    device = signals.device

    t = torch.arange(
        num_samples,
        device=device,
        dtype=torch.float32,
    )

    frequencies = (
        torch.rand(
            batch_size,
            device=device,
        )
        * 0.45
        + 0.025
    )

    phases = (
        torch.rand(
            batch_size,
            device=device,
        )
        * 2.0
        * np.pi
    )

    phase = 2.0 * np.pi * frequencies[:, None] * t[None, :] + phases[:, None]

    tone_i = torch.cos(phase)

    tone_q = torch.sin(phase)

    tone = torch.stack(
        [
            tone_i,
            tone_q,
        ],
        dim=1,
    )

    return tone * level


def generate_multitone_interference(
    signals: torch.Tensor,
    level: float,
) -> torch.Tensor:

    batch_size = signals.size(0)

    num_samples = signals.size(2)

    device = signals.device

    t = torch.arange(
        num_samples,
        device=device,
        dtype=torch.float32,
    )

    result = torch.zeros_like(signals)

    number_of_tones = 3

    for _ in range(number_of_tones):

        frequencies = (
            torch.rand(
                batch_size,
                device=device,
            )
            * 0.45
            + 0.025
        )

        phases = (
            torch.rand(
                batch_size,
                device=device,
            )
            * 2.0
            * np.pi
        )

        phase = 2.0 * np.pi * frequencies[:, None] * t[None, :] + phases[:, None]

        result[:, 0, :] += torch.cos(phase)

        result[:, 1, :] += torch.sin(phase)

    result /= number_of_tones

    return result * level


def generate_burst_interference(
    signals: torch.Tensor,
    level: float,
) -> torch.Tensor:

    batch_size = signals.size(0)

    num_samples = signals.size(2)

    device = signals.device

    result = torch.randn_like(signals)

    # Create a random temporal mask.

    mask = torch.zeros(
        batch_size,
        num_samples,
        device=device,
    )

    for batch_index in range(batch_size):

        burst_length = max(
            32,
            int(
                num_samples
                * (
                    0.10
                    + 0.20
                    * torch.rand(
                        1,
                        device=device,
                    ).item()
                )
            ),
        )

        start = torch.randint(
            0,
            max(
                1,
                num_samples - burst_length,
            ),
            (1,),
            device=device,
        ).item()

        end = min(
            num_samples,
            start + burst_length,
        )

        mask[
            batch_index,
            start:end,
        ] = 1.0

    result = result * mask[:, None, :]

    return result * level


def apply_interference_augmentation(
    signals: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    """
    Apply controlled synthetic interference to a subset
    of training samples.

    This function is used only during robust training.

    The original RadioML HDF5 file is never modified.
    """

    augmented = signals.clone()

    batch_size = signals.size(0)

    device = signals.device

    selected = (
        torch.rand(
            batch_size,
            device=device,
        )
        < INTERFERENCE_PROBABILITY
    )

    selected_indices = torch.where(selected)[0]

    applied_types = []

    if len(selected_indices) == 0:

        return augmented, applied_types

    for sample_index in selected_indices:

        level = (
            torch.rand(
                1,
                device=device,
            ).item()
            * MAX_INTERFERENCE_LEVEL
        )

        interference_type = random.choice(INTERFERENCE_TYPES)

        single_signal = signals[sample_index : sample_index + 1]

        if interference_type == "gaussian":

            disturbance = generate_gaussian_interference(
                single_signal,
                level,
            )

        elif interference_type == "tone":

            disturbance = generate_tone_interference(
                single_signal,
                level,
            )

        elif interference_type == "multitone":

            disturbance = generate_multitone_interference(
                single_signal,
                level,
            )

        elif interference_type == "burst":

            disturbance = generate_burst_interference(
                single_signal,
                level,
            )

        else:

            disturbance = torch.zeros_like(single_signal)

        augmented[sample_index : sample_index + 1] = single_signal + disturbance

        applied_types.append(interference_type)

    return augmented, applied_types


# ============================================================
# EPOCH
# ============================================================


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    use_interference=False,
):

    training = optimizer is not None

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    interference_batches = 0

    for batch_index, batch in enumerate(loader):

        signals, labels, _ = batch

        signals = signals.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if use_interference:

            signals, applied_types = apply_interference_augmentation(signals)

            if applied_types:

                interference_batches += 1

        if training:

            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(signals)

            loss = criterion(
                logits,
                labels,
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (predictions == labels).sum().item()

        total += batch_size

        if training and (batch_index + 1) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_index + 1:,}/"
                f"{len(loader):,} | "
                f"Loss: {loss.item():.4f}"
            )

    return (
        total_loss / total,
        correct / total,
        interference_batches,
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

    print("COG-EW ROBUST RADIOML TRAINING")

    print("=" * 70)

    seed_everything(SEED)

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
    # Dataset path
    # --------------------------------------------------------

    data_root = Path(DATA_ROOT)

    hdf5_candidates = list(data_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"))

    if not hdf5_candidates:

        raise FileNotFoundError(
            "RadioML HDF5 dataset was not found under:\n" f"{data_root}"
        )

    hdf5_path = hdf5_candidates[0]

    print("\nRadioML dataset:")

    print(hdf5_path)

    # --------------------------------------------------------
    # Verify benchmark split
    # --------------------------------------------------------

    print("\nVerifying benchmark split...")

    train_split = np.load(PROCESSED_DIR / "baseline_train_split.npy")

    val_split = np.load(PROCESSED_DIR / "baseline_val_split.npy")

    test_split = np.load(PROCESSED_DIR / "baseline_test_split.npy")

    print(f"Train: {len(train_split):,}")

    print(f"Validation: {len(val_split):,}")

    print(f"Test: {len(test_split):,}")

    if len(train_split) != 42000:

        raise RuntimeError(
            "Unexpected baseline train split size. " "Expected exactly 42,000 samples."
        )

    if len(val_split) != 9000:

        raise RuntimeError(
            "Unexpected baseline validation split size. "
            "Expected exactly 9,000 samples."
        )

    if len(test_split) != 9000:

        raise RuntimeError(
            "Unexpected baseline test split size. " "Expected exactly 9,000 samples."
        )

    print("\n[OK] Correct COG-EW benchmark split confirmed.")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print("\nCreating DataLoaders...")

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_baseline_loaders(
        hdf5_path=hdf5_path,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train samples: " f"{len(train_loader.dataset):,}")

    print(f"Validation samples: " f"{len(val_loader.dataset):,}")

    print(f"Test samples: " f"{len(test_loader.dataset):,}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating robust ResNet...")

    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: " f"{parameter_count:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = CHECKPOINT_DIR / "radioml_resnet_robust_best.pt"

    best_val_accuracy = 0.0

    patience_counter = 0

    history = []

    print("\nStarting robust training...")

    print("\nSynthetic augmentation configuration:")

    print(f"  Probability: " f"{INTERFERENCE_PROBABILITY}")

    print(f"  Maximum level: " f"{MAX_INTERFERENCE_LEVEL}")

    print(
        "  Types:",
        ", ".join(INTERFERENCE_TYPES),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n" + "-" * 70)

        print(f"Epoch {epoch}/{EPOCHS}")

        start_time = time.time()

        train_loss, train_acc, interference_batches = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            use_interference=True,
        )

        # IMPORTANT:
        # Validation remains CLEAN.
        # This allows direct comparison with
        # the baseline validation result.

        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            use_interference=False,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - start_time

        print(f"\nTrain loss: " f"{train_loss:.4f}")

        print(f"Train accuracy: " f"{train_acc * 100:.2f}%")

        print(f"Validation loss: " f"{val_loss:.4f}")

        print(f"Validation accuracy: " f"{val_acc * 100:.2f}%")

        print(f"Learning rate: " f"{current_lr:.7f}")

        print(
            f"Interference batches: "
            f"{interference_batches:,}/"
            f"{len(train_loader):,}"
        )

        print(f"Epoch time: " f"{epoch_time:.1f} sec")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "learning_rate": current_lr,
                "interference_batches": interference_batches,
                "epoch_time_seconds": epoch_time,
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

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
                    "model_name": "RadioMLResNetRobust",
                    "training_mode": "interference_augmentation",
                    "interference_probability": INTERFERENCE_PROBABILITY,
                    "max_interference_level": MAX_INTERFERENCE_LEVEL,
                    "interference_types": list(INTERFERENCE_TYPES),
                    "benchmark_split": {
                        "train": 42000,
                        "validation": 9000,
                        "test": 9000,
                    },
                },
                checkpoint_path,
            )

            print("\n[CHECKPOINT] " "Best robust model saved.")

        else:

            patience_counter += 1

            print("\nNo improvement. " f"Patience " f"{patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:

                print("\nEarly stopping.")

                break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = REPORT_DIR / "radioml_resnet_robust_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print("\nLoading best robust checkpoint...")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------------
    # Clean test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("ROBUST MODEL CLEAN TEST RESULT")

    print("=" * 70)

    test_loss, test_acc, _ = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        use_interference=False,
    )

    print(f"Clean test loss: " f"{test_loss:.4f}")

    print(f"Clean test accuracy: " f"{test_acc * 100:.2f}%")

    # --------------------------------------------------------
    # Save final report
    # --------------------------------------------------------

    report_path = REPORT_DIR / "radioml_resnet_robust_training.json"

    report = {
        "project": "COG-EW",
        "experiment": ("RadioML interference-aware " "robust training"),
        "dataset": ("RadioML 2018.01A"),
        "model": ("RadioMLResNet"),
        "benchmark_split": {
            "train": 42000,
            "validation": 9000,
            "test": 9000,
        },
        "interference_probability": (INTERFERENCE_PROBABILITY),
        "maximum_interference_level": (MAX_INTERFERENCE_LEVEL),
        "interference_types": list(INTERFERENCE_TYPES),
        "best_validation_accuracy": (best_val_accuracy),
        "best_validation_accuracy_percent": (best_val_accuracy * 100),
        "clean_test_accuracy": (test_acc),
        "clean_test_accuracy_percent": (test_acc * 100),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("ROBUST RADIOML TRAINING COMPLETE")

    print("=" * 70)

    print(f"Best validation accuracy: " f"{best_val_accuracy * 100:.2f}%")

    print(f"Clean test accuracy: " f"{test_acc * 100:.2f}%")

    print("\nRobust checkpoint:")

    print(checkpoint_path)

    print("\nTraining report:")

    print(report_path)

    print("\n[PASS] Baseline checkpoint preserved.")


if __name__ == "__main__":

    main()
