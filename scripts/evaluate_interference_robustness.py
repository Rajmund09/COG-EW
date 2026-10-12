from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score

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
# PATH RESOLUTION
# ============================================================


def resolve_hdf5_path() -> Path:

    data_root = Path(DATA_ROOT)

    # First try directories containing RadioML.
    for directory in data_root.glob("*RadioML*"):

        candidate = directory / "GOLD_XYZ_OSC.0001_1024.hdf5"

        if candidate.exists():
            return candidate

    # Then search recursively.
    for candidate in data_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"):

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate GOLD_XYZ_OSC.0001_1024.hdf5 "
        f"inside DATA_ROOT:\n{data_root}"
    )


HDF5_PATH = resolve_hdf5_path()


# ============================================================
# PROJECT PATHS
# ============================================================

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

SEED = 42


# ============================================================
# ROBUSTNESS CONFIGURATION
# ============================================================

# These are controlled simulation strengths.
#
# They are signal-domain perturbation parameters,
# NOT real RF transmitter settings.

INTERFERENCE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.35,
]


# ============================================================
# CLASS NAMES
# ============================================================

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


def apply_batch_interference(
    signals: torch.Tensor,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Apply controlled synthetic disturbances to a batch.

    Input:
        signals -> [batch, 2, 1024]

    Output:
        disturbed signals -> [batch, 2, 1024]

    The original tensor is not modified.
    """

    if strength <= 0.0:
        return signals.clone()

    disturbed = signals.clone()

    batch_size = disturbed.shape[0]

    number_of_channels = disturbed.shape[1]

    number_of_samples = disturbed.shape[2]

    # --------------------------------------------------------
    # 1. Additive Gaussian disturbance
    # --------------------------------------------------------

    noise = torch.randn(
        disturbed.shape,
        generator=generator,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

    disturbed = disturbed + noise * (strength * 0.50)

    # --------------------------------------------------------
    # 2. Tone-like synthetic disturbance
    # --------------------------------------------------------

    time_axis = torch.arange(
        number_of_samples,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

    # Use deterministic frequency/phase values generated
    # for every batch.
    frequencies = 0.03 + 0.22 * torch.rand(
        batch_size,
        generator=generator,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

    phases = (
        2.0
        * torch.pi
        * torch.rand(
            batch_size,
            generator=generator,
            device=disturbed.device,
            dtype=disturbed.dtype,
        )
    )

    tone = torch.sin(
        (2.0 * torch.pi * frequencies.unsqueeze(1) * time_axis.unsqueeze(0))
        + phases.unsqueeze(1)
    )

    tone = tone.unsqueeze(1)

    tone = tone.repeat(
        1,
        number_of_channels,
        1,
    )

    disturbed = disturbed + tone * (strength * 0.35)

    # --------------------------------------------------------
    # 3. Controlled impulsive disturbance
    # --------------------------------------------------------

    impulse_mask = torch.rand(
        disturbed.shape,
        generator=generator,
        device=disturbed.device,
        dtype=disturbed.dtype,
    ) < (
        0.01
        * min(
            1.0,
            strength / 0.10 if strength > 0 else 0.0,
        )
    )

    impulse_noise = torch.randn(
        disturbed.shape,
        generator=generator,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

    disturbed = torch.where(
        impulse_mask,
        disturbed + (impulse_noise * strength * 2.0),
        disturbed,
    )

    # --------------------------------------------------------
    # Re-normalize exactly like the dataset
    # --------------------------------------------------------

    mean = disturbed.mean(
        dim=2,
        keepdim=True,
    )

    std = disturbed.std(
        dim=2,
        keepdim=True,
        unbiased=False,
    )

    disturbed = (disturbed - mean) / (std + 1e-8)

    return disturbed


# ============================================================
# EVALUATION
# ============================================================


def evaluate_level(
    model: torch.nn.Module,
    loader,
    strength: float,
    level_seed: int,
):

    predictions = []

    labels = []

    confidences = []

    snrs = []

    generator = torch.Generator(device=DEVICE)

    generator.manual_seed(level_seed)

    total_batches = len(loader)

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

            disturbed_signals = apply_batch_interference(
                signals=signals,
                strength=strength,
                generator=generator,
            )

            logits = model(disturbed_signals)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1,
            )

            predictions.append(predicted.cpu().numpy())

            labels.append(batch_labels.numpy())

            confidences.append(confidence.cpu().numpy())

            snrs.append(batch_snrs.numpy())

            if batch_index % 50 == 0:

                print(
                    f"    Processed " f"{batch_index:,}/" f"{total_batches:,} batches"
                )

    predictions = np.concatenate(predictions)

    labels = np.concatenate(labels)

    confidences = np.concatenate(confidences)

    snrs = np.concatenate(snrs)

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    average_confidence = float(np.mean(confidences))

    return {
        "strength": float(strength),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "average_confidence": average_confidence,
        "average_confidence_percent": (average_confidence * 100.0),
        "samples": int(len(labels)),
        "predictions": predictions,
        "labels": labels,
        "snrs": snrs,
    }


# ============================================================
# SNR ROBUSTNESS
# ============================================================


def calculate_snr_results(
    result: dict,
):

    predictions = result["predictions"]

    labels = result["labels"]

    snrs = result["snrs"]

    output = {}

    for snr in sorted(np.unique(snrs)):

        mask = snrs == snr

        if not np.any(mask):
            continue

        accuracy = float(np.mean(predictions[mask] == labels[mask]))

        output[str(int(snr))] = {
            "accuracy": accuracy,
            "accuracy_percent": (accuracy * 100.0),
            "samples": int(np.sum(mask)),
        }

    return output


# ============================================================
# ROBUSTNESS PLOT
# ============================================================


def create_robustness_plot(
    results: list[dict],
):

    x = [item["strength"] for item in results]

    accuracy = [item["accuracy_percent"] for item in results]

    confidence = [item["average_confidence_percent"] for item in results]

    # --------------------------------------------------------
    # Accuracy plot
    # --------------------------------------------------------

    accuracy_path = FIGURE_DIR / "rf_interference_robustness.png"

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        accuracy,
        marker="o",
        linewidth=2,
        label="Accuracy",
    )

    plt.plot(
        x,
        confidence,
        marker="s",
        linewidth=2,
        label="Average confidence",
    )

    plt.xlabel("Synthetic interference strength")

    plt.ylabel("Percentage (%)")

    plt.title("COG-EW RadioML ResNet Robustness " "to Controlled RF Disturbances")

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

    plt.savefig(
        accuracy_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return accuracy_path


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

    print_header("COG-EW RADIO ML INTERFERENCE ROBUSTNESS EVALUATION")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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
    # Verify files
    # --------------------------------------------------------

    print("\nChecking required files...")

    if not HDF5_PATH.exists():

        raise FileNotFoundError(f"RadioML HDF5 dataset not found:\n" f"{HDF5_PATH}")

    print("[OK] RadioML dataset found.")

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(f"ResNet checkpoint not found:\n" f"{CHECKPOINT_PATH}")

    print("[OK] ResNet checkpoint found.")

    test_split_path = PROCESSED_DIR / "baseline_test_split.npy"

    if not test_split_path.exists():

        raise FileNotFoundError("Baseline test split not found:\n" f"{test_split_path}")

    test_indices = np.load(test_split_path)

    print("[OK] Baseline test split found.")

    print(f"Baseline test samples: " f"{len(test_indices):,}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print("\nCreating baseline DataLoader...")

    (
        _train_loader,
        _val_loader,
        test_loader,
    ) = create_baseline_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Test samples: " f"{len(test_loader.dataset):,}")

    print(f"Test batches: " f"{len(test_loader):,}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating RadioML ResNet...")

    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model parameters: " f"{parameter_count:,}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading baseline checkpoint...")

    checkpoint = torch.load(
        str(CHECKPOINT_PATH),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("[OK] Baseline checkpoint loaded.")

    if "epoch" in checkpoint:

        print(f"Checkpoint epoch: " f"{checkpoint['epoch']}")

    if "validation_accuracy" in checkpoint:

        print(
            "Checkpoint validation accuracy: "
            f"{checkpoint['validation_accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Evaluate levels
    # --------------------------------------------------------

    print_header("CONTROLLED INTERFERENCE RESULTS")

    results = []

    for level_index, strength in enumerate(INTERFERENCE_LEVELS):

        print("\n" + "-" * 70)

        print(f"Interference level: " f"{strength:.2f}")

        result = evaluate_level(
            model=model,
            loader=test_loader,
            strength=strength,
            level_seed=(SEED + level_index),
        )

        results.append(result)

        print(f"\nAccuracy: " f"{result['accuracy_percent']:.2f}%")

        print(f"Average confidence: " f"{result['average_confidence_percent']:.2f}%")

        print(f"Samples: " f"{result['samples']:,}")

    # --------------------------------------------------------
    # Baseline reference
    # --------------------------------------------------------

    clean_result = results[0]

    baseline_accuracy = clean_result["accuracy_percent"]

    print_header("ROBUSTNESS SUMMARY")

    print(f"Clean baseline accuracy: " f"{baseline_accuracy:.2f}%")

    summary_results = []

    for result in results:

        degradation = baseline_accuracy - result["accuracy_percent"]

        relative_degradation = 0.0

        if baseline_accuracy > 0:

            relative_degradation = degradation / baseline_accuracy * 100.0

        summary_results.append(
            {
                "strength": result["strength"],
                "accuracy_percent": result["accuracy_percent"],
                "average_confidence_percent": result["average_confidence_percent"],
                "accuracy_drop_percentage_points": degradation,
                "relative_accuracy_drop_percent": relative_degradation,
                "samples": result["samples"],
            }
        )

        print(
            f"Level {result['strength']:.2f}"
            f" | Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
            f" | Drop: "
            f"{degradation:.2f} pp"
        )

    # --------------------------------------------------------
    # SNR results
    # --------------------------------------------------------

    print_header("SNR ROBUSTNESS ANALYSIS")

    snr_results = {}

    for result in results:

        strength_key = f"{result['strength']:.2f}"

        snr_results[strength_key] = calculate_snr_results(result)

        print(f"\nInterference level " f"{strength_key}:")

        for snr, values in snr_results[strength_key].items():

            print(
                f"  SNR {int(snr):>3} dB: "
                f"{values['accuracy_percent']:6.2f}% "
                f"({values['samples']:,})"
            )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print("\nGenerating robustness plot...")

    robustness_plot = create_robustness_plot(results)

    print(robustness_plot)

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "experiment": ("RadioML ResNet " "Controlled Interference Robustness"),
        "dataset": "RadioML 2018.01A",
        "model": "RadioMLResNet",
        "checkpoint": str(CHECKPOINT_PATH),
        "evaluation_split": ("baseline_test_split.npy"),
        "test_samples": int(len(test_indices)),
        "num_classes": NUM_CLASSES,
        "seed": SEED,
        "interference_levels": (INTERFERENCE_LEVELS),
        "baseline_accuracy_percent": (baseline_accuracy),
        "results": summary_results,
        "snr_results": snr_results,
        "figure": str(robustness_plot),
    }

    report_path = REPORT_DIR / "radioml_interference_robustness.json"

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

    print_header("INTERFERENCE ROBUSTNESS EVALUATION COMPLETE")

    print(f"Baseline accuracy: " f"{baseline_accuracy:.2f}%")

    print(f"Highest tested interference: " f"{INTERFERENCE_LEVELS[-1]:.2f}")

    print(f"Final tested accuracy: " f"{summary_results[-1]['accuracy_percent']:.2f}%")

    print("\nReport:")

    print(report_path)

    print("\nFigure:")

    print(robustness_plot)

    print(
        "\n[PASS] Existing baseline checkpoint "
        "and baseline test split were preserved."
    )


if __name__ == "__main__":
    main()
