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
# PATHS
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

BASELINE_CHECKPOINT = CHECKPOINT_DIR / "radioml_resnet_best.pt"

ROBUST_CHECKPOINT = CHECKPOINT_DIR / "radioml_resnet_robust_best.pt"

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
# DATASET
# ============================================================


def resolve_hdf5_path() -> Path:

    data_root = Path(DATA_ROOT)

    for directory in data_root.glob("*RadioML*"):

        candidate = directory / "GOLD_XYZ_OSC.0001_1024.hdf5"

        if candidate.exists():
            return candidate

    for candidate in data_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"):

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate RadioML HDF5 dataset " f"inside:\n{data_root}"
    )


HDF5_PATH = resolve_hdf5_path()


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 24

BATCH_SIZE = 32

NUM_WORKERS = 0

SEED = 42


# ============================================================
# CONTROLLED DISTURBANCE LEVELS
# ============================================================

INTERFERENCE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.35,
]


# ============================================================
# HELPERS
# ============================================================


def print_header(title: str):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# CONTROLLED INTERFERENCE
# ============================================================


def apply_batch_interference(
    signals: torch.Tensor,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Controlled synthetic disturbance used exclusively
    for defensive robustness evaluation.

    Input:
        [batch, 2, 1024]

    Output:
        [batch, 2, 1024]

    The original input is not modified.
    """

    if strength <= 0.0:
        return signals.clone()

    disturbed = signals.clone()

    batch_size = disturbed.shape[0]

    channels = disturbed.shape[1]

    samples = disturbed.shape[2]

    # --------------------------------------------------------
    # Gaussian disturbance
    # --------------------------------------------------------

    noise = torch.randn(
        disturbed.shape,
        generator=generator,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

    disturbed = disturbed + noise * (strength * 0.50)

    # --------------------------------------------------------
    # Tone-like disturbance
    # --------------------------------------------------------

    time_axis = torch.arange(
        samples,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

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
        channels,
        1,
    )

    disturbed = disturbed + tone * (strength * 0.35)

    # --------------------------------------------------------
    # Controlled impulsive disturbance
    # --------------------------------------------------------

    probability = 0.01 * min(
        1.0,
        strength / 0.10 if strength > 0 else 0.0,
    )

    impulse_mask = (
        torch.rand(
            disturbed.shape,
            generator=generator,
            device=disturbed.device,
            dtype=disturbed.dtype,
        )
        < probability
    )

    impulse_noise = torch.randn(
        disturbed.shape,
        generator=generator,
        device=disturbed.device,
        dtype=disturbed.dtype,
    )

    disturbed = torch.where(
        impulse_mask,
        disturbed + impulse_noise * strength * 2.0,
        disturbed,
    )

    # --------------------------------------------------------
    # Normalize
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
# MODEL LOADING
# ============================================================


def load_model(
    checkpoint_path: Path,
):

    model = RadioMLResNet(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(
    model,
    loader,
    strength: float,
    seed: int,
):

    predictions = []

    labels = []

    confidences = []

    snrs = []

    generator = torch.Generator(device=DEVICE)

    generator.manual_seed(seed)

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

            disturbed = apply_batch_interference(
                signals=signals,
                strength=strength,
                generator=generator,
            )

            logits = model(disturbed)

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

                print(f"      Batch " f"{batch_index:,}/" f"{len(loader):,}")

    predictions = np.concatenate(predictions)

    labels = np.concatenate(labels)

    confidences = np.concatenate(confidences)

    snrs = np.concatenate(snrs)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "average_confidence": float(np.mean(confidences)),
        "average_confidence_percent": float(np.mean(confidences) * 100),
        "predictions": predictions,
        "labels": labels,
        "snrs": snrs,
    }


# ============================================================
# SNR ANALYSIS
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

        accuracy = float(np.mean(predictions[mask] == labels[mask]))

        output[str(int(snr))] = {
            "accuracy_percent": (accuracy * 100),
            "samples": int(np.sum(mask)),
        }

    return output


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

    print_header("COG-EW BASELINE VS ROBUST RADIO ML EVALUATION")

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

    print_header("CHECKING REQUIRED FILES")

    print(f"[OK] Dataset:\n{HDF5_PATH}")

    if not BASELINE_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Baseline checkpoint missing:\n" f"{BASELINE_CHECKPOINT}"
        )

    print("[OK] Baseline checkpoint found.")

    if not ROBUST_CHECKPOINT.exists():

        raise FileNotFoundError(f"Robust checkpoint missing:\n" f"{ROBUST_CHECKPOINT}")

    print("[OK] Robust checkpoint found.")

    test_split_path = PROCESSED_DIR / "baseline_test_split.npy"

    if not test_split_path.exists():

        raise FileNotFoundError(f"Test split missing:\n" f"{test_split_path}")

    test_indices = np.load(test_split_path)

    if len(test_indices) != 9000:

        raise RuntimeError("Expected exactly 9,000 " "test samples.")

    print(f"[OK] Test split: " f"{len(test_indices):,} samples")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print_header("CREATING TEST DATALOADER")

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
    # Load models
    # --------------------------------------------------------

    print_header("LOADING MODELS")

    baseline_model, baseline_checkpoint = load_model(BASELINE_CHECKPOINT)

    robust_model, robust_checkpoint = load_model(ROBUST_CHECKPOINT)

    print("[OK] Baseline model loaded.")

    print("[OK] Robust model loaded.")

    if "epoch" in baseline_checkpoint:

        print(
            "Baseline checkpoint epoch:",
            baseline_checkpoint["epoch"],
        )

    if "epoch" in robust_checkpoint:

        print(
            "Robust checkpoint epoch:",
            robust_checkpoint["epoch"],
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print_header("CONTROLLED ROBUSTNESS COMPARISON")

    comparison = []

    detailed_results = {}

    for level_index, strength in enumerate(INTERFERENCE_LEVELS):

        print()
        print("-" * 70)

        print(f"Interference level: " f"{strength:.2f}")

        # IMPORTANT:
        # Both models receive identical
        # deterministic perturbations.

        print("\n  Baseline evaluation:")

        baseline_result = evaluate_model(
            model=baseline_model,
            loader=test_loader,
            strength=strength,
            seed=SEED + level_index,
        )

        print("  Robust evaluation:")

        robust_result = evaluate_model(
            model=robust_model,
            loader=test_loader,
            strength=strength,
            seed=SEED + level_index,
        )

        baseline_accuracy = baseline_result["accuracy_percent"]

        robust_accuracy = robust_result["accuracy_percent"]

        baseline_confidence = baseline_result["average_confidence_percent"]

        robust_confidence = robust_result["average_confidence_percent"]

        accuracy_gain = robust_accuracy - baseline_accuracy

        comparison.append(
            {
                "strength": float(strength),
                "baseline_accuracy_percent": baseline_accuracy,
                "robust_accuracy_percent": robust_accuracy,
                "robust_accuracy_gain_pp": accuracy_gain,
                "baseline_confidence_percent": baseline_confidence,
                "robust_confidence_percent": robust_confidence,
                "robust_confidence_gain_pp": (robust_confidence - baseline_confidence),
            }
        )

        detailed_results[f"{strength:.2f}"] = {
            "baseline": {
                "accuracy_percent": baseline_accuracy,
                "confidence_percent": baseline_confidence,
                "snr": calculate_snr_results(baseline_result),
            },
            "robust": {
                "accuracy_percent": robust_accuracy,
                "confidence_percent": robust_confidence,
                "snr": calculate_snr_results(robust_result),
            },
        }

        print()
        print(f"  Baseline accuracy: " f"{baseline_accuracy:.2f}%")

        print(f"  Robust accuracy:   " f"{robust_accuracy:.2f}%")

        print(f"  Robust gain:       " f"{accuracy_gain:+.2f} pp")

        print(f"  Baseline confidence: " f"{baseline_confidence:.2f}%")

        print(f"  Robust confidence:   " f"{robust_confidence:.2f}%")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header("BASELINE VS ROBUST SUMMARY")

    print("\nLevel | Baseline | Robust | Gain")

    print("-" * 55)

    for item in comparison:

        print(
            f"{item['strength']:.2f}  | "
            f"{item['baseline_accuracy_percent']:7.2f}% | "
            f"{item['robust_accuracy_percent']:6.2f}% | "
            f"{item['robust_accuracy_gain_pp']:+6.2f} pp"
        )

    # --------------------------------------------------------
    # Best robustness gain
    # --------------------------------------------------------

    best_gain = max(
        comparison,
        key=lambda item: item["robust_accuracy_gain_pp"],
    )

    worst_gain = min(
        comparison,
        key=lambda item: item["robust_accuracy_gain_pp"],
    )

    print("\nBest robust-model gain:")

    print(
        f"  Level {best_gain['strength']:.2f}: "
        f"{best_gain['robust_accuracy_gain_pp']:+.2f} pp"
    )

    print("\nWorst robust-model change:")

    print(
        f"  Level {worst_gain['strength']:.2f}: "
        f"{worst_gain['robust_accuracy_gain_pp']:+.2f} pp"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print_header("GENERATING COMPARISON FIGURE")

    x = [item["strength"] for item in comparison]

    baseline_y = [item["baseline_accuracy_percent"] for item in comparison]

    robust_y = [item["robust_accuracy_percent"] for item in comparison]

    figure_path = FIGURE_DIR / "radioml_baseline_vs_robust.png"

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        baseline_y,
        marker="o",
        linewidth=2,
        label="Baseline ResNet",
    )

    plt.plot(
        x,
        robust_y,
        marker="s",
        linewidth=2,
        label="Robust ResNet",
    )

    plt.xlabel("Controlled synthetic interference strength")

    plt.ylabel("Classification accuracy (%)")

    plt.title("COG-EW RadioML Baseline vs Robust Model")

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
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[OK] Figure saved:\n" f"{figure_path}")

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "experiment": ("Baseline vs Robust " "RadioML classifier"),
        "dataset": ("RadioML 2018.01A"),
        "test_samples": int(len(test_indices)),
        "seed": SEED,
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "robust_checkpoint": str(ROBUST_CHECKPOINT),
        "interference_levels": (INTERFERENCE_LEVELS),
        "comparison": comparison,
        "detailed_results": detailed_results,
        "figure": str(figure_path),
    }

    report_path = REPORT_DIR / "radioml_baseline_vs_robust.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(f"[OK] Report saved:\n" f"{report_path}")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_header("BASELINE VS ROBUST EVALUATION COMPLETE")

    print("The baseline checkpoint was not modified.")

    print("The robust checkpoint was not modified.")

    print("The original RadioML dataset was not modified.")

    print("\nNext decision should be based on " "the measured robustness gains.")


if __name__ == "__main__":
    main()
