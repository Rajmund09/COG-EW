from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT DATASET
# ============================================================

from data.loaders.radioml_dataset import RadioMLDataset


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

OUTPUT_DIR = PROJECT_ROOT / "data" / "simulation" / "rf_interference"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

OUTPUT_DIR.mkdir(
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

# ============================================================
# RADIO ML DATASET PATH
# ============================================================

DATASETS_ROOT = Path(r"D:\Datasets")


def resolve_radioml_hdf5() -> Path:
    """
    Automatically locate the RadioML 2018.01A HDF5 dataset.

    This avoids hardcoding the Unicode folder name.
    """

    if not DATASETS_ROOT.exists():
        raise FileNotFoundError(f"Datasets directory not found:\n{DATASETS_ROOT}")

    # Look specifically for RadioML 2018.01A directories.
    candidates = [
        p
        for p in DATASETS_ROOT.iterdir()
        if p.is_dir() and "RadioML 2018.01A" in p.name
    ]

    if not candidates:
        raise FileNotFoundError(
            "RadioML 2018.01A directory was not found.\n\n"
            f"Searched in:\n{DATASETS_ROOT}"
        )

    # Search inside the matching directory for the HDF5 file.
    for dataset_dir in candidates:

        hdf5_path = dataset_dir / "GOLD_XYZ_OSC.0001_1024.hdf5"

        if hdf5_path.exists():
            return hdf5_path

        # Also support nested dataset layouts.
        matches = list(dataset_dir.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"))

        if matches:
            return matches[0]

    raise FileNotFoundError(
        "RadioML 2018.01A directory was found, "
        "but GOLD_XYZ_OSC.0001_1024.hdf5 was not found.\n\n"
        f"Dataset directories:\n" + "\n".join(str(p) for p in candidates)
    )


HDF5_PATH = resolve_radioml_hdf5()

SAMPLE_INDEX = 100

SEED = 42

NUM_SAMPLES_TO_SAVE = 8


# ============================================================
# CONTROLLED INTERFERENCE LEVELS
# ============================================================

# These are simulation parameters only.
#
# They represent progressively stronger synthetic
# disturbances applied to an IQ signal.
#
# They are NOT transmitter or RF hardware settings.

INTERFERENCE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.35,
]


# ============================================================
# RANDOM SEED
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# SIGNAL HELPERS
# ============================================================


def normalize_signal(signal: np.ndarray) -> np.ndarray:

    signal = signal.astype(np.float32)

    mean = np.mean(
        signal,
        axis=1,
        keepdims=True,
    )

    std = np.std(
        signal,
        axis=1,
        keepdims=True,
    )

    return (signal - mean) / (std + 1e-8)


def calculate_signal_power(
    signal: np.ndarray,
) -> float:

    return float(np.mean(signal**2))


# ============================================================
# SYNTHETIC INTERFERENCE GENERATORS
# ============================================================


def additive_noise(
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:

    noise = rng.normal(
        0.0,
        strength,
        size=signal.shape,
    ).astype(np.float32)

    return signal + noise


def tone_like_interference(
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:

    channels, samples = signal.shape

    t = np.arange(
        samples,
        dtype=np.float32,
    )

    frequency = rng.uniform(
        0.03,
        0.25,
    )

    phase = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    tone = np.sin(2.0 * np.pi * frequency * t + phase)

    tone = tone.astype(np.float32)

    interference = np.zeros_like(
        signal,
        dtype=np.float32,
    )

    interference[0] = tone
    interference[1] = tone

    return signal + strength * interference


def impulsive_interference(
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:

    result = signal.copy()

    channels, samples = result.shape

    number_of_events = max(
        1,
        int(samples * 0.01),
    )

    positions = rng.integers(
        0,
        samples,
        size=number_of_events,
    )

    for position in positions:

        width = int(
            rng.integers(
                1,
                8,
            )
        )

        start = max(
            0,
            position - width,
        )

        end = min(
            samples,
            position + width,
        )

        pulse = rng.normal(
            0.0,
            strength,
            size=(end - start,),
        ).astype(np.float32)

        result[
            :,
            start:end,
        ] += pulse

    return result


def apply_controlled_interference(
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:

    if strength <= 0.0:

        return signal.copy()

    result = signal.copy()

    # Combine several synthetic disturbance
    # mechanisms to create a robustness scenario.

    result = additive_noise(
        result,
        strength * 0.50,
    )

    result = tone_like_interference(
        result,
        strength * 0.35,
    )

    result = impulsive_interference(
        result,
        strength * 0.25,
    )

    return result


# ============================================================
# SAVE SIGNAL
# ============================================================


def save_signal(
    signal: np.ndarray,
    filename: str,
):

    output_path = OUTPUT_DIR / filename

    np.save(
        output_path,
        signal.astype(np.float32),
    )

    return output_path


# ============================================================
# PLOT
# ============================================================


def create_comparison_plot(
    clean_signal: np.ndarray,
    examples: list[tuple[float, np.ndarray]],
):

    fig, axes = plt.subplots(
        len(examples) + 1,
        1,
        figsize=(14, 12),
    )

    if len(examples) == 0:

        axes = [axes]

    # --------------------------------------------------------
    # Clean signal
    # --------------------------------------------------------

    clean_amplitude = np.sqrt(clean_signal[0] ** 2 + clean_signal[1] ** 2)

    axes[0].plot(clean_amplitude[:512])

    axes[0].set_title("Clean RadioML IQ Signal")

    axes[0].set_ylabel("Amplitude")

    axes[0].grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Interference examples
    # --------------------------------------------------------

    for index, (
        strength,
        signal,
    ) in enumerate(
        examples,
        start=1,
    ):

        amplitude = np.sqrt(signal[0] ** 2 + signal[1] ** 2)

        axes[index].plot(amplitude[:512])

        axes[index].set_title(f"Synthetic Interference " f"Strength = {strength:.2f}")

        axes[index].set_ylabel("Amplitude")

        axes[index].grid(
            True,
            alpha=0.3,
        )

    axes[-1].set_xlabel("Sample")

    plt.tight_layout()

    output_path = FIGURE_DIR / "rf_interference_examples.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================


def main():

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW CONTROLLED RF INTERFERENCE SIMULATION")
    print("=" * 70)

    print(
        "\nPurpose:"
        "\nGenerate controlled synthetic RF disturbances"
        "\nfor robustness testing of the RadioML perception model."
    )

    print("\nThis stage does NOT modify the original RadioML dataset.")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not HDF5_PATH.exists():

        raise FileNotFoundError(
            "\nRadioML HDF5 dataset not found:\n"
            f"{HDF5_PATH}\n\n"
            "Update HDF5_PATH in this script "
            "to your actual dataset location."
        )

    print(f"\nDataset:\n{HDF5_PATH}")

    # --------------------------------------------------------
    # Load one sample
    # --------------------------------------------------------

    dataset = RadioMLDataset(
        hdf5_path=HDF5_PATH,
        indices_path=(PROCESSED_DIR / "baseline_test_split.npy"),
        metadata_path=(PROCESSED_DIR / "radioml_index.npz"),
        normalize=True,
    )

    signal, label, snr = dataset[SAMPLE_INDEX]

    signal = signal.numpy()

    print(
        f"\nSelected sample:"
        f"\n  Dataset index: {SAMPLE_INDEX}"
        f"\n  Class ID: {int(label)}"
        f"\n  SNR: {int(snr)} dB"
        f"\n  Shape: {signal.shape}"
    )

    clean_power = calculate_signal_power(signal)

    print(f"\nClean signal power: " f"{clean_power:.6f}")

    # --------------------------------------------------------
    # Generate examples
    # --------------------------------------------------------

    examples = []

    print("\nGenerating controlled interference examples...")

    for strength in INTERFERENCE_LEVELS:

        disturbed = apply_controlled_interference(
            signal,
            strength,
        )

        disturbed = normalize_signal(disturbed)

        filename = f"sample_{SAMPLE_INDEX:06d}" f"_level_{strength:.2f}.npy"

        output_path = save_signal(
            disturbed,
            filename,
        )

        examples.append(
            (
                strength,
                disturbed,
            )
        )

        print(f"  Level {strength:.2f}" f" -> {output_path.name}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_path = create_comparison_plot(
        signal,
        examples,
    )

    print("\nGenerated figure:")

    print(plot_path)

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "project": "COG-EW",
        "stage": "controlled_rf_interference_simulation",
        "dataset": "RadioML 2018.01A",
        "sample_index": SAMPLE_INDEX,
        "class_id": int(label),
        "snr_db": int(snr),
        "signal_shape": list(signal.shape),
        "seed": SEED,
        "interference_levels": INTERFERENCE_LEVELS,
        "description": (
            "Synthetic signal-domain disturbances " "for ML robustness evaluation."
        ),
    }

    metadata_path = OUTPUT_DIR / "simulation_metadata.json"

    import json

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    dataset.close()

    print("\nMetadata:")

    print(metadata_path)

    print("\n" + "=" * 70)

    print("RF INTERFERENCE SIMULATION COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
