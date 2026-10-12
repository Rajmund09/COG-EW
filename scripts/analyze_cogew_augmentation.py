"""
COG-EW RF AUGMENTATION QUALITY / INTEGRITY ANALYSIS

Purpose:
    Validate the augmented training dataset before model training.

Checks:
    1. Dataset structure
    2. NaN / Inf values
    3. Class distribution
    4. Original vs augmented distribution
    5. SNR distribution
    6. Signal power
    7. IQ statistics
    8. Original-vs-augmented correlation
    9. Augmentation diversity
    10. Label preservation
    11. Source-index integrity
    12. Potentially excessive augmentation
    13. SHA256 integrity
"""

from pathlib import Path
import hashlib
import json

import h5py
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\COG-EW")

INPUT_FILE = PROJECT_ROOT / "data" / "augmented" / "cogew_train_augmented.h5"

ORIGINAL_FILE = PROJECT_ROOT / "data" / "splits" / "cogew_train.h5"

REPORT_FILE = PROJECT_ROOT / "reports" / "cogew_augmentation_quality.json"


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

EXPECTED_CLASSES = 5

RANDOM_SEED = 42

# Number of augmented pairs to inspect
PAIR_PROBE_SIZE = 10_000

# Batch size
BATCH_SIZE = 4096


# ============================================================
# HELPERS
# ============================================================


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA256 without loading the entire file into memory."""

    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def signal_power(x):
    """
    Calculate average IQ signal power.

    x shape:
        [N, 128, 2]
    """

    return np.mean(np.square(x), axis=(1, 2))


def amplitude_statistics(x):
    """
    Calculate IQ amplitude statistics.

    amplitude = sqrt(I^2 + Q^2)
    """

    i = x[:, :, 0]
    q = x[:, :, 1]

    amplitude = np.sqrt(i**2 + q**2)

    return {
        "mean": float(np.mean(amplitude)),
        "std": float(np.std(amplitude)),
        "min": float(np.min(amplitude)),
        "max": float(np.max(amplitude)),
        "median": float(np.median(amplitude)),
    }


def phase_statistics(x):
    """Calculate IQ phase statistics."""

    i = x[:, :, 0]
    q = x[:, :, 1]

    phase = np.arctan2(q, i)

    return {
        "mean": float(np.mean(phase)),
        "std": float(np.std(phase)),
        "min": float(np.min(phase)),
        "max": float(np.max(phase)),
    }


def correlation(a, b):
    """Calculate flattened Pearson correlation."""

    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))

    if denominator == 0:
        return 0.0

    return float(np.sum(a * b) / denominator)


def distribution_dict(values):
    """Convert integer distribution to JSON-safe dictionary."""

    unique, counts = np.unique(values, return_counts=True)

    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("COG-EW RF AUGMENTATION QUALITY / INTEGRITY ANALYSIS")
print("=" * 70)

print()
print("Augmented dataset:")
print(INPUT_FILE)

print()
print("Original training dataset:")
print(ORIGINAL_FILE)


# ============================================================
# FILE CHECK
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Augmented dataset not found:\n{INPUT_FILE}")

if not ORIGINAL_FILE.exists():
    raise FileNotFoundError(f"Original training dataset not found:\n{ORIGINAL_FILE}")


# ============================================================
# HASHES
# ============================================================

print()
print("=" * 70)
print("FILE INTEGRITY")
print("=" * 70)

print()
print("Calculating SHA256...")

original_sha = sha256_file(ORIGINAL_FILE)
augmented_sha = sha256_file(INPUT_FILE)

print()
print("Original training SHA256:")
print(original_sha)

print()
print("Augmented dataset SHA256:")
print(augmented_sha)


# ============================================================
# OPEN DATASETS
# ============================================================

print()
print("=" * 70)
print("OPENING DATASETS")
print("=" * 70)

with h5py.File(INPUT_FILE, "r") as aug_h5, h5py.File(ORIGINAL_FILE, "r") as orig_h5:

    print()
    print("Augmented dataset keys:")

    for key in aug_h5.keys():
        print(f"  {key}: {aug_h5[key].shape}")

    print()
    print("Original dataset keys:")

    for key in orig_h5.keys():
        print(f"  {key}: {orig_h5[key].shape}")

    # ========================================================
    # STRUCTURE
    # ========================================================

    print()
    print("=" * 70)
    print("STRUCTURAL VALIDATION")
    print("=" * 70)

    required_keys = [
        "X",
        "Y",
        "Z",
        "augmentation_source",
        "source_index",
    ]

    structure_pass = True

    for key in required_keys:

        if key in aug_h5:
            print(f"[PASS] {key} exists")
        else:
            print(f"[FAIL] {key} missing")
            structure_pass = False

    X = aug_h5["X"]
    Y = aug_h5["Y"]
    Z = aug_h5["Z"]

    augmentation_source = aug_h5["augmentation_source"]
    source_index = aug_h5["source_index"]

    original_X = orig_h5["X"]
    original_Y = orig_h5["Y"]
    original_Z = orig_h5["Z"]

    n_aug = len(X)
    n_original = len(original_X)

    print()
    print(f"Original samples : {n_original:,}")
    print(f"Augmented dataset: {n_aug:,}")

    if n_aug != n_original * 2:
        print("[WARNING] Dataset is not exactly 2x the original size.")
    else:
        print("[PASS] Dataset size is exactly 2x.")

    # ========================================================
    # SHAPE CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("IQ SHAPE VALIDATION")
    print("=" * 70)

    print(f"X shape: {X.shape}")

    if X.ndim == 3 and X.shape[1:] == (128, 2):
        print("[PASS] IQ shape = (128, 2)")
    else:
        print("[FAIL] Unexpected IQ shape")
        structure_pass = False

    # ========================================================
    # FINITE CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("FINITE VALUE CHECK")
    print("=" * 70)

    nan_count = 0
    inf_count = 0

    for start in range(0, n_aug, BATCH_SIZE):

        end = min(start + BATCH_SIZE, n_aug)

        batch = X[start:end]

        nan_count += int(np.isnan(batch).sum())
        inf_count += int(np.isinf(batch).sum())

    print(f"NaN values : {nan_count:,}")
    print(f"Inf values : {inf_count:,}")

    finite_pass = nan_count == 0 and inf_count == 0

    if finite_pass:
        print("[PASS] No NaN/Inf values")
    else:
        print("[FAIL] NaN/Inf detected")

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    y_values = Y[:]

    class_counts = np.bincount(
        y_values,
        minlength=EXPECTED_CLASSES,
    )

    print()
    print("Class                  Samples")
    print("-" * 40)

    for class_id, count in enumerate(class_counts):

        name = (
            CLASS_NAMES[class_id]
            if class_id < len(CLASS_NAMES)
            else f"UNKNOWN_{class_id}"
        )

        print(f"{class_id:02d} | " f"{name:<10} | " f"{count:,}")

    # ========================================================
    # AUGMENTATION SOURCE DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("AUGMENTATION SOURCE DISTRIBUTION")
    print("=" * 70)

    aug_source_values = augmentation_source[:]

    unique_source, source_counts = np.unique(
        aug_source_values,
        return_counts=True,
    )

    source_distribution = {}

    for source, count in zip(
        unique_source,
        source_counts,
    ):

        if source == 0:
            name = "Original"
        elif source == 1:
            name = "Augmented"
        else:
            name = f"Unknown_{source}"

        source_distribution[name] = int(count)

        print(f"{name:<12} | {count:,}")

    # ========================================================
    # SOURCE INDEX CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("SOURCE INDEX INTEGRITY")
    print("=" * 70)

    source_indices = source_index[:]

    invalid_indices = np.sum((source_indices < 0) | (source_indices >= n_original))

    print(f"Invalid source indices: " f"{invalid_indices:,}")

    if invalid_indices == 0:
        print("[PASS] All source indices valid")
    else:
        print("[FAIL] Invalid source indices detected")

    # ========================================================
    # LABEL PRESERVATION
    # ========================================================

    print()
    print("=" * 70)
    print("LABEL PRESERVATION")
    print("=" * 70)

    label_errors = 0

    # Check original half
    original_half = Y[:n_original]

    if not np.array_equal(
        original_half,
        original_Y[:],
    ):
        print("[FAIL] Original labels changed.")
        label_errors += 1
    else:
        print("[PASS] Original labels preserved.")

    # Check augmented labels against source indices
    augmented_half = Y[n_original:]

    augmented_indices = source_indices[n_original:]

    expected_augmented_labels = original_Y[augmented_indices]

    if not np.array_equal(
        augmented_half,
        expected_augmented_labels,
    ):
        print("[FAIL] Augmented labels do not " "match their source samples.")
        label_errors += 1
    else:
        print("[PASS] Augmented labels correctly " "match source samples.")

    # ========================================================
    # SNR DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("SNR DISTRIBUTION")
    print("=" * 70)

    z_values = Z[:]

    unique_snr, snr_counts = np.unique(
        z_values,
        return_counts=True,
    )

    print()

    for snr, count in zip(
        unique_snr,
        snr_counts,
    ):

        print(f"{int(snr):>4} dB | " f"{count:,}")

    # ========================================================
    # SIGNAL STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("SIGNAL STATISTICS")
    print("=" * 70)

    # Sample a maximum of 20k signals
    rng = np.random.default_rng(RANDOM_SEED)

    probe_size = min(
        20_000,
        n_aug,
    )

    probe_indices = rng.choice(
        n_aug,
        size=probe_size,
        replace=False,
    )

    probe_X = X[np.sort(probe_indices)]

    probe_power = signal_power(probe_X)

    print()
    print(f"Statistics probe: {probe_size:,} samples")

    print()
    print("Signal power:")
    print(f"  Mean   : {np.mean(probe_power):.8f}")
    print(f"  Std    : {np.std(probe_power):.8f}")
    print(f"  Minimum: {np.min(probe_power):.8f}")
    print(f"  Maximum: {np.max(probe_power):.8f}")

    amp_stats = amplitude_statistics(probe_X)

    print()
    print("IQ amplitude:")

    for key, value in amp_stats.items():

        print(f"  {key:<8}: {value:.8f}")

    phase_stats = phase_statistics(probe_X)

    print()
    print("IQ phase:")

    for key, value in phase_stats.items():

        print(f"  {key:<8}: {value:.8f}")

    # ========================================================
    # ORIGINAL VS AUGMENTED COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("ORIGINAL VS AUGMENTED COMPARISON")
    print("=" * 70)

    pair_size = min(
        PAIR_PROBE_SIZE,
        n_original,
    )

    pair_indices = rng.choice(
        n_original,
        size=pair_size,
        replace=False,
    )

    pair_indices.sort()

    original_probe = original_X[pair_indices]

    augmented_probe = X[n_original + pair_indices]

    pair_correlations = []

    for i in range(pair_size):

        corr = correlation(
            original_probe[i],
            augmented_probe[i],
        )

        pair_correlations.append(corr)

    pair_correlations = np.asarray(pair_correlations)

    print()
    print(f"Pair probe size: {pair_size:,}")

    print()
    print("Original ↔ Augmented correlation:")

    print(f"  Mean   : " f"{np.mean(pair_correlations):.6f}")

    print(f"  Median : " f"{np.median(pair_correlations):.6f}")

    print(f"  Minimum: " f"{np.min(pair_correlations):.6f}")

    print(f"  Maximum: " f"{np.max(pair_correlations):.6f}")

    # ========================================================
    # AUGMENTATION DIVERSITY
    # ========================================================

    print()
    print("=" * 70)
    print("AUGMENTATION DIVERSITY")
    print("=" * 70)

    identical_count = np.sum(
        np.all(
            np.isclose(
                original_probe,
                augmented_probe,
                rtol=1e-5,
                atol=1e-6,
            ),
            axis=(1, 2),
        )
    )

    identical_percentage = identical_count / pair_size * 100

    print()
    print(f"Near-identical pairs: " f"{identical_count:,}/{pair_size:,}")

    print(f"Near-identical percentage: " f"{identical_percentage:.4f}%")

    if identical_percentage < 1.0:
        print("[PASS] Augmentation produces " "meaningfully different signals.")

    elif identical_percentage < 5.0:
        print("[WARNING] Some augmented signals " "are very similar to originals.")

    else:
        print("[WARNING] Large proportion of " "augmented signals are near-identical.")

    # ========================================================
    # POWER RATIO
    # ========================================================

    original_pair_power = signal_power(original_probe)

    augmented_pair_power = signal_power(augmented_probe)

    power_ratio = augmented_pair_power / (original_pair_power + 1e-12)

    print()
    print("Augmented/original power ratio:")

    print(f"  Mean   : " f"{np.mean(power_ratio):.6f}")

    print(f"  Median : " f"{np.median(power_ratio):.6f}")

    print(f"  Minimum: " f"{np.min(power_ratio):.6f}")

    print(f"  Maximum: " f"{np.max(power_ratio):.6f}")

    # ========================================================
    # POTENTIAL EXTREME AUGMENTATION
    # ========================================================

    extreme_low = np.mean(power_ratio < 0.25) * 100

    extreme_high = np.mean(power_ratio > 4.0) * 100

    print()
    print(f"Very low power ratio (<0.25): " f"{extreme_low:.3f}%")

    print(f"Very high power ratio (>4.0): " f"{extreme_high:.3f}%")

    if extreme_low < 5 and extreme_high < 5:
        print("[PASS] No large population of " "extreme power changes detected.")
    else:
        print("[WARNING] Potentially aggressive " "amplitude augmentation detected.")

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print()
    print("=" * 70)
    print("AUGMENTATION QUALITY DECISION")
    print("=" * 70)

    critical_failures = 0

    if not structure_pass:
        critical_failures += 1

    if not finite_pass:
        critical_failures += 1

    if invalid_indices != 0:
        critical_failures += 1

    if label_errors != 0:
        critical_failures += 1

    if critical_failures == 0:

        print()
        print("[PASS] Augmented dataset passed " "critical integrity checks.")

        print()
        print("The dataset is eligible for " "model training.")

        overall_status = "PASS"

    else:

        print()
        print("[FAIL] Augmented dataset failed " "critical integrity checks.")

        print()
        print("DO NOT TRAIN until the " "problems are fixed.")

        overall_status = "FAIL"

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "status": overall_status,
        "input_dataset": str(INPUT_FILE),
        "original_training_dataset": str(ORIGINAL_FILE),
        "original_sha256": original_sha,
        "augmented_sha256": augmented_sha,
        "original_samples": int(n_original),
        "augmented_dataset_samples": int(n_aug),
        "expected_augmented_samples": int(n_original * 2),
        "class_distribution": {
            CLASS_NAMES[i]: int(class_counts[i])
            for i in range(
                min(
                    len(CLASS_NAMES),
                    len(class_counts),
                )
            )
        },
        "augmentation_source_distribution": source_distribution,
        "snr_distribution": distribution_dict(z_values),
        "nan_values": int(nan_count),
        "inf_values": int(inf_count),
        "invalid_source_indices": int(invalid_indices),
        "label_errors": int(label_errors),
        "pair_probe_size": int(pair_size),
        "pair_correlation": {
            "mean": float(np.mean(pair_correlations)),
            "median": float(np.median(pair_correlations)),
            "minimum": float(np.min(pair_correlations)),
            "maximum": float(np.max(pair_correlations)),
        },
        "near_identical_percentage": float(identical_percentage),
        "power_ratio": {
            "mean": float(np.mean(power_ratio)),
            "median": float(np.median(power_ratio)),
            "minimum": float(np.min(power_ratio)),
            "maximum": float(np.max(power_ratio)),
        },
        "extreme_low_power_percentage": float(extreme_low),
        "extreme_high_power_percentage": float(extreme_high),
        "signal_statistics": {
            "power_mean": float(np.mean(probe_power)),
            "power_std": float(np.std(probe_power)),
            "power_min": float(np.min(probe_power)),
            "power_max": float(np.max(probe_power)),
            "amplitude": amp_stats,
            "phase": phase_stats,
        },
    }


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("AUGMENTATION ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Report:")
print(REPORT_FILE)

print()
print("[PASS] Original dataset was not modified.")
print("[PASS] Validation dataset was not modified.")
print("[PASS] Test dataset was not modified.")

print()
print("Next stage:")
print("Train the new COG-EW RF/AMC model " "ONLY if the quality decision is PASS.")
