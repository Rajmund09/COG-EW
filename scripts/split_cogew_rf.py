"""
COG-EW
Leakage-Safe Train / Validation / Test Split

Input:
    data/processed/cogew_common_rf.h5

Output:
    data/splits/cogew_train.h5
    data/splits/cogew_validation.h5
    data/splits/cogew_test.h5

Important:
    - Does NOT modify source datasets.
    - Does NOT perform augmentation.
    - Does NOT train a model.
    - Splits BEFORE augmentation.
    - Uses deterministic random seed.
    - Stratifies by class + source + SNR.
"""

from pathlib import Path
import json
import hashlib

import h5py
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cogew_common_rf.h5"

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_FILE = OUTPUT_DIR / "cogew_train.h5"
VAL_FILE = OUTPUT_DIR / "cogew_validation.h5"
TEST_FILE = OUTPUT_DIR / "cogew_test.h5"

REPORT_FILE = PROJECT_ROOT / "reports" / "cogew_split_audit.json"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPECTED_CLASSES = {
    0: "8PSK",
    1: "BPSK",
    2: "QPSK",
    3: "16QAM",
    4: "64QAM",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def log(message=""):
    print(message)


def sha256_array(arr):
    """
    Create a hash from an array.

    Used for leakage verification.
    """
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def create_stratum_labels(y, z, source):
    """
    Create a stratification key using:

        class + SNR + source

    Example:

        class=2, SNR=10, source=RadioML2018

    becomes:

        2|10|RadioML2018
    """

    labels = []

    for cls, snr, src in zip(y, z, source):
        labels.append(f"{int(cls)}|{int(snr)}|{src}")

    return np.asarray(labels)


def split_indices_stratified(
    y,
    z,
    source,
    seed=42,
):
    """
    Deterministic stratified split.

    Each class/SNR/source group is divided into:

        70% train
        15% validation
        15% test
    """

    rng = np.random.default_rng(seed)

    strata = create_stratum_labels(y, z, source)

    unique_strata = np.unique(strata)

    train_indices = []
    val_indices = []
    test_indices = []

    log()
    log("=" * 70)
    log("STRATIFIED SPLITTING")
    log("=" * 70)

    log(f"Number of strata: {len(unique_strata)}")

    for counter, stratum in enumerate(unique_strata, start=1):

        indices = np.where(strata == stratum)[0]

        indices = indices.copy()

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        # Make sure all samples are assigned.
        n_test = n - n_train - n_val

        train_part = indices[:n_train]

        val_part = indices[n_train : n_train + n_val]

        test_part = indices[n_train + n_val :]

        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)

        if counter % 50 == 0 or counter == len(unique_strata):
            log(f"  Processed {counter}/{len(unique_strata)} strata")

    train_indices = np.asarray(train_indices, dtype=np.int64)

    val_indices = np.asarray(val_indices, dtype=np.int64)

    test_indices = np.asarray(test_indices, dtype=np.int64)

    # Shuffle each final split independently.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (train_indices, val_indices, test_indices)


# ============================================================
# DATASET VALIDATION
# ============================================================


def validate_input(h5):

    required = ["X", "Y", "Z", "source"]

    for key in required:

        if key not in h5:

            raise RuntimeError(
                f"Required dataset '{key}' " f"not found in merged dataset."
            )

    X = h5["X"]
    Y = h5["Y"]
    Z = h5["Z"]
    source = h5["source"]

    n = X.shape[0]

    if Y.shape[0] != n:
        raise RuntimeError("X/Y alignment failure.")

    if Z.shape[0] != n:
        raise RuntimeError("X/Z alignment failure.")

    if source.shape[0] != n:
        raise RuntimeError("X/source alignment failure.")

    if X.ndim != 3:
        raise RuntimeError(f"Unexpected X dimensions: {X.shape}")

    if X.shape[1:] != (128, 2):
        raise RuntimeError(f"Expected X shape (*,128,2), " f"got {X.shape}")

    y_unique = np.unique(Y[:])

    expected = np.array(list(EXPECTED_CLASSES.keys()))

    if not np.array_equal(np.sort(y_unique), expected):
        raise RuntimeError(f"Unexpected class labels: {y_unique}")

    log("[PASS] X/Y/Z/source alignment")
    log("[PASS] IQ shape = (128, 2)")
    log("[PASS] Five expected classes detected")

    return n


# ============================================================
# WRITE SPLIT
# ============================================================


def write_split(output_file, input_h5, indices, split_name):

    log()
    log(f"Writing {split_name} split...")
    log(f"Samples: {len(indices):,}")

    X_source = input_h5["X"]
    Y_source = input_h5["Y"]
    Z_source = input_h5["Z"]
    source_source = input_h5["source"]

    n = len(indices)
    batch_size = 4096

    X = np.empty((n, X_source.shape[1], X_source.shape[2]), dtype=X_source.dtype)
    Y = np.empty(n, dtype=Y_source.dtype)
    Z = np.empty(n, dtype=Z_source.dtype)
    source = np.empty(n, dtype=source_source.dtype)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        sorted_order = np.argsort(batch_idx)
        sorted_indices = batch_idx[sorted_order]

        X_batch_sorted = np.asarray(X_source[sorted_indices])
        Y_batch_sorted = np.asarray(Y_source[sorted_indices])
        Z_batch_sorted = np.asarray(Z_source[sorted_indices])
        source_batch_sorted = np.asarray(source_source[sorted_indices])

        undo_order = np.argsort(sorted_order)

        X[start:end] = X_batch_sorted[undo_order]
        Y[start:end] = Y_batch_sorted[undo_order]
        Z[start:end] = Z_batch_sorted[undo_order]
        source[start:end] = source_batch_sorted[undo_order]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as h5:

        h5.create_dataset(
            "X", data=X, compression="gzip", compression_opts=4, chunks=True
        )

        h5.create_dataset("Y", data=Y, compression="gzip", compression_opts=4)

        h5.create_dataset("Z", data=Z, compression="gzip", compression_opts=4)

        # Preserve source information.
        string_dtype = h5py.string_dtype(encoding="utf-8")

        h5.create_dataset("source", data=source.astype(str).astype(object), dtype=string_dtype)

        h5.create_dataset(
            "original_index", data=indices, compression="gzip", compression_opts=4
        )

        h5.attrs["project"] = "COG-EW"

        h5.attrs["split"] = split_name

        h5.attrs["random_seed"] = RANDOM_SEED

        h5.attrs["description"] = "Leakage-safe split created before augmentation."

        h5.attrs["train_ratio"] = TRAIN_RATIO
        h5.attrs["validation_ratio"] = VAL_RATIO
        h5.attrs["test_ratio"] = TEST_RATIO

    log(f"[OK] {output_file}")


# ============================================================
# SPLIT STATISTICS
# ============================================================


def calculate_statistics(h5_file):

    with h5py.File(h5_file, "r") as h5:

        y = h5["Y"][:]
        z = h5["Z"][:]
        source = h5["source"][:]

    source = np.asarray(
        [s.decode() if isinstance(s, bytes) else str(s) for s in source]
    )

    statistics = {
        "samples": int(len(y)),
        "classes": {},
        "snr": {},
        "sources": {},
        "class_source": {},
    }

    # Class distribution
    for class_id, class_name in EXPECTED_CLASSES.items():

        count = int(np.sum(y == class_id))

        statistics["classes"][class_name] = count

    # SNR distribution
    for snr in np.unique(z):

        statistics["snr"][str(int(snr))] = int(np.sum(z == snr))

    # Source distribution
    for src in np.unique(source):

        statistics["sources"][src] = int(np.sum(source == src))

    # Class + source
    for class_id, class_name in EXPECTED_CLASSES.items():

        statistics["class_source"][class_name] = {}

        for src in np.unique(source):

            count = int(np.sum((y == class_id) & (source == src)))

            statistics["class_source"][class_name][src] = count

    return statistics


# ============================================================
# LEAKAGE CHECK
# ============================================================


def leakage_check(input_h5, train_indices, val_indices, test_indices):

    log()
    log("=" * 70)
    log("LEAKAGE CHECK")
    log("=" * 70)

    # --------------------------------------------------------
    # Index overlap
    # --------------------------------------------------------

    train_set = set(train_indices.tolist())

    val_set = set(val_indices.tolist())

    test_set = set(test_indices.tolist())

    train_val_overlap = train_set & val_set

    train_test_overlap = train_set & test_set

    val_test_overlap = val_set & test_set

    if train_val_overlap:
        raise RuntimeError("Train/validation index leakage detected.")

    if train_test_overlap:
        raise RuntimeError("Train/test index leakage detected.")

    if val_test_overlap:
        raise RuntimeError("Validation/test index leakage detected.")

    log("[PASS] No train/validation index overlap")

    log("[PASS] No train/test index overlap")

    log("[PASS] No validation/test index overlap")

    # --------------------------------------------------------
    # Full coverage
    # --------------------------------------------------------

    n = input_h5["X"].shape[0]

    combined = np.concatenate([train_indices, val_indices, test_indices])

    if len(combined) != n:

        raise RuntimeError("Split coverage does not equal " "original dataset size.")

    if len(np.unique(combined)) != n:

        raise RuntimeError("Duplicate original indices detected.")

    log(f"[PASS] All {n:,} original samples " f"assigned exactly once")

    # --------------------------------------------------------
    # Content hashes
    #
    # We check a sample from each split.
    # --------------------------------------------------------

    log()
    log("Checking sample content hashes...")

    sample_count = min(100, len(train_indices), len(val_indices), len(test_indices))

    train_hashes = set()
    val_hashes = set()
    test_hashes = set()

    for idx in train_indices[:sample_count]:

        arr = input_h5["X"][int(idx)]

        train_hashes.add(sha256_array(arr))

    for idx in val_indices[:sample_count]:

        arr = input_h5["X"][int(idx)]

        val_hashes.add(sha256_array(arr))

    for idx in test_indices[:sample_count]:

        arr = input_h5["X"][int(idx)]

        test_hashes.add(sha256_array(arr))

    if train_hashes & val_hashes:
        raise RuntimeError("Possible train/validation content leakage.")

    if train_hashes & test_hashes:
        raise RuntimeError("Possible train/test content leakage.")

    if val_hashes & test_hashes:
        raise RuntimeError("Possible validation/test content leakage.")

    log("[PASS] Sample content hash check")


# ============================================================
# MAIN
# ============================================================


def main():

    log("=" * 70)
    log("COG-EW LEAKAGE-SAFE DATASET SPLIT")
    log("=" * 70)

    log()
    log(f"Input dataset:")
    log(INPUT_FILE)

    log()
    log("Split configuration:")
    log(f"  Train      : {TRAIN_RATIO:.0%}")
    log(f"  Validation : {VAL_RATIO:.0%}")
    log(f"  Test       : {TEST_RATIO:.0%}")
    log(f"  Seed       : {RANDOM_SEED}")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(f"Input dataset not found:\n{INPUT_FILE}")

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:

        raise RuntimeError("Split ratios must sum to 1.0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Open source dataset
    # --------------------------------------------------------

    with h5py.File(INPUT_FILE, "r") as input_h5:

        log()
        log("=" * 70)
        log("VALIDATING MERGED DATASET")
        log("=" * 70)

        n = validate_input(input_h5)

        log()
        log(f"Total samples: {n:,}")

        y = input_h5["Y"][:]
        z = input_h5["Z"][:]
        source = input_h5["source"][:]

        source = np.asarray(
            [s.decode() if isinstance(s, bytes) else str(s) for s in source]
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        train_indices, val_indices, test_indices = split_indices_stratified(
            y=y, z=z, source=source, seed=RANDOM_SEED
        )

        log()
        log("=" * 70)
        log("SPLIT SIZE")
        log("=" * 70)

        log(f"Train      : {len(train_indices):,} " f"({len(train_indices)/n:.2%})")

        log(f"Validation : {len(val_indices):,} " f"({len(val_indices)/n:.2%})")

        log(f"Test       : {len(test_indices):,} " f"({len(test_indices)/n:.2%})")

        # ----------------------------------------------------
        # Leakage checks
        # ----------------------------------------------------

        leakage_check(input_h5, train_indices, val_indices, test_indices)

        # ----------------------------------------------------
        # Write datasets
        # ----------------------------------------------------

        write_split(TRAIN_FILE, input_h5, train_indices, "train")

        write_split(VAL_FILE, input_h5, val_indices, "validation")

        write_split(TEST_FILE, input_h5, test_indices, "test")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    log()
    log("=" * 70)
    log("SPLIT DISTRIBUTION")
    log("=" * 70)

    train_stats = calculate_statistics(TRAIN_FILE)

    val_stats = calculate_statistics(VAL_FILE)

    test_stats = calculate_statistics(TEST_FILE)

    log()
    log("CLASS DISTRIBUTION")
    log("-" * 70)

    log(f"{'Class':<12}" f"{'Train':>12}" f"{'Validation':>14}" f"{'Test':>12}")

    for class_name in EXPECTED_CLASSES.values():

        log(
            f"{class_name:<12}"
            f"{train_stats['classes'][class_name]:>12,}"
            f"{val_stats['classes'][class_name]:>14,}"
            f"{test_stats['classes'][class_name]:>12,}"
        )

    log()
    log("SOURCE DISTRIBUTION")
    log("-" * 70)

    sources = sorted(
        train_stats["sources"].keys()
        | val_stats["sources"].keys()
        | test_stats["sources"].keys()
    )

    for src in sources:

        log(
            f"{src:<16}"
            f"{train_stats['sources'].get(src, 0):>12,}"
            f"{val_stats['sources'].get(src, 0):>14,}"
            f"{test_stats['sources'].get(src, 0):>12,}"
        )

    # --------------------------------------------------------
    # SNR distribution
    # --------------------------------------------------------

    log()
    log("SNR DISTRIBUTION")
    log("-" * 70)

    snrs = sorted(
        {
            int(x)
            for x in (
                set(train_stats["snr"].keys())
                | set(val_stats["snr"].keys())
                | set(test_stats["snr"].keys())
            )
        }
    )

    for snr in snrs:

        train_count = train_stats["snr"].get(str(snr), 0)

        val_count = val_stats["snr"].get(str(snr), 0)

        test_count = test_stats["snr"].get(str(snr), 0)

        log(
            f"{snr:>4} dB | "
            f"{train_count:>8,} | "
            f"{val_count:>8,} | "
            f"{test_count:>8,}"
        )

    # --------------------------------------------------------
    # Audit report
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "stage": "leakage_safe_dataset_split",
        "input_dataset": str(INPUT_FILE),
        "output_datasets": {
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
            "test": str(TEST_FILE),
        },
        "configuration": {
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "stratification": ["class", "snr", "source"],
        },
        "input_samples": int(n),
        "split_samples": {
            "train": int(len(train_indices)),
            "validation": int(len(val_indices)),
            "test": int(len(test_indices)),
        },
        "statistics": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },
        "leakage_checks": {
            "train_validation_overlap": 0,
            "train_test_overlap": 0,
            "validation_test_overlap": 0,
            "duplicate_original_indices": 0,
            "all_samples_assigned_once": True,
            "content_hash_sample_check": True,
        },
        "augmentation_performed": False,
        "training_performed": False,
        "source_datasets_modified": False,
        "notes": (
            "The dataset was split before augmentation. "
            "Augmentation must be applied only to the training "
            "split in the next stage. Validation and test sets "
            "must remain untouched."
        ),
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    log()
    log("=" * 70)
    log("LEAKAGE-SAFE SPLIT COMPLETE")
    log("=" * 70)

    log()
    log("[OK] Train dataset:")
    log(TRAIN_FILE)

    log()
    log("[OK] Validation dataset:")
    log(VAL_FILE)

    log()
    log("[OK] Test dataset:")
    log(TEST_FILE)

    log()
    log("[OK] Audit report:")
    log(REPORT_FILE)

    log()
    log("[PASS] No augmentation performed.")
    log("[PASS] No model training performed.")
    log("[PASS] Source merged dataset was not modified.")
    log("[PASS] No split overlap detected.")
    log("[PASS] Every source sample assigned exactly once.")

    log()
    log("NEXT STAGE:")
    log("Apply controlled RF data augmentation ONLY " "to the training split.")


if __name__ == "__main__":
    main()
