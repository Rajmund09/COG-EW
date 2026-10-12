"""
COG-EW: Leakage-Safe RF Dataset Harmonization + Merge
======================================================

INPUTS
------
D:\\COG-EW\\data\\intermediate\\rml2016_common128.h5
D:\\COG-EW\\data\\intermediate\\rml2018_common128.h5

OUTPUT
------
D:\\COG-EW\\data\\processed\\cogew_common_rf.h5

PURPOSE
-------
1. Validate both adapted datasets.
2. Verify common 5-class structure.
3. Remove invalid samples if present.
4. Balance the two sources deterministically.
5. Normalize each IQ sample using RMS normalization.
6. Preserve class, SNR and source metadata.
7. Merge both datasets.
8. Produce an audit report.

IMPORTANT
---------
- Original datasets are NEVER modified.
- No augmentation is performed here.
- No train/validation/test split is performed here.
- No model training is performed here.
- Augmentation will happen ONLY after splitting.
"""

from pathlib import Path
import json
import h5py
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\COG-EW")

RML2016_PATH = PROJECT_ROOT / "data" / "intermediate" / "rml2016_common128.h5"

RML2018_PATH = PROJECT_ROOT / "data" / "intermediate" / "rml2018_common128.h5"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "cogew_common_rf.h5"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = REPORT_DIR / "cogew_common_rf_merge_audit.json"


# Common classes
CLASS_MAP = {
    0: "8PSK",
    1: "BPSK",
    2: "QPSK",
    3: "16QAM",
    4: "64QAM",
}

EXPECTED_IQ_LENGTH = 128
EXPECTED_CHANNELS = 2

SEED = 20261008


# ============================================================
# UTILITY
# ============================================================


def stop(message):
    raise RuntimeError(message)


# ============================================================
# DATASET VALIDATION
# ============================================================


def validate_dataset(path, dataset_name):

    print()
    print("=" * 70)
    print(f"VALIDATING {dataset_name}")
    print("=" * 70)

    print(f"Path:")
    print(path)

    if not path.exists():
        stop(
            f"\nDataset not found:\n{path}\n\n"
            f"Make sure the previous preparation step completed successfully."
        )

    with h5py.File(path, "r") as h5:

        required = {"X", "Y", "Z"}

        missing = required - set(h5.keys())

        if missing:
            stop(f"{dataset_name} is missing datasets: " f"{sorted(missing)}")

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]

        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Z shape: {Z.shape}")

        print(f"X dtype: {X.dtype}")
        print(f"Y dtype: {Y.dtype}")
        print(f"Z dtype: {Z.dtype}")

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if X.ndim != 3:
            stop(f"{dataset_name}: X must be 3-dimensional.")

        if X.shape[1] != EXPECTED_IQ_LENGTH:
            stop(
                f"{dataset_name}: expected IQ length "
                f"{EXPECTED_IQ_LENGTH}, got {X.shape[1]}"
            )

        if X.shape[2] != EXPECTED_CHANNELS:
            stop(
                f"{dataset_name}: expected "
                f"{EXPECTED_CHANNELS} channels, got {X.shape[2]}"
            )

        if Y.ndim != 1:
            stop(f"{dataset_name}: Y must be 1-dimensional.")

        if Z.ndim != 1:
            stop(f"{dataset_name}: Z must be 1-dimensional.")

        n = X.shape[0]

        if len(Y) != n:
            stop(f"{dataset_name}: X/Y alignment failure.")

        if len(Z) != n:
            stop(f"{dataset_name}: X/Z alignment failure.")

        print("[PASS] X/Y/Z alignment")

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = Y[:]
        snr = Z[:]

        unique_labels = np.unique(labels)

        expected_labels = np.array(sorted(CLASS_MAP.keys()), dtype=unique_labels.dtype)

        if not np.array_equal(unique_labels, expected_labels):
            stop(
                f"{dataset_name}: unexpected labels.\n"
                f"Found: {unique_labels.tolist()}\n"
                f"Expected: {expected_labels.tolist()}"
            )

        print("[PASS] Expected five classes detected")

        # ----------------------------------------------------
        # Finite-value scan
        # ----------------------------------------------------

        invalid_values = 0

        chunk_size = 8192

        for start in range(0, n, chunk_size):

            end = min(start + chunk_size, n)

            block = X[start:end]

            invalid_values += int(np.count_nonzero(~np.isfinite(block)))

        if invalid_values > 0:

            stop(f"{dataset_name}: " f"{invalid_values:,} invalid values detected.")

        print("[PASS] No NaN/Inf values")

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        class_counts = {}

        print()
        print("Class distribution:")

        for class_id, class_name in CLASS_MAP.items():

            count = int(np.sum(labels == class_id))

            class_counts[class_name] = count

            print(f"  {class_id:02d} | " f"{class_name:<8} | " f"{count:,}")

        # ----------------------------------------------------
        # SNR information
        # ----------------------------------------------------

        snr_min = int(np.min(snr))
        snr_max = int(np.max(snr))

        print()
        print(f"SNR range: {snr_min} dB " f"to {snr_max} dB")

        return {
            "samples": int(n),
            "shape": list(X.shape),
            "classes": class_counts,
            "snr_min": snr_min,
            "snr_max": snr_max,
            "invalid_values": invalid_values,
        }


# ============================================================
# PREPARE DATASET
# ============================================================


def prepare_dataset(path, source_id, source_name, target_per_class):

    print()
    print("=" * 70)
    print(f"PREPARING {source_name}")
    print("=" * 70)

    rng = np.random.default_rng(SEED + source_id)

    with h5py.File(path, "r") as h5:

        X_source = h5["X"]

        Y_source = h5["Y"][:]

        Z_source = h5["Z"][:]

        selected_indices = []

        # ----------------------------------------------------
        # Equal number of samples per class
        # ----------------------------------------------------

        for class_id in CLASS_MAP:

            indices = np.flatnonzero(Y_source == class_id)

            if len(indices) < target_per_class:

                stop(
                    f"{source_name}: class "
                    f"{CLASS_MAP[class_id]} contains "
                    f"{len(indices):,} samples but "
                    f"{target_per_class:,} required."
                )

            chosen = rng.choice(indices, size=target_per_class, replace=False)

            selected_indices.append(chosen)

        selected_indices = np.concatenate(selected_indices)

        # Shuffle selected records
        rng.shuffle(selected_indices)

        n = len(selected_indices)

        X_output = np.empty(
            (n, EXPECTED_IQ_LENGTH, EXPECTED_CHANNELS), dtype=np.float32
        )

        Y_output = Y_source[selected_indices].astype(np.int8)

        Z_output = Z_source[selected_indices].astype(np.int16)

        source_output = np.full(n, source_id, dtype=np.int8)

        # ----------------------------------------------------
        # Sample-wise RMS normalization
        # ----------------------------------------------------

        batch_size = 4096

        for start in range(0, n, batch_size):

            end = min(start + batch_size, n)

            batch_idx = selected_indices[start:end]
            sorted_order = np.argsort(batch_idx)
            sorted_indices = batch_idx[sorted_order]

            X_batch_sorted = np.asarray(
                X_source[sorted_indices], dtype=np.float32
            )

            undo_order = np.argsort(sorted_order)
            X_batch = X_batch_sorted[undo_order]

            # RMS across IQ channels and time
            rms = np.sqrt(np.mean(X_batch**2, axis=(1, 2), keepdims=True))

            # Prevent division by zero
            rms = np.maximum(rms, 1e-8)

            X_batch = X_batch / rms

            X_output[start:end] = X_batch

            if start == 0 or end == n or end % 50000 < batch_size:

                print(f"  Normalized " f"{end:,}/{n:,}")

    return (X_output, Y_output, Z_output, source_output)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("COG-EW RF DATASET HARMONIZATION + MERGE")
    print("=" * 70)

    print()
    print("This stage does NOT train a model.")
    print("This stage does NOT perform augmentation.")
    print("This stage does NOT modify source datasets.")

    # ========================================================
    # STEP 1 — Validate datasets
    # ========================================================

    audit_2016 = validate_dataset(RML2016_PATH, "RadioML 2016.10A")

    audit_2018 = validate_dataset(RML2018_PATH, "RadioML 2018.01A")

    # ========================================================
    # STEP 2 — Determine balanced sample count
    # ========================================================

    print()
    print("=" * 70)
    print("DETERMINING CONTROLLED BALANCE")
    print("=" * 70)

    minimum_counts = {}

    for class_id, class_name in CLASS_MAP.items():

        count_2016 = audit_2016["classes"][class_name]

        count_2018 = audit_2018["classes"][class_name]

        minimum_counts[class_name] = min(count_2016, count_2018)

        print(
            f"{class_name:<8} | "
            f"2016={count_2016:,} | "
            f"2018={count_2018:,} | "
            f"usable={minimum_counts[class_name]:,}"
        )

    target_per_class = min(minimum_counts.values())

    print()
    print(f"Target per class per source: " f"{target_per_class:,}")

    expected_total = target_per_class * len(CLASS_MAP) * 2

    print(f"Expected merged samples: " f"{expected_total:,}")

    # ========================================================
    # STEP 3 — Prepare RadioML 2016
    # ========================================================

    X16, Y16, Z16, S16 = prepare_dataset(
        RML2016_PATH,
        source_id=0,
        source_name="RadioML 2016",
        target_per_class=target_per_class,
    )

    # ========================================================
    # STEP 4 — Prepare RadioML 2018
    # ========================================================

    X18, Y18, Z18, S18 = prepare_dataset(
        RML2018_PATH,
        source_id=1,
        source_name="RadioML 2018",
        target_per_class=target_per_class,
    )

    # ========================================================
    # STEP 5 — Merge
    # ========================================================

    print()
    print("=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)

    X = np.concatenate([X16, X18], axis=0)

    Y = np.concatenate([Y16, Y18], axis=0)

    Z = np.concatenate([Z16, Z18], axis=0)

    SOURCE = np.concatenate([S16, S18], axis=0)

    print(f"Before final shuffle: " f"{len(Y):,} samples")

    # Final deterministic shuffle
    rng = np.random.default_rng(SEED)

    permutation = rng.permutation(len(Y))

    X = X[permutation]
    Y = Y[permutation]
    Z = Z[permutation]
    SOURCE = SOURCE[permutation]

    print(f"After final shuffle: " f"{len(Y):,} samples")

    # ========================================================
    # STEP 6 — Final validation
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MERGED DATASET VALIDATION")
    print("=" * 70)

    if not np.isfinite(X).all():
        stop("Merged X contains NaN or Inf.")

    if not np.isfinite(Y).all():
        stop("Merged Y contains invalid values.")

    if not np.isfinite(Z).all():
        stop("Merged Z contains invalid values.")

    print("[PASS] X contains only finite values")
    print("[PASS] Y contains only finite values")
    print("[PASS] Z contains only finite values")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    merged_class_counts = {}

    print()
    print("FINAL CLASS DISTRIBUTION")

    for class_id, class_name in CLASS_MAP.items():

        count = int(np.sum(Y == class_id))

        merged_class_counts[class_name] = count

        print(f"{class_id:02d} | " f"{class_name:<8} | " f"{count:,}")

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    source_counts = {
        "RadioML2016": int(np.sum(SOURCE == 0)),
        "RadioML2018": int(np.sum(SOURCE == 1)),
    }

    print()
    print("FINAL SOURCE DISTRIBUTION")

    for source_name, count in source_counts.items():

        print(f"{source_name:<12} | " f"{count:,}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("SIGNAL STATISTICS")

    print(f"Mean: " f"{np.mean(X):.8f}")

    print(f"Std : " f"{np.std(X):.8f}")

    print(f"Min : " f"{np.min(X):.8f}")

    print(f"Max : " f"{np.max(X):.8f}")

    # ========================================================
    # STEP 7 — Save merged dataset
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("WRITING MERGED DATASET")
    print("=" * 70)

    with h5py.File(OUTPUT_PATH, "w") as h5:

        h5.create_dataset(
            "X",
            data=X,
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        h5.create_dataset(
            "Y", data=Y, dtype="int8", compression="gzip", compression_opts=4
        )

        h5.create_dataset(
            "Z", data=Z, dtype="int16", compression="gzip", compression_opts=4
        )

        h5.create_dataset(
            "source", data=SOURCE, dtype="int8", compression="gzip", compression_opts=4
        )

        # Metadata
        h5.attrs["project"] = "COG-EW"

        h5.attrs["dataset_type"] = "common_128_iq"

        h5.attrs["class_mapping"] = json.dumps(CLASS_MAP)

        h5.attrs["source_mapping"] = json.dumps(
            {
                "0": "RadioML2016.10A",
                "1": "RadioML2018.01A",
            }
        )

        h5.attrs["normalization"] = "per-sample RMS normalization"

        h5.attrs["augmentation"] = "not performed"

        h5.attrs["train_validation_test_split"] = "not performed"

        h5.attrs["model_training"] = "not performed"

    # ========================================================
    # STEP 8 — Audit report
    # ========================================================

    report = {
        "project": "COG-EW",
        "stage": "harmonization_and_controlled_merge",
        "seed": SEED,
        "inputs": {
            "RadioML2016": audit_2016,
            "RadioML2018": audit_2018,
        },
        "class_mapping": CLASS_MAP,
        "target_per_class_per_source": target_per_class,
        "merged_samples": int(len(Y)),
        "merged_shape": list(X.shape),
        "class_distribution": merged_class_counts,
        "source_distribution": source_counts,
        "snr_range": {
            "minimum": int(np.min(Z)),
            "maximum": int(np.max(Z)),
        },
        "normalization": {"method": "per-sample RMS normalization"},
        "quality": {
            "x_finite": bool(np.isfinite(X).all()),
            "y_finite": bool(np.isfinite(Y).all()),
            "z_finite": bool(np.isfinite(Z).all()),
            "mean": float(np.mean(X)),
            "std": float(np.std(X)),
            "minimum": float(np.min(X)),
            "maximum": float(np.max(X)),
        },
        "augmentation": False,
        "split_performed": False,
        "training_performed": False,
        "source_datasets_modified": False,
        "output": str(OUTPUT_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("HARMONIZATION + MERGE COMPLETE")
    print("=" * 70)

    print()
    print("[OK] Merged dataset:")
    print(OUTPUT_PATH)

    print()
    print("[OK] Audit report:")
    print(REPORT_PATH)

    print()
    print("[PASS] RadioML 2016 original dataset unchanged.")
    print("[PASS] RadioML 2018 original dataset unchanged.")
    print("[PASS] No augmentation performed.")
    print("[PASS] No model training performed.")
    print("[PASS] No train/validation/test split performed.")

    print()
    print("NEXT STEP:")
    print("Create a leakage-safe train/validation/test split " "BEFORE augmentation.")


if __name__ == "__main__":
    main()
