"""
COG-EW
RadioML 2018.01A -> Common 128-IQ Adapter

Purpose
-------
Prepare a clean, harmonized subset of RadioML 2018.01A that can be
compared/combined with the adapted RadioML 2016.10A dataset.

IMPORTANT
---------
- Original RadioML 2018 dataset is NEVER modified.
- No augmentation is performed here.
- No merging is performed here.
- No model training is performed here.
- Only the five modulation classes shared with our RadioML 2016
  adaptation are retained.

Common classes
--------------
8PSK
BPSK
QPSK
16QAM
64QAM

Input
-----
RadioML 2018.01A:
GOLD_XYZ_OSC.0001_1024.hdf5

Expected structure
------------------
X: [N, 1024, 2]
Y: [N, 24] one-hot labels
Z: [N, 1] SNR

Output
------
data/intermediate/rml2018_common128.h5

Output structure
----------------
X: [N, 128, 2]
Y: [N]
Z: [N]
source_index: [N]

The conversion from 1024 IQ samples to 128 IQ samples is performed
using deterministic contiguous windows of 128 samples.

For every original RadioML 2018 sample, only the FIRST 128 IQ samples
are retained. This keeps the transformation deterministic and avoids
creating artificial training examples at this stage.

A later dedicated augmentation/windowing experiment can evaluate
alternative strategies without contaminating this preparation step.
"""

from pathlib import Path
import json
import sys

import h5py
import numpy as np


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    Path(r"D:\Datasets")
    / "RadioML 2018.01A ⭐ Main RF benchmark"
    / "GOLD_XYZ_OSC.0001_1024.hdf5"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_PATH = OUTPUT_DIR / "rml2018_common128.h5"
REPORT_PATH = REPORT_DIR / "rml2018_common128_adaptation_audit.json"


# ======================================================================
# RADIO ML 2018 CLASS DEFINITIONS
# ======================================================================

RADIOML2018_CLASSES = [
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


# Classes shared with the adapted RadioML 2016 dataset.
COMMON_CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fail(message):
    print()
    print("[ERROR]", message)
    sys.exit(1)


def validate_paths():
    if not DATASET_PATH.exists():
        fail(f"RadioML 2018 dataset was not found:\n" f"{DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_class_indices():
    mapping = {}

    for class_name in COMMON_CLASS_NAMES:
        if class_name not in RADIOML2018_CLASSES:
            fail(
                f"Required class '{class_name}' is not present "
                f"in the RadioML 2018 class list."
            )

        mapping[class_name] = RADIOML2018_CLASSES.index(class_name)

    return mapping


# ======================================================================
# MAIN PREPARATION
# ======================================================================


def main():

    print_header("COG-EW RADIO ML 2018.01A -> COMMON 128-IQ ADAPTER")

    print()
    print("Input dataset:")
    print(DATASET_PATH)

    print()
    print("Output dataset:")
    print(OUTPUT_PATH)

    validate_paths()

    # --------------------------------------------------------------
    # CLASS MAPPING
    # --------------------------------------------------------------

    print_header("CLASS MAPPING")

    class_indices = get_class_indices()

    for class_name in COMMON_CLASS_NAMES:
        print(f"  RadioML 2018 ID {class_indices[class_name]:2d}" f" -> {class_name}")

    print()
    print("Excluded RadioML 2018 classes:")

    excluded = [name for name in RADIOML2018_CLASSES if name not in COMMON_CLASS_NAMES]

    for name in excluded:
        print(f"  [EXCLUDED] {name}")

    # New common label mapping.
    common_label_map = {
        "8PSK": 0,
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 3,
        "64QAM": 4,
    }

    print()
    print("Common label mapping:")

    for name, label in common_label_map.items():
        print(f"  {label} -> {name}")

    # --------------------------------------------------------------
    # OPEN DATASET
    # --------------------------------------------------------------

    print_header("OPENING RADIO ML 2018 DATASET")

    with h5py.File(DATASET_PATH, "r") as source:

        required_keys = {"X", "Y", "Z"}

        missing = required_keys.difference(source.keys())

        if missing:
            fail("Dataset is missing required keys: " + ", ".join(sorted(missing)))

        X = source["X"]
        Y = source["Y"]
        Z = source["Z"]

        print("Dataset keys:")
        for key in source.keys():
            print(f"  {key}: {source[key].shape}")

        print()
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)
        print("Z shape:", Z.shape)

        # ----------------------------------------------------------
        # STRUCTURAL VALIDATION
        # ----------------------------------------------------------

        print_header("STRUCTURAL VALIDATION")

        if len(X.shape) != 3:
            fail(f"Unexpected X dimensions: {X.shape}")

        if X.shape[1] != 1024 or X.shape[2] != 2:
            fail("Expected RadioML 2018 X shape " "[N, 1024, 2], got " f"{X.shape}")

        if Y.shape[0] != X.shape[0]:
            fail("X/Y sample counts do not match.")

        if Z.shape[0] != X.shape[0]:
            fail("X/Z sample counts do not match.")

        if Y.shape[1] != 24:
            fail(f"Expected 24 RadioML 2018 classes, " f"got {Y.shape[1]}")

        print("[PASS] X structure")
        print("[PASS] Y structure")
        print("[PASS] Z structure")
        print("[PASS] sample alignment")

        total_samples = X.shape[0]

        # ----------------------------------------------------------
        # DETERMINE TARGET INDICES
        # ----------------------------------------------------------

        print_header("LOCATING COMMON CLASS SAMPLES")

        target_indices = []

        class_counts = {}

        for class_name in COMMON_CLASS_NAMES:

            original_class_id = class_indices[class_name]

            # RadioML labels are one-hot.
            class_mask = np.asarray(Y[:, original_class_id]) == 1

            indices = np.flatnonzero(class_mask)

            class_counts[class_name] = int(len(indices))

            target_indices.append(indices)

            print(f"{class_name:8s} : " f"{len(indices):,} samples")

        target_indices = np.concatenate(target_indices)

        print()
        print(f"Total selected samples: " f"{len(target_indices):,}")

        # ----------------------------------------------------------
        # SORT INDICES
        # ----------------------------------------------------------

        # Preserve original dataset order.
        target_indices.sort()

        # ----------------------------------------------------------
        # OUTPUT HDF5
        # ----------------------------------------------------------

        print_header("CREATING COMMON 128-IQ DATASET")

        total_selected = len(target_indices)

        # We use extendable/chunked HDF5 datasets so that the complete
        # RadioML 2018 source is not duplicated in RAM.
        with h5py.File(OUTPUT_PATH, "w") as output:

            output_X = output.create_dataset(
                "X",
                shape=(total_selected, 128, 2),
                dtype=np.float32,
                chunks=(min(4096, total_selected), 128, 2),
                compression="gzip",
                compression_opts=4,
            )

            output_Y = output.create_dataset(
                "Y",
                shape=(total_selected,),
                dtype=np.int64,
                chunks=(min(4096, total_selected),),
                compression="gzip",
                compression_opts=4,
            )

            output_Z = output.create_dataset(
                "Z",
                shape=(total_selected,),
                dtype=np.int64,
                chunks=(min(4096, total_selected),),
                compression="gzip",
                compression_opts=4,
            )

            output_source_index = output.create_dataset(
                "source_index",
                shape=(total_selected,),
                dtype=np.int64,
                chunks=(min(4096, total_selected),),
                compression="gzip",
                compression_opts=4,
            )

            # Store metadata.
            output.attrs["source_dataset"] = "RadioML 2018.01A"
            output.attrs["source_file"] = str(DATASET_PATH)
            output.attrs["source_sample_length"] = 1024
            output.attrs["common_sample_length"] = 128
            output.attrs["iq_channels"] = 2
            output.attrs["augmentation_performed"] = False
            output.attrs["cleaning_performed"] = True
            output.attrs["merge_performed"] = False
            output.attrs["training_performed"] = False

            # ------------------------------------------------------
            # PROCESS IN CHUNKS
            # ------------------------------------------------------

            chunk_size = 4096

            written = 0

            invalid_nan = 0
            invalid_inf = 0
            invalid_shape = 0

            output_class_counts = {name: 0 for name in COMMON_CLASS_NAMES}

            print()
            print(f"Processing {total_selected:,} selected samples...")

            for start in range(0, total_selected, chunk_size):

                end = min(start + chunk_size, total_selected)

                source_indices = target_indices[start:end]

                # --------------------------------------------------
                # Load selected source samples.
                # --------------------------------------------------

                batch = np.asarray(X[source_indices], dtype=np.float32)

                labels_onehot = np.asarray(Y[source_indices], dtype=np.int64)

                snr_batch = np.asarray(Z[source_indices]).reshape(-1)

                # --------------------------------------------------
                # Validate finite values.
                # --------------------------------------------------

                nan_mask = np.isnan(batch).any(axis=(1, 2))

                inf_mask = np.isinf(batch).any(axis=(1, 2))

                invalid_mask = nan_mask | inf_mask

                invalid_nan += int(nan_mask.sum())
                invalid_inf += int(inf_mask.sum())

                valid_batch_mask = ~invalid_mask

                batch = batch[valid_batch_mask]
                labels_onehot = labels_onehot[valid_batch_mask]
                snr_batch = snr_batch[valid_batch_mask]

                valid_source_indices = source_indices[valid_batch_mask]

                if len(batch) == 0:
                    continue

                # --------------------------------------------------
                # Convert 1024 IQ -> 128 IQ.
                # --------------------------------------------------
                #
                # We intentionally take the first 128 samples.
                #
                # This is NOT augmentation.
                #
                # It is only a deterministic representation
                # harmonization step.
                # --------------------------------------------------

                batch_common = batch[:, :128, :]

                # --------------------------------------------------
                # Convert labels.
                # --------------------------------------------------

                original_ids = np.argmax(labels_onehot, axis=1)

                reverse_map = {
                    class_indices[name]: common_label_map[name]
                    for name in COMMON_CLASS_NAMES
                }

                common_labels = np.array(
                    [reverse_map[int(class_id)] for class_id in original_ids],
                    dtype=np.int64,
                )

                # --------------------------------------------------
                # Write.
                # --------------------------------------------------

                batch_count = len(batch_common)

                output_X[written : written + batch_count] = batch_common

                output_Y[written : written + batch_count] = common_labels

                output_Z[written : written + batch_count] = snr_batch.astype(np.int64)

                output_source_index[written : written + batch_count] = (
                    valid_source_indices
                )

                # Counts.
                for class_name, common_id in common_label_map.items():

                    count = int(np.sum(common_labels == common_id))

                    output_class_counts[class_name] += count

                written += batch_count

                if written % 100_000 < chunk_size or written == total_selected:
                    print(f"  Written " f"{written:,}/" f"{total_selected:,}")

            # ------------------------------------------------------
            # If invalid samples existed, trim output.
            # ------------------------------------------------------

            if written != total_selected:

                print()
                print("[WARNING] Some invalid samples were removed.")

                output_X.resize((written, 128, 2))

                output_Y.resize((written,))

                output_Z.resize((written,))

                output_source_index.resize((written,))

            actual_count = written

    # ==================================================================
    # OUTPUT VERIFICATION
    # ==================================================================

    print_header("OUTPUT VERIFICATION")

    with h5py.File(OUTPUT_PATH, "r") as check:

        out_X = check["X"]
        out_Y = check["Y"]
        out_Z = check["Z"]
        out_source = check["source_index"]

        print("X shape:", out_X.shape)
        print("Y shape:", out_Y.shape)
        print("Z shape:", out_Z.shape)
        print("source_index shape:", out_source.shape)

        # Small verification sample.
        verification_size = min(10_000, out_X.shape[0])

        verification = np.asarray(out_X[:verification_size])

        has_nan = bool(np.isnan(verification).any())

        has_inf = bool(np.isinf(verification).any())

        print()
        print("NaN in verification sample:", has_nan)

        print("Inf in verification sample:", has_inf)

        print("Minimum:", float(verification.min()))

        print("Maximum:", float(verification.max()))

        print("Mean:", float(verification.mean()))

        print("Std:", float(verification.std()))

        # Verify labels.
        unique_labels, label_counts = np.unique(out_Y[:], return_counts=True)

        print()
        print("COMMON CLASS DISTRIBUTION")

        reverse_common = {value: key for key, value in common_label_map.items()}

        for label, count in zip(unique_labels, label_counts):
            print(
                f"{int(label):02d} | "
                f"{reverse_common.get(int(label), 'UNKNOWN'):8s}"
                f" | {int(count):,}"
            )

        # Verify SNR distribution.
        unique_snr, snr_counts = np.unique(out_Z[:], return_counts=True)

        print()
        print("SNR RANGE")

        print(f"Minimum SNR: {int(unique_snr.min())} dB")

        print(f"Maximum SNR: {int(unique_snr.max())} dB")

    # ==================================================================
    # REPORT
    # ==================================================================

    report = {
        "project": "COG-EW",
        "operation": "RadioML 2018.01A common 128-IQ adaptation",
        "source": {
            "dataset": "RadioML 2018.01A",
            "path": str(DATASET_PATH),
            "original_shape": [int(total_samples), 1024, 2],
        },
        "output": {
            "path": str(OUTPUT_PATH),
            "shape": [int(actual_count), 128, 2],
        },
        "class_mapping": {
            name: {
                "source_class_id": int(class_indices[name]),
                "common_class_id": int(common_class_map),
            }
            for name, common_class_map in common_label_map.items()
        },
        "selected_class_counts": {
            name: int(count) for name, count in class_counts.items()
        },
        "output_class_counts": {
            name: int(count) for name, count in output_class_counts.items()
        },
        "excluded_classes": excluded,
        "quality": {
            "invalid_nan": int(invalid_nan),
            "invalid_inf": int(invalid_inf),
            "invalid_shape": int(invalid_shape),
        },
        "processing": {
            "source_modified": False,
            "augmentation_performed": False,
            "merge_performed": False,
            "training_performed": False,
            "representation_conversion": ("1024 IQ samples -> first 128 IQ samples"),
            "normalization_performed": False,
        },
        "next_stage": (
            "Perform independent harmonization, "
            "leakage-safe split, merge, and "
            "training-only augmentation."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2)

    # ==================================================================
    # FINAL
    # ==================================================================

    print_header("RADIO ML 2018 ADAPTATION COMPLETE")

    print()
    print("[OK] Intermediate dataset:")
    print(OUTPUT_PATH)

    print()
    print("[OK] Audit report:")
    print(REPORT_PATH)

    print()
    print("[PASS] Original RadioML 2018 dataset " "was NOT modified.")

    print("[PASS] No augmentation performed.")

    print("[PASS] No dataset merge performed.")

    print("[PASS] No model training performed.")

    print()
    print(
        "Next stage: leakage-safe harmonization " "and controlled multi-dataset merge."
    )


if __name__ == "__main__":
    main()
