from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = REPORT_DIR / "radioml_dataset_audit.json"


# ============================================================
# RADIO ML DATASET
# ============================================================

DATASET_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_SAMPLES = 2_555_904

EXPECTED_CLASSES = 24

EXPECTED_SAMPLES_PER_CLASS = EXPECTED_SAMPLES // EXPECTED_CLASSES

EXPECTED_IQ_LENGTH = 1024


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


def safe_float(value):

    value = float(value)

    if np.isfinite(value):
        return value

    return None


# ============================================================
# MAIN
# ============================================================


def main():

    print_header("COG-EW RADIO ML DATASET AUDIT")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(f"RadioML dataset not found:\n" f"{DATASET_PATH}")

    print("\nRadioML dataset:")

    print(DATASET_PATH)

    # --------------------------------------------------------
    # Open dataset
    # --------------------------------------------------------

    print_header("OPENING HDF5 DATASET")

    with h5py.File(
        DATASET_PATH,
        "r",
    ) as h5:

        print("\nDataset keys:")

        for key in h5.keys():

            print(f"  {key}: " f"{h5[key].shape}")

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]

        total_samples = X.shape[0]

        # ----------------------------------------------------
        # Structure
        # ----------------------------------------------------

        print_header("DATASET STRUCTURE")

        print(f"X shape: {X.shape}")

        print(f"Y shape: {Y.shape}")

        print(f"Z shape: {Z.shape}")

        print(f"X dtype: {X.dtype}")

        print(f"Y dtype: {Y.dtype}")

        print(f"Z dtype: {Z.dtype}")

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        print_header("STRUCTURAL VALIDATION")

        structural_checks = {
            "sample_count": (total_samples == EXPECTED_SAMPLES),
            "class_count": (Y.shape[1] == EXPECTED_CLASSES),
            "iq_length": (X.shape[1] == EXPECTED_IQ_LENGTH),
            "iq_channels": (X.shape[2] == 2),
            "label_alignment": (X.shape[0] == Y.shape[0] == Z.shape[0]),
        }

        for name, result in structural_checks.items():

            status = "[PASS]" if result else "[FAIL]"

            print(f"{status} {name}")

        # ----------------------------------------------------
        # Process in chunks
        # ----------------------------------------------------

        chunk_size = 20_000

        class_counts = np.zeros(
            EXPECTED_CLASSES,
            dtype=np.int64,
        )

        snr_values = []

        nan_count = 0

        inf_count = 0

        global_sum = 0.0

        global_sum_sq = 0.0

        global_count = 0

        min_value = np.inf

        max_value = -np.inf

        class_sum = np.zeros(
            EXPECTED_CLASSES,
            dtype=np.float64,
        )

        class_sum_sq = np.zeros(
            EXPECTED_CLASSES,
            dtype=np.float64,
        )

        class_count = np.zeros(
            EXPECTED_CLASSES,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Audit chunks
        # ----------------------------------------------------

        print_header("SCANNING DATASET")

        for start in range(
            0,
            total_samples,
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                total_samples,
            )

            x_chunk = np.asarray(X[start:end])

            y_chunk = np.asarray(Y[start:end])

            z_chunk = np.asarray(Z[start:end]).reshape(-1)

            # ----------------------------------------------
            # NaN / Inf
            # ----------------------------------------------

            nan_count += int(np.isnan(x_chunk).sum())

            inf_count += int(np.isinf(x_chunk).sum())

            # ----------------------------------------------
            # Global statistics
            # ----------------------------------------------

            finite_mask = np.isfinite(x_chunk)

            finite_values = x_chunk[finite_mask]

            if finite_values.size:

                global_sum += float(finite_values.sum(dtype=np.float64))

                global_sum_sq += float(
                    np.square(
                        finite_values,
                        dtype=np.float64,
                    ).sum(dtype=np.float64)
                )

                global_count += finite_values.size

                min_value = min(
                    min_value,
                    float(finite_values.min()),
                )

                max_value = max(
                    max_value,
                    float(finite_values.max()),
                )

            # ----------------------------------------------
            # Labels
            # ----------------------------------------------

            labels = np.argmax(
                y_chunk,
                axis=1,
            )

            for class_id in range(EXPECTED_CLASSES):

                mask = labels == class_id

                count = int(mask.sum())

                if count == 0:
                    continue

                class_counts[class_id] += count

                class_signals = x_chunk[mask]

                class_values = class_signals.reshape(-1)

                finite_class = class_values[np.isfinite(class_values)]

                if finite_class.size:

                    class_sum[class_id] += float(finite_class.sum(dtype=np.float64))

                    class_sum_sq[class_id] += float(
                        np.square(
                            finite_class,
                            dtype=np.float64,
                        ).sum(dtype=np.float64)
                    )

                    class_count[class_id] += finite_class.size

            # ----------------------------------------------
            # SNR
            # ----------------------------------------------

            snr_values.extend(z_chunk.tolist())

            if ((start // chunk_size) + 1) % 10 == 0:

                processed = min(
                    end,
                    total_samples,
                )

                print(f"  Processed " f"{processed:,}/" f"{total_samples:,}")

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if global_count > 0:

            global_mean = global_sum / global_count

            global_variance = global_sum_sq / global_count - global_mean**2

            global_std = np.sqrt(
                max(
                    global_variance,
                    0.0,
                )
            )

        else:

            global_mean = None
            global_std = None

        # ----------------------------------------------------
        # SNR analysis
        # ----------------------------------------------------

        snr_array = np.asarray(
            snr_values,
            dtype=np.float32,
        )

        unique_snrs, snr_counts = np.unique(
            snr_array,
            return_counts=True,
        )

        snr_distribution = {}

        for snr, count in zip(
            unique_snrs,
            snr_counts,
        ):

            snr_distribution[str(int(snr))] = int(count)

        # ----------------------------------------------------
        # Class analysis
        # ----------------------------------------------------

        class_distribution = {}

        for class_id in range(EXPECTED_CLASSES):

            count = int(class_counts[class_id])

            if class_count[class_id] > 0:

                mean = class_sum[class_id] / class_count[class_id]

                variance = class_sum_sq[class_id] / class_count[class_id] - mean**2

                std = np.sqrt(
                    max(
                        variance,
                        0.0,
                    )
                )

            else:

                mean = None
                std = None

            class_distribution[str(class_id)] = {
                "class_name": CLASS_NAMES[class_id],
                "samples": count,
                "percentage": (count / total_samples * 100.0),
                "mean": safe_float(mean),
                "std": safe_float(std),
            }

        # ----------------------------------------------------
        # Duplicate probe
        # ----------------------------------------------------

        print_header("DUPLICATE SAMPLE PROBE")

        # We deliberately do not load all 2.55M
        # samples into RAM.

        duplicate_probe_size = min(
            50_000,
            total_samples,
        )

        probe = np.asarray(X[:duplicate_probe_size])

        probe_flat = probe.reshape(
            duplicate_probe_size,
            -1,
        )

        _, unique_indices = np.unique(
            probe_flat,
            axis=0,
            return_index=True,
        )

        duplicate_count = duplicate_probe_size - len(unique_indices)

        duplicate_rate = duplicate_count / duplicate_probe_size * 100.0

        print(f"Probe samples: " f"{duplicate_probe_size:,}")

        print(f"Duplicate samples: " f"{duplicate_count:,}")

        print(f"Duplicate rate: " f"{duplicate_rate:.4f}%")

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        report = {
            "project": "COG-EW",
            "dataset": ("RadioML 2018.01A"),
            "path": str(DATASET_PATH),
            "structure": {
                "X_shape": list(X.shape),
                "Y_shape": list(Y.shape),
                "Z_shape": list(Z.shape),
                "X_dtype": str(X.dtype),
                "Y_dtype": str(Y.dtype),
                "Z_dtype": str(Z.dtype),
            },
            "structural_checks": structural_checks,
            "global_statistics": {
                "samples": int(total_samples),
                "mean": safe_float(global_mean),
                "std": safe_float(global_std),
                "minimum": safe_float(min_value),
                "maximum": safe_float(max_value),
            },
            "data_quality": {
                "nan_values": int(nan_count),
                "infinite_values": int(inf_count),
                "duplicate_probe": {
                    "samples": int(duplicate_probe_size),
                    "duplicates": int(duplicate_count),
                    "duplicate_rate_percent": float(duplicate_rate),
                },
            },
            "class_distribution": class_distribution,
            "snr_distribution": snr_distribution,
        }

        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
            )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print_header("RADIO ML DATASET QUALITY")

        print(f"Total samples: " f"{total_samples:,}")

        print(f"NaN values: " f"{nan_count:,}")

        print(f"Infinite values: " f"{inf_count:,}")

        print(f"Global mean: " f"{global_mean:.8f}")

        print(f"Global std: " f"{global_std:.8f}")

        print(f"Minimum: " f"{min_value:.8f}")

        print(f"Maximum: " f"{max_value:.8f}")

        print_header("CLASS DISTRIBUTION")

        print("ID | Class          | Samples")

        print("-" * 45)

        for class_id in range(EXPECTED_CLASSES):

            print(
                f"{class_id:2d} | "
                f"{CLASS_NAMES[class_id]:14s} | "
                f"{class_counts[class_id]:8,}"
            )

        print_header("SNR DISTRIBUTION")

        for snr in unique_snrs:

            count = snr_distribution[str(int(snr))]

            print(f"SNR {int(snr):>4} dB : " f"{count:,}")

        print_header("AUDIT COMPLETE")

        print("Report:")

        print(REPORT_PATH)

        print("\n[PASS] Original RadioML dataset " "was not modified.")

        print("[PASS] No training was performed.")

        print("[PASS] No augmentation was performed.")

        print("[PASS] No merging was performed.")


if __name__ == "__main__":
    main()
