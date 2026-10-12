from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


RADIO_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)

REPORT_DIR = Path(r"D:\COG-EW\reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "radioml_distribution.json"

CHUNK_SIZE = 100_000


def analyze():

    print("=" * 70)
    print("COG-EW RADIO ML FULL DISTRIBUTION ANALYSIS")
    print("=" * 70)

    with h5py.File(RADIO_PATH, "r") as h5:

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]

        total = X.shape[0]
        num_classes = Y.shape[1]

        print(f"\nTotal samples: {total:,}")
        print(f"Classes: {num_classes}")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Z shape: {Z.shape}")

        # --------------------------------------------------
        # Full Y distribution
        # --------------------------------------------------

        class_counts = np.zeros(
            num_classes,
            dtype=np.int64,
        )

        # --------------------------------------------------
        # Full Z distribution
        # --------------------------------------------------

        snr_counts = {}

        # --------------------------------------------------
        # Validate one-hot structure
        # --------------------------------------------------

        one_hot_valid = True

        min_row_sum = None
        max_row_sum = None

        # --------------------------------------------------
        # Process in chunks
        # --------------------------------------------------

        for start in range(
            0,
            total,
            CHUNK_SIZE,
        ):

            end = min(
                start + CHUNK_SIZE,
                total,
            )

            y_chunk = np.asarray(Y[start:end])

            z_chunk = np.asarray(Z[start:end]).reshape(-1)

            # One-hot validation
            row_sums = np.sum(
                y_chunk,
                axis=1,
            )

            if not np.all(
                np.isin(
                    y_chunk,
                    [0, 1],
                )
            ):
                one_hot_valid = False

            if not np.all(row_sums == 1):
                one_hot_valid = False

            chunk_min = int(np.min(row_sums))

            chunk_max = int(np.max(row_sums))

            if min_row_sum is None:
                min_row_sum = chunk_min
                max_row_sum = chunk_max
            else:
                min_row_sum = min(
                    min_row_sum,
                    chunk_min,
                )

                max_row_sum = max(
                    max_row_sum,
                    chunk_max,
                )

            # Class counts
            class_ids = np.argmax(
                y_chunk,
                axis=1,
            )

            counts = np.bincount(
                class_ids,
                minlength=num_classes,
            )

            class_counts += counts

            # SNR counts
            values, counts = np.unique(
                z_chunk,
                return_counts=True,
            )

            for value, count in zip(
                values,
                counts,
            ):

                value = int(value)

                snr_counts[value] = snr_counts.get(
                    value,
                    0,
                ) + int(count)

            print(
                f"\rProcessed " f"{end:,}/{total:,}",
                end="",
                flush=True,
            )

        print("\n")

    # ------------------------------------------------------
    # Percentages
    # ------------------------------------------------------

    class_percentages = class_counts / total * 100.0

    snr_percentages = {
        str(snr): (count / total * 100.0) for snr, count in sorted(snr_counts.items())
    }

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    report = {
        "dataset": "RadioML 2018.01A",
        "samples": int(total),
        "num_classes": int(num_classes),
        "one_hot_validation": {
            "valid": bool(one_hot_valid),
            "min_row_sum": int(min_row_sum),
            "max_row_sum": int(max_row_sum),
        },
        "class_distribution": {
            str(class_id): {
                "count": int(class_counts[class_id]),
                "percentage": float(class_percentages[class_id]),
            }
            for class_id in range(num_classes)
        },
        "snr_distribution": {
            str(snr): {
                "count": int(snr_counts[snr]),
                "percentage": float(snr_percentages[str(snr)]),
            }
            for snr in sorted(snr_counts)
        },
        "class_count_min": int(np.min(class_counts)),
        "class_count_max": int(np.max(class_counts)),
        "class_count_mean": float(np.mean(class_counts)),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ------------------------------------------------------
    # Console output
    # ------------------------------------------------------

    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for class_id in range(num_classes):

        print(
            f"Class {class_id:02d}: "
            f"{class_counts[class_id]:,} "
            f"({class_percentages[class_id]:.3f}%)"
        )

    print("\n" + "=" * 70)
    print("SNR DISTRIBUTION")
    print("=" * 70)

    for snr in sorted(snr_counts):

        print(
            f"SNR {snr:>3} dB: "
            f"{snr_counts[snr]:,} "
            f"({snr_percentages[str(snr)]:.3f}%)"
        )

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"One-hot valid: {one_hot_valid}")

    print(f"Minimum class count: " f"{np.min(class_counts):,}")

    print(f"Maximum class count: " f"{np.max(class_counts):,}")

    print(f"Class-count mean: " f"{np.mean(class_counts):,.2f}")

    print(f"\nReport saved to:\n" f"{REPORT_PATH}")


if __name__ == "__main__":
    analyze()
