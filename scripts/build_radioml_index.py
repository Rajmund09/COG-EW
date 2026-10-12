from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


RADIO_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)

OUTPUT_DIR = Path(r"D:\COG-EW\data\processed\radioml")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INDEX_PATH = OUTPUT_DIR / "radioml_index.npz"

REPORT_PATH = OUTPUT_DIR / "index_report.json"


def main():

    print("=" * 70)
    print("COG-EW RADIOML INDEX BUILDER")
    print("=" * 70)

    with h5py.File(
        RADIO_PATH,
        "r",
    ) as h5:

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]

        total = X.shape[0]

        print(f"\nSamples: {total:,}")

        # --------------------------------------------------
        # Read labels only
        # --------------------------------------------------

        print("Reading Y labels...")

        Y_all = np.asarray(
            Y[:],
            dtype=np.int8,
        )

        print("Converting one-hot " "labels to class IDs...")

        class_ids = np.argmax(
            Y_all,
            axis=1,
        ).astype(np.int8)

        del Y_all

        print("Reading Z SNR values...")

        snr = np.asarray(
            Z[:],
            dtype=np.int8,
        ).reshape(-1)

    sample_index = np.arange(
        total,
        dtype=np.int32,
    )

    np.savez_compressed(
        INDEX_PATH,
        sample_index=sample_index,
        class_id=class_ids,
        snr=snr,
    )

    report = {
        "dataset": "RadioML 2018.01A",
        "samples": int(total),
        "classes": int(len(np.unique(class_ids))),
        "snr_levels": [int(v) for v in np.unique(snr)],
        "class_counts": {
            str(int(c)): int(np.sum(class_ids == c)) for c in np.unique(class_ids)
        },
        "snr_counts": {str(int(s)): int(np.sum(snr == s)) for s in np.unique(snr)},
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

    print("\n[PASS] Index created.")

    print(f"\nIndex:\n{INDEX_PATH}")

    print(f"\nReport:\n{REPORT_PATH}")

    print("\nClass distribution:")

    for c, count in report["class_counts"].items():

        print(f"  Class {c}: " f"{count:,}")

    print("\nSNR levels:")

    print(report["snr_levels"])


if __name__ == "__main__":
    main()
