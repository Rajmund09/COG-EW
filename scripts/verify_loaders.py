from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.loaders import (
    RadioML2018Loader,
    NISTSeaDataLoader,
    IDFDSLoader,
)


RADIOML_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)


NIST_PATH = Path(
    r"D:\Datasets\NIST SEA-DATA compressed dataset" r"\NIST-Data-2026-09-19T22-19-1.zip"
)


IDF_PATH = Path(
    r"D:\Datasets\IDF-DS — Flight telemetry"
    r"\Holybro Pixhawk\processed"
    r"\Grouped flights\vuelo_100_sync.csv"
)


def find_nist_database():

    # NIST is stored inside ZIP archives.
    # We do not extract the multi-GB archive here.

    print("NIST ZIP detected.")

    print("Database inspection will be performed " "after safe extraction.")


def verify_radioml():

    print("\n" + "=" * 70)
    print("RADIO ML 2018 LOADER")
    print("=" * 70)

    if not RADIOML_PATH.exists():

        print(f"[SKIP] File not found:\n" f"{RADIOML_PATH}")

        return

    with RadioML2018Loader(RADIOML_PATH) as loader:

        metadata = loader.inspect()

        print(
            json.dumps(
                metadata,
                indent=2,
            )
        )

        print(f"\nDataset length: {len(loader):,}")

        x, y, z = loader.get_sample(0)

        print(f"Single X shape: {x.shape}")

        print(f"Single Y shape: {y.shape}")

        print(f"Single Z shape: {z.shape}")

        print(f"X dtype: {x.dtype}")

        print(f"Y dtype: {y.dtype}")

        print(f"Z dtype: {z.dtype}")

        indices = np.array(
            [0, 10, 100, 1000],
            dtype=np.int64,
        )

        xb, yb, zb = loader.get_batch(indices)

        print(f"\nBatch X shape: {xb.shape}")

        print(f"Batch Y shape: {yb.shape}")

        print(f"Batch Z shape: {zb.shape}")

        print("\n[PASS] RadioML loader " "read indexed samples only.")


def verify_nist_zip():

    print("\n" + "=" * 70)
    print("NIST SEA-DATA")
    print("=" * 70)

    if NIST_PATH.exists():

        print(f"[OK] NIST archive found:\n" f"{NIST_PATH}")

        find_nist_database()

    else:

        print("[SKIP] NIST archive not found.")


def verify_idfds():

    print("\n" + "=" * 70)
    print("IDF-DS LOADER")
    print("=" * 70)

    if not IDF_PATH.exists():

        print(f"[SKIP] IDF file not found:\n" f"{IDF_PATH}")

        return

    loader = IDFDSLoader(IDF_PATH)

    metadata = loader.inspect()

    print(f"Rows: {metadata['rows']:,}")

    print(f"Columns: {len(metadata['columns'])}")

    print("\nFirst columns:")

    for column in metadata["columns"][:20]:

        print(f"  - {column}")

    print("\n[PASS] IDF-DS loader initialized.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW LOADER VERIFICATION")
    print("=" * 70)

    verify_radioml()
    verify_nist_zip()
    verify_idfds()

    print("\n" + "=" * 70)
    print("LOADER VERIFICATION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
