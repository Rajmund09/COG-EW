from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


DATA_ROOT = Path(
    os.getenv(
        "DATA_ROOT",
        r"D:\Datasets",
    )
)

NIST_DIR = DATA_ROOT / "NIST SEA-DATA compressed dataset"

ZIP_PATH = NIST_DIR / "NIST-Data-2026-09-19T22-19-1.zip"

OUTPUT_DIR = Path(r"D:\COG-EW\data\processed") / "nist_sea"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    print("=" * 70)
    print("COG-EW NIST DUCKDB EXTRACTION")
    print("=" * 70)

    if not ZIP_PATH.exists():

        raise FileNotFoundError(f"NIST ZIP not found:\n" f"{ZIP_PATH}")

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as archive:

        duckdb_members = [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith(".duckdb")
        ]

        if not duckdb_members:

            raise RuntimeError("No .duckdb file was found " "inside the archive.")

        if len(duckdb_members) > 1:

            raise RuntimeError(
                "More than one .duckdb file "
                "was found. Manual selection "
                "is required."
            )

        info = duckdb_members[0]

        print(f"\nDuckDB member:\n" f"{info.filename}")

        compressed_mb = info.compress_size / (1024**2)

        uncompressed_mb = info.file_size / (1024**2)

        print(f"Compressed: " f"{compressed_mb:.2f} MB")

        print(f"Uncompressed: " f"{uncompressed_mb:.2f} MB")

        # ----------------------------------------------
        # Disk-space safety check
        # ----------------------------------------------

        usage = shutil.disk_usage(OUTPUT_DIR)

        free_gb = usage.free / (1024**3)

        required_gb = info.file_size / (1024**3) * 1.10

        print(f"\nFree disk space: " f"{free_gb:.2f} GB")

        print(f"Estimated required: " f"{required_gb:.2f} GB")

        if free_gb < required_gb:

            raise RuntimeError("Insufficient disk space " "for safe extraction.")

        # ----------------------------------------------
        # Safe output filename
        # ----------------------------------------------

        output_name = Path(info.filename).name

        output_path = OUTPUT_DIR / output_name

        print(f"\nOutput:\n" f"{output_path}")

        if output_path.exists():

            print("\n[OK] DuckDB already " "exists. No extraction " "performed.")

            return

        # ----------------------------------------------
        # Extract only the database
        # ----------------------------------------------

        with archive.open(info) as source, output_path.open("wb") as target:

            shutil.copyfileobj(
                source,
                target,
                length=1024 * 1024,
            )

    print("\n[PASS] DuckDB extracted.")

    print(f"Path:\n{output_path}")


if __name__ == "__main__":
    main()
