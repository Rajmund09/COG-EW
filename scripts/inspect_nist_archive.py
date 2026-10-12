from __future__ import annotations

import json
import zipfile
from pathlib import Path


ZIP_PATH = Path(
    r"D:\Datasets\NIST SEA-DATA compressed dataset" r"\NIST-Data-2026-09-19T22-19-1.zip"
)

REPORT_PATH = Path(r"D:\COG-EW\reports\nist_archive_inspection.json")


def main():

    print("=" * 70)
    print("COG-EW NIST SEA-DATA ARCHIVE INSPECTION")
    print("=" * 70)

    if not ZIP_PATH.exists():

        raise FileNotFoundError(f"NIST archive not found:\n" f"{ZIP_PATH}")

    print(f"\nArchive:\n{ZIP_PATH}")

    archive_size_gb = ZIP_PATH.stat().st_size / (1024**3)

    print(f"\nCompressed size: " f"{archive_size_gb:.3f} GB")

    files = []

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as archive:

        infos = archive.infolist()

        print(f"\nArchive members: " f"{len(infos):,}")

        for info in infos:

            suffix = Path(info.filename).suffix.lower()

            files.append(
                {
                    "name": info.filename,
                    "extension": suffix,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "directory": info.is_dir(),
                }
            )

    extension_counts = {}

    for item in files:

        ext = item["extension"] or "[none]"

        extension_counts[ext] = (
            extension_counts.get(
                ext,
                0,
            )
            + 1
        )

    report = {
        "archive": str(ZIP_PATH),
        "compressed_size_gb": round(
            archive_size_gb,
            3,
        ),
        "member_count": len(files),
        "extension_counts": extension_counts,
        "members": files,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print("\nFile types:")

    for ext, count in sorted(extension_counts.items()):

        print(f"  {ext}: {count:,}")

    print(f"\nReport saved:\n" f"{REPORT_PATH}")


if __name__ == "__main__":
    main()
