from __future__ import annotations

import json
import hashlib
from pathlib import Path

import h5py
import numpy as np


# ============================================================
# PROJECT / DATASET PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Your RF datasets are stored outside the project directory.
DATASET_ROOT = Path(r"D:\Datasets")

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "rf_dataset_inventory.json"


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".csv",
    ".parquet",
    ".pkl",
    ".pickle",
    ".mat",
    ".wav",
    ".iq",
    ".bin",
}


# ============================================================
# HELPERS
# ============================================================


def human_size(size_bytes: int) -> str:

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    size = float(size_bytes)

    for unit in units:

        if size < 1024:

            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_shape(obj):

    try:
        return list(obj.shape)
    except Exception:
        return None


# ============================================================
# HDF5 INSPECTION
# ============================================================


def inspect_hdf5(path: Path) -> dict:

    result = {
        "format": "HDF5",
        "datasets": [],
        "attributes": {},
        "error": None,
    }

    try:

        with h5py.File(path, "r") as h5:

            for key, value in h5.attrs.items():

                try:

                    if isinstance(value, np.ndarray):
                        result["attributes"][key] = value.tolist()
                    else:
                        result["attributes"][key] = value.item()

                except Exception:
                    result["attributes"][key] = str(value)

            def visitor(
                name,
                obj,
            ):

                if isinstance(
                    obj,
                    h5py.Dataset,
                ):

                    info = {
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "size": int(obj.size),
                    }

                    result["datasets"].append(info)

            h5.visititems(visitor)

    except Exception as exc:

        result["error"] = repr(exc)

    return result


# ============================================================
# NPZ INSPECTION
# ============================================================


def inspect_npz(path: Path) -> dict:

    result = {
        "format": "NPZ",
        "arrays": [],
        "error": None,
    }

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            for name in data.files:

                array = data[name]

                result["arrays"].append(
                    {
                        "name": name,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "size": int(array.size),
                    }
                )

    except Exception as exc:

        result["error"] = repr(exc)

    return result


# ============================================================
# NPY INSPECTION
# ============================================================


def inspect_npy(path: Path) -> dict:

    result = {
        "format": "NPY",
        "shape": None,
        "dtype": None,
        "error": None,
    }

    try:

        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        result["shape"] = list(array.shape)

        result["dtype"] = str(array.dtype)

        result["size"] = int(array.size)

    except Exception as exc:

        result["error"] = repr(exc)

    return result


# ============================================================
# GENERIC FILE INSPECTION
# ============================================================


def inspect_file(path: Path) -> dict:

    result = {
        "path": str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "size_human": human_size(path.stat().st_size),
    }

    extension = path.suffix.lower()

    if extension in {".h5", ".hdf5"}:

        result["details"] = inspect_hdf5(path)

    elif extension == ".npz":

        result["details"] = inspect_npz(path)

    elif extension == ".npy":

        result["details"] = inspect_npy(path)

    else:

        result["details"] = {
            "format": extension.lstrip(".").upper(),
            "inspection": (
                "Metadata inventory only. "
                "Detailed parser will be created after dataset identification."
            ),
        }

    return result


# ============================================================
# MAIN INVENTORY
# ============================================================


def main():

    print("=" * 78)
    print("COG-EW RF DATASET INVENTORY & COMPATIBILITY AUDIT")
    print("=" * 78)

    print()
    print("Dataset root:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(f"Dataset root does not exist:\n{DATASET_ROOT}")

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    print()
    print("Scanning datasets...")

    all_files = []

    for path in DATASET_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:

            all_files.append(path)

    all_files.sort(key=lambda item: str(item).lower())

    print(f"Candidate RF/data files found: " f"{len(all_files):,}")

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    inventory = []

    for index, path in enumerate(
        all_files,
        start=1,
    ):

        print()
        print(f"[{index}/{len(all_files)}] " f"{path.name}")

        print(f"  Path: {path}")

        print(f"  Size: " f"{human_size(path.stat().st_size)}")

        try:

            record = inspect_file(path)

            # Hash only files that are reasonably sized.
            # This prevents an enormous dataset from making
            # the inventory unnecessarily slow.
            if path.stat().st_size <= 2 * 1024**3:

                print("  Calculating SHA256...")

                record["sha256"] = sha256_file(path)

            else:

                record["sha256"] = "SKIPPED_FILE_LARGER_THAN_2GB"

            inventory.append(record)

        except Exception as exc:

            inventory.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "extension": path.suffix.lower(),
                    "error": repr(exc),
                }
            )

            print(f"  [ERROR] {exc}")

    # --------------------------------------------------------
    # Dataset grouping
    # --------------------------------------------------------

    extension_summary = {}

    for record in inventory:

        extension = record.get(
            "extension",
            "unknown",
        )

        extension_summary[extension] = (
            extension_summary.get(
                extension,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # Important RadioML detection
    # --------------------------------------------------------

    radioml_files = [
        record
        for record in inventory
        if "GOLD_XYZ_OSC.0001_1024" in record.get("name", "")
    ]

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "purpose": (
            "Inventory and compatibility audit " "before multi-dataset RF training."
        ),
        "dataset_root": str(DATASET_ROOT),
        "total_candidate_files": len(inventory),
        "extension_summary": extension_summary,
        "radioml_detected": len(radioml_files) > 0,
        "radioml_files": radioml_files,
        "files": inventory,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INVENTORY SUMMARY")
    print("=" * 78)

    print(f"Candidate files: " f"{len(inventory):,}")

    print()

    print("File types:")

    for extension, count in sorted(extension_summary.items()):

        print(f"  {extension or '[no extension]':10s}" f" : {count:,}")

    print()

    if radioml_files:

        print("[OK] RadioML 2018.01A detected:")

        for record in radioml_files:

            print(f"  {record['path']}")

    else:

        print("[WARNING] RadioML HDF5 file " "was not detected.")

    print()
    print("Inventory report:")
    print(REPORT_PATH)

    print()
    print("=" * 78)
    print("DATASET INVENTORY COMPLETE")
    print("=" * 78)

    print()
    print("IMPORTANT:")

    print("No datasets were modified.")

    print("No datasets were cleaned.")

    print("No datasets were merged.")

    print("No model checkpoints were modified.")

    print("No training was performed.")


if __name__ == "__main__":

    main()
