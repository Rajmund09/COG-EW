from __future__ import annotations

import hashlib
import json
import os
import pickle
import zipfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".pkl": "pickle",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".sigmf": "sigmf_metadata",
    ".sigmf-data": "sigmf_data",
    ".duckdb": "duckdb",
    ".csv": "csv",
    ".parquet": "parquet",
    ".zip": "zip",
}


DATASET_NAME_PATTERNS = {
    "radioml2016": [
        "rml2016",
        "radioml2016",
        "rml2016.10a",
        "rml2016.10a_dict",
    ],
    "radioml2018": [
        "radioml2018",
        "gold_xyz",
        "osc.0001_1024",
        "2018.01a",
    ],
    "nist_sea_data": [
        "nist",
        "sea-data",
        "sea_data",
        "seadata",
    ],
    "idf_ds": [
        "idf-ds",
        "idf_ds",
        "holybro",
        "pixhawk",
        "speedybee",
    ],
}


def get_project_root() -> Path:
    """
    Return the project root.

    scripts/discover_datasets.py
    -> scripts/
    -> project root
    """

    return Path(__file__).resolve().parents[1]


def get_data_root() -> Path:
    """
    Determine the dataset search root.

    Priority:
    1. DATA_ROOT environment variable
    2. D:/Datasets (if present)
    3. project/data
    """

    env_root = os.getenv("DATA_ROOT")

    if env_root:
        return Path(env_root).expanduser().resolve()

    d_datasets = Path("D:/Datasets")
    if d_datasets.exists():
        return d_datasets.resolve()

    return (get_project_root() / "data").resolve()



def format_size(size_bytes: int) -> str:
    """Convert bytes into a human-readable size."""

    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} B"


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA256.

    Large files are processed in chunks to avoid loading
    entire datasets into memory.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def identify_dataset(path: Path) -> str:
    """
    Identify a likely dataset from the filename/path.

    This is only a heuristic. It does not invent dataset
    contents or labels.
    """

    text = str(path).lower()

    for dataset_name, patterns in DATASET_NAME_PATTERNS.items():

        for pattern in patterns:

            if pattern.lower() in text:
                return dataset_name

    return "unknown"


def verify_zip(path: Path) -> dict[str, Any]:
    """Inspect ZIP integrity without extracting it."""

    result = {
        "valid": False,
        "file_count": None,
        "error": None,
    }

    try:
        with zipfile.ZipFile(path, "r") as archive:

            bad_file = archive.testzip()

            result["valid"] = bad_file is None
            result["file_count"] = len(archive.namelist())

            if bad_file:
                result["error"] = f"Corrupt archive member: {bad_file}"

    except Exception as exc:
        result["error"] = str(exc)

    return result


def verify_pickle(path: Path) -> dict[str, Any]:
    """
    Perform a lightweight pickle integrity check.

    We do not retain the loaded object.
    """

    result = {
        "valid": False,
        "object_type": None,
        "error": None,
    }

    try:
        with path.open("rb") as file:
            obj = pickle.load(file)

        result["valid"] = True
        result["object_type"] = type(obj).__name__

        del obj

    except Exception as exc:
        result["error"] = str(exc)

    return result


def verify_file(path: Path, file_format: str) -> dict[str, Any]:
    """
    Perform format-specific lightweight validation.
    """

    result: dict[str, Any] = {
        "valid": True,
        "details": {},
    }

    try:

        if file_format == "zip":
            result["details"] = verify_zip(path)

        elif file_format == "pickle":
            result["details"] = verify_pickle(path)

        elif file_format == "hdf5":
            try:
                import h5py

                with h5py.File(path, "r") as handle:
                    result["details"] = {
                        "valid": True,
                        "root_keys": list(handle.keys())[:20],
                    }

            except ImportError:
                result["details"] = {
                    "valid": None,
                    "message": "h5py is not installed.",
                }

        elif file_format == "duckdb":
            try:
                import duckdb

                connection = duckdb.connect(
                    database=str(path),
                    read_only=True,
                )

                tables = connection.execute("SHOW TABLES").fetchall()

                connection.close()

                result["details"] = {
                    "valid": True,
                    "tables": [row[0] for row in tables],
                }

            except ImportError:
                result["details"] = {
                    "valid": None,
                    "message": "duckdb is not installed.",
                }

        elif file_format == "parquet":
            try:
                import pandas as pd

                metadata = pd.read_parquet(
                    path,
                    engine="pyarrow",
                ).head(0)

                result["details"] = {
                    "valid": True,
                    "columns": list(metadata.columns),
                }

            except ImportError:
                result["details"] = {
                    "valid": None,
                    "message": "pandas/pyarrow unavailable.",
                }

        elif file_format == "csv":
            try:
                import pandas as pd

                sample = pd.read_csv(
                    path,
                    nrows=5,
                )

                result["details"] = {
                    "valid": True,
                    "columns": list(sample.columns),
                }

            except Exception as exc:
                result["details"] = {
                    "valid": False,
                    "error": str(exc),
                }

        elif file_format in {"sigmf_metadata", "sigmf_data"}:
            result["details"] = {
                "valid": True,
                "message": "SigMF file detected.",
            }

    except Exception as exc:
        result["valid"] = False
        result["details"] = {
            "error": str(exc),
        }

    return result


def scan_files(data_root: Path) -> list[dict[str, Any]]:
    """Recursively scan the dataset directory."""

    datasets: list[dict[str, Any]] = []

    for path in sorted(data_root.rglob("*")):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        file_format = SUPPORTED_EXTENSIONS[suffix]

        try:
            stat = path.stat()

            relative_path = path.relative_to(data_root)

            record = {
                "name": path.name,
                "dataset": identify_dataset(path),
                "path": str(path.resolve()),
                "relative_path": str(relative_path),
                "format": file_format,
                "extension": suffix,
                "size_bytes": stat.st_size,
                "size_human": format_size(stat.st_size),
                "modified_time": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "verification": verify_file(
                    path,
                    file_format,
                ),
            }

            datasets.append(record)

        except Exception as exc:

            datasets.append(
                {
                    "name": path.name,
                    "dataset": identify_dataset(path),
                    "path": str(path.resolve()),
                    "format": file_format,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return datasets


def print_report(
    data_root: Path,
    datasets: list[dict[str, Any]],
) -> None:
    """Print a human-readable dataset report."""

    print()
    print("=" * 80)
    print("COG-EW DATASET DISCOVERY")
    print("=" * 80)

    print(f"Search root: {data_root}")

    if not data_root.exists():

        print()
        print("[ERROR] Dataset root does not exist.")
        print(f"[ERROR] Expected directory: {data_root}")
        print()
        print("Set DATA_ROOT to your actual dataset directory.")
        return

    if not datasets:

        print()
        print("[WARNING] No supported dataset files were found.")
        print()
        print("Supported formats:")
        print(", ".join(SUPPORTED_EXTENSIONS.keys()))
        return

    print()
    print(f"Detected files: {len(datasets)}")
    print()

    for index, item in enumerate(datasets, start=1):

        print(f"[{index}] {item['name']}")
        print(f"    Dataset: {item['dataset']}")
        print(f"    Format:  {item['format']}")
        print(f"    Size:    {item.get('size_human', 'unknown')}")
        print(f"    Path:    {item['path']}")

        verification = item.get("verification", {})
        details = verification.get("details", {})

        if details:
            if details.get("valid") is True:
                print("    Check:   VALID")

            elif details.get("valid") is False:
                print("    Check:   INVALID")

                if details.get("error"):
                    print(f"    Error:   {details['error']}")

            elif details.get("valid") is None:
                print("    Check:   NOT AVAILABLE")

        print()


def save_inventory(
    data_root: Path,
    datasets: list[dict[str, Any]],
) -> Path:
    """Save machine-readable dataset inventory."""

    output_path = get_project_root() / "dataset_inventory.json"

    inventory = {
        "project": "COG-EW",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "datasets": datasets,
        "summary": {
            "files_detected": len(datasets),
            "dataset_types": sorted({item["dataset"] for item in datasets}),
            "formats": sorted({item["format"] for item in datasets}),
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            inventory,
            file,
            indent=2,
        )

    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data_root = get_data_root()

    print(f"DATA_ROOT: {data_root}")

    if not data_root.exists():

        print()
        print("[ERROR] Dataset root does not exist:")
        print(data_root)

        print()
        print("Create the directory or set DATA_ROOT.")

        return

    datasets = scan_files(data_root)

    inventory_path = save_inventory(
        data_root,
        datasets,
    )

    print_report(
        data_root,
        datasets,
    )

    print("=" * 80)
    print("INVENTORY")
    print("=" * 80)

    print(f"Saved to: {inventory_path}")

    print()
    print("Dataset discovery completed.")


if __name__ == "__main__":
    main()
