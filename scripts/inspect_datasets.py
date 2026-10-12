from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import yaml


# ============================================================
# COG-EW DATASET INSPECTOR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT = REPORT_DIR / "dataset_inspection.json"
MD_REPORT = REPORT_DIR / "dataset_inspection.md"


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """
    Convert objects into JSON-serializable values.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    return str(value)


def file_size_mb(path: Path) -> float:
    return round(
        path.stat().st_size / (1024**2),
        3,
    )


def file_size_gb(path: Path) -> float:
    return round(
        path.stat().st_size / (1024**3),
        3,
    )


# ------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f) or {}


def get_data_root(config: dict) -> Path:

    environment_root = os.getenv("DATA_ROOT")

    if environment_root:
        return Path(environment_root).expanduser().resolve()

    configured_root = config.get("data", {}).get("root", "")

    if configured_root.startswith("${"):
        inner = configured_root[2:-1]
        parts = inner.split(":", 1)
        var_name = parts[0]
        default_path = parts[1] if len(parts) > 1 else "data"
        env_val = os.getenv(var_name)
        if env_val:
            return Path(env_val).expanduser().resolve()
        configured_root = default_path

    if configured_root:
        path = Path(configured_root).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path.resolve()

    d_datasets = Path("D:/Datasets")
    if d_datasets.exists():
        return d_datasets.resolve()

    return (PROJECT_ROOT / "data").resolve()


# ------------------------------------------------------------
# HDF5 inspection
# ------------------------------------------------------------


def inspect_hdf5(path: Path) -> dict:

    result = {
        "format": "HDF5",
        "path": str(path),
        "size_mb": file_size_mb(path),
        "size_gb": file_size_gb(path),
        "datasets": [],
        "groups": [],
        "root_attributes": {},
        "errors": [],
    }

    try:

        with h5py.File(path, "r") as h5:

            result["root_attributes"] = {
                str(k): json_safe(v) for k, v in h5.attrs.items()
            }

            def visitor(name, obj):

                if isinstance(obj, h5py.Group):

                    result["groups"].append(name)

                elif isinstance(obj, h5py.Dataset):

                    dataset_info = {
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "ndim": obj.ndim,
                        "size": int(obj.size),
                        "chunks": json_safe(obj.chunks),
                        "compression": obj.compression,
                        "compression_opts": json_safe(obj.compression_opts),
                        "attributes": {
                            str(k): json_safe(v) for k, v in obj.attrs.items()
                        },
                    }

                    result["datasets"].append(dataset_info)

            h5.visititems(visitor)

    except Exception as exc:

        result["errors"].append(repr(exc))

    return result


# ------------------------------------------------------------
# ZIP inspection
# ------------------------------------------------------------


def inspect_zip(path: Path) -> dict:

    result = {
        "format": "ZIP",
        "path": str(path),
        "size_mb": file_size_mb(path),
        "members": [],
        "extensions": {},
        "errors": [],
    }

    try:

        with zipfile.ZipFile(path, "r") as archive:

            for info in archive.infolist():

                suffix = Path(info.filename).suffix.lower()

                result["members"].append(
                    {
                        "name": info.filename,
                        "compressed_size": info.compress_size,
                        "uncompressed_size": info.file_size,
                        "is_directory": info.is_dir(),
                    }
                )

                result["extensions"][suffix] = (
                    result["extensions"].get(
                        suffix,
                        0,
                    )
                    + 1
                )

    except Exception as exc:

        result["errors"].append(repr(exc))

    return result


# ------------------------------------------------------------
# SigMF inspection
# ------------------------------------------------------------


def inspect_sigmf(path: Path) -> dict:

    result = {
        "format": "SigMF",
        "path": str(path),
        "size_mb": file_size_mb(path),
        "metadata": {},
        "errors": [],
    }

    try:

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:

            metadata = json.load(f)

        result["metadata"] = json_safe(metadata)

    except Exception as exc:

        result["errors"].append(repr(exc))

    return result


# ------------------------------------------------------------
# CSV inspection
# ------------------------------------------------------------


def inspect_csv(path: Path) -> dict:

    result = {
        "format": "CSV",
        "path": str(path),
        "size_mb": file_size_mb(path),
        "columns": [],
        "dtypes": {},
        "sample_rows": [],
        "estimated_rows": None,
        "errors": [],
    }

    try:

        sample = pd.read_csv(
            path,
            nrows=10,
        )

        result["columns"] = list(sample.columns)

        result["dtypes"] = {
            str(column): str(dtype) for column, dtype in sample.dtypes.items()
        }

        result["sample_rows"] = json_safe(sample.to_dict(orient="records"))

        # Estimate row count without loading
        # the entire CSV into memory.

        with path.open("rb") as f:

            row_count = sum(1 for _ in f)

        result["estimated_rows"] = max(
            row_count - 1,
            0,
        )

    except Exception as exc:

        result["errors"].append(repr(exc))

    return result


# ------------------------------------------------------------
# Parquet inspection
# ------------------------------------------------------------


def inspect_parquet(path: Path) -> dict:

    result = {
        "format": "Parquet",
        "path": str(path),
        "size_mb": file_size_mb(path),
        "columns": [],
        "dtypes": {},
        "sample_rows": [],
        "errors": [],
    }

    try:

        sample = pd.read_parquet(path).head(10)

        result["columns"] = list(sample.columns)

        result["dtypes"] = {
            str(column): str(dtype) for column, dtype in sample.dtypes.items()
        }

        result["sample_rows"] = json_safe(sample.to_dict(orient="records"))

    except Exception as exc:

        result["errors"].append(repr(exc))

    return result


# ------------------------------------------------------------
# Generic file inspection
# ------------------------------------------------------------


def inspect_file(path: Path) -> dict:

    suffix = path.suffix.lower()

    if suffix in {".h5", ".hdf5"}:

        return inspect_hdf5(path)

    if suffix == ".zip":

        return inspect_zip(path)

    if suffix == ".csv":

        return inspect_csv(path)

    if suffix in {
        ".parquet",
        ".pq",
    }:

        return inspect_parquet(path)

    if suffix == ".sigmf-meta":

        return inspect_sigmf(path)

    return {
        "format": suffix or "unknown",
        "path": str(path),
        "size_mb": file_size_mb(path),
    }


# ------------------------------------------------------------
# Dataset discovery
# ------------------------------------------------------------


def discover_files(
    data_root: Path,
) -> list[Path]:

    supported = {
        ".h5",
        ".hdf5",
        ".zip",
        ".csv",
        ".parquet",
        ".pq",
        ".sigmf-meta",
    }

    files = []

    for path in data_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in supported:

            files.append(path)

    return sorted(files)


# ------------------------------------------------------------
# Prioritize important files
# ------------------------------------------------------------


def prioritize_files(
    files: list[Path],
) -> list[Path]:

    priority = []

    for path in files:

        name = path.name.lower()

        score = 0

        if "radioml" in str(path).lower():
            score += 100

        if "gold_xyz" in name:
            score += 200

        if "nist" in str(path).lower():
            score += 100

        if "sea" in str(path).lower():
            score += 50

        if "idf" in str(path).lower():
            score += 100

        if path.suffix.lower() in {
            ".h5",
            ".hdf5",
            ".sigmf-meta",
            ".zip",
        }:
            score += 20

        priority.append(
            (
                score,
                path,
            )
        )

    priority.sort(
        key=lambda x: (
            -x[0],
            str(x[1]).lower(),
        )
    )

    return [path for _, path in priority]


# ------------------------------------------------------------
# Markdown report
# ------------------------------------------------------------


def generate_markdown(
    report: dict,
) -> str:

    lines = []

    lines.append("# COG-EW Dataset Inspection Report")

    lines.append("")

    lines.append(f"Generated: {report['generated_at']}")

    lines.append("")

    lines.append(f"Data root: `{report['data_root']}`")

    lines.append("")

    lines.append(f"Files inspected: " f"{len(report['files'])}")

    lines.append("")

    lines.append("---")
    lines.append("")

    for item in report["files"]:

        lines.append(f"## {Path(item['path']).name}")

        lines.append("")

        lines.append(f"- Path: `{item['path']}`")

        lines.append(f"- Format: `{item['format']}`")

        lines.append(f"- Size: " f"{item.get('size_mb', 'N/A')} MB")

        if item.get("size_gb"):
            lines.append(f"- Size: " f"{item['size_gb']} GB")

        if item.get("datasets"):

            lines.append("")
            lines.append("### HDF5 datasets")
            lines.append("")

            lines.append("| Dataset | Shape | Dtype | Size |")
            lines.append("|---|---|---|---:|")

            for dataset in item["datasets"]:

                lines.append(
                    "| "
                    f"`{dataset['name']}` | "
                    f"`{dataset['shape']}` | "
                    f"`{dataset['dtype']}` | "
                    f"{dataset['size']} |"
                )

        if item.get("groups"):

            lines.append("")
            lines.append("### HDF5 groups")
            lines.append("")

            for group in item["groups"]:

                lines.append(f"- `{group}`")

        if item.get("root_attributes"):

            lines.append("")
            lines.append("### Root attributes")
            lines.append("")

            for key, value in item["root_attributes"].items():

                lines.append(f"- **{key}**: `{value}`")

        if item.get("extensions"):

            lines.append("")
            lines.append("### ZIP extensions")
            lines.append("")

            for extension, count in sorted(item["extensions"].items()):

                lines.append(f"- `{extension or '[no extension]'}`: " f"{count}")

        if item.get("columns"):

            lines.append("")
            lines.append("### Columns")
            lines.append("")

            for column in item["columns"]:

                lines.append(f"- `{column}`")

        if item.get("errors"):

            lines.append("")
            lines.append("### Errors")
            lines.append("")

            for error in item["errors"]:

                lines.append(f"- `{error}`")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW DATASET INSPECTION")
    print("=" * 70)

    config = load_config()

    data_root = get_data_root(config)

    print(f"\nDataset root: {data_root}")

    if not data_root.exists():

        raise FileNotFoundError(f"Dataset root does not exist: " f"{data_root}")

    print("\nSearching for supported dataset files...")

    files = discover_files(data_root)

    files = prioritize_files(files)

    print(f"Found {len(files)} supported files.")

    print("\nInspecting datasets...")

    report = {
        "project": "COG-EW",
        "generated_at": datetime.now().isoformat(),
        "data_root": str(data_root),
        "files": [],
    }

    for index, path in enumerate(
        files,
        start=1,
    ):

        print(f"[{index}/{len(files)}] " f"{path.name}")

        try:

            result = inspect_file(path)

            report["files"].append(result)

        except Exception as exc:

            report["files"].append(
                {
                    "path": str(path),
                    "format": path.suffix,
                    "errors": [repr(exc)],
                }
            )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with JSON_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save Markdown
    # --------------------------------------------------------

    markdown = generate_markdown(report)

    MD_REPORT.write_text(
        markdown,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print(f"\nJSON report:" f"\n{JSON_REPORT}")

    print(f"\nMarkdown report:" f"\n{MD_REPORT}")

    print("\nNo full HDF5 dataset was loaded into RAM.")


if __name__ == "__main__":
    main()
