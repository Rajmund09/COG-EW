from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import Counter

import h5py


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
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "rf_csv_profile.json"


# ============================================================
# DATA ROOT
# ============================================================

# Keep this consistent with your inventory script.
# Change only if your inventory script uses another root.

DATA_ROOTS = [
    Path(r"D:\Datasets"),
]


# ============================================================
# COLUMN KEYWORDS
# ============================================================

IQ_KEYWORDS = {
    "i",
    "q",
    "iq",
    "inphase",
    "in_phase",
    "quadrature",
    "real",
    "imag",
    "imaginary",
}

LABEL_KEYWORDS = {
    "label",
    "class",
    "modulation",
    "modulation_type",
    "signal_type",
    "target",
}

SNR_KEYWORDS = {
    "snr",
    "snr_db",
    "signal_to_noise_ratio",
}

RF_KEYWORDS = {
    "iq",
    "i",
    "q",
    "inphase",
    "quadrature",
    "complex",
    "signal",
    "modulation",
    "snr",
    "frequency",
    "phase",
    "amplitude",
}


# ============================================================
# HELPERS
# ============================================================


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def classify_columns(columns: list[str]) -> dict:

    normalized = {normalize_column_name(column) for column in columns}

    iq_columns = sorted(normalized.intersection(IQ_KEYWORDS))

    label_columns = sorted(normalized.intersection(LABEL_KEYWORDS))

    snr_columns = sorted(normalized.intersection(SNR_KEYWORDS))

    rf_columns = sorted(normalized.intersection(RF_KEYWORDS))

    return {
        "iq_columns": iq_columns,
        "label_columns": label_columns,
        "snr_columns": snr_columns,
        "rf_related_columns": rf_columns,
    }


def read_csv_header_and_sample(path: Path):

    try:

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as file:

            reader = csv.reader(file)

            header = next(reader, [])

            samples = []

            for _ in range(3):

                row = next(reader, None)

                if row is None:
                    break

                samples.append(row)

        return header, samples, None

    except Exception as exc:

        return [], [], str(exc)


def classify_dataset(
    path: Path,
    header: list[str],
    sample_rows: list[list[str]],
):

    columns = classify_columns(header)

    normalized = {normalize_column_name(column) for column in header}

    score = 0
    reasons = []

    if columns["iq_columns"]:
        score += 4
        reasons.append("IQ-related columns")

    if len(columns["iq_columns"]) >= 2:
        score += 3
        reasons.append("multiple IQ columns")

    if columns["label_columns"]:
        score += 3
        reasons.append("label/modulation column")

    if columns["snr_columns"]:
        score += 2
        reasons.append("SNR column")

    if columns["rf_related_columns"]:
        score += 1
        reasons.append("RF-related fields")

    filename_lower = path.name.lower()

    rf_filename_terms = [
        "radioml",
        "radio",
        "rf",
        "iq",
        "signal",
        "modulation",
        "wireless",
        "spectrum",
        "signal",
        "gnuradio",
    ]

    filename_matches = [term for term in rf_filename_terms if term in filename_lower]

    if filename_matches:
        score += 2
        reasons.append("RF-related filename")

    if score >= 7:
        category = "HIGH_PRIORITY_RF"

    elif score >= 4:
        category = "POSSIBLE_RF"

    elif score >= 2:
        category = "REVIEW"

    else:
        category = "LIKELY_NON_RF"

    return {
        "category": category,
        "score": score,
        "reasons": reasons,
        "columns": columns,
        "column_count": len(header),
        "sample_row_count": len(sample_rows),
    }


# ============================================================
# HDF5 INSPECTION
# ============================================================


def inspect_hdf5(path: Path):

    result = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return result

    try:

        with h5py.File(path, "r") as h5:

            result["keys"] = list(h5.keys())

            result["datasets"] = {}

            for key in h5.keys():

                item = h5[key]

                if hasattr(item, "shape"):

                    result["datasets"][key] = {
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                    }

            result["attributes"] = {
                str(key): str(value) for key, value in h5.attrs.items()
            }

    except Exception as exc:

        result["error"] = str(exc)

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("COG-EW RF DATASET CANDIDATE PROFILING")
    print("=" * 70)

    csv_files = []

    for root in DATA_ROOTS:

        if not root.exists():

            print(f"[WARNING] Dataset root does not exist: {root}")

            continue

        print(f"\nScanning:\n{root}")

        for path in root.rglob("*.csv"):

            csv_files.append(path)

    # Remove duplicates
    csv_files = sorted(set(csv_files))

    print(f"\nCSV files discovered: {len(csv_files):,}")

    # --------------------------------------------------------
    # Profile CSV files
    # --------------------------------------------------------

    profiles = []

    category_counts = Counter()

    header_counts = Counter()

    errors = []

    for index, path in enumerate(csv_files, start=1):

        header, samples, error = read_csv_header_and_sample(path)

        if error:

            errors.append(
                {
                    "path": str(path),
                    "error": error,
                }
            )

            continue

        classification = classify_dataset(
            path=path,
            header=header,
            sample_rows=samples,
        )

        profile = {
            "path": str(path),
            "filename": path.name,
            "parent": str(path.parent),
            "header": header,
            "classification": classification,
        }

        profiles.append(profile)

        category_counts[classification["category"]] += 1

        header_signature = tuple(normalize_column_name(column) for column in header)

        header_counts[header_signature] += 1

        if index % 1000 == 0:

            print(f"  Processed " f"{index:,}/{len(csv_files):,}")

    # --------------------------------------------------------
    # RadioML HDF5
    # --------------------------------------------------------

    radioml_files = []

    for root in DATA_ROOTS:

        if not root.exists():
            continue

        for path in root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"):

            radioml_files.append(path)

    hdf5_profiles = [inspect_hdf5(path) for path in sorted(set(radioml_files))]

    # --------------------------------------------------------
    # High priority files
    # --------------------------------------------------------

    high_priority = [
        item
        for item in profiles
        if item["classification"]["category"] == "HIGH_PRIORITY_RF"
    ]

    possible_rf = [
        item for item in profiles if item["classification"]["category"] == "POSSIBLE_RF"
    ]

    review = [
        item for item in profiles if item["classification"]["category"] == "REVIEW"
    ]

    # --------------------------------------------------------
    # Header patterns
    # --------------------------------------------------------

    common_headers = []

    for signature, count in header_counts.most_common(30):

        common_headers.append(
            {
                "count": count,
                "columns": list(signature),
            }
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "purpose": (
            "Profile CSV candidates before cleaning, "
            "standardization, merging, augmentation, "
            "and final model training."
        ),
        "csv_count": len(csv_files),
        "profiled_csv_count": len(profiles),
        "errors": errors,
        "category_counts": dict(category_counts),
        "high_priority_rf_count": len(high_priority),
        "possible_rf_count": len(possible_rf),
        "review_count": len(review),
        "common_headers": common_headers,
        "high_priority_rf": high_priority,
        "possible_rf": possible_rf,
        "review": review,
        "radioml_hdf5": hdf5_profiles,
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

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RF DATASET PROFILING SUMMARY")
    print("=" * 70)

    print(f"\nCSV files profiled: " f"{len(profiles):,}")

    print(f"HIGH_PRIORITY_RF: " f"{len(high_priority):,}")

    print(f"POSSIBLE_RF: " f"{len(possible_rf):,}")

    print(f"REVIEW: " f"{len(review):,}")

    print(f"LIKELY_NON_RF: " f"{category_counts.get('LIKELY_NON_RF', 0):,}")

    print(f"CSV read errors: " f"{len(errors):,}")

    print("\nMost common CSV structures:")

    for item in common_headers[:10]:

        columns = ", ".join(item["columns"])

        print(f"  {item['count']:>6,} files | " f"{columns[:150]}")

    print("\nRadioML HDF5 files:")

    for item in hdf5_profiles:

        print(f"  {item['path']}")

        if "datasets" in item:

            for name, info in item["datasets"].items():

                print(f"    {name}: " f"{info['shape']}")

    print("\nReport:")
    print(REPORT_PATH)

    print("\n" + "=" * 70)
    print("RF DATASET PROFILING COMPLETE")
    print("=" * 70)

    print("\nIMPORTANT:")

    print("No CSV files were modified.")

    print("No datasets were cleaned.")

    print("No datasets were merged.")

    print("No augmentation was performed.")

    print("No model training was performed.")


if __name__ == "__main__":
    main()
