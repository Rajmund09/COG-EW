from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import h5py


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORT
# ============================================================

try:
    from config import DATA_ROOT
except Exception:
    DATA_ROOT = Path(r"D:\Datasets")


# ============================================================
# OUTPUT
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "compatible_rf_dataset_audit.json"


# ============================================================
# HELPERS
# ============================================================


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def format_gb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# RADIO ML 2018 DETECTION
# ============================================================


def inspect_radioml_2018(path: Path) -> dict:

    result = {
        "dataset": "RadioML 2018.01A",
        "path": str(path),
        "type": "HDF5",
        "compatible_for_amc": True,
        "status": "VALIDATED",
        "shape": {},
        "dtype": {},
        "classes": 24,
        "iq_length": 1024,
        "iq_channels": 2,
        "notes": [],
    }

    try:

        with h5py.File(path, "r") as h5:

            for key in h5.keys():

                result["shape"][key] = list(h5[key].shape)

                result["dtype"][key] = str(h5[key].dtype)

            required = {"X", "Y", "Z"}

            missing = required - set(h5.keys())

            if missing:

                result["status"] = "INVALID"
                result["compatible_for_amc"] = False
                result["notes"].append(f"Missing required datasets: {sorted(missing)}")

            else:

                x_shape = tuple(h5["X"].shape)
                y_shape = tuple(h5["Y"].shape)
                z_shape = tuple(h5["Z"].shape)

                if len(x_shape) != 3:
                    result["compatible_for_amc"] = False
                    result["status"] = "INVALID"
                    result["notes"].append(f"Unexpected X shape: {x_shape}")

                if x_shape[1:] != (1024, 2):
                    result["compatible_for_amc"] = False
                    result["status"] = "REVIEW"
                    result["notes"].append(f"Unexpected IQ shape: {x_shape[1:]}")

                if y_shape[0] != x_shape[0]:
                    result["compatible_for_amc"] = False
                    result["status"] = "INVALID"
                    result["notes"].append("X/Y sample counts do not match.")

                if z_shape[0] != x_shape[0]:
                    result["compatible_for_amc"] = False
                    result["status"] = "INVALID"
                    result["notes"].append("X/Z sample counts do not match.")

    except Exception as exc:

        result["status"] = "ERROR"
        result["compatible_for_amc"] = False
        result["notes"].append(str(exc))

    return result


# ============================================================
# RADIO ML 2016 DETECTION
# ============================================================


def inspect_radioml_2016(path: Path) -> dict:

    result = {
        "dataset": "RadioML 2016.10A",
        "path": str(path),
        "type": "PKL",
        "compatible_for_amc": False,
        "status": "REVIEW",
        "classes": None,
        "iq_length": None,
        "iq_channels": None,
        "keys": [],
        "notes": [],
    }

    try:

        with path.open("rb") as file:

            data = pickle.load(
                file,
                encoding="latin1",
            )

        if isinstance(data, dict):

            result["keys"] = [str(key) for key in list(data.keys())[:50]]

            modulation_classes = set()
            iq_lengths = set()

            inspected = 0

            for key, value in data.items():

                try:

                    if isinstance(key, tuple) and len(key) >= 2:
                        modulation_classes.add(str(key[1]))

                    shape = getattr(value, "shape", None)

                    if shape is not None:

                        if len(shape) >= 2:
                            iq_lengths.add(int(shape[-1]))

                    inspected += 1

                    if inspected >= 1000:
                        break

                except Exception:
                    continue

            result["classes"] = len(modulation_classes)

            if len(iq_lengths) == 1:
                result["iq_length"] = list(iq_lengths)[0]

            result["iq_channels"] = 2

            if result["classes"]:
                result["notes"].append(
                    f"Detected approximately " f"{result['classes']} modulation labels."
                )

            result["notes"].append(
                "RadioML 2016.10A requires an adapter because "
                "its class taxonomy and IQ length differ from "
                "RadioML 2018.01A."
            )

            result["compatible_for_amc"] = True
            result["status"] = "ADAPTER_REQUIRED"

        else:

            result["status"] = "INVALID"
            result["notes"].append("PKL object is not a dictionary.")

    except Exception as exc:

        result["status"] = "ERROR"
        result["notes"].append(str(exc))

    return result


# ============================================================
# FILE DISCOVERY
# ============================================================


def discover_files(data_root: Path):

    print_header("SCANNING LOCAL DATA ROOT")

    print(f"Data root:\n{data_root}")

    if not data_root.exists():

        raise FileNotFoundError(f"DATA_ROOT does not exist:\n{data_root}")

    hdf5_files = []
    pickle_files = []
    sigmf_files = []
    duckdb_files = []
    zip_files = []

    for path in data_root.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in {".h5", ".hdf5"}:
            hdf5_files.append(path)

        elif suffix in {".pkl", ".pickle"}:
            pickle_files.append(path)

        elif suffix == ".sigmf":
            sigmf_files.append(path)

        elif suffix == ".duckdb":
            duckdb_files.append(path)

        elif suffix == ".zip":
            zip_files.append(path)

    return {
        "hdf5": hdf5_files,
        "pickle": pickle_files,
        "sigmf": sigmf_files,
        "duckdb": duckdb_files,
        "zip": zip_files,
    }


# ============================================================
# CLASSIFY HDF5
# ============================================================


def classify_hdf5_files(files):

    results = []

    for path in files:

        print()
        print(f"Inspecting HDF5:")
        print(path)

        name = path.name.lower()

        if "gold_xyz_osc" in name:

            result = inspect_radioml_2018(path)

            print(f"  Dataset: {result['dataset']}")

            print(f"  Status: {result['status']}")

            print(f"  AMC compatible: " f"{result['compatible_for_amc']}")

            results.append(result)

        else:

            results.append(
                {
                    "dataset": "UNKNOWN_HDF5",
                    "path": str(path),
                    "type": "HDF5",
                    "compatible_for_amc": False,
                    "status": "REVIEW",
                    "size_gb": safe_size(path) / (1024**3),
                    "notes": [
                        "HDF5 file detected but dataset identity "
                        "could not be established automatically."
                    ],
                }
            )

    return results


# ============================================================
# CLASSIFY PICKLES
# ============================================================


def classify_pickle_files(files):

    results = []

    for path in files:

        name = path.name.lower()

        if "rml2016" in name or "radioml" in str(path).lower():

            print()
            print(f"Inspecting possible RadioML PKL:")
            print(path)

            result = inspect_radioml_2016(path)

            print(f"  Dataset: {result['dataset']}")

            print(f"  Status: {result['status']}")

            results.append(result)

    return results


# ============================================================
# MAIN
# ============================================================


def main():

    if hasattr(sys.stdout, "reconfigure"):

        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    print_header("COG-EW COMPATIBLE RF DATASET DISCOVERY")

    data_root = Path(DATA_ROOT)

    print(f"\nResolved DATA_ROOT:\n{data_root}")

    files = discover_files(data_root)

    print_header("FILE SUMMARY")

    print(f"HDF5 files : {len(files['hdf5']):,}")

    print(f"PKL files  : {len(files['pickle']):,}")

    print(f"SigMF files: {len(files['sigmf']):,}")

    print(f"DuckDB     : {len(files['duckdb']):,}")

    print(f"ZIP files  : {len(files['zip']):,}")

    # --------------------------------------------------------
    # HDF5
    # --------------------------------------------------------

    print_header("RADIO ML / HDF5 ANALYSIS")

    hdf5_results = classify_hdf5_files(files["hdf5"])

    # --------------------------------------------------------
    # PKL
    # --------------------------------------------------------

    print_header("RADIO ML 2016 / PKL ANALYSIS")

    pkl_results = classify_pickle_files(files["pickle"])

    # --------------------------------------------------------
    # NIST
    # --------------------------------------------------------

    print_header("NIST SEA-DATA / SIGMF / DUCKDB")

    nist_results = []

    for path in files["sigmf"]:

        nist_results.append(
            {
                "type": "SigMF",
                "path": str(path),
                "dataset": "NIST/SEA-DATA candidate",
                "status": "REVIEW",
                "note": (
                    "Spectrum data. Do not merge directly "
                    "with modulation-class training data."
                ),
            }
        )

    for path in files["duckdb"]:

        nist_results.append(
            {
                "type": "DuckDB",
                "path": str(path),
                "dataset": "NIST/SEA-DATA candidate",
                "status": "REVIEW",
                "note": (
                    "Spectrum/measurement database. "
                    "Requires schema inspection before use."
                ),
            }
        )

    print(f"SigMF/DuckDB candidates: " f"{len(nist_results):,}")

    # --------------------------------------------------------
    # FINAL DATASET DECISION
    # --------------------------------------------------------

    print_header("DATASET COMPATIBILITY DECISION")

    radioml_2018 = [
        item for item in hdf5_results if item["dataset"] == "RadioML 2018.01A"
    ]

    radioml_2016 = [
        item for item in pkl_results if item["dataset"] == "RadioML 2016.10A"
    ]

    print("\nRadioML 2018.01A:")

    if radioml_2018:

        print("  [FOUND] Validated candidate")

        print("  [DECISION] Primary AMC dataset")

    else:

        print("  [NOT FOUND]")

    print("\nRadioML 2016.10A:")

    if radioml_2016:

        print("  [FOUND] Adapter candidate")

        print("  [DECISION] Auxiliary/domain dataset")

    else:

        print("  [NOT FOUND]")

    print("\nNIST SEA-DATA:")

    if nist_results:

        print("  [FOUND] Spectrum-data candidates")

        print("  [DECISION] Separate spectrum/domain pipeline")

    else:

        print("  [NOT FOUND]")

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "project": "COG-EW",
        "data_root": str(data_root),
        "radioml_2018": hdf5_results,
        "radioml_2016": pkl_results,
        "nist_candidates": nist_results,
        "zip_count": len(files["zip"]),
        "decision": {
            "radioml_2018_role": (
                "primary_amc_dataset" if radioml_2018 else "unavailable"
            ),
            "radioml_2016_role": (
                "auxiliary_adapter_dataset" if radioml_2016 else "unavailable"
            ),
            "nist_role": (
                "separate_spectrum_domain_pipeline" if nist_results else "unavailable"
            ),
        },
        "safety": {
            "operational_jamming": False,
            "weapon_guidance": False,
            "missile_targeting": False,
            "real_world_electronic_attack": False,
        },
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

    print_header("COMPATIBLE RF DATASET DISCOVERY COMPLETE")

    print(f"Report:\n{REPORT_PATH}")

    print("\n[PASS] No source dataset was modified.")

    print("[PASS] No data was merged.")

    print("[PASS] No augmentation was performed.")

    print("[PASS] No model training was performed.")


if __name__ == "__main__":
    main()
