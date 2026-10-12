"""
COG-EW — RadioML 2016.10A -> Common 128-IQ Adapter
===================================================

Purpose
-------
Create a SAFE, auditable intermediate dataset from RadioML 2016.10A
without modifying the original PKL.

Important design decision
-------------------------
Only exact/defensible modulation overlaps with RadioML 2018.01A are
kept for the first cross-dataset experiment:

    8PSK  -> 8PSK
    BPSK  -> BPSK
    QPSK  -> QPSK
    QAM16 -> 16QAM
    QAM64 -> 64QAM

The following RadioML 2016 classes are intentionally EXCLUDED:
    AM-DSB, AM-SSB
because RadioML 2018 distinguishes WC/SC subclasses and the 2016
labels do not provide enough information to determine that mapping.

GFSK, CPFSK, PAM4 and WBFM are also excluded from this first common
label space because there is no exact 2018 counterpart.

Input:
    RML2016.10a_dict.pkl

Output:
    data/intermediate/rml2016_common128.h5
    reports/rml2016_adaptation_audit.json

The common representation is 128 IQ samples:
    [N, 128, 2]

RadioML 2016 is already [N, 2, 128], so no interpolation/resampling
is performed. This avoids inventing signal samples.

This script does NOT modify the source PKL.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(r"D:\COG-EW")
DATASET_PATH = Path(
    r"D:\Datasets\DeepSig Dataset RadioML 2016.10A\RML2016.10a_dict.pkl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_H5 = OUTPUT_DIR / "rml2016_common128.h5"
REPORT_JSON = REPORT_DIR / "rml2016_adaptation_audit.json"

# Exact semantic overlap with RadioML 2018.01A.
CLASS_MAP = {
    "8PSK": "8PSK",
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "QAM16": "16QAM",
    "QAM64": "64QAM",
}

TARGET_CLASSES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

TARGET_CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_sample_array(arr: np.ndarray, key) -> tuple[np.ndarray, dict]:
    """
    Convert one RML2016 value from [N, 2, 128] to [N, 128, 2].

    Returns:
        converted array
        quality statistics
    """
    arr = np.asarray(arr)

    if arr.ndim != 3:
        fail(f"Unexpected dimensions for key {key}: {arr.shape}")

    if arr.shape[1:] != (2, 128):
        fail(f"Unexpected sample shape for key {key}: {arr.shape}")

    if arr.dtype.kind not in "fc":
        fail(f"Unexpected dtype for key {key}: {arr.dtype}")

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    finite = np.isfinite(arr)

    if finite.any():
        finite_values = arr[finite]
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        mean_value = float(finite_values.mean())
        std_value = float(finite_values.std())
    else:
        min_value = max_value = mean_value = std_value = None

    # Channel-first -> time-first.
    converted = np.transpose(arr, (0, 2, 1)).astype(np.float32, copy=False)

    quality = {
        "shape_original": list(arr.shape),
        "shape_converted": list(converted.shape),
        "dtype": str(arr.dtype),
        "nan_values": nan_count,
        "infinite_values": inf_count,
        "minimum": min_value,
        "maximum": max_value,
        "mean": mean_value,
        "std": std_value,
    }

    return converted, quality


def main() -> None:
    print("=" * 70)
    print("COG-EW RADIO ML 2016.10A -> COMMON 128-IQ ADAPTER")
    print("=" * 70)

    if not DATASET_PATH.exists():
        fail(f"Dataset not found:\n{DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nInput dataset:\n{DATASET_PATH}")
    print(f"Output dataset:\n{OUTPUT_H5}")

    print("\n" + "=" * 70)
    print("LOADING SOURCE DATASET")
    print("=" * 70)

    with DATASET_PATH.open("rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    if not isinstance(dataset, dict):
        fail(f"Expected dict, received {type(dataset)}")

    print(f"[OK] Loaded dictionary with {len(dataset):,} keys.")

    # ------------------------------------------------------------------
    # Discover classes and SNRs
    # ------------------------------------------------------------------
    source_class_counts = Counter()
    source_snr_counts = Counter()
    source_class_snr_counts = Counter()

    for key, value in dataset.items():
        if not isinstance(key, tuple) or len(key) != 2:
            fail(f"Unexpected key format: {key!r}")

        modulation, snr = key
        source_class_counts[str(modulation)] += len(value)
        source_snr_counts[int(snr)] += len(value)
        source_class_snr_counts[(str(modulation), int(snr))] += len(value)

    print("\nSource classes:")
    for cls, count in sorted(source_class_counts.items()):
        print(f"  {cls:10s}: {count:8,d}")

    # ------------------------------------------------------------------
    # Collect only exact-overlap classes
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASS MAPPING")
    print("=" * 70)

    for src, dst in CLASS_MAP.items():
        print(f"  {src:8s} -> {dst}")

    excluded = sorted(c for c in source_class_counts if c not in CLASS_MAP)

    print("\nExcluded classes:")
    for cls in excluded:
        print(f"  [EXCLUDED] {cls}")

    # ------------------------------------------------------------------
    # First pass: calculate size and quality
    # ------------------------------------------------------------------
    kept_samples = 0
    rejected_nan = 0
    rejected_inf = 0
    rejected_shape = 0

    kept_class_counts = Counter()
    kept_snr_counts = Counter()
    kept_class_snr_counts = Counter()

    quality_by_class = defaultdict(list)

    print("\n" + "=" * 70)
    print("VALIDATING COMPATIBLE SAMPLES")
    print("=" * 70)

    for key, value in dataset.items():
        modulation, snr = str(key[0]), int(key[1])

        if modulation not in CLASS_MAP:
            continue

        arr = np.asarray(value)

        if arr.ndim != 3 or arr.shape[1:] != (2, 128):
            rejected_shape += int(arr.shape[0]) if arr.ndim > 0 else 1
            continue

        nan_mask = np.isnan(arr).any(axis=(1, 2))
        inf_mask = np.isinf(arr).any(axis=(1, 2))

        valid_mask = ~(nan_mask | inf_mask)

        rejected_nan += int(nan_mask.sum())
        rejected_inf += int(inf_mask.sum())

        valid_count = int(valid_mask.sum())
        kept_samples += valid_count

        kept_class_counts[CLASS_MAP[modulation]] += valid_count
        kept_snr_counts[snr] += valid_count
        kept_class_snr_counts[(CLASS_MAP[modulation], snr)] += valid_count

        if valid_count:
            valid = arr[valid_mask]
            quality_by_class[modulation].append(
                {
                    "samples": valid_count,
                    "minimum": float(valid.min()),
                    "maximum": float(valid.max()),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                }
            )

    print(f"Compatible valid samples: {kept_samples:,}")
    print(f"Rejected NaN samples:      {rejected_nan:,}")
    print(f"Rejected Inf samples:      {rejected_inf:,}")
    print(f"Rejected shape samples:    {rejected_shape:,}")

    if kept_samples == 0:
        fail("No compatible samples survived validation.")

    # ------------------------------------------------------------------
    # Write HDF5
    # ------------------------------------------------------------------
    if OUTPUT_H5.exists():
        OUTPUT_H5.unlink()
        print(f"\n[INFO] Existing intermediate file replaced:\n{OUTPUT_H5}")

    print("\n" + "=" * 70)
    print("WRITING ADAPTED DATASET")
    print("=" * 70)

    report = {
        "project": "COG-EW",
        "source_dataset": str(DATASET_PATH),
        "output_dataset": str(OUTPUT_H5),
        "source_format": "RadioML 2016.10A dictionary PKL",
        "output_format": "HDF5",
        "source_sample_shape": [2, 128],
        "output_sample_shape": [128, 2],
        "target_classes": TARGET_CLASSES,
        "class_mapping": CLASS_MAP,
        "excluded_classes": excluded,
        "excluded_class_reason": (
            "No exact/common semantic label in RadioML 2018.01A "
            "for the first conservative merge experiment."
        ),
        "normalization": "none at adaptation stage",
        "augmentation": False,
        "resampling": False,
        "source_modified": False,
        "nan_rejected": rejected_nan,
        "inf_rejected": rejected_inf,
        "shape_rejected": rejected_shape,
    }

    with h5py.File(OUTPUT_H5, "w") as out:
        x_ds = out.create_dataset(
            "X",
            shape=(kept_samples, 128, 2),
            dtype="float32",
            chunks=(min(1024, kept_samples), 128, 2),
            compression="gzip",
            compression_opts=4,
        )

        y_ds = out.create_dataset(
            "Y",
            shape=(kept_samples,),
            dtype="int64",
            compression="gzip",
            compression_opts=4,
        )

        z_ds = out.create_dataset(
            "Z",
            shape=(kept_samples,),
            dtype="int64",
            compression="gzip",
            compression_opts=4,
        )

        cursor = 0

        for key, value in dataset.items():
            modulation, snr = str(key[0]), int(key[1])

            if modulation not in CLASS_MAP:
                continue

            arr = np.asarray(value)

            if arr.ndim != 3 or arr.shape[1:] != (2, 128):
                continue

            nan_mask = np.isnan(arr).any(axis=(1, 2))
            inf_mask = np.isinf(arr).any(axis=(1, 2))
            valid_mask = ~(nan_mask | inf_mask)

            if not valid_mask.any():
                continue

            # Keep only valid samples.
            valid = arr[valid_mask]

            # [N, 2, 128] -> [N, 128, 2]
            valid = np.transpose(valid, (0, 2, 1)).astype(np.float32, copy=False)

            n = valid.shape[0]
            end = cursor + n

            x_ds[cursor:end] = valid
            y_ds[cursor:end] = TARGET_CLASS_TO_ID[CLASS_MAP[modulation]]
            z_ds[cursor:end] = snr

            cursor = end

        # Metadata
        out.attrs["project"] = "COG-EW"
        out.attrs["source_dataset"] = str(DATASET_PATH)
        out.attrs["sample_length"] = 128
        out.attrs["channels"] = 2
        out.attrs["class_count"] = len(TARGET_CLASSES)
        out.attrs["classes"] = json.dumps(TARGET_CLASSES)
        out.attrs["class_mapping"] = json.dumps(CLASS_MAP)
        out.attrs["augmentation"] = "none"
        out.attrs["normalization"] = "none"

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OUTPUT VERIFICATION")
    print("=" * 70)

    with h5py.File(OUTPUT_H5, "r") as h5:
        x_shape = tuple(h5["X"].shape)
        y_shape = tuple(h5["Y"].shape)
        z_shape = tuple(h5["Z"].shape)

        x = h5["X"][: min(5000, kept_samples)]

        print(f"X shape: {x_shape}")
        print(f"Y shape: {y_shape}")
        print(f"Z shape: {z_shape}")

        print(f"NaN in verification sample: {bool(np.isnan(x).any())}")
        print(f"Inf in verification sample: {bool(np.isinf(x).any())}")
        print(f"Minimum: {float(x.min()):.8f}")
        print(f"Maximum: {float(x.max()):.8f}")
        print(f"Mean:     {float(x.mean()):.8f}")
        print(f"Std:      {float(x.std()):.8f}")

        if x_shape != (kept_samples, 128, 2):
            fail(f"Unexpected output X shape: {x_shape}")

        if y_shape != (kept_samples,):
            fail(f"Unexpected output Y shape: {y_shape}")

        if z_shape != (kept_samples,):
            fail(f"Unexpected output Z shape: {z_shape}")

    report["output_shape"] = {
        "X": list(x_shape),
        "Y": list(y_shape),
        "Z": list(z_shape),
    }
    report["kept_samples"] = kept_samples
    report["kept_class_counts"] = dict(kept_class_counts)
    report["kept_snr_counts"] = {str(k): v for k, v in sorted(kept_snr_counts.items())}
    report["source_class_counts"] = dict(source_class_counts)
    report["source_snr_counts"] = {
        str(k): v for k, v in sorted(source_snr_counts.items())
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("ADAPTATION COMPLETE")
    print("=" * 70)

    print(f"\n[OK] Intermediate dataset:")
    print(OUTPUT_H5)

    print("\n[OK] Audit report:")
    print(REPORT_JSON)

    print("\nKept classes:")
    for cls in TARGET_CLASSES:
        print(f"  {cls:8s}: {kept_class_counts[cls]:,}")

    print("\n[PASS] Original RadioML 2016.10A PKL was NOT modified.")
    print("[PASS] No augmentation performed.")
    print("[PASS] No model training performed.")
    print("[PASS] No RadioML 2018 samples were modified.")
    print("\nNext stage: independently clean/harmonize RadioML 2018 and")
    print("construct a leakage-safe common training/validation/test pipeline.")


if __name__ == "__main__":
    main()
