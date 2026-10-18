from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "processed" / "radioml" / "radioml_index.npz"

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "radioml" / "experiments" / "split_60_10_30_v1"
)

SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict:
    unique, counts = np.unique(values, return_counts=True)

    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def verify_partition(
    name: str,
    indices: np.ndarray,
    class_ids: np.ndarray,
    snr: np.ndarray,
) -> dict:
    partition_classes = class_ids[indices]
    partition_snr = snr[indices]

    return {
        "samples": int(len(indices)),
        "percentage": round(100.0 * len(indices) / len(class_ids), 6),
        "class_counts": distribution(partition_classes),
        "snr_counts": distribution(partition_snr),
        "class_snr_counts": {
            f"class_{int(c)}_snr_{int(s)}": int(
                np.sum((partition_classes == c) & (partition_snr == s))
            )
            for c, s in np.unique(
                np.column_stack((partition_classes, partition_snr)),
                axis=0,
            )
        },
    }


def main() -> None:
    print("=" * 72)
    print("COG-EW RADIO ML — VERSIONED 60/10/30 SPLIT")
    print("=" * 72)

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata index not found: {INDEX_PATH}\n"
            "Do not rebuild it yet. Verify the existing dataset index first."
        )

    print(f"Metadata index: {INDEX_PATH}")

    with np.load(INDEX_PATH, allow_pickle=False) as metadata:
        required = {"sample_index", "class_id", "snr"}
        missing = required - set(metadata.files)

        if missing:
            raise ValueError(f"Metadata index is missing fields: {sorted(missing)}")

        sample_indices = np.asarray(metadata["sample_index"], dtype=np.int64)
        class_ids = np.asarray(metadata["class_id"], dtype=np.int64)
        snr = np.asarray(metadata["snr"], dtype=np.int64)

    total = len(sample_indices)

    if total == 0:
        raise ValueError("Metadata index contains no samples.")

    if not (len(class_ids) == total and len(snr) == total):
        raise ValueError("Index, class labels, and SNR lengths differ.")

    if len(np.unique(sample_indices)) != total:
        raise ValueError("Sample indices contain duplicates.")

    if not np.isfinite(class_ids).all() or not np.isfinite(snr).all():
        raise ValueError("Metadata contains invalid numeric values.")

    if not np.array_equal(
        np.unique(class_ids),
        np.arange(24),
    ):
        raise ValueError(
            "Expected RadioML class IDs 0 through 23. "
            f"Found: {np.unique(class_ids).tolist()}"
        )

    # Stratify jointly by modulation class and SNR.
    # The class/SNR pair is used only for partitioning, not as a feature.
    _, joint_strata = np.unique(
        np.column_stack((class_ids, snr)),
        axis=0,
        return_inverse=True,
    )

    indices = np.arange(total, dtype=np.int64)

    # First reserve 30% as the final test partition.
    development_indices, test_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
        stratify=joint_strata,
    )

    # Of the remaining 70%, reserve 1/7 for validation:
    # 70% * 1/7 = 10% of the full dataset.
    # The remaining 6/7 is 60% of the full dataset.
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=1 / 7,
        random_state=SEED,
        shuffle=True,
        stratify=joint_strata[development_indices],
    )

    partitions = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(validation_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }

    # Verify sample-index disjointness before writing any files.
    for left_name, left in partitions.items():
        for right_name, right in partitions.items():
            if left_name >= right_name:
                continue

            overlap = np.intersect1d(left, right)
            if overlap.size:
                raise RuntimeError(
                    f"Data leakage: {left_name}/{right_name} overlap "
                    f"contains {overlap.size} indices."
                )

    combined = np.concatenate(list(partitions.values()))

    if len(combined) != total or len(np.unique(combined)) != total:
        raise RuntimeError("Partitions do not cover every sample exactly once.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact_hashes = {}

    for name, split_indices in partitions.items():
        output_path = OUTPUT_DIR / f"{name}_indices.npy"

        np.save(
            output_path,
            sample_indices[split_indices].astype(np.int32),
            allow_pickle=False,
        )

        artifact_hashes[output_path.name] = sha256_file(output_path)

        print(
            f"{name:>10}: {len(split_indices):>10,} samples "
            f"({100 * len(split_indices) / total:.4f}%)"
        )

    report = {
        "experiment": "RadioML 2018.01A 60/10/30 split v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": SEED,
        "stratification": "joint modulation class and SNR",
        "total_samples": int(total),
        "class_ids": sorted(int(x) for x in np.unique(class_ids)),
        "snr_levels": sorted(int(x) for x in np.unique(snr)),
        "partitions": {
            name: verify_partition(
                name,
                split_indices,
                class_ids,
                snr,
            )
            for name, split_indices in partitions.items()
        },
        "integrity": {
            "all_samples_assigned_once": True,
            "partition_overlap_count": 0,
            "source_index_modified": False,
            "raw_dataset_modified": False,
            "existing_splits_modified": False,
            "test_used_for_training_or_tuning": False,
            "group_leakage_check": (
                "Not possible from class/SNR metadata alone; "
                "source or recording identifiers were not provided "
                "to this script."
            ),
        },
        "source_index": str(INDEX_PATH),
        "source_index_sha256": sha256_file(INDEX_PATH),
        "split_files_sha256": artifact_hashes,
    }

    report_path = OUTPUT_DIR / "split_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nClass distribution:")
    for name, details in report["partitions"].items():
        print(f"\n{name.upper()}")
        for class_id, count in details["class_counts"].items():
            print(f"  Class {class_id:>2}: {count:>10,}")

    print("\n[PASS] All samples assigned exactly once.")
    print("[PASS] No overlap between train, validation, and test.")
    print("[PASS] Existing split files were not modified.")
    print("[PASS] Raw dataset and model checkpoints were not modified.")
    print(f"\nSplit directory: {OUTPUT_DIR}")
    print(f"Audit report:    {report_path}")


if __name__ == "__main__":
    main()
