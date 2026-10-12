from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"
INDEX_PATH = PROCESSED_DIR / "radioml_index.npz"
BASELINE_PATH = PROCESSED_DIR / "baseline_train_indices.npy"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def pure_numpy_stratified_split(
    indices: np.ndarray,
    strata: np.ndarray,
    test_ratio: float,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy stratified split function bypassing sklearn DLL block."""
    rng = np.random.default_rng(random_seed)
    unique_strata = np.unique(strata)

    train_list = []
    test_list = []

    for stratum in unique_strata:
        mask = (strata == stratum)
        stratum_indices = indices[mask].copy()
        rng.shuffle(stratum_indices)

        n_test = int(round(len(stratum_indices) * test_ratio))
        test_list.append(stratum_indices[:n_test])
        train_list.append(stratum_indices[n_test:])

    train_idx = np.concatenate(train_list)
    test_idx = np.concatenate(test_list)
    return train_idx, test_idx


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW BASELINE DATASET SPLIT (PURE NUMPY)")
    print("=" * 70)

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline subset file not found:\n{BASELINE_PATH}")

    index_data = np.load(INDEX_PATH)
    class_ids = index_data["class_id"]
    snr = index_data["snr"]

    baseline_indices = np.load(BASELINE_PATH)
    baseline_classes = class_ids[baseline_indices]
    baseline_snr = snr[baseline_indices]

    strata = baseline_classes.astype(np.int16) * 100 + (
        baseline_snr.astype(np.int16) + 20
    )

    print(f"\nTotal baseline samples: {len(baseline_indices):,}")
    print(f"Strata: {len(np.unique(strata))}")

    # Train (70%) vs Temp (30%)
    train_indices, temp_indices = pure_numpy_stratified_split(
        baseline_indices,
        strata,
        test_ratio=(VAL_RATIO + TEST_RATIO),
        random_seed=SEED,
    )

    # Temp split: 50% Val (15%), 50% Test (15%)
    temp_strata = strata[np.isin(baseline_indices, temp_indices)]
    val_indices, test_indices = pure_numpy_stratified_split(
        temp_indices,
        temp_strata,
        test_ratio=0.50,
        random_seed=SEED,
    )

    rng = np.random.default_rng(SEED)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_indices = train_indices.astype(np.int32)
    val_indices = val_indices.astype(np.int32)
    test_indices = test_indices.astype(np.int32)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "baseline_train_split.npy", train_indices)
    np.save(PROCESSED_DIR / "baseline_val_split.npy", val_indices)
    np.save(PROCESSED_DIR / "baseline_test_split.npy", test_indices)

    report = {
        "seed": SEED,
        "total": int(len(baseline_indices)),
        "train": {
            "samples": int(len(train_indices)),
            "percentage": float(len(train_indices) / len(baseline_indices) * 100),
        },
        "validation": {
            "samples": int(len(val_indices)),
            "percentage": float(len(val_indices) / len(baseline_indices) * 100),
        },
        "test": {
            "samples": int(len(test_indices)),
            "percentage": float(len(test_indices) / len(baseline_indices) * 100),
        },
    }

    with (PROCESSED_DIR / "baseline_split_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(f"Train      : {len(train_indices):,}")
    print(f"Validation : {len(val_indices):,}")
    print(f"Test       : {len(test_indices):,}")
    print(f"Total      : {len(baseline_indices):,}")

    print("\n[PASS] Baseline splits created.")


if __name__ == "__main__":
    main()
