from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INDEX_PATH = Path(r"D:\COG-EW\data\processed\radioml" r"\radioml_index.npz")
OUTPUT_DIR = Path(r"D:\COG-EW\data\processed\radioml")

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def pure_numpy_stratified_split(
    indices: np.ndarray,
    strata: np.ndarray,
    test_ratio: float,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy stratified split function.
    Bypasses sklearn binary DLL load dependencies.
    """
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

    print("=" * 70)
    print("COG-EW RADIOML TRAIN/VAL/TEST SPLIT (PURE NUMPY)")
    print("=" * 70)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found:\n{INDEX_PATH}")

    data = np.load(INDEX_PATH)
    indices = data["sample_index"]
    class_ids = data["class_id"]
    snr = data["snr"]

    # Combined stratification key
    strata = class_ids.astype(np.int16) * 100 + (snr.astype(np.int16) + 20)

    print(f"\nTotal samples: {len(indices):,}")
    print(f"Unique strata: {len(np.unique(strata))}")

    # ------------------------------------------------------
    # Train + temporary set split (30% temp)
    # ------------------------------------------------------
    temp_ratio = VALIDATION_SIZE + TEST_SIZE  # 0.30
    train_idx, temp_idx = pure_numpy_stratified_split(
        indices,
        strata,
        test_ratio=temp_ratio,
        random_seed=RANDOM_SEED,
    )

    # ------------------------------------------------------
    # Validation vs Test split (50% of temp set = 15% each)
    # ------------------------------------------------------
    temp_strata = strata[temp_idx]
    val_idx, test_idx = pure_numpy_stratified_split(
        temp_idx,
        temp_strata,
        test_ratio=0.50,
        random_seed=RANDOM_SEED,
    )

    # ------------------------------------------------------
    # Sort indices for efficient contiguous HDF5 reading
    # ------------------------------------------------------
    train_idx = np.sort(train_idx).astype(np.int32)
    val_idx = np.sort(val_idx).astype(np.int32)
    test_idx = np.sort(test_idx).astype(np.int32)

    # ------------------------------------------------------
    # Save split arrays
    # ------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "train_indices.npy", train_idx)
    np.save(OUTPUT_DIR / "val_indices.npy", val_idx)
    np.save(OUTPUT_DIR / "test_indices.npy", test_idx)

    report = {
        "random_seed": RANDOM_SEED,
        "train": {
            "samples": int(len(train_idx)),
            "percentage": float(len(train_idx) / len(indices) * 100),
        },
        "validation": {
            "samples": int(len(val_idx)),
            "percentage": float(len(val_idx) / len(indices) * 100),
        },
        "test": {
            "samples": int(len(test_idx)),
            "percentage": float(len(test_idx) / len(indices) * 100),
        },
        "total": int(len(train_idx) + len(val_idx) + len(test_idx)),
    }

    with (OUTPUT_DIR / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(
        f"Train      : {len(train_idx):,} ({len(train_idx)/len(indices)*100:.2f}%)"
    )
    print(
        f"Validation : {len(val_idx):,} ({len(val_idx)/len(indices)*100:.2f}%)"
    )
    print(
        f"Test       : {len(test_idx):,} ({len(test_idx)/len(indices)*100:.2f}%)"
    )
    print(f"Total      : {len(indices):,}")

    print("\n[PASS] RadioML splits created successfully.")


if __name__ == "__main__":
    main()
