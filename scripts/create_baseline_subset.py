from __future__ import annotations

import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(r"D:\COG-EW")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"

INDEX_PATH = PROCESSED_DIR / "radioml_index.npz"

OUTPUT_PATH = PROCESSED_DIR / "baseline_train_indices.npy"

SEED = 42

TARGET_SNRS = [
    -20,
    -10,
    0,
    10,
    20,
]

SAMPLES_PER_CLASS_SNR = 500


def main():

    print("=" * 70)
    print("COG-EW BASELINE TRAINING SUBSET")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    data = np.load(INDEX_PATH)

    indices = data["sample_index"]

    class_ids = data["class_id"]

    snr = data["snr"]

    selected = []

    for class_id in range(24):

        for snr_value in TARGET_SNRS:

            mask = (class_ids == class_id) & (snr == snr_value)

            candidates = indices[mask]

            if len(candidates) < (SAMPLES_PER_CLASS_SNR):

                raise RuntimeError(
                    f"Not enough samples for " f"class={class_id}, " f"SNR={snr_value}"
                )

            chosen = rng.choice(
                candidates,
                size=SAMPLES_PER_CLASS_SNR,
                replace=False,
            )

            selected.append(chosen)

    selected = np.concatenate(selected)

    rng.shuffle(selected)

    selected = selected.astype(np.int32)

    np.save(
        OUTPUT_PATH,
        selected,
    )

    print(f"\nClasses: 24")

    print(f"SNR levels: " f"{TARGET_SNRS}")

    print(f"Samples/class/SNR: " f"{SAMPLES_PER_CLASS_SNR}")

    print(f"\nTotal samples: " f"{len(selected):,}")

    print(f"\nOutput:")

    print(OUTPUT_PATH)

    print("\n[PASS] Baseline subset created.")


if __name__ == "__main__":
    main()
