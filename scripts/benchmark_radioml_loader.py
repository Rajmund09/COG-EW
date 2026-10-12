from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.loaders.radioml_loaders import (
    create_radioml_loaders,
)

HDF5_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "radioml"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW RADIOML DATA LOADER BENCHMARK")
    print("=" * 70)

    print(f"\nPyTorch: " f"{torch.__version__}")

    print(f"CUDA: " f"{torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

    print("\nCreating loader...")

    train_loader, _, _ = create_radioml_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    )

    iterator = iter(train_loader)

    times = []

    number_of_batches = 10

    print(f"\nBenchmarking " f"{number_of_batches} batches...")

    for batch_number in range(number_of_batches):

        start = time.perf_counter()

        signals, labels, snr = next(iterator)

        elapsed = time.perf_counter() - start

        times.append(elapsed)

        print(
            f"Batch "
            f"{batch_number + 1:02d}: "
            f"{elapsed:.4f} sec | "
            f"shape={tuple(signals.shape)}"
        )

    # Ignore first batch when estimating
    # steady-state performance.

    warm_times = times[1:]

    average_time = sum(warm_times) / len(warm_times)

    samples_per_second = 32 / average_time

    estimated_minutes = 1_789_008 / samples_per_second / 60

    print("\n" + "=" * 70)

    print("BENCHMARK RESULTS")

    print("=" * 70)

    print(f"First batch: " f"{times[0]:.4f} sec")

    print(f"Average subsequent batch: " f"{average_time:.4f} sec")

    print(f"Throughput: " f"{samples_per_second:.2f} samples/sec")

    print(
        f"Estimated time to read "
        f"one full training epoch: "
        f"{estimated_minutes:.2f} minutes"
    )

    print("\n[PASS] Benchmark completed.")


if __name__ == "__main__":
    main()
