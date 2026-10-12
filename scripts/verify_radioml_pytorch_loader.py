from __future__ import annotations

import os
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
    print("COG-EW PYTORCH RADIOML LOADER VERIFICATION")
    print("=" * 70)

    print(f"\nPyTorch: " f"{torch.__version__}")

    print(f"CUDA available: " f"{torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("\nCreating DataLoaders...")

    train_loader, val_loader, test_loader = create_radioml_loaders(
        hdf5_path=HDF5_PATH,
        processed_dir=PROCESSED_DIR,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    )

    print("\n[PASS] DataLoaders created.")

    print(f"\nTrain samples: " f"{len(train_loader.dataset):,}")

    print(f"Validation samples: " f"{len(val_loader.dataset):,}")

    print(f"Test samples: " f"{len(test_loader.dataset):,}")

    # ------------------------------------------------------
    # First batch
    # ------------------------------------------------------

    print("\nReading first training batch...")

    start = time.perf_counter()

    signals, labels, snr = next(iter(train_loader))

    elapsed = time.perf_counter() - start

    print(f"Batch loading time: " f"{elapsed:.4f} seconds")

    print("\nBatch information:")

    print(f"Signals shape: " f"{tuple(signals.shape)}")

    print(f"Signals dtype: " f"{signals.dtype}")

    print(f"Labels shape: " f"{tuple(labels.shape)}")

    print(f"Labels dtype: " f"{labels.dtype}")

    print(f"SNR shape: " f"{tuple(snr.shape)}")

    print(f"SNR dtype: " f"{snr.dtype}")

    print("\nSignal statistics:")

    print(f"Min: " f"{signals.min().item():.5f}")

    print(f"Max: " f"{signals.max().item():.5f}")

    print(f"Mean: " f"{signals.mean().item():.5f}")

    print(f"Std: " f"{signals.std().item():.5f}")

    print("\nLabels in batch:")

    print(labels.tolist())

    print("\nSNR values in batch:")

    print(snr.tolist())

    # ------------------------------------------------------
    # GPU transfer
    # ------------------------------------------------------

    if torch.cuda.is_available():

        print("\nTesting GPU transfer...")

        device = torch.device("cuda")

        gpu_start = time.perf_counter()

        signals_gpu = signals.to(
            device,
            non_blocking=True,
        )

        labels_gpu = labels.to(
            device,
            non_blocking=True,
        )

        torch.cuda.synchronize()

        gpu_elapsed = time.perf_counter() - gpu_start

        print(f"GPU transfer time: " f"{gpu_elapsed:.4f} seconds")

        print(f"GPU signal shape: " f"{tuple(signals_gpu.shape)}")

        print(f"GPU label shape: " f"{tuple(labels_gpu.shape)}")

        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

    # ------------------------------------------------------
    # Dataset cleanup
    # ------------------------------------------------------

    train_loader.dataset.close()
    val_loader.dataset.close()
    test_loader.dataset.close()

    print("\n" + "=" * 70)

    print("[PASS] PyTorch RadioML loader " "verification completed.")

    print("=" * 70)


if __name__ == "__main__":
    main()
