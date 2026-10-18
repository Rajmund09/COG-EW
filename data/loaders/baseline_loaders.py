from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from data.loaders.radioml_dataset import (
    RadioMLDataset,
)


def create_baseline_loaders(
    hdf5_path: str | Path,
    processed_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
):

    processed_dir = Path(processed_dir)

    metadata_path = processed_dir / "radioml_index.npz"

    train_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=(processed_dir / "baseline_train_split.npy"),
        metadata_path=metadata_path,
    )

    val_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=(processed_dir / "baseline_val_split.npy"),
        metadata_path=metadata_path,
    )

    test_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=(processed_dir / "baseline_test_split.npy"),
        metadata_path=metadata_path,
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
