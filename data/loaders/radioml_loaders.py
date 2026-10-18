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


def create_radioml_loaders(
    hdf5_path: str | Path,
    processed_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
):

    processed_dir = Path(processed_dir)

    train_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=(processed_dir / "train_indices.npy"),
    )

    val_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=(processed_dir / "val_indices.npy"),
    )

    test_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=(processed_dir / "test_indices.npy"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
