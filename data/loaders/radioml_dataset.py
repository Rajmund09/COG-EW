from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RadioMLDataset(Dataset):

    def __init__(
        self,
        hdf5_path: str | Path,
        indices_path: str | Path,
        metadata_path: str | Path | None = None,
        normalize: bool = True,
    ):

        self.hdf5_path = Path(hdf5_path)

        self.indices_path = Path(indices_path)

        self.normalize = normalize

        if not self.hdf5_path.exists():

            raise FileNotFoundError(f"HDF5 file not found:\n" f"{self.hdf5_path}")

        if not self.indices_path.exists():

            raise FileNotFoundError(f"Index file not found:\n" f"{self.indices_path}")

        self.indices = np.load(self.indices_path).astype(np.int64)

        self._h5 = None

        # Optional metadata cache.
        self.class_ids = None
        self.snr_values = None

        if metadata_path is not None:

            metadata_path = Path(metadata_path)

            if metadata_path.exists():

                metadata = np.load(metadata_path)

                all_indices = metadata["sample_index"]

                all_classes = metadata["class_id"]

                all_snr = metadata["snr"]

                # Build lookup arrays.

                self.class_ids = all_classes

                self.snr_values = all_snr

    def _open_hdf5(self):

        if self._h5 is None:

            self._h5 = h5py.File(
                self.hdf5_path,
                "r",
                swmr=True,
            )

    def __len__(self):

        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ):

        self._open_hdf5()

        actual_index = int(self.indices[index])

        # --------------------------------------------------
        # IQ signal
        # --------------------------------------------------

        signal = self._h5["X"][actual_index]

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        # HDF5:
        # (1024, 2)
        #
        # CNN:
        # (2, 1024)

        signal = signal.T

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        if self.normalize:

            mean = np.mean(
                signal,
                axis=1,
                keepdims=True,
            )

            std = np.std(
                signal,
                axis=1,
                keepdims=True,
            )

            signal = (signal - mean) / (std + 1e-8)

        signal = torch.from_numpy(signal.copy())

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        if self.class_ids is not None:

            label = int(self.class_ids[actual_index])

        else:

            label = int(np.argmax(self._h5["Y"][actual_index]))

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        # --------------------------------------------------
        # SNR
        # --------------------------------------------------

        if self.snr_values is not None:

            snr = int(self.snr_values[actual_index])

        else:

            snr = int(np.asarray(self._h5["Z"][actual_index]).reshape(-1)[0])

        snr = torch.tensor(
            snr,
            dtype=torch.int16,
        )

        return (
            signal,
            label,
            snr,
        )

    def close(self):

        if self._h5 is not None:

            self._h5.close()

            self._h5 = None

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass
