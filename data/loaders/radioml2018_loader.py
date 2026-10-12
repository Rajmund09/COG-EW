from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .base_loader import BaseDatasetLoader


class RadioML2018Loader(BaseDatasetLoader):
    """
    Memory-safe loader for RadioML 2018.01A.

    The HDF5 file remains on disk.
    Only requested samples are read.
    """

    def __init__(
        self,
        path: str | Path,
    ):

        super().__init__(path)

        self._file = None

        self._x_shape = None
        self._y_shape = None
        self._z_shape = None

        self._inspect_structure()

    # --------------------------------------------------------
    # File handling
    # --------------------------------------------------------

    def _open(self):

        if self._file is None:

            self._file = h5py.File(
                self.path,
                "r",
            )

        return self._file

    def close(self):

        if self._file is not None:

            self._file.close()
            self._file = None

    def __enter__(self):

        self._open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    def _inspect_structure(self):

        with h5py.File(
            self.path,
            "r",
        ) as h5:

            required = ["X", "Y", "Z"]

            missing = [name for name in required if name not in h5]

            if missing:

                raise ValueError("RadioML HDF5 is missing " f"datasets: {missing}")

            self._x_shape = tuple(h5["X"].shape)

            self._y_shape = tuple(h5["Y"].shape)

            self._z_shape = tuple(h5["Z"].shape)

    # --------------------------------------------------------
    # Basic properties
    # --------------------------------------------------------

    def __len__(self) -> int:

        return self._x_shape[0]

    @property
    def sample_shape(self):

        return self._x_shape[1:]

    @property
    def label_shape(self):

        return self._y_shape[1:]

    @property
    def auxiliary_shape(self):

        return self._z_shape[1:]

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def inspect(self) -> dict[str, Any]:

        with h5py.File(
            self.path,
            "r",
        ) as h5:

            return {
                "dataset": "RadioML 2018.01A",
                "path": str(self.path),
                "samples": int(h5["X"].shape[0]),
                "X": {
                    "shape": list(h5["X"].shape),
                    "dtype": str(h5["X"].dtype),
                },
                "Y": {
                    "shape": list(h5["Y"].shape),
                    "dtype": str(h5["Y"].dtype),
                },
                "Z": {
                    "shape": list(h5["Z"].shape),
                    "dtype": str(h5["Z"].dtype),
                },
                "file_size_gb": round(
                    self.path.stat().st_size / (1024**3),
                    3,
                ),
            }

    # --------------------------------------------------------
    # Sample access
    # --------------------------------------------------------

    def get_sample(
        self,
        index: int,
    ):

        if not 0 <= index < len(self):

            raise IndexError(f"Index {index} is outside " f"[0, {len(self) - 1}]")

        h5 = self._open()

        x = h5["X"][index]
        y = h5["Y"][index]
        z = h5["Z"][index]

        return (
            np.asarray(x),
            np.asarray(y),
            np.asarray(z),
        )

    # --------------------------------------------------------
    # Batch access
    # --------------------------------------------------------

    def get_batch(
        self,
        indices: Sequence[int],
    ):

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")

        if len(indices) == 0:

            raise ValueError("indices cannot be empty")

        if np.any(indices < 0) or np.any(indices >= len(self)):

            raise IndexError("One or more indices are " "outside the dataset range.")

        # h5py requires increasing indices
        # for efficient fancy indexing.

        order = np.argsort(indices)

        sorted_indices = indices[order]

        h5 = self._open()

        x_sorted = h5["X"][sorted_indices]

        y_sorted = h5["Y"][sorted_indices]

        z_sorted = h5["Z"][sorted_indices]

        reverse = np.argsort(order)

        x = x_sorted[reverse]
        y = y_sorted[reverse]
        z = z_sorted[reverse]

        return (
            np.asarray(x),
            np.asarray(y),
            np.asarray(z),
        )

    # --------------------------------------------------------
    # Small development sample
    # --------------------------------------------------------

    def get_sequential_batch(
        self,
        start: int,
        batch_size: int,
    ):

        if batch_size <= 0:

            raise ValueError("batch_size must be > 0")

        end = min(
            start + batch_size,
            len(self),
        )

        if start < 0 or start >= len(self):

            raise IndexError(f"Invalid start index: {start}")

        indices = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        return self.get_batch(indices)
