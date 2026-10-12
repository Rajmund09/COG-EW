from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .base_loader import BaseDatasetLoader


class IDFDSLoader(BaseDatasetLoader):
    """
    Loader for IDF-DS flight telemetry CSV files.

    Data are loaded in chunks where possible.
    """

    def __init__(
        self,
        path: str | Path,
    ):

        super().__init__(path)

        self._columns = None
        self._row_count = None

        self._inspect_structure()

    def _inspect_structure(self):

        sample = pd.read_csv(
            self.path,
            nrows=5,
        )

        self._columns = list(sample.columns)

        # Count rows without creating
        # a complete dataframe.

        with self.path.open("rb") as f:

            self._row_count = max(
                sum(1 for _ in f) - 1,
                0,
            )

    def __len__(self):

        return self._row_count

    @property
    def columns(self):

        return self._columns.copy()

    def inspect(self) -> dict[str, Any]:

        sample = pd.read_csv(
            self.path,
            nrows=5,
        )

        return {
            "dataset": "IDF-DS",
            "path": str(self.path),
            "rows": self._row_count,
            "columns": self.columns,
            "dtypes": {
                str(column): str(dtype) for column, dtype in sample.dtypes.items()
            },
            "sample": (sample.astype(str).to_dict(orient="records")),
            "file_size_mb": round(
                self.path.stat().st_size / (1024**2),
                3,
            ),
        }

    def get_sample(
        self,
        index: int,
    ):

        if not 0 <= index < len(self):

            raise IndexError(f"Invalid row index: {index}")

        df = pd.read_csv(
            self.path,
            skiprows=range(
                1,
                index + 1,
            ),
            nrows=1,
        )

        return df

    def get_batch(
        self,
        indices: Sequence[int],
    ):

        if len(indices) == 0:

            raise ValueError("indices cannot be empty")

        start = min(indices)
        end = max(indices)

        count = end - start + 1

        df = pd.read_csv(
            self.path,
            skiprows=range(
                1,
                start + 1,
            ),
            nrows=count,
        )

        # Preserve requested order if possible.
        positions = [i - start for i in indices]

        return df.iloc[positions].reset_index(drop=True)

    def iter_chunks(
        self,
        chunk_size: int = 10000,
    ):

        if chunk_size <= 0:

            raise ValueError("chunk_size must be > 0")

        return pd.read_csv(
            self.path,
            chunksize=chunk_size,
        )
