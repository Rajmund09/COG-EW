from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDatasetLoader(ABC):
    """
    Common interface for all COG-EW dataset loaders.
    """

    def __init__(self, path: str | Path):

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

    @abstractmethod
    def inspect(self) -> dict[str, Any]:
        """
        Return dataset metadata without loading
        the complete dataset into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return number of available samples/records.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int):
        """
        Return one sample.
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, indices):
        """
        Return a small indexed batch.
        """
        raise NotImplementedError
