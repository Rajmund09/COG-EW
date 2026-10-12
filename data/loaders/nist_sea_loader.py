from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from .base_loader import BaseDatasetLoader


class NISTSeaDataLoader(BaseDatasetLoader):
    """
    Loader for NIST SEA-DATA DuckDB datasets.

    This class inspects the database schema first.
    It does not assume table names or column names.
    """

    def __init__(
        self,
        path: str | Path,
        read_only: bool = True,
    ):

        super().__init__(path)

        self.read_only = read_only
        self._connection = None

    # --------------------------------------------------------
    # Connection
    # --------------------------------------------------------

    def connect(self):

        if self._connection is None:

            self._connection = duckdb.connect(
                database=str(self.path),
                read_only=self.read_only,
            )

        return self._connection

    def close(self):

        if self._connection is not None:

            self._connection.close()
            self._connection = None

    def __enter__(self):

        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    def list_tables(self):

        con = self.connect()

        result = con.execute("SHOW TABLES").fetchall()

        return [row[0] for row in result]

    # --------------------------------------------------------
    # Table schema
    # --------------------------------------------------------

    def describe_table(
        self,
        table_name: str,
    ):

        con = self.connect()

        # Identifier quoting prevents table names
        # containing special characters from breaking.

        escaped = table_name.replace('"', '""')

        query = f'DESCRIBE "{escaped}"'

        return con.execute(query).fetchdf()

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    def count_rows(
        self,
        table_name: str,
    ) -> int:

        con = self.connect()

        escaped = table_name.replace('"', '""')

        query = f"SELECT COUNT(*) " f'FROM "{escaped}"'

        return int(con.execute(query).fetchone()[0])

    # --------------------------------------------------------
    # Sample rows
    # --------------------------------------------------------

    def sample_rows(
        self,
        table_name: str,
        limit: int = 5,
    ):

        if limit <= 0:
            raise ValueError("limit must be > 0")

        con = self.connect()

        escaped = table_name.replace('"', '""')

        query = f"SELECT * " f'FROM "{escaped}" ' f"LIMIT {int(limit)}"

        return con.execute(query).fetchdf()

    # --------------------------------------------------------
    # Full inspection
    # --------------------------------------------------------

    def inspect(self) -> dict[str, Any]:

        tables = self.list_tables()

        result = {
            "dataset": "NIST SEA-DATA",
            "path": str(self.path),
            "file_size_gb": round(
                self.path.stat().st_size / (1024**3),
                3,
            ),
            "tables": [],
        }

        for table in tables:

            try:

                schema = self.describe_table(table)

                rows = self.count_rows(table)

                sample = self.sample_rows(
                    table,
                    limit=3,
                )

                result["tables"].append(
                    {
                        "name": table,
                        "rows": rows,
                        "columns": [
                            {
                                "name": str(row[0]),
                                "type": str(row[1]),
                            }
                            for _, row in schema.iterrows()
                        ],
                        "sample": (sample.astype(str).to_dict(orient="records")),
                    }
                )

            except Exception as exc:

                result["tables"].append(
                    {
                        "name": table,
                        "error": repr(exc),
                    }
                )

        return result

    # --------------------------------------------------------
    # Required BaseDatasetLoader methods
    # --------------------------------------------------------

    def __len__(self) -> int:

        tables = self.list_tables()

        if not tables:
            return 0

        return self.count_rows(tables[0])

    def get_sample(
        self,
        index: int,
    ):

        tables = self.list_tables()

        if not tables:

            raise RuntimeError("NIST database contains " "no tables.")

        table = tables[0]

        con = self.connect()

        escaped = table.replace('"', '""')

        query = f"SELECT * " f'FROM "{escaped}" ' f"LIMIT 1 OFFSET " f"{int(index)}"

        return con.execute(query).fetchdf()

    def get_batch(
        self,
        indices,
    ):

        # Generic indexed access is intentionally
        # conservative until the actual NIST schema
        # has been examined.

        if len(indices) == 0:

            raise ValueError("indices cannot be empty")

        tables = self.list_tables()

        if not tables:

            raise RuntimeError("NIST database contains " "no tables.")

        table = tables[0]

        start = min(indices)
        count = len(indices)

        con = self.connect()

        escaped = table.replace('"', '""')

        query = (
            f"SELECT * "
            f'FROM "{escaped}" '
            f"LIMIT {int(count)} "
            f"OFFSET {int(start)}"
        )

        return con.execute(query).fetchdf()
