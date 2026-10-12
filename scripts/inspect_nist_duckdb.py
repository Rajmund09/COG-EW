from __future__ import annotations

import json
from pathlib import Path

import duckdb


NIST_DIR = Path(r"D:\COG-EW\data\processed\nist_sea")

REPORT_PATH = Path(r"D:\COG-EW\reports") / "nist_duckdb_schema.json"


def main():

    print("=" * 70)
    print("COG-EW NIST DUCKDB SCHEMA INSPECTION")
    print("=" * 70)

    databases = list(NIST_DIR.glob("*.duckdb"))

    if not databases:

        raise FileNotFoundError(f"No DuckDB found in:\n" f"{NIST_DIR}")

    if len(databases) > 1:

        print("Multiple databases found:")

        for db in databases:
            print(f"  - {db.name}")

        raise RuntimeError("Keep one database for this " "inspection.")

    database = databases[0]

    print(f"\nDatabase:\n{database}")

    con = duckdb.connect(
        database=str(database),
        read_only=True,
    )

    try:

        tables = con.execute("SHOW TABLES").fetchall()

        table_names = [row[0] for row in tables]

        print(f"\nTables: " f"{len(table_names)}")

        report = {
            "database": str(database),
            "tables": [],
        }

        for table in table_names:

            print("\n" + "-" * 60)

            print(f"TABLE: {table}")

            escaped = table.replace(
                '"',
                '""',
            )

            schema = con.execute(f'DESCRIBE "{escaped}"').fetchall()

            row_count = con.execute(
                f"""
                SELECT COUNT(*)
                FROM "{escaped}"
                """
            ).fetchone()[0]

            print(f"Rows: {row_count:,}")

            columns = []

            for row in schema:

                column_name = str(row[0])

                column_type = str(row[1])

                columns.append(
                    {
                        "name": column_name,
                        "type": column_type,
                    }
                )

                print(f"  {column_name}: " f"{column_type}")

            sample = con.execute(
                f"""
                SELECT *
                FROM "{escaped}"
                LIMIT 5
                """
            ).fetchdf()

            report["tables"].append(
                {
                    "name": table,
                    "rows": int(row_count),
                    "columns": columns,
                    "sample": (sample.astype(str).to_dict(orient="records")),
                }
            )

        REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
            )

        print("\n" + "=" * 70)

        print(f"Report saved:\n" f"{REPORT_PATH}")

    finally:

        con.close()


if __name__ == "__main__":
    main()
