from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import h5py


RADIO_PATH = Path(
    r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark" r"\GOLD_XYZ_OSC.0001_1024.hdf5"
)

REPORT_DIR = Path(r"D:\COG-EW\reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = REPORT_DIR / "radioml_semantic_analysis.json"


def analyze_y(y):

    result = {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "min": None,
        "max": None,
        "unique_rows": None,
        "row_sums": None,
        "one_hot_like": False,
    }

    result["min"] = int(np.min(y))

    result["max"] = int(np.max(y))

    # We inspect only a manageable subset
    # before considering the entire array.

    sample_size = min(
        len(y),
        100_000,
    )

    sample = y[:sample_size]

    row_sums = np.sum(
        sample,
        axis=1,
    )

    result["row_sum_min"] = int(np.min(row_sums))

    result["row_sum_max"] = int(np.max(row_sums))

    result["row_sum_unique"] = [int(v) for v in np.unique(row_sums)[:50]]

    result["one_hot_like"] = bool(
        np.all(
            np.isin(
                sample,
                [0, 1],
            )
        )
        and np.all(row_sums == 1)
    )

    if result["one_hot_like"]:

        labels = np.argmax(
            sample,
            axis=1,
        )

        counts = np.bincount(
            labels,
            minlength=y.shape[1],
        )

        result["class_counts_sample"] = [int(v) for v in counts]

    return result


def analyze_z(z):

    result = {
        "shape": list(z.shape),
        "dtype": str(z.dtype),
    }

    sample_size = min(
        len(z),
        500_000,
    )

    sample = np.asarray(z[:sample_size]).reshape(-1)

    unique = np.unique(sample)

    result["min"] = int(np.min(sample))

    result["max"] = int(np.max(sample))

    result["unique_count"] = int(len(unique))

    result["unique_values_preview"] = [int(v) for v in unique[:100]]

    return result


def analyze_x(x):

    sample_size = min(
        len(x),
        1000,
    )

    sample = np.asarray(x[:sample_size])

    result = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "sample_count": sample_size,
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
    }

    # Per-channel statistics
    result["channel_statistics"] = []

    for channel in range(sample.shape[-1]):

        values = sample[..., channel]

        result["channel_statistics"].append(
            {
                "channel": channel,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )

    return result


def main():

    print("=" * 70)
    print("COG-EW RADIO ML SEMANTIC ANALYSIS")
    print("=" * 70)

    if not RADIO_PATH.exists():

        raise FileNotFoundError(f"RadioML file not found:\n" f"{RADIO_PATH}")

    with h5py.File(
        RADIO_PATH,
        "r",
    ) as h5:

        print("\nDatasets:")

        for name in h5.keys():

            print(f"  {name}: " f"{h5[name].shape} " f"{h5[name].dtype}")

        x = h5["X"]
        y = h5["Y"]
        z = h5["Z"]

        print("\nAnalyzing X...")
        x_info = analyze_x(x)

        print("Analyzing Y...")
        y_info = analyze_y(y)

        print("Analyzing Z...")
        z_info = analyze_z(z)

    report = {
        "dataset": "RadioML 2018.01A",
        "path": str(RADIO_PATH),
        "X": x_info,
        "Y": y_info,
        "Z": z_info,
    }

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
    print("RESULT")
    print("=" * 70)

    print("\nY analysis:")

    print(
        json.dumps(
            y_info,
            indent=2,
        )
    )

    print("\nZ analysis:")

    print(
        json.dumps(
            z_info,
            indent=2,
        )
    )

    print("\nX statistics:")

    print(
        json.dumps(
            x_info,
            indent=2,
        )
    )

    print(f"\nSaved report:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
