"""
RadioML 2018.01A Dataset Reader & Inspector Utility
(scripts/read_radioml_dataset.py)

Usage:
  1. Inspect general dataset shape & structure:
     python scripts/read_radioml_dataset.py --info

  2. Read a specific sample index (e.g. sample #5000):
     python scripts/read_radioml_dataset.py --sample-index 5000

  3. Query samples by modulation class and SNR:
     python scripts/read_radioml_dataset.py --modulation 16QAM --snr 14

  4. Export sample I/Q waveform to CSV:
     python scripts/read_radioml_dataset.py --sample-index 5000 --export-csv sample_5000.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.rf_inference_service import (
    RADIOML_24_CLASSES,
    THREAT_CATEGORIES,
    RFInferenceService,
)


def find_dataset_path() -> Path:
    candidates = [
        Path(r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"),
        Path(r"D:\Datasets\RadioML 2018.01A Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"),
        Path(r"D:\Datasets\GOLD_XYZ_OSC.0001_1024.hdf5"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    d_path = Path(r"D:\Datasets")
    if d_path.is_dir():
        for match in d_path.glob("*RadioML*/GOLD_XYZ_OSC.0001_1024.hdf5"):
            if match.is_file():
                return match

    raise FileNotFoundError("Could not find GOLD_XYZ_OSC.0001_1024.hdf5 in D:/Datasets")


def print_dataset_info(h5: h5py.File, path: Path):
    print("=" * 72)
    print("RADIOML 2018.01A DATASET SUMMARY & HARDWARE SPECIFICATIONS")
    print("=" * 72)
    clean_path = str(path).encode("ascii", "ignore").decode("ascii")
    print(f"File Path: {clean_path}")
    print(f"File Size: {path.stat().st_size / (1024**3):.2f} GB")
    print("-" * 72)
    print(f"Array 'X' (I/Q Signal Pulses): Shape {h5['X'].shape}, Dtype: {h5['X'].dtype}")
    print(f"Array 'Y' (One-Hot Labels):    Shape {h5['Y'].shape}, Dtype: {h5['Y'].dtype}")
    print(f"Array 'Z' (SNR in dB):         Shape {h5['Z'].shape}, Dtype: {h5['Z'].dtype}")
    print("-" * 72)
    print(f"Total Sample Pulses:     2,555,904")
    print(f"Samples per Modulation:  106,496 (across 24 modulation classes)")
    print(f"Samples per SNR Step:    98,304 (across 26 SNR levels from -20 dB to +30 dB)")
    print(f"Time Bins per Pulse:     1,024 complex samples (1024 I + 1024 Q)")
    print("-" * 72)
    print("24 Canonical Modulation Classes:")
    for i, mod in enumerate(RADIOML_24_CLASSES):
        print(f"  [{i:02d}] {mod:<10} - Threat: {THREAT_CATEGORIES.get(mod, {}).get('threat_level', 'UNKNOWN')}")
    print("=" * 72)


def inspect_sample(h5: h5py.File, index: int, export_csv: str | None = None):
    total = len(h5["X"])
    if index < 0 or index >= total:
        print(f"Error: Index {index} is out of bounds (0 to {total-1})")
        return

    raw_x = h5["X"][index]  # (1024, 2)
    raw_y = h5["Y"][index]  # (24,)
    raw_z = h5["Z"][index]  # (1,)

    class_idx = int(np.argmax(raw_y))
    mod_name = RADIOML_24_CLASSES[class_idx]
    snr_db = float(raw_z[0]) if hasattr(raw_z, "__len__") else float(raw_z)
    threat = THREAT_CATEGORIES.get(mod_name, {})

    i_chan = raw_x[:, 0]
    q_chan = raw_x[:, 1]

    p_i = float(np.mean(i_chan ** 2))
    p_q = float(np.mean(q_chan ** 2))
    p_tot = p_i + p_q
    crest = float(np.max(i_chan ** 2 + q_chan ** 2) / max(p_tot, 1e-9))

    print("=" * 72)
    print(f"RADIOML 2018.01A SAMPLE INSPECTION — INDEX #{index:,}")
    print("=" * 72)
    print(f"Modulation Class:      {mod_name} (Class Index: {class_idx})")
    print(f"Signal-to-Noise Ratio: {snr_db:.1f} dB")
    print(f"Tactical Threat Level: {threat.get('threat_level', 'UNKNOWN')}")
    print(f"Radar Category:        {threat.get('category', 'UNKNOWN')}")
    print(f"Recommended Jamming:   {threat.get('jamming_mode', 'UNKNOWN')}")
    print("-" * 72)
    print(f"In-Phase (I) Power:    {p_i:.5f}")
    print(f"Quadrature (Q) Power:  {p_q:.5f}")
    print(f"Total Signal Power:    {p_tot:.5f}")
    print(f"PAPR / Crest Factor:   {crest:.3f}")
    print(f"DC Offset (I / Q):     {np.mean(i_chan):.5f} / {np.mean(q_chan):.5f}")
    print("-" * 72)
    print("First 8 In-Phase (I) samples:")
    print(" ", [round(float(v), 4) for v in i_chan[:8]])
    print("First 8 Quadrature (Q) samples:")
    print(" ", [round(float(v), 4) for v in q_chan[:8]])
    print("=" * 72)

    # Run inference test if model is present
    try:
        svc = RFInferenceService(device="auto")
        sig = raw_x.T.astype(np.float32)
        res = svc.predict(sig)
        print(f"ResNet Neural Network Prediction: {res['predicted_class']} ({res['confidence_percent']}%)")
        print(f"Inference Latency:                {res['inference_ms']:.2f} ms")
        print(f"Result Verdict:                   {'CORRECT' if res['predicted_class'] == mod_name else 'MISMATCH'}")
        print("=" * 72)
    except Exception as e:
        print(f"[Notice] Model inference skipped: {e}")

    if export_csv:
        import pandas as pd
        df = pd.DataFrame({
            "Time_Step": np.arange(1024),
            "In_Phase_I": i_chan,
            "Quadrature_Q": q_chan,
        })
        df.to_csv(export_csv, index=False)
        print(f"Exported raw I/Q samples to CSV: {export_csv}")


def main():
    parser = argparse.ArgumentParser(description="RadioML 2018.01A Dataset Reader & Inspector")
    parser.add_argument("--info", action="store_true", help="Print dataset metadata and schema")
    parser.add_argument("--sample-index", type=int, default=5000, help="Index of sample to inspect (0 to 2,555,903)")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export raw I/Q samples to CSV")
    args = parser.parse_args()

    path = find_dataset_path()
    with h5py.File(path, "r") as h5:
        if args.info:
            print_dataset_info(h5, path)
        else:
            inspect_sample(h5, args.sample_index, args.export_csv)


if __name__ == "__main__":
    main()
