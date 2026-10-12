import os
import pickle
import numpy as np
from collections import Counter, defaultdict


PKL_PATH = r"D:\Datasets\DeepSig Dataset RadioML 2016.10A\RML2016.10a_dict.pkl"

print("=" * 70)
print("COG-EW RADIO ML 2016.10A DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset:")
print(PKL_PATH)

if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(PKL_PATH)

print("\n[OK] Dataset file found.")

print("\nLoading PKL...")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("[OK] Dataset loaded.")

print("\nDataset object type:")
print(type(data))

print("\nNumber of keys:")
print(len(data))

print("\n" + "=" * 70)
print("KEY STRUCTURE")
print("=" * 70)

for i, key in enumerate(data.keys()):
    print(f"{i + 1:03d}: {key}")

print("\n" + "=" * 70)
print("SAMPLE STRUCTURE")
print("=" * 70)

first_key = next(iter(data.keys()))

print(f"First key: {first_key}")

first_value = np.asarray(data[first_key])

print(f"Value type: {type(data[first_key])}")
print(f"Value shape: {first_value.shape}")
print(f"Value dtype: {first_value.dtype}")

print("\n" + "=" * 70)
print("MODULATION CLASSES")
print("=" * 70)

classes = sorted(set(key[0] for key in data.keys()))

for idx, cls in enumerate(classes):
    print(f"{idx:02d} | {cls}")

print(f"\nTotal classes: {len(classes)}")

print("\n" + "=" * 70)
print("SNR LEVELS")
print("=" * 70)

snrs = sorted(set(key[1] for key in data.keys()))

for snr in snrs:
    print(f"{snr:>4} dB")

print(f"\nTotal SNR levels: {len(snrs)}")

print("\n" + "=" * 70)
print("SAMPLES PER CLASS")
print("=" * 70)

class_counts = Counter()

for key, value in data.items():
    modulation = key[0]
    class_counts[modulation] += len(value)

for cls in classes:
    print(f"{cls:<12} : {class_counts[cls]:>8,}")

print("\n" + "=" * 70)
print("SAMPLES PER CLASS / SNR")
print("=" * 70)

for cls in classes:
    print(f"\n{cls}")

    for snr in snrs:
        key = (cls, snr)

        if key in data:
            count = len(data[key])
            print(f"  {snr:>4} dB : {count:>6,}")

print("\n" + "=" * 70)
print("IQ FORMAT CHECK")
print("=" * 70)

sample = first_value

print(f"Shape: {sample.shape}")

if sample.ndim == 3:
    print("[OK] Dataset appears to contain IQ/channel dimensions.")
elif sample.ndim == 2:
    print("[INFO] Dataset uses 2D sample representation.")
else:
    print("[WARNING] Unexpected sample dimensionality.")

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

sample_subset = first_value[: min(len(first_value), 10000)]

print(f"Inspection samples: {len(sample_subset):,}")

print(f"NaN present: {np.isnan(sample_subset).any()}")
print(f"Inf present: {np.isinf(sample_subset).any()}")
print(f"Minimum: {sample_subset.min():.6f}")
print(f"Maximum: {sample_subset.max():.6f}")
print(f"Mean: {sample_subset.mean():.6f}")
print(f"Std: {sample_subset.std():.6f}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("No source dataset was modified.")
print("No data was cleaned.")
print("No data was merged.")
print("No augmentation was performed.")
print("No model training was performed.")
