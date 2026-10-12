import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT = REPORT_DIR / "rf_model_card.json"


model_card = {
    "project": "COG-EW",
    "component": "RF Automatic Modulation Classification",
    "model": "1D CNN",
    "framework": "PyTorch",
    "classes": ["8PSK", "BPSK", "QPSK", "16QAM", "64QAM"],
    "input": {
        "format": "IQ",
        "shape": [128, 2],
        "channels": 2,
        "samples_per_signal": 128,
    },
    "datasets": ["RadioML 2016.10A", "RadioML 2018.01A"],
    "preprocessing": {
        "common_length": 128,
        "normalization": True,
        "augmentation": "training only",
    },
    "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "random_seed": 42},
    "augmentation": {
        "enabled": True,
        "factor": 2,
        "validation_augmented": False,
        "test_augmented": False,
    },
    "performance": {
        "best_validation_accuracy": 0.6488,
        "test_accuracy": 0.6475,
        "radio_ml_2016_accuracy": 0.6671,
        "radio_ml_2018_accuracy": 0.6279,
    },
    "class_performance": {
        "8PSK": {"precision": 0.4440, "recall": 0.8948, "f1": 0.5935},
        "BPSK": {"precision": 0.8485, "recall": 0.6992, "f1": 0.7667},
        "QPSK": {"precision": 0.7938, "recall": 0.6217, "f1": 0.6973},
        "16QAM": {"precision": 0.6987, "recall": 0.5350, "f1": 0.6060},
        "64QAM": {"precision": 0.7951, "recall": 0.4867, "f1": 0.6038},
    },
    "limitations": [
        "Performance decreases substantially at very low SNR.",
        "8PSK has high recall but relatively low precision.",
        "16QAM and 64QAM are frequently confused.",
        "RadioML 2018 performance is lower than RadioML 2016.",
        "The model is an RF signal classification component and should not be interpreted as a complete electronic-warfare decision system.",
    ],
    "integrity": {
        "test_data_augmented": False,
        "test_data_used_for_training": False,
        "test_data_used_for_model_selection": False,
        "source_datasets_modified": False,
    },
    "generated_at": datetime.now().isoformat(),
}


REPORT_DIR.mkdir(parents=True, exist_ok=True)


with open(OUTPUT, "w", encoding="utf-8") as f:

    json.dump(model_card, f, indent=4)


print("=" * 70)
print("COG-EW RF MODEL CARD")
print("=" * 70)

print()
print("[OK] Model card generated:")
print(OUTPUT)
