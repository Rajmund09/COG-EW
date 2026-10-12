"""
COG-EW
RF Test Sample Classification

Loads the trained RF CNN and directly reads a sample from
the untouched test dataset.

No dependency on test_prediction.json or sample_prediction.json.
"""

from pathlib import Path
import argparse
import json
import hashlib
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "cogew_rf_cnn_best.pt"
TEST_DATASET = PROJECT_ROOT / "data" / "splits" / "cogew_test.h5"
OUTPUT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

SOURCE_NAMES = {
    0: "RadioML2016",
    1: "RadioML2018",
}


# ============================================================
# MODEL
# ============================================================


class COGEWRFCNN(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            # Block 4
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# SHA256
# ============================================================


def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# MODEL LOADER
# ============================================================


def load_model(device):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    model = COGEWRFCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Support different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)

    model.to(device)
    model.eval()

    return model


# ============================================================
# LOAD TEST SAMPLE
# ============================================================


def load_test_sample(index):

    if not TEST_DATASET.exists():
        raise FileNotFoundError(f"Test dataset not found:\n{TEST_DATASET}")

    with h5py.File(TEST_DATASET, "r") as f:

        total = f["X"].shape[0]

        if index < 0 or index >= total:
            raise IndexError(
                f"Index {index} is invalid. " f"Valid range: 0 to {total - 1}"
            )

        x = f["X"][index]
        y = int(f["Y"][index])
        z = int(f["Z"][index])

        original_index = int(f["original_index"][index])

        source = int(f["source"][index])

    return (x, y, z, original_index, source)


# ============================================================
# PREDICTION
# ============================================================


def predict(model, x, device):

    # Original shape:
    # (128, 2)
    #
    # CNN requires:
    # (batch, channels, time)
    #
    # Convert:
    # (128, 2)
    # ->
    # (2, 128)

    x_tensor = torch.tensor(x, dtype=torch.float32)

    x_tensor = x_tensor.transpose(0, 1)

    x_tensor = x_tensor.unsqueeze(0)

    x_tensor = x_tensor.to(device)

    with torch.inference_mode():

        logits = model(x_tensor)

        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    confidence = float(probabilities[predicted_index].item())

    probabilities_np = probabilities.cpu().numpy()

    return (predicted_index, confidence, probabilities_np)


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(description="COG-EW RF test sample prediction")

    parser.add_argument(
        "--index", type=int, default=5000, help="Test dataset sample index"
    )

    parser.add_argument(
        "--save", type=str, default=None, help="Optional output JSON path"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("COG-EW RF TEST SAMPLE CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("Device :", device)

    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # PATH CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILE VALIDATION")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Test dataset:")
    print(TEST_DATASET)

    if not MODEL_PATH.exists():
        print("\n[ERROR] Model file does not exist.")
        sys.exit(1)

    if not TEST_DATASET.exists():
        print("\n[ERROR] Test dataset does not exist.")
        sys.exit(1)

    print("\n[PASS] Required files found")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = load_model(device)

    model_hash = calculate_sha256(MODEL_PATH)

    print()
    print("[PASS] Model loaded")
    print("SHA256:", model_hash)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING TEST SAMPLE")
    print("=" * 70)

    (x, true_label, snr, original_index, source_id) = load_test_sample(args.index)

    print()
    print("Test index      :", args.index)
    print("Original index  :", original_index)
    print("Source ID       :", source_id)
    print("Source          :", SOURCE_NAMES.get(source_id, f"Unknown ({source_id})"))
    print("SNR             :", snr, "dB")
    print("IQ shape        :", x.shape)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not np.isfinite(x).all():
        print("\n[ERROR] Sample contains NaN/Inf")
        sys.exit(1)

    if x.shape != (128, 2):
        print("\n[ERROR] Unexpected IQ shape:", x.shape)
        sys.exit(1)

    print("[PASS] IQ sample validated")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    (predicted_index, confidence, probabilities) = predict(model, x, device)

    predicted_class = CLASS_NAMES[predicted_index]

    true_class = CLASS_NAMES[true_label]

    correct = predicted_index == true_label

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print()
    print("True class       :", true_class)
    print("Predicted class  :", predicted_class)
    print("Confidence       :", f"{confidence * 100:.2f}%")

    print("Correct          :", "YES" if correct else "NO")

    print()
    print("CLASS PROBABILITIES")
    print("-" * 70)

    sorted_indices = np.argsort(probabilities)[::-1]

    for idx in sorted_indices:

        print(f"{CLASS_NAMES[idx]:10s} " f"{probabilities[idx] * 100:7.2f}%")

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    result = {
        "project": "COG-EW",
        "prediction_type": "RF Automatic Modulation Classification",
        "test_index": int(args.index),
        "original_index": int(original_index),
        "source_id": int(source_id),
        "source_name": SOURCE_NAMES.get(source_id, f"Unknown ({source_id})"),
        "snr_db": int(snr),
        "iq_shape": list(x.shape),
        "true_class": true_class,
        "true_class_id": int(true_label),
        "predicted_class": predicted_class,
        "predicted_class_id": int(predicted_index),
        "confidence": float(confidence),
        "confidence_percent": float(confidence * 100),
        "correct": bool(correct),
        "class_probabilities": {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        },
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "test_dataset": str(TEST_DATASET),
        "device": str(device),
    }

    if args.save:

        output_path = Path(args.save)

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    else:

        output_path = OUTPUT_DIR / "test_prediction.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(result, f, indent=2)

    print()
    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    print()
    print("[OK] Prediction saved:")
    print(output_path)


if __name__ == "__main__":
    main()
