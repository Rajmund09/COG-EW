"""
COG-EW RF AUTOMATIC MODULATION CLASSIFICATION
Single / batch RF-IQ inference

Purpose:
    Load the trained COG-EW RF CNN and classify IQ samples.

Supported classes:
    0 -> 8PSK
    1 -> BPSK
    2 -> QPSK
    3 -> 16QAM
    4 -> 64QAM

Expected input:
    NumPy array with shape:
        (128, 2)
    or
        (N, 128, 2)

The model expects:
    (batch, channels=2, time=128)

This script performs classification only.
It does not perform RF transmission, jamming,
weapon control, or autonomous engagement.
"""

from pathlib import Path
import argparse
import json
import hashlib

import h5py
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "cogew_rf_cnn_best.pt"

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]

NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL DEFINITION
# ============================================================


class COGEWRFCNN(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            # Block 4
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
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

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# LOAD MODEL
# ============================================================


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    print("=" * 70)
    print("LOADING COG-EW RF CNN")
    print("=" * 70)

    print(f"Model path : {MODEL_PATH}")
    print(f"Device     : {DEVICE}")

    if DEVICE.type == "cuda":

        print(f"GPU        : " f"{torch.cuda.get_device_name(0)}")

    print()

    model = COGEWRFCNN(num_classes=NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Support either a raw state_dict or a checkpoint dict.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    print("[PASS] Model loaded successfully")

    print(f"SHA256     : " f"{calculate_sha256(MODEL_PATH)}")

    print()

    return model


# ============================================================
# INPUT VALIDATION
# ============================================================


def validate_iq(x):

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:

        if x.shape != (128, 2):

            raise ValueError(
                "Expected single sample shape " "(128, 2), got " f"{x.shape}"
            )

        x = x[None, ...]

    elif x.ndim == 3:

        if x.shape[1:] != (128, 2):

            raise ValueError("Expected batch shape " "(N, 128, 2), got " f"{x.shape}")

    else:

        raise ValueError("Input must have shape " "(128, 2) or (N, 128, 2)")

    if not np.isfinite(x).all():

        raise ValueError("Input contains NaN or Inf values.")

    return x


# ============================================================
# PREDICTION
# ============================================================


@torch.no_grad()
def predict(model, x):

    x = validate_iq(x)

    # Convert:
    #
    # (N, 128, 2)
    #
    # to:
    #
    # (N, 2, 128)

    x = np.transpose(x, (0, 2, 1))

    tensor = torch.from_numpy(x)

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probabilities = torch.softmax(logits, dim=1)

    confidence, predictions = torch.max(probabilities, dim=1)

    predictions = predictions.cpu().numpy()

    confidence = confidence.cpu().numpy()

    probabilities = probabilities.cpu().numpy()

    results = []

    for i in range(len(predictions)):

        class_id = int(predictions[i])

        results.append(
            {
                "prediction_id": class_id,
                "prediction": CLASS_NAMES[class_id],
                "confidence": float(confidence[i]),
                "probabilities": {
                    CLASS_NAMES[j]: float(probabilities[i, j])
                    for j in range(NUM_CLASSES)
                },
            }
        )

    return results


# ============================================================
# LOAD HDF5 SAMPLE
# ============================================================


def load_h5_sample(h5_path, sample_index=0):

    h5_path = Path(h5_path)

    if not h5_path.exists():

        raise FileNotFoundError(f"Dataset not found:\n{h5_path}")

    with h5py.File(h5_path, "r") as f:

        if "X" not in f:

            raise KeyError("Dataset does not contain X.")

        total = f["X"].shape[0]

        if sample_index < 0 or sample_index >= total:

            raise IndexError(f"sample_index must be " f"between 0 and {total - 1}")

        x = f["X"][sample_index]

        result = {"X": x}

        if "Y" in f:

            result["Y"] = int(f["Y"][sample_index])

        if "Z" in f:

            result["Z"] = int(f["Z"][sample_index])

        if "source" in f:

            result["source"] = int(f["source"][sample_index])

    return result


# ============================================================
# DISPLAY RESULT
# ============================================================


def display_result(result, true_label=None, snr=None):

    print("=" * 70)
    print("COG-EW RF CLASSIFICATION RESULT")
    print("=" * 70)

    print(f"Predicted class : " f"{result['prediction']}")

    print(f"Confidence      : " f"{result['confidence'] * 100:.2f}%")

    if true_label is not None:

        print(f"True class      : " f"{CLASS_NAMES[true_label]}")

        correct = true_label == result["prediction_id"]

        print(f"Correct         : " f"{'YES' if correct else 'NO'}")

    if snr is not None:

        print(f"SNR             : " f"{snr} dB")

    print()

    print("CLASS PROBABILITIES")
    print("-" * 70)

    sorted_probs = sorted(
        result["probabilities"].items(), key=lambda item: item[1], reverse=True
    )

    for class_name, probability in sorted_probs:

        print(f"{class_name:<10} " f"{probability * 100:>8.2f}%")

    print()


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(description=("COG-EW RF CNN inference"))

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "splits" / "cogew_test.h5"),
        help="HDF5 dataset containing X",
    )

    parser.add_argument("--index", type=int, default=0, help="Sample index")

    parser.add_argument(
        "--save", type=str, default=None, help="Optional JSON output path"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load sample
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING RF-IQ SAMPLE")
    print("=" * 70)

    sample = load_h5_sample(args.dataset, args.index)

    print(f"Dataset : {args.dataset}")

    print(f"Index   : {args.index}")

    print(f"Shape   : {sample['X'].shape}")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict(model, sample["X"])[0]

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    true_label = sample.get("Y")

    snr = sample.get("Z")

    display_result(result, true_label=true_label, snr=snr)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    if args.save:

        output_path = Path(args.save)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset": str(Path(args.dataset).resolve()),
            "sample_index": args.index,
            "model": str(MODEL_PATH.resolve()),
            "result": result,
            "true_label": (int(true_label) if true_label is not None else None),
            "true_class": (CLASS_NAMES[true_label] if true_label is not None else None),
            "snr_db": (int(snr) if snr is not None else None),
        }

        with open(output_path, "w", encoding="utf-8") as f:

            json.dump(payload, f, indent=2)

        print(f"[OK] Prediction saved:\n" f"{output_path}")


if __name__ == "__main__":

    main()
