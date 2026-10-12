"""
COG-EW RF INFERENCE SERVICE
============================

Purpose:
    Reusable inference module for the trained COG-EW RF CNN.

This module:
    - Loads the best trained model
    - Accepts IQ samples
    - Performs RF modulation classification
    - Returns class probabilities
    - Estimates confidence
    - Does NOT perform jamming
    - Does NOT control aircraft/weapons
"""

from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "cogew_rf_cnn_best.pt"


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]


# ============================================================
# MODEL
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
# RF CLASSIFIER
# ============================================================


class RFClassifier:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(f"Model not found:\n{self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = COGEWRFCNN(num_classes=len(CLASS_NAMES))

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        self.model.eval()

        self.sha256 = calculate_sha256(self.model_path)

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def validate_iq(iq):

        iq = np.asarray(iq)

        if iq.shape != (128, 2):

            raise ValueError(f"Expected IQ shape (128, 2), " f"received {iq.shape}")

        if not np.isfinite(iq).all():

            raise ValueError("IQ input contains NaN or Inf values.")

        return iq.astype(np.float32, copy=False)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, iq):

        iq = self.validate_iq(iq)

        # Convert:
        #
        # (128, 2)
        #
        # into:
        #
        # (1, 2, 128)

        tensor = torch.from_numpy(iq.T).unsqueeze(0)

        tensor = tensor.to(self.device)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(logits, dim=1)[0]

        probabilities = probabilities.detach().cpu().numpy()

        predicted_index = int(np.argmax(probabilities))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(probabilities[predicted_index])

        probability_dict = {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probability_dict,
            "model_sha256": self.sha256,
            "device": str(self.device),
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("COG-EW RF INFERENCE SERVICE TEST")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_PATH)

    classifier = RFClassifier()

    print()
    print("Device:")
    print(classifier.device)

    print()
    print("Model SHA256:")
    print(classifier.sha256)

    # Demonstration input.
    #
    # Replace this with a real IQ sample when integrating
    # with the COG-EW RF analysis pipeline.

    demo_iq = np.zeros((128, 2), dtype=np.float32)

    result = classifier.predict(demo_iq)

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(f"Predicted class : " f"{result['predicted_class']}")

    print(f"Confidence      : " f"{result['confidence'] * 100:.2f}%")

    print()
    print("Class probabilities:")

    for name, probability in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):

        print(f"  {name:<8} " f"{probability * 100:6.2f}%")

    print()
    print("[PASS] RF inference service is operational.")
