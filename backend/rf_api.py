from pathlib import Path
import json
import hashlib
import sys

import h5py
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "cogew_rf_cnn_best.pt"

TEST_DATASET = ROOT_DIR / "data" / "splits" / "cogew_test.h5"


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
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL IMPORT
# ============================================================

# IMPORTANT:
# Replace this import with the actual location of the
# COGEWRFCNN class from your training script.
#
# If your train_cogew_rf_cnn.py defines COGEWRFCNN directly,
# we will move that model definition into a reusable module.

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from scripts.train_cogew_rf_cnn import COGEWRFCNN
except Exception as exc:
    raise ImportError(
        "Could not import COGEWRFCNN. "
        "Move the COGEWRFCNN model class into a reusable "
        "module such as scripts/cogew_rf_model.py."
    ) from exc


# ============================================================
# MODEL HASH
# ============================================================


def calculate_sha256(path: Path) -> str:

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# LOAD MODEL
# ============================================================


def load_rf_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = COGEWRFCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # Handle different checkpoint formats

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# GLOBAL MODEL
# ============================================================

print("=" * 70)
print("COG-EW FAST RF INFERENCE BACKEND")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")

MODEL = load_rf_model()

MODEL_SHA256 = calculate_sha256(MODEL_PATH)

print("Model loaded successfully.")
print(f"SHA256: {MODEL_SHA256}")


# ============================================================
# RF PREPROCESSING
# ============================================================


def prepare_iq(iq: np.ndarray) -> torch.Tensor:

    iq = np.asarray(iq, dtype=np.float32)

    if iq.shape != (128, 2):

        raise ValueError(f"Expected IQ shape (128, 2), " f"received {iq.shape}")

    if not np.isfinite(iq).all():

        raise ValueError("IQ sample contains NaN or Inf values.")

    # (128, 2)
    #      ↓
    # (2, 128)

    iq = iq.T

    tensor = torch.from_numpy(iq).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================


@torch.inference_mode()
def predict_iq(iq: np.ndarray):

    tensor = prepare_iq(iq)

    logits = MODEL(tensor)

    probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    confidence = float(probabilities[predicted_index].item())

    probability_dict = {
        CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probability_dict,
    }


# ============================================================
# TEST SAMPLE
# ============================================================


def load_test_sample(index: int):

    with h5py.File(TEST_DATASET, "r") as h5:

        X = h5["X"]

        Y = h5["Y"]

        Z = h5["Z"]

        if index < 0 or index >= len(X):

            raise IndexError(f"Test index must be between " f"0 and {len(X) - 1}")

        iq = X[index]

        label = int(Y[index])

        snr = int(Z[index])

        return (
            np.asarray(iq),
            label,
            snr,
        )


# ============================================================
# SIGNAL QUALITY
# ============================================================


def analyze_signal_quality(iq):

    i = iq[:, 0]
    q = iq[:, 1]

    power = float(np.mean(i**2 + q**2))

    # DC offset

    dc_offset = float(np.sqrt(np.mean(i) ** 2 + np.mean(q) ** 2))

    # IQ imbalance

    i_power = np.mean(i**2)
    q_power = np.mean(q**2)

    iq_imbalance = float(abs(i_power - q_power) / (i_power + q_power + 1e-12))

    # Approximate noise/SNR estimate

    signal = i + 1j * q

    signal_centered = signal - np.mean(signal)

    signal_variance = np.var(signal_centered)

    noise_estimate = np.var(np.diff(signal_centered)) / 2.0

    if noise_estimate <= 1e-12:

        estimated_snr = 50.0

    else:

        estimated_snr = 10.0 * np.log10(signal_variance / noise_estimate)

    # Quality classification

    if estimated_snr >= 10:

        quality = "GOOD"

    elif estimated_snr >= 0:

        quality = "MODERATE"

    else:

        quality = "POOR"

    return {
        "signal_power": power,
        "estimated_snr": float(estimated_snr),
        "iq_imbalance": iq_imbalance,
        "dc_offset": dc_offset,
        "signal_quality": quality,
    }
