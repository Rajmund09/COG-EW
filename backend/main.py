from pathlib import Path
import hashlib
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "cogew_rf_cnn_best.pt"
TEST_PATH = ROOT / "data" / "splits" / "cogew_test.h5"

DASHBOARD_PATH = ROOT / "dashboard"
PAGES_PATH = DASHBOARD_PATH / "pages"
ASSETS_PATH = DASHBOARD_PATH / "assets"


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
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ============================================================
# MODEL
# IMPORTANT:
# Architecture must match train_cogew_rf_cnn.py
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

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("=" * 70)
print("COG-EW FAST RF API")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

print("Loading model...")

MODEL = COGEWRFCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)


checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


MODEL.load_state_dict(state_dict)

MODEL.eval()

MODEL_HASH = calculate_sha256(MODEL_PATH)

print("[PASS] Model loaded")
print("SHA256:", MODEL_HASH)


# ============================================================
# LOAD TEST DATA ONCE
#
# This is deliberate.
#
# We DO NOT open HDF5 for every API request.
# ============================================================

print("Loading test dataset into RAM...")

with h5py.File(TEST_PATH, "r") as f:

    TEST_X = f["X"][:].astype(np.float32)

    TEST_Y = f["Y"][:].astype(np.int64)

    TEST_Z = f["Z"][:].astype(np.int64)

    TEST_SOURCE = f["source"][:].astype(np.int64)

    TEST_ORIGINAL_INDEX = f["original_index"][:].astype(np.int64)


print("[PASS] Test samples:", len(TEST_X))


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="COG-EW RF API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount("/assets", StaticFiles(directory=ASSETS_PATH), name="assets")


# ============================================================
# SAMPLE ACCESS
# ============================================================


def get_sample(index):

    if index < 0 or index >= len(TEST_X):

        raise HTTPException(
            status_code=404, detail=(f"Invalid test index. " f"Use 0-{len(TEST_X)-1}.")
        )

    return (
        TEST_X[index],
        int(TEST_Y[index]),
        int(TEST_Z[index]),
        int(TEST_SOURCE[index]),
        int(TEST_ORIGINAL_INDEX[index]),
    )


# ============================================================
# FAST MODEL INFERENCE
# ============================================================


@torch.inference_mode()
def predict_signal(iq):

    # Original:
    #
    # (128, 2)
    #
    # Required by CNN:
    #
    # (batch, channels, time)
    #
    # => (1, 2, 128)

    tensor = torch.from_numpy(iq.T).unsqueeze(0)

    tensor = tensor.to(DEVICE, non_blocking=True)

    logits = MODEL(tensor)

    probabilities = torch.softmax(logits, dim=1)[0]

    probabilities = probabilities.detach().cpu().numpy()

    predicted_index = int(np.argmax(probabilities))

    return (predicted_index, probabilities)


# ============================================================
# SIGNAL QUALITY
# ============================================================


def calculate_signal_quality(iq):

    i = iq[:, 0]

    q = iq[:, 1]

    complex_signal = i + 1j * q

    power = float(np.mean(np.abs(complex_signal) ** 2))

    dc_offset = float(np.abs(np.mean(complex_signal)))

    i_std = float(np.std(i))

    q_std = float(np.std(q))

    iq_imbalance = float(abs(i_std - q_std) / max(i_std + q_std, 1e-12))

    rms = float(np.sqrt(power))

    peak = float(np.max(np.abs(complex_signal)))

    papr = float(peak**2 / max(power, 1e-12))

    return {
        "signal_power": power,
        "rms_amplitude": rms,
        "peak_amplitude": peak,
        "papr": papr,
        "dc_offset": dc_offset,
        "iq_imbalance": iq_imbalance,
    }


# ============================================================
# QUALITY DECISION
# ============================================================


def determine_quality(confidence, metrics):

    if confidence >= 0.85 and metrics["iq_imbalance"] < 0.10:

        return "GOOD"

    if confidence >= 0.60:

        return "MODERATE"

    return "LOW"


# ============================================================
# DECISION SUPPORT
# ============================================================


def generate_decision(confidence, quality):

    if confidence >= 0.85 and quality == "GOOD":

        return {
            "state": "HIGH_CONFIDENCE_RF_TRACK",
            "action": "CONTINUE_SIMULATION",
            "reason": "RF classification confidence and signal quality are strong.",
        }

    if confidence >= 0.60:

        return {
            "state": "MEDIUM_CONFIDENCE_RF_TRACK",
            "action": "CLASSIFICATION_REVIEW",
            "reason": "Classification can be used for simulation, but additional RF evidence is recommended.",
        }

    return {
        "state": "LOW_CONFIDENCE_RF_TRACK",
        "action": "REQUEST_ADDITIONAL_EVIDENCE",
        "reason": "RF confidence is too low for a reliable classification decision.",
    }


# ============================================================
# HOME
# ============================================================


@app.get("/")
def home():

    return FileResponse(DASHBOARD_PATH / "index.html")


# ============================================================
# PAGES
# ============================================================


@app.get("/pages/{page_name}")
def dashboard_page(page_name: str):

    allowed_pages = {
        "overview": "overview.html",
        "rf_classification": "rf_classification.html",
        "signal_analysis": "signal_analysis.html",
        "decision_support": "decision_support.html",
        "performance": "performance.html",
    }

    if page_name not in allowed_pages:

        raise HTTPException(status_code=404, detail="Dashboard page not found.")

    return FileResponse(PAGES_PATH / allowed_pages[page_name])


# ============================================================
# HEALTH
# ============================================================


@app.get("/api/health")
def health():

    return {
        "status": "online",
        "device": str(DEVICE),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "model_loaded": True,
        "test_samples": len(TEST_X),
    }


# ============================================================
# MODEL INFO
# ============================================================


@app.get("/api/model")
def model_info():

    return {
        "model": "COGEW RFCNN",
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
        "model_sha256": MODEL_HASH,
        "test_samples": len(TEST_X),
        "test_accuracy": 64.75,
    }


# ============================================================
# PREDICTION
# ============================================================


@app.get("/api/predict/{index}")
def predict(index: int):

    (iq, true_class, snr, source, original_index) = get_sample(index)

    start = time.perf_counter()

    predicted_index, probabilities = predict_signal(iq)

    latency = (time.perf_counter() - start) * 1000

    probability_map = {
        CLASS_NAMES[i]: round(float(probabilities[i] * 100), 2)
        for i in np.argsort(probabilities)[::-1]
    }

    return {
        "index": index,
        "original_index": original_index,
        "source_id": source,
        "source": SOURCE_NAMES.get(source, str(source)),
        "snr_db": snr,
        "true_class": CLASS_NAMES[true_class],
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index] * 100), 2),
        "correct": predicted_index == true_class,
        "probabilities": probability_map,
        "latency_ms": round(latency, 3),
    }


# ============================================================
# COMPLETE SIGNAL ANALYSIS
# ============================================================


@app.get("/api/analyze/{index}")
def analyze(index: int):

    (iq, true_class, snr, source, original_index) = get_sample(index)

    predicted_index, probabilities = predict_signal(iq)

    confidence = float(probabilities[predicted_index])

    metrics = calculate_signal_quality(iq)

    quality = determine_quality(confidence, metrics)

    decision = generate_decision(confidence, quality)

    return {
        "index": index,
        "original_index": original_index,
        "source": SOURCE_NAMES.get(source, str(source)),
        "dataset_snr_db": snr,
        "true_class": CLASS_NAMES[true_class],
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2),
        "quality": quality,
        "reliability": (
            "HIGH" if confidence >= 0.85 else "MEDIUM" if confidence >= 0.60 else "LOW"
        ),
        "metrics": metrics,
        "decision": decision,
    }


print("=" * 70)
print("COG-EW FAST API READY")
print("=" * 70)
