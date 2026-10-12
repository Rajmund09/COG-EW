"""
======================================================================
COG-EW FAST DASHBOARD API
======================================================================

Purpose:
    FastAPI backend for the COG-EW RF monitoring dashboard.

Design goals:
    - Fast startup
    - Load CNN model only once
    - Avoid loading large HDF5 datasets repeatedly
    - Use existing JSON reports wherever possible
    - Lightweight REST API
    - Serve dashboard HTML/CSS/JS directly

IMPORTANT:
    This dashboard provides RF classification, signal-quality analysis,
    model monitoring and simulation decision support.

    It does not implement operational radar jamming, missile control,
    or target-engagement functionality.
======================================================================
"""

from pathlib import Path
import json
import hashlib
import sys
import time
from functools import lru_cache

import numpy as np
import h5py
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DASHBOARD_DIR = ROOT_DIR / "dashboard"
PAGES_DIR = DASHBOARD_DIR / "pages"
ASSETS_DIR = DASHBOARD_DIR / "assets"

MODEL_PATH = ROOT_DIR / "models" / "cogew_rf_cnn_best.pt"

TEST_DATASET = ROOT_DIR / "data" / "splits" / "cogew_test.h5"

REPORTS_DIR = ROOT_DIR / "reports"

SUMMARY_JSON = REPORTS_DIR / "cogew_rf_model_summary.json"
EVALUATION_JSON = REPORTS_DIR / "cogew_rf_cnn_test_evaluation.json"
QUALITY_JSON = REPORTS_DIR / "rf_signal_quality_analysis.json"
DECISION_JSON = REPORTS_DIR / "cogew_decision.json"
MODEL_CARD_JSON = REPORTS_DIR / "rf_model_card.json"


# ======================================================================
# CLASS DEFINITIONS
# ======================================================================

CLASS_NAMES = [
    "8PSK",
    "BPSK",
    "QPSK",
    "16QAM",
    "64QAM",
]


# ======================================================================
# MODEL
# ======================================================================


class COGEWRFCNN(nn.Module):
    """
    Must match the architecture used during training.
    """

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


# ======================================================================
# DEVICE
# ======================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# MODEL LOADING
# ======================================================================


@lru_cache(maxsize=1)
def load_model():
    """
    Load the CNN only once.

    This is important for dashboard speed.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = COGEWRFCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
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

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ======================================================================
# FILE UTILITIES
# ======================================================================


def load_json(path: Path):

    if not path.exists():
        return {
            "status": "missing",
            "file": str(path),
        }

    try:

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:

        return {
            "status": "error",
            "file": str(path),
            "error": str(exc),
        }


def calculate_sha256(path: Path):

    if not path.exists():
        return None

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ======================================================================
# FAST TEST SAMPLE ACCESS
# ======================================================================


def load_test_sample(index: int):

    if not TEST_DATASET.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET}")

    with h5py.File(TEST_DATASET, "r") as h5:

        total = h5["X"].shape[0]

        if index < 0 or index >= total:
            raise IndexError(f"Index must be between 0 and {total - 1}")

        x = np.asarray(h5["X"][index], dtype=np.float32)

        y = int(h5["Y"][index])

        z = int(h5["Z"][index])

        original_index = int(h5["original_index"][index])

        source = int(h5["source"][index])

    return {
        "x": x,
        "y": y,
        "snr": z,
        "original_index": original_index,
        "source": source,
    }


# ======================================================================
# RF PREDICTION
# ======================================================================


@torch.inference_mode()
def predict_signal(x: np.ndarray):

    if x.shape != (128, 2):
        raise ValueError(f"Expected signal shape (128, 2), got {x.shape}")

    start_time = time.perf_counter()

    signal = torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE,
    )

    # (128, 2) -> (1, 2, 128)
    signal = signal.transpose(0, 1)
    signal = signal.unsqueeze(0)

    model = load_model()

    logits = model(signal)

    probabilities = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000

    predicted_index = int(torch.argmax(probabilities).item())
    confidence_prob = float(probabilities[predicted_index].item())

    # Raw float probabilities that sum to 1.0 (between 0 and 1)
    prob_dict = {
        CLASS_NAMES[i]: float(probabilities[i].item())
        for i in range(len(CLASS_NAMES))
    }

    # Percentages for UI display
    prob_pct_dict = {
        CLASS_NAMES[i]: round(float(probabilities[i].item()) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "predicted_index": predicted_index,
        "confidence_prob": confidence_prob,
        "confidence_percent": round(confidence_prob * 100, 2),
        "probabilities": prob_dict,
        "probability_percentages": prob_pct_dict,
        "inference_ms": round(latency_ms, 3),
    }


# ======================================================================
# SIGNAL QUALITY
# ======================================================================


def analyze_signal_quality(x):

    i = x[:, 0]
    q = x[:, 1]

    complex_signal = i + 1j * q

    power = float(np.mean(np.abs(complex_signal) ** 2))

    i_power = float(np.mean(i**2))

    q_power = float(np.mean(q**2))

    imbalance = abs(i_power - q_power)

    denominator = i_power + q_power + 1e-12

    iq_imbalance = float(imbalance / denominator)

    dc_offset = float(np.abs(np.mean(complex_signal)))

    if power < 0.5:

        quality = "LOW"

    elif power < 1.5:

        quality = "MODERATE"

    else:

        quality = "GOOD"

    return {
        "signal_power": round(power, 6),
        "iq_imbalance": round(iq_imbalance, 6),
        "dc_offset": round(dc_offset, 6),
        "signal_quality": quality,
    }


# ======================================================================
# APPLICATION
# ======================================================================

app = FastAPI(
    title="COG-EW RF Dashboard API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/api/docs", include_in_schema=False)
def redirect_api_docs():
    return RedirectResponse(url="/docs")


# ======================================================================
# STATIC FILES
# ======================================================================

app.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR)),
    name="assets",
)


# ======================================================================
# HEALTH
# ======================================================================


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "online": True,
        "device": str(DEVICE),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "cuda_available": torch.cuda.is_available(),
        "model_exists": MODEL_PATH.exists(),
        "model_available": MODEL_PATH.exists(),
        "test_dataset_exists": TEST_DATASET.exists(),
        "test_dataset_available": TEST_DATASET.exists(),
    }


# ======================================================================
# SYSTEM INFORMATION
# ======================================================================


@app.get("/api/system")
def system():

    return {
        "project": "COG-EW",
        "component": "RF Automatic Modulation Classification",
        "device": str(DEVICE),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "model": MODEL_PATH.name,
        "model_sha256": calculate_sha256(MODEL_PATH),
        "classes": CLASS_NAMES,
    }


# ======================================================================
# MODEL SUMMARY
# ======================================================================


@app.get("/api/model-summary")
def model_summary():

    return load_json(SUMMARY_JSON)


# ======================================================================
# TEST EVALUATION
# ======================================================================


@app.get("/api/evaluation")
def evaluation():

    return load_json(EVALUATION_JSON)


# ======================================================================
# SIGNAL QUALITY REPORT
# ======================================================================


@app.get("/api/signal-quality")
def signal_quality():

    return load_json(QUALITY_JSON)


# ======================================================================
# DECISION SUPPORT
# ======================================================================


@app.get("/api/decision")
def decision():

    return load_json(DECISION_JSON)


# ======================================================================
# MODEL CARD
# ======================================================================


@app.get("/api/model-card")
def model_card():

    return load_json(MODEL_CARD_JSON)


@app.get("/api/metrics")
def metrics():

    return load_json(EVALUATION_JSON)


# ======================================================================
# TEST SAMPLE REQUEST
# ======================================================================


class SampleRequest(BaseModel):

    test_index: int | None = None
    index: int | None = None

    @property
    def target_index(self) -> int:
        if self.test_index is not None:
            return self.test_index
        if self.index is not None:
            return self.index
        return 5000


@app.post("/api/predict")
@app.post("/api/predict-test")
def predict_test(request: SampleRequest):

    try:
        idx = request.target_index
        sample = load_test_sample(idx)

        prediction = predict_signal(sample["x"])

        quality = analyze_signal_quality(sample["x"])

        true_class = CLASS_NAMES[sample["y"]]
        is_correct = bool(prediction["predicted_index"] == sample["y"])

        return {
            "test_index": idx,
            "index": idx,
            "original_index": sample["original_index"],
            "source_id": sample["source"],
            "source": ("RadioML2016" if sample["source"] == 0 else "RadioML2018"),
            "source_name": ("RadioML2016" if sample["source"] == 0 else "RadioML2018"),
            "snr_db": sample["snr"],
            "snr": sample["snr"],
            "true_class": true_class,
            "predicted_class": prediction["predicted_class"],
            "predicted_index": prediction["predicted_index"],
            "confidence": prediction["confidence_prob"],
            "confidence_percent": prediction["confidence_percent"],
            "inference_ms": prediction["inference_ms"],
            "correct": is_correct,
            "probabilities": prediction["probabilities"],
            "prediction": {
                "predicted_class": prediction["predicted_class"],
                "predicted_index": prediction["predicted_index"],
                "confidence": prediction["confidence_percent"],
                "probabilities": prediction["probability_percentages"],
            },
            "quality": quality,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ======================================================================
# DASHBOARD PAGES
# ======================================================================


@app.get("/")
def dashboard_home():

    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/pages/{page_name}")
def dashboard_page(page_name: str):

    allowed_pages = {
        "overview.html",
        "rf_classification.html",
        "signal_analysis.html",
        "decision_support.html",
        "model_performance.html",
        "performance.html",
    }

    if page_name not in allowed_pages:
        raise HTTPException(
            status_code=404,
            detail="Dashboard page not found",
        )

    # Alias performance.html to model_performance.html if needed
    actual_name = "model_performance.html" if page_name == "performance.html" else page_name
    path = PAGES_DIR / actual_name
    if not path.exists() and (PAGES_DIR / page_name).exists():
        path = PAGES_DIR / page_name

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard page file does not exist",
        )

    return FileResponse(path)


# ======================================================================
# STARTUP
# ======================================================================


@app.on_event("startup")
def startup_event():

    print()
    print("=" * 70)
    print("COG-EW FASTAPI DASHBOARD")
    print("=" * 70)

    print(f"Device : {DEVICE}")

    print(f"Model  : {MODEL_PATH}")

    # Load model once.
    try:

        load_model()

        print("[PASS] CNN loaded into memory")

    except Exception as exc:

        print("[WARNING] Model could not be loaded:")

        print(exc)

    print("[PASS] FastAPI dashboard ready")

    print("=" * 70)
