from pydantic import BaseModel
from typing import List, Optional


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    snr: Optional[float] = None
    signal_quality: Optional[str] = None
    reliability: Optional[str] = None
    decision: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model: str


class SignalAnalysisResponse(BaseModel):
    predicted_class: str
    confidence: float
    true_class: Optional[str] = None

    signal_power: float
    estimated_snr: float
    iq_imbalance: float
    dc_offset: float

    signal_quality: str
    reliability: str
    decision: str
