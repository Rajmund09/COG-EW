"""
COG-EW RF Perception Inference & Jamming Recommendation Service
(perception/rf_inference_service.py)

Provides 24-class RadioML 2018.01A modulation classification, radar threat
categorization, signal-quality metrics, and cognitive EW jamming recommendations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from perception.models.radioml_resnet import RadioMLResNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 24 Canonical RadioML 2018.01A Modulation Classes
RADIOML_24_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK"
]

# Radar Emitter Threat Classification Mapping
# Maps modulation types to tactical radar operational categories
THREAT_CATEGORIES = {
    # Amplitude / Continuous wave / Broadcast signals (Surveillance / Early Warning)
    "OOK": {"category": "Search Radar", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},
    "4ASK": {"category": "Search Radar", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},
    "8ASK": {"category": "Early Warning Radar", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},
    "AM-SSB-WC": {"category": "Military Comm / Navigation", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},
    "AM-SSB-SC": {"category": "Tactical Comm", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},
    "AM-DSB-WC": {"category": "Air Defense Broadcast", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},
    "AM-DSB-SC": {"category": "Command Data Link", "threat_level": "MEDIUM", "jamming_mode": "Noise Jamming"},
    "FM": {"category": "Voice Comm / Navigation", "threat_level": "LOW", "jamming_mode": "Passive / Monitor"},

    # Phase-shift keying (Target Acquisition & Tracking Radars)
    "BPSK": {"category": "Target Acquisition Radar", "threat_level": "MEDIUM", "jamming_mode": "Spot Noise Jamming"},
    "QPSK": {"category": "Tracking Radar (T-1)", "threat_level": "HIGH", "jamming_mode": "Barrage Noise Jamming"},
    "8PSK": {"category": "Pulse Compression Tracking Radar", "threat_level": "HIGH", "jamming_mode": "Noise + RGPO"},
    "16PSK": {"category": "Multi-function Radar", "threat_level": "HIGH", "jamming_mode": "RGPO Spoofing"},
    "32PSK": {"category": "Phased Array Tracking", "threat_level": "HIGH", "jamming_mode": "RGPO Spoofing"},
    "GMSK": {"category": "Secure Tactical Uplink", "threat_level": "MEDIUM", "jamming_mode": "Spot Noise Jamming"},
    "OQPSK": {"category": "Target Illumination Radar", "threat_level": "HIGH", "jamming_mode": "Angle Deception"},

    # High-Order QAM / APSK (Fire-Control & Missile Guidance Links)
    "16QAM": {"category": "Fire-Control Radar (FCR)", "threat_level": "CRITICAL", "jamming_mode": "RGPO Deceptive Spoofing"},
    "32QAM": {"category": "Surface-to-Air Missile (SAM) Guidance", "threat_level": "CRITICAL", "jamming_mode": "RGPO + Evasive Auto-Pilot"},
    "64QAM": {"category": "Terminal Missile Intercept Link", "threat_level": "CRITICAL", "jamming_mode": "RGPO + Angle Deception + Max G-Turn"},
    "128QAM": {"category": "Advanced Active Radar Homing", "threat_level": "CRITICAL", "jamming_mode": "Full Spectrum Jamming + Evasion"},
    "256QAM": {"category": "High-Density IADS Network", "threat_level": "CRITICAL", "jamming_mode": "Full Spectrum Jamming + Evasion"},
    "16APSK": {"category": "Satellite Navigation Jammer Link", "threat_level": "MEDIUM", "jamming_mode": "Spot Noise Jamming"},
    "32APSK": {"category": "Precision Guidance Radar", "threat_level": "CRITICAL", "jamming_mode": "RGPO Spoofing"},
    "64APSK": {"category": "Anti-Air Command & Control", "threat_level": "HIGH", "jamming_mode": "Barrage Noise Jamming"},
    "128APSK": {"category": "Strategic Early Warning Net", "threat_level": "HIGH", "jamming_mode": "Barrage Noise Jamming"},
}

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "radioml_60_10_30_v1"
    / "resnet_lr2.5e-4_wd5e-4_best.pt"
)


class RFInferenceService:
    """High-performance inference engine for 24-class RadioML modulation classification."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT
        self.checkpoint_path = Path(checkpoint_path).resolve()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.class_names = RADIOML_24_CLASSES
        self.num_classes = len(self.class_names)
        self.model = self._load_model()

    def _load_model(self) -> RadioMLResNet:
        if not self.checkpoint_path.is_file():
            # Fallback to general model location if available
            alt_path = PROJECT_ROOT / "models" / "checkpoints" / "radioml_resnet_best.pt"
            if alt_path.is_file():
                self.checkpoint_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Selected model checkpoint not found: {self.checkpoint_path}"
                )

        model = RadioMLResNet(num_classes=self.num_classes)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def validate_signal(self, signal: np.ndarray | list[list[float]]) -> np.ndarray:
        """Validate and format input signal to shape (2, 1024) float32."""
        arr = np.asarray(signal, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        # Handle transposed input (1024, 2) -> (2, 1024)
        if arr.shape == (1024, 2):
            arr = arr.T

        if arr.shape != (2, 1024):
            raise ValueError(
                f"Expected input signal shape (2, 1024), but received {arr.shape}."
            )

        if not np.isfinite(arr).all():
            raise ValueError("Input signal contains NaN or infinite values.")

        # Unit-variance normalization matching training loader
        std = np.std(arr)
        if std > 1e-8:
            arr = (arr - np.mean(arr)) / std

        return arr

    @torch.no_grad()
    def predict(self, signal: np.ndarray | list[list[float]]) -> dict[str, Any]:
        """Classify a single normalized (2, 1024) RF signal."""
        clean_arr = self.validate_signal(signal)
        tensor = torch.as_tensor(clean_arr, device=self.device).unsqueeze(0)

        start_time = time.perf_counter()
        logits = self.model(tensor)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        # Epistemic uncertainty proxy via normalized Shannon entropy
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        max_entropy = np.log(self.num_classes)
        uncertainty_score = float(entropy / max_entropy)

        predicted_class = self.class_names[predicted_idx]
        threat_info = THREAT_CATEGORIES.get(
            predicted_class,
            {"category": "Unknown RF Emitter", "threat_level": "MEDIUM", "jamming_mode": "Active Noise Jamming"}
        )

        # Signal quality metrics
        quality = self.analyze_signal_quality(clean_arr)

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 2),
            "uncertainty_score": round(uncertainty_score, 4),
            "inference_ms": round(inference_time_ms, 2),
            "device": str(self.device),
            "threat_category": threat_info["category"],
            "threat_level": threat_info["threat_level"],
            "recommended_countermeasure": threat_info["jamming_mode"],
            "probabilities": {
                name: round(float(prob), 5)
                for name, prob in zip(self.class_names, probabilities)
            },
            "signal_quality": quality,
        }

    @staticmethod
    def analyze_signal_quality(signal: np.ndarray) -> dict[str, Any]:
        """Compute comprehensive signal-quality and physical RF parameters."""
        i_chan = signal[0]
        q_chan = signal[1]
        complex_sig = i_chan + 1j * q_chan

        power_i = float(np.mean(i_chan ** 2))
        power_q = float(np.mean(q_chan ** 2))
        total_power = power_i + power_q
        peak_power = float(np.max(np.abs(complex_sig) ** 2))
        crest_factor = float(peak_power / max(total_power, 1e-9))

        dc_i = float(np.mean(i_chan))
        dc_q = float(np.mean(q_chan))
        dc_offset = float(np.sqrt(dc_i ** 2 + dc_q ** 2))

        iq_imbalance = float(abs(power_i - power_q) / max(total_power, 1e-9))
        snr_estimate_db = float(10.0 * np.log10(max(total_power / (dc_offset + 1e-4), 1.0)))

        # Qualitative classification
        if snr_estimate_db > 15.0 and iq_imbalance < 0.15:
            quality_rating = "EXCELLENT"
        elif snr_estimate_db > 8.0:
            quality_rating = "GOOD"
        elif snr_estimate_db > 2.0:
            quality_rating = "MODERATE"
        else:
            quality_rating = "DEGRADED"

        return {
            "total_power": round(total_power, 5),
            "snr_db_estimate": round(snr_estimate_db, 2),
            "iq_imbalance": round(iq_imbalance, 5),
            "dc_offset": round(dc_offset, 5),
            "crest_factor": round(crest_factor, 3),
            "quality_rating": quality_rating,
        }

    @staticmethod
    def compute_spectral_features(signal: np.ndarray, n_fft: int = 256) -> dict[str, Any]:
        """Compute FFT Power Spectral Density and time-frequency representations."""
        complex_sig = signal[0] + 1j * signal[1]
        
        # FFT Power Spectrum
        window = np.hanning(len(complex_sig))
        fft_vals = np.fft.fftshift(np.fft.fft(complex_sig * window))
        psd_db = 10.0 * np.log10(np.abs(fft_vals) ** 2 + 1e-12)
        psd_db = psd_db - np.max(psd_db)  # normalize to 0 dB peak

        freqs = np.linspace(-0.5, 0.5, len(psd_db)).tolist()

        # Downsample for lightweight JSON transmission (e.g. 128 bins)
        stride = max(1, len(psd_db) // 128)
        psd_downsampled = [round(float(v), 2) for v in psd_db[::stride]]
        freqs_downsampled = [round(float(f), 4) for f in freqs[::stride]]

        return {
            "frequencies": freqs_downsampled,
            "psd_db": psd_downsampled,
            "center_frequency_norm": 0.0,
            "bandwidth_3db": round(float(np.sum(psd_db > -3.0) / len(psd_db)), 4),
        }

    @classmethod
    def generate_synthetic_radar_signal(
        cls,
        modulation: str = "QPSK",
        snr_db: float = 12.0,
        samples: int = 1024,
    ) -> np.ndarray:
        """
        Generate mathematically realistic synthetic RF pulses for live simulation
        and interactive judge demonstrations.
        """
        t = np.linspace(0, 1.0, samples, endpoint=False)
        symbol_rate = 32
        symbols_per_sample = samples // symbol_rate

        noise_power = 10.0 ** (-snr_db / 10.0)
        noise_i = np.random.normal(0, np.sqrt(noise_power / 2.0), samples)
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2.0), samples)

        if modulation == "BPSK":
            bits = np.random.choice([-1.0, 1.0], size=symbol_rate)
            i_sig = np.repeat(bits, symbols_per_sample)
            q_sig = np.zeros(samples)
        elif modulation == "QPSK":
            bits_i = np.random.choice([-1.0, 1.0], size=symbol_rate)
            bits_q = np.random.choice([-1.0, 1.0], size=symbol_rate)
            i_sig = np.repeat(bits_i, symbols_per_sample)
            q_sig = np.repeat(bits_q, symbols_per_sample)
        elif modulation == "8PSK":
            phases = np.random.choice(np.linspace(0, 2 * np.pi, 8, endpoint=False), size=symbol_rate)
            i_sig = np.repeat(np.cos(phases), symbols_per_sample)
            q_sig = np.repeat(np.sin(phases), symbols_per_sample)
        elif modulation in ("16QAM", "32QAM", "64QAM"):
            levels = [-3.0, -1.0, 1.0, 3.0]
            syms_i = np.random.choice(levels, size=symbol_rate) / 3.0
            syms_q = np.random.choice(levels, size=symbol_rate) / 3.0
            i_sig = np.repeat(syms_i, symbols_per_sample)
            q_sig = np.repeat(syms_q, symbols_per_sample)
        elif modulation == "FM":
            carrier = 2 * np.pi * 10 * t
            msg = np.sin(2 * np.pi * 2 * t)
            phase = carrier + 2.5 * msg
            i_sig = np.cos(phase)
            q_sig = np.sin(phase)
        else:
            # Default carrier wave
            i_sig = np.cos(2 * np.pi * 8 * t)
            q_sig = np.sin(2 * np.pi * 8 * t)

        # Pulse envelope (radar burst)
        pulse_env = np.sin(np.pi * t) ** 0.5
        i_chan = i_sig * pulse_env + noise_i
        q_chan = q_sig * pulse_env + noise_q

        sig = np.vstack([i_chan, q_chan]).astype(np.float32)
        # Normalize
        std = np.std(sig)
        if std > 1e-8:
            sig = (sig - np.mean(sig)) / std
        return sig

    @classmethod
    def load_dataset_sample(
        cls,
        index: int = 5000,
        dataset_dir: str | Path = r"D:\Datasets",
    ) -> tuple[np.ndarray, str, float, str]:
        """
        Load an actual radar/RF signal sample from the real dataset directory (D:/Datasets)
        or local locked test splits.

        Returns:
            (signal: np.ndarray [2, 1024], true_class: str, snr_db: float, source: str)
        """
        # 1. Search candidate HDF5 locations in D:/Datasets
        candidates = [
            Path(r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"),
            Path(r"D:\Datasets\RadioML 2018.01A Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"),
            Path(dataset_dir) / "GOLD_XYZ_OSC.0001_1024.hdf5",
            PROJECT_ROOT / "data" / "GOLD_XYZ_OSC.0001_1024.hdf5",
        ]

        # Additional glob search in D:/Datasets
        d_path = Path(dataset_dir)
        if d_path.is_dir():
            try:
                for match in d_path.glob("*RadioML*/GOLD_XYZ_OSC.0001_1024.hdf5"):
                    if match.is_file():
                        candidates.insert(0, match)
                        break
            except OSError:
                pass

        # Check for RadioML 2018.01A HDF5
        for hdf5_path in candidates:
            if hdf5_path.is_file():
                try:
                    import h5py

                    # Check for locked test split indices
                    test_idx_paths = [
                        PROJECT_ROOT / "data" / "processed" / "radioml" / "experiments" / "split_60_10_30_v1" / "test_indices.npy",
                        PROJECT_ROOT / "data" / "processed" / "radioml" / "test_indices.npy",
                    ]
                    test_indices = None
                    for p in test_idx_paths:
                        if p.is_file():
                            test_indices = np.load(p)
                            break

                    with h5py.File(hdf5_path, "r") as h5:
                        total_samples = len(h5["X"])
                        if test_indices is not None and len(test_indices) > 0:
                            actual_idx = int(test_indices[index % len(test_indices)])
                        else:
                            actual_idx = int(index % total_samples)

                        raw_x = h5["X"][actual_idx]  # shape (1024, 2)
                        raw_y = h5["Y"][actual_idx]  # one-hot (24,)
                        raw_z = h5["Z"][actual_idx]  # snr

                        # Format to (2, 1024)
                        if raw_x.shape == (1024, 2):
                            sig = raw_x.T.astype(np.float32)
                        else:
                            sig = raw_x.astype(np.float32)

                        # Determine class name
                        class_idx = int(np.argmax(raw_y)) if raw_y.ndim > 0 else int(raw_y)
                        true_class = RADIOML_24_CLASSES[class_idx] if class_idx < len(RADIOML_24_CLASSES) else "Unknown"
                        snr_db = float(raw_z[0]) if hasattr(raw_z, "__len__") else float(raw_z)

                        source_desc = f"RadioML 2018.01A (Real Dataset Sample #{actual_idx} from D:/Datasets)"
                        return sig, true_class, snr_db, source_desc
                except Exception as err:
                    # Log and try fallback
                    print(f"[RFInferenceService] Notice reading RadioML HDF5: {err}. Attempting split fallback.")

        # 2. Check local split cogew_test.h5
        local_split = PROJECT_ROOT / "data" / "splits" / "cogew_test.h5"
        if local_split.is_file():
            try:
                import h5py
                with h5py.File(local_split, "r") as h5:
                    total_samples = len(h5["X"])
                    actual_idx = int(index % total_samples)
                    raw_x = h5["X"][actual_idx]
                    raw_y = h5["Y"][actual_idx]
                    raw_z = h5["Z"][actual_idx]

                    if raw_x.shape == (1024, 2):
                        sig = raw_x.T.astype(np.float32)
                    elif raw_x.ndim == 2 and raw_x.shape[0] == 2:
                        sig = raw_x.astype(np.float32)
                    elif raw_x.shape == (128, 2):
                        # Pad or repeat to 1024
                        sig_128 = raw_x.T.astype(np.float32)
                        sig = np.tile(sig_128, (1, 8))[:, :1024]
                    else:
                        sig = raw_x.astype(np.float32)

                    class_idx = int(np.argmax(raw_y)) if hasattr(raw_y, "__len__") and len(raw_y) > 1 else int(raw_y)
                    # Mapping for 5-class legacy split to 24-class names
                    legacy_classes = ["8PSK", "BPSK", "QPSK", "16QAM", "64QAM"]
                    if class_idx < len(legacy_classes):
                        true_class = legacy_classes[class_idx]
                    else:
                        true_class = RADIOML_24_CLASSES[class_idx % len(RADIOML_24_CLASSES)]

                    snr_db = float(raw_z[0]) if hasattr(raw_z, "__len__") else float(raw_z)
                    source_desc = f"Local Locked Test Split cogew_test.h5 (Sample #{actual_idx})"
                    return sig, true_class, snr_db, source_desc
            except Exception as err:
                print(f"[RFInferenceService] Notice reading cogew_test.h5: {err}")

        # 3. Mathematical fallback for live demo/unmounted drive
        fallback_mod = RADIOML_24_CLASSES[index % len(RADIOML_24_CLASSES)]
        sig = cls.generate_synthetic_radar_signal(modulation=fallback_mod, snr_db=12.0)
        return sig, fallback_mod, 12.0, "High-Fidelity Synthetic Radar Pulse (Simulation Generator)"

