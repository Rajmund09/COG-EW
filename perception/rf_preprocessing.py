"""
RF Signal Preprocessing (perception/rf_preprocessing.py)
Extracts pulse parameters (PRF, pulse width, bandwidth, Direction of Arrival [DoA]) from raw IQ/pulse data.
"""

import numpy as np


class RFPulsePreprocessor:
    """
    Simulates RF front-end receiver processing and pulse descriptor word (PDW) generation.
    """

    def __init__(self, sample_rate_ghz: float = 1.0):
        self.sample_rate_ghz = sample_rate_ghz

    def extract_pulse_descriptors(self, raw_signal: np.ndarray, rx_position: np.ndarray, emitter_pos: np.ndarray) -> dict:
        """
        Generate Pulse Descriptor Word (PDW) vector from intercepted RF energy.
        """
        # Direction of Arrival (DoA) calculation
        relative_vec = emitter_pos - rx_position
        doa_rad = np.arctan2(relative_vec[1], relative_vec[0])
        
        # Synthetic noise injection on pulse measurements
        doa_noise = np.random.normal(0, np.radians(0.5))  # 0.5 deg noise
        
        pdw = {
            "center_freq_ghz": 3.0 + np.random.normal(0, 0.05),
            "pulse_width_us": 10.0 + np.random.normal(0, 0.1),
            "prf_hz": 1000.0 + np.random.normal(0, 5.0),
            "bandwidth_mhz": 5.0 + np.random.normal(0, 0.2),
            "doa_rad": doa_rad + doa_noise,
            "snr_db": 15.0 + np.random.normal(0, 1.0)
        }
        return pdw

    def pdw_to_vector(self, pdw: dict) -> np.ndarray:
        """Convert PDW dictionary into standardized normalized feature vector."""
        return np.array([
            pdw["center_freq_ghz"] / 15.0,
            pdw["pulse_width_us"] / 100.0,
            pdw["prf_hz"] / 10000.0,
            pdw["bandwidth_mhz"] / 50.0,
            pdw["doa_rad"] / np.pi,
            pdw["snr_db"] / 40.0
        ], dtype=np.float32)
