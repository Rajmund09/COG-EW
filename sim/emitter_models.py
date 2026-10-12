"""
RF / Emitter Environment Models (sim/emitter_models.py)
Defines radar emitters, scan patterns, multi-static IADS layout, and detection probability calculations.
"""

import numpy as np


class Emitter:
    """Represents an individual radar emitter in an IADS network."""

    def __init__(
        self,
        emitter_id: str,
        emitter_type: str,
        position: np.ndarray,
        freq_ghz: float = 3.0,
        prf_hz: float = 1000.0,
        max_range_km: float = 200.0,
        peak_power_kw: float = 100.0
    ):
        self.id = emitter_id
        self.type = emitter_type  # 'search', 'tracking', 'fire_control'
        self.position = np.array(position, dtype=np.float64)
        self.freq_ghz = freq_ghz
        self.prf_hz = prf_hz
        self.max_range_m = max_range_km * 1000.0
        self.peak_power_kw = peak_power_kw
        
        self.is_active = True
        self.scan_angle = 0.0  # current antenna azimuth scan angle (rad)
        self.scan_speed = np.radians(60.0) if emitter_type == "search" else np.radians(120.0)

    def update_scan(self, dt: float):
        """Update mechanical or electronic antenna scan phase."""
        self.scan_angle = (self.scan_angle + self.scan_speed * dt) % (2 * np.pi)

    def compute_snr(self, target_pos: np.ndarray, rcs_sqm: float = 0.1, jamming_noise_db: float = 0.0) -> float:
        """
        Radar Equation calculation for Signal-to-Noise Ratio (SNR) in dB.
        """
        distance = np.linalg.norm(target_pos - self.position)
        if distance > self.max_range_m:
            return -100.0  # Out of maximum detection range

        # Basic R^4 path loss calculation
        c = 3e8
        wavelength = c / (self.freq_ghz * 1e9)
        pt = self.peak_power_kw * 1e3
        gain = 35.0  # dB antenna gain
        gain_linear = 10 ** (gain / 10.0)

        numerator = pt * (gain_linear ** 2) * (wavelength ** 2) * rcs_sqm
        denominator = ((4 * np.pi) ** 3) * (distance ** 4)
        
        pr = numerator / max(denominator, 1e-12)  # Received power (W)
        thermal_noise = 1e-13  # Baseline noise floor (W)
        
        # Jamming elevates effective noise floor
        jamming_linear = thermal_noise * (10 ** (jamming_noise_db / 10.0))
        total_noise = thermal_noise + jamming_linear

        snr_linear = pr / total_noise
        return 10.0 * np.log10(max(snr_linear, 1e-10))

    def compute_p_detection(self, snr_db: float) -> float:
        """Compute detection probability using Swerling 1 model approximation."""
        snr_linear = 10 ** (snr_db / 10.0)
        p_fa = 1e-6  # False alarm probability
        threshold = -np.log(p_fa)
        
        pd = np.exp(-threshold / (1.0 + snr_linear))
        return float(np.clip(pd, 0.0, 1.0))


class EmitterNetwork:
    """Manages the network of IADS radar emitters."""

    def __init__(self, config_list: list = None):
        self.emitters = []
        if config_list:
            for cfg in config_list:
                self.emitters.append(Emitter(
                    emitter_id=cfg["id"],
                    emitter_type=cfg["type"],
                    position=cfg["position"],
                    freq_ghz=cfg.get("freq_ghz", 3.0),
                    prf_hz=cfg.get("prf_hz", 1000),
                    max_range_km=cfg.get("max_range_km", 200.0)
                ))
        else:
            # Default single search & tracking setup
            self.emitters = [
                Emitter("s1", "search", [150000.0, 30000.0], 3.0, 1000, 250.0),
                Emitter("t1", "tracking", [200000.0, 25000.0], 8.0, 3000, 120.0),
            ]

    def update(self, dt: float):
        for e in self.emitters:
            e.update_scan(dt)

    def evaluate_threats(self, target_pos: np.ndarray, jamming_map: dict = None) -> list:
        """Evaluate detection state and SNR for all active emitters."""
        if jamming_map is None:
            jamming_map = {}

        results = []
        for e in self.emitters:
            jam_db = jamming_map.get(e.id, 0.0)
            snr = e.compute_snr(target_pos, jamming_noise_db=jam_db)
            pd = e.compute_p_detection(snr)
            results.append({
                "id": e.id,
                "type": e.type,
                "position": e.position,
                "snr_db": snr,
                "p_detection": pd,
                "freq_ghz": e.freq_ghz,
                "prf_hz": e.prf_hz
            })
        return results
