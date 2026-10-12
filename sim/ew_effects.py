"""
Electronic Warfare Effects Model (sim/ew_effects.py)
Simulates active noise jamming, deceptive spoofing, and track degradation impacts on enemy radar.
"""

import numpy as np


class EWEffectsModel:
    """
    Computes the electromagnetic countermeasure effects on radar emitters:
    1. Noise Jamming: Elevates radar noise floor (J/N ratio).
    2. Range Gate Pull-Off (RGPO): Induces range delay error.
    3. Angle Deception: Offsets Direction-of-Arrival estimation.
    """

    def __init__(self, max_power_dbw: float = 30.0):
        self.max_power_dbw = max_power_dbw

    def compute_countermeasures(
        self,
        ew_action: np.ndarray,
        target_pos: np.ndarray,
        emitters: list
    ) -> dict:
        """
        ew_action vector: [mode_idx, power_level (0-1), freq_offset, bandwidth]
        mode_idx: 0=passive, 1=noise_jamming, 2=rgpo_spoofing, 3=angle_deception
        """
        mode_idx = int(ew_action[0])
        power_ratio = np.clip(ew_action[1], 0.0, 1.0)
        power_dbw = power_ratio * self.max_power_dbw

        jamming_map = {}
        spoofing_offsets = {}

        for e in emitters:
            dist = np.linalg.norm(target_pos - e["position"])
            
            if mode_idx == 1:  # Active Noise Jamming
                # 1/R^2 path loss for one-way jamming transmission
                path_loss_db = 20 * np.log10(dist + 1.0)
                effective_j_n = power_dbw - path_loss_db + 80.0
                jamming_map[e["id"]] = max(0.0, effective_j_n)
                spoofing_offsets[e["id"]] = np.array([0.0, 0.0])

            elif mode_idx == 2:  # Deceptive Spoofing (RGPO)
                # Induces track error offset in radar tracking filters
                jamming_map[e["id"]] = 5.0
                offset_dist = 500.0 * power_ratio  # meters range error
                direction = (target_pos - e["position"]) / max(dist, 1.0)
                spoofing_offsets[e["id"]] = direction * offset_dist

            elif mode_idx == 3:  # Angle Deception
                jamming_map[e["id"]] = 3.0
                # Cross-eye / angle offset
                angle_offset = np.radians(5.0 * power_ratio)
                spoofing_offsets[e["id"]] = np.array([np.cos(angle_offset), np.sin(angle_offset)]) * 200.0

            else:  # Passive / None
                jamming_map[e["id"]] = 0.0
                spoofing_offsets[e["id"]] = np.array([0.0, 0.0])

        return {
            "jamming_noise_db": jamming_map,
            "spoofing_offsets": spoofing_offsets,
            "power_consumed_kw": (10 ** (power_dbw / 10.0)) / 1000.0
        }
