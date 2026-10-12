"""
Situation Vector Builder (perception/situation_builder.py)
Synthesizes classified emitters, vehicle state, and spatial threat zones into a unified situation vector.
"""

import numpy as np


class SituationVectorBuilder:
    """
    Assembles multi-modal perception feeds into a compact feature vector for RL decision networks.
    """

    def __init__(self, vector_dim: int = 64):
        self.vector_dim = vector_dim

    def build_situation_vector(
        self,
        vehicle_kinematics: np.ndarray,
        classified_emitters: list,
        target_relative_pos: np.ndarray
    ) -> np.ndarray:
        """
        Synthesize state vector.
        """
        vec = np.zeros(self.vector_dim, dtype=np.float32)

        # 1. Kinematics (6)
        vec[0:len(vehicle_kinematics)] = vehicle_kinematics

        # 2. Target vector (2)
        vec[6:8] = target_relative_pos

        # 3. Emitter threats summary (up to 4 emitters * 8 features = 32)
        idx = 8
        for emitter in classified_emitters[:4]:
            vec[idx] = emitter.get("p_detection", 0.0)
            vec[idx + 1] = emitter.get("snr_db", -100.0) / 50.0
            vec[idx + 2] = emitter.get("freq_ghz", 0.0) / 15.0
            vec[idx + 3] = emitter.get("prf_hz", 0.0) / 5000.0
            idx += 8

        return vec
