"""
Attention Visualizer (explain/attention_visualizer.py)
Extracts model attention weights and highlights key radar threat drivers.
"""

import numpy as np


class AttentionVisualizer:
    """
    Extracts attention weights across spatial threat locations and emitter feeds.
    """

    def __init__(self):
        pass

    def compute_emitter_attention(self, situation_vector: np.ndarray, num_emitters: int = 4) -> np.ndarray:
        """
        Calculates relative attention weights assigned to each radar emitter.
        """
        threat_features = situation_vector[8:8 + num_emitters * 8]
        weights = []
        for i in range(num_emitters):
            p_det = threat_features[i * 8] if i * 8 < len(threat_features) else 0.0
            weights.append(p_det)

        weights = np.array(weights, dtype=np.float32)
        total = np.sum(weights) + 1e-8
        return weights / total
