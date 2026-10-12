"""
Input Saliency Analyzer (explain/saliency_analyzer.py)
Computes gradient-based feature saliencies to quantify input impact on action choices.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SaliencyAnalyzer:
    """
    Computes input feature saliencies using PyTorch gradient computation.
    """

    def __init__(self, policy_net=None):
        self.policy_net = policy_net

    def compute_saliency(self, situation_vector: np.ndarray) -> np.ndarray:
        if HAS_TORCH and self.policy_net is not None:
            state_tensor = torch.FloatTensor(situation_vector).unsqueeze(0)
            state_tensor.requires_grad = True

            features, value = self.policy_net(state_tensor)
            value.backward()

            saliency = state_tensor.grad.abs().squeeze(0).numpy()
            return saliency / (np.max(saliency) + 1e-8)
        else:
            # Heuristic saliency weights
            saliency = np.ones_like(situation_vector, dtype=np.float32)
            return saliency / np.sum(saliency)
