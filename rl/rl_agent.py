"""
RL Agent Wrapper (rl/rl_agent.py)
Agent wrapper handling neural network policy interactions, experience storage, and action selection.
"""

import numpy as np

try:
    import torch
    from rl.policy_network import MultiHeadPolicyNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PPOAgent:
    """
    Proximal Policy Optimization Agent for Joint EW and Guidance.
    """

    def __init__(self, state_dim: int = 16, device: str = "cpu"):
        self.state_dim = state_dim
        self.device_str = device

        if HAS_TORCH:
            self.device = torch.device(device)
            self.policy = MultiHeadPolicyNet(state_dim=state_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3.0e-4)
        else:
            self.policy = None

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        if HAS_TORCH and self.policy is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.policy.sample_action(state_tensor, deterministic=deterministic)
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
        else:
            # Fallback heuristic / random sampling when PyTorch is not installed
            guidance_act = np.random.uniform(-0.1, 0.1, size=(2,)).astype(np.float32)
            ew_mode = float(np.random.choice([0, 1, 2, 3]))
            ew_power = float(np.random.uniform(0.2, 0.8))
            action = np.array([guidance_act[0], guidance_act[1], ew_mode, ew_power], dtype=np.float32)
            return action, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    def save_checkpoint(self, filepath: str):
        if HAS_TORCH and self.policy is not None:
            torch.save(self.policy.state_dict(), filepath)
        else:
            print(f"[!] Warning: PyTorch not installed. Skipping checkpoint save to {filepath}")

    def load_checkpoint(self, filepath: str):
        if HAS_TORCH and self.policy is not None:
            self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        else:
            print(f"[!] Warning: PyTorch not installed. Skipping checkpoint load from {filepath}")
