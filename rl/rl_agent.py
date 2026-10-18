"""
COG-EW: Reinforcement Learning Agent Wrapper

Provides policy inference and checkpoint management.

Safety and reliability:
- Requires PyTorch; never substitutes random actions silently.
- Validates state dimensions and numerical values.
- Validates policy outputs.
- Validates checkpoint keys and tensor shapes before loading.
- Loads tensor checkpoints with weights_only=True.
- Saves checkpoints atomically to reduce the risk of partial files.

Note:
This class does not implement PPO optimization. The production
training loop must be implemented and tested separately.
"""

import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    try:
        from rl.policy_network import MultiHeadPolicyNet
    except ModuleNotFoundError:
        from policy_network import MultiHeadPolicyNet

    HAS_TORCH = True
else:
    MultiHeadPolicyNet = None
    HAS_TORCH = False


class PPOAgent:
    """Policy wrapper for inference and checkpoint management."""

    def __init__(
        self,
        state_dim: int = 16,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required to initialize PPOAgent. "
                "Install a compatible PyTorch build before continuing. "
                "Random-action fallback is disabled."
            )

        if not isinstance(state_dim, int) or isinstance(state_dim, bool):
            raise TypeError("state_dim must be an integer.")

        if state_dim < 1:
            raise ValueError("state_dim must be at least 1.")

        self.state_dim = state_dim
        self.device_str = str(device)

        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device: {device!r}") from exc

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable. "
                "Use device='cpu' or check the PyTorch installation."
            )

        self.policy = MultiHeadPolicyNet(state_dim=self.state_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=3.0e-4,
        )

    def _validate_state(self, state) -> np.ndarray:
        """Convert and validate one observation."""
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()

        try:
            state_array = np.asarray(state, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("State must be convertible to a float32 array.") from exc

        if state_array.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.state_dim},), "
                f"received {state_array.shape}."
            )

        if not np.isfinite(state_array).all():
            raise ValueError("State contains NaN or infinite values.")

        return state_array

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ):
        """
        Select an action using the loaded or initialized policy.

        Returns:
            action: NumPy array with four entries.
            log_prob: NumPy array containing the action log probability.
            value: NumPy array containing the estimated state value.

        This method does not train or update the policy.
        """
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")

        state_array = self._validate_state(state)

        state_tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.policy.eval()

        with torch.no_grad():
            action, log_prob, value = self.policy.sample_action(
                state_tensor,
                deterministic=deterministic,
            )

        if action is None or log_prob is None or value is None:
            raise RuntimeError("Policy inference returned missing outputs.")

        if action.shape != (1, 4):
            raise RuntimeError(
                f"Expected policy action shape (1, 4), "
                f"received {tuple(action.shape)}."
            )

        if not torch.isfinite(action).all().item():
            raise RuntimeError("Policy returned NaN or infinite action values.")

        if not torch.isfinite(log_prob).all().item():
            raise RuntimeError("Policy returned a non-finite log probability.")

        if not torch.isfinite(value).all().item():
            raise RuntimeError("Policy returned a non-finite value estimate.")

        return (
            action.detach().cpu().numpy()[0],
            log_prob.detach().cpu().numpy()[0],
            value.detach().cpu().numpy()[0],
        )

    def save_checkpoint(self, filepath: str):
        """
        Save policy weights atomically.

        The saved format remains a raw state dictionary for
        compatibility with the existing evaluation entry point.
        """
        if self.policy is None:
            raise RuntimeError("Cannot save a checkpoint: policy is not initialized.")

        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            key: tensor.detach().cpu()
            for key, tensor in self.policy.state_dict().items()
        }

        temporary_path = None

        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)

            torch.save(state_dict, temporary_path)
            os.replace(temporary_path, path)

        except Exception as exc:
            raise RuntimeError(f"Failed to save checkpoint to {path}: {exc}") from exc

        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _extract_state_dict(checkpoint):
        """Accept a raw state dict or a recognized wrapper format."""
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint must contain a state dictionary.")

        # Current project checkpoints are raw state dictionaries.
        # Also accept common wrapped formats for compatibility.
        for key in ("policy_state_dict", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break

        if not checkpoint:
            raise ValueError("Checkpoint state dictionary is empty.")

        if not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in checkpoint.items()
        ):
            raise ValueError(
                "Checkpoint must contain only string keys and tensor values."
            )

        return checkpoint

    def load_checkpoint(self, filepath: str):
        """
        Load compatible policy weights.

        Missing, malformed, or incompatible checkpoints raise errors.
        No randomly initialized policy is reported as loaded.
        """
        if self.policy is None:
            raise RuntimeError("Cannot load a checkpoint: policy is not initialized.")

        path = Path(filepath).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "This PyTorch version does not support the required "
                "weights_only checkpoint loading option. Upgrade PyTorch "
                "before loading checkpoints."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not safely load checkpoint {path}: {exc}"
            ) from exc

        state_dict = self._extract_state_dict(checkpoint)
        expected_state = self.policy.state_dict()

        expected_keys = set(expected_state.keys())
        received_keys = set(state_dict.keys())

        missing_keys = sorted(expected_keys - received_keys)
        unexpected_keys = sorted(received_keys - expected_keys)

        if missing_keys or unexpected_keys:
            raise ValueError(
                "Checkpoint keys do not match the policy architecture. "
                f"Missing keys: {missing_keys}; "
                f"unexpected keys: {unexpected_keys}."
            )

        shape_errors = []

        for key, expected_tensor in expected_state.items():
            received_tensor = state_dict[key]

            if received_tensor.shape != expected_tensor.shape:
                shape_errors.append(
                    f"{key}: expected {tuple(expected_tensor.shape)}, "
                    f"received {tuple(received_tensor.shape)}"
                )

        if shape_errors:
            raise ValueError(
                "Checkpoint tensor shapes do not match the policy: "
                + "; ".join(shape_errors)
            )

        # Apply weights only after validating keys and shapes.
        try:
            self.policy.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint validation succeeded, but loading failed: {exc}"
            ) from exc

        self.policy.eval()

        print(f"[PASS] Policy checkpoint loaded successfully: {path}")
