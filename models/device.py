from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_device(requested_device: str | None = None) -> torch.device:
    """
    Get the PyTorch device to use for training and inference.

    If requested_device is provided (e.g. 'cuda', 'cpu', 'cuda:0'), it will attempt to use it.
    Otherwise, it defaults to CUDA if available, falling back to CPU.
    """
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Requested CUDA device '%s', but CUDA is not available. Falling back to CPU.",
                requested_device,
            )
            return torch.device("cpu")
        return device

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using GPU for training: %s", device_name)
        return torch.device("cuda")

    logger.info("CUDA not available. Using CPU for training.")
    return torch.device("cpu")


def to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors, tuples, lists, or dicts of tensors to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: to_device(val, device) for key, val in data.items()}
    return data
