from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from perception.models.radioml_resnet import (
    RadioMLResNet,
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW RADIOML RESNET VERIFICATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = RadioMLResNet(num_classes=24).to(device)

    model.eval()

    parameters = sum(p.numel() for p in model.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel:")

    print(model)

    print(f"\nTotal parameters: " f"{parameters:,}")

    print(f"Trainable parameters: " f"{trainable:,}")

    x = torch.randn(
        32,
        2,
        1024,
        device=device,
    )

    with torch.no_grad():

        y = model(x)

    print("\nInput:")

    print(tuple(x.shape))

    print("\nOutput:")

    print(tuple(y.shape))

    predictions = torch.argmax(
        y,
        dim=1,
    )

    print("\nPrediction shape:")

    print(tuple(predictions.shape))

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / 1024**2

        print(f"\nGPU memory allocated: " f"{allocated:.2f} MB")

    assert y.shape == (
        32,
        24,
    )

    assert predictions.shape == (32,)

    print("\n[PASS] " "RadioML ResNet verification completed.")


if __name__ == "__main__":
    main()
