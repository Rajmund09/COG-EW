from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from perception.models.radioml_cnn import (
    RadioMLCNN,
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("COG-EW RADIOML CNN VERIFICATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

    model = RadioMLCNN(num_classes=24)

    model = model.to(device)

    print("\nModel:")

    print(model)

    # ------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nParameters:")

    print(f"Total: " f"{total_params:,}")

    print(f"Trainable: " f"{trainable_params:,}")

    # ------------------------------------------------------
    # Dummy input
    # ------------------------------------------------------

    x = torch.randn(
        32,
        2,
        1024,
        device=device,
    )

    print("\nInput shape:")

    print(tuple(x.shape))

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(x)

    print("\nOutput shape:")

    print(tuple(output.shape))

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predictions = torch.argmax(
        output,
        dim=1,
    )

    print("\nPrediction shape:")

    print(tuple(predictions.shape))

    print("\nPredicted classes:")

    print(predictions.tolist())

    # ------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / 1024**2

        reserved = torch.cuda.memory_reserved() / 1024**2

        print("\nGPU memory:")

        print(f"Allocated: " f"{allocated:.2f} MB")

        print(f"Reserved: " f"{reserved:.2f} MB")

    print("\n" + "=" * 70)

    print("[PASS] RadioML CNN verification " "completed.")

    print("=" * 70)


if __name__ == "__main__":
    main()
