from __future__ import annotations

import torch
import torch.nn as nn


class RadioMLCNN(nn.Module):
    """
    Lightweight 1D CNN baseline for RadioML IQ signals.

    Input:
        (batch, 2, 1024)

    Output:
        (batch, 24)
    """

    def __init__(
        self,
        num_classes: int = 24,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(
                in_channels=2,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 4
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                256,
                128,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.classifier(x)

        return x
