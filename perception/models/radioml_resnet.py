from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = x + residual

        x = self.relu(x)

        return x


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation attention.

    Learns which feature channels are
    important for RF classification.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ):

        super().__init__()

        hidden = max(
            channels // reduction,
            8,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                hidden,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden,
                channels,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):

        batch, channels, _ = x.shape

        weights = self.pool(x).view(
            batch,
            channels,
        )

        weights = self.fc(weights).view(
            batch,
            channels,
            1,
        )

        return x * weights


class MultiScaleBlock(nn.Module):
    """
    Extracts RF patterns at different
    temporal scales.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        branch_channels = out_channels // 3

        self.branch_3 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_5 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_7 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv1d(
                branch_channels * 3,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x3 = self.branch_3(x)

        x5 = self.branch_5(x)

        x7 = self.branch_7(x)

        x = torch.cat(
            [x3, x5, x7],
            dim=1,
        )

        return self.project(x)


class RadioMLResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 24,
    ):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                2,
                64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.multi_scale = MultiScaleBlock(
            64,
            128,
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.residual1 = ResidualBlock1D(128)

        self.residual2 = ResidualBlock1D(128)

        self.attention = SEBlock1D(128)

        self.downsample = nn.Sequential(
            nn.Conv1d(
                128,
                256,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.residual3 = ResidualBlock1D(256)

        self.residual4 = ResidualBlock1D(256)

        self.final_attention = SEBlock1D(256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                256,
                128,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.multi_scale(x)

        x = self.pool1(x)

        x = self.residual1(x)

        x = self.residual2(x)

        x = self.attention(x)

        x = self.downsample(x)

        x = self.residual3(x)

        x = self.residual4(x)

        x = self.final_attention(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x
