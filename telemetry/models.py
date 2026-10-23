"""
======================================================================
COG-EW MODERN & LATEST FLIGHT TELEMETRY NEURAL ARCHITECTURES
(telemetry/models.py)
======================================================================

Implements modern deep learning architectures for aerospace flight trajectory,
attitude (roll/pitch/yaw), and dynamic G-load temporal forecasting using the
Instituto IDF Flight Telemetry Database.

Architectures:
  1. FlightTransformer:
     State-of-the-art Transformer encoder with Multi-Head Self-Attention,
     sinusoidal positional encoding, and feedforward projection head.
  2. TCNResNet:
     Temporal Convolutional Network with dilated 1D causal convolutions,
     residual connections, and exponential receptive field for high-G dynamics.
  3. BiLSTMAttention:
     Bidirectional LSTM coupled with a temporal Bahdanau-style additive attention
     mechanism for sequence-level feature aggregation.
  4. KinematicMLP:
     Deep Multi-Layer Perceptron with kinematic residual skip connection
     serving as a fast inference baseline.
======================================================================
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. POSITIONAL ENCODING
# ======================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal flight telemetry sequences."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


# ======================================================================
# 2. FLIGHT TRANSFORMER (MODERN SOTA)
# ======================================================================

class FlightTransformer(nn.Module):
    """
    Modern Transformer model for multi-horizon flight telemetry forecasting.
    Uses multi-head self-attention across historical Pixhawk telemetry steps
    to predict future positions, attitudes, and G-loads.
    """

    def __init__(
        self,
        in_features: int = 11,
        out_features: int = 11,
        input_steps: int = 20,
        pred_steps: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = "FlightTransformer"
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.in_features = in_features
        self.out_features = out_features

        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_steps + 50)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Multi-step prediction head
        self.decoder_proj = nn.Sequential(
            nn.Linear(d_model * input_steps, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_steps * out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_steps, in_features)
        b = x.size(0)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)  # (B, input_steps, d_model)
        h_flat = h.reshape(b, -1)
        out = self.decoder_proj(h_flat)
        return out.view(b, self.pred_steps, self.out_features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 3. TEMPORAL CONVOLUTIONAL NETWORK (TCN-RESNET)
# ======================================================================

class ChausalConv1dBlock(nn.Module):
    """Dilated residual 1D convolution block with weight normalization and dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.dropout1(self.relu1(out))

        out = self.conv2(out)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.dropout2(self.relu2(out))

        return self.norm(out + residual)


class TCNResNet(nn.Module):
    """
    Dilated Temporal Convolutional Network with residual connections.
    Captures fast kinematic transients and dynamic accelerations.
    """

    def __init__(
        self,
        in_features: int = 11,
        out_features: int = 11,
        input_steps: int = 20,
        pred_steps: int = 10,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = "TCNResNet"
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.in_features = in_features
        self.out_features = out_features

        if channels is None:
            channels = [64, 128, 128, 64]

        layers = []
        num_levels = len(channels)
        in_c = in_features
        for i in range(num_levels):
            dilation_size = 2**i
            out_c = channels[i]
            layers.append(ChausalConv1dBlock(in_c, out_c, kernel_size, dilation=dilation_size, dropout=dropout))
            in_c = out_c

        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1] * input_steps, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_steps * out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_steps, in_features) -> permute to (B, in_features, input_steps)
        b = x.size(0)
        x_conv = x.permute(0, 2, 1)
        feat = self.network(x_conv)  # (B, C_last, input_steps)
        feat_flat = feat.reshape(b, -1)
        out = self.fc(feat_flat)
        return out.view(b, self.pred_steps, self.out_features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 4. BILSTM ATTENTION
# ======================================================================

class TemporalAttention(nn.Module):
    """Additive Bahdanau temporal attention across LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        # rnn_outputs: (B, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (B, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (B, hidden_dim)
        return context


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Bahdanau Temporal Attention.
    Recurrent sequence model balancing past trajectory integration and attitude dynamics.
    """

    def __init__(
        self,
        in_features: int = 11,
        out_features: int = 11,
        input_steps: int = 20,
        pred_steps: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = "BiLSTMAttention"
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.in_features = in_features
        self.out_features = out_features

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_steps * out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        lstm_out, _ = self.lstm(x)  # (B, seq_len, 2*hidden_dim)
        context = self.attention(lstm_out)  # (B, 2*hidden_dim)
        out = self.fc(context)
        return out.view(b, self.pred_steps, self.out_features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 5. KINEMATIC MLP BASELINE
# ======================================================================

class KinematicMLP(nn.Module):
    """
    Fast kinematic MLP baseline with residual jump connections.
    """

    def __init__(
        self,
        in_features: int = 11,
        out_features: int = 11,
        input_steps: int = 20,
        pred_steps: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.model_name = "KinematicMLP"
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.in_features = in_features
        self.out_features = out_features

        in_dim = input_steps * in_features
        out_dim = pred_steps * out_features

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x_flat = x.reshape(b, -1)
        out = self.net(x_flat)
        return out.view(b, self.pred_steps, self.out_features)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# MODEL FACTORY
# ======================================================================

def create_model(model_name: str, in_features: int = 11, out_features: int = 11, input_steps: int = 20, pred_steps: int = 10) -> nn.Module:
    """Factory to instantiate any of the modern flight telemetry models."""
    name_lower = model_name.lower().replace("-", "").replace("_", "")
    if "transformer" in name_lower:
        return FlightTransformer(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps)
    elif "tcn" in name_lower or "resnet" in name_lower:
        return TCNResNet(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps)
    elif "lstm" in name_lower:
        return BiLSTMAttention(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps)
    elif "mlp" in name_lower or "baseline" in name_lower:
        return KinematicMLP(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: FlightTransformer, TCNResNet, BiLSTMAttention, KinematicMLP.")


def get_all_models(in_features: int = 11, out_features: int = 11, input_steps: int = 20, pred_steps: int = 10) -> dict[str, nn.Module]:
    """Return dictionary of instantiated candidate models for benchmark comparison."""
    return {
        "FlightTransformer": FlightTransformer(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps),
        "TCNResNet": TCNResNet(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps),
        "BiLSTMAttention": BiLSTMAttention(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps),
        "KinematicMLP": KinematicMLP(in_features=in_features, out_features=out_features, input_steps=input_steps, pred_steps=pred_steps),
    }
