"""
Emitter Signal Classifier Neural Network (perception/emitter_classifier.py)
1D CNN / Transformer classifier for real-time radar type and operational mode classification.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Module:
        pass
    nn = type('nn', (), {'Module': Module})


class EmitterClassifierNet(nn.Module):
    """
    1D CNN architecture for classifying Pulse Descriptor Word (PDW) sequences into radar categories:
    [0: Early Warning Search, 1: Altitude Search, 2: Target Tracking, 3: Fire Control / Missile Guidance]
    """

    def __init__(self, input_dim: int = 6, num_classes: int = 4):
        if HAS_TORCH:
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if not HAS_TORCH:
            return None
        if x.dim() == 2:
            x = x.unsqueeze(2)  # Add sequence dimension

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = torch.mean(h, dim=2)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)
        return F.softmax(logits, dim=-1)
