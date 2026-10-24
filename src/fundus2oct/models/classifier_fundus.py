"""
src/fundus2oct/models/classifier_fundus.py

Model A: Fundus-only Disease Classifier.
Extracts semantic features from 2D color retinal fundus photographs and outputs
class probability logits for pathology detection (e.g., Macular Degeneration vs Other).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard residual convolutional block with GroupNorm and SiLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        num_groups = 8 if out_ch % 8 == 0 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + res)


class FundusClassifier(nn.Module):
    """
    Model A: Retinal Fundus image disease classifier.
    Input: (B, in_channels=3, 256, 256)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.is_trained = False

        # Feature Backbone: Downsamples 256x256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),  # 128x128
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        self.layer1 = ConvBlock(base_channels, base_channels * 2, stride=2)   # 64x64
        self.layer2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)  # 32x32
        self.layer3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)  # 16x16
        self.layer4 = ConvBlock(base_channels * 8, feature_dim, stride=2)        # 8x8

        # Pooling & Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate pooled feature vector (B, feature_dim)."""
        if x.ndim != 4:
            raise ValueError(f"FundusClassifier expects 4D tensor (B, C, H, W), got {x.shape}")
        h = self.init_conv(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        feat = self.pool(h).flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits (B, num_classes)."""
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        logits = self.head(feat)
        return logits

    def load_checkpoint(self, path: Union[str, Path], device: torch.device = torch.device("cpu")):
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint not found or empty at: {p}")
        checkpoint = torch.load(p, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict)
        self.to(device)
        self.is_trained = True
