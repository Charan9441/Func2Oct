"""
src/fundus2oct/models/classifier_multimodal.py

Model B: Fundus + Synthetic OCT Multimodal Disease Classifier.
Fuses complementary optical en-face representations (from 2D Fundus photographs)
with depth-resolved cross-sectional structural representations (from Synthetic OCT B-scans).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundus2oct.models.classifier_fundus import ConvBlock


class MultimodalFusion(nn.Module):
    """
    Gated Multimodal Fusion Unit.
    Combines Fundus features and OCT features using a learnable gating mechanism
    that dynamically weighs the complementary modalities.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        # Gating gate
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        # Joint non-linear projection
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, f_fundus: torch.Tensor, f_oct: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([f_fundus, f_oct], dim=-1)
        z = self.gate(concat)  # Gating weight in [0, 1]
        fused = z * f_fundus + (1.0 - z) * f_oct
        return self.proj(concat) + fused


class MultimodalClassifier(nn.Module):
    """
    Model B: Multimodal Disease Classifier (Fundus + Synthetic OCT).
    Inputs:
        fundus: (B, 3, 256, 256)
        oct:    (B, 1, 256, 256)
    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        fundus_channels: int = 3,
        oct_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.is_trained = False

        # Fundus Stream Feature Extractor
        self.fundus_init = nn.Sequential(
            nn.Conv2d(fundus_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.fundus_l1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.fundus_l2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.fundus_l3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.fundus_l4 = ConvBlock(base_channels * 8, feature_dim, stride=2)
        self.fundus_pool = nn.AdaptiveAvgPool2d((1, 1))

        # OCT Stream Feature Extractor (single-channel cross-sectional)
        self.oct_init = nn.Sequential(
            nn.Conv2d(oct_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.oct_l1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.oct_l2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.oct_l3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.oct_l4 = ConvBlock(base_channels * 8, feature_dim, stride=2)
        self.oct_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Cross-Modal Fusion
        self.fusion = MultimodalFusion(feature_dim=feature_dim, dropout=dropout)

        # Classification Head
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def extract_fundus_features(self, fundus: torch.Tensor) -> torch.Tensor:
        h = self.fundus_init(fundus)
        h = self.fundus_l1(h)
        h = self.fundus_l2(h)
        h = self.fundus_l3(h)
        h = self.fundus_l4(h)
        return self.fundus_pool(h).flatten(1)

    def extract_oct_features(self, oct_img: torch.Tensor) -> torch.Tensor:
        h = self.oct_init(oct_img)
        h = self.oct_l1(h)
        h = self.oct_l2(h)
        h = self.oct_l3(h)
        h = self.oct_l4(h)
        return self.oct_pool(h).flatten(1)

    def extract_multimodal_features(
        self,
        fundus: torch.Tensor,
        oct_img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_fundus = self.extract_fundus_features(fundus)
        f_oct = self.extract_oct_features(oct_img)
        f_fused = self.fusion(f_fundus, f_oct)
        return f_fundus, f_oct, f_fused

    def forward(self, fundus: torch.Tensor, oct_img: torch.Tensor) -> torch.Tensor:
        """Forward pass fusing both modalities into classification logits."""
        _, _, f_fused = self.extract_multimodal_features(fundus, oct_img)
        f_fused = self.dropout(f_fused)
        logits = self.head(f_fused)
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
