"""
src/fundus2oct/models/classifier_multimodal.py

Model B: Multimodal 3D Retinal Disease Classifier.
Fuses complementary 2D optical en-face representations (from color retinal fundus
photographs via EyeFound foundation ViT backbone) with depth-resolved cross-sectional
structural representations (from 3D synthetic OCT volumes via a 3D CNN encoder).

Supports:
1. End-to-end multimodal classification from 2D fundus images and 3D synthetic OCT volumes.
2. Robust support for 2D OCT B-scan slices via automatic volumetric depth extension.
3. Direct classification from precomputed EyeFound foundation embeddings and/or OCT embeddings.
4. Gated multimodal fusion with learnable cross-modal balance gating.
5. Penultimate multimodal representation feature extraction for comparative AUROC uplift & t-SNE.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fundus2oct.models.fundus_encoder import (
    EyeFoundViTBackbone,
    FundusEncoder,
    build_fundus_encoder,
)
from fundus2oct.models.classifier_fundus import ConvBlock


# ---------------------------------------------------------------------------
# 1. 3D Convolutional Residual Block for Volumetric OCT
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """
    3D Residual Convolutional Block for OCT Volume feature extraction.
    Applies two 3D convolutional layers with GroupNorm and SiLU activation,
    paired with a 3D residual shortcut connection.
    Supports both full 3D volumetric scans (B, C, D, H, W) and single-slice
    cross sections (B, C, 1, H, W).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
    ):
        super().__init__()
        stride_tuple = stride if isinstance(stride, tuple) else (1, stride, stride)
        num_groups = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else 1)

        self.conv1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=3, stride=stride_tuple, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride_tuple, bias=False),
                nn.GroupNorm(num_groups, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + res)


# ---------------------------------------------------------------------------
# 2. 3D CNN OCT Volume Feature Extractor
# ---------------------------------------------------------------------------

class OCT3DEncoder(nn.Module):
    """
    3D Convolutional Feature Extractor for Synthetic OCT Volumes.
    Extracts cross-sectional structural representations (retinal layers, subretinal fluid,
    drusen, geographic atrophy) across depth, height, and width.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        feature_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Initial volumetric stem: 7x7 spatial with 3-slice depth receptive field
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.GroupNorm(8 if base_channels % 8 == 0 else 1, base_channels),
            nn.SiLU(),
        )

        # 4-stage residual 3D downsampling hierarchy
        self.stage1 = ConvBlock3D(base_channels, base_channels * 2, stride=2)
        self.stage2 = ConvBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.stage3 = ConvBlock3D(base_channels * 4, base_channels * 8, stride=2)
        self.stage4 = ConvBlock3D(base_channels * 8, feature_dim, stride=2)

        # Global 3D Volumetric Adaptive Pooling to (1, 1, 1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Non-linear projection to normalized feature space
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 5D tensor (B, C, D, H, W)
        Returns:
            features: (B, feature_dim)
        """
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        pooled = self.pool(h).flatten(1)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# 3. Gated Multimodal Cross-Modal Fusion Unit
# ---------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """
    Gated Multimodal Fusion Unit.
    Combines EyeFound fundus features (2D optical en-face) and OCT features (3D cross-sectional)
    using a learnable modal balance gating mechanism that dynamically weighs the complementary
    modalities according to diagnostic relevance.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim

        # Dynamic cross-modal gating weight in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

        # Joint non-linear cross-modal interaction projection
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, f_fundus: torch.Tensor, f_oct: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([f_fundus, f_oct], dim=-1)
        z = self.gate(concat)  # Gating weight in [0, 1]
        fused = z * f_fundus + (1.0 - z) * f_oct
        return self.proj(concat) + fused


# ---------------------------------------------------------------------------
# 4. Main Multimodal 3D Disease Classifier (Model B)
# ---------------------------------------------------------------------------

class MultimodalClassifier(nn.Module):
    """
    Model B: Multimodal 3D Disease Classifier (2D EyeFound Fundus + 3D Synthetic OCT).
    Fuses complementary optical en-face representations (from 2D retinal fundus photographs
    via EyeFound foundation ViT backbone) with depth-resolved cross-sectional structural
    representations (from 3D synthetic OCT volumes via a 3D CNN encoder).

    Inputs:
        fundus: (B, 3, 256, 256) image tensor OR (B, embed_dim) precomputed EyeFound embedding
        oct:    (B, 1, D, H, W) 3D volume OR (B, 1, 256, 256) 2D B-scan slice
    Output:
        logits: (B, num_classes) unnormalized disease classification logits
    """

    def __init__(
        self,
        fundus_channels: int = 3,
        oct_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        feature_dim: int = 256,
        embed_dim: int = 512,
        dropout: float = 0.3,
        backbone: str = "eyefound",
        freeze_backbone: bool = False,
        encoder_checkpoint: Optional[Union[str, Path]] = "checkpoints/fundus_encoder.pt",
    ):
        super().__init__()
        self.fundus_channels = fundus_channels
        self.oct_channels = oct_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.backbone_type = backbone.lower()
        self.freeze_backbone = freeze_backbone
        self.is_trained = False

        # --- A. 2D EyeFound Fundus Stream ---
        if self.backbone_type in ("eyefound", "vit", "retfound"):
            self.fundus_encoder = FundusEncoder(
                backbone="eyefound",
                embed_dim=embed_dim,
                spatial_features=False,
                img_size=256,
                in_channels=fundus_channels,
                checkpoint_path=encoder_checkpoint,
            )
            if self.freeze_backbone:
                for param in self.fundus_encoder.parameters():
                    param.requires_grad = False
                self.fundus_encoder.eval()

            # EyeFound Embedding Projection (embed_dim -> feature_dim)
            self.fundus_proj = nn.Sequential(
                nn.Linear(embed_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
            self.fundus_cnn = None

        else:
            # Fallback 2D CNN stream for backward compatibility
            self.fundus_encoder = None
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
            self.fundus_proj = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )

        # --- B. 3D CNN Synthetic OCT Stream ---
        self.oct_3d_encoder = OCT3DEncoder(
            in_channels=oct_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        # Retain backward-compatible references for older checkpoints
        self.oct_init = self.oct_3d_encoder.stem
        self.oct_l1 = self.oct_3d_encoder.stage1
        self.oct_l2 = self.oct_3d_encoder.stage2
        self.oct_l3 = self.oct_3d_encoder.stage3
        self.oct_l4 = self.oct_3d_encoder.stage4
        self.oct_pool = self.oct_3d_encoder.pool

        # --- C. Cross-Modal Fusion ---
        self.fusion = MultimodalFusion(feature_dim=feature_dim, dropout=dropout)

        # --- D. Final Classification Head ---
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_fundus_features(self, fundus: torch.Tensor) -> torch.Tensor:
        """
        Extracts EyeFound fundus features (B, feature_dim).
        Handles:
        - 4D Image tensor (B, 3, H, W) via EyeFound ViT backbone
        - 2D Precomputed Embedding vector (B, embed_dim) or (B, feature_dim)
        """
        if fundus.ndim == 2:
            if fundus.shape[1] == self.feature_dim:
                return fundus
            if fundus.shape[1] == self.embed_dim:
                return self.fundus_proj(fundus)
            # Generic linear projection fallback for other dims
            proj_layer = nn.Linear(fundus.shape[1], self.feature_dim).to(fundus.device)
            return proj_layer(fundus)

        if fundus.ndim != 4:
            raise ValueError(f"Expected fundus tensor (B, 3, H, W) or (B, D), got {fundus.shape}")

        if self.fundus_encoder is not None:
            emb = self.fundus_encoder(fundus)
            if isinstance(emb, tuple):
                emb = emb[0]
            return self.fundus_proj(emb)

        # Fallback CNN path
        h = self.fundus_init(fundus)
        h = self.fundus_l1(h)
        h = self.fundus_l2(h)
        h = self.fundus_l3(h)
        h = self.fundus_l4(h)
        feat = self.fundus_pool(h).flatten(1)
        return self.fundus_proj(feat)

    def extract_oct_features(self, oct_data: torch.Tensor) -> torch.Tensor:
        """
        Extracts 3D CNN OCT features (B, feature_dim).
        Handles:
        - 5D Volume tensor (B, C, D, H, W)
        - 4D B-scan slice tensor (B, C, H, W) -> auto-expanded to (B, C, 1, H, W)
        - 3D Volume tensor (B, D, H, W) -> auto-expanded to (B, 1, D, H, W)
        - 2D Precomputed Embedding vector (B, feature_dim)
        """
        if oct_data.ndim == 2:
            if oct_data.shape[1] == self.feature_dim:
                return oct_data
            proj = nn.Linear(oct_data.shape[1], self.feature_dim).to(oct_data.device)
            return proj(oct_data)

        # Normalize input to 5D (B, C, D, H, W)
        if oct_data.ndim == 4:
            # 2D B-scan slice (B, C, H, W) -> unsqueeze depth to (B, C, 1, H, W)
            oct_vol = oct_data.unsqueeze(2)
        elif oct_data.ndim == 3:
            # (B, H, W) -> (B, 1, 1, H, W)
            oct_vol = oct_data.unsqueeze(1).unsqueeze(2)
        elif oct_data.ndim == 5:
            # Full 3D Volume (B, C, D, H, W)
            oct_vol = oct_data
        else:
            raise ValueError(
                f"Expected 3D OCT volume (B, C, D, H, W) or 2D B-scan (B, C, H, W), got {oct_data.shape}"
            )

        return self.oct_3d_encoder(oct_vol)

    def extract_multimodal_features(
        self,
        fundus: torch.Tensor,
        oct_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts individual fundus, OCT, and fused cross-modal features.
        Returns:
            f_fundus: (B, feature_dim) EyeFound fundus feature
            f_oct:    (B, feature_dim) 3D CNN OCT feature
            f_fused:  (B, feature_dim) Gated fused multimodal representation
        """
        f_fundus = self.extract_fundus_features(fundus)
        f_oct = self.extract_oct_features(oct_data)
        f_fused = self.fusion(f_fundus, f_oct)
        return f_fundus, f_oct, f_fused

    def forward(self, fundus: torch.Tensor, oct_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass fusing both modalities into disease classification logits.
        Returns:
            logits: (B, num_classes)
        """
        _, _, f_fused = self.extract_multimodal_features(fundus, oct_data)
        f_fused = self.dropout(f_fused)
        logits = self.head(f_fused)
        return logits

    def predict_proba(self, fundus: torch.Tensor, oct_data: torch.Tensor) -> torch.Tensor:
        """Returns normalized class probabilities via softmax (B, num_classes)."""
        logits = self.forward(fundus, oct_data)
        return F.softmax(logits, dim=-1)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict: bool = False,
    ):
        """Loads trained model weights from disk."""
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint not found or empty at: {p}")
        checkpoint = torch.load(p, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Support backward compatibility if state_dict has older prefix variations
        adjusted_sd = {}
        for k, v in state_dict.items():
            # If checkpoint has older 2D oct_init keys, map them cleanly
            if k.startswith("oct_init.") and not hasattr(self, "oct_init"):
                adjusted_sd[k.replace("oct_init.", "oct_3d_encoder.stem.")] = v
            else:
                adjusted_sd[k] = v

        try:
            self.load_state_dict(adjusted_sd, strict=strict)
        except Exception as e:
            # Non-strict fallback for partial weight transfer
            print(f"[MultimodalClassifier] Non-strict checkpoint load: {e}")
            self.load_state_dict(adjusted_sd, strict=False)

        self.to(device)
        self.is_trained = True
        print(f"[MultimodalClassifier] Successfully loaded checkpoint from {p}")

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int = 0,
        val_loss: float = 0.0,
        val_acc: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Saves model weights along with training and architecture metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "state_dict": self.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "metrics": metrics or {},
            "model_type": "multimodal",
            "backbone": self.backbone_type,
            "embed_dim": self.embed_dim,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
        }
        torch.save(payload, p)
        print(f"[MultimodalClassifier] Checkpoint saved to {p}")


# Architectural and Domain Aliases
Multimodal3DClassifier = MultimodalClassifier
EyeFoundOCT3DClassifier = MultimodalClassifier


def build_multimodal_classifier(
    num_classes: int = 2,
    feature_dim: int = 256,
    embed_dim: int = 512,
    base_channels: int = 32,
    dropout: float = 0.3,
    backbone: str = "eyefound",
    checkpoint_path: Optional[Union[str, Path]] = "checkpoints/classifier_multimodal.pt",
    encoder_checkpoint: Optional[Union[str, Path]] = "checkpoints/fundus_encoder.pt",
    device: Union[str, torch.device] = "cpu",
) -> MultimodalClassifier:
    """Convenience factory function for instantiating the EyeFound + 3D OCT multimodal classifier."""
    model = MultimodalClassifier(
        fundus_channels=3,
        oct_channels=1,
        num_classes=num_classes,
        base_channels=base_channels,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        dropout=dropout,
        backbone=backbone,
        encoder_checkpoint=encoder_checkpoint,
    )
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists() and p.stat().st_size > 0:
            try:
                model.load_checkpoint(p, device=device)
            except Exception as e:
                print(f"[build_multimodal_classifier] Notice: Could not load checkpoint {p}: {e}")
    model.to(device)
    return model
