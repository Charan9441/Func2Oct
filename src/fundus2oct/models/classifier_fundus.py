"""
src/fundus2oct/models/classifier_fundus.py

Model A: Baseline 2D Retinal Fundus Image Disease Classifier.
Attaches a Multi-Layer Perceptron (MLP) classification head to EyeFound foundation
fundus embeddings for multi-class and binary retinal pathology detection (e.g., Macular
Degeneration [MD] vs. Other / Normal).

Supports:
1. End-to-end inference from 2D fundus images (B, 3, 256, 256) via EyeFound ViT backbone.
2. Direct classification from precomputed EyeFound embedding vectors (B, embed_dim).
3. Penultimate representation feature extraction for multimodal fusion (Model B) and t-SNE.
4. Backward compatibility with convolutional architectures via ConvBlock.
"""

from __future__ import annotations

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


class ConvBlock(nn.Module):
    """
    Standard residual convolutional block with GroupNorm and SiLU.
    Retained for backward compatibility with convolutional pipelines and Model B fusion.
    """

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


class MLPClassificationHead(nn.Module):
    """
    Multi-Layer Perceptron (MLP) classification head for multi-class retinal pathology detection.
    Attaches to EyeFound foundation fundus embeddings (e.g. 512-dim).
    Projects through non-linear layers with LayerNorm and Dropout to prevent overfitting.
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # First MLP projection stage -> hidden_dim
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(p=dropout)

        # Second MLP refinement stage
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(p=dropout)

        # Final classification projection
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Projects embedding through MLP up to penultimate representation (B, hidden_dim)."""
        h = self.drop1(self.act1(self.norm1(self.fc1(x))))
        h = self.drop2(self.act2(self.norm2(self.fc2(h))))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes unnormalized classification logits (B, num_classes)."""
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


class FundusClassifier(nn.Module):
    """
    Model A: Retinal Fundus Image Disease Classifier.
    Attaches an MLP classification head to EyeFound fundus embeddings for multi-class
    retinal pathology detection.

    Inputs:
    - 2D Fundus Image tensor: (B, 3, 256, 256)
    OR
    - Precomputed EyeFound embeddings: (B, embed_dim)

    Outputs:
    - Classification Logits: (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        feature_dim: int = 256,
        dropout: float = 0.3,
        backbone: str = "eyefound",
        embed_dim: int = 512,
        freeze_backbone: bool = False,
        encoder_checkpoint: Optional[Union[str, Path]] = "checkpoints/fundus_encoder.pt",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.backbone_type = backbone.lower()
        self.freeze_backbone = freeze_backbone
        self.is_trained = False

        if self.backbone_type in ("eyefound", "vit", "retfound"):
            # EyeFound Foundation Model Backbone
            self.encoder = FundusEncoder(
                backbone="eyefound",
                embed_dim=embed_dim,
                spatial_features=False,
                img_size=256,
                in_channels=in_channels,
                checkpoint_path=encoder_checkpoint,
            )
            if self.freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()

            # MLP Head attached to EyeFound embeddings
            self.head = MLPClassificationHead(
                in_features=embed_dim,
                hidden_dim=feature_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
            self.cnn_backbone = None

        else:
            # Fallback CNN backbone
            self.encoder = None
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
            )
            self.layer1 = ConvBlock(base_channels, base_channels * 2, stride=2)
            self.layer2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
            self.layer3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
            self.layer4 = ConvBlock(base_channels * 8, feature_dim, stride=2)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts penultimate feature representation (B, feature_dim).
        Handles both 4D image inputs (B, C, H, W) and 2D embedding inputs (B, embed_dim).
        """
        if x.ndim == 2:
            # Input is already embedding vector (B, embed_dim)
            if hasattr(self.head, "extract_features"):
                return self.head.extract_features(x)
            return x

        if x.ndim != 4:
            raise ValueError(
                f"FundusClassifier expects 4D image tensor (B, C, H, W) or 2D embedding (B, D), got {x.shape}"
            )

        if self.encoder is not None:
            # Extract EyeFound foundation representation
            emb = self.encoder(x)
            if isinstance(emb, tuple):
                emb = emb[0]
            # Project through MLP head to penultimate feature_dim
            feat = self.head.extract_features(emb)
            return feat

        # Fallback CNN feature path
        h = self.init_conv(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        feat = self.pool(h).flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning classification logits (B, num_classes).
        Supports either raw 2D fundus images or pre-extracted EyeFound embeddings.
        """
        if x.ndim == 2:
            # Input is embedding vector (B, embed_dim)
            return self.head(x)

        if x.ndim != 4:
            raise ValueError(
                f"FundusClassifier expects 4D image tensor (B, C, H, W) or 2D embedding (B, D), got {x.shape}"
            )

        if self.encoder is not None:
            # Pass through EyeFound encoder
            emb = self.encoder(x)
            if isinstance(emb, tuple):
                emb = emb[0]
            # Classify using MLP head attached to EyeFound embeddings
            logits = self.head(emb)
            return logits

        # Fallback CNN path
        feat = self.extract_features(x)
        logits = self.head(feat)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized class probabilities via softmax (B, num_classes)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict: bool = False,
    ):
        """Loads trained classifier weights from disk."""
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint not found or empty at: {p}")
        checkpoint = torch.load(p, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=strict)
        self.to(device)
        self.is_trained = True
        print(f"[FundusClassifier] Successfully loaded checkpoint from {p}")

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int = 0,
        val_loss: float = 0.0,
        val_acc: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Saves model weights along with training metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "state_dict": self.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "metrics": metrics or {},
            "model_type": "fundus_only",
            "backbone": self.backbone_type,
            "embed_dim": self.embed_dim,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
        }
        torch.save(payload, p)
        print(f"[FundusClassifier] Checkpoint saved to {p}")


# Alias matching domain nomenclature
EyeFoundFundusClassifier = FundusClassifier


def build_fundus_classifier(
    num_classes: int = 2,
    feature_dim: int = 256,
    dropout: float = 0.3,
    backbone: str = "eyefound",
    embed_dim: int = 512,
    checkpoint_path: Optional[Union[str, Path]] = "checkpoints/classifier_fundus.pt",
    encoder_checkpoint: Optional[Union[str, Path]] = "checkpoints/fundus_encoder.pt",
    device: Union[str, torch.device] = "cpu",
) -> FundusClassifier:
    """Convenience factory function for instantiating the EyeFound-based 2D classifier."""
    model = FundusClassifier(
        in_channels=3,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout=dropout,
        backbone=backbone,
        embed_dim=embed_dim,
        encoder_checkpoint=encoder_checkpoint,
    )
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists() and p.stat().st_size > 0:
            model.load_checkpoint(p, device=device)
    model.to(device)
    return model
