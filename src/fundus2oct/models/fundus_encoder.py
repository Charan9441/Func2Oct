"""
src/fundus2oct/models/fundus_encoder.py

Retinal Fundus Image Representation & Conditioning Encoder.
Supports:
1. 'eyefound' / 'eyefound_stub' foundation ViT backbone (inspired by Yukun Zhou / rmaphoh RETFound/EyeFound MAE)
2. 'resnet50' torchvision backbone
3. 'baseline_cnn' lightweight clinical convolutional feature extractor

Extracts:
- Global semantic conditioning vector: (B, embed_dim) (e.g. 512-dim)
- Optional 2D spatial feature grid: (B, embed_dim, H', W') for cross-attention
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Foundation ViT Backbone (EyeFound / RETFound-style Masked Autoencoder ViT)
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Multi-Head Attention and MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class EyeFoundViTBackbone(nn.Module):
    """
    EyeFound Retinal Vision Transformer Backbone.
    Extracts high-dimensional anatomical representations from retinal color fundus photographs.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x_patches = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_patches), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0]  # (B, D)
        patch_feats = x[:, 1:]  # (B, N, D)
        return cls_feat, patch_feats


# ---------------------------------------------------------------------------
# 2. Baseline Lightweight Convolutional Backbone
# ---------------------------------------------------------------------------

class BaselineCNNBackbone(nn.Module):
    """Lightweight 4-stage residual convolutional feature extractor for fundus images."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def make_stage(in_c: int, out_c: int):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(),
            )

        self.stage1 = make_stage(64, 128)
        self.stage2 = make_stage(128, 256)
        self.stage3 = make_stage(256, embed_dim)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)  # (B, embed_dim, H', W')

        cls_feat = self.pool(x3).flatten(1)  # (B, embed_dim)
        return cls_feat, x3


# ---------------------------------------------------------------------------
# 3. Main FundusEncoder Model Wrapper
# ---------------------------------------------------------------------------

class FundusEncoder(nn.Module):
    """
    Unified Fundus Encoder module for conditioning the downstream OCT Latent Diffusion model.

    Returns:
    - Global conditioning embedding c: (B, embed_dim)
    - Optional spatial representation: (B, embed_dim, H', W')
    """

    def __init__(
        self,
        backbone: str = "eyefound",
        embed_dim: int = 512,
        spatial_features: bool = True,
        img_size: int = 256,
        in_channels: int = 3,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.embed_dim = embed_dim
        self.spatial_features = spatial_features
        self.img_size = img_size

        if self.backbone_name in ("eyefound", "eyefound_stub", "retfound", "vit"):
            self.backbone = EyeFoundViTBackbone(
                img_size=img_size,
                patch_size=16,
                in_chans=in_channels,
                embed_dim=embed_dim,
                depth=6,
                num_heads=8,
            )
        elif self.backbone_name in ("resnet50", "resnet"):
            from torchvision.models import resnet50, ResNet50_Weights
            base = resnet50(weights=ResNet50_Weights.DEFAULT if checkpoint_path is None else None)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:  # 'baseline_cnn' or fallback
            self.backbone = BaselineCNNBackbone(in_channels=in_channels, embed_dim=embed_dim)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path], strict: bool = False):
        """Loads weights from checkpoint file if it exists and is non-empty."""
        path = Path(checkpoint_path)
        if not path.exists() or path.stat().st_size == 0:
            print(f"[FundusEncoder] Checkpoint not found or empty at {path}. Using initial weights.")
            return

        try:
            state = torch.load(path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            self.load_state_dict(state, strict=strict)
            print(f"[FundusEncoder] Successfully loaded weights from {path}")
        except Exception as e:
            print(f"[FundusEncoder] Warning: Failed to load checkpoint {path}: {e}")

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Retinal fundus image tensor of shape (B, 3, H, W)
        Returns:
            global_embed: (B, embed_dim)
            (optional) spatial_features: (B, embed_dim, H', W') if spatial_features=True
        """
        if self.backbone_name in ("eyefound", "eyefound_stub", "retfound", "vit"):
            cls_feat, patch_feats = self.backbone(x)
            global_embed = self.projection(cls_feat)

            if self.spatial_features:
                # Reshape (B, N, D) -> (B, D, grid_h, grid_w)
                B, N, D = patch_feats.shape
                grid_sz = int(math.sqrt(N))
                spatial_grid = patch_feats.transpose(1, 2).reshape(B, D, grid_sz, grid_sz)
                return global_embed, spatial_grid
            return global_embed

        elif self.backbone_name in ("resnet50", "resnet"):
            feat = self.backbone(x)
            feat = self.proj(feat)  # (B, embed_dim, H', W')
            cls_feat = self.pool(feat).flatten(1)
            global_embed = self.projection(cls_feat)
            if self.spatial_features:
                return global_embed, feat
            return global_embed

        else:
            cls_feat, spatial_grid = self.backbone(x)
            global_embed = self.projection(cls_feat)
            if self.spatial_features:
                return global_embed, spatial_grid
            return global_embed


def build_fundus_encoder(
    backbone: str = "eyefound",
    embed_dim: int = 512,
    spatial_features: bool = True,
    checkpoint_path: Optional[Union[str, Path]] = "checkpoints/fundus_encoder.pt",
) -> FundusEncoder:
    """Convenience factory function matching application config contracts."""
    return FundusEncoder(
        backbone=backbone,
        embed_dim=embed_dim,
        spatial_features=spatial_features,
        checkpoint_path=checkpoint_path,
    )
