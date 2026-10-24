"""
tests/test_diffusion.py

Unit and architecture tests for Conditional Latent Video/Volume Diffusion Model (LVDM):
- Verifies 3D UNet contains 4 residual blocks (res1, res2, res3, res4).
- Verifies shape integrity for 3D latent volumes (B, C, D, H, W) and 2D B-scans (B, C, 1, H, W).
- Verifies conditioning on EyeFound fundus embeddings (512-dim).
- Verifies Gaussian forward diffusion and CFG / DDIM sampling.
"""

import pytest
import torch

from fundus2oct.models.fundus_encoder import FundusEncoder, build_fundus_encoder
from fundus2oct.models.conditional_diffusion import (
    ConditionalLVDM3D,
    ResBlock3D,
    UNet3D,
    build_lvdm,
    load_diffusion_checkpoint,
)


def test_unet3d_4_residual_blocks():
    """Verifies that UNet3D contains exactly 4 core residual blocks."""
    unet = UNet3D(
        in_channels=8,
        out_channels=8,
        base_channels=32,
        cond_dim=512,
        time_dim=128,
    )

    # 1. Check named attributes exist
    assert hasattr(unet, "res1"), "UNet3D must have res1 block"
    assert hasattr(unet, "res2"), "UNet3D must have res2 block"
    assert hasattr(unet, "res3"), "UNet3D must have res3 block"
    assert hasattr(unet, "res4"), "UNet3D must have res4 block"

    # 2. Check types
    assert isinstance(unet.res1, ResBlock3D)
    assert isinstance(unet.res2, ResBlock3D)
    assert isinstance(unet.res3, ResBlock3D)
    assert isinstance(unet.res4, ResBlock3D)

    # 3. Check res_blocks list has length 4
    assert hasattr(unet, "res_blocks")
    assert len(unet.res_blocks) == 4


def test_unet3d_forward_shapes():
    """Verifies UNet3D preserves tensor shapes for 3D volumes and single B-scans."""
    unet = UNet3D(
        in_channels=8,
        out_channels=8,
        base_channels=16,
        cond_dim=512,
        time_dim=64,
    )
    unet.eval()

    # Case 1: Single B-scan slice (D=1): (B=2, C=8, D=1, H=32, W=32)
    x1 = torch.randn(2, 8, 1, 32, 32)
    t = torch.tensor([10, 50], dtype=torch.long)
    cond = torch.randn(2, 512)

    with torch.no_grad():
        out1 = unet(x1, t, cond)
    assert out1.shape == (2, 8, 1, 32, 32), f"Expected shape (2, 8, 1, 32, 32), got {out1.shape}"

    # Case 2: Multi-slice OCT volume (D=4): (B=1, C=8, D=4, H=32, W=32)
    x2 = torch.randn(1, 8, 4, 32, 32)
    t2 = torch.tensor([100], dtype=torch.long)
    cond2 = torch.randn(1, 512)

    with torch.no_grad():
        out2 = unet(x2, t2, cond2)
    assert out2.shape == (1, 8, 4, 32, 32), f"Expected shape (1, 8, 4, 32, 32), got {out2.shape}"


def test_lvdm_diffusion_loss_forward():
    """Verifies ConditionalLVDM3D training forward pass computes finite MSE noise prediction loss."""
    model = build_lvdm(
        latent_channels=8,
        cond_dim=512,
        base_channels=16,
        timesteps=100,
    )
    model.train()

    z0 = torch.randn(2, 8, 1, 32, 32)
    cond = torch.randn(2, 512)

    loss_dict = model(z0, cond=cond, apply_dropout=True)

    assert "loss" in loss_dict
    assert "predicted_noise" in loss_dict
    assert "target_noise" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].item() > 0.0
    assert loss_dict["predicted_noise"].shape == (2, 8, 1, 32, 32)


def test_lvdm_ddim_sampling():
    """Verifies that DDIM sampling produces synthetic latent volumes with specified shape."""
    model = build_lvdm(
        latent_channels=8,
        cond_dim=512,
        base_channels=16,
        timesteps=50,
    )
    model.eval()

    cond = torch.randn(2, 512)
    target_shape = (2, 8, 1, 32, 32)

    with torch.no_grad():
        synth_latents = model.sample_ddim(
            cond=cond,
            shape=target_shape,
            steps=5,
            guidance_scale=2.5,
        )

    assert synth_latents.shape == target_shape
    assert torch.isfinite(synth_latents).all()


def test_end_to_end_eyefound_conditioning():
    """Verifies end-to-end integration: Fundus image -> EyeFound embedding -> LVDM loss."""
    fundus_encoder = build_fundus_encoder(
        backbone="eyefound",
        embed_dim=512,
        spatial_features=False,
    )
    fundus_encoder.eval()

    lvdm = build_lvdm(
        latent_channels=8,
        cond_dim=512,
        base_channels=16,
        timesteps=100,
    )
    lvdm.eval()

    # Retinal fundus image
    fundus_img = torch.randn(2, 3, 256, 256)
    # OCT latent volume
    oct_latent = torch.randn(2, 8, 1, 32, 32)

    with torch.no_grad():
        cond = fundus_encoder(fundus_img)
        if isinstance(cond, tuple):
            cond = cond[0]
        assert cond.shape == (2, 512)

        out = lvdm(oct_latent, cond=cond, apply_dropout=False)

    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0.0
