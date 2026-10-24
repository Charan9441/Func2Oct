"""
tests/test_shapes.py

Unit and shape tests for OCT VAE:
- Validates 2D B-scan and 3D volume input/output tensor dimensions.
- Verifies Gaussian latent distribution parameters (mu, logvar) and 8x spatial downsampling.
- Confirms finite loss calculations (L1 + MSE reconstruction and non-negative KL divergence).
- Verifies error handling for missing/empty checkpoints.
- Tests latent encoding pipeline end-to-end.
"""

from pathlib import Path
import pytest
import torch

from fundus2oct.models.oct_vae import (
    CheckpointError,
    OCTAutoencoder2D,
    OCTVAE2D,
    OCTVAE3D,
    build_oct_vae,
    load_vae_checkpoint,
)
from fundus2oct.data.dataset import OCTReconstructionDataset
from fundus2oct.inference.encode_latents import encode_oct_dataset


def test_vae_2d_default_8x32x32_latent_mapping():
    """Verifies default OCTVAE2D maps (B, 1, 256, 256) B-scans to (B, 8, 32, 32) latent space."""
    model = OCTVAE2D()
    model.eval()

    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = model.encode_latent(x, stochastic=False)
        recon = model.decode(z)

    assert mu.shape == (2, 8, 32, 32), f"Expected mu shape (2, 8, 32, 32), got {mu.shape}"
    assert logvar.shape == (2, 8, 32, 32), f"Expected logvar shape (2, 8, 32, 32), got {logvar.shape}"
    assert z.shape == (2, 8, 32, 32), f"Expected latent z shape (2, 8, 32, 32), got {z.shape}"
    assert recon.shape == (2, 1, 256, 256), f"Expected recon shape (2, 1, 256, 256), got {recon.shape}"


def test_autoencoder_2d_8x32x32_mapping():
    """Verifies OCTAutoencoder2D maps (B, 1, 256, 256) B-scans to (B, 8, 32, 32) latent and reconstructs."""
    model = OCTAutoencoder2D(in_channels=1, latent_channels=8, base_channels=32)
    model.eval()

    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        z = model.encode(x)
        recon = model.decode(z)
        out_recon, out_z = model(x)
        recon_helper = model.reconstruct(x)

    assert z.shape == (2, 8, 32, 32), f"Expected latent shape (2, 8, 32, 32), got {z.shape}"
    assert recon.shape == (2, 1, 256, 256), f"Expected recon shape (2, 1, 256, 256), got {recon.shape}"
    assert out_recon.shape == (2, 1, 256, 256)
    assert out_z.shape == (2, 8, 32, 32)
    assert recon_helper.shape == (2, 1, 256, 256)
    assert recon.min() >= -1.0
    assert recon.max() <= 1.0


def test_vae_2d_encoder_shapes():
    """Verifies that OCTEncoder2D downsamples (B, 1, 256, 256) by 8x into (B, 4, 32, 32) when configured."""
    model = OCTVAE2D(
        in_channels=1,
        latent_channels=4,
        base_channels=16,
        channel_multipliers=(1, 2, 4, 8),
    )
    model.eval()

    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        mu, logvar = model.encode(x)

    assert mu.shape == (2, 4, 32, 32), f"Expected mu shape (2, 4, 32, 32), got {mu.shape}"
    assert logvar.shape == (2, 4, 32, 32), f"Expected logvar shape (2, 4, 32, 32), got {logvar.shape}"
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()


def test_vae_2d_decoder_shapes():
    """Verifies that OCTDecoder2D reconstructs (B, 4, 32, 32) into (B, 1, 256, 256) in [-1, 1]."""
    model = OCTVAE2D(
        in_channels=1,
        latent_channels=4,
        base_channels=16,
        channel_multipliers=(1, 2, 4, 8),
    )
    model.eval()

    z = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        x_recon = model.decode(z)

    assert x_recon.shape == (2, 1, 256, 256), f"Expected recon shape (2, 1, 256, 256), got {x_recon.shape}"
    assert x_recon.min() >= -1.0
    assert x_recon.max() <= 1.0


def test_vae_2d_forward_and_reparameterization():
    """Tests forward pass and verifies deterministic vs stochastic reparameterization."""
    model = OCTVAE2D(base_channels=16)
    model.eval()

    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        # Deterministic mode returns mu exactly
        z_det = model.reparameterize(mu, logvar, stochastic=False)
        assert torch.equal(z_det, mu)

        # Stochastic mode produces variation
        z_stoch1 = model.reparameterize(mu, logvar, stochastic=True)
        z_stoch2 = model.reparameterize(mu, logvar, stochastic=True)
        assert not torch.equal(z_stoch1, z_stoch2)


def test_vae_2d_finite_losses():
    """Verifies that loss_function returns finite positive reconstruction loss and non-negative KL."""
    model = OCTVAE2D(base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    x_recon, mu, logvar = model(x, stochastic=True)

    losses = model.loss_function(x, x_recon, mu, logvar, kl_weight=1e-4)

    assert "loss" in losses
    assert "recon_loss" in losses
    assert "kl_loss" in losses

    total_loss = losses["loss"]
    recon_loss = losses["recon_loss"]
    kl_loss = losses["kl_loss"]

    assert torch.isfinite(total_loss)
    assert recon_loss.item() > 0.0
    assert kl_loss.item() >= 0.0


def test_vae_3d_shapes():
    """Verifies 3D volumetric VAE shapes across depth, height, and width."""
    model = OCTVAE3D(
        in_channels=1,
        latent_channels=4,
        base_channels=8,
        channel_multipliers=(1, 2, 4, 8),
    )
    model.eval()

    # 3D volume: (Batch=1, Channels=1, Depth=16, Height=128, Width=128)
    x = torch.randn(1, 1, 16, 128, 128)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        assert mu.shape == (1, 4, 2, 16, 16)

        z = model.reparameterize(mu, logvar, stochastic=False)
        x_recon = model.decode(z)
        assert x_recon.shape == (1, 1, 16, 128, 128)


def test_missing_and_empty_checkpoint_error(tmp_path):
    """Verifies that missing or 0-byte checkpoint raises CheckpointError when strict."""
    model = OCTVAE2D(base_channels=16)

    # 1. Non-existent file
    non_existent = tmp_path / "missing_model.pt"
    with pytest.raises(CheckpointError) as exc_info:
        load_vae_checkpoint(model, non_existent, allow_untrained_fallback=False)
    assert "not found" in str(exc_info.value).lower()

    # 2. Empty (0-byte) file
    empty_file = tmp_path / "empty_model.pt"
    empty_file.touch()
    with pytest.raises(CheckpointError) as exc_info:
        load_vae_checkpoint(model, empty_file, allow_untrained_fallback=False)
    assert "empty" in str(exc_info.value).lower()


def test_dataset_oct_loading():
    """Verifies that OCTReconstructionDataset produces the expected tensor shapes."""
    dataset = OCTReconstructionDataset(
        target_size=(128, 128),
        is_3d=False,
        normalize="bipolar",
        synthetic_count=4,
    )
    assert len(dataset) == 4
    item = dataset[0]
    assert "oct" in item
    assert "sample_id" in item
    assert item["oct"].shape == (1, 128, 128)
    assert item["oct"].dtype == torch.float32


def test_encode_latents_pipeline_smoke(tmp_path):
    """Smoke test for latent encoding pipeline on synthetic dataset."""
    out_dir = tmp_path / "latents_out"
    summary = encode_oct_dataset(
        checkpoint_path=tmp_path / "dummy.pt",
        output_dir=out_dir,
        batch_size=2,
        is_3d=False,
        stochastic=False,
        allow_untrained=True,  # Architecture smoke test
        synthetic_fallback_count=4,
    )

    assert summary["status"] == "success"
    assert summary["total_encoded"] in (4, 20) or summary["total_encoded"] > 0
    assert summary["is_trained"] is False
    assert (out_dir / "all_latents.pt").exists()
    assert (out_dir / "latent_manifest.csv").exists()
