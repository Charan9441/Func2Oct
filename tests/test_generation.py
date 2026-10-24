"""
tests/test_generation.py

Unit tests for classification models, multimodal fusion,
and pipeline integrity guards.
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier
from fundus2oct.models.pipeline import Fundus2OCTPipeline, GeneratorNotReadyError
from fundus2oct.data.dataset import DiseaseClassificationDataset


def test_fundus_classifier_shapes():
    """Verifies Model A output logits shape and feature dimensions."""
    model = FundusClassifier(in_channels=3, num_classes=2, feature_dim=128)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(x)
        features = model.extract_features(x)

    assert logits.shape == (2, 2)
    assert features.shape == (2, 128)
    assert torch.isfinite(logits).all()


def test_multimodal_classifier_shapes():
    """Verifies Model B dual-stream forward pass and fusion dimensions."""
    model = MultimodalClassifier(
        fundus_channels=3,
        oct_channels=1,
        num_classes=2,
        feature_dim=128,
    )
    model.eval()

    fundus = torch.randn(2, 3, 256, 256)
    oct_img = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        logits = model(fundus, oct_img)
        f_f, f_o, f_fused = model.extract_multimodal_features(fundus, oct_img)

    assert logits.shape == (2, 2)
    assert f_f.shape == (2, 128)
    assert f_o.shape == (2, 128)
    assert f_fused.shape == (2, 128)
    assert torch.isfinite(logits).all()


def test_pipeline_integrity_guard_when_untrained():
    """Ensures pipeline raises GeneratorNotReadyError when diffusion weights are untrained."""
    pipeline = Fundus2OCTPipeline()
    # Force diffusion_ready to False
    pipeline.diffusion_ready = False

    fundus = torch.randn(1, 3, 256, 256)
    with pytest.raises(GeneratorNotReadyError) as exc_info:
        pipeline.generate(fundus, allow_smoke_test=False)

    assert "blocked" in str(exc_info.value).lower()
    assert "diffusion" in str(exc_info.value).lower()


def test_disease_classification_dataset_class_weights():
    """Verifies balanced class weights computation on synthetic split."""
    ds = DiseaseClassificationDataset(
        manifest_path="nonexistent.csv",
        mode="fundus_only",
        synthetic_fallback_count=10,
    )
    assert len(ds) == 10
    weights = ds.get_class_weights()
    assert weights.shape == (2,)
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()


def test_lvdm_unet_forward_shapes():
    """Verifies that UNet3D accepts 3D volumetric latents and outputs matching noise shapes."""
    from fundus2oct.models.conditional_diffusion import UNet3D

    unet = UNet3D(
        in_channels=8,
        out_channels=8,
        base_channels=16,
        cond_dim=512,
        time_dim=64,
    )
    unet.eval()

    z_t = torch.randn(2, 8, 4, 32, 32)
    t = torch.tensor([10, 250], dtype=torch.long)
    cond = torch.randn(2, 512)

    with torch.no_grad():
        eps_pred = unet(z_t, t, cond)

    assert eps_pred.shape == (2, 8, 4, 32, 32), f"Expected (2, 8, 4, 32, 32), got {eps_pred.shape}"
    assert torch.isfinite(eps_pred).all()


def test_lvdm_reverse_diffusion_ddim_sampling():
    """Verifies accelerated DDIM reverse diffusion sampling produces finite latents."""
    from fundus2oct.models.conditional_diffusion import build_lvdm

    diffusion = build_lvdm(
        latent_channels=8,
        cond_dim=512,
        base_channels=16,
        timesteps=50,
    )
    diffusion.eval()

    cond = torch.randn(1, 512)
    latent_shape = (1, 8, 4, 32, 32)
    with torch.no_grad():
        z_0 = diffusion.sample_ddim(cond=cond, shape=latent_shape, steps=3, guidance_scale=1.5)

    assert z_0.shape == latent_shape
    assert torch.isfinite(z_0).all()


def test_pipeline_generate_32_slice_volume():
    """Verifies that Fundus2OCTPipeline end-to-end generates a complete 32-slice 3D OCT volume."""
    from fundus2oct.data.auto_capture import create_demo_fundus

    pipeline = Fundus2OCTPipeline()
    fundus = create_demo_fundus(blur_level=0.0)

    # Generate 32 slices using fast 2-step reverse diffusion with allow_untrained=True
    result = pipeline.generate_volume(
        fundus=fundus,
        num_slices=32,
        steps=2,
        guidance_scale=2.0,
        seed=123,
        allow_untrained=True,
        return_numpy=True,
    )

    volume = result["volume"]
    slices = result["slices"]
    meta = result["metadata"]

    assert isinstance(volume, np.ndarray)
    assert volume.shape == (32, 256, 256), f"Expected volume shape (32, 256, 256), got {volume.shape}"
    assert volume.dtype == np.uint8
    assert len(slices) == 32
    assert slices[0].shape == (256, 256)
    assert meta["num_slices"] == 32
    assert "elapsed_sec" in meta

