"""
tests/test_generation.py

Unit tests for classification models, multimodal fusion,
and pipeline integrity guards.
"""

from pathlib import Path
import pytest
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
