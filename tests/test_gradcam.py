"""
tests/test_gradcam.py

Comprehensive unit tests for the Grad-CAM interpretability module:
- Output dimensions and finite values
- Range bounded in [0.0, 1.0]
- Handling of constant activations / zero gradients without NaNs
- Explicit vs predicted target class selection
- Lifecycle hook cleanup and model train/eval state restoration
- Exception safety (hooks removed even on forward failure)
- Invalid layer detection
- Overlay visualization and image denormalization
- Compatibility with FundusClassifier and MultimodalClassifier
"""

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundus2oct.evaluation.gradcam import (
    GradCAM,
    GradCAMResult,
    compute_gradcam,
    find_target_layer,
    overlay_heatmap_on_image,
)
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier


class SimpleConvNet(nn.Module):
    """Small synthetic CNN with identifiable conv layers for isolated testing."""

    def __init__(self, in_ch: int = 3, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        feat = self.pool(h).flatten(1)
        return self.fc(feat)


def test_gradcam_output_shape_and_finite_range():
    """Verifies that Grad-CAM returns a normalized (H, W) array bounded in [0.0, 1.0]."""
    model = SimpleConvNet(in_ch=3, num_classes=2)
    x = torch.randn(1, 3, 32, 32)

    result = compute_gradcam(model, inputs=x, target_layer=model.conv2)

    assert isinstance(result, GradCAMResult)
    assert result.heatmap.shape == (32, 32)
    assert np.all(np.isfinite(result.heatmap))
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0
    assert result.target_class in (0, 1)
    assert result.predicted_class in (0, 1)
    assert len(result.class_probabilities) == 2


def test_gradcam_explicit_target_class():
    """Verifies that requesting specific target classes generates class-specific CAMs."""
    model = SimpleConvNet(in_ch=3, num_classes=3)
    x = torch.randn(1, 3, 32, 32)

    cam_c0 = compute_gradcam(model, inputs=x, target_class=0, target_layer="conv2")
    cam_c2 = compute_gradcam(model, inputs=x, target_class=2, target_layer="conv2")

    assert cam_c0.target_class == 0
    assert cam_c2.target_class == 2
    assert np.all(np.isfinite(cam_c0.heatmap))
    assert np.all(np.isfinite(cam_c2.heatmap))


def test_constant_input_zero_gradients_no_nans():
    """Verifies that zero gradients or constant activations return zeroed heatmaps without NaNs."""
    model = SimpleConvNet(in_ch=3, num_classes=2)
    # Zero tensor input
    x = torch.zeros(1, 3, 32, 32)

    result = compute_gradcam(model, inputs=x, target_layer=model.conv2)

    assert np.all(np.isfinite(result.heatmap))
    assert not np.isnan(result.heatmap).any()
    assert not np.isinf(result.heatmap).any()


def test_hooks_removed_and_model_state_restored():
    """Verifies that forward/backward hooks are removed and model training state is preserved."""
    model = SimpleConvNet()
    model.train(True)  # Start in training mode
    x = torch.randn(1, 3, 32, 32)

    explainer = GradCAM(model, target_layer=model.conv2)
    # Target module should have 0 forward hooks initially
    assert len(explainer.target_module._forward_hooks) == 0

    _ = explainer.generate(x)

    # After execution: hooks removed, model returned to training mode
    assert len(explainer.target_module._forward_hooks) == 0
    assert model.training is True

    # Check case when model was originally in eval mode
    model.eval()
    _ = explainer.generate(x)
    assert len(explainer.target_module._forward_hooks) == 0
    assert model.training is False


def test_hooks_cleaned_up_on_forward_exception():
    """Verifies exception safety: hooks are removed even if forward pass raises an exception."""
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            _ = self.conv(x)
            raise RuntimeError("Simulated runtime error during forward pass")

    bad_model = FailingModel()
    explainer = GradCAM(bad_model, target_layer=bad_model.conv)

    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        explainer.generate(torch.randn(1, 3, 16, 16))

    # Hook must be cleaned up in finally block
    assert len(explainer.target_module._forward_hooks) == 0


def test_invalid_target_layer_raises_value_error():
    """Verifies that specifying a non-existent or non-module layer raises ValueError."""
    model = SimpleConvNet()
    with pytest.raises(ValueError, match="Target layer 'non_existent_layer' not found"):
        _ = GradCAM(model, target_layer="non_existent_layer")

    class NoConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    with pytest.raises(ValueError, match="Unable to find a convolutional layer"):
        _ = GradCAM(NoConvNet())


def test_overlay_heatmap_on_various_image_types():
    """Verifies that overlay_heatmap_on_image works for PIL, NumPy, and PyTorch tensors."""
    h, w = 64, 64
    heatmap = np.random.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)

    # 1. PIL Image
    pil_img = Image.new("RGB", (w, h), color=(128, 64, 32))
    overlay_pil = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5)
    assert overlay_pil.shape == (h, w, 3)
    assert overlay_pil.dtype == np.uint8

    # 2. NumPy Array uint8
    np_img = np.full((h, w, 3), 100, dtype=np.uint8)
    overlay_np = overlay_heatmap_on_image(np_img, heatmap, alpha=0.5)
    assert overlay_np.shape == (h, w, 3)
    assert overlay_np.dtype == np.uint8

    # 3. Normalized PyTorch tensor (ImageNet)
    tensor_img = torch.randn(3, h, w)
    overlay_tensor = overlay_heatmap_on_image(tensor_img, heatmap, alpha=0.5, normalize_mode="imagenet")
    assert overlay_tensor.shape == (h, w, 3)
    assert overlay_tensor.dtype == np.uint8


def test_fundus_classifier_compatibility():
    """Verifies Grad-CAM on real FundusClassifier architecture."""
    model = FundusClassifier(in_channels=3, num_classes=2, base_channels=8, feature_dim=32, backbone="cnn")
    x = torch.randn(1, 3, 64, 64)

    # Automatic target layer discovery should find layer4
    result = compute_gradcam(model, inputs=x)

    assert result.target_layer_name == "layer4"
    assert result.heatmap.shape == (64, 64)
    assert np.all(np.isfinite(result.heatmap))
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0


def test_multimodal_classifier_compatibility():
    """Verifies Grad-CAM on real MultimodalClassifier architecture for fundus stream."""
    model = MultimodalClassifier(
        fundus_channels=3,
        oct_channels=1,
        num_classes=2,
        base_channels=8,
        feature_dim=32,
        backbone="cnn",
    )
    fundus = torch.randn(1, 3, 64, 64)
    oct_img = torch.randn(1, 1, 64, 64)

    # Fundus stream explanation
    res_fundus = compute_gradcam(model, inputs=(fundus, oct_img), modality="fundus")
    assert res_fundus.target_layer_name == "fundus_l4"
    assert res_fundus.modality_explained == "fundus"
    assert res_fundus.heatmap.shape == (64, 64)

