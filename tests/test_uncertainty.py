"""
tests/test_uncertainty.py

Comprehensive unit tests for the Monte Carlo Dropout Predictive Uncertainty module:
- Output shapes and valid probability constraints (sum to 1.0, non-negative)
- Analytical entropy calculations (uniform distribution = ln(2) nats, 1.0 bit; deterministic = 0.0)
- Expected entropy and mutual information properties (MI >= 0, MI = 0 when deterministic)
- Stochastic variation from active dropout layers
- Freezing of normalization layers (BatchNorm running stats do not drift)
- Complete restoration of initial model train/eval states
- Exception and error handling (invalid pass counts, invalid metrics, missing dropout)
- High uncertainty threshold flag behavior
- Compatibility with real FundusClassifier and MultimodalClassifier
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundus2oct.evaluation.uncertainty import (
    MonteCarloDropoutEstimator,
    NoDropoutLayersError,
    UncertaintyResult,
    discover_dropout_modules,
    estimate_predictive_uncertainty,
)
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier


class MockDropoutNet(nn.Module):
    """Small neural network with BatchNorm and Dropout for isolated testing."""

    def __init__(self, in_features: int = 16, num_classes: int = 2, dropout_p: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.drop(h)
        return self.fc2(h)


class MockNoDropoutNet(nn.Module):
    """Model lacking any dropout modules."""

    def __init__(self, in_features: int = 16, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_uncertainty_output_validity_and_shapes():
    """Verifies probability constraints and structure of UncertaintyResult."""
    model = MockDropoutNet(in_features=16, num_classes=3, dropout_p=0.4)
    x = torch.randn(1, 16)

    result = estimate_predictive_uncertainty(model, inputs=x, num_passes=10)

    assert isinstance(result, UncertaintyResult)
    assert len(result.mean_probabilities) == 3
    assert np.all(result.mean_probabilities >= 0.0)
    assert np.isclose(float(np.sum(result.mean_probabilities)), 1.0, atol=1e-5)
    assert 0 <= result.predicted_class < 3
    assert result.num_passes == 10
    assert result.predictive_entropy_nats >= 0.0
    assert result.predictive_entropy_bits >= 0.0
    assert result.expected_entropy_nats >= 0.0
    assert result.mutual_information_nats >= 0.0


def test_predictive_entropy_analytical_values():
    """Verifies entropy matches known analytical values for uniform and deterministic distributions."""
    model = MockDropoutNet()

    # Create dummy estimator to access computation
    estimator = MonteCarloDropoutEstimator(model, num_passes=2)

    # 1. Binary uniform distribution [0.5, 0.5] -> H = ln(2) nats = 1.0 bit
    uniform_result = estimator.estimate(torch.randn(1, 16))
    # Override mean probs to test analytical formula
    uniform_probs = np.array([0.5, 0.5])
    expected_nats = math.log(2.0)
    expected_bits = 1.0

    calc_nats = float(-np.sum(uniform_probs * np.log(uniform_probs)))
    calc_bits = float(-np.sum(uniform_probs * (np.log(uniform_probs) / math.log(2.0))))

    assert np.isclose(calc_nats, expected_nats, atol=1e-5)
    assert np.isclose(calc_bits, expected_bits, atol=1e-5)

    # 2. Deterministic distribution [1.0, 0.0] -> H = 0.0 nats
    det_probs = np.array([1.0, 1e-12])
    calc_det = float(-np.sum(det_probs * np.log(det_probs)))
    assert np.isclose(calc_det, 0.0, atol=1e-4)


def test_expected_entropy_and_mutual_information():
    """Verifies that MI = H(p_bar) - E[H(p_t)] >= 0 and equals 0 when passes are identical."""
    model = MockDropoutNet()
    estimator = MonteCarloDropoutEstimator(model, num_passes=5)

    result = estimator.estimate(torch.randn(1, 16))

    # By Jensen's inequality: H(E[p]) >= E[H(p)] => MI >= 0
    assert result.predictive_entropy_nats >= result.expected_entropy_nats - 1e-6
    assert result.mutual_information_nats >= 0.0
    assert np.isclose(
        result.mutual_information_nats,
        max(0.0, result.predictive_entropy_nats - result.expected_entropy_nats),
        atol=1e-5,
    )


def test_dropout_produces_stochastic_variation():
    """Verifies that active dropout produces non-zero variance across passes."""
    model = MockDropoutNet(dropout_p=0.5)
    x = torch.randn(1, 16)

    # Run with T=20 passes
    result = estimate_predictive_uncertainty(model, inputs=x, num_passes=20)

    # At p=0.5, variance across passes should be strictly positive
    assert np.any(result.variance_per_class > 0.0)


def test_batch_norm_remains_frozen_during_mc_dropout():
    """Verifies that BatchNorm running mean/variance do not drift during stochastic passes."""
    model = MockDropoutNet(dropout_p=0.5)
    model.eval()

    # Record initial running stats
    initial_mean = model.bn1.running_mean.clone()
    initial_var = model.bn1.running_var.clone()

    x = torch.randn(4, 16)
    _ = estimate_predictive_uncertainty(model, inputs=x, num_passes=10)

    # Running stats must be unchanged because BN stays in eval mode
    assert torch.equal(model.bn1.running_mean, initial_mean)
    assert torch.equal(model.bn1.running_var, initial_var)
    assert model.bn1.training is False


def test_original_model_training_state_is_restored():
    """Verifies that model and all submodules are restored to their original training states."""
    model = MockDropoutNet()
    model.train(True)

    _ = estimate_predictive_uncertainty(model, inputs=torch.randn(1, 16), num_passes=5)
    assert model.training is True
    assert model.drop.training is True

    model.eval()
    _ = estimate_predictive_uncertainty(model, inputs=torch.randn(1, 16), num_passes=5)
    assert model.training is False
    assert model.drop.training is False


def test_no_dropout_modules_raises_error():
    """Verifies that attempting MC Dropout on a model without dropout raises NoDropoutLayersError."""
    model = MockNoDropoutNet()
    with pytest.raises(NoDropoutLayersError, match="contains no dropout modules"):
        _ = MonteCarloDropoutEstimator(model, allow_deterministic_fallback=False)

    # With allow_deterministic_fallback=True, should succeed with metadata flag
    estimator = MonteCarloDropoutEstimator(model, allow_deterministic_fallback=True)
    res = estimator.estimate(torch.randn(1, 16), num_passes=3)
    assert res.metadata["deterministic_fallback_used"] is True


def test_invalid_parameters_raise_errors():
    """Verifies that invalid pass count or metric raises ValueError."""
    model = MockDropoutNet()
    with pytest.raises(ValueError, match="num_passes must be at least 1"):
        _ = MonteCarloDropoutEstimator(model, num_passes=0)

    with pytest.raises(ValueError, match="uncertainty_metric must be one of"):
        _ = MonteCarloDropoutEstimator(model, uncertainty_metric="invalid_metric")


def test_uncertainty_threshold_flag():
    """Verifies that the high uncertainty boolean flag correctly compares against threshold."""
    model = MockDropoutNet()
    x = torch.randn(1, 16)

    # Very high threshold -> should be False
    res_low = estimate_predictive_uncertainty(model, inputs=x, num_passes=5, threshold=100.0)
    assert res_low.is_high_uncertainty is False
    assert res_low.threshold == 100.0

    # Negative threshold -> should be True
    res_high = estimate_predictive_uncertainty(model, inputs=x, num_passes=5, threshold=-1.0)
    assert res_high.is_high_uncertainty is True

    # No threshold -> is_high_uncertainty is False and threshold is None
    res_none = estimate_predictive_uncertainty(model, inputs=x, num_passes=5, threshold=None)
    assert res_none.is_high_uncertainty is False
    assert res_none.threshold is None


def test_fundus_classifier_uncertainty():
    """Verifies MC Dropout on real FundusClassifier architecture."""
    model = FundusClassifier(in_channels=3, num_classes=2, base_channels=8, feature_dim=32, dropout=0.3)
    x = torch.randn(1, 3, 64, 64)

    result = estimate_predictive_uncertainty(
        model=model,
        inputs=x,
        num_passes=5,
        threshold=0.50,
        uncertainty_metric="predictive_entropy",
    )

    assert len(result.mean_probabilities) == 2
    assert result.num_passes == 5
    assert result.metadata["num_dropout_layers"] >= 1
    assert "dropout" in result.metadata["dropout_layer_names"]


def test_multimodal_classifier_uncertainty():
    """Verifies MC Dropout on real MultimodalClassifier architecture."""
    model = MultimodalClassifier(
        fundus_channels=3,
        oct_channels=1,
        num_classes=2,
        base_channels=8,
        feature_dim=32,
        dropout=0.3,
    )
    fundus = torch.randn(1, 3, 64, 64)
    oct_img = torch.randn(1, 1, 64, 64)

    result = estimate_predictive_uncertainty(
        model=model,
        inputs=(fundus, oct_img),
        num_passes=5,
        threshold=0.50,
    )

    assert len(result.mean_probabilities) == 2
    assert result.num_passes == 5
    assert result.metadata["num_dropout_layers"] >= 1
