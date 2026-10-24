"""
tests/test_metrics.py

Unit tests for VAE validation metric tracking and benchmark monitoring:
- Verifies MSE, MAE, PSNR, SSIM against target thresholds.
- Verifies benchmark logic for MSE <= 0.0089 and MAE <= 0.0587.
- Tests multi-batch stateful tracking with VAEMetricTracker.
"""

import math
import pytest
import torch

from fundus2oct.evaluation.metrics import (
    TARGET_VAE_MAE,
    TARGET_VAE_MSE,
    VAEMetricTracker,
    compute_l1,
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_reconstruction_metrics,
    compute_ssim,
    evaluate_vae_benchmarks,
)


def test_target_benchmark_constants():
    """Confirms target benchmarks match requirement: MSE <= 0.0089, MAE <= 0.0587."""
    assert TARGET_VAE_MSE == 0.0089
    assert TARGET_VAE_MAE == 0.0587


def test_compute_mse_mae_identical_tensors():
    """Identical tensors should produce zero error, maximum PSNR, and perfect SSIM."""
    x = torch.randn(4, 1, 64, 64)
    mse = compute_mse(x, x)
    mae = compute_mae(x, x)
    psnr = compute_psnr(x, x)
    ssim = compute_ssim(x, x)

    assert mse == 0.0
    assert mae == 0.0
    assert psnr >= 50.0
    assert math.isclose(ssim, 1.0, abs_tol=1e-4)


def test_compute_mse_mae_known_delta():
    """Constant delta should produce predictable analytical MSE and MAE."""
    x = torch.zeros(2, 1, 32, 32)
    delta = 0.05
    y = x + delta

    mse = compute_mse(x, y)
    mae = compute_mae(x, y)
    l1 = compute_l1(x, y)

    assert math.isclose(mse, delta ** 2, rel_tol=1e-5)
    assert math.isclose(mae, delta, rel_tol=1e-5)
    assert math.isclose(l1, delta, rel_tol=1e-5)


def test_compute_psnr_scalar_and_tensor():
    """Tests PSNR with both scalar MSE and tensor input."""
    # With data_range=2.0 and MSE=0.01:
    # PSNR = 10 * log10(4.0 / 0.01) = 10 * log10(400) ~ 26.02 dB
    psnr_scalar = compute_psnr(0.01, data_range=2.0)
    assert math.isclose(psnr_scalar, 26.0205999, rel_tol=1e-4)

    # Tensor input
    x = torch.zeros(1, 1, 16, 16)
    y = torch.full((1, 1, 16, 16), 0.1)  # MSE = 0.01
    psnr_tensor = compute_psnr(x, y, data_range=2.0)
    assert math.isclose(psnr_tensor, psnr_scalar, rel_tol=1e-4)


def test_compute_ssim_2d_and_3d():
    """Verifies SSIM computation on 2D images and 3D volume inputs."""
    # 2D
    x2d = torch.randn(2, 1, 64, 64).clamp(-1.0, 1.0)
    noisy2d = (x2d + 0.02 * torch.randn_like(x2d)).clamp(-1.0, 1.0)
    ssim2d = compute_ssim(x2d, noisy2d, data_range=2.0)
    assert 0.8 < ssim2d <= 1.0

    # 3D
    x3d = torch.randn(1, 1, 8, 32, 32).clamp(-1.0, 1.0)
    noisy3d = (x3d + 0.02 * torch.randn_like(x3d)).clamp(-1.0, 1.0)
    ssim3d = compute_ssim(x3d, noisy3d, data_range=2.0)
    assert 0.8 < ssim3d <= 1.0


def test_evaluate_vae_benchmarks_pass_and_fail():
    """Verifies benchmark logic against target MSE (0.0089) and MAE (0.0587)."""
    # 1. Both pass
    res_pass = evaluate_vae_benchmarks(mse=0.0075, mae=0.0450)
    assert res_pass["mse_passed"] is True
    assert res_pass["mae_passed"] is True
    assert res_pass["all_passed"] is True
    assert res_pass["mse_margin"] > 0
    assert res_pass["mae_margin"] > 0
    assert "BENCHMARKS MET" in res_pass["summary"]

    # 2. MSE fails
    res_fail_mse = evaluate_vae_benchmarks(mse=0.0095, mae=0.0450)
    assert res_fail_mse["mse_passed"] is False
    assert res_fail_mse["mae_passed"] is True
    assert res_fail_mse["all_passed"] is False
    assert res_fail_mse["mse_margin"] < 0
    assert "BENCHMARKS PENDING" in res_fail_mse["summary"]

    # 3. MAE fails
    res_fail_mae = evaluate_vae_benchmarks(mse=0.0075, mae=0.0650)
    assert res_fail_mae["mse_passed"] is True
    assert res_fail_mae["mae_passed"] is False
    assert res_fail_mae["all_passed"] is False
    assert res_fail_mae["mae_margin"] < 0

    # 4. Exact threshold boundary (equality passes <= target)
    res_boundary = evaluate_vae_benchmarks(mse=0.0089, mae=0.0587)
    assert res_boundary["mse_passed"] is True
    assert res_boundary["mae_passed"] is True
    assert res_boundary["all_passed"] is True


def test_vae_metric_tracker_multi_batch():
    """Verifies that VAEMetricTracker accurately accumulates across batches."""
    tracker = VAEMetricTracker(target_mse=0.0089, target_mae=0.0587)

    # Batch 1: delta = 0.05 (MSE = 0.0025, MAE = 0.050) -> Both pass
    x1 = torch.zeros(4, 1, 32, 32)
    y1 = x1 + 0.05
    b1_metrics = tracker.update(y1, x1, compute_ssim_batch=True)
    assert math.isclose(b1_metrics["mse"], 0.0025, rel_tol=1e-4)
    assert math.isclose(b1_metrics["mae"], 0.050, rel_tol=1e-4)

    # Batch 2: delta = 0.02 (MSE = 0.0004, MAE = 0.020) -> Both pass
    x2 = torch.zeros(4, 1, 32, 32)
    y2 = x2 + 0.02
    tracker.update(y2, x2, compute_ssim_batch=True)

    summary = tracker.compute()
    assert summary["total_samples"] == 8
    assert summary["total_batches"] == 2

    # Expected average MSE = (0.0025 + 0.0004) / 2 = 0.00145
    # Expected average MAE = (0.050 + 0.020) / 2 = 0.035
    assert math.isclose(summary["mse"], 0.00145, rel_tol=1e-4)
    assert math.isclose(summary["mae"], 0.035, rel_tol=1e-4)

    bench = summary["benchmarks"]
    assert bench["all_passed"] is True
    assert bench["mse_passed"] is True
    assert bench["mae_passed"] is True

    # Formatting summary output
    summary_text = tracker.format_summary()
    assert "BENCHMARKS MET" in summary_text
    assert "MSE" in summary_text
    assert "MAE" in summary_text


def test_shape_mismatch_raises_error():
    """Tensors of different shapes should raise ValueError."""
    x = torch.zeros(2, 1, 32, 32)
    y = torch.zeros(2, 1, 64, 64)

    with pytest.raises(ValueError):
        compute_mse(x, y)
    with pytest.raises(ValueError):
        compute_mae(x, y)
    with pytest.raises(ValueError):
        compute_ssim(x, y)
