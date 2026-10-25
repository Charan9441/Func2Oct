"""
tests/test_generation_evaluation.py

Unit and integration tests for generation fidelity metrics:
- Fréchet distance mathematical computation (identity, positive distance, shape checks)
- 2D B-scan FID extraction and calculation
- 3D Volumetric FVD extraction and calculation
- Retinal axial depth reflectance profile alignment and Pearson correlation
- End-to-end evaluation pipeline and artifact generation
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from fundus2oct.evaluation.fid_fvd import (
    OCT2DFeatureExtractor,
    OCT3DVolumeFeatureExtractor,
    calculate_frechet_distance,
    compute_axial_depth_profile,
    compute_fid,
    compute_fvd,
    compute_statistics,
    create_mock_oct_pairs,
    evaluate_depth_profile_fidelity,
    evaluate_paired_generation,
)


def test_frechet_distance_identical_distributions():
    """Verifies that the Fréchet distance between identical distributions is exactly 0.0."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.5, 0.3],
        [0.1, 0.3, 0.8],
    ])

    dist = calculate_frechet_distance(mu, sigma, mu, sigma)
    assert dist == pytest.approx(0.0, abs=1e-5)


def test_frechet_distance_distinct_distributions():
    """Verifies that distinct distributions yield a strictly positive Fréchet distance."""
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.eye(2)
    mu2 = np.array([2.0, 3.0])
    sigma2 = np.eye(2) * 1.5

    dist = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert dist > 0.0
    # Mean difference squared is 2^2 + 3^2 = 13.0
    assert dist > 13.0


def test_frechet_distance_shape_mismatch_raises_error():
    """Verifies ValueError when dimensions of mean or covariance matrices mismatch."""
    mu1 = np.array([1.0, 2.0])
    sigma1 = np.eye(2)
    mu2 = np.array([1.0, 2.0, 3.0])
    sigma2 = np.eye(3)

    with pytest.raises(ValueError, match="shape mismatch"):
        calculate_frechet_distance(mu1, sigma1, mu2, sigma1)

    with pytest.raises(ValueError, match="shape mismatch"):
        calculate_frechet_distance(mu1, sigma1, mu1, sigma2)


def test_compute_statistics():
    """Verifies empirical mean and covariance calculation from feature matrices."""
    features = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    mu, sigma = compute_statistics(features)
    assert np.allclose(mu, [3.0, 4.0])
    assert sigma.shape == (2, 2)


def test_2d_bscan_fid():
    """Verifies 2D B-scan FID extraction and calculation."""
    b, c, h, w = 6, 1, 64, 64
    rng = torch.Generator().manual_seed(42)
    real_bscans = torch.randn(b, c, h, w, generator=rng)
    fake_bscans = real_bscans + 0.1 * torch.randn(b, c, h, w, generator=rng)

    fid_score = compute_fid(real_bscans, fake_bscans, batch_size=4)
    assert isinstance(fid_score, float)
    assert fid_score >= 0.0


def test_3d_volumetric_fvd():
    """Verifies 3D Volumetric FVD extraction and calculation."""
    b, c, d, h, w = 4, 1, 8, 32, 32
    rng = torch.Generator().manual_seed(42)
    real_vols = torch.randn(b, c, d, h, w, generator=rng)
    fake_vols = real_vols + 0.05 * torch.randn(b, c, d, h, w, generator=rng)

    fvd_score = compute_fvd(real_vols, fake_vols, batch_size=2)
    assert isinstance(fvd_score, float)
    assert fvd_score >= 0.0


def test_axial_depth_profile_and_correlation():
    """Verifies axial depth profile calculation and layer reflectance correlation."""
    d, h, w = 16, 64, 64
    # Create volume with structured depth reflectance (asymmetric peak at top)
    vol_real = torch.zeros(d, h, w)
    vol_real[:, 5:20, :] = 1.0  # Simulated superficial retinal layer

    prof = compute_axial_depth_profile(vol_real)
    assert len(prof) == h
    assert prof[17] > prof[0]  # Peak at row 17

    # Identical volume should have correlation 1.0
    fidelity_identical = evaluate_depth_profile_fidelity(vol_real, vol_real)
    assert fidelity_identical["profile_correlation"] == pytest.approx(1.0, abs=1e-3)
    assert fidelity_identical["profile_mse"] == pytest.approx(0.0, abs=1e-5)

    # Inverted volume should have negative correlation
    vol_inverted = torch.flip(vol_real, dims=[1])
    fidelity_inverted = evaluate_depth_profile_fidelity(vol_real, vol_inverted)
    assert fidelity_inverted["profile_correlation"] < 0.0


def test_end_to_end_evaluate_paired_generation():
    """Verifies full paired evaluation runner and summary dictionary."""
    real_vols, fake_vols, sample_ids = create_mock_oct_pairs(num_samples=4, d=8, h=32, w=32)
    df_metrics, summary_stats = evaluate_paired_generation(
        real_volumes=real_vols,
        fake_volumes=fake_vols,
        sample_ids=sample_ids,
        device=torch.device("cpu"),
    )

    assert len(df_metrics) == 4
    expected_cols = {"sample_id", "mse", "mae", "psnr_db", "ssim", "axial_profile_correlation"}
    assert expected_cols.issubset(set(df_metrics.columns))

    assert "mean_psnr_db" in summary_stats
    assert "mean_ssim" in summary_stats
    assert "mean_axial_correlation" in summary_stats
    assert "distribution_metrics" in summary_stats
    assert "bscan_fid_2d" in summary_stats["distribution_metrics"]
    assert "volumetric_fvd_3d" in summary_stats["distribution_metrics"]
