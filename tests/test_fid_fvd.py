"""
tests/test_fid_fvd.py

Unit tests for generative distribution metrics:
- Frechet Inception Distance (FID) for 2D B-scans.
- Frechet Video Distance (FVD) across 3D OCT volume slices.
- Numerical stability, identical distributions, symmetry, and slice profiling.
- Fast lightweight extractors and streaming accumulation.
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from fundus2oct.evaluation.fid_fvd import (
    FIDCalculator,
    FVDCalculator,
    InceptionV3FeatureExtractor,
    SimpleFeatureExtractor2D,
    SimpleFeatureExtractor3D,
    Volume3DFeatureExtractor,
    compute_activation_statistics,
    compute_fid,
    compute_fid_from_features,
    compute_frechet_distance,
    compute_fvd,
    compute_fvd_from_features,
    compute_slice_wise_fid,
    evaluate_generative_distributions,
)


def test_frechet_distance_identical_distributions():
    """Identical distributions must produce Fréchet distance strictly 0.0."""
    mu = np.array([1.0, 2.0, -1.0])
    sigma = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])

    fd = compute_frechet_distance(mu, sigma, mu, sigma)
    assert fd == 0.0


def test_frechet_distance_known_mean_shift():
    """Shift in mean with identical covariance produces distance equal to ||delta_mu||^2."""
    mu1 = np.zeros(4)
    delta = 2.0
    mu2 = np.full(4, delta)
    sigma = np.eye(4) * 1.5

    fd = compute_frechet_distance(mu1, sigma, mu2, sigma)
    expected = 4 * (delta ** 2)
    assert math.isclose(fd, expected, rel_tol=1e-5)


def test_frechet_distance_known_1d_variance():
    """For 1D Gaussians, FD = (mu1 - mu2)^2 + (s1 - s2)^2."""
    mu1 = np.array([2.0])
    mu2 = np.array([5.0])
    s1 = 2.0  # std dev
    s2 = 5.0
    sigma1 = np.array([[s1 ** 2]])
    sigma2 = np.array([[s2 ** 2]])

    fd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    expected = (5.0 - 2.0) ** 2 + (5.0 - 2.0) ** 2  # 9 + 9 = 18
    assert math.isclose(fd, expected, rel_tol=1e-5)


def test_frechet_distance_symmetry():
    """Fréchet distance must be symmetric: d(A, B) == d(B, A)."""
    mu1 = np.array([0.5, -1.2])
    sigma1 = np.array([[1.0, 0.3], [0.3, 2.0]])
    mu2 = np.array([1.5, 0.8])
    sigma2 = np.array([[2.5, -0.4], [-0.4, 1.8]])

    d12 = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    d21 = compute_frechet_distance(mu2, sigma2, mu1, sigma1)

    assert d12 > 0.0
    assert math.isclose(d12, d21, rel_tol=1e-5)


def test_frechet_distance_singular_covariance():
    """Handles low-rank or singular covariance matrices without raising exceptions."""
    mu1 = np.zeros(5)
    mu2 = np.ones(5)
    # Singular rank-1 matrix
    v = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    sigma1 = v.dot(v.T)
    sigma2 = np.eye(5) * 0.5

    fd = compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-5)
    assert fd >= 0.0
    assert np.isfinite(fd)


def test_compute_activation_statistics():
    """Verifies empirical mean and covariance calculation."""
    rng = np.random.RandomState(42)
    features = rng.normal(loc=3.0, scale=2.0, size=(100, 4))

    mu, sigma = compute_activation_statistics(features)
    assert mu.shape == (4,)
    assert sigma.shape == (4, 4)
    assert np.allclose(mu, [3.0, 3.0, 3.0, 3.0], atol=0.6)
    # Diagonal should be close to 4.0 (2.0^2)
    assert np.allclose(np.diag(sigma), [4.0, 4.0, 4.0, 4.0], atol=1.0)


def test_fid_from_features_zero_for_same():
    """compute_fid_from_features with identical activations returns 0."""
    feat = np.random.randn(20, 16)
    fid = compute_fid_from_features(feat, feat)
    assert fid == 0.0


def test_simple_feature_extractor_2d_and_fid():
    """Tests 2D B-scan feature extractor and FID computation."""
    extractor = SimpleFeatureExtractor2D(in_channels=1, feature_dim=32, device="cpu")
    real_scans = torch.randn(8, 1, 64, 64)
    fake_scans = real_scans + 0.1 * torch.randn_like(real_scans)

    # Self-FID
    fid_self = compute_fid(real_scans, real_scans, feature_extractor=extractor, batch_size=4, device="cpu")
    assert math.isclose(fid_self, 0.0, abs_tol=1e-5)

    # Delta FID
    fid_diff = compute_fid(real_scans, fake_scans, feature_extractor=extractor, batch_size=4, device="cpu")
    assert fid_diff > 0.0


def test_simple_feature_extractor_3d_and_fvd():
    """Tests 3D volume feature extractor and FVD computation."""
    extractor = SimpleFeatureExtractor3D(in_channels=1, feature_dim=32, device="cpu")
    real_volumes = torch.randn(4, 1, 8, 32, 32)
    fake_volumes = real_volumes + 0.15 * torch.randn_like(real_volumes)

    # Self-FVD
    fvd_self = compute_fvd(real_volumes, real_volumes, feature_extractor=extractor, batch_size=2, device="cpu")
    assert math.isclose(fvd_self, 0.0, abs_tol=1e-5)

    # Different FVD
    fvd_diff = compute_fvd(real_volumes, fake_volumes, feature_extractor=extractor, batch_size=2, device="cpu")
    assert fvd_diff > 0.0


def test_slice_wise_fid():
    """Tests depth slice-by-slice FID profiling across OCT volumes."""
    extractor = SimpleFeatureExtractor2D(in_channels=1, feature_dim=16, device="cpu")
    num_volumes = 6
    depth = 5
    real_vol = torch.randn(num_volumes, depth, 32, 32)
    fake_vol = real_vol + 0.1 * torch.randn_like(real_vol)

    res = compute_slice_wise_fid(
        real_vol,
        fake_vol,
        feature_extractor=extractor,
        batch_size=4,
        device="cpu",
    )

    assert "per_slice" in res
    assert len(res["per_slice"]) == depth
    assert 0 <= res["best_slice"] < depth
    assert 0 <= res["worst_slice"] < depth
    assert res["mean_slice_fid"] >= 0.0
    assert res["min_slice_fid"] <= res["max_slice_fid"]


def test_streaming_calculators():
    """Tests stateful FIDCalculator and FVDCalculator accumulation."""
    extractor2d = SimpleFeatureExtractor2D(in_channels=1, feature_dim=16, device="cpu")
    calc = FIDCalculator(feature_extractor=extractor2d, device="cpu")

    scans1 = torch.randn(5, 1, 32, 32)
    scans2 = torch.randn(5, 1, 32, 32)
    calc.update_real(scans1)
    calc.update_fake(scans2)

    fid = calc.compute()
    assert fid >= 0.0

    calc.reset()
    assert len(calc.real_features) == 0
    assert len(calc.fake_features) == 0


def test_evaluate_generative_distributions_combined():
    """Verifies combined evaluation (FID + FVD + slice-wise) with lightweight extractors."""
    ext2d = SimpleFeatureExtractor2D(in_channels=1, feature_dim=16, device="cpu")
    ext3d = SimpleFeatureExtractor3D(in_channels=1, feature_dim=16, device="cpu")

    real_v = torch.randn(4, 4, 32, 32)
    fake_v = real_v + 0.05 * torch.randn_like(real_v)

    results = evaluate_generative_distributions(
        real_volumes=real_v,
        fake_volumes=fake_v,
        compute_slice_fid_flag=True,
        extractor_2d=ext2d,
        extractor_3d=ext3d,
        batch_size=2,
        device="cpu",
    )

    assert "fid" in results
    assert "fvd" in results
    assert "slice_wise_fid" in results
    assert results["num_real_volumes"] == 4
    assert results["num_fake_volumes"] == 4
    assert results["volume_depth"] == 4
    assert results["fid"] >= 0.0
    assert results["fvd"] >= 0.0


def test_inception_v3_feature_extractor_shapes():
    """Verifies InceptionV3FeatureExtractor forward pass and output dimensionality (2048)."""
    extractor = InceptionV3FeatureExtractor(pretrained=False, device="cpu")
    # Grayscale B-scans (2, 1, 64, 64)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        feat = extractor(x)
    assert feat.shape == (2, 2048)
    assert torch.isfinite(feat).all()


def test_volume_3d_feature_extractor_shapes():
    """Verifies Volume3DFeatureExtractor forward pass and output dimensionality (512)."""
    extractor = Volume3DFeatureExtractor(pretrained=False, target_slices=8, device="cpu")
    # Volume (1, 8, 32, 32)
    vol = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        feat = extractor(vol)
    assert feat.shape == (1, 512)
    assert torch.isfinite(feat).all()


def test_exact_subspace_projection_fid():
    """Verifies that subspace projection matches full high-dimensional FID calculation."""
    rng = np.random.RandomState(42)
    n_real, n_fake, d_feat = 3, 3, 128
    feat_real = rng.randn(n_real, d_feat).astype(np.float64)
    feat_fake = rng.randn(n_fake, d_feat).astype(np.float64) + 0.5

    # 1. Full high-dimensional calculation
    mu1, s1 = compute_activation_statistics(feat_real)
    mu2, s2 = compute_activation_statistics(feat_fake)
    fid_full = compute_frechet_distance(mu1, s1, mu2, s2)

    # 2. Subspace projection via compute_fid_from_features
    fid_subspace = compute_fid_from_features(feat_real, feat_fake)

    assert fid_full >= 0.0
    assert fid_subspace >= 0.0
    assert math.isclose(fid_full, fid_subspace, rel_tol=1e-4, abs_tol=1e-4)


def test_generation_evaluation_cli_execution(tmp_path):
    """Verifies that evaluate_generation.py CLI executes cleanly in demo mode and exports artifacts."""
    import subprocess
    import sys
    import json

    cli_script = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_generation.py"
    out_dir = tmp_path / "gen_eval_reports"

    cmd = [
        sys.executable,
        str(cli_script),
        "--demo",
        "--use-lightweight-extractor",
        "--max-samples", "2",
        "--output-dir", str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"CLI exited with {res.returncode}. Stderr: {res.stderr}"

    # Verify generated artifacts
    json_report = out_dir / "generation_evaluation.json"
    csv_report = out_dir / "generation_evaluation.csv"
    slice_csv = out_dir / "slice_fid_profile.csv"
    png_plot = out_dir / "generation_evaluation.png"

    assert json_report.exists()
    assert csv_report.exists()
    assert slice_csv.exists()
    assert png_plot.exists()

    with open(json_report, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "fid" in data
    assert "fvd" in data
    assert "reconstruction" in data
    assert "slice_wise_fid" in data
    assert data["num_real_volumes"] == 2

