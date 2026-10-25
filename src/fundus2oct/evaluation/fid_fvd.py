"""
src/fundus2oct/evaluation/fid_fvd.py

Distributional fidelity metrics for generative 2D OCT B-scans and 3D OCT volumes:
- Fréchet Inception Distance (FID) for 2D B-scans
- Fréchet Video/Volume Distance (FVD) for 3D OCT volumes (32-slice stacks)
- Axial Depth Intensity Profile correlation (optical layer fidelity)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundus2oct.evaluation.metrics import compute_reconstruction_metrics


# ==============================================================================
# 1. CORE FRÉCHET DISTANCE CALCULATION
# ==============================================================================

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculates the Fréchet distance between two multivariate Gaussians:
        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))
        
    Args:
        mu1: Mean vector for real distribution (D,)
        sigma1: Covariance matrix for real distribution (D, D)
        mu2: Mean vector for synthetic distribution (D,)
        sigma2: Covariance matrix for synthetic distribution (D, D)
        eps: Small regularization offset for numerical stability
        
    Returns:
        Scalar Fréchet distance (lower is better, 0.0 indicates identical distributions).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance matrices shape mismatch: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2

    # Product of covariance matrices
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    # Numerical stability check for imaginary artifacts from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    dist = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, dist)


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes empirical mean vector and covariance matrix from a feature matrix (N, D)."""
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D matrix of shape (N, D), got {features.shape}")
    n_samples = features.shape[0]
    if n_samples < 2:
        # Fallback for degenerate single-sample test case
        mu = np.mean(features, axis=0)
        sigma = np.eye(features.shape[1]) * 1e-6
        return mu, sigma

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ==============================================================================
# 2. FEATURE EXTRACTORS (2D B-SCAN & 3D VOLUME)
# ==============================================================================

class OCT2DFeatureExtractor(nn.Module):
    """
    Lightweight, deterministic feature extractor for 2D OCT B-scans.
    Produces 128-dimensional embedding representations for computing B-scan FID.
    Can operate fully offline without external pretrained download dependencies.
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] == 3:
            # Convert 3-channel to 1-channel grayscale if needed
            x = x.mean(dim=1, keepdim=True)

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        out = self.pool(h).flatten(start_dim=1)
        return out


class OCT3DVolumeFeatureExtractor(nn.Module):
    """
    3D Spatio-Temporal / Volumetric feature extractor for 32-slice OCT volumes.
    Computes volumetric embeddings for Fréchet Video/Volume Distance (FVD).
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, feature_dim, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn4 = nn.BatchNorm3d(feature_dim)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, C, D, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, D, H, W)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        out = self.pool(h).flatten(start_dim=1)
        return out


# ==============================================================================
# 3. HIGH-LEVEL FID & FVD COMPUTATION
# ==============================================================================

def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    extractor: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
) -> float:
    """
    Computes 2D Fréchet Inception Distance (FID) between real and synthetic OCT B-scans.
    
    Args:
        real_images: Tensor of real B-scans (N, C, H, W) or (N, H, W)
        fake_images: Tensor of synthetic B-scans (M, C, H, W) or (M, H, W)
        extractor: Optional feature extractor module (defaults to OCT2DFeatureExtractor)
        device: Torch device
        batch_size: Mini-batch extraction size
        
    Returns:
        Scalar FID score (0.0 represents identical distributions).
    """
    if device is None:
        device = real_images.device

    if extractor is None:
        extractor = OCT2DFeatureExtractor().to(device)
    extractor.eval()

    def extract_all(tensors: torch.Tensor) -> np.ndarray:
        feats = []
        n = len(tensors)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = tensors[i:i + batch_size].to(device)
                feat = extractor(batch).cpu().numpy()
                feats.append(feat)
        return np.concatenate(feats, axis=0)

    feats_real = extract_all(real_images)
    feats_fake = extract_all(fake_images)

    mu_real, sigma_real = compute_statistics(feats_real)
    mu_fake, sigma_fake = compute_statistics(feats_fake)

    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def compute_fvd(
    real_volumes: torch.Tensor,
    fake_volumes: torch.Tensor,
    extractor: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
) -> float:
    """
    Computes 3D Fréchet Video/Volume Distance (FVD) between real and synthetic OCT volumes.
    
    Args:
        real_volumes: Tensor of real 3D volumes (N, C, D, H, W) or (N, D, H, W)
        fake_volumes: Tensor of synthetic 3D volumes (M, C, D, H, W) or (M, D, H, W)
        extractor: Optional 3D feature extractor module (defaults to OCT3DVolumeFeatureExtractor)
        device: Torch device
        batch_size: Mini-batch extraction size
        
    Returns:
        Scalar FVD score.
    """
    if device is None:
        device = real_volumes.device

    if extractor is None:
        extractor = OCT3DVolumeFeatureExtractor().to(device)
    extractor.eval()

    def extract_all(tensors: torch.Tensor) -> np.ndarray:
        feats = []
        n = len(tensors)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = tensors[i:i + batch_size].to(device)
                feat = extractor(batch).cpu().numpy()
                feats.append(feat)
        return np.concatenate(feats, axis=0)

    feats_real = extract_all(real_volumes)
    feats_fake = extract_all(fake_volumes)

    mu_real, sigma_real = compute_statistics(feats_real)
    mu_fake, sigma_fake = compute_statistics(feats_fake)

    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


# ==============================================================================
# 4. AXIAL DEPTH INTENSITY PROFILE (RETINAL LAYER INTEGRITY)
# ==============================================================================

def compute_axial_depth_profile(oct_tensor: torch.Tensor) -> np.ndarray:
    """
    Computes the 1D axial depth intensity profile of an OCT volume or B-scan.
    Averages optical intensity laterally across width (W) and slices (D) to produce
    an axial depth reflectance curve of length H (height).
    
    Retinal layers (e.g., ILM peak, nuclear layer dips, RPE/Bruch's peak)
    exhibit characteristic reflections along this curve.
    """
    if oct_tensor.dim() == 5:
        # (B, C, D, H, W) -> average across B, C, D, W
        profile = oct_tensor.mean(dim=(0, 1, 2, 4)).detach().cpu().numpy()
    elif oct_tensor.dim() == 4:
        # (C, D, H, W) or (B, C, H, W)
        if oct_tensor.shape[1] > 3:  # (C, D, H, W) where D is depth
            profile = oct_tensor.mean(dim=(0, 1, 3)).detach().cpu().numpy()
        else:
            profile = oct_tensor.mean(dim=(0, 1, 3)).detach().cpu().numpy()
    elif oct_tensor.dim() == 3:
        # (D, H, W) or (C, H, W)
        profile = oct_tensor.mean(dim=(0, 2)).detach().cpu().numpy()
    elif oct_tensor.dim() == 2:
        # (H, W) -> average along width
        profile = oct_tensor.mean(dim=1).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported OCT tensor dimension for depth profile: {oct_tensor.shape}")

    return profile


def evaluate_depth_profile_fidelity(
    real_volume: torch.Tensor,
    fake_volume: torch.Tensor,
) -> Dict[str, float]:
    """
    Compares the axial depth reflectance profile of synthetic OCT vs ground-truth real OCT:
    - profile_correlation: Pearson correlation r in [-1.0, 1.0] (1.0 = identical layer reflection)
    - profile_mse: Mean squared error between normalized depth profiles
    - profile_mae: Mean absolute error between normalized depth profiles
    """
    prof_real = compute_axial_depth_profile(real_volume)
    prof_fake = compute_axial_depth_profile(fake_volume)

    if len(prof_real) != len(prof_fake):
        raise ValueError(f"Depth profile length mismatch: real {len(prof_real)} vs fake {len(prof_fake)}")

    # Pearson correlation
    std_real = np.std(prof_real)
    std_fake = np.std(prof_fake)
    if std_real > 1e-8 and std_fake > 1e-8:
        corr = float(np.corrcoef(prof_real, prof_fake)[0, 1])
    else:
        corr = 1.0 if np.allclose(prof_real, prof_fake) else 0.0

    mse = float(np.mean((prof_real - prof_fake) ** 2))
    mae = float(np.mean(np.abs(prof_real - prof_fake)))

    return {
        "profile_correlation": round(corr, 4),
        "profile_mse": round(mse, 6),
        "profile_mae": round(mae, 6),
    }


# ==============================================================================
# 5. HIGH-LEVEL EVALUATION RUNNER & PAIR GENERATOR
# ==============================================================================

def evaluate_paired_generation(
    real_volumes: List[torch.Tensor],
    fake_volumes: List[torch.Tensor],
    sample_ids: List[str],
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes per-sample and global distribution metrics for real vs synthetic OCT volumes.
    """
    if device is None:
        device = torch.device("cpu")

    records = []
    real_tensors_list = []
    fake_tensors_list = []

    for idx, (real_vol, fake_vol, s_id) in enumerate(zip(real_volumes, fake_volumes, sample_ids)):
        if real_vol.dim() == 4 and real_vol.shape[0] == 1:
            real_vol = real_vol.squeeze(0)
        if fake_vol.dim() == 4 and fake_vol.shape[0] == 1:
            fake_vol = fake_vol.squeeze(0)

        if real_vol.shape != fake_vol.shape:
            fake_vol = torch.nn.functional.interpolate(
                fake_vol.unsqueeze(0).unsqueeze(0).float(),
                size=real_vol.shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        recon_metrics = compute_reconstruction_metrics(fake_vol, real_vol, data_range=2.0)
        depth_metrics = evaluate_depth_profile_fidelity(real_vol, fake_vol)

        records.append({
            "sample_id": s_id,
            "mse": round(recon_metrics["mse"], 6),
            "mae": round(recon_metrics["mae"], 6),
            "psnr_db": round(recon_metrics["psnr"], 2),
            "ssim": round(recon_metrics.get("ssim", 0.0), 4),
            "axial_profile_correlation": depth_metrics["profile_correlation"],
            "axial_profile_mse": depth_metrics["profile_mse"],
        })

        real_tensors_list.append(real_vol.unsqueeze(0))
        fake_tensors_list.append(fake_vol.unsqueeze(0))

    df_metrics = pd.DataFrame(records)

    all_real = torch.cat(real_tensors_list, dim=0).unsqueeze(1).float()
    all_fake = torch.cat(fake_tensors_list, dim=0).unsqueeze(1).float()

    b, c, d, h, w = all_real.shape
    real_bscans = all_real.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    fake_bscans = all_fake.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    fid_score = compute_fid(real_bscans, fake_bscans, device=device)
    fvd_score = compute_fvd(all_real, all_fake, device=device)

    summary_stats = {
        "num_evaluated_samples": len(records),
        "mean_psnr_db": round(float(df_metrics["psnr_db"].mean()), 2),
        "std_psnr_db": round(float(df_metrics["psnr_db"].std()), 2) if len(df_metrics) > 1 else 0.0,
        "mean_ssim": round(float(df_metrics["ssim"].mean()), 4),
        "std_ssim": round(float(df_metrics["ssim"].std()), 4) if len(df_metrics) > 1 else 0.0,
        "mean_mae": round(float(df_metrics["mae"].mean()), 6),
        "mean_mse": round(float(df_metrics["mse"].mean()), 6),
        "mean_axial_correlation": round(float(df_metrics["axial_profile_correlation"].mean()), 4),
        "distribution_metrics": {
            "bscan_fid_2d": round(float(fid_score), 4),
            "volumetric_fvd_3d": round(float(fvd_score), 4),
        },
    }

    return df_metrics, summary_stats


def create_mock_oct_pairs(
    num_samples: int = 10,
    d: int = 32,
    h: int = 256,
    w: int = 256,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Generates synthetic paired OCT tensors for smoke test validation."""
    reals = []
    fakes = []
    sample_ids = []
    rng = np.random.default_rng(42)

    for i in range(num_samples):
        base = np.zeros((d, h, w), dtype=np.float32)
        base[:, 55:65, :] += 0.8  # Simulated ILM
        base[:, 110:130, :] += 0.4  # Simulated IPL/OPL
        base[:, 180:195, :] += 0.9  # Simulated RPE
        base += rng.normal(0, 0.05, size=(d, h, w)).astype(np.float32)
        base = np.clip(base, 0.0, 1.0) * 2.0 - 1.0

        synthetic = base + rng.normal(0, 0.08, size=(d, h, w)).astype(np.float32)
        synthetic = np.clip(synthetic, -1.0, 1.0)

        reals.append(torch.from_numpy(base))
        fakes.append(torch.from_numpy(synthetic))
        sample_ids.append(f"eval_eye_{i:03d}")

    return reals, fakes, sample_ids
