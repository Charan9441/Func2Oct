"""
src/fundus2oct/evaluation/fid_fvd.py

Distributional Generative Distance Metrics for Retinal OCT Synthesis:
1. Frechet Inception Distance (FID) on individual 2D B-scans.
2. Frechet Video Distance (FVD) across 3D OCT volume slices.
3. Slice-wise depth FID profiling (anterior to posterior depth tracking).

Mathematical Formulation:
Fréchet Distance (Wasserstein-2 distance between two multivariate Gaussians):
    d^2((μ_r, Σ_r), (μ_g, Σ_g)) = ||μ_r - μ_g||_2^2 + Tr(Σ_r + Σ_g - 2 * (Σ_r * Σ_g)^{1/2})

Numerical Stability:
Uses matrix square root with regularized diagonal shifts, imaginary component suppression,
and singular covariance stabilization.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("fundus2oct.evaluation.fid_fvd")


# =====================================================================
# 1. CORE MATHEMATICAL FORMULATION: FRECHET DISTANCE
# =====================================================================

def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Computes the Fréchet distance (Wasserstein-2 distance) between two multivariate Gaussians:
        d^2 = ||mu1 - mu2||_2^2 + Tr(sigma1 + sigma2 - 2 * sqrtm(sigma1 * sigma2))

    Args:
        mu1: Mean vector of distribution 1, shape (D,).
        sigma1: Covariance matrix of distribution 1, shape (D, D).
        mu2: Mean vector of distribution 2, shape (D,).
        sigma2: Covariance matrix of distribution 2, shape (D, D).
        eps: Small regularization added to the diagonal if covariance product is singular.

    Returns:
        float: Computed Fréchet distance (non-negative).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance matrices shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    if sigma1.shape[0] != sigma1.shape[1] or sigma1.shape[0] != mu1.shape[0]:
        raise ValueError(
            f"Dimension mismatch between mean ({mu1.shape}) and covariance ({sigma1.shape})"
        )

    # If distributions are identical (e.g. self-comparison), distance is strictly 0.0
    if np.array_equal(mu1, mu2) and np.array_equal(sigma1, sigma2):
        return 0.0

    diff = mu1 - mu2
    mean_diff_sq = float(diff.dot(diff))

    # Product of covariance matrices
    res = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    covmean = res[0] if isinstance(res, tuple) else res

    # Handle numerical singularities or NaN/inf
    if not np.isfinite(covmean).all():
        logger.warning("Fréchet distance matrix sqrt produced non-finite values. Applying diagonal regularization.")
        offset = np.eye(sigma1.shape[0]) * eps
        res_reg = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = res_reg[0] if isinstance(res_reg, tuple) else res_reg

    # Discard negligible imaginary components produced by roundoff error
    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            logger.debug("Fréchet matrix square root has notable imaginary component (max=%.4e); taking real part.", max_imag)
        covmean = covmean.real

    tr_covmean = float(np.trace(covmean))
    tr_sigma1 = float(np.trace(sigma1))
    tr_sigma2 = float(np.trace(sigma2))

    dist = mean_diff_sq + tr_sigma1 + tr_sigma2 - 2.0 * tr_covmean

    # Floating point precision clamping (prevent -0.000000001)
    return float(max(0.0, dist))


def compute_activation_statistics(
    features: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes empirical mean vector and covariance matrix from activation feature vectors.

    Args:
        features: Array or tensor of shape (N, D) where N is number of samples, D is feature dim.
        eps: Small diagonal jitter added when N <= D to ensure positive semi-definiteness.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mu, sigma)
            mu: shape (D,)
            sigma: shape (D, D)
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    elif features.ndim != 2:
        features = features.reshape(features.shape[0], -1)

    n_samples, feat_dim = features.shape
    if n_samples == 0:
        raise ValueError("Cannot compute activation statistics for an empty feature array.")

    mu = np.mean(features, axis=0)

    if n_samples == 1:
        # Single sample cannot define variance; return zero covariance with eps jitter
        sigma = np.eye(feat_dim) * eps
        return mu, sigma

    sigma = np.cov(features, rowvar=False)
    sigma = np.atleast_2d(sigma)

    # If samples <= dimensions, covariance matrix is low-rank / singular
    if n_samples <= feat_dim or np.linalg.matrix_rank(sigma) < feat_dim:
        sigma = sigma + np.eye(feat_dim) * eps

    return mu, sigma


def compute_fid_from_features(
    feat_real: Union[np.ndarray, torch.Tensor],
    feat_fake: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> float:
    """Computes FID given pre-extracted feature representations for real and fake samples."""
    if isinstance(feat_real, torch.Tensor):
        feat_real = feat_real.detach().cpu().numpy()
    if isinstance(feat_fake, torch.Tensor):
        feat_fake = feat_fake.detach().cpu().numpy()

    feat_real = np.asarray(feat_real, dtype=np.float64)
    feat_fake = np.asarray(feat_fake, dtype=np.float64)
    if feat_real.ndim == 1:
        feat_real = feat_real.reshape(-1, 1)
    elif feat_real.ndim != 2:
        feat_real = feat_real.reshape(feat_real.shape[0], -1)

    if feat_fake.ndim == 1:
        feat_fake = feat_fake.reshape(-1, 1)
    elif feat_fake.ndim != 2:
        feat_fake = feat_fake.reshape(feat_fake.shape[0], -1)

    n_r, d_r = feat_real.shape
    n_f, d_f = feat_fake.shape
    if d_r != d_f:
        raise ValueError(f"Feature dimension mismatch: {d_r} vs {d_f}")

    # Fast exact subspace projection when sample count is smaller than feature dimension (e.g. N + M + 1 < D)
    # Outside the affine span of {X, Y, mu_r - mu_f}, both covariances are isotropic eps*I, which cancel out identically.
    if (n_r + n_f + 1) < d_r:
        mu_r = np.mean(feat_real, axis=0)
        mu_f = np.mean(feat_fake, axis=0)
        diff = mu_r - mu_f

        denom_r = np.sqrt(max(1, n_r - 1))
        denom_f = np.sqrt(max(1, n_f - 1))
        candidates = np.vstack([
            (feat_real - mu_r) / denom_r,
            (feat_fake - mu_f) / denom_f,
            diff.reshape(1, -1),
        ])
        Q, _ = np.linalg.qr(candidates.T)
        k = Q.shape[1]

        feat_real_proj = feat_real.dot(Q)
        feat_fake_proj = feat_fake.dot(Q)
        mu_real, sigma_real = compute_activation_statistics(feat_real_proj, eps=eps)
        mu_fake, sigma_fake = compute_activation_statistics(feat_fake_proj, eps=eps)
        return compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=eps)

    mu_real, sigma_real = compute_activation_statistics(feat_real, eps=eps)
    mu_fake, sigma_fake = compute_activation_statistics(feat_fake, eps=eps)
    return compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=eps)


# =====================================================================
# 2. 2D B-SCAN FEATURE EXTRACTORS (INCEPTION-V3 & LIGHTWEIGHT)
# =====================================================================

class InceptionV3FeatureExtractor(nn.Module):
    """
    Standard Inception-V3 feature extractor for 2D OCT B-scans.
    Extracts 2048-dimensional feature representations from the penultimate pooling layer.
    Preprocesses inputs:
    - Automatically replicates 1-channel grayscale B-scans to 3 channels.
    - Bilinearly interpolates to standard (299, 299) spatial resolution.
    - Applies standard ImageNet mean/std normalization.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = 2048

        try:
            import torchvision.models as models
            # Attempt to initialize InceptionV3
            if pretrained and weights_path is None:
                try:
                    weights = models.Inception_V3_Weights.DEFAULT
                    self.inception = models.inception_v3(weights=weights, transform_input=False)
                except Exception as e:
                    logger.warning(
                        "Could not download InceptionV3 default weights (%s). Falling back to un-pretrained model.",
                        str(e),
                    )
                    self.inception = models.inception_v3(weights=None, transform_input=False)
            elif weights_path is not None:
                self.inception = models.inception_v3(weights=None, transform_input=False)
                state = torch.load(weights_path, map_location="cpu")
                self.inception.load_state_dict(state)
            else:
                self.inception = models.inception_v3(weights=None, transform_input=False)

            # Replace classification head with Identity
            self.inception.fc = nn.Identity()
            self.inception.eval()
            self.inception.to(self.device)
            for p in self.inception.parameters():
                p.requires_grad = False
        except Exception as err:
            logger.error("Failed to build InceptionV3: %s", err)
            raise

        # Normalization constants (ImageNet standard)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))

    def _preprocess_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Standardizes batch of B-scans for Inception-V3."""
        # Shape handling: (B, H, W) -> (B, 1, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected B-scan tensor of shape (B, C, H, W) or (B, H, W), got {x.shape}")

        # Ensure on proper device & float
        x = x.to(self.device).float()

        # Handle 1-channel grayscale -> 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3, :, :]

        # Normalize to [0, 1] range if in [-1, 1] or [0, 255]
        min_val = x.min()
        max_val = x.max()
        if min_val < 0.0:  # Assuming [-1, 1]
            x = (x + 1.0) / 2.0
        elif max_val > 1.5:  # Assuming [0, 255]
            x = x / 255.0
        x = torch.clamp(x, 0.0, 1.0)

        # Bilinear resize to 299x299 if needed
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 2048-dim features from B-scans.
        Returns tensor of shape (B, 2048).
        """
        x_norm = self._preprocess_batch(x)
        features = self.inception(x_norm)
        if isinstance(features, tuple):
            features = features[0]
        return features.flatten(start_dim=1)


class SimpleFeatureExtractor2D(nn.Module):
    """
    Lightweight, deterministic 2D feature extractor for fast tests and offline evaluation.
    Produces 128-dim features without requiring large external pretrained weights.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128, device: Optional[str] = None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.eval()
        self.to(self.device)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.to(self.device).float()
        if x.shape[1] > 1:
            x = x[:, :1, :, :]
        # Normalize to [0, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        elif x.max() > 1.5:
            x = x / 255.0
        feat = self.net(x)
        return feat.view(feat.shape[0], -1)


# =====================================================================
# 3. 3D VOLUMETRIC / VIDEO FEATURE EXTRACTORS (R3D-18 & LIGHTWEIGHT)
# =====================================================================

class Volume3DFeatureExtractor(nn.Module):
    """
    3D Spatio-Temporal feature extractor for 3D OCT volumes (e.g., 32-slice volumes).
    Uses R3D-18 (ResNet3D) to extract 512-dimensional representations capturing volumetric coherence.
    
    Expects input volume tensors of shape:
        - (B, 1, D, H, W) or (B, D, H, W) or (B, C, D, H, W)
    """

    def __init__(
        self,
        pretrained: bool = False,
        weights_path: Optional[Union[str, Path]] = None,
        target_slices: int = 32,
        spatial_size: Tuple[int, int] = (112, 112),
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = 512
        self.target_slices = target_slices
        self.spatial_size = spatial_size

        try:
            import torchvision.models.video as vmodels
            if pretrained and weights_path is None:
                try:
                    weights = vmodels.R3D_18_Weights.DEFAULT
                    self.r3d = vmodels.r3d_18(weights=weights)
                except Exception as e:
                    logger.warning("Could not download R3D_18 default weights (%s). Initializing un-pretrained.", str(e))
                    self.r3d = vmodels.r3d_18(weights=None)
            elif weights_path is not None:
                self.r3d = vmodels.r3d_18(weights=None)
                state = torch.load(weights_path, map_location="cpu")
                self.r3d.load_state_dict(state)
            else:
                self.r3d = vmodels.r3d_18(weights=None)

            self.r3d.fc = nn.Identity()
            self.r3d.eval()
            self.r3d.to(self.device)
            for p in self.r3d.parameters():
                p.requires_grad = False
        except Exception as err:
            logger.error("Failed to build R3D-18 volume feature extractor: %s", err)
            raise

    def _preprocess_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses 3D volume into (B, 3, T, H, W) normalized for 3D CNNs.
        """
        # (B, D, H, W) -> (B, 1, D, H, W)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            # Single volume (D, H, W) -> (1, 1, D, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim != 5:
            raise ValueError(f"Expected 3D volume tensor with 4 or 5 dims, got shape {x.shape}")

        x = x.to(self.device).float()

        # Channel replication (1 -> 3)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3, :, :, :]

        # Normalize to [0, 1]
        min_v = x.min()
        max_v = x.max()
        if min_v < 0:
            x = (x + 1.0) / 2.0
        elif max_v > 1.5:
            x = x / 255.0
        x = torch.clamp(x, 0.0, 1.0)

        b, c, d, h, w = x.shape

        # Slices adjustment: if depth != target_slices or spatial != spatial_size
        need_temporal = (d != self.target_slices)
        need_spatial = ((h, w) != self.spatial_size)

        if need_temporal or need_spatial:
            x = F.interpolate(
                x,
                size=(self.target_slices, self.spatial_size[0], self.spatial_size[1]),
                mode="trilinear",
                align_corners=False,
            )

        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim volumetric representation across slices."""
        x_prep = self._preprocess_volume(x)
        feat = self.r3d(x_prep)
        return feat.flatten(start_dim=1)


class SimpleFeatureExtractor3D(nn.Module):
    """
    Lightweight, fast 3D CNN feature extractor for fast unit tests and smoke tests.
    Produces 128-dim features across volume slices without downloading large models.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128, device: Optional[str] = None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cpu")
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, feature_dim, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.eval()
        self.to(self.device)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(self.device).float()
        if x.shape[1] > 1:
            x = x[:, :1, :, :, :]
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        elif x.max() > 1.5:
            x = x / 255.0
        feat = self.net(x)
        return feat.view(feat.shape[0], -1)


# =====================================================================
# 4. HIGH-LEVEL COMPUTATION FUNCTIONS: FID, FVD & SLICE-WISE FID
# =====================================================================

def extract_features(
    data_input: Union[torch.Tensor, np.ndarray, Sequence[Any], DataLoader],
    model: nn.Module,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Extracts features for an arbitrary sequence/tensor of images or volumes.
    """
    dev = torch.device(device) if device else getattr(model, "device", torch.device("cpu"))
    features_list: List[np.ndarray] = []

    if isinstance(data_input, DataLoader):
        for batch in data_input:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch.get("oct") or batch.get("image") or batch.get("volume")
            else:
                x = batch
            if x is not None:
                feat = model(x.to(dev))
                features_list.append(feat.detach().cpu().numpy())
    elif isinstance(data_input, torch.Tensor):
        num_items = data_input.shape[0]
        for i in range(0, num_items, batch_size):
            batch = data_input[i : i + batch_size].to(dev)
            feat = model(batch)
            features_list.append(feat.detach().cpu().numpy())
    elif isinstance(data_input, np.ndarray):
        t_data = torch.from_numpy(data_input)
        num_items = t_data.shape[0]
        for i in range(0, num_items, batch_size):
            batch = t_data[i : i + batch_size].to(dev)
            feat = model(batch)
            features_list.append(feat.detach().cpu().numpy())
    elif isinstance(data_input, Sequence):
        # List of tensors or arrays
        batches = [data_input[i : i + batch_size] for i in range(0, len(data_input), batch_size)]
        for b in batches:
            if isinstance(b[0], torch.Tensor):
                stacked = torch.stack(b, dim=0).to(dev)
            elif isinstance(b[0], np.ndarray):
                stacked = torch.from_numpy(np.stack(b, axis=0)).to(dev)
            else:
                stacked = torch.stack([torch.tensor(item) for item in b], dim=0).to(dev)
            feat = model(stacked)
            features_list.append(feat.detach().cpu().numpy())
    else:
        raise TypeError(f"Unsupported data_input type for feature extraction: {type(data_input)}")

    if not features_list:
        return np.empty((0, getattr(model, "feature_dim", 128)), dtype=np.float32)

    return np.concatenate(features_list, axis=0)


def compute_fid(
    real_scans: Union[torch.Tensor, np.ndarray, Sequence[Any], DataLoader],
    fake_scans: Union[torch.Tensor, np.ndarray, Sequence[Any], DataLoader],
    feature_extractor: Optional[nn.Module] = None,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    eps: float = 1e-6,
) -> float:
    """
    Computes Frechet Inception Distance (FID) on individual 2D B-scans.

    Args:
        real_scans: Reference authentic B-scans (Tensor, ndarray, DataLoader, or List).
        fake_scans: Synthesized B-scans to evaluate.
        feature_extractor: 2D feature extractor module (defaults to InceptionV3FeatureExtractor).
        batch_size: Mini-batch size for inference.
        device: Computation device ('cpu', 'cuda', etc.).
        eps: Small numerical stabilizer.

    Returns:
        float: Fréchet Inception Distance (FID).
    """
    if feature_extractor is None:
        feature_extractor = InceptionV3FeatureExtractor(device=device)

    feat_real = extract_features(real_scans, feature_extractor, batch_size=batch_size, device=device)
    feat_fake = extract_features(fake_scans, feature_extractor, batch_size=batch_size, device=device)

    return compute_fid_from_features(feat_real, feat_fake, eps=eps)


def compute_fvd_from_features(
    feat_real: Union[np.ndarray, torch.Tensor],
    feat_fake: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> float:
    """Computes FVD given pre-extracted 3D spatio-temporal feature representations."""
    return compute_fid_from_features(feat_real, feat_fake, eps=eps)


def compute_fvd(
    real_volumes: Union[torch.Tensor, np.ndarray, Sequence[Any], DataLoader],
    fake_volumes: Union[torch.Tensor, np.ndarray, Sequence[Any], DataLoader],
    feature_extractor: Optional[nn.Module] = None,
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
    eps: float = 1e-6,
) -> float:
    """
    Computes Frechet Video Distance (FVD) across 3D volume slices.
    Treats depth slices across the OCT volume as temporal frames in a sequence to measure
    inter-slice volumetric and structural consistency.

    Args:
        real_volumes: Reference authentic 3D OCT volumes (shape: B, 1, D, H, W or B, D, H, W).
        fake_volumes: Synthesized 3D OCT volumes.
        feature_extractor: 3D spatio-temporal extractor (defaults to Volume3DFeatureExtractor).
        batch_size: Mini-batch size.
        device: Computation device.
        eps: Small numerical stabilizer.

    Returns:
        float: Fréchet Video Distance (FVD).
    """
    if feature_extractor is None:
        feature_extractor = Volume3DFeatureExtractor(device=device)

    feat_real = extract_features(real_volumes, feature_extractor, batch_size=batch_size, device=device)
    feat_fake = extract_features(fake_volumes, feature_extractor, batch_size=batch_size, device=device)

    return compute_fvd_from_features(feat_real, feat_fake, eps=eps)


def compute_slice_wise_fid(
    real_volumes: Union[torch.Tensor, np.ndarray],
    fake_volumes: Union[torch.Tensor, np.ndarray],
    feature_extractor: Optional[nn.Module] = None,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    eps: float = 1e-6,
    precomputed_features_real: Optional[np.ndarray] = None,
    precomputed_features_fake: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Performs depth slice-by-slice FID profiling across the OCT volume.
    Extracts B-scans at depth index d (from slice 0 to slice D-1) across both cohorts,
    evaluating anatomical fidelity progression from anterior to central to posterior slices.

    Returns:
        Dict[str, Any]: {
            "per_slice": Dict[int, float],  # slice_index -> FID
            "mean_slice_fid": float,
            "min_slice_fid": float,
            "max_slice_fid": float,
            "best_slice": int,
            "worst_slice": int,
        }
    """
    if isinstance(real_volumes, np.ndarray):
        real_volumes = torch.from_numpy(real_volumes)
    if isinstance(fake_volumes, np.ndarray):
        fake_volumes = torch.from_numpy(fake_volumes)

    # Standardize to (B, D, H, W)
    if real_volumes.ndim == 5 and real_volumes.shape[1] == 1:
        real_volumes = real_volumes.squeeze(1)
    if fake_volumes.ndim == 5 and fake_volumes.shape[1] == 1:
        fake_volumes = fake_volumes.squeeze(1)

    if real_volumes.ndim != 4 or fake_volumes.ndim != 4:
        raise ValueError(
            f"Expected (B, D, H, W) volumes for slice-wise evaluation. Got real {real_volumes.shape}, fake {fake_volumes.shape}"
        )

    b_real, d_real, h_real, w_real = real_volumes.shape
    b_fake, d_fake, h_fake, w_fake = fake_volumes.shape
    depth = min(d_real, d_fake)

    if precomputed_features_real is not None and precomputed_features_fake is not None:
        feat_r_all = precomputed_features_real
        feat_f_all = precomputed_features_fake
    else:
        if feature_extractor is None:
            feature_extractor = InceptionV3FeatureExtractor(device=device)
        real_flat = real_volumes.reshape(-1, 1, h_real, w_real)
        fake_flat = fake_volumes.reshape(-1, 1, h_fake, w_fake)
        feat_r_flat = extract_features(real_flat, feature_extractor, batch_size=batch_size, device=device)
        feat_f_flat = extract_features(fake_flat, feature_extractor, batch_size=batch_size, device=device)
        feat_r_all = feat_r_flat.reshape(b_real, d_real, -1)
        feat_f_all = feat_f_flat.reshape(b_fake, d_fake, -1)

    per_slice: Dict[int, float] = {}

    for d in range(depth):
        fid_d = compute_fid_from_features(
            feat_r_all[:, d, :],
            feat_f_all[:, d, :],
            eps=eps,
        )
        per_slice[d] = float(fid_d)

    values = list(per_slice.values())
    best_slice = int(min(per_slice, key=per_slice.get))  # type: ignore
    worst_slice = int(max(per_slice, key=per_slice.get))  # type: ignore

    return {
        "per_slice": per_slice,
        "mean_slice_fid": float(np.mean(values)),
        "min_slice_fid": float(np.min(values)),
        "max_slice_fid": float(np.max(values)),
        "std_slice_fid": float(np.std(values)),
        "best_slice": best_slice,
        "worst_slice": worst_slice,
    }


# =====================================================================
# 5. STATEFUL CALCULATORS FOR STREAMING ACCUMULATION
# =====================================================================

class FIDCalculator:
    """
    Streaming accumulator for computing 2D FID across large sets of B-scans
    without keeping all high-dimensional feature arrays in memory simultaneously.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.extractor = feature_extractor or InceptionV3FeatureExtractor(device=device)
        self.device = device
        self.real_features: List[np.ndarray] = []
        self.fake_features: List[np.ndarray] = []

    def update_real(self, scans: Union[torch.Tensor, np.ndarray], batch_size: int = 32) -> None:
        feat = extract_features(scans, self.extractor, batch_size=batch_size, device=self.device)
        self.real_features.append(feat)

    def update_fake(self, scans: Union[torch.Tensor, np.ndarray], batch_size: int = 32) -> None:
        feat = extract_features(scans, self.extractor, batch_size=batch_size, device=self.device)
        self.fake_features.append(feat)

    def compute(self, eps: float = 1e-6) -> float:
        if not self.real_features or not self.fake_features:
            raise RuntimeError("Must provide both real and fake scans before calling compute()")
        all_real = np.concatenate(self.real_features, axis=0)
        all_fake = np.concatenate(self.fake_features, axis=0)
        return compute_fid_from_features(all_real, all_fake, eps=eps)

    def reset(self) -> None:
        self.real_features.clear()
        self.fake_features.clear()


class FVDCalculator:
    """
    Streaming accumulator for computing 3D FVD across volumetric OCT cohorts.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.extractor = feature_extractor or Volume3DFeatureExtractor(device=device)
        self.device = device
        self.real_features: List[np.ndarray] = []
        self.fake_features: List[np.ndarray] = []

    def update_real(self, volumes: Union[torch.Tensor, np.ndarray], batch_size: int = 16) -> None:
        feat = extract_features(volumes, self.extractor, batch_size=batch_size, device=self.device)
        self.real_features.append(feat)

    def update_fake(self, volumes: Union[torch.Tensor, np.ndarray], batch_size: int = 16) -> None:
        feat = extract_features(volumes, self.extractor, batch_size=batch_size, device=self.device)
        self.fake_features.append(feat)

    def compute(self, eps: float = 1e-6) -> float:
        if not self.real_features or not self.fake_features:
            raise RuntimeError("Must provide both real and fake volumes before calling compute()")
        all_real = np.concatenate(self.real_features, axis=0)
        all_fake = np.concatenate(self.fake_features, axis=0)
        return compute_fvd_from_features(all_real, all_fake, eps=eps)

    def reset(self) -> None:
        self.real_features.clear()
        self.fake_features.clear()


# =====================================================================
# 6. UNIFIED GENERATION EVALUATION WRAPPER
# =====================================================================

def evaluate_generative_distributions(
    real_volumes: Union[torch.Tensor, np.ndarray],
    fake_volumes: Union[torch.Tensor, np.ndarray],
    compute_slice_fid_flag: bool = True,
    extractor_2d: Optional[nn.Module] = None,
    extractor_3d: Optional[nn.Module] = None,
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Unified evaluation function executing:
    - 2D B-scan FID across pooled volume slices.
    - 3D Volumetric FVD across spatio-temporal sequences.
    - Slice-wise depth FID profile (if enabled).

    Args:
        real_volumes: Authentic OCT volumes, shape (N, 1, D, H, W) or (N, D, H, W).
        fake_volumes: Synthetic OCT volumes, shape (M, 1, D, H, W) or (M, D, H, W).
        compute_slice_fid_flag: Whether to calculate per-slice FID depth progression.
        extractor_2d: Optional custom 2D feature extractor.
        extractor_3d: Optional custom 3D volume extractor.
        batch_size: Batch size for feature extraction.
        device: Target execution device.
        eps: Regularization factor.

    Returns:
        Dict containing "fid", "fvd", "num_real_volumes", "num_fake_volumes",
        "total_b_scans_evaluated", and optionally "slice_wise_fid".
    """
    if isinstance(real_volumes, np.ndarray):
        real_volumes = torch.from_numpy(real_volumes)
    if isinstance(fake_volumes, np.ndarray):
        fake_volumes = torch.from_numpy(fake_volumes)

    # Standardize to (N, D, H, W)
    if real_volumes.ndim == 5 and real_volumes.shape[1] == 1:
        real_volumes_4d = real_volumes.squeeze(1)
    else:
        real_volumes_4d = real_volumes

    if fake_volumes.ndim == 5 and fake_volumes.shape[1] == 1:
        fake_volumes_4d = fake_volumes.squeeze(1)
    else:
        fake_volumes_4d = fake_volumes

    n_real, d_real, h_real, w_real = real_volumes_4d.shape
    n_fake, d_fake, h_fake, w_fake = fake_volumes_4d.shape

    # 1. Flatten volumes into pools of individual 2D B-scans for 2D FID
    # (N, D, H, W) -> (N * D, 1, H, W)
    real_b_scans = real_volumes_4d.reshape(-1, 1, h_real, w_real)
    fake_b_scans = fake_volumes_4d.reshape(-1, 1, h_fake, w_fake)

    logger.info("Computing 2D FID on %d real vs %d fake B-scans...", real_b_scans.shape[0], fake_b_scans.shape[0])
    if extractor_2d is None:
        extractor_2d = InceptionV3FeatureExtractor(device=device)

    feat_real_bscans = extract_features(real_b_scans, extractor_2d, batch_size=batch_size * 2, device=device)
    feat_fake_bscans = extract_features(fake_b_scans, extractor_2d, batch_size=batch_size * 2, device=device)
    fid_val = compute_fid_from_features(feat_real_bscans, feat_fake_bscans, eps=eps)

    # 2. Compute 3D Volumetric FVD
    logger.info("Computing 3D FVD on %d real vs %d fake OCT volumes...", n_real, n_fake)
    fvd_val = compute_fvd(
        real_volumes_4d,
        fake_volumes_4d,
        feature_extractor=extractor_3d,
        batch_size=batch_size,
        device=device,
        eps=eps,
    )

    results: Dict[str, Any] = {
        "fid": float(fid_val),
        "fvd": float(fvd_val),
        "num_real_volumes": int(n_real),
        "num_fake_volumes": int(n_fake),
        "num_real_bscans": int(real_b_scans.shape[0]),
        "num_fake_bscans": int(fake_b_scans.shape[0]),
        "volume_depth": int(min(d_real, d_fake)),
    }

    # 3. Slice-wise FID Profiling
    if compute_slice_fid_flag:
        logger.info("Computing depth slice-wise FID across %d slices...", results["volume_depth"])
        feat_r_by_slice = feat_real_bscans.reshape(n_real, d_real, -1)
        feat_f_by_slice = feat_fake_bscans.reshape(n_fake, d_fake, -1)
        slice_res = compute_slice_wise_fid(
            real_volumes_4d,
            fake_volumes_4d,
            feature_extractor=extractor_2d,
            batch_size=batch_size * 2,
            device=device,
            eps=eps,
            precomputed_features_real=feat_r_by_slice,
            precomputed_features_fake=feat_f_by_slice,
        )
        results["slice_wise_fid"] = slice_res

    return results
