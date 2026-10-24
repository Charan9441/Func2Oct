"""
src/fundus2oct/evaluation/__init__.py

Evaluation package for Fundus2OCT models:
- Reconstruction metrics: MSE, MAE, PSNR, SSIM, and target benchmark tracking (MSE <= 0.0089, MAE <= 0.0587).
"""

from .metrics import (
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

__all__ = [
    "TARGET_VAE_MSE",
    "TARGET_VAE_MAE",
    "compute_mse",
    "compute_mae",
    "compute_l1",
    "compute_psnr",
    "compute_ssim",
    "compute_reconstruction_metrics",
    "evaluate_vae_benchmarks",
    "VAEMetricTracker",
]
