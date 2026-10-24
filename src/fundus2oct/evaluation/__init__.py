"""
src/fundus2oct/evaluation/__init__.py

Evaluation, validation, and benchmarking package for Fund2Oct:
- Reconstruction metrics: MSE, MAE, PSNR, SSIM, and target benchmark tracking (MSE <= 0.0089, MAE <= 0.0587).
- Comparative disease classification: Model A (Fundus) vs. Model B (Multimodal), AUROC uplift (4-8% target).
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
from .compare_models import (
    compare_model_predictions,
    compute_paired_comparison,
    compute_single_model_metrics,
    export_comparison_reports,
    plot_auroc_uplift,
    plot_metrics_comparison,
    validate_inputs,
)

__all__ = [
    # Reconstruction
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
    # Model Comparison & AUROC Uplift
    "compare_model_predictions",
    "compute_paired_comparison",
    "compute_single_model_metrics",
    "export_comparison_reports",
    "plot_auroc_uplift",
    "plot_metrics_comparison",
    "validate_inputs",
]
