"""
src/fundus2oct/evaluation/__init__.py

Evaluation, validation, and benchmarking package for Fund2Oct:
- Reconstruction metrics: MSE, MAE, PSNR, SSIM, and target benchmark tracking (MSE <= 0.0089, MAE <= 0.0587).
- Comparative disease classification: Model A (Fundus) vs. Model B (Multimodal), AUROC uplift (4-8% target).
- Generative fidelity metrics: Fréchet Inception Distance (FID), Fréchet Volume Distance (FVD), Axial Depth Profiles.
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
from .fid_fvd import (
    OCT2DFeatureExtractor,
    OCT3DVolumeFeatureExtractor,
    calculate_frechet_distance,
    compute_axial_depth_profile,
    compute_fid,
    compute_fvd,
    create_mock_oct_pairs,
    evaluate_depth_profile_fidelity,
    evaluate_paired_generation,
)
from .gradcam import (
    GradCAM,
    GradCAMResult,
    compute_gradcam,
    find_target_layer,
    overlay_heatmap_on_image,
    save_gradcam_artifacts,
)
from .uncertainty import (
    MonteCarloDropoutEstimator,
    NoDropoutLayersError,
    UncertaintyResult,
    discover_dropout_modules,
    estimate_predictive_uncertainty,
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
    # Generative Fidelity (FID/FVD/Depth Profile)
    "OCT2DFeatureExtractor",
    "OCT3DVolumeFeatureExtractor",
    "calculate_frechet_distance",
    "compute_axial_depth_profile",
    "compute_fid",
    "compute_fvd",
    "evaluate_depth_profile_fidelity",
    "evaluate_paired_generation",
    "create_mock_oct_pairs",
    # Interpretability & Grad-CAM
    "GradCAM",
    "GradCAMResult",
    "compute_gradcam",
    "find_target_layer",
    "overlay_heatmap_on_image",
    "save_gradcam_artifacts",
    # Predictive Uncertainty (MC Dropout)
    "MonteCarloDropoutEstimator",
    "UncertaintyResult",
    "estimate_predictive_uncertainty",
    "discover_dropout_modules",
    "NoDropoutLayersError",
]

