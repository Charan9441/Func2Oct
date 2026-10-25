"""
src/fundus2oct/evaluation/__init__.py

Evaluation, validation, and benchmarking package for Fund2Oct:
- Reconstruction metrics: MSE, MAE, PSNR, SSIM, and target benchmark tracking (MSE <= 0.0089, MAE <= 0.0587).
- Comparative disease classification: Model A (Fundus) vs. Model B (Multimodal), AUROC uplift (4-8% target).
- Generative fidelity metrics: Fréchet Inception Distance (FID), Fréchet Video/Volume Distance (FVD), Slice-wise depth FID.
- Model interpretability: Grad-CAM gradient-weighted class activation mapping and retinal layer attribution.
- Predictive uncertainty: Monte Carlo Dropout (epistemic/aleatoric uncertainty, entropy, mutual information, safety flags).
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
    # Generative Metrics: FID & FVD
    "compute_frechet_distance",
    "compute_activation_statistics",
    "compute_fid_from_features",
    "compute_fvd_from_features",
    "compute_fid",
    "compute_fvd",
    "compute_slice_wise_fid",
    "evaluate_generative_distributions",
    "InceptionV3FeatureExtractor",
    "Volume3DFeatureExtractor",
    "SimpleFeatureExtractor2D",
    "SimpleFeatureExtractor3D",
    "FIDCalculator",
    "FVDCalculator",
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
