"""
src/fundus2oct/config.py

Configuration dataclasses and YAML loading utilities for the Fundus2OCT architecture.
Provides structured access to preprocessing, model architectures, paths, and training parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import torch


@dataclass
class PreprocessingConfig:
    fundus_size: Tuple[int, int] = (256, 256)
    fundus_channels: int = 3
    preserve_fov: bool = True
    fov_mask_threshold: int = 15
    fundus_normalize: str = "imagenet"  # "imagenet", "unit", "bipolar"

    oct_size: Tuple[int, int] = (256, 256)
    oct_channels: int = 1
    volume_depth: int = 16
    oct_normalize: str = "bipolar"  # "bipolar" [-1, 1], "unit" [0, 1]
    invert_colormap: bool = False


@dataclass
class QualityControlConfig:
    min_resolution: Tuple[int, int] = (128, 128)
    blur_laplacian_threshold: float = 50.0
    min_contrast_std: float = 15.0
    underexposure_max_percent: float = 40.0
    overexposure_max_percent: float = 25.0


@dataclass
class VAEConfig:
    in_channels: int = 1
    latent_channels: int = 4
    base_channels: int = 32
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    is_3d: bool = False
    kl_weight: float = 1e-4
    learning_rate: float = 1e-4


@dataclass
class FundusEncoderConfig:
    backbone: str = "baseline_cnn"  # "baseline_cnn", "resnet50", "eyefound_stub"
    embed_dim: int = 512
    spatial_features: bool = True
    pretrained: bool = False


@dataclass
class DiffusionConfig:
    latent_channels: int = 4
    cond_dim: int = 512
    model_channels: int = 64
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4])
    timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sampling_steps: int = 50
    guidance_scale: float = 2.5
    is_3d: bool = False


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    fundus_dir: Path = Path("data/fundus")
    oct_dir: Path = Path("data/oct")
    manifest_path: Path = Path("data/manifests/dataset.csv")
    checkpoints_dir: Path = Path("checkpoints")
    outputs_dir: Path = Path("outputs")
    synthetic_oct_dir: Path = Path("outputs/synthetic_oct")
    evaluation_reports_dir: Path = Path("outputs/evaluation_reports")
    segmentation_masks_dir: Path = Path("outputs/segmentation_masks")


@dataclass
class CheckpointsConfig:
    oct_vae: Path = Path("checkpoints/oct_vae.pt")
    fundus_encoder: Path = Path("checkpoints/fundus_encoder.pt")
    diffusion: Path = Path("checkpoints/diffusion.pt")
    classifier_fundus: Path = Path("checkpoints/classifier_fundus.pt")
    classifier_multimodal: Path = Path("checkpoints/classifier_multimodal.pt")


@dataclass
class InferenceConfig:
    default_candidates: int = 3
    max_candidates: int = 5
    seed: int = 42
    guidance_scale: float = 2.5
    steps: int = 50
    allow_mock_fallback: bool = False


@dataclass
class InterpretabilityConfig:
    gradcam_target_layer: str = "auto"
    gradcam_alpha: float = 0.45
    gradcam_colormap: str = "jet"


@dataclass
class UncertaintyConfig:
    num_passes: int = 15
    metric: str = "predictive_entropy"
    threshold: Optional[float] = 0.50
    seed: int = 42


@dataclass
class EvaluationConfig:
    enable_fid: bool = False
    fid_min_samples: int = 50
    calculate_ssim_psnr: bool = True
    intensity_profile_bins: int = 64
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)



@dataclass
class ClassifierConfig:
    num_classes: int = 2
    backbone: str = "custom_cnn"  # "custom_cnn", "resnet18", "resnet34"
    feature_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 8


@dataclass
class AppConfig:
    project_name: str = "Fundus2OCT"
    version: str = "0.1.0"
    device: str = "auto"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    qc: QualityControlConfig = field(default_factory=QualityControlConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    fundus_encoder: FundusEncoderConfig = field(default_factory=FundusEncoderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    classifiers: ClassifierConfig = field(default_factory=ClassifierConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def resolve_device(self) -> torch.device:
        """Resolves target torch.device considering system capabilities and config."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")


def load_config(config_path: Union[str, Path] = "configs/default.yaml") -> AppConfig:
    """Loads configuration from YAML file or returns defaults if file not found."""
    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig()

    if "project" in raw:
        cfg.project_name = raw["project"].get("name", cfg.project_name)
        cfg.version = raw["project"].get("version", cfg.version)
        cfg.device = raw["project"].get("device", cfg.device)
        cfg.seed = raw["project"].get("seed", cfg.seed)

    if "paths" in raw:
        p = raw["paths"]
        cfg.paths.data_dir = Path(p.get("data_dir", cfg.paths.data_dir))
        cfg.paths.fundus_dir = Path(p.get("fundus_dir", cfg.paths.fundus_dir))
        cfg.paths.oct_dir = Path(p.get("oct_dir", cfg.paths.oct_dir))
        cfg.paths.manifest_path = Path(p.get("manifest_path", cfg.paths.manifest_path))
        cfg.paths.checkpoints_dir = Path(p.get("checkpoints_dir", cfg.paths.checkpoints_dir))
        cfg.paths.outputs_dir = Path(p.get("outputs_dir", cfg.paths.outputs_dir))
        cfg.paths.synthetic_oct_dir = Path(p.get("synthetic_oct_dir", cfg.paths.synthetic_oct_dir))
        cfg.paths.evaluation_reports_dir = Path(p.get("evaluation_reports_dir", cfg.paths.evaluation_reports_dir))
        cfg.paths.segmentation_masks_dir = Path(p.get("segmentation_masks_dir", cfg.paths.segmentation_masks_dir))

    if "checkpoints" in raw:
        cp = raw["checkpoints"]
        cfg.checkpoints.oct_vae = Path(cp.get("oct_vae", cfg.checkpoints.oct_vae))
        cfg.checkpoints.fundus_encoder = Path(cp.get("fundus_encoder", cfg.checkpoints.fundus_encoder))
        cfg.checkpoints.diffusion = Path(cp.get("diffusion", cfg.checkpoints.diffusion))
        cfg.checkpoints.classifier_fundus = Path(cp.get("classifier_fundus", cfg.checkpoints.classifier_fundus))
        cfg.checkpoints.classifier_multimodal = Path(cp.get("classifier_multimodal", cfg.checkpoints.classifier_multimodal))

    if "preprocessing" in raw:
        pre = raw["preprocessing"]
        if "fundus" in pre:
            f = pre["fundus"]
            cfg.preprocessing.fundus_size = tuple(f.get("image_size", cfg.preprocessing.fundus_size))
            cfg.preprocessing.fundus_channels = f.get("channels", cfg.preprocessing.fundus_channels)
            cfg.preprocessing.preserve_fov = f.get("preserve_fov", cfg.preprocessing.preserve_fov)
            cfg.preprocessing.fov_mask_threshold = f.get("fov_mask_threshold", cfg.preprocessing.fov_mask_threshold)
            cfg.preprocessing.fundus_normalize = f.get("normalize", cfg.preprocessing.fundus_normalize)
        if "oct" in pre:
            o = pre["oct"]
            cfg.preprocessing.oct_size = tuple(o.get("image_size", cfg.preprocessing.oct_size))
            cfg.preprocessing.oct_channels = o.get("channels", cfg.preprocessing.oct_channels)
            cfg.preprocessing.volume_depth = o.get("volume_depth", cfg.preprocessing.volume_depth)
            cfg.preprocessing.oct_normalize = o.get("normalize", cfg.preprocessing.oct_normalize)
            cfg.preprocessing.invert_colormap = o.get("invert_colormap", cfg.preprocessing.invert_colormap)

    if "quality_control" in raw:
        qc = raw["quality_control"]
        cfg.qc.min_resolution = tuple(qc.get("min_resolution", cfg.qc.min_resolution))
        cfg.qc.blur_laplacian_threshold = float(qc.get("blur_laplacian_threshold", cfg.qc.blur_laplacian_threshold))
        cfg.qc.min_contrast_std = float(qc.get("min_contrast_std", cfg.qc.min_contrast_std))
        cfg.qc.underexposure_max_percent = float(qc.get("underexposure_max_percent", cfg.qc.underexposure_max_percent))
        cfg.qc.overexposure_max_percent = float(qc.get("overexposure_max_percent", cfg.qc.overexposure_max_percent))

    if "models" in raw:
        m = raw["models"]
        if "oct_vae" in m:
            v = m["oct_vae"]
            cfg.vae.in_channels = v.get("in_channels", cfg.vae.in_channels)
            cfg.vae.latent_channels = v.get("latent_channels", cfg.vae.latent_channels)
            cfg.vae.base_channels = v.get("base_channels", cfg.vae.base_channels)
            cfg.vae.channel_multipliers = v.get("channel_multipliers", cfg.vae.channel_multipliers)
            cfg.vae.is_3d = v.get("is_3d", cfg.vae.is_3d)
            cfg.vae.kl_weight = float(v.get("kl_weight", cfg.vae.kl_weight))
            cfg.vae.learning_rate = float(v.get("learning_rate", cfg.vae.learning_rate))
        if "fundus_encoder" in m:
            fe = m["fundus_encoder"]
            cfg.fundus_encoder.backbone = fe.get("backbone", cfg.fundus_encoder.backbone)
            cfg.fundus_encoder.embed_dim = fe.get("embed_dim", cfg.fundus_encoder.embed_dim)
            cfg.fundus_encoder.spatial_features = fe.get("spatial_features", cfg.fundus_encoder.spatial_features)
        if "conditional_diffusion" in m:
            cd = m["conditional_diffusion"]
            cfg.diffusion.latent_channels = cd.get("latent_channels", cfg.diffusion.latent_channels)
            cfg.diffusion.cond_dim = cd.get("cond_dim", cfg.diffusion.cond_dim)
            cfg.diffusion.model_channels = cd.get("model_channels", cfg.diffusion.model_channels)
            cfg.diffusion.channel_multipliers = cd.get("channel_multipliers", cfg.diffusion.channel_multipliers)
            cfg.diffusion.timesteps = cd.get("timesteps", cfg.diffusion.timesteps)
            cfg.diffusion.beta_schedule = cd.get("beta_schedule", cfg.diffusion.beta_schedule)
            cfg.diffusion.beta_start = float(cd.get("beta_start", cfg.diffusion.beta_start))
            cfg.diffusion.beta_end = float(cd.get("beta_end", cfg.diffusion.beta_end))
            cfg.diffusion.sampling_steps = cd.get("sampling_steps", cfg.diffusion.sampling_steps)
            cfg.diffusion.guidance_scale = float(cd.get("guidance_scale", cfg.diffusion.guidance_scale))
            cfg.diffusion.is_3d = cd.get("is_3d", cfg.diffusion.is_3d)
        if "classifiers" in m:
            cl = m["classifiers"]
            cfg.classifiers.num_classes = cl.get("num_classes", cfg.classifiers.num_classes)
            cfg.classifiers.backbone = cl.get("backbone", cfg.classifiers.backbone)
            cfg.classifiers.feature_dim = cl.get("feature_dim", cfg.classifiers.feature_dim)
            cfg.classifiers.dropout = float(cl.get("dropout", cfg.classifiers.dropout))

    if "training" in raw:
        tr = raw["training"]
        cfg.classifiers.epochs = tr.get("epochs_classifier", cfg.classifiers.epochs)
        cfg.classifiers.batch_size = tr.get("batch_size", cfg.classifiers.batch_size)
        cfg.classifiers.weight_decay = float(tr.get("weight_decay", cfg.classifiers.weight_decay))

    if "inference" in raw:
        inf = raw["inference"]
        cfg.inference.default_candidates = inf.get("default_candidates", cfg.inference.default_candidates)
        cfg.inference.max_candidates = inf.get("max_candidates", cfg.inference.max_candidates)
        cfg.inference.seed = inf.get("seed", cfg.inference.seed)
        cfg.inference.guidance_scale = float(inf.get("guidance_scale", cfg.inference.guidance_scale))
        cfg.inference.steps = inf.get("steps", cfg.inference.steps)
        cfg.inference.allow_mock_fallback = inf.get("allow_mock_fallback", cfg.inference.allow_mock_fallback)

    if "evaluation" in raw:
        ev = raw["evaluation"]
        cfg.evaluation.enable_fid = ev.get("enable_fid", cfg.evaluation.enable_fid)
        cfg.evaluation.fid_min_samples = ev.get("fid_min_samples", cfg.evaluation.fid_min_samples)
        cfg.evaluation.calculate_ssim_psnr = ev.get("calculate_ssim_psnr", cfg.evaluation.calculate_ssim_psnr)
        cfg.evaluation.intensity_profile_bins = ev.get("intensity_profile_bins", cfg.evaluation.intensity_profile_bins)

        if "interpretability" in ev:
            interp = ev["interpretability"]
            cfg.evaluation.interpretability.gradcam_target_layer = interp.get(
                "gradcam_target_layer", cfg.evaluation.interpretability.gradcam_target_layer
            )
            cfg.evaluation.interpretability.gradcam_alpha = float(
                interp.get("gradcam_alpha", cfg.evaluation.interpretability.gradcam_alpha)
            )
            cfg.evaluation.interpretability.gradcam_colormap = interp.get(
                "gradcam_colormap", cfg.evaluation.interpretability.gradcam_colormap
            )

        if "uncertainty" in ev:
            unc = ev["uncertainty"]
            cfg.evaluation.uncertainty.num_passes = unc.get(
                "num_passes", cfg.evaluation.uncertainty.num_passes
            )
            cfg.evaluation.uncertainty.metric = unc.get(
                "metric", cfg.evaluation.uncertainty.metric
            )
            thresh = unc.get("threshold", cfg.evaluation.uncertainty.threshold)
            cfg.evaluation.uncertainty.threshold = float(thresh) if thresh is not None else None
            cfg.evaluation.uncertainty.seed = unc.get(
                "seed", cfg.evaluation.uncertainty.seed
            )

    return cfg
