"""
src/fundus2oct/evaluation/uncertainty.py

Predictive Uncertainty Estimation using Monte Carlo Dropout (MC Dropout)
for 2D Fundus and Multimodal Retinal Disease Classifiers.

Provides epistemic (model) and aleatoric (data) uncertainty quantification:
- Mean predictive probability: \\bar{p}(y|x) = (1/T) \\sum_{t=1}^T p_t(y|x)
- Predictive entropy (Total Uncertainty): H(\\bar{p}) = -\\sum_c \\bar{p}_c \\log(\\bar{p}_c)
- Expected entropy (Aleatoric Uncertainty): E[H(p_t)] = (1/T) \\sum_t H(p_t)
- Mutual Information (Epistemic Uncertainty): MI = H(\\bar{p}) - E[H(p_t)]
- High-uncertainty flagging based on configurable thresholds.

References:
- Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", ICML 2016.
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoDropoutLayersError(ValueError):
    """Raised when Monte Carlo Dropout is requested but the model contains no dropout modules."""
    pass


@dataclass
class UncertaintyResult:
    """
    Structured outcome of Monte Carlo Dropout predictive uncertainty evaluation.

    Attributes:
        mean_probabilities: Mean predictive probabilities across stochastic passes (C,).
        predicted_class: Argmax class index based on mean predictive probabilities.
        predictive_entropy_nats: Total predictive entropy H(\\bar{p}) in nats (natural logarithm).
        predictive_entropy_bits: Total predictive entropy in bits (base-2 logarithm).
        expected_entropy_nats: Expected entropy E[H(p_t)] (aleatoric uncertainty) in nats.
        mutual_information_nats: Mutual information MI = H(\\bar{p}) - E[H(p_t)] (epistemic uncertainty).
        variance_per_class: Empirical variance of class probabilities across stochastic passes (C,).
        uncertainty_score: Scalar metric selected for threshold evaluation.
        uncertainty_metric: Identifier of the metric used for scoring (e.g. 'predictive_entropy').
        threshold: Configured threshold applied for the uncertainty flag (or None).
        is_high_uncertainty: Boolean flag indicating if uncertainty_score exceeds threshold.
        num_passes: Number of stochastic MC Dropout passes executed.
        stochastic_probabilities: Array of individual pass probabilities (T, C).
        metadata: Model and execution telemetry.
    """
    mean_probabilities: np.ndarray
    predicted_class: int
    predictive_entropy_nats: float
    predictive_entropy_bits: float
    expected_entropy_nats: float
    mutual_information_nats: float
    variance_per_class: np.ndarray
    uncertainty_score: float
    uncertainty_metric: str = "predictive_entropy"
    threshold: Optional[float] = None
    is_high_uncertainty: bool = False
    num_passes: int = 15
    stochastic_probabilities: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_passes: bool = False) -> Dict[str, Any]:
        """Converts result to a clean, JSON-serializable dictionary."""
        d = {
            "predicted_class": int(self.predicted_class),
            "mean_probabilities": [float(p) for p in self.mean_probabilities],
            "predictive_entropy_nats": float(self.predictive_entropy_nats),
            "predictive_entropy_bits": float(self.predictive_entropy_bits),
            "expected_entropy_nats": float(self.expected_entropy_nats),
            "mutual_information_nats": float(self.mutual_information_nats),
            "variance_per_class": [float(v) for v in self.variance_per_class],
            "uncertainty_score": float(self.uncertainty_score),
            "uncertainty_metric": str(self.uncertainty_metric),
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "is_high_uncertainty": bool(self.is_high_uncertainty),
            "num_passes": int(self.num_passes),
            "metadata": self.metadata,
        }
        if include_passes and self.stochastic_probabilities is not None:
            d["stochastic_probabilities"] = self.stochastic_probabilities.tolist()
        return d

    def summary_text(self) -> str:
        """Formats a human-readable diagnostic summary with clinical disclaimer."""
        flag_badge = "⚠️ HIGH UNCERTAINTY" if self.is_high_uncertainty else "🟢 LOW UNCERTAINTY"
        thresh_str = f"{self.threshold:.4f}" if self.threshold is not None else "None (unthresholded)"
        summary = (
            f"### Predictive Uncertainty Assessment: {flag_badge}\n\n"
            f"- **Predicted Class**: `{self.predicted_class}` "
            f"(Confidence: `{self.mean_probabilities[self.predicted_class] * 100:.1f}%`)\n"
            f"- **Uncertainty Score ({self.uncertainty_metric})**: `{self.uncertainty_score:.4f}`\n"
            f"- **Decision Threshold**: `{thresh_str}`\n"
            f"- **Total Entropy H(p)**: `{self.predictive_entropy_nats:.4f} nats` (`{self.predictive_entropy_bits:.4f} bits`)\n"
            f"- **Epistemic Uncertainty (MI)**: `{self.mutual_information_nats:.4f} nats`\n"
            f"- **Aleatoric Uncertainty (E[H])**: `{self.expected_entropy_nats:.4f} nats`\n"
            f"- **Stochastic MC Passes**: `{self.num_passes}`\n\n"
            f"> [!NOTE]\n"
            f"> **Clinical Disclaimer**: Uncertainty metrics represent research model variance and entropy "
            f"under Monte Carlo Dropout. They do not constitute clinically calibrated error probabilities."
        )
        return summary


def discover_dropout_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Discovers all active dropout modules within a model.

    Detects leaf nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, and custom dropout
    modules with p > 0, strictly excluding root containers or non-dropout parent models.
    """
    dropout_modules: List[Tuple[str, nn.Module]] = []
    dropout_types = (
        nn.Dropout,
        getattr(nn, "Dropout1d", nn.Dropout),
        nn.Dropout2d,
        nn.Dropout3d,
    )
    for name, module in model.named_modules():
        # The top-level model container itself is never a dropout module
        if module is model:
            continue

        is_standard_dropout = isinstance(module, dropout_types)
        cls_name = module.__class__.__name__
        is_leaf = len(list(module.children())) == 0
        p = getattr(module, "p", None)

        is_custom_dropout = (
            is_leaf
            and "dropout" in cls_name.lower()
            and isinstance(p, (float, int))
            and not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict))
        )

        if is_standard_dropout or is_custom_dropout:
            if p is None or float(p) > 0.0:
                dropout_modules.append((name, module))

    return dropout_modules



class MonteCarloDropoutEstimator:
    """
    Monte Carlo Dropout Predictive Uncertainty Estimator.

    Executes stochastic forward inference by keeping dropout active while maintaining
    evaluation mode for normalization layers (BatchNorm, GroupNorm, LayerNorm).
    """

    def __init__(
        self,
        model: nn.Module,
        num_passes: int = 15,
        threshold: Optional[float] = None,
        uncertainty_metric: str = "predictive_entropy",
        allow_deterministic_fallback: bool = False,
    ):
        """
        Args:
            model: PyTorch classifier (e.g. FundusClassifier or MultimodalClassifier).
            num_passes: Number of stochastic forward passes (must be >= 1, recommended >= 10).
            threshold: Optional decision boundary threshold for high-uncertainty flag.
            uncertainty_metric: Metric for thresholding ("predictive_entropy", "mutual_information",
                                "expected_entropy", or "variance").
            allow_deterministic_fallback: If True, warns rather than raising error when no dropout layers exist.
        """
        if num_passes < 1:
            raise ValueError(f"num_passes must be at least 1, got {num_passes}")

        valid_metrics = ("predictive_entropy", "mutual_information", "expected_entropy", "variance")
        if uncertainty_metric not in valid_metrics:
            raise ValueError(f"uncertainty_metric must be one of {valid_metrics}, got '{uncertainty_metric}'")

        self.model = model
        self.num_passes = num_passes
        self.threshold = threshold
        self.uncertainty_metric = uncertainty_metric
        self.allow_deterministic_fallback = allow_deterministic_fallback

        # Audit available dropout layers
        self.dropout_modules = discover_dropout_modules(self.model)
        if len(self.dropout_modules) == 0:
            msg = (
                f"Model '{model.__class__.__name__}' contains no dropout modules with p > 0. "
                "Monte Carlo Dropout requires stochastic dropout layers to measure model uncertainty."
            )
            if not self.allow_deterministic_fallback:
                raise NoDropoutLayersError(msg)

    def estimate(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        num_passes: Optional[int] = None,
        threshold: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        eps: float = 1e-8,
    ) -> UncertaintyResult:
        """
        Executes Monte Carlo Dropout inference and computes epistemic/aleatoric uncertainty.

        Args:
            inputs: Tensor of shape (1, C, H, W) or (B, C, H, W), or tuple/dict of tensors for multimodal models.
            num_passes: Override number of stochastic passes (default: self.num_passes).
            threshold: Override uncertainty threshold (default: self.threshold).
            seed: Optional random seed for reproducible dropout sampling.
            device: Optional torch.device to move inputs to.
            eps: Numerical stability constant to prevent log(0).

        Returns:
            UncertaintyResult containing predictive probabilities, entropy, MI, and uncertainty flags.
        """
        t_passes = num_passes if num_passes is not None else self.num_passes
        applied_threshold = threshold if threshold is not None else self.threshold

        if t_passes < 1:
            raise ValueError(f"num_passes must be at least 1, got {t_passes}")

        # Store initial training state of all submodules
        module_states = {mod: mod.training for mod in self.model.modules()}
        was_model_training = self.model.training

        # Set model to eval mode (freezes BatchNorm/GroupNorm statistics)
        self.model.eval()

        # Reactivate ONLY the discovered dropout modules
        for _, drop_mod in self.dropout_modules:
            drop_mod.train(True)

        if seed is not None:
            torch.manual_seed(seed)

        pass_probs: List[torch.Tensor] = []

        try:
            # Prepare forward arguments
            forward_args, forward_kwargs = self._prepare_inputs(inputs, device=device)

            with torch.no_grad():
                for _ in range(t_passes):
                    logits = self.model(*forward_args, **forward_kwargs)
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError(f"Model forward output must be a torch.Tensor, got {type(logits)}")

                    if logits.ndim == 1:
                        logits = logits.unsqueeze(0)

                    # For binary or multiclass classification with >= 2 logits, use softmax
                    # For single logit binary, use sigmoid
                    if logits.shape[-1] == 1:
                        p1 = torch.sigmoid(logits)
                        probs = torch.cat([1.0 - p1, p1], dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)

                    pass_probs.append(probs.cpu())

        finally:
            # Restore all modules to their exact original states
            for mod, orig_state in module_states.items():
                mod.train(orig_state)
            self.model.train(was_model_training)

        # Shape: (T, B, C) -> For single sample (B=1): (T, C)
        stacked_probs = torch.stack(pass_probs, dim=0)  # (T, B, C)
        sample_probs = stacked_probs[:, 0, :].numpy()   # (T, C)

        # 1. Mean Predictive Probability: p_bar = (1/T) * sum_t p_t
        mean_probs = np.mean(sample_probs, axis=0)  # (C,)
        predicted_class = int(np.argmax(mean_probs))

        # 2. Predictive Entropy (Total Uncertainty): H(p_bar) = - sum_c p_bar_c * log(p_bar_c)
        clipped_mean = np.clip(mean_probs, eps, 1.0)
        predictive_entropy_nats = float(-np.sum(clipped_mean * np.log(clipped_mean)))
        predictive_entropy_bits = float(-np.sum(clipped_mean * (np.log(clipped_mean) / math.log(2.0))))

        # 3. Expected Entropy (Aleatoric Uncertainty): E[H(p_t)] = (1/T) * sum_t H(p_t)
        clipped_samples = np.clip(sample_probs, eps, 1.0)
        sample_entropies = -np.sum(clipped_samples * np.log(clipped_samples), axis=1)  # (T,)
        expected_entropy_nats = float(np.mean(sample_entropies))

        # 4. Mutual Information (Epistemic / Model Uncertainty): MI = H(p_bar) - E[H(p_t)]
        mutual_info_nats = max(0.0, float(predictive_entropy_nats - expected_entropy_nats))

        # 5. Empirical Variance across passes: Var(p_c)
        if t_passes > 1:
            var_per_class = np.var(sample_probs, axis=0, ddof=1)
        else:
            var_per_class = np.zeros_like(mean_probs)

        # Select score based on configured metric
        if self.uncertainty_metric == "predictive_entropy":
            uncertainty_score = predictive_entropy_nats
        elif self.uncertainty_metric == "mutual_information":
            uncertainty_score = mutual_info_nats
        elif self.uncertainty_metric == "expected_entropy":
            uncertainty_score = expected_entropy_nats
        elif self.uncertainty_metric == "variance":
            uncertainty_score = float(np.max(var_per_class))
        else:
            uncertainty_score = predictive_entropy_nats

        # Decision flag based on threshold
        is_high = False
        if applied_threshold is not None:
            is_high = bool(uncertainty_score > applied_threshold)

        return UncertaintyResult(
            mean_probabilities=mean_probs,
            predicted_class=predicted_class,
            predictive_entropy_nats=predictive_entropy_nats,
            predictive_entropy_bits=predictive_entropy_bits,
            expected_entropy_nats=expected_entropy_nats,
            mutual_information_nats=mutual_info_nats,
            variance_per_class=var_per_class,
            uncertainty_score=uncertainty_score,
            uncertainty_metric=self.uncertainty_metric,
            threshold=applied_threshold,
            is_high_uncertainty=is_high,
            num_passes=t_passes,
            stochastic_probabilities=sample_probs,
            metadata={
                "model_class": self.model.__class__.__name__,
                "num_dropout_layers": len(self.dropout_modules),
                "dropout_layer_names": [name for name, _ in self.dropout_modules],
                "deterministic_fallback_used": len(self.dropout_modules) == 0,
            },
        )

    def _prepare_inputs(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Standardizes input tensors and handles device transfers."""
        forward_args: List[torch.Tensor] = []
        forward_kwargs: Dict[str, torch.Tensor] = {}

        def to_dev(t: torch.Tensor) -> torch.Tensor:
            return t.to(device) if device is not None else t

        if isinstance(inputs, torch.Tensor):
            t = to_dev(inputs)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            forward_args.append(t)

        elif isinstance(inputs, (tuple, list)):
            if len(inputs) == 0:
                raise ValueError("Inputs tuple cannot be empty.")
            for item in inputs:
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"Tuple input elements must be torch.Tensor, got {type(item)}")
                t = to_dev(item)
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                forward_args.append(t)

        elif isinstance(inputs, dict):
            for k, item in inputs.items():
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"Dictionary input '{k}' must be torch.Tensor, got {type(item)}")
                t = to_dev(item)
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                forward_kwargs[k] = t

        else:
            raise TypeError(f"Unsupported inputs format: {type(inputs)}")

        return forward_args, forward_kwargs


def estimate_predictive_uncertainty(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    num_passes: int = 15,
    threshold: Optional[float] = None,
    uncertainty_metric: str = "predictive_entropy",
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    allow_deterministic_fallback: bool = False,
) -> UncertaintyResult:
    """
    Functional interface to estimate predictive uncertainty via Monte Carlo Dropout.

    Args:
        model: PyTorch classifier module.
        inputs: Input tensor(s) for the classifier.
        num_passes: Number of stochastic forward passes (default: 15).
        threshold: Optional cutoff threshold for the high-uncertainty flag.
        uncertainty_metric: Metric evaluated against threshold (default: 'predictive_entropy').
        seed: Optional reproducibility random seed.
        device: Optional target device for computation.
        allow_deterministic_fallback: Allow execution when model has no dropout layers.

    Returns:
        UncertaintyResult dataclass with epistemic/aleatoric uncertainty and flags.
    """
    estimator = MonteCarloDropoutEstimator(
        model=model,
        num_passes=num_passes,
        threshold=threshold,
        uncertainty_metric=uncertainty_metric,
        allow_deterministic_fallback=allow_deterministic_fallback,
    )
    return estimator.estimate(inputs=inputs, seed=seed, device=device)
