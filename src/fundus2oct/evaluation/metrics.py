"""
src/fundus2oct/evaluation/metrics.py

Reconstruction evaluation metrics and benchmark validation tracking for OCT VAE / Autoencoder.
Monitors reconstruction fidelity against established clinical & model quality benchmarks:
- Target MSE <= 0.0089
- Target MAE <= 0.0587
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# =====================================================================
# Target VAE / Autoencoder Benchmarks
# =====================================================================
TARGET_VAE_MSE: float = 0.0089
TARGET_VAE_MAE: float = 0.0587


# =====================================================================
# Core Metric Functions
# =====================================================================

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Mean Squared Error (MSE) between prediction and ground truth.
    Both tensors are expected to be on the same scale (e.g. [-1.0, 1.0]).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in compute_mse: pred {pred.shape} vs target {target.shape}")
    return float(F.mse_loss(pred.float(), target.float(), reduction="mean").item())


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Mean Absolute Error (MAE / L1) between prediction and ground truth.
    Both tensors are expected to be on the same scale (e.g. [-1.0, 1.0]).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in compute_mae: pred {pred.shape} vs target {target.shape}")
    return float(F.l1_loss(pred.float(), target.float(), reduction="mean").item())


# Alias for L1
compute_l1 = compute_mae


def compute_psnr(
    pred: Union[torch.Tensor, float],
    target: Optional[torch.Tensor] = None,
    data_range: float = 2.0,
    eps: float = 1e-10,
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR in dB).
    If pred is a scalar float or int, it is treated directly as the MSE value.
    If pred and target are tensors, MSE is computed first.
    Default data_range=2.0 corresponds to normalized B-scans in [-1.0, 1.0].
    """
    if isinstance(pred, (int, float)):
        mse_val = float(pred)
    else:
        if target is None:
            raise ValueError("target must be provided when pred is a Tensor")
        mse_val = compute_mse(pred, target)

    if mse_val <= eps:
        return 100.0  # Numerical ceiling for near-perfect reconstruction
    return float(10.0 * math.log10((data_range ** 2) / mse_val))


def _gaussian_window_2d(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Creates a 2D Gaussian kernel tensor."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
    return kernel_2d.repeat(channels, 1, 1, 1)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM) on 2D images or 3D volumes.
    Expects tensors of shape (B, C, H, W) or (B, C, D, H, W).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in compute_ssim: pred {pred.shape} vs target {target.shape}")

    pred = pred.float()
    target = target.float()

    # Flatten 3D volume (B, C, D, H, W) -> (B*D, C, H, W) if needed
    if pred.dim() == 5:
        b, c, d, h, w = pred.shape
        pred = pred.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        target = target.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    channels = pred.shape[1]
    window = _gaussian_window_2d(window_size, sigma, channels).to(device=pred.device, dtype=pred.dtype)

    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def compute_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    compute_ssim_metric: bool = True,
) -> Dict[str, float]:
    """
    Computes complete reconstruction metric dictionary:
    - mse: Mean Squared Error
    - mae: Mean Absolute Error (L1)
    - psnr: Peak Signal-to-Noise Ratio (dB)
    - ssim: Structural Similarity (optional)
    """
    mse = compute_mse(pred, target)
    mae = compute_mae(pred, target)
    psnr = compute_psnr(mse, data_range=data_range)

    res = {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
    }
    if compute_ssim_metric:
        res["ssim"] = compute_ssim(pred, target, data_range=data_range)

    return res


# =====================================================================
# Target Benchmark Evaluation
# =====================================================================

def evaluate_vae_benchmarks(
    mse: float,
    mae: float,
    target_mse: float = TARGET_VAE_MSE,
    target_mae: float = TARGET_VAE_MAE,
) -> Dict[str, Any]:
    """
    Checks validation MSE and MAE against clinical target benchmarks:
    - MSE <= target_mse (default: 0.0089)
    - MAE <= target_mae (default: 0.0587)

    Returns detailed diagnostic dictionary and pass/fail statuses.
    """
    mse_passed = bool(mse <= target_mse)
    mae_passed = bool(mae <= target_mae)
    all_passed = bool(mse_passed and mae_passed)

    # Positive margin indicates beating the threshold (better than target)
    mse_margin = target_mse - mse
    mae_margin = target_mae - mae

    summary_parts = []
    summary_parts.append(f"MSE: {mse:.6f} <= {target_mse:.4f} [{'PASS' if mse_passed else 'FAIL'}]")
    summary_parts.append(f"MAE: {mae:.6f} <= {target_mae:.4f} [{'PASS' if mae_passed else 'FAIL'}]")
    overall_status = "BENCHMARKS MET" if all_passed else "BENCHMARKS PENDING"
    summary = f"{overall_status} | " + " | ".join(summary_parts)

    return {
        "target_mse": target_mse,
        "target_mae": target_mae,
        "mse": mse,
        "mae": mae,
        "mse_passed": mse_passed,
        "mae_passed": mae_passed,
        "all_passed": all_passed,
        "mse_margin": mse_margin,
        "mae_margin": mae_margin,
        "summary": summary,
    }


# =====================================================================
# Metric Tracker Class
# =====================================================================

class VAEMetricTracker:
    """
    Stateful metric accumulator for VAE validation and test loops.
    Tracks sample-weighted or batch-averaged MSE, MAE, PSNR, and SSIM,
    and provides benchmark compliance reporting.
    """

    def __init__(
        self,
        target_mse: float = TARGET_VAE_MSE,
        target_mae: float = TARGET_VAE_MAE,
        data_range: float = 2.0,
    ):
        self.target_mse = target_mse
        self.target_mae = target_mae
        self.data_range = data_range
        self.reset()

    def reset(self) -> None:
        """Resets all metric accumulators."""
        self.total_samples: int = 0
        self.total_batches: int = 0
        self.total_mse: float = 0.0
        self.total_mae: float = 0.0
        self.total_ssim: float = 0.0
        self._computed: bool = False

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_ssim_batch: bool = False,
    ) -> Dict[str, float]:
        """
        Updates running metric accumulators with a new batch.
        Returns the instantaneous metrics for the current batch.
        """
        batch_size = pred.shape[0]
        batch_mse = compute_mse(pred, target)
        batch_mae = compute_mae(pred, target)

        self.total_mse += batch_mse * batch_size
        self.total_mae += batch_mae * batch_size
        self.total_samples += batch_size
        self.total_batches += 1

        batch_metrics = {
            "mse": batch_mse,
            "mae": batch_mae,
            "psnr": compute_psnr(batch_mse, data_range=self.data_range),
        }

        if compute_ssim_batch:
            batch_ssim = compute_ssim(pred, target, data_range=self.data_range)
            self.total_ssim += batch_ssim * batch_size
            batch_metrics["ssim"] = batch_ssim

        return batch_metrics

    def compute(self) -> Dict[str, Any]:
        """
        Computes aggregate metrics over all accumulated samples and evaluates benchmarks.
        """
        if self.total_samples == 0:
            avg_mse = 0.0
            avg_mae = 0.0
            avg_psnr = 0.0
            avg_ssim = 0.0
        else:
            avg_mse = self.total_mse / self.total_samples
            avg_mae = self.total_mae / self.total_samples
            avg_psnr = compute_psnr(avg_mse, data_range=self.data_range)
            avg_ssim = (self.total_ssim / self.total_samples) if self.total_ssim > 0 else 0.0

        benchmarks = evaluate_vae_benchmarks(
            mse=avg_mse,
            mae=avg_mae,
            target_mse=self.target_mse,
            target_mae=self.target_mae,
        )

        return {
            "mse": avg_mse,
            "mae": avg_mae,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "benchmarks": benchmarks,
        }

    def format_summary(self) -> str:
        """Returns formatted string showing metrics and benchmark statuses."""
        metrics = self.compute()
        b = metrics["benchmarks"]
        status_tag = "[PASS - ALL BENCHMARKS MET]" if b["all_passed"] else "[PENDING - BENCHMARKS NOT MET]"

        return (
            f"VAE Validation Metrics ({metrics['total_samples']} samples):\n"
            f"  MSE:  {metrics['mse']:.6f} (Target: <= {self.target_mse:.4f}) -> {'PASS' if b['mse_passed'] else 'FAIL'}\n"
            f"  MAE:  {metrics['mae']:.6f} (Target: <= {self.target_mae:.4f}) -> {'PASS' if b['mae_passed'] else 'FAIL'}\n"
            f"  PSNR: {metrics['psnr']:.2f} dB\n"
            f"  Overall: {status_tag}"
        )
