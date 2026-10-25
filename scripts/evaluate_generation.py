#!/usr/bin/env python3
"""
scripts/evaluate_generation.py

Quantitative Generative Benchmark Evaluation CLI for Retinal OCT Synthesis:
- Frechet Inception Distance (FID) on individual 2D B-scans.
- Frechet Video Distance (FVD) across 3D OCT volume slices.
- Depth-wise slice FID profiling (anterior to posterior volume slices).
- Paired reconstruction metrics (MSE, MAE, PSNR, SSIM) when reference OCT is available.
- Visual publication-quality diagnostic report generation.

Usage:
    # Run in demo / synthetic verification mode:
    python scripts/evaluate_generation.py --demo

    # Run on generated volumes vs authentic references:
    python scripts/evaluate_generation.py \\
        --synthetic-dir outputs/synthetic_oct \\
        --real-dir data/oct \\
        --output-dir outputs/evaluation_reports
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure src/ is on sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.data.preprocessing import (
    denormalize_to_numpy,
    preprocess_oct_bscan,
    preprocess_oct_volume,
)
from fundus2oct.evaluation.fid_fvd import (
    FIDCalculator,
    FVDCalculator,
    InceptionV3FeatureExtractor,
    SimpleFeatureExtractor2D,
    SimpleFeatureExtractor3D,
    Volume3DFeatureExtractor,
    compute_fid,
    compute_fvd,
    compute_slice_wise_fid,
    evaluate_generative_distributions,
)
from fundus2oct.evaluation.metrics import (
    TARGET_VAE_MAE,
    TARGET_VAE_MSE,
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_reconstruction_metrics,
    compute_ssim,
    evaluate_vae_benchmarks,
)

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    nib = None
    HAS_NIBABEL = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_generation")


# =====================================================================
# 1. VOLUME LOADING & PREPROCESSING UTILITIES
# =====================================================================

SUPPORTED_VOL_EXTS = {".npy", ".nii", ".nii.gz", ".tif", ".tiff"}
SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_volume_from_file(
    file_path: Union[str, Path],
    num_scans: int = 32,
    target_size: Tuple[int, int] = (256, 256),
) -> Optional[np.ndarray]:
    """
    Loads a 3D OCT volume from disk and standardizes it to shape (num_scans, H, W) in [0, 1].
    Supports NumPy (.npy), NIfTI (.nii, .nii.gz), multi-page TIFF (.tif, .tiff).
    """
    path = Path(file_path)
    if not path.exists():
        return None

    vol_array: Optional[np.ndarray] = None

    try:
        if path.suffix == ".npy":
            arr = np.load(path)
            vol_array = arr
        elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
            if HAS_NIBABEL and nib is not None:
                nii = nib.load(str(path))
                vol_array = np.asarray(nii.get_fdata(), dtype=np.float32)
                # NIfTI usually (H, W, D) -> transpose to (D, H, W)
                if vol_array.ndim == 3 and vol_array.shape[0] != num_scans and vol_array.shape[2] == num_scans:
                    vol_array = np.transpose(vol_array, (2, 0, 1))
            else:
                logger.warning("nibabel not installed; cannot load %s", path)
                return None
        elif path.suffix in (".tif", ".tiff"):
            # Multi-page TIFF
            with Image.open(path) as img:
                pages = []
                for i in range(getattr(img, "n_frames", 1)):
                    img.seek(i)
                    pages.append(np.array(img.convert("L")))
                vol_array = np.stack(pages, axis=0)
        else:
            return None
    except Exception as exc:
        logger.error("Failed to read volume %s: %s", path, exc)
        return None

    if vol_array is None:
        return None

    # Handle shape & squeeze excess singleton channels
    if vol_array.ndim == 4:
        if vol_array.shape[0] in (1, 3):
            vol_array = vol_array[0]  # (D, H, W)
        elif vol_array.shape[-1] in (1, 3):
            vol_array = vol_array[..., 0]

    # Preprocess via preprocess_oct_volume
    standardized = preprocess_oct_volume(
        vol_array,
        num_scans=num_scans,
        target_size=target_size,
    )
    if isinstance(standardized, torch.Tensor):
        standardized = standardized.cpu().numpy()

    # Normalize to [0.0, 1.0]
    min_v = float(standardized.min())
    max_v = float(standardized.max())
    if max_v > min_v:
        standardized = (standardized - min_v) / (max_v - min_v)
    else:
        standardized = np.zeros_like(standardized)

    return standardized.astype(np.float32)


def collect_volumes_from_directory(
    directory: Union[str, Path],
    num_scans: int = 32,
    max_samples: Optional[int] = None,
    target_size: Tuple[int, int] = (256, 256),
) -> List[Tuple[str, np.ndarray]]:
    """
    Discovers and loads 3D OCT volumes from a directory.
    Returns list of (identifier, volume_array) tuples with shape (num_scans, H, W).
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    collected: List[Tuple[str, np.ndarray]] = []

    # 1. Look for volume files (.npy, .nii, .nii.gz, .tif)
    candidates = sorted(
        [p for p in dir_path.iterdir() if any(p.name.endswith(ext) for ext in SUPPORTED_VOL_EXTS)]
    )

    for p in candidates:
        vol = load_volume_from_file(p, num_scans=num_scans, target_size=target_size)
        if vol is not None:
            collected.append((p.stem, vol))
            if max_samples and len(collected) >= max_samples:
                break

    # 2. If no volume files, look for subdirectories of 2D B-scans
    if not collected:
        subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
        for sd in subdirs:
            img_files = sorted(
                [f for f in sd.iterdir() if f.suffix.lower() in SUPPORTED_IMG_EXTS]
            )
            if img_files:
                slices = []
                for f in img_files:
                    try:
                        im = Image.open(f).convert("L")
                        slices.append(np.array(im))
                    except Exception:
                        continue
                if slices:
                    vol = preprocess_oct_volume(slices, num_scans=num_scans, target_size=target_size)
                    if isinstance(vol, torch.Tensor):
                        vol = vol.cpu().numpy()
                    min_v, max_v = float(vol.min()), float(vol.max())
                    if max_v > min_v:
                        vol = (vol - min_v) / (max_v - min_v)
                    collected.append((sd.name, vol.astype(np.float32)))
                    if max_samples and len(collected) >= max_samples:
                        break

    return collected


def generate_demo_volumes(
    count: int = 10,
    num_scans: int = 32,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic realistic demonstration OCT volume cohorts (real vs synthetic)
    simulating retinal layer structures, foveal depressions, and optical speckle.
    Used for unit testing, offline smoke tests, and demonstrative CLI runs.
    """
    rng = np.random.RandomState(seed)
    real_vols = np.zeros((count, num_scans, height, width), dtype=np.float32)
    fake_vols = np.zeros((count, num_scans, height, width), dtype=np.float32)

    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    for i in range(count):
        # Base foveal curvature
        fovea_depth = rng.uniform(10.0, 18.0)
        curvature = fovea_depth * np.exp(-((x_grid - width / 2.0) ** 2) / (2.0 * (width / 5.0) ** 2))

        for z in range(num_scans):
            # Depth attenuation factor (fovea in central slices)
            dist_from_center = abs(z - num_scans / 2.0) / (num_scans / 2.0)
            fovea_weight = max(0.0, 1.0 - dist_from_center * 0.8)

            # Layer definitions (ILM, IPL, INL, OPL, RPE, Choroid)
            rpe_y = int(height * 0.65 + rng.uniform(-3, 3))
            retina_y = rpe_y - (curvature * fovea_weight) - 30

            # Real scan synthesis
            real_scan = np.zeros((height, width), dtype=np.float32)
            # Retinal tissue reflection
            retina_mask = (y_grid >= retina_y) & (y_grid <= rpe_y)
            real_scan[retina_mask] = 0.35 + rng.normal(0.0, 0.05, size=retina_mask.sum())
            # Hyper-reflective RPE complex
            rpe_mask = (y_grid >= rpe_y - 4) & (y_grid <= rpe_y + 2)
            real_scan[rpe_mask] = 0.90 + rng.normal(0.0, 0.05, size=rpe_mask.sum())
            # Background optical speckle noise
            real_scan += rng.rayleigh(scale=0.08, size=(height, width))
            real_vols[i, z] = np.clip(real_scan, 0.0, 1.0)

            # Synthetic scan synthesis (reconstructed with mild blur / slight divergence)
            fake_scan = real_scan.copy()
            # Slight structural difference for synthetic generator
            fake_scan += rng.normal(0.0, 0.04, size=(height, width))
            fake_vols[i, z] = np.clip(fake_scan, 0.0, 1.0)

    return real_vols, fake_vols


# =====================================================================
# 2. PAIRED RECONSTRUCTION METRICS EVALUATION
# =====================================================================

def compute_paired_volume_metrics(
    real_volumes: np.ndarray,
    fake_volumes: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes paired pixel-level reconstruction fidelity metrics across matching volumes:
    MSE, MAE, PSNR (dB), and SSIM.
    Checks adherence to target clinical VAE benchmarks (MSE <= 0.0089, MAE <= 0.0587).
    """
    n_pairs = min(len(real_volumes), len(fake_volumes))
    if n_pairs == 0:
        return {}

    mse_list: List[float] = []
    mae_list: List[float] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []

    for idx in range(n_pairs):
        r_vol = torch.from_numpy(real_volumes[idx]).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        f_vol = torch.from_numpy(fake_volumes[idx]).unsqueeze(0).unsqueeze(0)

        # Map from [0, 1] to [-1, 1] to match VAE benchmark convention
        r_bipolar = r_vol * 2.0 - 1.0
        f_bipolar = f_vol * 2.0 - 1.0

        mse_val = compute_mse(f_bipolar, r_bipolar)
        mae_val = compute_mae(f_bipolar, r_bipolar)
        psnr_val = compute_psnr(f_bipolar, r_bipolar, data_range=2.0)
        ssim_val = compute_ssim(f_bipolar, r_bipolar, data_range=2.0)

        mse_list.append(mse_val)
        mae_list.append(mae_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    mean_mse = float(np.mean(mse_list))
    mean_mae = float(np.mean(mae_list))
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))

    # Benchmark checking against clinical targets
    benchmarks = evaluate_vae_benchmarks(mse=mean_mse, mae=mean_mae)

    return {
        "num_pairs_evaluated": n_pairs,
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "mean_psnr_db": mean_psnr,
        "mean_ssim": mean_ssim,
        "target_mse": TARGET_VAE_MSE,
        "target_mae": TARGET_VAE_MAE,
        "mse_passed": benchmarks["mse_passed"],
        "mae_passed": benchmarks["mae_passed"],
        "all_benchmarks_passed": benchmarks["all_passed"],
        "psnr_passed": bool(mean_psnr >= 30.0),
        "ssim_passed": bool(mean_ssim >= 0.85),
    }


# =====================================================================
# 3. REPORT EXPORTING & VISUALIZATION GENERATION
# =====================================================================

def plot_generative_evaluation_report(
    eval_results: Dict[str, Any],
    real_vol: Optional[np.ndarray],
    fake_vol: Optional[np.ndarray],
    output_png: Path,
) -> None:
    """
    Renders and saves a comprehensive, publication-quality 4-panel diagnostic figure:
    1. Top-Left: Generative Distribution Distances (FID vs FVD bar chart).
    2. Top-Right: Depth Slice-wise FID Profile (axial anatomical curve across slices).
    3. Bottom-Left: Reconstruction Fidelity & Target Benchmarks.
    4. Bottom-Right: Representative Real vs Synthetic B-scan cross-section comparison.
    """
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    # -------------------------------------------------------------
    # Panel 1: Generative Distribution Distances (FID & FVD)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    metrics_names = ["2D B-Scan FID", "3D Volumetric FVD"]
    fid_val = eval_results.get("fid", 0.0)
    fvd_val = eval_results.get("fvd", 0.0)
    values = [fid_val, fvd_val]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax1.bar(metrics_names, values, color=colors, width=0.45, edgecolor="black", linewidth=1.2, alpha=0.88)
    ax1.set_title("Distributional Generative Distance", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Fréchet Distance (Lower is Better)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 10)

    # -------------------------------------------------------------
    # Panel 2: Slice-wise FID Depth Profile
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    slice_data = eval_results.get("slice_wise_fid", {})
    per_slice = slice_data.get("per_slice", {})

    if per_slice:
        slices = sorted(per_slice.keys(), key=lambda k: int(k))
        scores = [per_slice[s] for s in slices]
        mean_s_fid = slice_data.get("mean_slice_fid", float(np.mean(scores)))

        ax2.plot(slices, scores, marker="o", color="#2ca02c", linewidth=2.2, label="Slice FID")
        ax2.axhline(mean_s_fid, color="red", linestyle="--", linewidth=1.5, label=f"Mean FID ({mean_s_fid:.2f})")
        ax2.set_title("Slice-wise FID Profile (Anterior -> Fovea -> Posterior)", fontsize=14, fontweight="bold", pad=12)
        ax2.set_xlabel("Volume Slice Index (Depth)", fontsize=11)
        ax2.set_ylabel("B-scan FID", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)
    else:
        ax2.text(0.5, 0.5, "Slice-wise FID profiling disabled\nor insufficient data", ha="center", va="center", fontsize=12)
        ax2.set_title("Slice-wise FID Profile", fontsize=14, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 3: Reconstruction Fidelity & Target Benchmarks
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    recon = eval_results.get("reconstruction", {})
    if recon:
        rec_metrics = ["PSNR (dB)", "SSIM x 50", "MAE x 500", "MSE x 2000"]
        psnr_v = recon.get("mean_psnr_db", 0.0)
        ssim_v = recon.get("mean_ssim", 0.0) * 50.0
        mae_v = recon.get("mean_mae", 0.0) * 500.0
        mse_v = recon.get("mean_mse", 0.0) * 2000.0
        rec_vals = [psnr_v, ssim_v, mae_v, mse_v]

        colors_rec = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        bars3 = ax3.bar(rec_metrics, rec_vals, color=colors_rec, width=0.5, edgecolor="black", alpha=0.85)
        ax3.set_title("Paired Reconstruction Fidelity & Benchmarks", fontsize=14, fontweight="bold", pad=12)
        ax3.set_ylabel("Scaled Value", fontsize=11)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotations
        labels_raw = [
            f"{recon.get('mean_psnr_db', 0):.1f} dB\n(Target >30)",
            f"{recon.get('mean_ssim', 0):.3f}\n(Target >0.85)",
            f"{recon.get('mean_mae', 0):.4f}\n(Target <{TARGET_VAE_MAE})",
            f"{recon.get('mean_mse', 0):.4f}\n(Target <{TARGET_VAE_MSE})",
        ]
        for bar, label in zip(bars3, labels_raw):
            h = bar.get_height()
            ax3.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax3.set_ylim(0, max(rec_vals) * 1.35 if rec_vals else 10)
    else:
        ax3.text(0.5, 0.5, "Unpaired Evaluation Mode\n(Reconstruction requires aligned pairs)", ha="center", va="center", fontsize=12)
        ax3.set_title("Reconstruction Metrics", fontsize=14, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Real vs Synthetic Central B-scan Comparison
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    if real_vol is not None and fake_vol is not None:
        c_idx = real_vol.shape[0] // 2
        real_slice = real_vol[c_idx]
        fake_slice = fake_vol[c_idx]

        # Combine side-by-side with dividing line
        h, w = real_slice.shape
        side_by_side = np.ones((h, w * 2 + 10), dtype=np.float32)
        side_by_side[:, :w] = real_slice
        side_by_side[:, w + 10 :] = fake_slice

        im4 = ax4.imshow(side_by_side, cmap="gray", aspect="auto")
        ax4.set_title(f"Central B-Scan: Real (Left) vs Synthetic (Right) [Slice #{c_idx}]", fontsize=13, fontweight="bold", pad=12)
        ax4.axis("off")
        cbar = plt.colorbar(im4, ax=ax4, fraction=0.03, pad=0.02)
        cbar.set_label("Normalized Reflectivity", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No volume images available to render", ha="center", va="center", fontsize=12)
        ax4.axis("off")

    fig.suptitle(
        f"Fundus2OCT Generative Distribution Evaluation Report\nEvaluated on {eval_results.get('num_real_volumes', 0)} Real vs {eval_results.get('num_fake_volumes', 0)} Synthetic Volumes",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved generative evaluation plot: %s", output_png)


def export_evaluation_artifacts(
    eval_results: Dict[str, Any],
    output_dir: Path,
    real_sample: Optional[np.ndarray] = None,
    fake_sample: Optional[np.ndarray] = None,
    save_plots: bool = True,
) -> Dict[str, Path]:
    """
    Exports structured JSON, CSV summaries, slice profile CSV, and PNG figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}

    # 1. JSON Report
    json_path = output_dir / "generation_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    artifacts["json"] = json_path
    logger.info("Saved JSON report: %s", json_path)

    # 2. Flat Summary CSV
    summary_data = {
        "metric": [
            "2D B-Scan FID",
            "3D Volumetric FVD",
            "Mean Slice-wise FID",
            "Min Slice-wise FID",
            "Max Slice-wise FID",
            "Real Volumes Count",
            "Synthetic Volumes Count",
            "Total Real B-Scans",
            "Total Synthetic B-Scans",
            "Volume Slices Evaluated",
        ],
        "value": [
            eval_results.get("fid", 0.0),
            eval_results.get("fvd", 0.0),
            eval_results.get("slice_wise_fid", {}).get("mean_slice_fid", 0.0),
            eval_results.get("slice_wise_fid", {}).get("min_slice_fid", 0.0),
            eval_results.get("slice_wise_fid", {}).get("max_slice_fid", 0.0),
            eval_results.get("num_real_volumes", 0),
            eval_results.get("num_fake_volumes", 0),
            eval_results.get("num_real_bscans", 0),
            eval_results.get("num_fake_bscans", 0),
            eval_results.get("volume_depth", 0),
        ],
    }

    # Include paired reconstruction if present
    recon = eval_results.get("reconstruction", {})
    if recon:
        summary_data["metric"].extend([
            "Reconstruction MSE",
            "Reconstruction MAE",
            "Reconstruction PSNR (dB)",
            "Reconstruction SSIM",
            "Target MSE Benchmark (<= 0.0089)",
            "Target MAE Benchmark (<= 0.0587)",
            "All VAE Benchmarks Passed",
        ])
        summary_data["value"].extend([
            recon.get("mean_mse", 0.0),
            recon.get("mean_mae", 0.0),
            recon.get("mean_psnr_db", 0.0),
            recon.get("mean_ssim", 0.0),
            recon.get("mse_passed", False),
            recon.get("mae_passed", False),
            recon.get("all_benchmarks_passed", False),
        ])

    csv_path = output_dir / "generation_evaluation.csv"
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    artifacts["summary_csv"] = csv_path
    logger.info("Saved summary CSV: %s", csv_path)

    # 3. Slice-wise Profile CSV
    per_slice = eval_results.get("slice_wise_fid", {}).get("per_slice", {})
    if per_slice:
        slice_df = pd.DataFrame([
            {"slice_index": int(k), "bscan_fid": float(v)}
            for k, v in sorted(per_slice.items(), key=lambda x: int(x[0]))
        ])
        slice_csv_path = output_dir / "slice_fid_profile.csv"
        slice_df.to_csv(slice_csv_path, index=False)
        artifacts["slice_csv"] = slice_csv_path
        logger.info("Saved slice-wise FID profile: %s", slice_csv_path)

    # 4. Diagnostic Plot
    if save_plots:
        png_path = output_dir / "generation_evaluation.png"
        plot_generative_evaluation_report(eval_results, real_sample, fake_sample, png_path)
        artifacts["plot"] = png_path

    return artifacts


# =====================================================================
# 4. CLI ARGUMENT PARSING & MAIN EXECUTION
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative Generative Benchmark Evaluation CLI (FID & FVD) for Retinal OCT Synthesis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default=str(project_root / "outputs" / "synthetic_oct"),
        help="Path to directory containing synthesized OCT volumes (.npy, .nii.gz, .tif).",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=str(project_root / "data" / "oct"),
        help="Path to directory containing reference authentic OCT volumes/scans.",
    )
    parser.add_argument(
        "--real-manifest",
        type=str,
        default=str(project_root / "data" / "manifests" / "dataset.csv"),
        help="Path to dataset CSV manifest with real OCT paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs" / "evaluation_reports"),
        help="Output directory to save evaluation metrics, CSVs, and plots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for feature extraction inference.",
    )
    parser.add_argument(
        "--num-scans",
        type=int,
        default=32,
        help="Number of B-scan slices per 3D volume (default: 32).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of volumes to evaluate per cohort (for rapid validation).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--compute-slice-fid",
        action="store_true",
        default=True,
        help="Compute depth-wise slice-by-slice FID profile.",
    )
    parser.add_argument(
        "--no-slice-fid",
        dest="compute_slice_fid",
        action="store_false",
        help="Disable depth-wise slice FID computation.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demonstration mode with synthesized realistic OCT volume pairs.",
    )
    parser.add_argument(
        "--use-lightweight-extractor",
        action="store_true",
        help="Use fast lightweight CNN feature extractors instead of full Inception/R3D.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating and saving matplotlib visualization charts.",
    )
    return parser.parse_args()


def print_summary_table(results: Dict[str, Any]) -> None:
    """Prints a polished ASCII evaluation summary table."""
    print("\n" + "=" * 76)
    print("      FUNDUS2OCT GENERATIVE MODEL DISTRIBUTION EVALUATION (FID / FVD)")
    print("=" * 76)
    print(f" Timestamp:                 {results.get('timestamp', 'N/A')}")
    print(f" Real Volumes Evaluated:    {results.get('num_real_volumes', 0)}")
    print(f" Synthetic Volumes:         {results.get('num_fake_volumes', 0)}")
    print(f" B-Scans Per Volume:        {results.get('volume_depth', 0)}")
    print(f" Total Real B-Scans:        {results.get('num_real_bscans', 0)}")
    print(f" Total Synthetic B-Scans:   {results.get('num_fake_bscans', 0)}")
    print("-" * 76)
    print(f" 2D B-Scan FID:             {results.get('fid', 0.0):.4f} (Lower is Better)")
    print(f" 3D Volumetric FVD:         {results.get('fvd', 0.0):.4f} (Lower is Better)")

    slice_data = results.get("slice_wise_fid", {})
    if slice_data:
        print(f" Mean Slice-wise FID:       {slice_data.get('mean_slice_fid', 0.0):.4f}")
        print(f" Best Slice (Min FID):      Slice #{slice_data.get('best_slice', 0)} ({slice_data.get('min_slice_fid', 0.0):.4f})")
        print(f" Worst Slice (Max FID):     Slice #{slice_data.get('worst_slice', 0)} ({slice_data.get('max_slice_fid', 0.0):.4f})")

    recon = results.get("reconstruction", {})
    if recon:
        print("-" * 76)
        print(" PAIRED RECONSTRUCTION FIDELITY (Target Tracking):")
        print(f"  MSE:                      {recon.get('mean_mse', 0.0):.6f}  [Target <= {TARGET_VAE_MSE}] -> {'PASSED' if recon.get('mse_passed') else 'FAILED'}")
        print(f"  MAE:                      {recon.get('mean_mae', 0.0):.6f}  [Target <= {TARGET_VAE_MAE}] -> {'PASSED' if recon.get('mae_passed') else 'FAILED'}")
        print(f"  PSNR:                     {recon.get('mean_psnr_db', 0.0):.2f} dB   [Target > 30.0 dB]  -> {'PASSED' if recon.get('psnr_passed') else 'FAILED'}")
        print(f"  SSIM:                     {recon.get('mean_ssim', 0.0):.4f}     [Target > 0.85]     -> {'PASSED' if recon.get('ssim_passed') else 'FAILED'}")
        print(f"  Overall Benchmarks:       {'ALL PASSED' if recon.get('all_benchmarks_passed') else 'ATTENTION NEEDED'}")

    print("=" * 76 + "\n")


def main() -> int:
    args = parse_args()

    # Load project configuration
    cfg = None
    if Path(args.config).exists():
        try:
            cfg = load_config(args.config)
        except Exception as exc:
            logger.warning("Could not load config file %s: %s", args.config, exc)

    # 1. Device Resolution
    if args.device == "auto":
        target_dev = getattr(getattr(cfg, "project", None), "device", "auto") if cfg else "auto"
        if target_dev == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = target_dev
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # 2. Check if running in demo mode or loading real/synthetic directories
    real_volumes: List[np.ndarray] = []
    fake_volumes: List[np.ndarray] = []
    is_demo = args.demo

    if not is_demo:
        logger.info("Scanning for synthesized OCT volumes in: %s", args.synthetic_dir)
        synth_collected = collect_volumes_from_directory(
            args.synthetic_dir,
            num_scans=args.num_scans,
            max_samples=args.max_samples,
        )
        fake_volumes = [v for _, v in synth_collected]

        logger.info("Scanning for authentic reference OCT volumes in: %s", args.real_dir)
        real_collected = collect_volumes_from_directory(
            args.real_dir,
            num_scans=args.num_scans,
            max_samples=args.max_samples,
        )
        real_volumes = [v for _, v in real_collected]

        if len(fake_volumes) == 0 or len(real_volumes) == 0:
            logger.warning(
                "Insufficient volumes found (Real: %d, Synthetic: %d). "
                "Switching automatically to demonstration mode (--demo).",
                len(real_volumes),
                len(fake_volumes),
            )
            is_demo = True

    if is_demo:
        sample_count = args.max_samples or 12
        logger.info("Executing evaluation in demonstration mode with %d synthetic paired volumes...", sample_count)
        real_arr, fake_arr = generate_demo_volumes(
            count=sample_count,
            num_scans=args.num_scans,
            seed=42,
        )
    else:
        real_arr = np.stack(real_volumes, axis=0)
        fake_arr = np.stack(fake_volumes, axis=0)

    # 3. Feature Extractor Setup
    extractor_2d: Optional[torch.nn.Module] = None
    extractor_3d: Optional[torch.nn.Module] = None

    if args.use_lightweight_extractor:
        logger.info("Using lightweight feature extractors (SimpleFeatureExtractor2D / 3D).")
        extractor_2d = SimpleFeatureExtractor2D(device=device)
        extractor_3d = SimpleFeatureExtractor3D(device=device)
    else:
        logger.info("Initializing InceptionV3 (2D) and R3D-18 (3D) feature extractors...")
        extractor_2d = InceptionV3FeatureExtractor(device=device)
        extractor_3d = Volume3DFeatureExtractor(device=device, target_slices=args.num_scans)

    # 4. Execution of Distributional Evaluation (FID & FVD)
    start_time = time.time()
    logger.info("Starting FID & FVD evaluation...")
    eval_results = evaluate_generative_distributions(
        real_volumes=real_arr,
        fake_volumes=fake_arr,
        compute_slice_fid_flag=args.compute_slice_fid,
        extractor_2d=extractor_2d,
        extractor_3d=extractor_3d,
        batch_size=args.batch_size,
        device=device,
    )
    elapsed = time.time() - start_time
    logger.info("Generative evaluation completed in %.2f seconds.", elapsed)

    # 5. Paired Reconstruction Evaluation
    # If shapes match and cohorts are paired (e.g. in demo mode or paired synthesis)
    if len(real_arr) == len(fake_arr):
        logger.info("Cohorts have matching sample counts; computing paired reconstruction metrics...")
        recon_results = compute_paired_volume_metrics(real_arr, fake_arr)
        eval_results["reconstruction"] = recon_results

    # Metadata
    eval_results["timestamp"] = datetime.now().isoformat()
    eval_results["device"] = str(device)
    eval_results["elapsed_seconds"] = round(elapsed, 2)
    eval_results["is_demo_mode"] = is_demo

    # 6. Export Reports & Plots
    output_dir = Path(args.output_dir)
    real_sample_vol = real_arr[0] if len(real_arr) > 0 else None
    fake_sample_vol = fake_arr[0] if len(fake_arr) > 0 else None

    artifacts = export_evaluation_artifacts(
        eval_results=eval_results,
        output_dir=output_dir,
        real_sample=real_sample_vol,
        fake_sample=fake_sample_vol,
        save_plots=not args.no_plots,
    )

    # 7. Print Console Summary
    print_summary_table(eval_results)

    logger.info("All evaluation artifacts saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
