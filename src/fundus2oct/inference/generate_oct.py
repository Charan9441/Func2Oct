"""
src/fundus2oct/inference/generate_oct.py

High-level inference and batch generation module for synthetic OCT volumes.
Exposes generate_oct_volume, batch_generate_synthetic_oct, save_nifti_volume, and save_tiff_volume.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from fundus2oct.config import AppConfig, load_config
from fundus2oct.models.pipeline import Fundus2OCTPipeline, GeneratorNotReadyError
from fundus2oct.data.auto_capture import create_demo_fundus

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    nib = None
    HAS_NIBABEL = False


def save_nifti_volume(
    volume_np: np.ndarray,
    output_path: Path,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Saves a 3D OCT volume array (Depth, Height, Width) into standard NIfTI (.nii.gz)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.name.endswith((".nii", ".nii.gz")):
        output_path = output_path.with_suffix(".nii.gz")

    d, h, w = volume_np.shape
    nii_data = np.transpose(volume_np, (2, 1, 0))  # (W, H, D)

    if HAS_NIBABEL:
        affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
        nii_img = nib.Nifti1Image(nii_data.astype(np.uint8), affine)
        nii_img.header.set_xyzt_units("mm")
        nib.save(nii_img, str(output_path))
    else:
        np.savez_compressed(output_path.with_suffix(".npz"), volume=volume_np)
        output_path = output_path.with_suffix(".npz")

    return output_path


def save_tiff_volume(
    volume_np: np.ndarray,
    output_path: Path,
) -> Path:
    """Saves a 3D OCT volume array (Depth, Height, Width) as a multi-page TIFF stack (.tif)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.suffix.lower() in (".tif", ".tiff"):
        output_path = output_path.with_suffix(".tif")

    num_slices = volume_np.shape[0]
    pil_slices = [Image.fromarray(volume_np[i]) for i in range(num_slices)]
    pil_slices[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_slices[1:],
    )
    return output_path


def generate_oct_volume(
    fundus: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    pipeline: Optional[Fundus2OCTPipeline] = None,
    num_slices: int = 32,
    steps: int = 50,
    guidance_scale: float = 2.5,
    seed: int = 42,
    device: str = "auto",
    allow_untrained: bool = True,
    allow_smoke_test: bool = False,
    return_numpy: bool = True,
) -> Dict[str, Any]:
    """Convenience functional interface to synthesize a 32-slice 3D OCT volume."""
    if pipeline is None:
        cfg = load_config()
        pipeline = Fundus2OCTPipeline(config=cfg, device=device)

    return pipeline.generate_volume(
        fundus=fundus,
        num_slices=num_slices,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        allow_untrained=allow_untrained,
        allow_smoke_test=allow_smoke_test,
        return_numpy=return_numpy,
    )


def batch_generate_synthetic_oct(
    pipeline: Optional[Fundus2OCTPipeline] = None,
    count: int = 500,
    output_dir: Union[str, Path] = "outputs/synthetic_oct",
    num_slices: int = 32,
    formats: Tuple[str, ...] = ("nifti", "tiff"),
    steps: int = 2,
    guidance_scale: float = 2.5,
    seed: int = 42,
    device: str = "auto",
    allow_untrained: bool = True,
    save_fundus_paired: bool = True,
) -> Path:
    """Batch generates paired 3D OCT volumes and saves structured NIfTI and TIFF formats."""
    if pipeline is None:
        cfg = load_config()
        pipeline = Fundus2OCTPipeline(config=cfg, device=device)

    from generate_oct import batch_generate_synthetic_oct as _batch_gen
    return _batch_gen(
        pipeline=pipeline,
        count=count,
        output_dir=output_dir,
        num_slices=num_slices,
        formats=formats,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        allow_untrained=allow_untrained,
        save_fundus_paired=save_fundus_paired,
    )


__all__ = [
    "generate_oct_volume",
    "batch_generate_synthetic_oct",
    "save_nifti_volume",
    "save_tiff_volume",
    "Fundus2OCTPipeline",
    "GeneratorNotReadyError",
]
