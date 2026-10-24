"""
src/fundus2oct/inference/generate_oct.py

High-level inference module for synthetic OCT generation.
Exposes generate_oct_volume and CLI entry points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from fundus2oct.config import AppConfig, load_config
from fundus2oct.models.pipeline import Fundus2OCTPipeline, GeneratorNotReadyError
from fundus2oct.data.auto_capture import create_demo_fundus


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
    """
    Convenience functional interface to synthesize a 32-slice 3D OCT volume
    from a 2D fundus image via LVDM reverse diffusion.

    Args:
        fundus: Retinal fundus image or file path
        pipeline: Optional pre-loaded Fundus2OCTPipeline instance
        num_slices: Number of B-scan slices in depth (default: 32)
        steps: DDIM reverse diffusion sampling steps
        guidance_scale: Classifier-free guidance scale
        seed: Deterministic random seed
        device: Execution device ('auto', 'cuda', 'cpu')
        allow_untrained: Permit sampling with initialized diffusion model
        allow_smoke_test: Permit zero-tensor smoke test fallback
        return_numpy: Return volume as (32, 256, 256) uint8 array
    Returns:
        Dictionary with 'volume', 'slices', 'fundus_preprocessed', 'metadata'
    """
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


__all__ = [
    "generate_oct_volume",
    "Fundus2OCTPipeline",
    "GeneratorNotReadyError",
]
