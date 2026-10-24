#!/usr/bin/env python3
"""
generate_oct.py

End-to-End Inference CLI for Cross-Modality Retinal OCT Volume Synthesis.
Accepts 2D color retinal fundus images and synthesizes coherent 32-slice 3D OCT volumes
via Latent Volumetric Diffusion Model (LVDM) reverse diffusion.

Usage:
    # Generate 32-slice 3D OCT volume from a fundus image:
    python generate_oct.py --input path/to/fundus.png --output outputs/synthetic_oct/volume_3d.npy

    # Run demonstration mode with synthetic fundus and animated GIF export:
    python generate_oct.py --demo --save_gif --output_dir outputs/synthetic_oct/demo_run

    # Fast inference (20 sampling steps):
    python generate_oct.py --demo --steps 20 --device auto --allow_untrained
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image
import torch

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fundus2oct.config import load_config
from fundus2oct.data.auto_capture import create_demo_fundus
from fundus2oct.models.pipeline import Fundus2OCTPipeline, GeneratorNotReadyError


def save_volume_outputs(
    volume_np: np.ndarray,
    output_path: Optional[Path],
    output_dir: Optional[Path],
    save_gif: bool = True,
    fps: int = 10,
) -> None:
    """Saves 3D OCT volume array, individual B-scan slice images, and animated sweep GIF."""
    num_slices, h, w = volume_np.shape

    # 1. Save 3D Volume as NumPy array or TIFF
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".npy":
            np.save(str(output_path), volume_np)
            print(f"[Export] Saved 3D OCT volume ({num_slices}x{h}x{w}) to: {output_path}")
        elif output_path.suffix.lower() == ".npz":
            np.savez_compressed(str(output_path), volume=volume_np)
            print(f"[Export] Saved compressed 3D OCT volume to: {output_path}")
        elif output_path.suffix.lower() in (".tif", ".tiff"):
            # Multi-page TIFF
            pil_slices = [Image.fromarray(volume_np[i]) for i in range(num_slices)]
            pil_slices[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_slices[1:],
            )
            print(f"[Export] Saved multi-page TIFF volume to: {output_path}")
        else:
            np.save(str(output_path), volume_np)
            print(f"[Export] Saved 3D OCT volume to: {output_path}")

    # 2. Save individual B-scan slices if output_dir specified
    if output_dir is not None:
        slices_dir = output_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_slices):
            slice_img = Image.fromarray(volume_np[i])
            slice_path = slices_dir / f"bscan_slice_{i:03d}.png"
            slice_img.save(slice_path)
        print(f"[Export] Exported {num_slices} individual B-scan slices to: {slices_dir}")

        # 3. Save animated sweep GIF
        if save_gif:
            gif_path = output_dir / "oct_volume_sweep.gif"
            pil_frames = [Image.fromarray(volume_np[i]) for i in range(num_slices)]
            # Loop forward and backward for smooth volumetric sweep
            loop_frames = pil_frames + pil_frames[::-1]
            loop_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=loop_frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            print(f"[Export] Saved animated volumetric sweep GIF to: {gif_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fundus2OCT: Generate 32-slice 3D OCT Volumes from 2D Fundus Images via LVDM Reverse Diffusion"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input 2D retinal fundus image (.png, .jpg, .tif)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a realistic synthetic fundus photograph for demonstration/testing",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Target output file path for 3D OCT volume (.npy, .npz, .tiff)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/synthetic_oct",
        help="Target directory to save individual B-scan slice images and sweep GIF",
    )
    parser.add_argument(
        "--num_slices", "-s",
        type=int,
        default=32,
        help="Number of B-scan slices in the generated 3D OCT volume (default: 32)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="DDIM reverse diffusion sampling steps (default: 50)",
    )
    parser.add_argument(
        "--guidance_scale", "-g",
        type=float,
        default=2.5,
        help="Classifier-Free Guidance (CFG) scale (default: 2.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', or 'cpu')",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        default=True,
        help="Export an animated depth-sweep GIF of the 32 B-scans",
    )
    parser.add_argument(
        "--allow_untrained",
        action="store_true",
        default=True,
        help="Allow generation with initialized architecture if diffusion checkpoint is untrained",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke-test fallback",
    )

    args = parser.parse_args()

    # Determine input fundus
    if args.demo or (args.input is None):
        print("[Info] Demo mode active: Generating synthetic retinal fundus photograph...")
        fundus_input = create_demo_fundus(blur_level=0.0, macula_dist_dd=2.7)
        input_desc = "Synthetic Demo Fundus (512x512)"
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[Error] Specified input image not found: {input_path}")
            sys.exit(1)
        fundus_input = input_path
        input_desc = str(input_path)

    # Initialize Fundus2OCT Pipeline
    print("[Pipeline] Initializing Fundus2OCT LVDM generative pipeline...")
    cfg = load_config()
    pipeline = Fundus2OCTPipeline(config=cfg, device=args.device)

    status = pipeline.get_status()
    print(f"[Status] Device: {status['device']}")
    print(f"[Status] OCT VAE Ready: {status['oct_vae_ready']}")
    print(f"[Status] Fundus Encoder Ready: {status['fundus_encoder_ready']}")
    print(f"[Status] Conditional Diffusion Ready: {status['diffusion_ready']}")

    # Setup output paths
    out_dir = Path(args.output_dir) if args.output_dir else None
    if args.output:
        out_path = Path(args.output)
    elif out_dir:
        out_path = out_dir / "volume_3d.npy"
    else:
        out_path = Path("outputs/synthetic_oct/volume_3d.npy")

    print(f"[Inference] Generating {args.num_slices}-slice 3D OCT volume from: {input_desc}")
    print(f"[Inference] Sampling Steps: {args.steps} | CFG Scale: {args.guidance_scale} | Seed: {args.seed}")

    try:
        t0 = time.time()
        result = pipeline.generate_volume(
            fundus=fundus_input,
            num_slices=args.num_slices,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            allow_smoke_test=args.smoke_test,
            allow_untrained=args.allow_untrained,
            return_numpy=True,
        )
        elapsed = time.time() - t0
        volume_np = result["volume"]
        print(f"[Success] Synthesized {volume_np.shape[0]} slices ({volume_np.shape[1]}x{volume_np.shape[2]}) in {elapsed:.2f}s!")

        # Export outputs
        save_volume_outputs(
            volume_np=volume_np,
            output_path=out_path,
            output_dir=out_dir,
            save_gif=args.save_gif,
        )

        print("[Done] End-to-end 3D OCT synthesis completed successfully.")

    except GeneratorNotReadyError as e:
        print(f"\n[BLOCKED BY INTEGRITY GUARD] {e}")
        print("To run inference for development/testing, pass --allow_untrained or --smoke_test.")
        sys.exit(2)


if __name__ == "__main__":
    main()
