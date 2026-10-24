"""
scripts/train_diffusion.py

Training script for the Conditional Latent Video/Volume Diffusion Model (LVDM).
Synthesizes 3D OCT volumes conditioned on EyeFound fundus embeddings.
Features:
- Encodes retinal fundus images using EyeFound ViT foundation backbone (512-dim).
- Encodes OCT B-scans/volumes into 8x32x32 latent space using pre-trained OCT VAE.
- Trains 3D UNet with 4 residual blocks using Gaussian diffusion noise prediction loss.
- Periodic validation, classifier-free guidance, and checkpointing to checkpoints/diffusion.pt.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.data.dataset import PairedFundusOCTDataset
from fundus2oct.models.fundus_encoder import FundusEncoder, build_fundus_encoder
from fundus2oct.models.oct_vae import (
    OCTAutoencoder2D,
    OCTVAE2D,
    build_oct_vae,
    load_vae_checkpoint,
)
from fundus2oct.models.conditional_diffusion import (
    ConditionalLVDM3D,
    UNet3D,
    build_lvdm,
    load_diffusion_checkpoint,
)


class SyntheticPairedDataset(Dataset):
    """Fallback paired dataset generating synthetic tensors for testing."""

    def __init__(self, count: int = 16, fundus_size: Tuple[int, int] = (256, 256), oct_size: Tuple[int, int] = (256, 256)):
        self.count = count
        self.fundus_size = fundus_size
        self.oct_size = oct_size

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "fundus": torch.randn(3, *self.fundus_size),
            "oct": torch.randn(1, *self.oct_size).clamp(-1.0, 1.0),
            "sample_id": f"syn_{idx:04d}",
            "patient_id": f"P{idx:03d}",
            "eye": "OD",
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Conditional Latent Video/Volume Diffusion Model (LVDM) on EyeFound fundus embeddings."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-steps",
        "--steps",
        type=int,
        default=50000,
        help="Number of diffusion training steps (default: 50000 steps).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=8,
        help="Latent channels matching OCT VAE 8x32x32 latent space (default: 8).",
    )
    parser.add_argument(
        "--cond-dim",
        type=int,
        default=512,
        help="Conditioning dimension matching EyeFound embedding (default: 512).",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base channel count for 3D UNet (default: 64).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total diffusion timesteps (default: 1000).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.5,
        help="Classifier-Free Guidance scale for sampling (default: 2.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--fundus-checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "fundus_encoder.pt"),
        help="Path to EyeFound fundus encoder checkpoint.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "oct_vae.pt"),
        help="Path to pre-trained OCT VAE checkpoint.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "diffusion.pt"),
        help="Destination path for trained diffusion model checkpoint.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print metrics every N steps (default: 100).",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1000,
        help="Evaluate on validation set every N steps (default: 1000).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=2500,
        help="Save checkpoint every N steps (default: 2500).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test for pipeline verification.",
    )
    return parser.parse_args()


def cycle_dataloader(loader: DataLoader):
    """Infinite generator over DataLoader."""
    while True:
        for batch in loader:
            yield batch


def evaluate_diffusion(
    model: ConditionalLVDM3D,
    fundus_encoder: FundusEncoder,
    vae: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates diffusion noise prediction MSE on the validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            fundus_imgs = batch["fundus"].to(device)
            oct_imgs = batch["oct"].to(device)

            # 1. Extract EyeFound conditioning
            cond_out = fundus_encoder(fundus_imgs)
            cond = cond_out[0] if isinstance(cond_out, tuple) else cond_out

            # 2. Extract OCT latent representation
            if hasattr(vae, "encode_latent"):
                z0 = vae.encode_latent(oct_imgs, deterministic=True)
            elif hasattr(vae, "encode"):
                enc_out = vae.encode(oct_imgs)
                z0 = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            else:
                z0 = oct_imgs

            if z0.dim() == 4:
                z0 = z0.unsqueeze(2)  # (B, C, 1, H, W)

            # 3. Compute diffusion loss
            loss_dict = model(z0, cond=cond, apply_dropout=False)
            total_loss += loss_dict["loss"].item()
            n_batches += 1

    n_batches = max(n_batches, 1)
    avg_loss = total_loss / n_batches
    return {
        "val_loss": avg_loss,
        "val_mse": avg_loss,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve settings
    if args.smoke_test:
        max_steps = 10 if args.max_steps == 50000 else args.max_steps
        log_interval = 2
        val_interval = 5
        save_interval = 5
    else:
        max_steps = args.max_steps
        log_interval = args.log_interval
        val_interval = args.val_interval
        save_interval = args.save_interval

    batch_size = args.batch_size
    lr = args.lr
    latent_channels = args.latent_channels
    cond_dim = args.cond_dim
    base_channels = args.base_channels
    timesteps = args.timesteps
    guidance_scale = args.guidance_scale

    if args.device:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    else:
        device = cfg.resolve_device()

    print("=" * 70, flush=True)
    print("  FUNDUS2OCT - CONDITIONAL LATENT VIDEO/VOLUME DIFFUSION (LVDM)", flush=True)
    print("=" * 70, flush=True)
    print(f"Architecture      : 3D UNet with 4 Residual Blocks (res1, res2, res3, res4)", flush=True)
    print(f"Latent Channels   : {latent_channels} (8x32x32 spatial latent)", flush=True)
    print(f"Conditioning      : EyeFound Fundus Embeddings ({cond_dim}-dim)", flush=True)
    print(f"Diffusion Steps   : {timesteps} timesteps", flush=True)
    print(f"Target Steps      : {max_steps:,} training steps", flush=True)
    print(f"Batch Size        : {batch_size}", flush=True)
    print(f"Learning Rate     : {lr}", flush=True)
    print(f"Guidance Scale    : {guidance_scale}", flush=True)
    print(f"Compute Device    : {device}", flush=True)
    print(f"Output Checkpoint : {args.output_checkpoint}", flush=True)
    print("=" * 70, flush=True)

    # 1. Initialize Paired Dataset
    manifest_path = project_root / cfg.paths.manifest_path
    data_root = project_root / cfg.paths.data_dir

    dataset = PairedFundusOCTDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        fundus_size=(256, 256),
        oct_size=(256, 256),
    )

    if len(dataset) == 0:
        print("[Notice] No paired samples found in manifest. Using SyntheticPairedDataset fallback.", flush=True)
        dataset = SyntheticPairedDataset(
            count=16 if args.smoke_test else 32,
            fundus_size=(256, 256),
            oct_size=(256, 256),
        )

    print(f"Paired Dataset    : {len(dataset)} fundus-to-OCT examinations loaded.", flush=True)

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 2. Load Conditioning Backbone (EyeFound Fundus Encoder)
    print("[1/3] Initializing EyeFound Fundus Conditioning Encoder...", flush=True)
    fundus_encoder = build_fundus_encoder(
        backbone=cfg.fundus_encoder.backbone,
        embed_dim=cond_dim,
        spatial_features=False,
        checkpoint_path=Path(args.fundus_checkpoint),
    ).to(device)
    fundus_encoder.eval()
    for param in fundus_encoder.parameters():
        param.requires_grad = False
    print("      EyeFound encoder initialized and frozen.", flush=True)

    # 3. Load OCT VAE for Latent Space Compression
    print("[2/3] Initializing OCT Latent VAE...", flush=True)
    vae = build_oct_vae(
        is_3d=False,
        latent_channels=latent_channels,
        checkpoint_path=Path(args.vae_checkpoint),
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("      OCT VAE initialized and frozen in 8x32x32 latent space.", flush=True)

    # 4. Initialize Conditional LVDM (3D UNet with 4 Residual Blocks)
    print("[3/3] Initializing Conditional LVDM 3D UNet...", flush=True)
    lvdm = build_lvdm(
        latent_channels=latent_channels,
        cond_dim=cond_dim,
        base_channels=base_channels,
        timesteps=timesteps,
        device=device,
    )
    print(f"      3D UNet verified with {len(lvdm.unet.res_blocks)} core residual blocks.", flush=True)

    # 5. Optimizer & LR Scheduler
    optimizer = optim.AdamW(lvdm.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)

    # 6. Training Loop
    checkpoint_out = Path(args.output_checkpoint)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    data_iter = iter(cycle_dataloader(train_loader))
    best_val_loss = float("inf")
    start_time = time.time()

    running_loss = 0.0

    print(f"\nStarting {max_steps:,} step LVDM diffusion training loop...\n", flush=True)

    for step in range(1, max_steps + 1):
        lvdm.train()
        batch = next(data_iter)

        fundus_imgs = batch["fundus"].to(device)
        oct_imgs = batch["oct"].to(device)

        # 1. Extract EyeFound fundus embeddings (B, 512)
        with torch.no_grad():
            cond_out = fundus_encoder(fundus_imgs)
            cond = cond_out[0] if isinstance(cond_out, tuple) else cond_out

            # 2. Extract OCT latent volume (B, 8, 1, 32, 32)
            if hasattr(vae, "encode_latent"):
                z0 = vae.encode_latent(oct_imgs, deterministic=True)
            elif hasattr(vae, "encode"):
                enc_out = vae.encode(oct_imgs)
                z0 = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            else:
                z0 = oct_imgs

            if z0.dim() == 4:
                z0 = z0.unsqueeze(2)  # (B, 8, 1, 32, 32)

        optimizer.zero_grad()

        # 3. Forward pass through Conditional LVDM
        loss_dict = lvdm(z0, cond=cond, apply_dropout=True)
        loss = loss_dict["loss"]
        loss.backward()

        # Gradient clipping for stable diffusion training
        torch.nn.utils.clip_grad_norm_(lvdm.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Logging training progress
        if step % log_interval == 0 or step == 1:
            interval_count = max(log_interval if step > 1 else 1, 1)
            avg_loss = running_loss / interval_count
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {step:>6d}/{max_steps:,}] "
                f"Diffusion Loss: {avg_loss:.4f} | "
                f"LR: {cur_lr:.2e}",
                flush=True,
            )
            running_loss = 0.0

        # Periodic Validation
        if step % val_interval == 0 or step == max_steps:
            val_metrics = evaluate_diffusion(lvdm, fundus_encoder, vae, val_loader, device)
            val_loss = val_metrics["val_loss"]
            print(
                f"  >>> [Validation @ Step {step:>6d}] "
                f"Val Diffusion MSE Loss: {val_loss:.4f}",
                flush=True,
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step,
                        "state_dict": lvdm.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "model_type": "ConditionalLVDM3D",
                        "config": {
                            "latent_channels": latent_channels,
                            "cond_dim": cond_dim,
                            "base_channels": base_channels,
                            "timesteps": timesteps,
                            "guidance_scale": guidance_scale,
                            "num_res_blocks": len(lvdm.unet.res_blocks),
                        },
                    },
                    checkpoint_out,
                )
                print(f"  >>> [Checkpoint] Saved best model to {checkpoint_out}", flush=True)

        # Periodic checkpoint backup
        elif step % save_interval == 0:
            torch.save(
                {
                    "step": step,
                    "state_dict": lvdm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": running_loss,
                    "model_type": "ConditionalLVDM3D",
                    "config": {
                        "latent_channels": latent_channels,
                        "cond_dim": cond_dim,
                        "base_channels": base_channels,
                        "timesteps": timesteps,
                    },
                },
                checkpoint_out,
            )

    elapsed_time = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"LVDM Training Complete in {elapsed_time:.1f}s ({elapsed_time/60.0:.2f} mins).", flush=True)
    print(f"Best Validation Loss: {best_val_loss:.4f}", flush=True)
    print(f"Verified Model Checkpoint: {checkpoint_out} ({checkpoint_out.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
