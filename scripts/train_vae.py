#!/usr/bin/env python3
"""
scripts/train_vae.py

Trains the OCT Autoencoder / Variational Autoencoder (VAE) for 2D B-scans.
Architecture maps 256x256 B-scans to an 8x32x32 latent space (8x spatial compression).

Features:
- Step-based training loop with 50,000 steps by default.
- Reconstruction loss optimization: L1 + MSE pixel-level reconstruction with PSNR tracking.
- Optional KL-divergence regularization (VAE mode).
- Automatic dataset loading from data/oct and data/manifests/dataset.csv with synthetic fallback.
- Periodic evaluation, validation loss logging, and best checkpoint saving to checkpoints/oct_vae.pt.
- Fast --smoke-test mode for rapid architectural and pipeline verification.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.data.dataset import OCTReconstructionDataset
from fundus2oct.models.oct_vae import (
    OCTAutoencoder2D,
    OCTVAE2D,
    build_oct_vae,
)


def compute_psnr(mse_val: float, max_val: float = 2.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (dB) for [-1.0, 1.0] range (dynamic range = 2.0)."""
    if mse_val <= 1e-10:
        return 50.0
    return float(10.0 * math.log10((max_val ** 2) / mse_val))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the OCT Autoencoder / VAE mapping 256x256 B-scans to 8x32x32 latent space."
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
        help="Number of training steps (default: 50000 steps).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for AdamW optimizer (default: 2e-4).",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=8,
        help="Number of latent channels for 8x32x32 latent space (default: 8).",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1e-4,
        help="Weight for KL-divergence loss in VAE mode (default: 1e-4; set 0.0 for pure reconstruction).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vae",
        choices=["vae", "autoencoder"],
        help="Model architecture: 'vae' (probabilistic) or 'autoencoder' (deterministic).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "oct_vae.pt"),
        help="Destination path for best checkpoint.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Print loss metrics every N steps (default: 500).",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=2500,
        help="Evaluate on validation set every N steps (default: 2500).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick 100-step smoke test for verification.",
    )
    return parser.parse_args()


def cycle_dataloader(loader: DataLoader):
    """Infinite generator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


def evaluate(model, val_loader, kl_weight, device) -> Dict[str, float]:
    """Runs evaluation over validation dataset."""
    model.eval()
    total_recon = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["oct"].to(device)
            if isinstance(model, OCTAutoencoder2D):
                x_recon, _ = model(x)
                losses = model.loss_function(x, x_recon)
            else:
                x_recon, mu, logvar = model(x, stochastic=False)
                losses = model.loss_function(x, x_recon, mu, logvar, kl_weight=kl_weight)

            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_mse += losses["mse"].item()
            total_l1 += losses["l1"].item()
            n_batches += 1

    n_batches = max(n_batches, 1)
    avg_mse = total_mse / n_batches
    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "mse": avg_mse,
        "l1": total_l1 / n_batches,
        "psnr": compute_psnr(avg_mse),
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
    kl_weight = args.kl_weight
    latent_channels = args.latent_channels
    is_autoencoder = args.model_type == "autoencoder"

    if args.device:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    else:
        device = cfg.resolve_device()

    print("=" * 70)
    print("  FUNDUS2OCT - OCT B-SCAN AUTOENCODER / VAE TRAINING")
    print("=" * 70)
    print(f"Architecture      : {'OCTAutoencoder2D (Deterministic)' if is_autoencoder else 'OCTVAE2D (Probabilistic)'}")
    print(f"Input Shape       : (B, 1, 256, 256)")
    print(f"Latent Space      : (B, {latent_channels}, 32, 32) -> 8x Spatial Compression")
    print(f"Target Steps      : {max_steps:,} steps")
    print(f"Batch Size        : {batch_size}")
    print(f"Learning Rate     : {lr}")
    print(f"KL Loss Weight    : {kl_weight if not is_autoencoder else 'N/A (Autoencoder)'}")
    print(f"Compute Device    : {device}")
    print(f"Output Checkpoint : {args.output_checkpoint}")
    print("=" * 70)

    # 1. Initialize Dataset
    manifest_file = project_root / cfg.paths.manifest_path
    oct_dir = project_root / cfg.paths.oct_dir

    dataset = OCTReconstructionDataset(
        manifest_path=manifest_file if manifest_file.exists() else None,
        data_dir=oct_dir if oct_dir.exists() else None,
        target_size=(256, 256),
        is_3d=False,
        normalize=cfg.preprocessing.oct_normalize,
        synthetic_count=16 if args.smoke_test else 0,
    )

    if len(dataset) == 0:
        print("[Notice] No local OCT files found in data/oct and manifest is empty.")
        print("         Engaging synthetic dataset for continuous training.")
        dataset = OCTReconstructionDataset(
            target_size=(256, 256),
            is_3d=False,
            normalize=cfg.preprocessing.oct_normalize,
            synthetic_count=32,
        )

    print(f"Dataset Loaded    : {len(dataset)} OCT B-scan samples.")

    # Split into train/validation (80/20)
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

    # 2. Build Model
    if is_autoencoder:
        model = OCTAutoencoder2D(
            in_channels=1,
            latent_channels=latent_channels,
            base_channels=cfg.vae.base_channels,
            channel_multipliers=tuple(cfg.vae.channel_multipliers),
        ).to(device)
    else:
        model = OCTVAE2D(
            in_channels=1,
            latent_channels=latent_channels,
            base_channels=cfg.vae.base_channels,
            channel_multipliers=tuple(cfg.vae.channel_multipliers),
        ).to(device)

    # 3. Optimizer & LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)

    # 4. Training Loop (50k steps)
    checkpoint_out = Path(args.output_checkpoint)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    data_iter = iter(cycle_dataloader(train_loader))
    best_recon_loss = float("inf")
    start_time = time.time()

    running_loss = 0.0
    running_recon = 0.0
    running_mse = 0.0
    running_l1 = 0.0
    running_kl = 0.0

    print(f"\nStarting {max_steps:,} step training loop...\n")

    for step in range(1, max_steps + 1):
        model.train()
        batch = next(data_iter)
        x = batch["oct"].to(device)

        optimizer.zero_grad()

        if is_autoencoder:
            x_recon, _ = model(x)
            losses = model.loss_function(x, x_recon)
        else:
            x_recon, mu, logvar = model(x, stochastic=True)
            losses = model.loss_function(x, x_recon, mu, logvar, kl_weight=kl_weight)

        loss = losses["loss"]
        loss.backward()

        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_recon += losses["recon_loss"].item()
        running_mse += losses["mse"].item()
        running_l1 += losses["l1"].item()
        if "kl_loss" in losses:
            running_kl += losses["kl_loss"].item()

        # Log training metrics
        if step % log_interval == 0 or step == 1:
            interval_count = max(log_interval if step > 1 else 1, 1)
            avg_loss = running_loss / interval_count
            avg_recon = running_recon / interval_count
            avg_mse = running_mse / interval_count
            avg_l1 = running_l1 / interval_count
            avg_kl = running_kl / interval_count
            cur_lr = scheduler.get_last_lr()[0]
            psnr = compute_psnr(avg_mse)

            if is_autoencoder:
                print(
                    f"[Step {step:>6d}/{max_steps:,}] "
                    f"Recon Loss: {avg_recon:.4f} (MSE: {avg_mse:.4f}, L1: {avg_l1:.4f}) | "
                    f"PSNR: {psnr:5.2f} dB | LR: {cur_lr:.2e}",
                    flush=True,
                )
            else:
                print(
                    f"[Step {step:>6d}/{max_steps:,}] "
                    f"Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} (MSE: {avg_mse:.4f}, L1: {avg_l1:.4f}) | "
                    f"KL: {avg_kl:.4f} | PSNR: {psnr:5.2f} dB | LR: {cur_lr:.2e}",
                    flush=True,
                )

            running_loss = 0.0
            running_recon = 0.0
            running_mse = 0.0
            running_l1 = 0.0
            running_kl = 0.0

        # Run Validation
        if step % val_interval == 0 or step == max_steps:
            val_metrics = evaluate(model, val_loader, kl_weight, device)
            print(
                f"  >>> [Validation @ Step {step}] "
                f"Val Recon Loss: {val_metrics['recon_loss']:.4f} (MSE: {val_metrics['mse']:.4f}, L1: {val_metrics['l1']:.4f}) | "
                f"Val PSNR: {val_metrics['psnr']:.2f} dB",
                flush=True,
            )

            # Save best checkpoint
            if val_metrics["recon_loss"] < best_recon_loss:
                best_recon_loss = val_metrics["recon_loss"]
                torch.save(
                    {
                        "step": step,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_recon_loss": best_recon_loss,
                        "val_psnr": val_metrics["psnr"],
                        "model_type": args.model_type,
                        "is_3d": False,
                        "config": {
                            "in_channels": 1,
                            "latent_channels": latent_channels,
                            "latent_shape": (latent_channels, 32, 32),
                            "input_size": (256, 256),
                            "base_channels": cfg.vae.base_channels,
                            "channel_multipliers": cfg.vae.channel_multipliers,
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
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_type": args.model_type,
                    "config": {
                        "in_channels": 1,
                        "latent_channels": latent_channels,
                        "latent_shape": (latent_channels, 32, 32),
                        "input_size": (256, 256),
                    },
                },
                checkpoint_out,
            )

    elapsed_time = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"Training Complete in {elapsed_time:.1f}s ({elapsed_time/60.0:.2f} mins).", flush=True)
    print(f"Best Validation Reconstruction Loss: {best_recon_loss:.4f}", flush=True)
    print(f"Verified Model Checkpoint: {checkpoint_out} ({checkpoint_out.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
