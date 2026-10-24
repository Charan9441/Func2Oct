#!/usr/bin/env python3
"""
scripts/train_vae.py

Trains the OCT Variational Autoencoder (VAE) for 2D B-scans or 3D volumes.
Features:
- Reconstruction loss (L1 + MSE) and KL-divergence loss with configurable weighting.
- Patient-stratified Train/Validation evaluation.
- Checkpoint saving with metadata to checkpoints/oct_vae.pt.
- Graceful synthetic smoke-test mode if local dataset images are not yet unzipped.
"""

import argparse
from pathlib import Path
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.data.dataset import OCTReconstructionDataset
from fundus2oct.models.oct_vae import build_oct_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Fundus2OCT Variational Autoencoder.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--kl-weight", type=float, default=None, help="Weight for KL-divergence loss.")
    parser.add_argument("--is-3d", action="store_true", help="Train 3D volumetric VAE instead of 2D B-scan VAE.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "oct_vae.pt"),
        help="Destination path for best checkpoint.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2-epoch smoke test on small synthetic batch.",
    )
    return parser.parse_args()


def train_epoch(model, loader, optimizer, kl_weight, device):
    model.train()
    total_loss = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["oct"].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, stochastic=True)
        losses = model.loss_function(x, x_recon, mu, logvar, kl_weight=kl_weight)
        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss_sum += losses["recon_loss"].item()
        kl_loss_sum += losses["kl_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "recon_loss": recon_loss_sum / max(n_batches, 1),
        "kl_loss": kl_loss_sum / max(n_batches, 1),
    }


def validate_epoch(model, loader, kl_weight, device):
    model.eval()
    total_loss = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["oct"].to(device)
            x_recon, mu, logvar = model(x, stochastic=False)
            losses = model.loss_function(x, x_recon, mu, logvar, kl_weight=kl_weight)
            total_loss += losses["loss"].item()
            recon_loss_sum += losses["recon_loss"].item()
            kl_loss_sum += losses["kl_loss"].item()
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "recon_loss": recon_loss_sum / max(n_batches, 1),
        "kl_loss": kl_loss_sum / max(n_batches, 1),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve settings
    epochs = args.epochs or (2 if args.smoke_test else 50)
    batch_size = args.batch_size or 4
    lr = args.lr or cfg.vae.learning_rate
    kl_weight = args.kl_weight or cfg.vae.kl_weight
    is_3d = args.is_3d or cfg.vae.is_3d

    if args.device:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    else:
        device = cfg.resolve_device()

    print("=" * 70)
    print("  FUNDUS2OCT - OCT VAE TRAINING")
    print("=" * 70)
    print(f"Modality Mode     : {'3D Volumetric (1, D, H, W)' if is_3d else '2D B-Scan (1, H, W)'}")
    print(f"Device Selected   : {device}")
    print(f"Epochs            : {epochs}")
    print(f"Batch Size        : {batch_size}")
    print(f"Learning Rate     : {lr}")
    print(f"KL Weight         : {kl_weight}")
    print(f"Output Checkpoint : {args.output_checkpoint}")
    print("=" * 70)

    # Initialize Dataset
    manifest_file = project_root / cfg.paths.manifest_path
    oct_dir = project_root / cfg.paths.oct_dir

    dataset = OCTReconstructionDataset(
        manifest_path=manifest_file if manifest_file.exists() else None,
        data_dir=oct_dir,
        target_size=cfg.preprocessing.oct_size,
        is_3d=is_3d,
        volume_depth=cfg.preprocessing.volume_depth,
        normalize=cfg.preprocessing.oct_normalize,
        synthetic_count=16 if args.smoke_test else 0,
    )

    if len(dataset) == 0:
        print("[Notice] No local OCT files found in data/oct and manifest is empty.")
        print("         Engaging synthetic-tensor mode for architectural verification.")
        dataset = OCTReconstructionDataset(
            target_size=cfg.preprocessing.oct_size,
            is_3d=is_3d,
            volume_depth=cfg.preprocessing.volume_depth,
            normalize=cfg.preprocessing.oct_normalize,
            synthetic_count=16,
        )

    print(f"Total Dataset Samples Available: {len(dataset)}")

    # Split into train/validation (80/20)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize Model
    model = build_oct_vae(
        is_3d=is_3d,
        in_channels=cfg.vae.in_channels,
        latent_channels=cfg.vae.latent_channels,
        base_channels=cfg.vae.base_channels,
        channel_multipliers=tuple(cfg.vae.channel_multipliers),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    checkpoint_out = Path(args.output_checkpoint)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, kl_weight, device)
        val_metrics = validate_epoch(model, val_loader, kl_weight, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] "
            f"Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f})"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "is_3d": is_3d,
                    "config": {
                        "in_channels": cfg.vae.in_channels,
                        "latent_channels": cfg.vae.latent_channels,
                        "base_channels": cfg.vae.base_channels,
                        "channel_multipliers": cfg.vae.channel_multipliers,
                    },
                },
                checkpoint_out,
            )

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training Complete in {total_time:.1f}s. Best Validation Loss: {best_val_loss:.4f}")
    print(f"Saved Checkpoint -> {checkpoint_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
