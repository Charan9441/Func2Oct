#!/usr/bin/env python3
"""
scripts/train_classifiers.py

Trains disease classification models to benchmark Fundus-only vs. Multimodal (Fundus + Synthetic OCT):
- Model A: Fundus-only classifier (classifier_fundus.py)
- Model B: Fundus + Synthetic OCT multimodal classifier (classifier_multimodal.py)

Strict Research Protocol:
1. Both models utilize identical patient-stratified train/val/test splits.
2. Class weights are derived strictly from training data.
3. If synthetic OCT generation checkpoint is missing/untrained, reports the blocker honestly.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.data.dataset import DiseaseClassificationDataset
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier
from fundus2oct.models.pipeline import Fundus2OCTPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fundus-only and Multimodal Disease Classifiers.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="both",
        choices=["fundus_only", "multimodal", "both"],
        help="Model architecture to train ('fundus_only', 'multimodal', or 'both').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(project_root / "data" / "manifests" / "dataset.csv"),
        help="Path to dataset manifest CSV.",
    )
    parser.add_argument(
        "--synthetic-oct-dir",
        type=str,
        default=str(project_root / "outputs" / "synthetic_oct"),
        help="Directory containing pre-generated synthetic OCT B-scans.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick 2-epoch smoke test on small synthetic dataset.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, is_multimodal: bool = False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        fundus = batch["fundus"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if is_multimodal:
            oct_img = batch["oct"].to(device)
            logits = model(fundus, oct_img)
        else:
            logits = model(fundus)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * fundus.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def validate_epoch(model, loader, criterion, device, is_multimodal: bool = False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            fundus = batch["fundus"].to(device)
            labels = batch["label"].to(device)

            if is_multimodal:
                oct_img = batch["oct"].to(device)
                logits = model(fundus, oct_img)
            else:
                logits = model(fundus)

            loss = criterion(logits, labels)
            total_loss += loss.item() * fundus.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    checkpoint_path: Path,
    epochs: int,
    lr: float,
    device: torch.device,
    is_multimodal: bool = False,
) -> Dict[str, Any]:
    print(f"\n--- Training {model_name} ---")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, is_multimodal=is_multimodal)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, is_multimodal=is_multimodal)
        el = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{epochs:03d} [{el:.1f}s] "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": best_val_acc,
                    "model_type": "multimodal" if is_multimodal else "fundus_only",
                },
                checkpoint_path,
            )

    total_time = time.time() - start_time
    print(f"[{model_name}] Finished in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}, Acc: {best_val_acc*100:.1f}%")
    print(f"Saved Checkpoint -> {checkpoint_path}")
    return {"best_val_loss": best_val_loss, "best_val_acc": best_val_acc, "checkpoint": str(checkpoint_path)}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed or cfg.seed
    set_seed(seed)

    device = torch.device(args.device if args.device else cfg.resolve_device())
    epochs = args.epochs or (2 if args.smoke_test else cfg.classifiers.epochs)
    batch_size = args.batch_size or cfg.classifiers.batch_size
    lr = args.lr or cfg.classifiers.learning_rate

    manifest_file = Path(args.manifest)
    synth_oct_dir = Path(args.synthetic_oct_dir)

    print("=" * 70)
    print("  FUNDUS2OCT - COMPARATIVE DISEASE CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Model Selection  : {args.model_type.upper()}")
    print(f"Target Labels    : Binary 0 = OTHER (Control), 1 = MD (Macular Degeneration)")
    print(f"Device           : {device}")
    print(f"Epochs           : {epochs}")
    print(f"Batch Size       : {batch_size}")
    print(f"Learning Rate    : {lr}")
    print("=" * 70)

    # 1. Audit Synthetic Generator Readiness for Model B
    run_model_b = args.model_type in ("multimodal", "both")
    pipeline = Fundus2OCTPipeline(config=cfg, device=device)

    if run_model_b and not pipeline.is_generator_ready():
        has_pregenerated = synth_oct_dir.exists() and any(synth_oct_dir.glob("*.png"))
        if not has_pregenerated and not args.smoke_test:
            print("\n[RESEARCH INTEGRITY BLOCKER - MODEL B]")
            print("  The conditional diffusion generator checkpoint ('diffusion.pt') is untrained,")
            print("  and no pre-generated synthetic OCT B-scans were found in outputs/synthetic_oct/.")
            print("  Per research protocol: untrained/random noise cannot be substituted as synthetic OCT.")
            print("  To train Model B, complete Milestone 3 (Diffusion Training) or run with --smoke-test.")
            if args.model_type == "multimodal":
                return 1
            print("  -> Skipping Model B and proceeding with Model A (Fundus-only).\n")
            run_model_b = False

    # 2. Prepare Datasets (Identical Patient Splits)
    smoke_count = 16 if args.smoke_test else 0

    train_ds_fundus = DiseaseClassificationDataset(
        manifest_path=manifest_file,
        mode="fundus_only",
        split="train",
        synthetic_fallback_count=smoke_count,
    )
    val_ds_fundus = DiseaseClassificationDataset(
        manifest_path=manifest_file,
        mode="fundus_only",
        split="val",
        synthetic_fallback_count=smoke_count // 2 if smoke_count else 0,
    )

    if len(train_ds_fundus) == 0 and not args.smoke_test:
        print("[Notice] Manifest image paths not found locally. Enabling smoke-test synthetic mode.")
        train_ds_fundus = DiseaseClassificationDataset(
            manifest_path=manifest_file,
            mode="fundus_only",
            split="train",
            synthetic_fallback_count=16,
        )
        val_ds_fundus = DiseaseClassificationDataset(
            manifest_path=manifest_file,
            mode="fundus_only",
            split="val",
            synthetic_fallback_count=8,
        )

    # Class weights for handling class imbalance (fit on training data only)
    class_weights = train_ds_fundus.get_class_weights().to(device)
    print(f"Calculated Training Class Weights: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader_fundus = DataLoader(train_ds_fundus, batch_size=batch_size, shuffle=True)
    val_loader_fundus = DataLoader(val_ds_fundus, batch_size=batch_size, shuffle=False)

    results: Dict[str, Any] = {}

    # 3. Train Model A (Fundus-only)
    if args.model_type in ("fundus_only", "both"):
        model_a = FundusClassifier(
            in_channels=3,
            num_classes=cfg.classifiers.num_classes,
            feature_dim=cfg.classifiers.feature_dim,
            dropout=cfg.classifiers.dropout,
        ).to(device)

        ckpt_a = Path(cfg.checkpoints.classifier_fundus)
        results["model_a"] = train_single_model(
            model_name="Model A (Fundus-Only)",
            model=model_a,
            train_loader=train_loader_fundus,
            val_loader=val_loader_fundus,
            criterion=criterion,
            checkpoint_path=ckpt_a,
            epochs=epochs,
            lr=lr,
            device=device,
            is_multimodal=False,
        )

    # 4. Train Model B (Fundus + Synthetic OCT)
    if run_model_b:
        train_ds_multi = DiseaseClassificationDataset(
            manifest_path=manifest_file,
            mode="multimodal",
            split="train",
            synthetic_oct_dir=synth_oct_dir,
            synthetic_fallback_count=smoke_count,
        )
        val_ds_multi = DiseaseClassificationDataset(
            manifest_path=manifest_file,
            mode="multimodal",
            split="val",
            synthetic_oct_dir=synth_oct_dir,
            synthetic_fallback_count=smoke_count // 2 if smoke_count else 0,
        )

        train_loader_multi = DataLoader(train_ds_multi, batch_size=batch_size, shuffle=True)
        val_loader_multi = DataLoader(val_ds_multi, batch_size=batch_size, shuffle=False)

        model_b = MultimodalClassifier(
            fundus_channels=3,
            oct_channels=1,
            num_classes=cfg.classifiers.num_classes,
            feature_dim=cfg.classifiers.feature_dim,
            dropout=cfg.classifiers.dropout,
        ).to(device)

        ckpt_b = Path(cfg.checkpoints.classifier_multimodal)
        results["model_b"] = train_single_model(
            model_name="Model B (Multimodal: Fundus + Synthetic OCT)",
            model=model_b,
            train_loader=train_loader_multi,
            val_loader=val_loader_multi,
            criterion=criterion,
            checkpoint_path=ckpt_b,
            epochs=epochs,
            lr=lr,
            device=device,
            is_multimodal=True,
        )

    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name.upper()}: Best Val Loss={r['best_val_loss']:.4f}, Val Acc={r['best_val_acc']*100:.1f}%")
        print(f"             Checkpoint -> {r['checkpoint']}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
