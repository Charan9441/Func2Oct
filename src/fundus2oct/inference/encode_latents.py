"""
src/fundus2oct/inference/encode_latents.py

Latent encoding engine for Optical Coherence Tomography (OCT) B-scans and volumes.
Compresses OCT inputs through a trained VAE into continuous latent representations:
- Evaluates with torch.no_grad() in eval mode.
- Extracts latent parameters (mu, logvar) and deterministic/stochastic latent z.
- Saves serialized latent tensors and a metadata CSV manifest.
- Enforces strict checkpoint compatibility checks, preventing untrained output from being mistaken for learned representations.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fundus2oct.config import AppConfig, load_config
from fundus2oct.data.dataset import OCTReconstructionDataset
from fundus2oct.models.oct_vae import (
    CheckpointError,
    build_oct_vae,
    load_vae_checkpoint,
)


def encode_oct_dataset(
    config: Optional[AppConfig] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    is_3d: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    stochastic: bool = False,
    allow_untrained: bool = False,
    synthetic_fallback_count: int = 0,
) -> Dict[str, Any]:
    """
    Encodes an entire OCT dataset into VAE latent representations.
    
    Args:
        config: Application configuration.
        checkpoint_path: Path to trained VAE checkpoint (defaults to config path).
        output_dir: Directory where latent tensors and metadata manifest are saved.
        data_dir: Directory containing OCT images.
        manifest_path: Path to dataset manifest CSV.
        batch_size: DataLoader batch size.
        is_3d: Whether processing 3D volumetric scans.
        device: Target execution device.
        stochastic: If True, samples z = mu + sigma*eps; if False, uses deterministic z = mu.
        allow_untrained: If True, allows execution with random weights for smoke testing.
        synthetic_fallback_count: Number of synthetic tensors to encode if data directory is empty.

    Returns:
        Summary dictionary containing output paths, sample counts, and verification metadata.
    """
    cfg = config or load_config()

    # 1. Resolve Device
    if device is None:
        target_device = cfg.resolve_device()
    elif isinstance(device, str):
        target_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    else:
        target_device = device

    # 2. Resolve Paths
    ckpt_file = Path(checkpoint_path or cfg.checkpoints.oct_vae)
    out_path = Path(output_dir or cfg.paths.outputs_dir / "oct_latents")
    out_path.mkdir(parents=True, exist_ok=True)

    man_file = Path(manifest_path or cfg.paths.manifest_path)
    oct_path = Path(data_dir or cfg.paths.oct_dir)

    # 3. Build Model
    model = build_oct_vae(
        is_3d=is_3d,
        in_channels=cfg.vae.in_channels,
        latent_channels=cfg.vae.latent_channels,
        base_channels=cfg.vae.base_channels,
        channel_multipliers=tuple(cfg.vae.channel_multipliers),
    ).to(target_device)

    # 4. Load Checkpoint with Integrity Check
    try:
        ckpt_info = load_vae_checkpoint(
            model=model,
            checkpoint_path=ckpt_file,
            device=target_device,
            allow_untrained_fallback=allow_untrained,
        )
    except CheckpointError as e:
        if not allow_untrained:
            raise e
        print(f"[SMOKE-TEST NOTICE] {e}")
        ckpt_info = {"status": "untrained", "reason": str(e)}

    is_trained = model.is_trained
    if not is_trained:
        print("\n" + "!" * 70)
        print("  WARNING: VAE IS RUNNING WITH UNTRAINED / RANDOM WEIGHTS")
        print("  Outputs represent architecture tests only, NOT learned representations.")
        print("!" * 70 + "\n")

    model.eval()

    # 5. Initialize Dataset & Loader
    dataset = OCTReconstructionDataset(
        manifest_path=man_file if man_file.exists() else None,
        data_dir=oct_path if oct_path.exists() else None,
        target_size=cfg.preprocessing.oct_size,
        is_3d=is_3d,
        volume_depth=cfg.preprocessing.volume_depth,
        normalize=cfg.preprocessing.oct_normalize,
        synthetic_count=synthetic_fallback_count,
    )

    if len(dataset) == 0:
        if synthetic_fallback_count == 0 and allow_untrained:
            print("[Notice] Dataset empty; generating 8 synthetic samples for architecture smoke test.")
            dataset = OCTReconstructionDataset(
                target_size=cfg.preprocessing.oct_size,
                is_3d=is_3d,
                volume_depth=cfg.preprocessing.volume_depth,
                normalize=cfg.preprocessing.oct_normalize,
                synthetic_count=8,
            )
        else:
            raise ValueError(f"No OCT samples found in manifest '{man_file}' or directory '{oct_path}'.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Encoding {len(dataset)} OCT samples into latent space on device: {target_device}...")
    start_time = time.time()

    encoded_records: List[Dict[str, Any]] = []
    latents_dict: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["oct"].to(target_device)
            sample_ids = batch["sample_id"]
            classes = batch["class_name"]
            paths = batch["path"]

            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar, stochastic=stochastic)

            # Move to CPU for serialization
            z_cpu = z.cpu()
            mu_cpu = mu.cpu()
            logvar_cpu = logvar.cpu()

            for i in range(x.shape[0]):
                s_id = str(sample_ids[i])
                latent_tensor = z_cpu[i]  # Shape: (C_lat, H/8, W/8) or (C_lat, D/8, H/8, W/8)
                latents_dict[s_id] = latent_tensor

                # Save individual sample latent tensor
                sample_file = out_path / f"{s_id}_latent.pt"
                torch.save(
                    {
                        "sample_id": s_id,
                        "latent": latent_tensor,
                        "mu": mu_cpu[i],
                        "logvar": logvar_cpu[i],
                        "is_trained": is_trained,
                        "is_3d": is_3d,
                        "stochastic": stochastic,
                    },
                    sample_file,
                )

                encoded_records.append({
                    "sample_id": s_id,
                    "class_name": str(classes[i]),
                    "original_path": str(paths[i]),
                    "latent_file": str(sample_file),
                    "latent_shape": str(list(latent_tensor.shape)),
                    "mu_norm": float(torch.norm(mu_cpu[i]).item()),
                    "logvar_mean": float(logvar_cpu[i].mean().item()),
                    "is_trained": is_trained,
                })

    elapsed = time.time() - start_time

    # Save consolidated latent dictionary
    consolidated_path = out_path / "all_latents.pt"
    torch.save(
        {
            "latents": latents_dict,
            "is_trained": is_trained,
            "is_3d": is_3d,
            "latent_channels": cfg.vae.latent_channels,
            "total_samples": len(encoded_records),
        },
        consolidated_path,
    )

    # Save metadata CSV manifest
    manifest_df = pd.DataFrame(encoded_records)
    manifest_out = out_path / "latent_manifest.csv"
    manifest_df.to_csv(manifest_out, index=False)

    print(f"Successfully encoded {len(encoded_records)} samples in {elapsed:.2f}s.")
    print(f"Latents saved to directory: {out_path}")
    print(f"Manifest written to: {manifest_out}")

    return {
        "status": "success",
        "total_encoded": len(encoded_records),
        "is_trained": is_trained,
        "is_3d": is_3d,
        "elapsed_seconds": elapsed,
        "output_dir": str(out_path),
        "consolidated_file": str(consolidated_path),
        "manifest_file": str(manifest_out),
        "sample_latent_shape": list(latents_dict[list(latents_dict.keys())[0]].shape),
    }
