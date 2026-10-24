#!/usr/bin/env python3
"""
scripts/encode_latents.py

Command-line interface to encode an OCT dataset into VAE latent representations.
Example:
    python scripts/encode_latents.py --checkpoint checkpoints/oct_vae.pt --output-dir outputs/oct_latents
    python scripts/encode_latents.py --smoke-test --allow-untrained
"""

import argparse
from pathlib import Path
import sys

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.inference.encode_latents import encode_oct_dataset


def main():
    parser = argparse.ArgumentParser(description="Encode OCT dataset into VAE continuous latent space.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to VAE checkpoint (defaults to configs/default.yaml path).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs" / "oct_latents"),
        help="Directory where latents and manifest will be saved.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing OCT images to encode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--is-3d",
        action="store_true",
        help="Process 3D volumetric OCT stacks instead of 2D B-scans.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastically (z = mu + sigma*eps) instead of deterministic z = mu.",
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Permit execution with random weights for smoke testing / architecture verification.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick architecture smoke test on 8 synthetic samples.",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    print("=" * 70)
    print("  FUNDUS2OCT - LATENT ENCODING PIPELINE")
    print("=" * 70)

    try:
        results = encode_oct_dataset(
            config=cfg,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            is_3d=args.is_3d,
            stochastic=args.stochastic,
            allow_untrained=args.allow_untrained or args.smoke_test,
            synthetic_fallback_count=8 if args.smoke_test else 0,
        )

        print("\n" + "=" * 70)
        print("  LATENT ENCODING SUCCESSFUL")
        print("=" * 70)
        print(f"Total Encoded Samples : {results['total_encoded']}")
        print(f"Latent Shape Per Item : {results['sample_latent_shape']}")
        print(f"Is Trained Weights    : {results['is_trained']}")
        print(f"Output Directory      : {results['output_dir']}")
        print(f"Consolidated File     : {results['consolidated_file']}")
        print(f"Manifest File         : {results['manifest_file']}")
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Latent encoding failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
