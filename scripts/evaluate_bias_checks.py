#!/usr/bin/env python3
"""
scripts/evaluate_bias_checks.py

Command-line Evaluation Runner for Retinal Classifier Bias Checks
and Cross-Dataset External Generalization (IDRiD and ARMD).

Usage:
    # Run with default or configured paths:
    python scripts/evaluate_bias_checks.py \\
        --checkpoint checkpoints/classifier_fundus.pt \\
        --idrid-dir data/external/idrid \\
        --armd-dir data/external/armd \\
        --output-dir outputs/evaluation_reports

    # Offline Smoke Test (when external datasets are not on disk):
    python scripts/evaluate_bias_checks.py --smoke-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

# Ensure src/ is on sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.evaluation.bias_checks import (
    evaluate_bias_and_generalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bias Checks and Cross-Dataset Generalization Evaluation on Retinal Classifiers."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "classifier_fundus.pt"),
        help="Path to trained classifier checkpoint (.pt).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["fundus", "multimodal"],
        default="fundus",
        help="Classifier architecture type ('fundus' or 'multimodal').",
    )
    parser.add_argument(
        "--idrid-dir",
        type=str,
        default=None,
        help="Path to IDRiD dataset directory or manifest file.",
    )
    parser.add_argument(
        "--armd-dir",
        type=str,
        default=None,
        help="Path to ARMD dataset directory or manifest file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs" / "evaluation_reports"),
        help="Directory to save generated JSON, CSV, and Markdown audit reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target execution device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Fixed classification decision threshold (default: 0.50).",
    )
    parser.add_argument(
        "--min-subgroup-samples",
        type=int,
        default=10,
        help="Minimum sample count required to evaluate subgroup without small-sample warning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility random seed for bootstrapping and data sampling.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute synthetic verification run when external datasets are not on disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Device resolution
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n" + "=" * 76)
    print("      FUNDUS2OCT: BIAS CHECKS & CROSS-DATASET GENERALIZATION AUDIT")
    print("=" * 76)
    print(f" Checkpoint:               {args.checkpoint}")
    print(f" Model Type:               {args.model_type}")
    print(f" Target Device:            {device}")
    print(f" Decision Threshold:       {args.threshold}")
    print(f" Min Subgroup Size:        {args.min_subgroup_samples}")
    print(f" Smoke Test Mode:          {'Enabled' if args.smoke_test else 'Disabled'}")
    print("-" * 76)

    try:
        results = evaluate_bias_and_generalization(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            idrid_path=args.idrid_dir,
            armd_path=args.armd_dir,
            output_dir=args.output_dir,
            device=device,
            threshold=args.threshold,
            min_subgroup_samples=args.min_subgroup_samples,
            seed=args.seed,
            smoke_test=args.smoke_test,
        )

        print("\n[+] Audit Execution Completed Successfully.")
        print(f"    Validation Reports:    {len(results.get('validation_reports', []))}")
        print(f"    Evaluated Datasets:    {len(results.get('dataset_metrics', []))}")
        print(f"    Incompatible Cohorts:  {len(results.get('incompatible_datasets', []))}")
        print(f"    Subgroup Breakdowns:   {len(results.get('subgroup_metrics', []))}")
        print("\nGenerated Output Artifacts:")
        for name, path in results.get("artifacts", {}).items():
            print(f"  - {name}: {path}")
        print("=" * 76 + "\n")
        return 0

    except Exception as exc:
        print(f"\n[!] Bias Check Execution Failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
