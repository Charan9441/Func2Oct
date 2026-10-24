#!/usr/bin/env python3
"""
scripts/validate_pairs.py

Validates the Fundus-OCT pairing manifest:
1. Verifies patient-level split exclusivity (ensures zero data leakage between train, val, and test).
2. Verifies file existence on disk for all paths referenced in the manifest.
3. Outputs diagnostic statistics: paired vs. unpaired samples, class distributions, and warnings.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.data.pairing import validate_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate Fundus-OCT dataset pairing manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(project_root / "data" / "manifests" / "dataset.csv"),
        help="Path to manifest CSV file (default: data/manifests/dataset.csv)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    print("=" * 70)
    print("  FUNDUS2OCT PAIRING & DATASET INTEGRITY VALIDATOR")
    print("=" * 70)
    print(f"Manifest Path: {manifest_path}\n")

    if not manifest_path.exists():
        print(f"[Warning] Manifest file not found at: {manifest_path}")
        print("Run 'python scripts/inspect_dataset.py' to generate the manifest.")
        return 0

    if manifest_path.stat().st_size == 0:
        print(f"[Warning] Manifest file is empty (0 bytes).")
        return 0

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV manifest: {e}")
        return 1

    report = validate_manifest(df)

    print(f"Total Records in Manifest : {report['total_records']}")
    print(f"  - Paired Samples        : {report['paired_records']}")
    print(f"  - Fundus-Only Samples   : {report['fundus_only']}")
    print(f"  - OCT-Only Samples      : {report['oct_only']}")

    if report["split_counts"]:
        print("\nSplit Distribution:")
        for split_name, count in report["split_counts"].items():
            pct = (count / report["total_records"]) * 100
            print(f"  - {split_name:<10} : {count:>5} samples ({pct:>5.1f}%)")

    if report["patient_leakage_detected"]:
        print("\n[CRITICAL ERROR] Patient leakage detected between splits!")
        for p in report["leakage_patients"]:
            print(f"  - Patient: {p}")
    else:
        print("\n[✓ PASS] Zero patient leakage: Patient IDs are strictly isolated across splits.")

    if report["missing_files"]:
        print(f"\n[Warning] {len(report['missing_files'])} file(s) referenced in manifest were not found on disk.")
        for p in report["missing_files"][:5]:
            print(f"  - Missing: {p}")
        if len(report["missing_files"]) > 5:
            print(f"  ... and {len(report['missing_files']) - 5} more.")
    else:
        print("[✓ PASS] All referenced files exist on disk.")

    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY: " + ("PASSED" if report["is_valid"] else "FAILED (Check Warnings)"))
    print("=" * 70 + "\n")

    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
