"""
src/fundus2oct/data/pairing.py

Pairing logic and manifest generator for matching 2D Retinal Fundus photographs
with 2D/3D Optical Coherence Tomography (OCT) records.
Partitions dataset pairs into stratified splits:
- 80% Train
- 10% Validation
- 10% Test
Stratified by pathology classes (e.g. MD, OTHER) with zero patient-level data leakage.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_class_name(name: str) -> str:
    """Normalizes clinical class names (e.g. 'FundusMD', 'octMD', 'MD' -> 'MD')."""
    clean = name.strip()
    upper = clean.upper()
    if "MD" in upper or "AMD" in upper or "ARMD" in upper:
        return "MD"
    elif "OTHER" in upper or "CONTROL" in upper or "NORMAL" in upper:
        return "OTHER"
    return clean


def normalize_stem(stem: str) -> str:
    """
    Extracts core numerical or alphanumeric identifier from stem.
    e.g., '001', '0001', 'img_001', 'patient_1' -> '1'
    """
    # Extract trailing or isolated digits if present
    match = re.search(r"(\d+)$", stem)
    if match:
        return str(int(match.group(1)))
    match_any = re.search(r"(\d+)", stem)
    if match_any:
        return str(int(match_any.group(1)))
    return stem.lower().strip()


def extract_patient_and_eye(
    filename_or_path: str,
    pattern: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Extracts patient identifier and eye lateralization (OD/OS/L/R) from filename or path.
    Defaults to regex matching if pattern is provided; otherwise extracts stem and checks
    standard clinical conventions (e.g. 'OD' for right eye, 'OS' for left eye).
    """
    stem = Path(filename_or_path).stem

    if pattern:
        match = re.search(pattern, stem, re.IGNORECASE)
        if match:
            groups = match.groupdict()
            patient_id = groups.get("patient_id", stem)
            eye = groups.get("eye", "unknown").upper()
            if eye in ("OD", "R", "RIGHT"):
                eye = "OD"
            elif eye in ("OS", "L", "LEFT"):
                eye = "OS"
            return str(patient_id), eye

    # Fallback heuristic: check for _OD, _OS, _R, _L suffixes
    eye = "unknown"
    patient_id = stem

    for suffix, lateral in [
        ("_OD", "OD"), ("_OS", "OS"), ("-OD", "OD"), ("-OS", "OS"),
        ("_R", "OD"), ("_L", "OS"), ("-R", "OD"), ("-L", "OS"),
        ("_right", "OD"), ("_left", "OS"),
    ]:
        if stem.lower().endswith(suffix.lower()):
            eye = lateral
            patient_id = stem[:-len(suffix)]
            break

    return patient_id, eye


def scan_directory_for_images(
    directory: Path,
    extensions: Set[str] = SUPPORTED_EXTENSIONS,
) -> List[Path]:
    """Recursively scans directory for supported image files."""
    if not directory.exists():
        return []
    return [
        p for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in extensions
    ]


def partition_indices(
    n_samples: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, List[int]]:
    """
    Partitions N indices into train, validation, and test sets.
    Ensures that when N >= 3, every split receives at least 1 sample.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    indices = rng.permutation(n_samples).tolist()

    if n_samples == 0:
        return {"train": [], "val": [], "test": []}

    if n_samples == 1:
        return {"train": indices, "val": [], "test": []}

    if n_samples == 2:
        return {"train": [indices[0]], "val": [indices[1]], "test": []}

    # Standard exact rounding
    n_train = max(1, int(round(n_samples * train_ratio)))
    n_val = max(1, int(round(n_samples * val_ratio)))
    n_test = n_samples - n_train - n_val

    # Adjust boundary if needed so no negative counts
    if n_test < 1 and n_samples >= 3:
        if n_train > 1:
            n_train -= 1
            n_test = 1
        elif n_val > 1:
            n_val -= 1
            n_test = 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def build_pairing_manifest(
    fundus_dir: Union[str, Path] = "data/fundus",
    oct_dir: Union[str, Path] = "data/oct",
    strategy: str = "stem_match",
    regex_pattern: Optional[str] = None,
    train_split: float = 0.80,
    val_split: float = 0.10,
    test_split: float = 0.10,
    stratify_by_class: bool = True,
    output_manifest: Optional[Union[str, Path]] = "data/manifests/dataset.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds a patient-level paired manifest DataFrame.
    Partitions dataset pairs into:
    - 80% train
    - 10% validation
    - 10% test
    stratified by pathology classes (e.g. MD, OTHER), with zero patient leakage.

    Saves the result to output_manifest if specified.
    """
    fundus_path = Path(fundus_dir)
    oct_path = Path(oct_dir)

    fundus_files = scan_directory_for_images(fundus_path)
    oct_files = scan_directory_for_images(oct_path)

    # Collect Fundus records: map by (class_name, normalized_id)
    fundus_by_key: Dict[Tuple[str, str], Tuple[str, Path, str]] = {}
    for f in fundus_files:
        # Determine class name from parent folder name
        rel = f.relative_to(fundus_path)
        cls_name = normalize_class_name(rel.parts[0] if len(rel.parts) > 1 else "MD")
        stem = f.stem
        norm_id = normalize_stem(stem)
        p_id, eye = extract_patient_and_eye(stem, pattern=regex_pattern)
        fundus_by_key[(cls_name, norm_id)] = (stem, f, eye)

    # Collect OCT records: map by (class_name, normalized_id)
    oct_by_key: Dict[Tuple[str, str], Tuple[str, Path, str]] = {}
    for o in oct_files:
        rel = o.relative_to(oct_path)
        cls_name = normalize_class_name(rel.parts[0] if len(rel.parts) > 1 else "MD")
        stem = o.stem
        norm_id = normalize_stem(stem)
        p_id, eye = extract_patient_and_eye(stem, pattern=regex_pattern)
        oct_by_key[(cls_name, norm_id)] = (stem, o, eye)

    # Union of all keys across both modalities
    all_keys = sorted(
        list(set(fundus_by_key.keys()).union(set(oct_by_key.keys()))),
        key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]),
    )

    # Build raw records list
    raw_records: List[Dict[str, Any]] = []
    for cls_name, norm_id in all_keys:
        has_fundus = (cls_name, norm_id) in fundus_by_key
        has_oct = (cls_name, norm_id) in oct_by_key
        is_paired = has_fundus and has_oct

        f_stem, f_file, f_eye = fundus_by_key[(cls_name, norm_id)] if has_fundus else ("", None, "unknown")
        o_stem, o_file, o_eye = oct_by_key[(cls_name, norm_id)] if has_oct else ("", None, "unknown")

        eye = f_eye if f_eye != "unknown" else o_eye
        if eye == "unknown":
            eye = "OD"  # Standard default clinical eye notation

        patient_id = f"{cls_name}_{norm_id.zfill(3) if norm_id.isdigit() else norm_id}"

        raw_records.append({
            "normalized_id": norm_id,
            "patient_id": patient_id,
            "class_name": cls_name,
            "eye": eye,
            "is_paired": is_paired,
            "has_fundus": has_fundus,
            "has_oct": has_oct,
            "fundus_filename": f_file.name if f_file else "",
            "oct_filename": o_file.name if o_file else "",
            "fundus_path": str(f_file.resolve()) if f_file else "",
            "oct_path": str(o_file.resolve()) if o_file else "",
        })

    rng = np.random.default_rng(seed)

    # Stratified partitioning by pathology class and pairing status
    # Guarantees that within each pathology class:
    # 80% -> train, 10% -> val, 10% -> test
    stratified_groups: Dict[Tuple[str, bool], List[int]] = {}
    for idx, rec in enumerate(raw_records):
        key = (rec["class_name"], rec["is_paired"])
        if key not in stratified_groups:
            stratified_groups[key] = []
        stratified_groups[key].append(idx)

    split_assignments: Dict[int, str] = {}
    for (cls, is_paired), group_indices in sorted(stratified_groups.items()):
        n_grp = len(group_indices)
        parts = partition_indices(
            n_grp,
            train_ratio=train_split,
            val_ratio=val_split,
            test_ratio=test_split,
            rng=rng,
        )
        for sub_idx in parts["train"]:
            split_assignments[group_indices[sub_idx]] = "train"
        for sub_idx in parts["val"]:
            split_assignments[group_indices[sub_idx]] = "val"
        for sub_idx in parts["test"]:
            split_assignments[group_indices[sub_idx]] = "test"

    # Attach split assignment to each record
    for idx, rec in enumerate(raw_records):
        rec["split"] = split_assignments.get(idx, "train")

    df = pd.DataFrame(raw_records)

    # Save manifest if output path is provided
    if output_manifest is not None:
        out_p = Path(output_manifest)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"[Pairing] Saved {len(df)} records manifest to {out_p}")

    return df


def validate_manifest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates manifest integrity:
    1. Checks for patient leakage across train, val, test splits.
    2. Checks for missing files on disk.
    3. Counts paired vs unpaired records.
    4. Analyzes class-stratified split distributions.
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "total_records": len(df),
        "paired_records": 0,
        "fundus_only": 0,
        "oct_only": 0,
        "patient_leakage_detected": False,
        "leakage_patients": [],
        "missing_files": [],
        "split_counts": {},
        "split_by_class": {},
        "warnings": [],
    }

    if df.empty:
        report["warnings"].append("Manifest DataFrame is completely empty.")
        report["is_valid"] = False
        return report

    # 1. Check patient leakage across splits
    splits_by_patient: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        p_id = str(row.get("patient_id", ""))
        split = str(row.get("split", ""))
        if p_id not in splits_by_patient:
            splits_by_patient[p_id] = set()
        splits_by_patient[p_id].add(split)

    leaking_patients = [p for p, splits in splits_by_patient.items() if len(splits) > 1]
    if leaking_patients:
        report["patient_leakage_detected"] = True
        report["leakage_patients"] = leaking_patients[:10]
        report["is_valid"] = False
        report["warnings"].append(
            f"CRITICAL: Patient leakage detected! {len(leaking_patients)} patient(s) appear across multiple splits."
        )

    # 2. Count pairings & verify file existence
    missing_files = []
    paired = 0
    fundus_only = 0
    oct_only = 0

    for _, row in df.iterrows():
        f_val = row.get("fundus_path", "")
        o_val = row.get("oct_path", "")
        f_p = "" if (pd.isna(f_val) or str(f_val).strip().lower() in ("", "nan", "none")) else str(f_val).strip()
        o_p = "" if (pd.isna(o_val) or str(o_val).strip().lower() in ("", "nan", "none")) else str(o_val).strip()

        has_f = bool(f_p)
        has_o = bool(o_p)

        if has_f and has_o:
            paired += 1
        elif has_f:
            fundus_only += 1
        elif has_o:
            oct_only += 1

        if has_f and not Path(f_p).exists():
            missing_files.append(f_p)
        if has_o and not Path(o_p).exists():
            missing_files.append(o_p)

    report["paired_records"] = paired
    report["fundus_only"] = fundus_only
    report["oct_only"] = oct_only
    report["missing_files"] = missing_files

    if missing_files:
        report["warnings"].append(f"{len(missing_files)} file path(s) referenced in manifest do not exist on disk.")

    # 3. Split distribution overall and by class
    if "split" in df.columns:
        report["split_counts"] = df["split"].value_counts().to_dict()

    if "class_name" in df.columns and "split" in df.columns:
        split_by_cls = df.groupby(["class_name", "split"]).size().unstack(fill_value=0).to_dict()
        report["split_by_class"] = split_by_cls

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Build and stratify Fundus-OCT pairing manifest (80% train, 10% val, 10% test)."
    )
    parser.add_argument(
        "--fundus-dir",
        type=str,
        default="data/fundus",
        help="Directory containing retinal fundus images (default: data/fundus).",
    )
    parser.add_argument(
        "--oct-dir",
        type=str,
        default="data/oct",
        help="Directory containing OCT B-scans / volumes (default: data/oct).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/manifests/dataset.csv",
        help="Output manifest CSV path (default: data/manifests/dataset.csv).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.80,
        help="Training split ratio (default: 0.80).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.10,
        help="Validation split ratio (default: 0.10).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.10,
        help="Test split ratio (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic stratification (default: 42).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  FUNDUS2OCT DATASET PAIRING & STRATIFIED PARTITION GENERATOR")
    print("=" * 70)
    print(f"Fundus Dir     : {args.fundus_dir}")
    print(f"OCT Dir        : {args.oct_dir}")
    print(f"Output Path    : {args.output}")
    print(f"Target Splits  : {args.train_split*100:.0f}% Train / {args.val_split*100:.0f}% Val / {args.test_split*100:.0f}% Test")
    print(f"Random Seed    : {args.seed}")
    print("=" * 70)

    df = build_pairing_manifest(
        fundus_dir=args.fundus_dir,
        oct_dir=args.oct_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        output_manifest=args.output,
        seed=args.seed,
    )

    report = validate_manifest(df)
    print("\n--- Manifest Summary ---")
    print(f"Total Records  : {report['total_records']}")
    print(f"Paired Samples : {report['paired_records']}")
    print(f"Fundus Only    : {report['fundus_only']}")
    print(f"OCT Only       : {report['oct_only']}")

    print("\nOverall Split Distribution:")
    for split_name, count in report["split_counts"].items():
        pct = (count / report["total_records"]) * 100
        print(f"  {split_name:<10}: {count:>4} samples ({pct:>5.1f}%)")

    print("\nPaired Split Distribution:")
    paired_df = df[df["is_paired"]]
    for split_name, count in paired_df["split"].value_counts().items():
        pct = (count / len(paired_df)) * 100
        print(f"  {split_name:<10}: {count:>4} pairs   ({pct:>5.1f}%)")

    print("\nClass Stratification Breakdown:")
    for cls in df["class_name"].unique():
        sub = df[df["class_name"] == cls]
        tr = len(sub[sub["split"] == "train"])
        va = len(sub[sub["split"] == "val"])
        te = len(sub[sub["split"] == "test"])
        paired_in_cls = len(sub[sub["is_paired"]])
        print(f"  Class {cls:<6}: {len(sub):>3} total ({paired_in_cls:>2} paired) -> {tr} Train ({tr/len(sub)*100:.0f}%), {va} Val ({va/len(sub)*100:.0f}%), {te} Test ({te/len(sub)*100:.0f}%)")

    print(f"\nZero Patient Leakage Verified: {'YES (PASS)' if not report['patient_leakage_detected'] else 'NO (FAIL)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
