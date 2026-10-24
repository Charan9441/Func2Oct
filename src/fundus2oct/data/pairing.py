"""
src/fundus2oct/data/pairing.py

Pairing logic and manifest generator for matching 2D Retinal Fundus photographs
with 2D/3D Optical Coherence Tomography (OCT) records.
Guarantees strict patient-level partition isolation to prevent data leakage.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def build_pairing_manifest(
    fundus_dir: Path,
    oct_dir: Path,
    strategy: str = "stem_match",
    regex_pattern: Optional[str] = None,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds a patient-level paired manifest DataFrame.
    Guarantees:
    1. Zero patient leakage: All images for any single patient belong to exactly one split.
    2. Explicit tracking of paired, fundus-only, and oct-only samples.
    """
    fundus_files = scan_directory_for_images(Path(fundus_dir))
    oct_files = scan_directory_for_images(Path(oct_dir))

    # Map by key: (patient_id, eye, stem)
    fundus_map: Dict[Tuple[str, str, str], Path] = {}
    for f in fundus_files:
        p_id, eye = extract_patient_and_eye(f.name, pattern=regex_pattern)
        stem = f.stem
        key = (p_id, eye, stem) if strategy == "stem_match" else (p_id, eye, "")
        fundus_map[key] = f

    oct_map: Dict[Tuple[str, str, str], Path] = {}
    for o in oct_files:
        p_id, eye = extract_patient_and_eye(o.name, pattern=regex_pattern)
        stem = o.stem
        key = (p_id, eye, stem) if strategy == "stem_match" else (p_id, eye, "")
        oct_map[key] = o

    all_keys = sorted(list(set(fundus_map.keys()).union(set(oct_map.keys()))))

    # Collect unique patient IDs to stratify splits without leakage
    unique_patients = sorted(list({k[0] for k in all_keys}))
    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(unique_patients).tolist()

    n_patients = len(shuffled_patients)
    n_train = int(n_patients * train_split)
    n_val = int(n_patients * val_split)

    patient_to_split: Dict[str, str] = {}
    for i, p in enumerate(shuffled_patients):
        if i < n_train:
            patient_to_split[p] = "train"
        elif i < n_train + n_val:
            patient_to_split[p] = "val"
        else:
            patient_to_split[p] = "test"

    records = []
    for p_id, eye, stem in all_keys:
        f_path = fundus_map.get((p_id, eye, stem), None)
        o_path = oct_map.get((p_id, eye, stem), None)
        is_paired = (f_path is not None) and (o_path is not None)
        split = patient_to_split.get(p_id, "train")

        records.append({
            "patient_id": p_id,
            "eye": eye,
            "sample_key": stem or p_id,
            "is_paired": is_paired,
            "fundus_path": str(f_path) if f_path else "",
            "oct_path": str(o_path) if o_path else "",
            "split": split,
        })

    return pd.DataFrame(records)


def validate_manifest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates manifest integrity:
    1. Checks for patient leakage across train, val, test splits.
    2. Checks for missing files on disk.
    3. Counts paired vs unpaired records.
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
        report["leakage_patients"] = leaking_patients[:10]  # Show up to 10
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
        f_p = str(row.get("fundus_path", "")).strip()
        o_p = str(row.get("oct_path", "")).strip()

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

    # 3. Split distribution
    if "split" in df.columns:
        report["split_counts"] = df["split"].value_counts().to_dict()

    return report
