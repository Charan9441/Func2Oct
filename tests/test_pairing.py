"""
tests/test_pairing.py

Unit tests for dataset pairing, patient extraction,
and split stratification with zero patient leakage.
"""

from pathlib import Path
import pandas as pd
import pytest

from fundus2oct.data.pairing import (
    extract_patient_and_eye,
    build_pairing_manifest,
    validate_manifest,
)


def test_extract_patient_and_eye():
    """Tests extraction of patient ID and lateralization from standard naming conventions."""
    # Suffix notation
    p_id, eye = extract_patient_and_eye("patient012_OD.png")
    assert p_id == "patient012"
    assert eye == "OD"

    p_id, eye = extract_patient_and_eye("subj_55-OS.jpg")
    assert p_id == "subj_55"
    assert eye == "OS"

    # Regex pattern
    pattern = r"(?P<patient_id>\d+)_(?P<eye>OD|OS)"
    p_id, eye = extract_patient_and_eye("1042_OD.png", pattern=pattern)
    assert p_id == "1042"
    assert eye == "OD"


def test_validate_manifest_zero_leakage():
    """Verifies that validate_manifest passes when patients are strictly isolated."""
    df = pd.DataFrame([
        {"patient_id": "P01", "eye": "OD", "fundus_path": "a.png", "oct_path": "b.png", "split": "train"},
        {"patient_id": "P01", "eye": "OS", "fundus_path": "c.png", "oct_path": "d.png", "split": "train"},
        {"patient_id": "P02", "eye": "OD", "fundus_path": "e.png", "oct_path": "f.png", "split": "val"},
        {"patient_id": "P03", "eye": "OS", "fundus_path": "g.png", "oct_path": "h.png", "split": "test"},
    ])
    report = validate_manifest(df)
    assert report["is_valid"] is True
    assert report["patient_leakage_detected"] is False
    assert report["paired_records"] == 4


def test_validate_manifest_detects_leakage():
    """Verifies that validate_manifest catches patient appearing in multiple splits."""
    df_leak = pd.DataFrame([
        {"patient_id": "P01", "eye": "OD", "fundus_path": "a.png", "oct_path": "b.png", "split": "train"},
        {"patient_id": "P01", "eye": "OS", "fundus_path": "c.png", "oct_path": "d.png", "split": "val"},  # Leakage!
        {"patient_id": "P02", "eye": "OD", "fundus_path": "e.png", "oct_path": "f.png", "split": "test"},
    ])
    report = validate_manifest(df_leak)
    assert report["patient_leakage_detected"] is True
    assert report["is_valid"] is False
    assert "P01" in report["leakage_patients"]


def test_build_pairing_manifest_stratified_80_10_10(tmp_path):
    """Verifies that build_pairing_manifest partitions pairs into 80% train, 10% val, 10% test stratified by class."""
    # Create mock dataset with 2 classes: MD (10 samples) and OTHER (10 samples)
    fundus_dir = tmp_path / "fundus"
    oct_dir = tmp_path / "oct"
    out_manifest = tmp_path / "dataset.csv"

    (fundus_dir / "MD").mkdir(parents=True)
    (fundus_dir / "OTHER").mkdir(parents=True)
    (oct_dir / "MD").mkdir(parents=True)

    for i in range(1, 11):
        (fundus_dir / "MD" / f"{i:03d}.png").touch()
        (oct_dir / "MD" / f"{i:04d}.png").touch()
        (fundus_dir / "OTHER" / f"{i:03d}.png").touch()

    df = build_pairing_manifest(
        fundus_dir=fundus_dir,
        oct_dir=oct_dir,
        train_split=0.80,
        val_split=0.10,
        test_split=0.10,
        output_manifest=out_manifest,
        seed=42,
    )

    assert out_manifest.exists()
    assert len(df) == 20
    assert "split" in df.columns

    # Verify paired records (MD only in this test, 10 samples)
    paired_df = df[df["is_paired"]]
    assert len(paired_df) == 10
    paired_splits = paired_df["split"].value_counts().to_dict()
    assert paired_splits["train"] == 8   # 80% of 10
    assert paired_splits["val"] == 1     # 10% of 10
    assert paired_splits["test"] == 1    # 10% of 10

    # Verify overall records
    splits = df["split"].value_counts().to_dict()
    assert splits["train"] == 16  # 80% of 20
    assert splits["val"] == 2     # 10% of 20
    assert splits["test"] == 2    # 10% of 20

    # Verify zero leakage
    report = validate_manifest(df)
    assert report["is_valid"] is True
    assert report["patient_leakage_detected"] is False
