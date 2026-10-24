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
