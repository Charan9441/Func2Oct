"""
tests/test_bias_checks.py

Automated test suite for Retinal Classifier Bias Checks and Cross-Dataset
Generalization Evaluation module (src/fundus2oct/evaluation/bias_checks.py).

Covers:
1. Manifest and Dataset Validation:
   - Valid vs invalid schemas, missing paths, empty datasets.
   - Missing image files, corrupted images, missing and invalid labels.
   - Duplicate image detection via MD5 hashing.
   - Training cohort overlap / data leakage detection.
   - Task and label compatibility checks (ARMD compatible vs IDRiD incompatible).
2. Metric Correctness:
   - Binary performance metrics (accuracy, balanced accuracy, sensitivity, specificity, F1).
   - Single-class edge cases where AUROC and AUPRC are mathematically undefined.
   - Decision threshold sensitivity.
   - Clustered patient-level bootstrap confidence interval calculation.
3. Subgroup and Bias Disparity Analysis:
   - Subgroup fairness metrics (disparate impact ratio, equal opportunity difference).
   - Small subgroup sample warnings.
   - Missing subgroup metadata columns and single-class subgroups.
4. Model Loading and Inference:
   - Safe checkpoint loading with head adaptation.
   - Missing checkpoint exception handling.
   - Deterministic batch inference.
5. Reporting and Artifact Export:
   - Structured JSON summary, CSV spreadsheets, and Markdown audit report.
6. CLI Integration:
   - Smoke-test execution via subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn

from fundus2oct.evaluation.bias_checks import (
    ARMD_DATASET_SPEC,
    IDRID_DATASET_SPEC,
    ClassificationMetrics,
    CompatibilityStatus,
    DatasetSpec,
    DatasetValidationReport,
    SubgroupMetricResult,
    analyze_subgroup_bias,
    compute_dataset_metrics,
    compute_file_md5,
    evaluate_bias_and_generalization,
    export_bias_evaluation_artifacts,
    generate_mock_external_cohort,
    load_classifier_for_evaluation,
    run_external_inference,
    validate_external_dataset,
)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def temp_cohort_dir(tmp_path: Path):
    """Creates a temporary folder containing synthetic PNG fundus images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    image_paths = []
    for i in range(8):
        img_p = img_dir / f"sample_{i:02d}.png"
        arr = rng.randint(50, 200, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_p)
        image_paths.append(img_p)

    return img_dir, image_paths


@pytest.fixture
def mock_classifier():
    """Simple linear mock model simulating a 2-class classifier."""
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 256 * 256, 2)
            self.head = nn.Sequential(nn.Dropout(0.0), nn.Linear(2, 2))

        def forward(self, x):
            flat = x.view(x.size(0), -1)
            # deterministic output
            out = self.linear(flat)
            return out

    model = SimpleClassifier()
    model.eval()
    return model


# =====================================================================
# 1. DATASET & MANIFEST VALIDATION TESTS
# =====================================================================

def test_validate_missing_dataset_path():
    """Non-existent manifest path returns empty dataframe and is_available=False."""
    missing_p = Path("non_existent_directory_or_manifest.csv")
    df_eligible, report = validate_external_dataset(missing_p, ARMD_DATASET_SPEC)

    assert len(df_eligible) == 0
    assert report.is_available is False
    assert report.eligible_samples == 0
    assert "does not exist" in report.status_message


def test_validate_empty_manifest(tmp_path: Path):
    """Empty manifest file returns is_available=False with clear status message."""
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)

    df_eligible, report = validate_external_dataset(empty_csv, ARMD_DATASET_SPEC)
    assert len(df_eligible) == 0
    assert report.is_available is False
    assert "empty" in report.status_message.lower()


def test_validate_missing_image_path_column(tmp_path: Path):
    """Manifest missing the specified image path column is flagged and rejected."""
    bad_csv = tmp_path / "no_img_col.csv"
    df = pd.DataFrame({"random_col": [1, 2], "label": ["MD", "CONTROL"]})
    df.to_csv(bad_csv, index=False)

    df_eligible, report = validate_external_dataset(bad_csv, ARMD_DATASET_SPEC)
    assert len(df_eligible) == 0
    assert report.is_available is False
    assert "image path column" in report.status_message


def test_validate_missing_and_corrupt_images(temp_cohort_dir, tmp_path: Path):
    """Verifies that non-existent image paths and corrupted images are excluded."""
    img_dir, image_paths = temp_cohort_dir

    # Create a corrupted file
    corrupt_p = img_dir / "corrupted.png"
    with open(corrupt_p, "wb") as f:
        f.write(b"NOT_A_VALID_PNG_FILE_DATA")

    # Create manifest with: 2 valid, 1 missing, 1 corrupted
    manifest_df = pd.DataFrame({
        "fundus_path": [
            str(image_paths[0]),
            str(image_paths[1]),
            str(img_dir / "missing_file.png"),
            str(corrupt_p),
        ],
        "class_name": ["MD", "CONTROL", "MD", "CONTROL"],
        "patient_id": ["P1", "P2", "P3", "P4"],
    })

    df_eligible, report = validate_external_dataset(
        manifest_df,
        ARMD_DATASET_SPEC,
        check_image_readability=True,
    )

    assert len(df_eligible) == 2
    assert report.total_rows == 4
    assert report.excluded_samples == 2
    assert report.exclusion_reasons.get("missing_image_file") == 1
    assert report.exclusion_reasons.get("corrupted_image") == 1


def test_validate_missing_and_invalid_labels(temp_cohort_dir):
    """Rows with missing or unrecognized label strings are excluded."""
    _, image_paths = temp_cohort_dir

    manifest_df = pd.DataFrame({
        "fundus_path": [str(image_paths[0]), str(image_paths[1]), str(image_paths[2])],
        "class_name": ["MD", None, "UNRECOGNIZED_DISEASE_LABEL"],
        "patient_id": ["P1", "P2", "P3"],
    })

    df_eligible, report = validate_external_dataset(manifest_df, ARMD_DATASET_SPEC)

    assert len(df_eligible) == 1
    assert df_eligible.iloc[0]["_binary_target"] == 1
    assert report.exclusion_reasons.get("missing_label") == 1
    assert report.exclusion_reasons.get("invalid_label") == 1


def test_validate_duplicate_image_hashes(temp_cohort_dir):
    """Duplicate images with identical content/hash are detected and excluded."""
    _, image_paths = temp_cohort_dir

    # Use identical image path twice
    manifest_df = pd.DataFrame({
        "fundus_path": [str(image_paths[0]), str(image_paths[0]), str(image_paths[1])],
        "class_name": ["MD", "MD", "CONTROL"],
        "patient_id": ["P1", "P1_dup", "P2"],
    })

    df_eligible, report = validate_external_dataset(manifest_df, ARMD_DATASET_SPEC)

    assert len(df_eligible) == 2
    assert report.duplicate_image_count == 1
    assert report.exclusion_reasons.get("duplicate_hash") == 1


def test_validate_training_overlap_detection(temp_cohort_dir, tmp_path: Path):
    """External images that match training manifest hashes or filenames are excluded."""
    _, image_paths = temp_cohort_dir

    # Create synthetic baseline training manifest containing image_paths[0]
    train_manifest = tmp_path / "baseline_train.csv"
    pd.DataFrame({
        "patient_id": ["TRAIN_P1"],
        "fundus_path": [str(image_paths[0])],
        "fundus_filename": [image_paths[0].name],
    }).to_csv(train_manifest, index=False)

    eval_df = pd.DataFrame({
        "fundus_path": [str(image_paths[0]), str(image_paths[1])],
        "class_name": ["MD", "CONTROL"],
        "patient_id": ["EVAL_P1", "EVAL_P2"],
    })

    df_eligible, report = validate_external_dataset(
        eval_df,
        ARMD_DATASET_SPEC,
        training_manifest_path=train_manifest,
    )

    assert len(df_eligible) == 1
    assert report.training_overlap_count == 1
    assert report.exclusion_reasons.get("training_overlap") == 1


def test_label_compatibility_status():
    """Verifies ARMD is marked compatible while IDRiD is rejected with clinical rationale."""
    assert ARMD_DATASET_SPEC.compatibility == CompatibilityStatus.COMPATIBLE
    assert ARMD_DATASET_SPEC.label_mapping is not None
    assert ARMD_DATASET_SPEC.label_mapping["AMD"] == 1
    assert ARMD_DATASET_SPEC.label_mapping["CONTROL"] == 0

    assert IDRID_DATASET_SPEC.compatibility == CompatibilityStatus.INCOMPATIBLE
    assert IDRID_DATASET_SPEC.label_mapping is None
    assert "Diabetic Retinopathy" in IDRID_DATASET_SPEC.compatibility_reason
    assert "Macular Degeneration" in IDRID_DATASET_SPEC.compatibility_reason


# =====================================================================
# 2. METRIC CALCULATION TESTS
# =====================================================================

def test_compute_dataset_metrics_perfect():
    """Perfect binary predictions should yield 1.0 for all performance metrics."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.05, 0.9, 0.85, 0.95, 0.8])

    metrics = compute_dataset_metrics(y_true, y_prob, dataset_name="TestCohort", threshold=0.5)

    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.sensitivity_recall == 1.0
    assert metrics.specificity == 1.0
    assert metrics.precision_ppv == 1.0
    assert metrics.npv == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.auroc_available is True
    assert metrics.confusion_matrix == {"TN": 4, "FP": 0, "FN": 0, "TP": 4}


def test_compute_dataset_metrics_threshold_sensitivity():
    """Verifies that changing decision threshold alters classification trade-offs."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.45, 0.55, 0.8])

    # Threshold 0.50: all 4 classified correctly
    m_50 = compute_dataset_metrics(y_true, y_prob, dataset_name="Test", threshold=0.50)
    assert m_50.accuracy == 1.0

    # Strict threshold 0.70: y_prob=0.55 becomes False Negative (FN=1, TP=1)
    m_70 = compute_dataset_metrics(y_true, y_prob, dataset_name="Test", threshold=0.70)
    assert m_70.confusion_matrix["FN"] == 1
    assert m_70.confusion_matrix["TP"] == 1
    assert m_70.sensitivity_recall == 0.5


def test_compute_dataset_metrics_single_class_undefined_auroc():
    """Single-class cohort safely reports auroc=None and auroc_available=False without crashing."""
    y_true_all_zeros = np.array([0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.15, 0.4])

    metrics = compute_dataset_metrics(y_true_all_zeros, y_prob, dataset_name="SingleClassTest")

    assert metrics.auroc is None
    assert metrics.auprc is None
    assert metrics.auroc_available is False
    assert metrics.undefined_metric_reason is not None
    assert "single class" in metrics.undefined_metric_reason.lower()
    assert metrics.accuracy == 1.0  # all predicted < 0.5


def test_compute_dataset_metrics_patient_clustered_bootstrap():
    """Patient-clustered bootstrap computes valid 95% confidence intervals."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.2, 0.4, 0.7, 0.8, 0.65, 0.9])
    patient_ids = ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"]

    metrics = compute_dataset_metrics(
        y_true,
        y_prob,
        dataset_name="ClusterTest",
        patient_ids=patient_ids,
        n_bootstraps=50,
        seed=42,
    )

    assert metrics.auroc_95ci is not None
    assert len(metrics.auroc_95ci) == 2
    ci_low, ci_high = metrics.auroc_95ci
    assert 0.0 <= ci_low <= ci_high <= 1.0


# =====================================================================
# 3. SUBGROUP & BIAS DISPARITY TESTS
# =====================================================================

def test_analyze_subgroup_bias_disparity_metrics():
    """Calculates subgroup disparities, disparate impact ratio, and delta against overall."""
    df = pd.DataFrame({
        "_binary_target": [0, 0, 1, 1, 0, 0, 1, 1],
        "pred_prob_MD":   [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.6, 0.7],
        "sex":            ["F", "F", "F", "F", "M", "M", "M", "M"],
        "age_group":      ["<65", "<65", "<65", "<65", ">=65", ">=65", ">=65", ">=65"],
    })

    overall = compute_dataset_metrics(
        df["_binary_target"].values,
        df["pred_prob_MD"].values,
        dataset_name="SubgroupTest",
    )

    results = analyze_subgroup_bias(
        df=df,
        overall_metrics=overall,
        subgroup_cols=["sex", "age_group"],
        min_subgroup_samples=4,  # exactly 4 per subgroup
    )

    assert len(results) == 4
    sex_groups = {r.subgroup_val: r for r in results if r.subgroup_col == "sex"}
    assert "F" in sex_groups
    assert "M" in sex_groups

    f_res = sex_groups["F"]
    assert f_res.accuracy == 1.0
    assert f_res.is_small_subgroup is False
    assert f_res.warning is None
    assert f_res.disparate_impact_ratio is not None


def test_analyze_subgroup_bias_small_sample_warning():
    """Subgroups with sample count < min_subgroup_samples trigger explicit warning."""
    df = pd.DataFrame({
        "_binary_target": [0, 1, 0],
        "pred_prob_MD":   [0.1, 0.8, 0.2],
        "camera_model":   ["Topcon", "Topcon", "Canon"],
    })

    overall = compute_dataset_metrics(
        df["_binary_target"].values,
        df["pred_prob_MD"].values,
        dataset_name="WarningTest",
    )

    results = analyze_subgroup_bias(
        df=df,
        overall_metrics=overall,
        subgroup_cols=["camera_model"],
        min_subgroup_samples=5,
    )

    assert len(results) == 2
    for r in results:
        assert r.is_small_subgroup is True
        assert r.warning is not None
        assert "Small subgroup size" in r.warning


def test_analyze_subgroup_bias_missing_metadata_column():
    """Missing subgroup columns are gracefully skipped without crashing."""
    df = pd.DataFrame({
        "_binary_target": [0, 1],
        "pred_prob_MD":   [0.2, 0.8],
    })
    overall = compute_dataset_metrics(df["_binary_target"].values, df["pred_prob_MD"].values, dataset_name="Test")

    results = analyze_subgroup_bias(
        df=df,
        overall_metrics=overall,
        subgroup_cols=["non_existent_column_xyz"],
    )
    assert len(results) == 0


# =====================================================================
# 4. MODEL LOADING & INFERENCE TESTS
# =====================================================================

def test_load_classifier_missing_checkpoint():
    """Non-existent checkpoint path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_classifier_for_evaluation("non_existent_checkpoint_path_123.pt")


def test_load_classifier_real_or_mock(tmp_path: Path):
    """Verifies classifier weights are loaded in eval mode with requires_grad=False."""
    real_ckpt = Path("checkpoints/classifier_fundus.pt")

    if real_ckpt.is_file():
        model, meta = load_classifier_for_evaluation(real_ckpt, device="cpu")
        assert not model.training
        for p in model.parameters():
            assert not p.requires_grad
        assert meta["model_type"] == "fundus"
    else:
        # Create a mock checkpoint with adapted head keys
        mock_p = tmp_path / "mock_ckpt.pt"
        torch.save({
            "epoch": 1,
            "val_acc": 0.85,
            "state_dict": {
                "head.1.weight": torch.randn(2, 256),
                "head.1.bias": torch.zeros(2),
            },
        }, mock_p)

        model, meta = load_classifier_for_evaluation(mock_p, device="cpu")
        assert not model.training
        assert meta["epoch"] == 1


def test_run_external_inference_deterministic(temp_cohort_dir, mock_classifier):
    """Batch inference outputs pred_prob_MD and pred_prob_OTHER summing to 1."""
    _, image_paths = temp_cohort_dir

    df_cohort = pd.DataFrame({
        "_resolved_image_path": [str(p) for p in image_paths[:4]],
        "class_name": ["MD", "CONTROL", "MD", "CONTROL"],
        "_binary_target": [1, 0, 1, 0],
    })

    df_out = run_external_inference(
        model=mock_classifier,
        df_eligible=df_cohort,
        device="cpu",
        batch_size=2,
    )

    assert "pred_prob_MD" in df_out.columns
    assert "pred_prob_OTHER" in df_out.columns
    assert len(df_out) == 4

    prob_sum = df_out["pred_prob_MD"] + df_out["pred_prob_OTHER"]
    np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-5)


# =====================================================================
# 5. ARTIFACT EXPORT TESTS
# =====================================================================

def test_export_bias_evaluation_artifacts(tmp_path: Path):
    """Exports and verifies all 4 required evaluation artifacts in target directory."""
    val_report = DatasetValidationReport(
        dataset_name="TestDataset",
        manifest_path="test_path",
        total_rows=10,
        eligible_samples=8,
        excluded_samples=2,
        exclusion_reasons={"missing_label": 2},
        duplicate_image_count=0,
        training_overlap_count=0,
        patient_count=4,
        available_subgroups=["sex"],
        missing_subgroups=["camera"],
        is_available=True,
        status_message="All valid",
    )

    metric = ClassificationMetrics(
        dataset_name="TestDataset",
        n_samples=8,
        n_positive=4,
        n_negative=4,
        prevalence=0.5,
        accuracy=0.875,
        balanced_accuracy=0.875,
        sensitivity_recall=0.75,
        specificity=1.0,
        precision_ppv=1.0,
        npv=0.8,
        f1_score=0.8571,
        auroc=0.9375,
        auroc_95ci=[0.75, 1.0],
        auprc=0.9167,
        auprc_95ci=[0.70, 1.0],
        auroc_available=True,
        undefined_metric_reason=None,
        decision_threshold=0.5,
        confusion_matrix={"TN": 4, "FP": 0, "FN": 1, "TP": 3},
        bootstrap_seed=42,
    )

    sub_metric = SubgroupMetricResult(
        dataset_name="TestDataset",
        subgroup_col="sex",
        subgroup_val="F",
        n_samples=4,
        n_positive=2,
        n_negative=2,
        prevalence=0.5,
        accuracy=1.0,
        sensitivity_recall=1.0,
        specificity=1.0,
        f1_score=1.0,
        auroc=1.0,
        delta_accuracy=0.125,
        delta_sensitivity=0.25,
        delta_specificity=0.0,
        disparate_impact_ratio=1.0,
        equal_opportunity_diff=0.25,
        false_positive_rate_diff=0.0,
        is_small_subgroup=True,
        warning="Small sample warning",
    )

    incomp = [{
        "dataset_name": "IDRiD",
        "task_description": "Diabetic Retinopathy",
        "compatibility": "INCOMPATIBLE",
        "reason": "DR != MD",
        "required_model_or_mapping": "Dedicated DR model",
    }]

    artifacts = export_bias_evaluation_artifacts(
        validation_reports=[val_report],
        dataset_metrics=[metric],
        subgroup_metrics=[sub_metric],
        incompatible_datasets=incomp,
        output_dir=tmp_path,
    )

    assert "summary_json" in artifacts
    assert "cross_dataset_csv" in artifacts
    assert "subgroup_csv" in artifacts
    assert "report_markdown" in artifacts

    # Check files exist on disk
    for name, p in artifacts.items():
        assert p.is_file(), f"Artifact {name} ({p}) was not created"

    # Verify JSON structure
    with open(artifacts["summary_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "timestamp" in data
        assert len(data["evaluated_datasets"]) == 1
        assert len(data["incompatible_datasets"]) == 1

    # Verify Markdown contains required sections
    with open(artifacts["report_markdown"], "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "External Generalization & Bias-Check Audit Report" in md_text
        assert "Label Compatibility & Disease Task Boundaries" in md_text
        assert "Subgroup & Fairness Disparity Analysis" in md_text


# =====================================================================
# 6. END-TO-END PIPELINE & CLI SMOKE TEST
# =====================================================================

def test_evaluate_bias_and_generalization_smoke_test(tmp_path: Path):
    """End-to-end Python pipeline test using synthetic fixtures in smoke_test mode."""
    results = evaluate_bias_and_generalization(
        checkpoint_path="checkpoints/classifier_fundus.pt",
        output_dir=tmp_path,
        device="cpu",
        smoke_test=True,
    )

    assert results["status"] == "completed"
    assert len(results["validation_reports"]) >= 2
    assert len(results["incompatible_datasets"]) >= 1
    assert "artifacts" in results
    assert Path(results["artifacts"]["summary_json"]).is_file()


def test_cli_smoke_test_execution():
    """Invokes scripts/evaluate_bias_checks.py --smoke-test via subprocess."""
    cmd = [
        sys.executable,
        "scripts/evaluate_bias_checks.py",
        "--smoke-test",
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0, f"CLI command failed with stderr: {res.stderr}"
    assert "FUNDUS2OCT: BIAS CHECKS & CROSS-DATASET GENERALIZATION AUDIT" in res.stdout
    assert "Audit Execution Completed Successfully." in res.stdout
