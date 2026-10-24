"""
tests/test_compare_models.py

Comprehensive test suite for Model A vs. Model B comparison and AUROC uplift evaluation:
1. Correct uplift calculations (absolute difference, percentage points, relative uplift).
2. Correct matching and alignment of sample IDs across models.
3. Detection and rejection of mismatched test cohorts.
4. Robust validation against missing predictions, invalid/out-of-bounds probabilities, and duplicates.
5. Graceful handling of single-class evaluation cohorts (AUROC marked unavailable without crash).
6. Complete artifact generation (CSV, JSON, AUROC uplift plot, metrics comparison plot).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fundus2oct.evaluation.compare_models import (
    compare_model_predictions,
    compute_paired_comparison,
    compute_single_model_metrics,
    export_comparison_reports,
    plot_auroc_uplift,
    plot_metrics_comparison,
    validate_inputs,
)


@pytest.fixture
def sample_predictions():
    """Generates aligned mock predictions for Model A and Model B."""
    n = 20
    rng = np.random.default_rng(42)
    sample_ids = [f"sample_{i:03d}" for i in range(n)]
    y_true = np.array([0] * 10 + [1] * 10)  # 10 negative, 10 positive

    # Model A: Base predictions (AUROC ~ 0.70)
    prob_a = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.15, size=n), 0.05, 0.95)
    # Model B: Improved predictions (AUROC ~ 0.78, uplift ~ 5-10%)
    prob_b = np.clip(0.2 + 0.6 * y_true + rng.normal(0, 0.12, size=n), 0.05, 0.95)

    df_a = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_true,
        "pred_prob_MD": prob_a,
    })
    df_b = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_true,
        "pred_prob_MD": prob_b,
    })
    return df_a, df_b


def test_uplift_calculations():
    """Tests exact mathematical formulas for absolute difference, percentage points, and relative uplift."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Probabilities constructed for known AUROCs:
    # Model A: 0.60
    # Model B: 0.80
    y_prob_a = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.8])
    y_prob_b = np.array([0.1, 0.2, 0.4, 0.3, 0.7, 0.8, 0.6, 0.9])

    res = compute_paired_comparison(y_true, y_prob_a, y_prob_b, n_bootstraps=100, seed=42)

    assert res["comparison_possible"] is True
    auc_a = res["model_a_auroc"]
    auc_b = res["model_b_auroc"]

    expected_abs_diff = auc_b - auc_a
    expected_pp_change = (auc_b - auc_a) * 100.0
    expected_rel_uplift = ((auc_b - auc_a) / auc_a) * 100.0

    assert res["absolute_auroc_diff"] == pytest.approx(expected_abs_diff, abs=1e-3)
    assert res["percentage_points_change"] == pytest.approx(expected_pp_change, abs=1e-1)
    assert res["relative_uplift_pct"] == pytest.approx(expected_rel_uplift, abs=1e-1)

    # Target 4-8% investigation structure
    target_info = res["target_investigation"]
    assert "target_status" in target_info
    assert "target_met" in target_info
    assert target_info["target_range_pct"] == [4.0, 8.0]


def test_sample_id_matching_and_alignment(sample_predictions):
    """Verifies that sample IDs are correctly matched and sorted even if shuffled in input DataFrames."""
    df_a, df_b = sample_predictions

    # Shuffle df_b ordering
    df_b_shuffled = df_b.sample(frac=1.0, random_state=123).reset_index(drop=True)

    df_a_aligned, df_b_aligned = validate_inputs(df_a, df_b_shuffled)

    assert list(df_a_aligned["sample_id"]) == list(df_b_aligned["sample_id"])
    assert list(df_a_aligned["sample_id"]) == sorted(df_a["sample_id"].tolist())
    assert np.array_equal(df_a_aligned["true_label"], df_b_aligned["true_label"])


def test_mismatched_test_cohorts_error(sample_predictions):
    """Verifies ValueError when Model A and Model B test sets contain different patient samples."""
    df_a, df_b = sample_predictions
    # Change one sample ID in df_b
    df_b_mismatched = df_b.copy()
    df_b_mismatched.loc[0, "sample_id"] = "different_patient_999"

    with pytest.raises(ValueError, match="Mismatched test cohorts"):
        validate_inputs(df_a, df_b_mismatched)


def test_duplicate_sample_ids_error(sample_predictions):
    """Verifies ValueError when duplicate sample IDs are present."""
    df_a, df_b = sample_predictions
    df_a_dup = pd.concat([df_a, df_a.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        validate_inputs(df_a_dup, df_b)


def test_missing_required_columns_error(sample_predictions):
    """Verifies ValueError when required columns are missing."""
    df_a, df_b = sample_predictions
    df_a_missing = df_a.drop(columns=["pred_prob_MD"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_inputs(df_a_missing, df_b)


def test_label_mismatch_error(sample_predictions):
    """Verifies ValueError when ground truth labels contradict each other."""
    df_a, df_b = sample_predictions
    df_b_bad_label = df_b.copy()
    df_b_bad_label.loc[0, "true_label"] = 1 - df_b_bad_label.loc[0, "true_label"]

    with pytest.raises(ValueError, match="Ground-truth label mismatch"):
        validate_inputs(df_a, df_b_bad_label)


def test_out_of_bounds_probabilities_error(sample_predictions):
    """Verifies ValueError when probabilities are outside [0, 1] or NaN."""
    df_a, df_b = sample_predictions

    # Case 1: Probability > 1.0
    df_a_high = df_a.copy()
    df_a_high.loc[0, "pred_prob_MD"] = 1.05
    with pytest.raises(ValueError, match="out of bounds"):
        validate_inputs(df_a_high, df_b)

    # Case 2: Probability < 0.0
    df_a_low = df_a.copy()
    df_a_low.loc[0, "pred_prob_MD"] = -0.01
    with pytest.raises(ValueError, match="out of bounds"):
        validate_inputs(df_a_low, df_b)

    # Case 3: NaN probability
    df_b_nan = df_b.copy()
    df_b_nan.loc[0, "pred_prob_MD"] = np.nan
    with pytest.raises(ValueError, match="NaN or Infinite"):
        validate_inputs(df_a, df_b_nan)


def test_single_class_evaluation_cohort():
    """Verifies that single-class cohorts gracefully mark AUROC unavailable without crashing."""
    y_true_single = np.array([1, 1, 1, 1, 1])
    y_prob = np.array([0.7, 0.8, 0.6, 0.9, 0.85])

    metrics = compute_single_model_metrics(y_true_single, y_prob)

    assert metrics["auroc_available"] is False
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert "Single class" in metrics["reason"]

    # Comparative paired comparison
    comp = compute_paired_comparison(y_true_single, y_prob, y_prob)
    assert comp["comparison_possible"] is False
    assert "Single class" in comp["reason"]


def test_artifact_generation(tmp_path, sample_predictions):
    """Verifies that all 4 required artifacts are generated and valid."""
    df_a, df_b = sample_predictions
    out_dir = tmp_path / "eval_reports"

    res = compare_model_predictions(
        preds_a_input=df_a,
        preds_b_input=df_b,
        output_dir=out_dir,
        n_bootstraps=50,
        seed=42,
    )

    csv_file = Path(res["artifacts"]["csv"])
    json_file = Path(res["artifacts"]["json"])
    auroc_png = Path(res["artifacts"]["auroc_png"])
    metrics_png = Path(res["artifacts"]["metrics_png"])

    # 1. Check CSV artifact
    assert csv_file.exists()
    assert csv_file.stat().st_size > 50
    df_res = pd.read_csv(csv_file)
    expected_cols = {"metric", "model_a_fundus", "model_b_multimodal", "absolute_difference", "percentage_points_change", "relative_uplift_pct"}
    assert expected_cols.issubset(set(df_res.columns))
    assert "AUROC" in df_res["metric"].values

    # 2. Check JSON artifact
    assert json_file.exists()
    assert json_file.stat().st_size > 100
    import json
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "model_a_fundus" in data
    assert "model_b_multimodal" in data
    assert "comparative_metrics" in data
    assert "interpretation" in data

    # 3. Check PNG plots
    assert auroc_png.exists()
    assert auroc_png.stat().st_size > 5000  # Non-empty, valid image

    assert metrics_png.exists()
    assert metrics_png.stat().st_size > 5000  # Non-empty, valid image

    # 4. Check Report text
    assert "MODEL COMPARISON & AUROC UPLIFT EVALUATION REPORT" in res["report_text"]
    assert "Clinical Disclaimer" in res["report_text"]
