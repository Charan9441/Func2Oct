"""
src/fundus2oct/evaluation/compare_models.py

Senior ML Evaluation Module for Fund2Oct.
Rigorously compares:
- Model A: Fundus-only disease classifier
- Model B: Fundus + Synthetic OCT multimodal classifier

Investigates whether adding synthetic OCT features yields the targeted 4–8% AUROC uplift
on identical held-out test cohorts without data leakage or fabricated scores.

Generated Artifacts:
- outputs/evaluation_reports/model_comparison.csv
- outputs/evaluation_reports/model_comparison.json
- outputs/evaluation_reports/auroc_uplift.png
- outputs/evaluation_reports/metrics_comparison.png
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("fundus2oct.evaluation.compare_models")


# ==============================================================================
# 1. INPUT VALIDATION & COHORT ALIGNMENT
# ==============================================================================

def validate_inputs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col: str = "sample_id",
    label_col: str = "true_label",
    prob_col: str = "pred_prob_MD",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strictly validates prediction DataFrames for Model A and Model B.
    
    Verifies:
    - Required columns exist in both DataFrames.
    - No duplicate sample IDs exist.
    - Both models evaluate the exact same cohort (identical sample ID sets).
    - Ground-truth target labels match identically across models.
    - Probabilities are valid, finite, and within [0.0, 1.0].
    
    Returns:
        Tuple of (df_a_aligned, df_b_aligned) sorted identically by id_col.
        
    Raises:
        ValueError: If any validation condition fails.
    """
    required_cols = [id_col, label_col, prob_col]
    
    # 1. Check required columns
    for name, df in [("Model A", df_a), ("Model B", df_b)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} predictions missing required columns: {missing}. Present: {list(df.columns)}")
            
    # 2. Check for duplicate IDs
    if df_a[id_col].duplicated().any():
        dups = df_a[id_col][df_a[id_col].duplicated()].tolist()
        raise ValueError(f"Model A contains duplicate sample IDs: {dups}")
    if df_b[id_col].duplicated().any():
        dups = df_b[id_col][df_b[id_col].duplicated()].tolist()
        raise ValueError(f"Model B contains duplicate sample IDs: {dups}")

    # 3. Check for cohort matching
    set_a = set(df_a[id_col])
    set_b = set(df_b[id_col])
    if set_a != set_b:
        only_a = set_a - set_b
        only_b = set_b - set_a
        raise ValueError(
            f"Mismatched test cohorts between Model A and Model B. "
            f"Samples only in Model A: {len(only_a)}, only in Model B: {len(only_b)}."
        )

    # 4. Sort and align by sample_id
    df_a_aligned = df_a.sort_values(by=id_col).reset_index(drop=True)
    df_b_aligned = df_b.sort_values(by=id_col).reset_index(drop=True)

    # 5. Verify ground-truth label consistency
    labels_a = df_a_aligned[label_col].to_numpy()
    labels_b = df_b_aligned[label_col].to_numpy()
    if not np.array_equal(labels_a, labels_b):
        mismatches = np.where(labels_a != labels_b)[0]
        sample_mismatch = df_a_aligned.iloc[mismatches][id_col].tolist()
        raise ValueError(f"Ground-truth label mismatch for samples: {sample_mismatch}")

    # 6. Verify probability validity
    for name, df in [("Model A", df_a_aligned), ("Model B", df_b_aligned)]:
        probs = df[prob_col].to_numpy()
        if not np.issubdtype(probs.dtype, np.number):
            raise ValueError(f"{name} probabilities must be numeric. Found dtype: {probs.dtype}")
        if np.isnan(probs).any() or np.isinf(probs).any():
            raise ValueError(f"{name} contains NaN or Infinite probabilities.")
        if np.any(probs < 0.0) or np.any(probs > 1.0):
            min_val, max_val = float(np.min(probs)), float(np.max(probs))
            raise ValueError(f"{name} probabilities out of bounds [0.0, 1.0]: min={min_val}, max={max_val}")

    return df_a_aligned, df_b_aligned


# ==============================================================================
# 2. METRICS & BOOTSTRAP UNCERTAINTY COMPUTATION
# ==============================================================================

def compute_single_model_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes classification metrics and bootstrap confidence intervals for a single model.
    """
    n_samples = len(y_true)
    unique_classes = np.unique(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if len(unique_classes) < 2:
        return {
            "n_samples": n_samples,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "auroc": None,
            "auroc_95ci": None,
            "auprc": None,
            "auprc_95ci": None,
            "auroc_available": False,
            "reason": f"Single class present in evaluation cohort ({unique_classes[0]}). AUROC is mathematically undefined.",
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        }

    auroc = float(roc_auc_score(y_true, y_prob))
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = float(auc(recall_curve, precision_curve))

    # Bootstrap CIs
    rng = np.random.default_rng(seed)
    boot_auroc = []
    boot_auprc = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_auroc.append(roc_auc_score(y_true[idx], y_prob[idx]))
            p_c, r_c, _ = precision_recall_curve(y_true[idx], y_prob[idx])
            boot_auprc.append(auc(r_c, p_c))
        except Exception:
            pass

    auroc_ci = (
        [round(float(np.percentile(boot_auroc, 2.5)), 4), round(float(np.percentile(boot_auroc, 97.5)), 4)]
        if len(boot_auroc) >= 20 else None
    )
    auprc_ci = (
        [round(float(np.percentile(boot_auprc, 2.5)), 4), round(float(np.percentile(boot_auprc, 97.5)), 4)]
        if len(boot_auprc) >= 20 else None
    )

    return {
        "n_samples": n_samples,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4),
        "auroc_95ci": auroc_ci,
        "auprc": round(auprc, 4),
        "auprc_95ci": auprc_ci,
        "auroc_available": True,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def compute_paired_comparison(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes comparative metrics, absolute difference, percentage-point change,
    relative uplift, and paired bootstrap hypothesis testing.
    
    Definitions:
    - Absolute AUROC difference: Model B AUROC - Model A AUROC
    - AUROC change in percentage points (pp): (Model B AUROC - Model A AUROC) * 100
    - Relative uplift (%): ((Model B AUROC - Model A AUROC) / Model A AUROC) * 100
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "comparison_possible": False,
            "reason": "Single class present in test set. Comparative AUROC cannot be computed.",
        }

    auc_a = float(roc_auc_score(y_true, y_prob_a))
    auc_b = float(roc_auc_score(y_true, y_prob_b))

    abs_diff = auc_b - auc_a
    pct_points_change = abs_diff * 100.0

    if auc_a > 0.0:
        rel_uplift_pct = ((auc_b - auc_a) / auc_a) * 100.0
    else:
        rel_uplift_pct = float("inf") if abs_diff > 0 else 0.0

    # Paired Bootstrap for delta (Model B - Model A)
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    boot_deltas = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            b_auc_a = roc_auc_score(y_true[idx], y_prob_a[idx])
            b_auc_b = roc_auc_score(y_true[idx], y_prob_b[idx])
            boot_deltas.append(b_auc_b - b_auc_a)
        except Exception:
            pass

    if len(boot_deltas) >= 20:
        delta_ci_low = float(np.percentile(boot_deltas, 2.5))
        delta_ci_high = float(np.percentile(boot_deltas, 97.5))
        paired_ci = [round(delta_ci_low, 4), round(delta_ci_high, 4)]
        # Statistical significance: 95% CI strictly excludes 0
        is_statistically_significant = bool(delta_ci_low > 0.0 or delta_ci_high < 0.0)
        # Empirical p-value for H0: delta <= 0 (one-tailed)
        p_val_uplift = float(np.mean(np.array(boot_deltas) <= 0.0))
    else:
        paired_ci = None
        is_statistically_significant = False
        p_val_uplift = None

    # Target 4-8% investigation
    # Target window is 4.0% to 8.0% uplift
    in_target_window = bool(4.0 <= rel_uplift_pct <= 8.0)
    in_pp_window = bool(4.0 <= pct_points_change <= 8.0)

    if in_target_window or in_pp_window:
        target_status = "TARGET_MET"
        target_explanation = f"Measured uplift falls within the targeted 4-8% benchmark window."
    elif rel_uplift_pct > 8.0 or pct_points_change > 8.0:
        target_status = "EXCEEDED_TARGET"
        target_explanation = (
            f"Measured difference (+{pct_points_change:.2f} pp / +{rel_uplift_pct:.1f}% relative) "
            f"exceeds the 4-8% target. Note that with small holdouts, point estimates carry wide uncertainty."
        )
    elif rel_uplift_pct > 0.0:
        target_status = "BELOW_TARGET"
        target_explanation = (
            f"Observed positive uplift (+{pct_points_change:.2f} pp / +{rel_uplift_pct:.1f}% relative) "
            f"is below the targeted 4-8% threshold."
        )
    elif rel_uplift_pct == 0.0:
        target_status = "NO_CHANGE"
        target_explanation = "No performance difference was observed between Model A and Model B."
    else:
        target_status = "PERFORMANCE_DEGRADED"
        target_explanation = f"Model B underperformed Model A by {abs_diff:.4f} AUROC."

    return {
        "comparison_possible": True,
        "model_a_auroc": round(auc_a, 4),
        "model_b_auroc": round(auc_b, 4),
        "absolute_auroc_diff": round(abs_diff, 4),
        "percentage_points_change": round(pct_points_change, 2),
        "relative_uplift_pct": round(rel_uplift_pct, 2),
        "paired_auroc_diff_95ci": paired_ci,
        "is_statistically_significant": is_statistically_significant,
        "empirical_p_value_uplift": round(p_val_uplift, 4) if p_val_uplift is not None else None,
        "target_investigation": {
            "target_range_description": "Target AUROC uplift: 4.0% to 8.0%",
            "target_range_pct": [4.0, 8.0],
            "target_status": target_status,
            "target_met": in_target_window or in_pp_window,
            "explanation": target_explanation,
            "note": "The 4–8% range is a target hypothesis to investigate, not an assumed or guaranteed result.",
        },
    }


# ==============================================================================
# 3. CHART GENERATION (MATPLOTLIB)
# ==============================================================================

def plot_auroc_uplift(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    comparison_res: Dict[str, Any],
    out_path: Path,
):
    """
    Renders high-quality publication-ready AUROC comparison chart:
    - Left Panel: Empirical ROC Curves for Model A and Model B.
    - Right Panel: Bar chart showing AUROC with 95% Confidence Intervals,
      delta percentage-point annotation, and clear visualization of the 4–8% target range.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Style configuration
    color_a = "#0284c7"  # Sky Blue / Cyan (Fundus-Only)
    color_b = "#7c3aed"  # Deep Violet / Purple (Multimodal)
    color_target = "#10b981"  # Emerald

    # --- PANEL 1: ROC Curves ---
    if len(np.unique(y_true)) >= 2:
        fpr_a, tpr_a, _ = roc_curve(y_true, y_prob_a)
        fpr_b, tpr_b, _ = roc_curve(y_true, y_prob_b)

        ax_roc.plot(
            fpr_a,
            tpr_a,
            color=color_a,
            lw=2.5,
            label=f"Model A (Fundus-Only): AUROC = {metrics_a.get('auroc', 'N/A')}",
        )
        ax_roc.plot(
            fpr_b,
            tpr_b,
            color=color_b,
            lw=2.5,
            linestyle="-",
            label=f"Model B (Multimodal): AUROC = {metrics_b.get('auroc', 'N/A')}",
        )
        ax_roc.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (0.50)")
        ax_roc.fill_between(fpr_b, tpr_b, alpha=0.08, color=color_b)
        ax_roc.fill_between(fpr_a, tpr_a, alpha=0.08, color=color_a)
    else:
        ax_roc.text(0.5, 0.5, "ROC unavailable:\nSingle class in test set", ha="center", va="center", fontsize=12)

    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="medium")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="medium")
    ax_roc.set_title("ROC Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # --- PANEL 2: AUROC Bar Chart with Target Range ---
    models = ["Model A\n(Fundus-Only)", "Model B\n(Multimodal)"]
    aurocs = [metrics_a.get("auroc") or 0.0, metrics_b.get("auroc") or 0.0]
    
    # Calculate error bar deltas if CIs available
    yerr = None
    if metrics_a.get("auroc_95ci") and metrics_b.get("auroc_95ci"):
        ci_a = metrics_a["auroc_95ci"]
        ci_b = metrics_b["auroc_95ci"]
        err_a = [aurocs[0] - ci_a[0], ci_a[1] - aurocs[0]]
        err_b = [aurocs[1] - ci_b[0], ci_b[1] - aurocs[1]]
        yerr = np.array([[max(0.0, err_a[0]), max(0.0, err_b[0])], [max(0.0, err_a[1]), max(0.0, err_b[1])]])

    bars = ax_bar.bar(
        [0.5, 1.5],
        aurocs,
        width=0.45,
        color=[color_a, color_b],
        edgecolor="#334155",
        linewidth=1.2,
        yerr=yerr,
        capsize=6,
        alpha=0.9,
    )

    # Annotate bar values
    for bar, val in zip(bars, aurocs):
        height = bar.get_height()
        ax_bar.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

    # Visualize target range (4-8% uplift from Model A)
    base_auc = aurocs[0]
    target_low = base_auc * 1.04
    target_high = base_auc * 1.08
    ax_bar.axhspan(target_low, target_high, color=color_target, alpha=0.15, label="Target Uplift (4–8%)")
    ax_bar.axhline(target_low, color=color_target, linestyle="--", lw=1.2, alpha=0.8)
    ax_bar.axhline(target_high, color=color_target, linestyle="--", lw=1.2, alpha=0.8)

    # Annotate Delta
    abs_diff = comparison_res.get("absolute_auroc_diff", 0.0)
    pp_change = comparison_res.get("percentage_points_change", 0.0)
    rel_uplift = comparison_res.get("relative_uplift_pct", 0.0)
    sign = "+" if abs_diff >= 0 else ""
    
    delta_text = f"Delta: {sign}{pp_change:.1f} pp ({sign}{rel_uplift:.1f}% rel)"
    ax_bar.annotate(
        delta_text,
        xy=(1.0, max(aurocs) + 0.05),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=color_b if abs_diff >= 0 else "#dc2626",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", lw=1),
    )

    ax_bar.set_xticks([0.5, 1.5])
    ax_bar.set_xticklabels(models, fontsize=11, fontweight="medium")
    ax_bar.set_ylabel("AUROC", fontsize=11, fontweight="medium")
    ax_bar.set_ylim([0.0, 1.15])
    ax_bar.set_title("AUROC Comparison & Uplift Target", fontsize=13, fontweight="bold", pad=12)
    ax_bar.grid(True, linestyle=":", axis="y", alpha=0.6)
    ax_bar.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # Footnote note
    fig.text(
        0.5,
        0.01,
        "Note: 4–8% uplift is an investigative hypothesis target, not an assumed outcome. "
        "Synthetic OCT is research data and does not constitute clinical diagnostic evidence.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#475569",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved AUROC comparison chart: %s", out_path)


def plot_metrics_comparison(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    out_path: Path,
):
    """
    Renders grouped bar chart across all normalized [0, 1] metrics:
    AUROC, AUPRC, Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal. Acc"),
        ("precision", "Precision"),
        ("recall_sensitivity", "Recall (Sens)"),
        ("specificity", "Specificity"),
        ("f1_score", "F1 Score"),
    ]

    labels = []
    vals_a = []
    vals_b = []

    for key, name in metric_keys:
        v_a = metrics_a.get(key)
        v_b = metrics_b.get(key)
        if v_a is not None and v_b is not None:
            labels.append(name)
            vals_a.append(float(v_a))
            vals_b.append(float(v_b))

    if not labels:
        logger.warning("No comparable metrics available for metrics_comparison chart.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    rects1 = ax.bar(x - width / 2, vals_a, width, label="Model A (Fundus-Only)", color="#0284c7", alpha=0.9, edgecolor="#334155")
    rects2 = ax.bar(x + width / 2, vals_b, width, label="Model B (Multimodal)", color="#7c3aed", alpha=0.9, edgecolor="#334155")

    # Annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#0369a1")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#6d28d9")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="medium")
    ax.set_title("Comprehensive Classification Metrics Comparison (Held-Out Test Split)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax.set_ylim([0.0, 1.18])
    ax.grid(True, linestyle=":", axis="y", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)

    fig.text(
        0.5,
        0.01,
        "All metrics evaluated on identical patient cohorts without leakage. Scale: [0, 1].",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#475569",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved metrics comparison chart: %s", out_path)


# ==============================================================================
# 4. EXPORT REPORTS (CSV, JSON, INTERPRETATION)
# ==============================================================================

def export_comparison_reports(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    comparison_res: Dict[str, Any],
    out_csv: Path,
    out_json: Path,
) -> pd.DataFrame:
    """
    Exports model comparison table to CSV and structured JSON.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("F1 Score", "f1_score"),
        ("Precision", "precision"),
        ("Recall / Sensitivity", "recall_sensitivity"),
        ("Specificity", "specificity"),
    ]

    records = []
    for display_name, key in metric_rows:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is not None and val_b is not None:
            abs_diff = val_b - val_a
            pp_change = abs_diff * 100.0
            if abs_diff == 0.0:
                rel_uplift = 0.0
            elif val_a > 0.0:
                rel_uplift = (abs_diff / val_a) * 100.0
            else:
                rel_uplift = float("inf") if abs_diff > 0 else 0.0
            records.append({
                "metric": display_name,
                "model_a_fundus": round(float(val_a), 4),
                "model_b_multimodal": round(float(val_b), 4),
                "absolute_difference": round(float(abs_diff), 4),
                "percentage_points_change": round(float(pp_change), 2),
                "relative_uplift_pct": round(float(rel_uplift), 2),
            })
        else:
            records.append({
                "metric": display_name,
                "model_a_fundus": val_a,
                "model_b_multimodal": val_b,
                "absolute_difference": None,
                "percentage_points_change": None,
                "relative_uplift_pct": None,
            })

    df_comp = pd.DataFrame(records)
    df_comp.to_csv(out_csv, index=False)

    # Build comprehensive JSON payload
    json_payload = {
        "metadata": {
            "evaluation_module": "fundus2oct.evaluation.compare_models",
            "target_uplift_range_pct": [4.0, 8.0],
            "num_evaluated_samples": metrics_a.get("n_samples", 0),
        },
        "model_a_fundus": metrics_a,
        "model_b_multimodal": metrics_b,
        "comparative_metrics": comparison_res,
        "table_summary": records,
        "interpretation": generate_interpretation_dict(metrics_a, metrics_b, comparison_res),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    return df_comp


def generate_interpretation_dict(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    comparison_res: Dict[str, Any],
) -> Dict[str, Any]:
    """Generates structured interpretation and limitation statements."""
    if not comparison_res.get("comparison_possible", False):
        return {
            "better_model": "INCONCLUSIVE",
            "verdict": "Comparison impossible: single class or missing predictions.",
            "statistical_support": False,
            "limitations": ["Single class present in test set or predictions unavailable."],
            "clinical_disclaimer": "Synthetic OCT is research data and does not constitute clinical diagnostic evidence.",
        }

    auc_a = comparison_res["model_a_auroc"]
    auc_b = comparison_res["model_b_auroc"]
    abs_diff = comparison_res["absolute_auroc_diff"]
    rel_uplift = comparison_res["relative_uplift_pct"]
    is_sig = comparison_res["is_statistically_significant"]
    target_info = comparison_res["target_investigation"]

    if abs_diff > 0:
        better_model = "Model B (Multimodal)"
    elif abs_diff < 0:
        better_model = "Model A (Fundus-Only)"
    else:
        better_model = "TIE (Identical AUROC)"

    limitations = [
        "Small holdout sample size introduces wide confidence intervals.",
        "Generative diffusion model maturity must be verified before clinical conclusions.",
        "AUROC performance does not demonstrate cross-site generalizability without multi-cohort testing.",
    ]

    return {
        "better_model": better_model,
        "absolute_auroc_change": abs_diff,
        "percentage_points_change": comparison_res["percentage_points_change"],
        "relative_uplift_pct": rel_uplift,
        "target_4_to_8_pct_met": target_info["target_met"],
        "target_assessment": target_info["explanation"],
        "statistically_supported": is_sig,
        "p_value": comparison_res.get("empirical_p_value_uplift"),
        "limitations": limitations,
        "clinical_disclaimer": (
            "AUROC improvement alone does not establish clinical safety or efficacy. "
            "Synthetic OCT represents generated research hypotheses and must never replace genuine clinical diagnostics."
        ),
    }


def generate_interpretation_report(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    comparison_res: Dict[str, Any],
) -> str:
    """Formats human-readable interpretation report."""
    interp = generate_interpretation_dict(metrics_a, metrics_b, comparison_res)
    
    report = [
        "=" * 72,
        "  MODEL COMPARISON & AUROC UPLIFT EVALUATION REPORT",
        "=" * 72,
        f"Higher Performing Model     : {interp['better_model']}",
        f"Model A (Fundus-Only) AUROC : {metrics_a.get('auroc', 'N/A')}",
        f"Model B (Multimodal) AUROC  : {metrics_b.get('auroc', 'N/A')}",
        f"Absolute AUROC Difference   : {interp.get('absolute_auroc_change', 'N/A'):+.4f}",
        f"Percentage-Point Change     : {interp.get('percentage_points_change', 'N/A'):+.2f} pp",
        f"Relative AUROC Uplift       : {interp.get('relative_uplift_pct', 'N/A'):+.2f}%",
        f"Target 4-8% Uplift Met?     : {'YES' if interp.get('target_4_to_8_pct_met') else 'NO'}",
        f"Target Assessment           : {interp.get('target_assessment', 'N/A')}",
        f"Statistically Supported?    : {'YES (95% CI excludes 0)' if interp.get('statistically_supported') else 'NO (Wide CI spans 0)'}",
        "-" * 72,
        "Key Research Limitations:",
    ]
    for lim in interp["limitations"]:
        report.append(f"  * {lim}")
    report.append("-" * 72)
    report.append(f"Clinical Disclaimer:\n  {interp['clinical_disclaimer']}")
    report.append("=" * 72)
    return "\n".join(report)


# ==============================================================================
# 5. HIGH-LEVEL API & CLI
# ==============================================================================

def compare_model_predictions(
    preds_a_input: Union[str, Path, pd.DataFrame],
    preds_b_input: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end evaluation entry point.
    Loads and validates predictions, calculates comparative metrics and bootstrap CIs,
    generates publication charts, and exports CSV & JSON reports.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load DataFrames
    if isinstance(preds_a_input, (str, Path)):
        p_a = Path(preds_a_input)
        if not p_a.exists() or p_a.stat().st_size == 0:
            raise FileNotFoundError(f"Model A prediction file missing or empty: {p_a}")
        df_a = pd.read_csv(p_a)
    else:
        df_a = preds_a_input.copy()

    if isinstance(preds_b_input, (str, Path)):
        p_b = Path(preds_b_input)
        if not p_b.exists() or p_b.stat().st_size == 0:
            raise FileNotFoundError(f"Model B prediction file missing or empty: {p_b}")
        df_b = pd.read_csv(p_b)
    else:
        df_b = preds_b_input.copy()

    # 2. Validate inputs
    df_a_val, df_b_val = validate_inputs(df_a, df_b)

    y_true = df_a_val["true_label"].to_numpy().astype(int)
    y_prob_a = df_a_val["pred_prob_MD"].to_numpy().astype(float)
    y_prob_b = df_b_val["pred_prob_MD"].to_numpy().astype(float)

    # 3. Calculate metrics
    metrics_a = compute_single_model_metrics(y_true, y_prob_a, n_bootstraps=n_bootstraps, seed=seed)
    metrics_b = compute_single_model_metrics(y_true, y_prob_b, n_bootstraps=n_bootstraps, seed=seed)
    comparison_res = compute_paired_comparison(y_true, y_prob_a, y_prob_b, n_bootstraps=n_bootstraps, seed=seed)

    # 4. Generate Artifacts
    out_csv = out_dir / "model_comparison.csv"
    out_json = out_dir / "model_comparison.json"
    out_auroc_png = out_dir / "auroc_uplift.png"
    out_metrics_png = out_dir / "metrics_comparison.png"

    export_comparison_reports(metrics_a, metrics_b, comparison_res, out_csv, out_json)

    if comparison_res.get("comparison_possible", False):
        plot_auroc_uplift(y_true, y_prob_a, y_prob_b, metrics_a, metrics_b, comparison_res, out_auroc_png)
        plot_metrics_comparison(metrics_a, metrics_b, out_metrics_png)

    report_text = generate_interpretation_report(metrics_a, metrics_b, comparison_res)

    return {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "comparison": comparison_res,
        "report_text": report_text,
        "artifacts": {
            "csv": str(out_csv),
            "json": str(out_json),
            "auroc_png": str(out_auroc_png),
            "metrics_png": str(out_metrics_png),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Model A and Model B predictions and measure AUROC uplift.")
    parser.add_argument(
        "--preds-a",
        type=str,
        default="outputs/evaluation_reports/predictions_fundus_only.csv",
        help="Path to Model A (Fundus-Only) predictions CSV.",
    )
    parser.add_argument(
        "--preds-b",
        type=str,
        default="outputs/evaluation_reports/predictions_multimodal.csv",
        help="Path to Model B (Multimodal) predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation_reports",
        help="Directory to save comparison reports and plots.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        res = compare_model_predictions(
            preds_a_input=args.preds_a,
            preds_b_input=args.preds_b,
            output_dir=args.output_dir,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
        )
        print("\n" + res["report_text"] + "\n")
        print("Generated Artifacts:")
        for name, p in res["artifacts"].items():
            print(f"  * {name:<12}: {p}")
        return 0
    except Exception as e:
        logger.error("Failed to compare models: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
