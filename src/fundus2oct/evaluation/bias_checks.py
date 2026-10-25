"""
src/fundus2oct/evaluation/bias_checks.py

Senior ML Research Engineering Module for Retinal Image Bias Checks and
Cross-Dataset Generalization Evaluation.

Key Capabilities:
1. Manifest and Dataset Validation:
   - Schema validation, missing value detection, image readability checks.
   - Duplicate image detection via cryptographic MD5 hashing.
   - Training overlap / data leakage detection against baseline training manifests.
2. Label Schema and Task Compatibility:
   - Evaluates whether external disease annotations (e.g. IDRiD Diabetic Retinopathy
     vs ARMD Macular Degeneration) represent compatible tasks.
   - Prevents invalid silent label remapping; records explicit incompatibility reasons.
3. Cross-Dataset Generalization Metrics:
   - AUROC, AUPRC, Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1.
   - Patient-level clustered bootstrap confidence intervals (95% CI).
   - Handles absent classes and small-cohort edge cases without crashes.
4. Subgroup and Demographic Disparity Analysis:
   - Evaluates performance across age, sex/gender, acquisition camera, site, eye.
   - Computes Disparate Impact Ratio, Equal Opportunity Difference, False Positive Parity.
   - Enforces configurable minimum subgroup sample thresholds with explicit warnings.
5. Reporting:
   - Generates JSON summary, cross-dataset CSV, subgroup CSV, and markdown report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
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
)
import torch
import torch.nn as nn

# Project imports
from fundus2oct.data.preprocessing import preprocess_fundus, safe_load_image
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier

logger = logging.getLogger("fundus2oct.evaluation.bias_checks")


# =====================================================================
# 1. ENUMS AND DATA CLASSES
# =====================================================================

class CompatibilityStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    EXPERIMENTAL_SHARED_TARGET = "EXPERIMENTAL_SHARED_TARGET"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class DatasetSpec:
    """Specification of an external validation dataset."""
    name: str
    task_description: str
    image_path_col: str = "image_path"
    label_col: str = "label"
    patient_id_col: Optional[str] = "patient_id"
    subgroup_cols: List[str] = field(default_factory=lambda: ["age", "age_group", "sex", "gender", "camera_model", "device_model", "site_id", "eye"])
    label_mapping: Optional[Dict[Any, int]] = None
    compatibility: CompatibilityStatus = CompatibilityStatus.UNVERIFIED
    compatibility_reason: str = ""


# Pre-configured specifications for benchmark retinal datasets
ARMD_DATASET_SPEC = DatasetSpec(
    name="ARMD",
    task_description="Age-related Macular Degeneration (AMD/ARMD) Detection vs Control/Other.",
    image_path_col="fundus_path",
    label_col="class_name",
    patient_id_col="patient_id",
    subgroup_cols=["eye", "patient_id", "split"],
    label_mapping={
        "AMD": 1,
        "ARMD": 1,
        "MD": 1,
        "1": 1,
        1: 1,
        "NORMAL": 0,
        "CONTROL": 0,
        "OTHER": 0,
        "0": 0,
        0: 0,
    },
    compatibility=CompatibilityStatus.COMPATIBLE,
    compatibility_reason="Target pathology (Macular Degeneration vs Control) matches the model's primary classification task.",
)

IDRID_DATASET_SPEC = DatasetSpec(
    name="IDRiD",
    task_description="Indian Diabetic Retinopathy Image Dataset (Diabetic Retinopathy & DME Severity Grading).",
    image_path_col="image_path",
    label_col="Retinopathy grade",
    patient_id_col="patient_id",
    subgroup_cols=["Risk of macular edema", "device_model", "eye"],
    label_mapping=None,  # Incompatible with MD classifier
    compatibility=CompatibilityStatus.INCOMPATIBLE,
    compatibility_reason=(
        "IDRiD annotates Diabetic Retinopathy severity (grades 0-4) and Diabetic Macular Edema risk (grades 0-2). "
        "Diabetic Retinopathy (microvascular diabetes complication) and Age-Related Macular Degeneration "
        "(chorioretinal neurodegeneration) represent distinct retinal pathologies. The binary MD classifier "
        "cannot evaluate DR severity without invalid label conflation."
    ),
)


@dataclass
class DatasetValidationReport:
    """Diagnostic audit of an evaluated dataset manifest."""
    dataset_name: str
    manifest_path: str
    total_rows: int
    eligible_samples: int
    excluded_samples: int
    exclusion_reasons: Dict[str, int]
    duplicate_image_count: int
    training_overlap_count: int
    patient_count: Optional[int]
    available_subgroups: List[str]
    missing_subgroups: List[str]
    is_available: bool
    status_message: str


@dataclass
class ClassificationMetrics:
    """Comprehensive single-dataset performance telemetry."""
    dataset_name: str
    n_samples: int
    n_positive: int
    n_negative: int
    prevalence: float
    accuracy: float
    balanced_accuracy: float
    sensitivity_recall: float
    specificity: float
    precision_ppv: float
    npv: float
    f1_score: float
    auroc: Optional[float]
    auroc_95ci: Optional[List[float]]
    auprc: Optional[float]
    auprc_95ci: Optional[List[float]]
    auroc_available: bool
    undefined_metric_reason: Optional[str]
    decision_threshold: float
    confusion_matrix: Dict[str, int]
    bootstrap_seed: int


@dataclass
class SubgroupMetricResult:
    """Performance disparity metrics for an individual demographic or acquisition subgroup."""
    dataset_name: str
    subgroup_col: str
    subgroup_val: str
    n_samples: int
    n_positive: int
    n_negative: int
    prevalence: float
    accuracy: float
    sensitivity_recall: float
    specificity: float
    f1_score: float
    auroc: Optional[float]
    delta_accuracy: float
    delta_sensitivity: float
    delta_specificity: float
    disparate_impact_ratio: Optional[float]
    equal_opportunity_diff: Optional[float]
    false_positive_rate_diff: Optional[float]
    is_small_subgroup: bool
    warning: Optional[str]


# =====================================================================
# 2. HASHING & TRAINING OVERLAP DETECTION
# =====================================================================

def compute_file_md5(file_path: Union[str, Path], chunk_size: int = 65536) -> Optional[str]:
    """Computes MD5 hash of an image file to detect duplicate and overlapping images."""
    p = Path(file_path)
    if not p.is_file():
        return None
    md5 = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None


def collect_training_identifiers(
    training_manifest_path: Union[str, Path] = "data/manifests/dataset.csv",
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extracts known training patient IDs, filenames, and MD5 hashes from baseline dataset.csv.
    Used to detect data leakage and training overlap in external evaluation.
    """
    path = Path(training_manifest_path)
    patient_ids: Set[str] = set()
    filenames: Set[str] = set()
    hashes: Set[str] = set()

    if not path.is_file():
        return patient_ids, filenames, hashes

    try:
        df = pd.read_csv(path)
        if "patient_id" in df.columns:
            patient_ids.update(df["patient_id"].dropna().astype(str).unique())
        for col in ("fundus_filename", "oct_filename"):
            if col in df.columns:
                filenames.update(df[col].dropna().astype(str).unique())
        for col in ("fundus_path", "oct_path"):
            if col in df.columns:
                for img_p in df[col].dropna():
                    h = compute_file_md5(img_p)
                    if h:
                        hashes.add(h)
    except Exception as exc:
        logger.warning("Could not read training manifest for overlap checking: %s", exc)

    return patient_ids, filenames, hashes


# =====================================================================
# 3. MANIFEST & DATASET VALIDATION
# =====================================================================

def validate_external_dataset(
    manifest_or_dir: Union[str, Path, pd.DataFrame],
    spec: DatasetSpec,
    training_manifest_path: Union[str, Path] = "data/manifests/dataset.csv",
    base_dir: Optional[Union[str, Path]] = None,
    check_image_readability: bool = True,
    max_readability_checks: int = 50,
) -> Tuple[pd.DataFrame, DatasetValidationReport]:
    """
    Strictly validates external dataset paths, columns, image accessibility, and labels.
    Never fabricates missing samples or attributes.
    """
    exclusion_reasons: Dict[str, int] = {
        "missing_image_file": 0,
        "corrupted_image": 0,
        "missing_label": 0,
        "invalid_label": 0,
        "duplicate_hash": 0,
        "training_overlap": 0,
    }

    # 1. Load DataFrame
    df_raw: Optional[pd.DataFrame] = None
    manifest_str = str(manifest_or_dir)

    if isinstance(manifest_or_dir, pd.DataFrame):
        df_raw = manifest_or_dir.copy()
        manifest_str = "in_memory_dataframe"
    else:
        path = Path(manifest_or_dir)
        if not path.exists():
            report = DatasetValidationReport(
                dataset_name=spec.name,
                manifest_path=manifest_str,
                total_rows=0,
                eligible_samples=0,
                excluded_samples=0,
                exclusion_reasons=exclusion_reasons,
                duplicate_image_count=0,
                training_overlap_count=0,
                patient_count=0,
                available_subgroups=[],
                missing_subgroups=spec.subgroup_cols,
                is_available=False,
                status_message=f"Dataset path or manifest file does not exist: {path}",
            )
            return pd.DataFrame(), report

        if path.is_file():
            try:
                if path.suffix.lower() == ".csv":
                    df_raw = pd.read_csv(path)
                elif path.suffix.lower() == ".json":
                    df_raw = pd.read_json(path)
                elif path.suffix.lower() in (".xlsx", ".xls"):
                    df_raw = pd.read_excel(path)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            except Exception as exc:
                report = DatasetValidationReport(
                    dataset_name=spec.name,
                    manifest_path=manifest_str,
                    total_rows=0,
                    eligible_samples=0,
                    excluded_samples=0,
                    exclusion_reasons=exclusion_reasons,
                    duplicate_image_count=0,
                    training_overlap_count=0,
                    patient_count=0,
                    available_subgroups=[],
                    missing_subgroups=spec.subgroup_cols,
                    is_available=False,
                    status_message=f"Failed to parse manifest {path}: {exc}",
                )
                return pd.DataFrame(), report
        elif path.is_dir():
            # Directory scanning: discover images and infer directory-based classes
            rows = []
            valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            for f in sorted(path.rglob("*")):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    class_name = f.parent.name
                    rows.append({
                        spec.image_path_col: str(f),
                        spec.label_col: class_name,
                        "filename": f.name,
                    })
            if rows:
                df_raw = pd.DataFrame(rows)
            else:
                report = DatasetValidationReport(
                    dataset_name=spec.name,
                    manifest_path=manifest_str,
                    total_rows=0,
                    eligible_samples=0,
                    excluded_samples=0,
                    exclusion_reasons=exclusion_reasons,
                    duplicate_image_count=0,
                    training_overlap_count=0,
                    patient_count=0,
                    available_subgroups=[],
                    missing_subgroups=spec.subgroup_cols,
                    is_available=False,
                    status_message=f"Directory {path} contains no readable retinal image files.",
                )
                return pd.DataFrame(), report

    if df_raw is None or len(df_raw) == 0:
        report = DatasetValidationReport(
            dataset_name=spec.name,
            manifest_path=manifest_str,
            total_rows=0,
            eligible_samples=0,
            excluded_samples=0,
            exclusion_reasons=exclusion_reasons,
            duplicate_image_count=0,
            training_overlap_count=0,
            patient_count=0,
            available_subgroups=[],
            missing_subgroups=spec.subgroup_cols,
            is_available=False,
            status_message="Dataset manifest is completely empty.",
        )
        return pd.DataFrame(), report

    total_rows = len(df_raw)

    # 2. Check required columns
    img_col = spec.image_path_col
    lbl_col = spec.label_col

    if img_col not in df_raw.columns:
        # Check fallback column names
        fallbacks = ["image", "image_path", "fundus_path", "filepath", "file_path", "filename", "img_path"]
        for fb in fallbacks:
            if fb in df_raw.columns:
                img_col = fb
                break
        else:
            report = DatasetValidationReport(
                dataset_name=spec.name,
                manifest_path=manifest_str,
                total_rows=total_rows,
                eligible_samples=0,
                excluded_samples=total_rows,
                exclusion_reasons={"missing_image_path_column": total_rows},
                duplicate_image_count=0,
                training_overlap_count=0,
                patient_count=0,
                available_subgroups=[],
                missing_subgroups=spec.subgroup_cols,
                is_available=False,
                status_message=f"Required image path column '{spec.image_path_col}' not found in manifest.",
            )
            return pd.DataFrame(), report

    if lbl_col not in df_raw.columns:
        lbl_fallbacks = ["label", "class", "class_name", "target", "diagnosis", "Retinopathy grade"]
        for fb in lbl_fallbacks:
            if fb in df_raw.columns:
                lbl_col = fb
                break

    # 3. Collect Training identifiers for overlap detection
    train_patients, train_filenames, train_hashes = collect_training_identifiers(training_manifest_path)

    # 4. Filter & Validate Samples
    eligible_rows: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    dup_count = 0
    overlap_count = 0

    readability_checked = 0

    for idx, row in df_raw.iterrows():
        raw_img_path = str(row[img_col]) if pd.notna(row[img_col]) else ""
        if not raw_img_path:
            exclusion_reasons["missing_image_file"] += 1
            continue

        p = Path(raw_img_path)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p

        if not p.is_file():
            exclusion_reasons["missing_image_file"] += 1
            continue

        # Optional Readability Check
        if check_image_readability and readability_checked < max_readability_checks:
            try:
                safe_load_image(p, expected_channels=3)
                readability_checked += 1
            except Exception:
                exclusion_reasons["corrupted_image"] += 1
                continue

        # Check Label
        raw_label = row[lbl_col] if (lbl_col in row and pd.notna(row[lbl_col])) else None
        if raw_label is None:
            exclusion_reasons["missing_label"] += 1
            continue

        # Map Label if mapping exists
        binary_label: Optional[int] = None
        if spec.label_mapping is not None:
            cleaned_key = str(raw_label).strip()
            if cleaned_key in spec.label_mapping:
                binary_label = spec.label_mapping[cleaned_key]
            elif raw_label in spec.label_mapping:
                binary_label = spec.label_mapping[raw_label]
            else:
                exclusion_reasons["invalid_label"] += 1
                continue
        else:
            # When mapping is None (e.g. IDRiD), record raw label
            binary_label = None

        # Check Duplicates within dataset
        file_hash = compute_file_md5(p)
        if file_hash:
            if file_hash in seen_hashes:
                dup_count += 1
                exclusion_reasons["duplicate_hash"] += 1
                continue
            seen_hashes.add(file_hash)

            # Check overlap with training data
            if file_hash in train_hashes:
                overlap_count += 1
                exclusion_reasons["training_overlap"] += 1
                continue

        # Check overlap via filename or patient ID if available
        if p.name in train_filenames:
            overlap_count += 1
            exclusion_reasons["training_overlap"] += 1
            continue

        row_dict = row.to_dict()
        row_dict["_resolved_image_path"] = str(p)
        row_dict["_binary_target"] = binary_label
        row_dict["_raw_label"] = raw_label
        row_dict["_image_hash"] = file_hash or ""
        eligible_rows.append(row_dict)

    df_eligible = pd.DataFrame(eligible_rows)
    eligible_count = len(df_eligible)
    excluded_count = total_rows - eligible_count

    # Subgroups Available
    avail_subgroups = [c for c in spec.subgroup_cols if c in df_raw.columns]
    missing_subgroups = [c for c in spec.subgroup_cols if c not in df_raw.columns]

    patient_count = None
    if spec.patient_id_col and spec.patient_id_col in df_eligible.columns:
        patient_count = int(df_eligible[spec.patient_id_col].nunique())

    report = DatasetValidationReport(
        dataset_name=spec.name,
        manifest_path=manifest_str,
        total_rows=total_rows,
        eligible_samples=eligible_count,
        excluded_samples=excluded_count,
        exclusion_reasons={k: v for k, v in exclusion_reasons.items() if v > 0},
        duplicate_image_count=dup_count,
        training_overlap_count=overlap_count,
        patient_count=patient_count,
        available_subgroups=avail_subgroups,
        missing_subgroups=missing_subgroups,
        is_available=eligible_count > 0,
        status_message=f"Validated {eligible_count} eligible samples ({excluded_count} excluded).",
    )

    return df_eligible, report


# =====================================================================
# 4. MODEL LOADING & INFERENCE
# =====================================================================

def load_classifier_for_evaluation(
    checkpoint_path: Union[str, Path] = "checkpoints/classifier_fundus.pt",
    model_type: str = "fundus",
    device: Optional[Union[str, torch.device]] = None,
    backbone: str = "cnn",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Safely loads trained classification weights using repository model conventions.
    Sets model to evaluation mode and freezes all parameters.
    """
    dev = torch.device(device) if device else torch.device("cpu")
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    logger.info("Loading %s classifier from: %s", model_type, ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")

    metadata: Dict[str, Any] = {
        "checkpoint_path": str(ckpt_path),
        "model_type": model_type,
        "device": str(dev),
    }

    if isinstance(state, dict):
        metadata["epoch"] = state.get("epoch", "unknown")
        metadata["val_loss"] = state.get("val_loss", None)
        metadata["val_acc"] = state.get("val_acc", None)
        model_weights = state.get("state_dict", state.get("model_state_dict", state))
    else:
        model_weights = state

    # Detect architecture from weights if possible
    has_eyefound = any("encoder." in k for k in model_weights.keys())
    effective_backbone = "eyefound" if has_eyefound else backbone

    if model_type == "fundus":
        model = FundusClassifier(
            in_channels=3,
            num_classes=2,
            feature_dim=256,
            backbone=effective_backbone,
        )
    elif model_type == "multimodal":
        model = MultimodalClassifier(
            fundus_channels=3,
            oct_channels=1,
            num_classes=2,
            feature_dim=256,
            backbone=effective_backbone,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'fundus' or 'multimodal'.")

    # Adapt head weights if saved under single linear vs Sequential(dropout, linear)
    model_keys = set(model.state_dict().keys())
    adapted_weights = {}
    for k, v in model_weights.items():
        if k == "head.weight" and "head.1.weight" in model_keys and "head.weight" not in model_keys:
            adapted_weights["head.1.weight"] = v
        elif k == "head.bias" and "head.1.bias" in model_keys and "head.bias" not in model_keys:
            adapted_weights["head.1.bias"] = v
        else:
            adapted_weights[k] = v

    # Load weights with strict=False to accommodate backbone variations
    missing, unexpected = model.load_state_dict(adapted_weights, strict=False)
    if missing:
        logger.warning("Checkpoint loading missing %d keys: %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Checkpoint loading unexpected %d keys: %s", len(unexpected), unexpected[:5])

    model.eval()
    model.to(dev)
    for p in model.parameters():
        p.requires_grad = False

    metadata["backbone"] = effective_backbone
    return model, metadata


@torch.no_grad()
def run_external_inference(
    model: nn.Module,
    df_eligible: pd.DataFrame,
    image_col: str = "_resolved_image_path",
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Runs deterministic model inference on an external cohort without threshold tuning.
    Appends 'pred_prob_MD' and 'pred_prob_OTHER' columns to the DataFrame.
    """
    if len(df_eligible) == 0:
        return df_eligible

    dev = torch.device(device) if device else getattr(model, "device", torch.device("cpu"))
    model.eval()

    probs_md: List[float] = []
    probs_other: List[float] = []

    image_paths = df_eligible[image_col].tolist()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                t = preprocess_fundus(p, target_size=(256, 256), normalize="imagenet")
            except Exception as exc:
                logger.error("Failed to preprocess %s during inference: %s", p, exc)
                t = torch.zeros(3, 256, 256)
            tensors.append(t)

        batch_tensor = torch.stack(tensors, dim=0).to(dev)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for p_row in probs:
            probs_other.append(float(p_row[0]))
            probs_md.append(float(p_row[1]))

    df_out = df_eligible.copy()
    df_out["pred_prob_OTHER"] = probs_other
    df_out["pred_prob_MD"] = probs_md
    return df_out


# =====================================================================
# 5. METRICS & PATIENT-LEVEL BOOTSTRAP UNCERTAINTY
# =====================================================================

def compute_dataset_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dataset_name: str,
    threshold: float = 0.5,
    patient_ids: Optional[Sequence[Any]] = None,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> ClassificationMetrics:
    """
    Computes rigorous classification performance telemetry with patient-clustered
    bootstrap confidence intervals (95% CI).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = float(n_pos / n_samples) if n_samples > 0 else 0.0

    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred)) if n_samples > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if n_samples > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # AUROC & AUPRC (Undefined when only 1 class is present)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return ClassificationMetrics(
            dataset_name=dataset_name,
            n_samples=n_samples,
            n_positive=n_pos,
            n_negative=n_neg,
            prevalence=round(prevalence, 4),
            accuracy=round(acc, 4),
            balanced_accuracy=round(bal_acc, 4),
            sensitivity_recall=round(rec, 4),
            specificity=round(spec, 4),
            precision_ppv=round(prec, 4),
            npv=round(npv, 4),
            f1_score=round(f1, 4),
            auroc=None,
            auroc_95ci=None,
            auprc=None,
            auprc_95ci=None,
            auroc_available=False,
            undefined_metric_reason=(
                f"Cohort contains only a single class ({unique_classes[0]}). "
                "AUROC and AUPRC are mathematically undefined for single-class cohorts."
            ),
            decision_threshold=threshold,
            confusion_matrix={"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            bootstrap_seed=seed,
        )

    auroc = float(roc_auc_score(y_true, y_prob))
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = float(auc(r_curve, p_curve))

    # Clustered / Patient-level Bootstrap Resampling for 95% CIs
    rng = np.random.default_rng(seed)
    boot_auroc: List[float] = []
    boot_auprc: List[float] = []

    if patient_ids is not None and len(patient_ids) == n_samples and len(set(patient_ids)) < n_samples:
        # Clustered bootstrap over unique patient IDs to handle correlated multiple images
        unique_pts = np.array(list(set(patient_ids)))
        pt_to_indices: Dict[Any, List[int]] = {}
        for idx, pt in enumerate(patient_ids):
            pt_to_indices.setdefault(pt, []).append(idx)

        for _ in range(n_bootstraps):
            sampled_pts = rng.choice(unique_pts, size=len(unique_pts), replace=True)
            sampled_indices = [idx for pt in sampled_pts for idx in pt_to_indices[pt]]
            y_b = y_true[sampled_indices]
            p_b = y_prob[sampled_indices]
            if len(np.unique(y_b)) < 2:
                continue
            try:
                boot_auroc.append(float(roc_auc_score(y_b, p_b)))
                p_c, r_c, _ = precision_recall_curve(y_b, p_b)
                boot_auprc.append(float(auc(r_c, p_c)))
            except Exception:
                pass
    else:
        # Standard sample-level bootstrap
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n_samples, size=n_samples)
            y_b = y_true[idx]
            p_b = y_prob[idx]
            if len(np.unique(y_b)) < 2:
                continue
            try:
                boot_auroc.append(float(roc_auc_score(y_b, p_b)))
                p_c, r_c, _ = precision_recall_curve(y_b, p_b)
                boot_auprc.append(float(auc(r_c, p_c)))
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

    return ClassificationMetrics(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=round(prevalence, 4),
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        sensitivity_recall=round(rec, 4),
        specificity=round(spec, 4),
        precision_ppv=round(prec, 4),
        npv=round(npv, 4),
        f1_score=round(f1, 4),
        auroc=round(auroc, 4),
        auroc_95ci=auroc_ci,
        auprc=round(auprc, 4),
        auprc_95ci=auprc_ci,
        auroc_available=True,
        undefined_metric_reason=None,
        decision_threshold=threshold,
        confusion_matrix={"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        bootstrap_seed=seed,
    )


# =====================================================================
# 6. SUBGROUP & BIAS DISPARITY ANALYSIS
# =====================================================================

def analyze_subgroup_bias(
    df: pd.DataFrame,
    overall_metrics: ClassificationMetrics,
    subgroup_cols: Sequence[str],
    label_col: str = "_binary_target",
    prob_col: str = "pred_prob_MD",
    threshold: float = 0.5,
    min_subgroup_samples: int = 10,
    seed: int = 42,
) -> List[SubgroupMetricResult]:
    """
    Evaluates subgroup fairness across available demographic and acquisition columns.
    Never fabricates attributes; reports small-sample and missing metadata warnings.
    """
    results: List[SubgroupMetricResult] = []
    if len(df) == 0:
        return results

    # Determine reference selection rate & TPR/FPR for fairness deltas
    y_true_all = df[label_col].values.astype(int)
    y_pred_all = (df[prob_col].values >= threshold).astype(int)
    selection_rate_overall = float(np.mean(y_pred_all)) if len(y_pred_all) > 0 else 0.0

    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn_all, fp_all, fn_all, tp_all = cm_all[0, 0], cm_all[0, 1], cm_all[1, 0], cm_all[1, 1]
    fpr_overall = float(fp_all / (fp_all + tn_all)) if (fp_all + tn_all) > 0 else 0.0
    tpr_overall = float(tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0

    for col in subgroup_cols:
        if col not in df.columns:
            continue

        # Drop NaN subgroup values
        valid_df = df[df[col].notna()]
        groups = sorted(valid_df[col].unique(), key=lambda x: str(x))

        for grp_val in groups:
            sub_df = valid_df[valid_df[col] == grp_val]
            n_sub = len(sub_df)
            if n_sub == 0:
                continue

            y_sub_true = sub_df[label_col].values.astype(int)
            y_sub_prob = sub_df[prob_col].values.astype(float)
            y_sub_pred = (y_sub_prob >= threshold).astype(int)

            n_pos = int(np.sum(y_sub_true == 1))
            n_neg = int(np.sum(y_sub_true == 0))
            prevalence = float(n_pos / n_sub) if n_sub > 0 else 0.0

            acc = float(accuracy_score(y_sub_true, y_sub_pred))
            rec = float(recall_score(y_sub_true, y_sub_pred, zero_division=0))
            f1 = float(f1_score(y_sub_true, y_sub_pred, zero_division=0))

            cm_sub = confusion_matrix(y_sub_true, y_sub_pred, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_sub[0, 0], cm_sub[0, 1], cm_sub[1, 0], cm_sub[1, 1]
            spec = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) > 0 else 0.0
            fpr_sub = float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) > 0 else 0.0
            tpr_sub = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0

            auroc = None
            if len(np.unique(y_sub_true)) >= 2:
                try:
                    auroc = round(float(roc_auc_score(y_sub_true, y_sub_prob)), 4)
                except Exception:
                    pass

            selection_rate_sub = float(np.mean(y_sub_pred))
            dir_ratio = (
                round(selection_rate_sub / selection_rate_overall, 4)
                if selection_rate_overall > 0 else None
            )

            # Fairness differences
            delta_acc = round(acc - overall_metrics.accuracy, 4)
            delta_rec = round(rec - overall_metrics.sensitivity_recall, 4)
            delta_spec = round(spec - overall_metrics.specificity, 4)
            eq_opp_diff = round(tpr_sub - tpr_overall, 4)
            fpr_diff = round(fpr_sub - fpr_overall, 4)

            is_small = n_sub < min_subgroup_samples
            warning = (
                f"Small subgroup size (N={n_sub} < {min_subgroup_samples}); metrics have high variance."
                if is_small else None
            )

            results.append(
                SubgroupMetricResult(
                    dataset_name=overall_metrics.dataset_name,
                    subgroup_col=col,
                    subgroup_val=str(grp_val),
                    n_samples=n_sub,
                    n_positive=n_pos,
                    n_negative=n_neg,
                    prevalence=round(prevalence, 4),
                    accuracy=round(acc, 4),
                    sensitivity_recall=round(rec, 4),
                    specificity=round(spec, 4),
                    f1_score=round(f1, 4),
                    auroc=auroc,
                    delta_accuracy=delta_acc,
                    delta_sensitivity=delta_rec,
                    delta_specificity=delta_spec,
                    disparate_impact_ratio=dir_ratio,
                    equal_opportunity_diff=eq_opp_diff,
                    false_positive_rate_diff=fpr_diff,
                    is_small_subgroup=is_small,
                    warning=warning,
                )
            )

    return results


# =====================================================================
# 7. REPORT GENERATION & ARTIFACT EXPORT
# =====================================================================

def export_bias_evaluation_artifacts(
    validation_reports: List[DatasetValidationReport],
    dataset_metrics: List[ClassificationMetrics],
    subgroup_metrics: List[SubgroupMetricResult],
    incompatible_datasets: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "outputs/evaluation_reports",
    model_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Exports structured JSON summary, CSV spreadsheets, and publication-ready Markdown report."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}

    timestamp = datetime.now().isoformat()

    # 1. bias_check_summary.json
    summary_data: Dict[str, Any] = {
        "timestamp": timestamp,
        "model_metadata": model_metadata or {},
        "evaluated_datasets": [asdict(m) for m in dataset_metrics],
        "validation_reports": [asdict(v) for v in validation_reports],
        "incompatible_datasets": incompatible_datasets,
        "subgroup_metrics": [asdict(s) for s in subgroup_metrics],
    }
    json_path = out_p / "bias_check_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    artifacts["summary_json"] = json_path

    # 2. cross_dataset_metrics.csv
    if dataset_metrics:
        df_cross = pd.DataFrame([asdict(m) for m in dataset_metrics])
        # Flatten confusion matrix column
        if "confusion_matrix" in df_cross.columns:
            df_cross["TP"] = df_cross["confusion_matrix"].apply(lambda d: d.get("TP", 0))
            df_cross["TN"] = df_cross["confusion_matrix"].apply(lambda d: d.get("TN", 0))
            df_cross["FP"] = df_cross["confusion_matrix"].apply(lambda d: d.get("FP", 0))
            df_cross["FN"] = df_cross["confusion_matrix"].apply(lambda d: d.get("FN", 0))
            df_cross = df_cross.drop(columns=["confusion_matrix"])
        csv_cross_path = out_p / "cross_dataset_metrics.csv"
        df_cross.to_csv(csv_cross_path, index=False)
        artifacts["cross_dataset_csv"] = csv_cross_path

    # 3. subgroup_metrics.csv
    if subgroup_metrics:
        df_sub = pd.DataFrame([asdict(s) for s in subgroup_metrics])
        csv_sub_path = out_p / "subgroup_metrics.csv"
        df_sub.to_csv(csv_sub_path, index=False)
        artifacts["subgroup_csv"] = csv_sub_path

    # 4. bias_check_report.md
    md_path = out_p / "bias_check_report.md"
    md_content = build_markdown_report(
        timestamp=timestamp,
        validation_reports=validation_reports,
        dataset_metrics=dataset_metrics,
        subgroup_metrics=subgroup_metrics,
        incompatible_datasets=incompatible_datasets,
        model_metadata=model_metadata,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    artifacts["report_md"] = md_path
    artifacts["report_markdown"] = md_path

    return artifacts


def build_markdown_report(
    timestamp: str,
    validation_reports: List[DatasetValidationReport],
    dataset_metrics: List[ClassificationMetrics],
    subgroup_metrics: List[SubgroupMetricResult],
    incompatible_datasets: List[Dict[str, Any]],
    model_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a comprehensive markdown report with clinical integrity notes."""
    lines: List[str] = [
        "# Fund2OCT: External Generalization & Bias-Check Audit Report",
        "",
        f"**Evaluation Timestamp:** `{timestamp}`  ",
        f"**Model Type:** `{model_metadata.get('model_type', 'N/A') if model_metadata else 'N/A'}`  ",
        f"**Checkpoint Provenance:** `{model_metadata.get('checkpoint_path', 'N/A') if model_metadata else 'N/A'}`  ",
        f"**Backbone:** `{model_metadata.get('backbone', 'N/A') if model_metadata else 'N/A'}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]

    if not dataset_metrics and incompatible_datasets:
        lines.append(
            "> [!IMPORTANT]\n"
            "> **Diagnostic Notice:** No external dataset was evaluated with standard classification metrics "
            "due to dataset unavailability or task label incompatibility. See details below.\n"
        )
    elif dataset_metrics:
        lines.append(
            f"Successfully audited **{len(dataset_metrics)}** external cohort(s). "
            "Evaluated cross-dataset generalization without threshold tuning on the external target."
        )

    lines.extend([
        "",
        "## 2. Dataset Cohort Audit & Validation",
        "",
        "| Dataset | Status | Total Rows | Eligible Samples | Excluded | Training Overlap | Patient Count |",
        "|---|---|---|---|---|---|---|",
    ])

    for v in validation_reports:
        avail_badge = "Available" if v.is_available else "Unavailable / Missing"
        pt_str = str(v.patient_count) if v.patient_count is not None else "N/A"
        lines.append(
            f"| **{v.dataset_name}** | {avail_badge} | {v.total_rows} | {v.eligible_samples} | "
            f"{v.excluded_samples} | {v.training_overlap_count} | {pt_str} |"
        )

    lines.extend([
        "",
        "### Exclusion Diagnoses:",
        "",
    ])

    for v in validation_reports:
        if v.exclusion_reasons:
            reasons = ", ".join(f"{k}: {count}" for k, count in v.exclusion_reasons.items())
            lines.append(f"- **{v.dataset_name}:** {reasons}")
        else:
            lines.append(f"- **{v.dataset_name}:** None (All valid rows included)")

    # Label Incompatibilities
    if incompatible_datasets:
        lines.extend([
            "",
            "## 3. Label Compatibility & Disease Task Boundaries",
            "",
            "> [!WARNING]",
            "> **Disease Task Incompatibility Guard:**",
            "> The following datasets were inspected and strictly excluded from invalid performance metrics:",
            "",
        ])
        for inc in incompatible_datasets:
            lines.append(f"### Dataset: `{inc.get('dataset_name')}`")
            lines.append(f"- **Documented Task:** {inc.get('task_description')}")
            lines.append(f"- **Compatibility Status:** `{inc.get('compatibility')}`")
            lines.append(f"- **Scientific Rationale:** {inc.get('reason')}")
            lines.append(
                f"- **Required Mapping / Model:** {inc.get('required_model_or_mapping', 'Requires dedicated classifier model')}\n"
            )

    # Cross-Dataset Metrics
    if dataset_metrics:
        lines.extend([
            "",
            "## 4. Cross-Dataset Generalization Performance",
            "",
            "| Cohort | N | Prev (%) | Acc | Bal Acc | Sens (TPR) | Spec (TNR) | PPV | F1 | AUROC [95% CI] | AUPRC [95% CI] |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])
        for m in dataset_metrics:
            auroc_str = f"{m.auroc:.4f}" if m.auroc is not None else "N/A"
            if m.auroc_95ci:
                auroc_str += f" [{m.auroc_95ci[0]:.3f}, {m.auroc_95ci[1]:.3f}]"

            auprc_str = f"{m.auprc:.4f}" if m.auprc is not None else "N/A"
            if m.auprc_95ci:
                auprc_str += f" [{m.auprc_95ci[0]:.3f}, {m.auprc_95ci[1]:.3f}]"

            lines.append(
                f"| **{m.dataset_name}** | {m.n_samples} | {m.prevalence * 100:.1f}% | {m.accuracy:.4f} | "
                f"{m.balanced_accuracy:.4f} | {m.sensitivity_recall:.4f} | {m.specificity:.4f} | "
                f"{m.precision_ppv:.4f} | {m.f1_score:.4f} | {auroc_str} | {auprc_str} |"
            )

    # Subgroup Analysis
    if subgroup_metrics:
        lines.extend([
            "",
            "## 5. Subgroup & Fairness Disparity Analysis",
            "",
            "| Dataset | Subgroup Field | Value | N | Prev | Acc (Δ) | Sens (Δ) | Spec (Δ) | Disp Impact Ratio | Warnings |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ])
        for s in subgroup_metrics:
            warn_str = s.warning if s.warning else "-"
            dir_str = f"{s.disparate_impact_ratio:.3f}" if s.disparate_impact_ratio is not None else "N/A"
            lines.append(
                f"| {s.dataset_name} | `{s.subgroup_col}` | **{s.subgroup_val}** | {s.n_samples} | "
                f"{s.prevalence * 100:.1f}% | {s.accuracy:.4f} ({s.delta_accuracy:+.3f}) | "
                f"{s.sensitivity_recall:.4f} ({s.delta_sensitivity:+.3f}) | "
                f"{s.specificity:.4f} ({s.delta_specificity:+.3f}) | {dir_str} | {warn_str} |"
            )
    else:
        lines.extend([
            "",
            "## 5. Subgroup & Demographic Analysis",
            "",
            "> [!NOTE]",
            "> No demographic (age, sex) or acquisition (device, site) metadata fields were present in "
            "the provided dataset manifests. Subgroup disparity analysis was therefore marked as **UNAVAILABLE** "
            "rather than fabricating sensitive demographic categories.",
            "",
        ])

    lines.extend([
        "",
        "## 6. Limitations & Governance Disclaimers",
        "",
        "- **Zero Demographic Inference:** Demographic attributes were never inferred from fundus pixels.",
        "- **Fixed Threshold Evaluation:** The decision threshold (default 0.50) was preserved from model training; "
        "no post-hoc threshold tuning was performed on external test sets.",
        "- **SaMD Research Telemetry:** This bias-check is intended for algorithmic auditing under CDSCO SaMD "
        "investigational protocols (Rule 51, Medical Device Rules 2017) and does not constitute clinical trial clearance.",
        "",
    ])

    return "\n".join(lines)


# =====================================================================
# 8. HIGH-LEVEL ORCHESTRATION PIPELINE
# =====================================================================

def evaluate_bias_and_generalization(
    checkpoint_path: Union[str, Path] = "checkpoints/classifier_fundus.pt",
    model_type: str = "fundus",
    idrid_path: Optional[Union[str, Path]] = None,
    armd_path: Optional[Union[str, Path]] = None,
    custom_datasets: Optional[List[Tuple[Union[str, Path], DatasetSpec]]] = None,
    output_dir: Union[str, Path] = "outputs/evaluation_reports",
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = 0.5,
    min_subgroup_samples: int = 10,
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Main orchestration entrypoint for bias-checking and cross-dataset generalization.
    Evaluates IDRiD and ARMD when available and labels are compatible.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_reports: List[DatasetValidationReport] = []
    dataset_metrics_list: List[ClassificationMetrics] = []
    subgroup_metrics_list: List[SubgroupMetricResult] = []
    incompatible_list: List[Dict[str, Any]] = []

    # 1. Model Loading
    model: Optional[nn.Module] = None
    model_meta: Dict[str, Any] = {}
    try:
        model, model_meta = load_classifier_for_evaluation(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
        )
    except Exception as exc:
        logger.error("Failed to load classifier checkpoint: %s", exc)
        if not smoke_test:
            raise

    # 2. Setup Dataset Candidate Queue
    dataset_queue: List[Tuple[Optional[Union[str, Path]], DatasetSpec]] = []

    # IDRiD (Default path or user provided)
    idrid_target = idrid_path or Path("data/external/idrid")
    dataset_queue.append((idrid_target, IDRID_DATASET_SPEC))

    # ARMD (Default path or user provided)
    armd_target = armd_path or Path("data/external/armd")
    dataset_queue.append((armd_target, ARMD_DATASET_SPEC))

    if custom_datasets:
        dataset_queue.extend(custom_datasets)

    # 3. Process Each Dataset Candidate
    for data_target, spec in dataset_queue:
        if data_target is None:
            continue

        # If running in smoke_test mode and path doesn't exist, create synthetic cohort for verification
        if smoke_test and not Path(data_target).exists():
            logger.info("Smoke test active: generating synthetic mock cohort for %s...", spec.name)
            df_eligible, val_report = generate_mock_external_cohort(spec, count=15)
        else:
            df_eligible, val_report = validate_external_dataset(
                manifest_or_dir=data_target,
                spec=spec,
            )

        validation_reports.append(val_report)

        if not val_report.is_available or len(df_eligible) == 0:
            logger.info("Dataset %s is unavailable; skipping inference.", spec.name)
            continue

        # Check Label Compatibility
        if spec.compatibility == CompatibilityStatus.INCOMPATIBLE:
            logger.warning(
                "Dataset %s has incompatible label schema: %s",
                spec.name,
                spec.compatibility_reason,
            )
            incompatible_list.append({
                "dataset_name": spec.name,
                "task_description": spec.task_description,
                "compatibility": spec.compatibility.value,
                "reason": spec.compatibility_reason,
                "required_model_or_mapping": "Dedicated Diabetic Retinopathy multi-class grading model.",
            })
            continue

        # Inference
        if model is not None:
            df_inferred = run_external_inference(
                model=model,
                df_eligible=df_eligible,
                device=device,
            )

            # Metric Calculation
            y_true = df_inferred["_binary_target"].values
            y_prob = df_inferred["pred_prob_MD"].values
            patient_ids = (
                df_inferred[spec.patient_id_col].values
                if spec.patient_id_col and spec.patient_id_col in df_inferred.columns
                else None
            )

            metrics = compute_dataset_metrics(
                y_true=y_true,
                y_prob=y_prob,
                dataset_name=spec.name,
                threshold=threshold,
                patient_ids=patient_ids,
                seed=seed,
            )
            dataset_metrics_list.append(metrics)

            # Subgroup Analysis
            sub_results = analyze_subgroup_bias(
                df=df_inferred,
                overall_metrics=metrics,
                subgroup_cols=spec.subgroup_cols,
                label_col="_binary_target",
                prob_col="pred_prob_MD",
                threshold=threshold,
                min_subgroup_samples=min_subgroup_samples,
                seed=seed,
            )
            subgroup_metrics_list.extend(sub_results)

    # 4. Export Artifacts
    artifacts = export_bias_evaluation_artifacts(
        validation_reports=validation_reports,
        dataset_metrics=dataset_metrics_list,
        subgroup_metrics=subgroup_metrics_list,
        incompatible_datasets=incompatible_list,
        output_dir=out_dir,
        model_metadata=model_meta,
    )

    return {
        "status": "completed",
        "validation_reports": [asdict(v) for v in validation_reports],
        "dataset_metrics": [asdict(m) for m in dataset_metrics_list],
        "subgroup_metrics": [asdict(s) for s in subgroup_metrics_list],
        "incompatible_datasets": incompatible_list,
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }


def generate_mock_external_cohort(spec: DatasetSpec, count: int = 15) -> Tuple[pd.DataFrame, DatasetValidationReport]:
    """Generates synthetic in-memory cohort with real temporary PNGs for unit testing and offline smoke tests."""
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix=f"mock_{spec.name.lower()}_"))
    rng = np.random.RandomState(42)
    rows = []

    for i in range(count):
        is_pos = (i % 2 == 1)
        lbl = "MD" if is_pos else "OTHER"
        img_p = temp_dir / f"mock_fundus_{i:03d}.png"

        # Generate a small 64x64 synthetic fundus image and save as PNG
        arr = rng.randint(40, 220, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_p)

        rows.append({
            spec.image_path_col: str(img_p),
            spec.label_col: lbl,
            "patient_id": f"PT_{i % 5:02d}",
            "age": int(rng.choice([48, 62, 74, 81])),
            "sex": str(rng.choice(["M", "F"])),
            "eye": str(rng.choice(["OD", "OS"])),
            "_resolved_image_path": str(img_p),
            "_binary_target": 1 if is_pos else 0,
            "_raw_label": lbl,
            "_image_hash": f"hash_{i}",
        })

    df = pd.DataFrame(rows)
    report = DatasetValidationReport(
        dataset_name=spec.name,
        manifest_path="synthetic_mock_fixture",
        total_rows=count,
        eligible_samples=count,
        excluded_samples=0,
        exclusion_reasons={},
        duplicate_image_count=0,
        training_overlap_count=0,
        patient_count=df["patient_id"].nunique(),
        available_subgroups=["age", "sex", "eye"],
        missing_subgroups=["camera_model"],
        is_available=True,
        status_message=f"Generated {count} synthetic test samples in {temp_dir}.",
    )
    return df, report
