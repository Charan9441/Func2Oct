#!/usr/bin/env python3
"""
scripts/evaluate_classifiers.py

Evaluates disease classification models on the held-out test split:
- Model A: Fundus-only Classifier
- Model B: Fundus + Synthetic OCT Multimodal Classifier

Computes:
- AUROC, AUPRC
- Accuracy, Balanced Accuracy, Precision, Recall, F1
- Confusion Matrix (TN, FP, FN, TP)
- 95% Bootstrap Confidence Intervals
- Comparative analysis: Delta performance from adding synthetic OCT features.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.config import load_config
from fundus2oct.data.dataset import DiseaseClassificationDataset
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate disease classification models on test split.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(project_root / "data" / "manifests" / "dataset.csv"),
        help="Path to dataset manifest CSV.",
    )
    parser.add_argument(
        "--checkpoint-fundus",
        type=str,
        default=str(project_root / "checkpoints" / "classifier_fundus.pt"),
        help="Path to Model A checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-multimodal",
        type=str,
        default=str(project_root / "checkpoints" / "classifier_multimodal.pt"),
        help="Path to Model B checkpoint.",
    )
    parser.add_argument(
        "--synthetic-oct-dir",
        type=str,
        default=str(project_root / "outputs" / "synthetic_oct"),
        help="Directory containing pre-generated synthetic OCT B-scans.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs" / "evaluation_reports"),
        help="Directory to save evaluation reports and prediction CSVs.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke-test evaluation on 10 synthetic samples.",
    )
    return parser.parse_args()


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval for AUROC."""
    if len(np.unique(y_true)) < 2:
        return (0.0, 0.0)

    scores = []
    rng = np.random.default_rng(42)
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            score = roc_auc_score(y_true[idx], y_prob[idx])
            scores.append(score)
        except Exception:
            pass

    if not scores:
        return (0.0, 0.0)

    low = float(np.percentile(scores, 100 * (alpha / 2.0)))
    high = float(np.percentile(scores, 100 * (1.0 - alpha / 2.0)))
    return (round(low, 3), round(high, 3))


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_multimodal: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_sample_ids = []
    all_class_names = []

    with torch.no_grad():
        for batch in loader:
            fundus = batch["fundus"].to(device)
            labels = batch["label"].cpu().numpy()

            if is_multimodal:
                oct_img = batch["oct"].to(device)
                logits = model(fundus, oct_img)
            else:
                logits = model(fundus)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])  # Positive class MD probability
            all_sample_ids.extend(batch["sample_id"])
            all_class_names.extend(batch["class_name"])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Calculate metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Binary metrics (Positive class = 1 / MD)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        auprc = float(auc(recall_curve, precision_curve)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auprc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    ci_low, ci_high = bootstrap_ci(y_true, y_prob) if not np.isnan(auroc) else (0.0, 0.0)

    metrics = {
        "total_test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
        "auroc_95ci": [ci_low, ci_high] if not np.isnan(auroc) else None,
        "auprc": round(auprc, 4) if not np.isnan(auprc) else None,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }

    df_preds = pd.DataFrame({
        "sample_id": all_sample_ids,
        "class_name": all_class_names,
        "true_label": y_true,
        "pred_label": y_pred,
        "pred_prob_MD": np.round(y_prob, 4),
        "is_correct": y_true == y_pred,
    })

    return metrics, df_preds


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device if args.device else cfg.resolve_device())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(args.manifest)
    synth_oct_dir = Path(args.synthetic_oct_dir)

    print("=" * 70)
    print("  FUNDUS2OCT - COMPARATIVE DISEASE CLASSIFICATION BENCHMARK")
    print("=" * 70)
    print("Test Split Evaluation (Strict Holdout)")
    print(f"Positive Class (1) : MD (Macular Degeneration)")
    print(f"Negative Class (0) : OTHER (Control / Non-MD)")
    print(f"Device             : {device}")
    print("=" * 70)

    smoke_count = 10 if args.smoke_test else 0

    # 1. Prepare Test Split Dataset (Zero leakage, identical samples)
    test_ds_fundus = DiseaseClassificationDataset(
        manifest_path=manifest_file,
        mode="fundus_only",
        split="test",
        synthetic_fallback_count=smoke_count,
    )

    if len(test_ds_fundus) == 0 and not args.smoke_test:
        print("[Notice] No local test images found in data folder. Running synthetic smoke test.")
        test_ds_fundus = DiseaseClassificationDataset(
            manifest_path=manifest_file,
            mode="fundus_only",
            split="test",
            synthetic_fallback_count=10,
        )

    test_loader_fundus = DataLoader(test_ds_fundus, batch_size=4, shuffle=False)
    print(f"Total Test Split Samples: {len(test_ds_fundus)}")

    benchmark_report: Dict[str, Any] = {
        "metadata": {
            "device": str(device),
            "smoke_test": args.smoke_test or (len(test_ds_fundus) == smoke_count),
            "num_test_samples": len(test_ds_fundus),
            "positive_class": "MD (label=1)",
            "negative_class": "OTHER (label=0)",
        },
        "models": {},
    }

    # 2. Evaluate Model A (Fundus-Only)
    ckpt_a = Path(args.checkpoint_fundus)
    model_a_evaluated = False

    if ckpt_a.exists() and ckpt_a.stat().st_size > 1024:
        print("\n--> Evaluating Model A (Fundus-Only)...")
        model_a = FundusClassifier(
            in_channels=3,
            num_classes=cfg.classifiers.num_classes,
            feature_dim=cfg.classifiers.feature_dim,
            dropout=cfg.classifiers.dropout,
        )
        model_a.load_checkpoint(ckpt_a, device=device)
        metrics_a, df_a = evaluate_model(model_a, test_loader_fundus, device=device, is_multimodal=False)
        benchmark_report["models"]["fundus_only"] = metrics_a

        # Save predictions
        preds_a_file = out_dir / "predictions_fundus_only.csv"
        df_a.to_csv(preds_a_file, index=False)
        print(f"    Accuracy: {metrics_a['accuracy']*100:.1f}%, F1: {metrics_a['f1_score']:.3f}, AUROC: {metrics_a['auroc']}")
        print(f"    Predictions saved -> {preds_a_file}")
        model_a_evaluated = True
    else:
        print(f"\n[Notice] Model A checkpoint not found at {ckpt_a}. Run 'python scripts/train_classifiers.py' first.")

    # 3. Evaluate Model B (Fundus + Synthetic OCT)
    ckpt_b = Path(args.checkpoint_multimodal)
    model_b_evaluated = False

    if ckpt_b.exists() and ckpt_b.stat().st_size > 1024:
        print("\n--> Evaluating Model B (Multimodal: Fundus + Synthetic OCT)...")
        test_ds_multi = DiseaseClassificationDataset(
            manifest_path=manifest_file,
            mode="multimodal",
            split="test",
            synthetic_oct_dir=synth_oct_dir,
            synthetic_fallback_count=len(test_ds_fundus) if args.smoke_test else 0,
        )
        test_loader_multi = DataLoader(test_ds_multi, batch_size=4, shuffle=False)

        model_b = MultimodalClassifier(
            fundus_channels=3,
            oct_channels=1,
            num_classes=cfg.classifiers.num_classes,
            feature_dim=cfg.classifiers.feature_dim,
            dropout=cfg.classifiers.dropout,
        )
        model_b.load_checkpoint(ckpt_b, device=device)
        metrics_b, df_b = evaluate_model(model_b, test_loader_multi, device=device, is_multimodal=True)
        benchmark_report["models"]["multimodal"] = metrics_b

        # Save predictions
        preds_b_file = out_dir / "predictions_multimodal.csv"
        df_b.to_csv(preds_b_file, index=False)
        print(f"    Accuracy: {metrics_b['accuracy']*100:.1f}%, F1: {metrics_b['f1_score']:.3f}, AUROC: {metrics_b['auroc']}")
        print(f"    Predictions saved -> {preds_b_file}")
        model_b_evaluated = True
    else:
        print(f"\n[Notice] Model B checkpoint not found at {ckpt_b}.")
        print("         To evaluate Model B, train it after synthetic OCTs are generated.")

    # 4. Comparative Delta Summary Table
    print("\n" + "=" * 70)
    print("  COMPARATIVE PERFORMANCE BENCHMARK (TEST SPLIT)")
    print("=" * 70)

    if model_a_evaluated and model_b_evaluated:
        ma = benchmark_report["models"]["fundus_only"]
        mb = benchmark_report["models"]["multimodal"]

        print(f"{'Metric':<22} | {'Model A (Fundus)':<18} | {'Model B (Multimodal)':<20} | {'Delta':<10}")
        print("-" * 75)
        for metric_key, label in [
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("f1_score", "F1 Score"),
            ("recall_sensitivity", "Recall / Sensitivity"),
            ("specificity", "Specificity"),
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
        ]:
            val_a = ma.get(metric_key)
            val_b = mb.get(metric_key)
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                sign = "+" if delta >= 0 else ""
                print(f"{label:<22} | {val_a:<18.4f} | {val_b:<20.4f} | {sign}{delta:.4f}")
            else:
                print(f"{label:<22} | {str(val_a):<18} | {str(val_b):<20} | N/A")

        benchmark_report["comparison_delta"] = {
            "accuracy_delta": round(mb["accuracy"] - ma["accuracy"], 4),
            "f1_delta": round(mb["f1_score"] - ma["f1_score"], 4),
        }

        # 4b. Generate comparative uplift artifacts and charts
        try:
            from fundus2oct.evaluation.compare_models import compare_model_predictions
            comp_res = compare_model_predictions(
                preds_a_input=preds_a_file,
                preds_b_input=preds_b_file,
                output_dir=out_dir,
            )
            print("\n" + comp_res["report_text"])
            print("Generated Uplift Artifacts:")
            for art_name, art_path in comp_res["artifacts"].items():
                print(f"  * {art_name:<12}: {art_path}")
        except Exception as err:
            print(f"[Warning] Could not generate comparative uplift charts: {err}")
    elif model_a_evaluated:
        ma = benchmark_report["models"]["fundus_only"]
        print(f"Model A (Fundus-Only) Results (Test N={ma['total_test_samples']}):")
        print(f"  * Accuracy          : {ma['accuracy']*100:.1f}%")
        print(f"  * Balanced Accuracy : {ma['balanced_accuracy']*100:.1f}%")
        print(f"  * F1 Score          : {ma['f1_score']:.4f}")
        print(f"  * Recall/Sens       : {ma['recall_sensitivity']*100:.1f}%")
        print(f"  * Specificity       : {ma['specificity']*100:.1f}%")
        print(f"  * Confusion Matrix  : {ma['confusion_matrix']}")
        print("\nModel B (Multimodal) was not evaluated because synthetic OCT checkpoint is pending.")
    else:
        print("No trained checkpoints found to evaluate. Run 'python scripts/train_classifiers.py' first.")

    # 5. Export JSON Report
    json_path = out_dir / "classification_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\nMachine-readable benchmark report saved -> {json_path}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
