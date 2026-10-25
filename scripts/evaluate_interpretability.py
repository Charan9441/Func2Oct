#!/usr/bin/env python3
"""
scripts/evaluate_interpretability.py

Command-line evaluation runner for Grad-CAM Saliency Mapping and
Monte Carlo Dropout Predictive Uncertainty Quantification on Retinal Classifiers.

Usage:
    python scripts/evaluate_interpretability.py --image data/fundus/sample.png
    python scripts/evaluate_interpretability.py --smoke-test
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import cv2
import numpy as np
from PIL import Image
import torch

# Add src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from fundus2oct.data.auto_capture import create_demo_fundus
from fundus2oct.data.preprocessing import preprocess_fundus
from fundus2oct.evaluation.gradcam import compute_gradcam, save_gradcam_artifacts
from fundus2oct.evaluation.uncertainty import estimate_predictive_uncertainty
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM saliency mapping and MC Dropout uncertainty evaluation."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input retinal fundus photograph.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["fundus", "multimodal"],
        default="fundus",
        help="Classifier architecture to evaluate ('fundus' or 'multimodal').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model weights checkpoint.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class index to explain (None = model's top predicted class).",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=15,
        help="Number of stochastic Monte Carlo Dropout forward passes.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="High-uncertainty decision threshold for selected metric.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["predictive_entropy", "mutual_information", "expected_entropy", "variance"],
        default="predictive_entropy",
        help="Uncertainty metric to compare against threshold.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Grad-CAM overlay blending opacity (0.0 to 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs" / "interpretability"),
        help="Output directory to save heatmaps and uncertainty reports.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute smoke test using synthetic calibrated fundus image.",
    )
    return parser.parse_args()


def load_classifier(
    model_type: str,
    checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Loads specified classifier architecture and loads checkpoint if available."""
    if model_type == "fundus":
        model = FundusClassifier(in_channels=3, num_classes=2)
        default_ckpt = project_root / "checkpoints" / "classifier_fundus.pt"
    else:
        model = MultimodalClassifier(fundus_channels=3, oct_channels=1, num_classes=2)
        default_ckpt = project_root / "checkpoints" / "classifier_multimodal.pt"

    target_ckpt = Path(checkpoint_path) if checkpoint_path else default_ckpt
    if target_ckpt.exists() and target_ckpt.stat().st_size > 0:
        try:
            model.load_checkpoint(target_ckpt, device=device)
            print(f"[Info] Loaded trained checkpoint: {target_ckpt}")
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint {target_ckpt}: {e}. Running with initialized weights.")
    else:
        print(f"[Warning] Checkpoint not found at {target_ckpt}. Running with initialized weights.")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Fund2OCT — Interpretability & Predictive Uncertainty Benchmark")
    print("=" * 70)
    print(f"Device:             {device}")
    print(f"Model Architecture: {args.model_type.upper()}")
    print(f"MC Dropout Passes:  {args.mc_passes}")
    print(f"Uncertainty Metric: {args.metric} (Threshold: {args.threshold})")
    print(f"Output Directory:   {out_dir}")

    # 1. Load Image
    if args.smoke_test or args.image is None:
        print("[Info] Smoke-test mode: Generating synthetic calibrated fundus photograph...")
        raw_arr = create_demo_fundus(blur_level=0.0)
        pil_image = Image.fromarray(raw_arr)
    else:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Specified image file not found: {img_path}")
        pil_image = Image.open(img_path).convert("RGB")

    # 2. Preprocess to normalized tensor (1, 3, 256, 256)
    img_tensor = preprocess_fundus(
        pil_image,
        target_size=(256, 256),
        center_crop=True,
        normalize="imagenet",
    ).unsqueeze(0).to(device)

    # 3. Load Model
    model = load_classifier(args.model_type, checkpoint_path=args.checkpoint, device=device)

    # 4. Prepare inputs
    if args.model_type == "multimodal":
        dummy_oct = torch.zeros(1, 1, 256, 256, device=device)
        model_inputs = (img_tensor, dummy_oct)
        modality = "fundus"
    else:
        model_inputs = img_tensor
        modality = "fundus"

    # 5. Compute Grad-CAM
    print("\n[1/3] Computing Grad-CAM Feature Attribution Saliency Map...")
    cam_result = compute_gradcam(
        model=model,
        inputs=model_inputs,
        target_class=args.target_class,
        modality=modality,
    )

    class_names = ["Control / Non-pathology (0)", "Macular Degeneration (1)"]
    target_class_name = class_names[cam_result.target_class]
    pred_class_name = class_names[cam_result.predicted_class]

    print(f"  Target Layer:    {cam_result.target_layer_name}")
    print(f"  Modality:        {cam_result.modality_explained}")
    print(f"  Explained Class: {target_class_name} (Predicted: {pred_class_name})")

    # 6. Compute Monte Carlo Dropout Predictive Uncertainty
    print(f"\n[2/3] Executing {args.mc_passes} Stochastic Monte Carlo Dropout Passes...")
    unc_result = estimate_predictive_uncertainty(
        model=model,
        inputs=model_inputs,
        num_passes=args.mc_passes,
        threshold=args.threshold,
        uncertainty_metric=args.metric,
        seed=42,
        device=device,
    )

    flag_status = "HIGH UNCERTAINTY [FLAGGED]" if unc_result.is_high_uncertainty else "LOW UNCERTAINTY [PASSED]"
    print(f"  Predicted Class:        {class_names[unc_result.predicted_class]}")
    print(f"  Confidence:             {unc_result.mean_probabilities[unc_result.predicted_class] * 100:.2f}%")
    print(f"  Predictive Entropy:     {unc_result.predictive_entropy_nats:.4f} nats ({unc_result.predictive_entropy_bits:.4f} bits)")
    print(f"  Epistemic (Mutual Info):{unc_result.mutual_information_nats:.4f} nats")
    print(f"  Aleatoric (Expected H): {unc_result.expected_entropy_nats:.4f} nats")
    print(f"  Decision Flag:          {flag_status}")

    # 7. Save Visual and JSON Artifacts
    print("\n[3/3] Exporting Evaluation Artifacts...")
    saved_images = save_gradcam_artifacts(
        result=cam_result,
        output_dir=out_dir,
        filename_prefix=f"{args.model_type}_sample",
        original_image=pil_image,
        alpha=args.alpha,
    )

    report_payload: Dict[str, Any] = {
        "model_type": args.model_type,
        "is_trained": getattr(model, "is_trained", False),
        "gradcam": {
            "target_layer": cam_result.target_layer_name,
            "modality_explained": cam_result.modality_explained,
            "target_class": cam_result.target_class,
            "predicted_class": cam_result.predicted_class,
            "class_probabilities": cam_result.class_probabilities.tolist(),
            "heatmap_path": str(saved_images.get("heatmap")),
            "overlay_path": str(saved_images.get("overlay")),
        },
        "uncertainty": unc_result.to_dict(include_passes=True),
        "clinical_disclaimer": (
            "Uncertainty scores and Grad-CAM heatmaps represent research telemetry under MC Dropout "
            "and gradient-weighted activation mapping. They do not constitute calibrated diagnostic error "
            "probabilities or causal biological assertions."
        ),
    }

    report_path = out_dir / f"{args.model_type}_uncertainty_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"  [+] Grad-CAM Heatmap:  {saved_images.get('heatmap')}")
    print(f"  [+] Saliency Overlay:  {saved_images.get('overlay')}")
    print(f"  [+] Diagnostic Report: {report_path}")
    print("=" * 70)
    print(" Evaluation Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
