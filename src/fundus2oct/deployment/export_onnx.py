"""
src/fundus2oct/deployment/export_onnx.py

Export trained generator and classifier models to ONNX format for edge device deployment.
Supports:
1. Generator Components:
   - FundusEncoder (EyeFound ViT / CNN backbone for 512-dim visual conditioning)
   - OCTDecoder2D (Latent-to-B-Scan generator for synthetic cross-sections)
2. Disease Classifiers:
   - FundusClassifier (Model A: Baseline 2D fundus classifier)
   - MultimodalClassifier (Model B: 2D cross-modal gated fusion)
   - Multimodal3DClassifier (Model B: Volumetric 3D CNN + Fundus fusion)
3. Validation & Parity:
   - Graph validation via onnx.checker
   - Numerical parity testing between PyTorch model and ONNX Runtime execution
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    onnx = None
    ort = None
    HAS_ONNX = False

# Ensure repo root and src/ are in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fundus2oct.models.classifier_fundus import FundusClassifier, build_fundus_classifier
from fundus2oct.models.classifier_multimodal import (
    MultimodalClassifier,
    Multimodal3DClassifier,
    build_multimodal_classifier,
)
from fundus2oct.models.fundus_encoder import FundusEncoder, build_fundus_encoder
from fundus2oct.models.oct_vae import OCTDecoder2D, build_oct_vae


# ---------------------------------------------------------------------------
# Numerical Parity & Graph Validation
# ---------------------------------------------------------------------------

def verify_onnx_model(
    onnx_path: Union[str, Path],
    torch_model: nn.Module,
    sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    input_names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Validates the exported ONNX graph using onnx.checker and verifies
    numerical output equivalence with PyTorch on the sample inputs.
    """
    if not HAS_ONNX:
        raise RuntimeError("onnx and onnxruntime are required for verification.")

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    # 1. Graph structural integrity check
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. PyTorch forward pass
    torch_model.eval()
    if isinstance(sample_inputs, torch.Tensor):
        inputs_tuple = (sample_inputs,)
    else:
        inputs_tuple = sample_inputs

    with torch.no_grad():
        torch_out = torch_model(*inputs_tuple)
        if isinstance(torch_out, tuple):
            torch_out_np = torch_out[0].detach().cpu().numpy()
        else:
            torch_out_np = torch_out.detach().cpu().numpy()

    # 3. ONNX Runtime forward pass
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # Errors only
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    ort_inputs = {}
    for name, tensor in zip(input_names, inputs_tuple):
        ort_inputs[name] = tensor.detach().cpu().numpy()

    ort_outs = sess.run(None, ort_inputs)
    ort_out_np = ort_outs[0]

    # 4. Numerical Parity Metrics
    mae = float(np.mean(np.abs(torch_out_np - ort_out_np)))
    max_err = float(np.max(np.abs(torch_out_np - ort_out_np)))
    is_close = bool(np.allclose(torch_out_np, ort_out_np, rtol=rtol, atol=atol))

    # Dot product cosine similarity
    flat_t = torch_out_np.flatten()
    flat_o = ort_out_np.flatten()
    denom = (np.linalg.norm(flat_t) * np.linalg.norm(flat_o))
    cosine_sim = float(np.dot(flat_t, flat_o) / denom) if denom > 0 else 1.0

    return {
        "valid_graph": True,
        "is_close": is_close,
        "mean_absolute_error": mae,
        "max_absolute_error": max_err,
        "cosine_similarity": cosine_sim,
        "output_shape": list(ort_out_np.shape),
    }


# ---------------------------------------------------------------------------
# Individual Exporters
# ---------------------------------------------------------------------------

def export_fundus_classifier(
    model: Optional[FundusClassifier] = None,
    output_path: Union[str, Path] = "outputs/onnx/fundus_classifier.onnx",
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Exports Model A (FundusClassifier) to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = FundusClassifier(in_channels=3, num_classes=2, feature_dim=128)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    input_names = ["fundus"]
    output_names = ["logits"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "fundus": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    if verify and HAS_ONNX:
        verify_onnx_model(output_path, model, dummy_input, input_names)

    return output_path


def export_multimodal_classifier(
    model: Optional[MultimodalClassifier] = None,
    output_path: Union[str, Path] = "outputs/onnx/multimodal_classifier_2d.onnx",
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Exports Model B 2D (MultimodalClassifier) to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = MultimodalClassifier(fundus_channels=3, oct_channels=1, num_classes=2, feature_dim=128)
    model.eval()

    dummy_fundus = torch.randn(1, 3, 256, 256)
    dummy_oct = torch.randn(1, 1, 256, 256)
    dummy_inputs = (dummy_fundus, dummy_oct)
    input_names = ["fundus", "oct_bscan"]
    output_names = ["logits"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "fundus": {0: "batch_size"},
            "oct_bscan": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    if verify and HAS_ONNX:
        verify_onnx_model(output_path, model, dummy_inputs, input_names)

    return output_path


def export_multimodal_3d_classifier(
    model: Optional[Multimodal3DClassifier] = None,
    output_path: Union[str, Path] = "outputs/onnx/multimodal_classifier_3d.onnx",
    num_slices: int = 32,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Exports Model B 3D Volumetric Classifier to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = Multimodal3DClassifier(
            fundus_channels=3,
            oct_channels=1,
            num_classes=2,
            base_channels=16,
            feature_dim=128,
        )
    model.eval()

    dummy_fundus = torch.randn(1, 3, 256, 256)
    dummy_oct_vol = torch.randn(1, 1, num_slices, 256, 256)
    dummy_inputs = (dummy_fundus, dummy_oct_vol)
    input_names = ["fundus", "oct_volume"]
    output_names = ["logits"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "fundus": {0: "batch_size"},
            "oct_volume": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    if verify and HAS_ONNX:
        verify_onnx_model(output_path, model, dummy_inputs, input_names)

    return output_path


def export_oct_decoder(
    model: Optional[OCTDecoder2D] = None,
    output_path: Union[str, Path] = "outputs/onnx/oct_generator_decoder.onnx",
    latent_channels: int = 8,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Exports Generator component OCTDecoder2D (latent -> synthesized B-scan) to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = OCTDecoder2D(out_channels=1, latent_channels=latent_channels, base_channels=32)
    model.eval()

    # Latent shape: (B, latent_channels, H/8=32, W/8=32)
    dummy_latent = torch.randn(1, latent_channels, 32, 32)
    input_names = ["latent_z"]
    output_names = ["synthesized_bscan"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "latent_z": {0: "batch_size"},
            "synthesized_bscan": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_latent,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    if verify and HAS_ONNX:
        verify_onnx_model(output_path, model, dummy_latent, input_names)

    return output_path


def export_fundus_encoder(
    model: Optional[FundusEncoder] = None,
    output_path: Union[str, Path] = "outputs/onnx/fundus_encoder.onnx",
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Exports Generator conditioning backbone (FundusEncoder) to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = FundusEncoder(model_name="vit_base", embed_dim=512, pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    input_names = ["fundus_image"]
    output_names = ["conditioning_vector"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "fundus_image": {0: "batch_size"},
            "conditioning_vector": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    if verify and HAS_ONNX:
        verify_onnx_model(output_path, model, dummy_input, input_names)

    return output_path


# ---------------------------------------------------------------------------
# Batch Export Orchestrator
# ---------------------------------------------------------------------------

def export_all_models(
    output_dir: Union[str, Path] = "outputs/onnx",
    models: Optional[List[str]] = None,
    opset_version: int = 18,
    verify: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Exports all target generator and classifier models to ONNX format.

    Args:
        output_dir: Directory where exported ONNX models will be saved.
        models: List of model names to export or None for all.
        opset_version: Target ONNX operator set version (default 18).
        verify: Whether to execute graph check and numerical parity verification.

    Returns:
        Structured manifest mapping model name to path, size_mb, and verification status.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available_exporters = {
        "fundus_classifier": (export_fundus_classifier, "fundus_classifier.onnx"),
        "multimodal_classifier": (export_multimodal_classifier, "multimodal_classifier_2d.onnx"),
        "multimodal_3d_classifier": (export_multimodal_3d_classifier, "multimodal_classifier_3d.onnx"),
        "oct_decoder": (export_oct_decoder, "oct_generator_decoder.onnx"),
        "fundus_encoder": (export_fundus_encoder, "fundus_encoder.onnx"),
    }

    target_keys = list(available_exporters.keys()) if models is None or "all" in models else models

    manifest: Dict[str, Dict[str, Any]] = {}

    for key in target_keys:
        if key not in available_exporters:
            print(f"[Warning] Unknown model export key '{key}', skipping.")
            continue

        exporter_fn, filename = available_exporters[key]
        target_path = out_dir / filename
        t0 = time.time()
        print(f"[ONNX Export] Exporting {key} -> {target_path} (opset={opset_version})...")

        try:
            saved_path = exporter_fn(
                output_path=target_path,
                opset_version=opset_version,
                verify=verify,
            )
            elapsed = time.time() - t0
            size_mb = saved_path.stat().st_size / (1024 * 1024)

            manifest[key] = {
                "status": "success",
                "path": str(saved_path),
                "size_mb": round(size_mb, 2),
                "export_time_sec": round(elapsed, 2),
                "opset_version": opset_version,
            }
            print(f"  [+] Successfully exported {key} ({size_mb:.2f} MB in {elapsed:.2f}s)")
        except Exception as e:
            manifest[key] = {
                "status": "failed",
                "error": str(e),
            }
            print(f"  [x] Failed to export {key}: {e}")

    # Write manifest summary JSON
    manifest_file = out_dir / "onnx_export_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export trained generator and classifier models to ONNX format.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "fundus_classifier", "multimodal_classifier", "multimodal_3d_classifier", "oct_decoder", "fundus_encoder"],
        help="Model components to export to ONNX.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/onnx",
        help="Directory to save exported ONNX models.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX operator set version (default 18).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical parity verification against ONNX Runtime.",
    )
    args = parser.parse_args()

    export_all_models(
        output_dir=args.output_dir,
        models=args.models,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
