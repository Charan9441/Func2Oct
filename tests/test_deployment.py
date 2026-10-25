"""
tests/test_deployment.py

Unit tests for edge deployment module:
- ONNX export of generator and classifier models
- Graph structural validation via onnx.checker
- PyTorch vs. ONNX Runtime numerical parity verification
- Post-training dynamic INT8 quantization (size reduction & execution)
- Post-training static INT8 quantization via RetinalCalibrationDataReader
- Edge latency and compression telemetry benchmarking
"""

import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure repo root and src/ are in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

onnx = pytest.importorskip("onnx", reason="onnx not installed in test environment")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed in test environment")

from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier
from fundus2oct.models.oct_vae import OCTDecoder2D
from fundus2oct.deployment.export_onnx import (
    export_fundus_classifier,
    export_multimodal_classifier,
    export_oct_decoder,
    verify_onnx_model,
)
from fundus2oct.deployment.quantize_int8 import (
    RetinalCalibrationDataReader,
    quantize_model_dynamic,
    quantize_model_static,
    benchmark_model,
    compare_fp32_vs_int8,
)


@pytest.fixture(scope="module")
def temp_export_dir(tmp_path_factory):
    """Provides a temporary directory for test model exports and quantization."""
    temp_dir = tmp_path_factory.mktemp("edge_deployment_test")
    return temp_dir


def test_export_fundus_classifier_onnx(temp_export_dir):
    """Tests ONNX export and graph validation for FundusClassifier (Model A)."""
    onnx_path = temp_export_dir / "test_fundus_classifier.onnx"
    model = FundusClassifier(in_channels=3, num_classes=2, feature_dim=64)
    model.eval()

    saved_path = export_fundus_classifier(
        model=model,
        output_path=onnx_path,
        opset_version=18,
        dynamic_batch=True,
        verify=True,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    # Load and check with onnx
    onnx_model = onnx.load(str(saved_path))
    onnx.checker.check_model(onnx_model)

    # Test inference with ONNX Runtime
    sess = ort.InferenceSession(str(saved_path), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(2, 3, 256, 256).astype(np.float32)
    outs = sess.run(None, {"fundus": dummy_input})

    assert len(outs) == 1
    assert outs[0].shape == (2, 2)


def test_export_oct_decoder_generator_onnx(temp_export_dir):
    """Tests ONNX export for OCTDecoder2D generator component (latent -> B-scan)."""
    onnx_path = temp_export_dir / "test_oct_decoder.onnx"
    model = OCTDecoder2D(out_channels=1, latent_channels=8, base_channels=16)
    model.eval()

    saved_path = export_oct_decoder(
        model=model,
        output_path=onnx_path,
        latent_channels=8,
        opset_version=18,
        dynamic_batch=True,
        verify=True,
    )

    assert saved_path.exists()

    sess = ort.InferenceSession(str(saved_path), providers=["CPUExecutionProvider"])
    dummy_latent = np.random.randn(1, 8, 32, 32).astype(np.float32)
    outs = sess.run(None, {"latent_z": dummy_latent})

    assert len(outs) == 1
    assert outs[0].shape == (1, 1, 256, 256)


def test_export_multimodal_classifier_onnx(temp_export_dir):
    """Tests ONNX export for MultimodalClassifier (Model B 2D)."""
    onnx_path = temp_export_dir / "test_multimodal_2d.onnx"
    model = MultimodalClassifier(fundus_channels=3, oct_channels=1, num_classes=2, feature_dim=64)
    model.eval()

    saved_path = export_multimodal_classifier(
        model=model,
        output_path=onnx_path,
        opset_version=18,
        dynamic_batch=True,
        verify=True,
    )

    assert saved_path.exists()

    sess = ort.InferenceSession(str(saved_path), providers=["CPUExecutionProvider"])
    dummy_f = np.random.randn(1, 3, 256, 256).astype(np.float32)
    dummy_o = np.random.randn(1, 1, 256, 256).astype(np.float32)
    outs = sess.run(None, {"fundus": dummy_f, "oct_bscan": dummy_o})

    assert len(outs) == 1
    assert outs[0].shape == (1, 2)


def test_dynamic_int8_quantization(temp_export_dir):
    """Verifies dynamic INT8 quantization compresses model size and executes cleanly."""
    fp32_path = temp_export_dir / "test_oct_decoder.onnx"
    if not fp32_path.exists():
        model = OCTDecoder2D(out_channels=1, latent_channels=8, base_channels=16)
        export_oct_decoder(model, fp32_path, verify=False)

    int8_path = temp_export_dir / "test_oct_decoder_int8_dynamic.onnx"
    quantize_model_dynamic(fp32_path, int8_path)

    assert int8_path.exists()
    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    assert int8_size < fp32_size  # Model must shrink!

    # Verify execution in ONNX Runtime
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    dummy_latent = np.random.randn(1, 8, 32, 32).astype(np.float32)
    outs = sess.run(None, {"latent_z": dummy_latent})
    assert outs[0].shape == (1, 1, 256, 256)


def test_static_int8_quantization_and_calibration(temp_export_dir):
    """Verifies static INT8 post-training quantization with calibration reader."""
    fp32_path = temp_export_dir / "test_oct_decoder.onnx"
    if not fp32_path.exists():
        model = OCTDecoder2D(out_channels=1, latent_channels=8, base_channels=16)
        export_oct_decoder(model, fp32_path, verify=False)

    reader = RetinalCalibrationDataReader(fp32_path, num_samples=4)
    sample = reader.get_next()
    assert sample is not None
    assert "latent_z" in sample
    assert sample["latent_z"].shape == (1, 8, 32, 32)
    reader.rewind()

    stat_int8_path = temp_export_dir / "test_oct_decoder_int8_static.onnx"
    quantize_model_static(fp32_path, stat_int8_path, calibration_data_reader=reader)

    assert stat_int8_path.exists()
    assert stat_int8_path.stat().st_size < fp32_path.stat().st_size


def test_benchmarking_and_comparison(temp_export_dir):
    """Verifies edge latency and compression benchmarking metrics."""
    fp32_path = temp_export_dir / "test_oct_decoder.onnx"
    int8_path = temp_export_dir / "test_oct_decoder_int8_dynamic.onnx"

    dummy_input = {"latent_z": np.random.randn(1, 8, 32, 32).astype(np.float32)}
    res = compare_fp32_vs_int8(fp32_path, int8_path, sample_inputs=dummy_input, num_runs=5)

    assert "compression_ratio" in res
    assert res["compression_ratio"] > 1.0
    assert res["size_reduction_percent"] > 20.0
    assert res["cosine_similarity"] > 0.90
    assert "fp32" in res
    assert "int8" in res
    assert res["fp32"]["throughput_fps"] > 0
    assert res["int8"]["throughput_fps"] > 0
