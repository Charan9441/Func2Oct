"""
src/fundus2oct/deployment/__init__.py

Edge deployment and optimization module for Func2OCT:
- Model export to ONNX format (generators and classifiers)
- INT8 dynamic and static post-training quantization
- Calibration data reader for retinal activations
- Edge latency, throughput, and compression benchmarking
"""

from .export_onnx import (
    export_fundus_classifier,
    export_multimodal_classifier,
    export_multimodal_3d_classifier,
    export_oct_decoder,
    export_fundus_encoder,
    export_all_models,
    verify_onnx_model,
)
from .quantize_int8 import (
    RetinalCalibrationDataReader,
    quantize_model_dynamic,
    quantize_model_static,
    benchmark_model,
    compare_fp32_vs_int8,
    quantize_edge_suite,
)

__all__ = [
    "export_fundus_classifier",
    "export_multimodal_classifier",
    "export_multimodal_3d_classifier",
    "export_oct_decoder",
    "export_fundus_encoder",
    "export_all_models",
    "verify_onnx_model",
    "RetinalCalibrationDataReader",
    "quantize_model_dynamic",
    "quantize_model_static",
    "benchmark_model",
    "compare_fp32_vs_int8",
    "quantize_edge_suite",
]
