"""
src/fundus2oct/deployment/quantize_int8.py

Performs post-training INT8 quantization (dynamic and static) on exported ONNX models
for high-efficiency edge device deployment (e.g., embedded medical edge gateways,
portable fundus cameras, Raspberry Pi / Jetson / x86 mobile workstations).

Key Capabilities:
1. Dynamic INT8 Quantization:
   - Quantizes weight tensors to 8-bit signed integers (QInt8).
   - Fast, zero calibration dataset requirement; optimizes memory footprint and latency.
2. Static Post-Training Quantization (PTQ):
   - Quantizes both activations and weights using calibration batches via RetinalCalibrationDataReader.
   - Calculates activation dynamic range, scales, and zero-points (QDQ / QOperator formats).
3. Benchmarking & Compression Telemetry:
   - Measures storage reduction (MB and % compression).
   - Measures inference latency (mean, median p50, 95th percentile p95, and throughput FPS).
   - Validates numerical parity (MAE, Cosine Similarity) between FP32 and INT8 models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    HAS_ORT_QUANT = True
except ImportError:
    onnx = None
    ort = None
    CalibrationDataReader = object  # type: ignore
    HAS_ORT_QUANT = False

# Ensure repo root and src/ are in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fundus2oct.data.auto_capture import create_demo_fundus


# ---------------------------------------------------------------------------
# Calibration Data Reader for Static Quantization
# ---------------------------------------------------------------------------

class RetinalCalibrationDataReader(CalibrationDataReader):
    """
    Supplies realistic retinal fundus, OCT cross-section, and latent tensors
    to calibrate activation quantization scales and zero-points during static PTQ.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        num_samples: int = 16,
        batch_size: int = 1,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.current_idx = 0

        # Inspect model input tensor requirements from ONNX graph
        sess = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in sess.get_inputs()]
        self.input_shapes = {inp.name: inp.shape for inp in sess.get_inputs()}
        self.input_types = {inp.name: inp.type for inp in sess.get_inputs()}

        # Pre-generate diverse synthetic calibration data
        self.calibration_batches = self._prepare_calibration_data()

    def _prepare_calibration_data(self) -> List[Dict[str, np.ndarray]]:
        batches: List[Dict[str, np.ndarray]] = []
        for i in range(self.num_samples):
            feed_dict: Dict[str, np.ndarray] = {}
            for name, shape in self.input_shapes.items():
                # Resolve dynamic batch size
                b = self.batch_size if shape[0] is None or isinstance(shape[0], str) else shape[0]

                if "fundus" in name.lower():
                    # High quality retinal fundus image
                    blur = (i % 4) * 0.15
                    sample_img = create_demo_fundus(blur_level=blur, noise_sigma=1.0 + i * 0.5)
                    # Convert to normalized (B, 3, 256, 256) float32
                    resized = cv2_resize_safe(sample_img, (256, 256))
                    tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                    batch_arr = np.repeat(tensor[np.newaxis, ...], b, axis=0)
                    feed_dict[name] = batch_arr

                elif "oct_volume" in name.lower():
                    # 3D OCT volume (B, 1, D=32, 256, 256)
                    d = 32 if len(shape) >= 3 and (shape[2] is None or isinstance(shape[2], str)) else (shape[2] if len(shape) >= 3 else 32)
                    vol = np.random.uniform(0.0, 1.0, (b, 1, d, 256, 256)).astype(np.float32)
                    feed_dict[name] = vol

                elif "oct" in name.lower():
                    # 2D OCT B-scan (B, 1, 256, 256)
                    bscan = np.random.uniform(0.0, 1.0, (b, 1, 256, 256)).astype(np.float32)
                    feed_dict[name] = bscan

                elif "latent" in name.lower() or "z" in name.lower():
                    # VAE Decoder Latent (B, 8, 32, 32)
                    c = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else 8
                    h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 32
                    w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 32
                    lat = np.random.normal(0.0, 1.0, (b, c, h, w)).astype(np.float32)
                    feed_dict[name] = lat

                else:
                    # Generic input fallback
                    concrete_shape = [
                        self.batch_size if (dim is None or isinstance(dim, str)) else dim
                        for dim in shape
                    ]
                    feed_dict[name] = np.random.randn(*concrete_shape).astype(np.float32)

            batches.append(feed_dict)
        return batches

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.current_idx < len(self.calibration_batches):
            data = self.calibration_batches[self.current_idx]
            self.current_idx += 1
            return data
        return None

    def rewind(self):
        self.current_idx = 0

    def __len__(self) -> int:
        return len(self.calibration_batches)


def cv2_resize_safe(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Helper to resize numpy image safely without cv2 hard-dependency."""
    try:
        import cv2
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        return np.array(Image.fromarray(img).resize(target_size, Image.BILINEAR))


# ---------------------------------------------------------------------------
# Quantization Engines
# ---------------------------------------------------------------------------

def quantize_model_dynamic(
    input_onnx_path: Union[str, Path],
    output_onnx_path: Union[str, Path],
    weight_type: Any = None,
    op_types_to_quantize: Optional[List[str]] = None,
) -> Path:
    """
    Applies Dynamic Post-Training Quantization (INT8 weights, dynamic activations).
    Ideal for general edge CPU inference without calibration datasets.
    """
    if not HAS_ORT_QUANT:
        raise RuntimeError("onnxruntime.quantization is required for dynamic quantization.")

    in_p = Path(input_onnx_path)
    out_p = Path(output_onnx_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {in_p}")

    w_type = weight_type or QuantType.QInt8

    # Quantize weights dynamically
    quantize_dynamic(
        model_input=str(in_p),
        model_output=str(out_p),
        weight_type=w_type,
        op_types_to_quantize=op_types_to_quantize,
    )

    return out_p


def quantize_model_static(
    input_onnx_path: Union[str, Path],
    output_onnx_path: Union[str, Path],
    calibration_data_reader: Optional[CalibrationDataReader] = None,
    quant_format: Any = None,
    per_channel: bool = True,
    reduce_range: bool = False,
    num_calibration_samples: int = 16,
) -> Path:
    """
    Applies Static Post-Training Quantization (INT8 weights and INT8 activations).
    Calibrates activation dynamic range for minimal quantization loss and maximum edge speedup.
    """
    if not HAS_ORT_QUANT:
        raise RuntimeError("onnxruntime.quantization is required for static quantization.")

    in_p = Path(input_onnx_path)
    out_p = Path(output_onnx_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {in_p}")

    # Use default RetinalCalibrationDataReader if none provided
    reader = calibration_data_reader or RetinalCalibrationDataReader(
        model_path=in_p,
        num_samples=num_calibration_samples,
    )

    q_format = quant_format or QuantFormat.QDQ

    quantize_static(
        model_input=str(in_p),
        model_output=str(out_p),
        calibration_data_reader=reader,
        quant_format=q_format,
        per_channel=per_channel,
        reduce_range=reduce_range,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    return out_p


# ---------------------------------------------------------------------------
# Benchmarking & Parity Evaluation
# ---------------------------------------------------------------------------

def benchmark_model(
    onnx_path: Union[str, Path],
    sample_inputs: Optional[Dict[str, np.ndarray]] = None,
    num_warmup: int = 5,
    num_runs: int = 30,
) -> Dict[str, Any]:
    """
    Measures file size, latency distribution (mean, p50, p95), and throughput.
    """
    p = Path(onnx_path)
    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {p}")

    size_mb = p.stat().st_size / (1024 * 1024)

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(str(p), sess_opts, providers=["CPUExecutionProvider"])

    # Prepare inputs if not provided
    if sample_inputs is None:
        reader = RetinalCalibrationDataReader(p, num_samples=1)
        feed_dict = reader.get_next() or {}
    else:
        feed_dict = sample_inputs

    # Warmup runs
    for _ in range(num_warmup):
        sess.run(None, feed_dict)

    # Benchmark runs
    latencies_ms: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, feed_dict)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_arr = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies_arr))
    median_p50 = float(np.median(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    min_ms = float(np.min(latencies_arr))
    max_ms = float(np.max(latencies_arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "model_path": str(p),
        "size_mb": round(size_mb, 2),
        "mean_latency_ms": round(mean_ms, 2),
        "p50_latency_ms": round(median_p50, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "min_latency_ms": round(min_ms, 2),
        "max_latency_ms": round(max_ms, 2),
        "throughput_fps": round(fps, 1),
    }


def compare_fp32_vs_int8(
    fp32_path: Union[str, Path],
    int8_path: Union[str, Path],
    sample_inputs: Optional[Dict[str, np.ndarray]] = None,
    num_runs: int = 25,
) -> Dict[str, Any]:
    """
    Evaluates compression ratio, latency speedup, and numerical fidelity
    between an original FP32 ONNX model and its INT8 quantized counterpart.
    """
    fp32_bench = benchmark_model(fp32_path, sample_inputs, num_runs=num_runs)
    int8_bench = benchmark_model(int8_path, sample_inputs, num_runs=num_runs)

    # Compute compression and speedup
    compression_ratio = fp32_bench["size_mb"] / int8_bench["size_mb"] if int8_bench["size_mb"] > 0 else 1.0
    size_reduction_pct = (1.0 - (int8_bench["size_mb"] / fp32_bench["size_mb"])) * 100.0 if fp32_bench["size_mb"] > 0 else 0.0
    speedup = fp32_bench["mean_latency_ms"] / int8_bench["mean_latency_ms"] if int8_bench["mean_latency_ms"] > 0 else 1.0

    # Numerical fidelity check
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    if sample_inputs is None:
        reader = RetinalCalibrationDataReader(fp32_path, num_samples=1)
        feed_dict = reader.get_next() or {}
    else:
        feed_dict = sample_inputs

    out_fp32 = sess_fp32.run(None, feed_dict)[0]
    out_int8 = sess_int8.run(None, feed_dict)[0]

    mae = float(np.mean(np.abs(out_fp32 - out_int8)))
    max_err = float(np.max(np.abs(out_fp32 - out_int8)))

    flat_32 = out_fp32.flatten()
    flat_8 = out_int8.flatten()
    denom = (np.linalg.norm(flat_32) * np.linalg.norm(flat_8))
    cosine_sim = float(np.dot(flat_32, flat_8) / denom) if denom > 0 else 1.0

    return {
        "fp32": fp32_bench,
        "int8": int8_bench,
        "compression_ratio": round(compression_ratio, 2),
        "size_reduction_percent": round(size_reduction_pct, 1),
        "latency_speedup_factor": round(speedup, 2),
        "mean_absolute_error": round(mae, 5),
        "max_absolute_error": round(max_err, 5),
        "cosine_similarity": round(cosine_sim, 5),
    }


# ---------------------------------------------------------------------------
# Batch Edge Quantization Suite
# ---------------------------------------------------------------------------

def quantize_edge_suite(
    input_dir: Union[str, Path] = "outputs/onnx",
    output_dir: Union[str, Path] = "outputs/onnx_int8",
    method: str = "all",
    benchmark: bool = True,
) -> Dict[str, Any]:
    """
    Processes all ONNX models in input_dir, applies INT8 dynamic and/or static
    post-training quantization, and compiles an edge deployment report.

    Args:
        input_dir: Directory containing FP32 .onnx models.
        output_dir: Directory to store INT8 .onnx models and benchmark report.
        method: "dynamic", "static", or "all".
        benchmark: Whether to run latency and compression benchmarking.

    Returns:
        Structured quantization and telemetry dictionary.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = list(in_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"[Warning] No .onnx files found in {in_dir}. Export models first via export_onnx.py.")
        return {}

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_directory": str(in_dir),
        "output_directory": str(out_dir),
        "method": method,
        "models": {},
    }

    for model_path in onnx_files:
        base_name = model_path.stem
        model_results: Dict[str, Any] = {}

        # 1. Dynamic INT8 Quantization
        if method in ("dynamic", "all"):
            dyn_out = out_dir / f"{base_name}_int8_dynamic.onnx"
            print(f"[Quantize] Dynamic INT8: {model_path.name} -> {dyn_out.name}...")
            t0 = time.time()
            try:
                quantize_model_dynamic(model_path, dyn_out)
                elapsed = time.time() - t0
                size_mb = dyn_out.stat().st_size / (1024 * 1024)
                dyn_res = {
                    "status": "success",
                    "path": str(dyn_out),
                    "size_mb": round(size_mb, 2),
                    "quant_time_sec": round(elapsed, 2),
                }
                if benchmark:
                    cmp = compare_fp32_vs_int8(model_path, dyn_out)
                    dyn_res["benchmark"] = cmp
                    print(f"  [+] Dynamic INT8: {size_mb:.2f} MB ({cmp['size_reduction_percent']}% reduction, {cmp['latency_speedup_factor']}x speedup, cosine sim: {cmp['cosine_similarity']})")
                else:
                    print(f"  [+] Dynamic INT8 created ({size_mb:.2f} MB)")
                model_results["dynamic"] = dyn_res
            except Exception as e:
                model_results["dynamic"] = {"status": "failed", "error": str(e)}
                print(f"  [x] Dynamic quantization failed for {model_path.name}: {e}")

        # 2. Static INT8 Quantization
        if method in ("static", "all"):
            stat_out = out_dir / f"{base_name}_int8_static.onnx"
            print(f"[Quantize] Static INT8 (PTQ): {model_path.name} -> {stat_out.name}...")
            t0 = time.time()
            try:
                quantize_model_static(model_path, stat_out, num_calibration_samples=16)
                elapsed = time.time() - t0
                size_mb = stat_out.stat().st_size / (1024 * 1024)
                stat_res = {
                    "status": "success",
                    "path": str(stat_out),
                    "size_mb": round(size_mb, 2),
                    "quant_time_sec": round(elapsed, 2),
                }
                if benchmark:
                    cmp = compare_fp32_vs_int8(model_path, stat_out)
                    stat_res["benchmark"] = cmp
                    print(f"  [+] Static INT8: {size_mb:.2f} MB ({cmp['size_reduction_percent']}% reduction, {cmp['latency_speedup_factor']}x speedup, cosine sim: {cmp['cosine_similarity']})")
                else:
                    print(f"  [+] Static INT8 created ({size_mb:.2f} MB)")
                model_results["static"] = stat_res
            except Exception as e:
                model_results["static"] = {"status": "failed", "error": str(e)}
                print(f"  [x] Static quantization failed for {model_path.name}: {e}")

        summary["models"][base_name] = model_results

    # Save summary report JSON
    report_file = out_dir / "int8_quantization_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-training INT8 quantization for edge device deployment.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/onnx",
        help="Directory containing exported FP32 ONNX models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/onnx_int8",
        help="Directory to save quantized INT8 models.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "dynamic", "static"],
        help="Quantization technique: dynamic, static (PTQ), or all.",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip latency and numerical parity benchmarking.",
    )
    args = parser.parse_args()

    quantize_edge_suite(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        benchmark=not args.no_benchmark,
    )


if __name__ == "__main__":
    main()
