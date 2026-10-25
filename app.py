"""
app.py

Func2OCT - Real-Time Retinal Auto-Capture & Quality Control Web Studio.
Integrates live optic disc detection, foveal macula localization, geometric distance
validation, and Modified Laplacian / Tenengrad frequency-domain focus clarity scoring
into an interactive auto-capture feedback loop.
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from fundus2oct.data.preprocessing import preprocess_fundus
from fundus2oct.data.quality_control import (
    QCStatus,
    check_fundus_quality,
    detect_optic_disc,
    detect_macula,
    verify_optic_disc_macula_geometry,
    compute_focus_clarity_score,
    draw_geometric_verification,
    draw_focus_clarity,
)
from fundus2oct.data.auto_capture import (
    AutoCaptureEngine,
    AutoCaptureDecision,
    GuidanceCue,
    create_demo_fundus,
)
from fundus2oct.evaluation.gradcam import compute_gradcam
from fundus2oct.evaluation.uncertainty import estimate_predictive_uncertainty

# Persistent Auto-Capture Engine instance for stateful video streams
_global_auto_capture_engine = AutoCaptureEngine(
    min_clarity_score=0.50,
    min_od_confidence=0.30,
    min_macula_confidence=0.25,
    required_consecutive_frames=3,
)

# Persistent model cache for web demo
_classifier_cache: Dict[str, Any] = {}


def get_cached_classifier(model_type: str = "Fundus-only Classifier (Model A)"):
    """Loads and caches the requested classifier model."""
    if model_type in _classifier_cache:
        return _classifier_cache[model_type]

    device = torch.device("cpu")
    if "Fundus-only" in model_type:
        from fundus2oct.models.classifier_fundus import FundusClassifier
        model = FundusClassifier()
        ckpt_path = Path("checkpoints/classifier_fundus.pt")
        if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
            try:
                model.load_checkpoint(ckpt_path, device=device)
            except Exception as e:
                print(f"[Warning] Failed to load FundusClassifier checkpoint: {e}")
        model.eval()
        _classifier_cache[model_type] = model
        return model
    else:
        from fundus2oct.models.classifier_multimodal import MultimodalClassifier
        model = MultimodalClassifier()
        ckpt_path = Path("checkpoints/classifier_multimodal.pt")
        if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
            try:
                model.load_checkpoint(ckpt_path, device=device)
            except Exception as e:
                print(f"[Warning] Failed to load MultimodalClassifier checkpoint: {e}")
        model.eval()
        _classifier_cache[model_type] = model
        return model



def process_auto_capture_stream(
    input_image: Optional[Image.Image],
    focus_threshold: float,
    required_frames: int,
    engine_reset: bool,
) -> Tuple[Optional[np.ndarray], str, float, Optional[np.ndarray], str]:
    """Process a live frame or uploaded image through the Auto-Capture Feedback Loop."""
    global _global_auto_capture_engine

    if engine_reset:
        _global_auto_capture_engine.reset()

    _global_auto_capture_engine.min_clarity_score = float(focus_threshold)
    _global_auto_capture_engine.required_consecutive_frames = int(required_frames)

    if input_image is None:
        # Default placeholder frame
        frame = create_demo_fundus(blur_level=0.0)
    else:
        frame = np.array(input_image)

    # Run auto-capture feedback loop evaluation
    decision: AutoCaptureDecision = _global_auto_capture_engine.process_frame(frame)

    # Format telemetry text
    status_icon = "🟢" if decision.should_capture else ("🟡" if decision.is_focus_valid else "🔴")
    status_summary = (
        f"### {status_icon} Feedback: {decision.feedback_message}\n\n"
        f"- **Guidance Cue**: `{decision.guidance_cue.value}`\n"
        f"- **Lock-in Buffer**: `{decision.consecutive_passes} / {_global_auto_capture_engine.required_consecutive_frames}` frames\n"
        f"- **Focus Clarity**: `{decision.metrics.get('clarity_score', 0.0) * 100:.1f}%` "
        f"(MLV: `{decision.metrics.get('modified_laplacian_variance', 0.0):.1f}`, "
        f"Freq Spec: `{decision.metrics.get('tenengrad_spectral_energy', 0.0):.4f}`)\n"
        f"- **Optic Disc**: `{decision.metrics.get('od_confidence', 0.0) * 100:.0f}% confidence` "
        f"({'✓ Valid' if decision.is_optic_disc_valid else '✗ Invalid'})\n"
        f"- **Macula**: `{decision.metrics.get('macula_confidence', 0.0) * 100:.0f}% confidence` "
        f"({'✓ Valid' if decision.is_macula_valid else '✗ Invalid'})\n"
        f"- **Retinal Distance**: `{decision.metrics.get('distance_dd', 0.0):.2f} DD` "
        f"({'✓ Anatomical' if decision.is_geometry_valid else '✗ Out of bounds'})\n"
    )

    captured_frame = frame if decision.should_capture else None
    action_text = "📸 AUTO-CAPTURE TRIGGERED!" if decision.should_capture else "Searching / Holding focus..."

    return decision.annotated_frame, status_summary, decision.stability_progress, captured_frame, action_text


def full_quality_control_inspection(
    image: Optional[Image.Image],
    verify_geom: bool,
    expected_eye: str,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """Execute full clinical quality control screening and landmark verification."""
    if image is None:
        return None, "Please upload or select an image to inspect.", {}

    img_arr = np.array(image)
    eye_param = None if expected_eye == "Auto-detect" else expected_eye
    report = check_fundus_quality(img_arr, verify_geometry=verify_geom, expected_laterality=eye_param)

    # Generate rich annotated visualization
    annotated = img_arr.copy()
    try:
        od = detect_optic_disc(img_arr)
        mac = detect_macula(img_arr, optic_disc=od)
        if report.geometric_qc is not None:
            annotated = draw_geometric_verification(annotated, od, mac, report.geometric_qc)
        if report.focus_clarity is not None:
            annotated = draw_focus_clarity(annotated, report.focus_clarity)
    except Exception:
        pass

    summary = report.summary_text()
    return annotated, summary, report.metrics


def run_interpretability_and_uncertainty(
    image: Optional[Image.Image],
    model_choice: str,
    target_class_choice: str,
    num_passes: int,
    uncertainty_metric: str,
    threshold: float,
    alpha: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any]]:
    """Generates Grad-CAM visual explanation and Monte Carlo Dropout uncertainty metrics."""
    if image is None:
        raw_arr = create_demo_fundus(blur_level=0.0)
        image = Image.fromarray(raw_arr)

    model = get_cached_classifier(model_choice)
    is_trained = getattr(model, "is_trained", False)

    # Preprocess fundus image to normalized tensor (1, 3, 256, 256)
    img_tensor = preprocess_fundus(
        image,
        target_size=(256, 256),
        center_crop=True,
        normalize="imagenet",
    ).unsqueeze(0)

    # Target class parsing
    if "Auto" in target_class_choice:
        target_class = None
    elif "0:" in target_class_choice:
        target_class = 0
    else:
        target_class = 1

    # Multimodal vs single-stream inputs
    if "Multimodal" in model_choice:
        dummy_oct = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
        inputs: Any = (img_tensor, dummy_oct)
        modality = "fundus"
    else:
        inputs = img_tensor
        modality = "fundus"

    try:
        # Compute Grad-CAM
        cam_result = compute_gradcam(
            model=model,
            inputs=inputs,
            target_class=target_class,
            modality=modality,
        )

        # Compute Monte Carlo Dropout Predictive Uncertainty
        unc_result = estimate_predictive_uncertainty(
            model=model,
            inputs=inputs,
            num_passes=int(num_passes),
            threshold=float(threshold),
            uncertainty_metric=uncertainty_metric,
            seed=42,
        )

        # Visualizations
        overlay_img = cam_result.overlay(image, alpha=float(alpha))
        heatmap_uint8 = np.uint8(255.0 * cam_result.heatmap)
        heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

        # Diagnostic text
        class_names = ["Control / Non-pathology (0)", "Macular Degeneration (1)"]
        pred_name = class_names[unc_result.predicted_class]
        confidence_pct = unc_result.mean_probabilities[unc_result.predicted_class] * 100.0

        flag_icon = "⚠️" if unc_result.is_high_uncertainty else "🟢"
        flag_title = "HIGH PREDICTIVE UNCERTAINTY" if unc_result.is_high_uncertainty else "CONFIDENT / LOW UNCERTAINTY"
        status_note = ""
        if not is_trained:
            status_note = "\n\n> [!WARNING]\n> **Untrained Checkpoint Warning**: Classifier checkpoint weights were initialized or uncalibrated. Displaying research telemetry."

        summary_md = (
            f"### {flag_icon} Diagnosis: `{pred_name}` ({confidence_pct:.1f}% confidence)\n\n"
            f"**Decision Status**: `{flag_title}`\n\n"
            f"- **Explained Layer**: `{cam_result.target_layer_name}` (Modality: `{cam_result.modality_explained}`)\n"
            f"- **Target Class Explained**: `{class_names[cam_result.target_class]}`\n"
            f"- **Uncertainty Score ({unc_result.uncertainty_metric})**: `{unc_result.uncertainty_score:.4f}` "
            f"(Threshold: `{unc_result.threshold:.4f}`)\n"
            f"- **Total Predictive Entropy**: `{unc_result.predictive_entropy_nats:.4f} nats` (`{unc_result.predictive_entropy_bits:.4f} bits`)\n"
            f"- **Epistemic Uncertainty (Mutual Info)**: `{unc_result.mutual_information_nats:.4f} nats`\n"
            f"- **Aleatoric Uncertainty (Expected Entropy)**: `{unc_result.expected_entropy_nats:.4f} nats`\n"
            f"- **Monte Carlo Stochastic Passes**: `{unc_result.num_passes}`\n"
            f"{status_note}\n\n"
            f"> [!NOTE]\n"
            f"> **Clinical Disclaimer**: Grad-CAM visual heatmaps highlight regional associative feature importance, "
            f"not biological ground truth. Predictive uncertainty reflects model variance under MC Dropout and is not "
            f"a calibrated diagnostic error probability."
        )

        telemetry = {
            "gradcam": {
                "target_layer": cam_result.target_layer_name,
                "target_class": cam_result.target_class,
                "predicted_class": cam_result.predicted_class,
                "class_probabilities": cam_result.class_probabilities.tolist(),
            },
            "uncertainty": unc_result.to_dict(),
        }

        return overlay_img, heatmap_rgb, summary_md, telemetry

    except Exception as e:
        err_msg = f"### ❌ Interpretability Pipeline Error\n\n`{str(e)}`"
        return None, None, err_msg, {"error": str(e)}


def build_app() -> gr.Blocks:
    """Build the Gradio Web Application."""
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Func2OCT - Auto-Capture & Quality Control Studio", theme=theme) as demo:
        gr.Markdown(
            """
            # 👁️ Func2OCT: Retinal Auto-Capture & Diagnostic QC Studio
            ### Real-Time Retinal Landmark Detection, Focus Clarity Scoring, & Geometric Verification
            """
        )

        with gr.Tabs():
            # Tab 1: Live Auto-Capture Simulator & Feedback Loop
            with gr.TabItem("🎯 Real-Time Auto-Capture"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_feed = gr.Image(
                            label="Live Fundus Camera / Simulated Video Feed",
                            type="pil",
                            sources=["upload", "webcam"],
                        )
                        with gr.Row():
                            focus_thresh = gr.Slider(
                                minimum=0.20,
                                maximum=0.80,
                                value=0.50,
                                step=0.05,
                                label="Minimum Focus Clarity Threshold",
                            )
                            lockin_frames = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Stability Lock-in Frames",
                            )
                            reset_btn = gr.Checkbox(label="Reset State Buffer", value=False)

                        run_capture_btn = gr.Button("⚡ Evaluate Live Frame", variant="primary")

                    with gr.Column(scale=5):
                        hud_display = gr.Image(label="Live Augmented Reality HUD Overlay", type="numpy")
                        stability_bar = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            interactive=False,
                            label="Lock-in Progress",
                        )
                        feedback_box = gr.Markdown(value="Waiting for frame evaluation...")
                        captured_box = gr.Image(label="Frozen Auto-Captured High-Res Frame", type="numpy")
                        capture_status_badge = gr.Textbox(label="Auto-Capture Action", interactive=False)

                run_capture_btn.click(
                    fn=process_auto_capture_stream,
                    inputs=[input_feed, focus_thresh, lockin_frames, reset_btn],
                    outputs=[hud_display, feedback_box, stability_bar, captured_box, capture_status_badge],
                )

            # Tab 2: Full Diagnostic QC & Landmark Inspector
            with gr.TabItem("🔬 Static Quality Control Screening"):
                with gr.Row():
                    with gr.Column(scale=5):
                        qc_image_in = gr.Image(label="Upload Retinal Fundus Photograph", type="pil")
                        with gr.Row():
                            verify_geom_box = gr.Checkbox(label="Verify Landmark Geometry", value=True)
                            eye_select = gr.Radio(
                                choices=["Auto-detect", "OD (Right Eye)", "OS (Left Eye)"],
                                value="Auto-detect",
                                label="Expected Laterality",
                            )
                        run_qc_btn = gr.Button("Run Diagnostic Quality Screening", variant="primary")

                    with gr.Column(scale=5):
                        qc_image_out = gr.Image(label="Augmented QC Diagnostic Visualization", type="numpy")
                        qc_report_out = gr.Textbox(label="Quality Control Report", lines=10)
                        qc_metrics_out = gr.JSON(label="Numerical Telemetry")

                run_qc_btn.click(
                    fn=full_quality_control_inspection,
                    inputs=[qc_image_in, verify_geom_box, eye_select],
                    outputs=[qc_image_out, qc_report_out, qc_metrics_out],
                )

            # Tab 3: Synthetic Calibration Generator
            with gr.TabItem("🧪 Calibration & Synthetic Generator"):
                gr.Markdown("Generate calibrated synthetic fundus images to test auto-capture responses under controlled blur and displacement.")
                with gr.Row():
                    with gr.Column():
                        blur_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Focus Ring Blur (0.0=Sharp, 1.0=Severe)")
                        disc_x_slider = gr.Slider(100, 300, value=160, step=10, label="Optic Disc Center X")
                        macula_dd_slider = gr.Slider(1.5, 4.2, value=2.7, step=0.1, label="Macula Distance (Disc Diameters)")
                        gen_btn = gr.Button("Generate Test Image", variant="secondary")
                    with gr.Column():
                        gen_output = gr.Image(label="Generated Synthetic Fundus", type="numpy")

                gen_btn.click(
                    fn=create_demo_fundus,
                    inputs=[blur_slider, disc_x_slider, gr.Number(value=250, visible=False), macula_dd_slider],
                    outputs=gen_output,
                )

            # Tab 4: Grad-CAM Interpretability & Predictive Uncertainty
            with gr.TabItem("🔍 Interpretability & Uncertainty"):
                gr.Markdown(
                    """
                    ### 🔍 Grad-CAM Saliency Maps & Monte Carlo Dropout Uncertainty Flagging
                    Visualize convolutional feature attention across retinal landmarks and quantify epistemic/aleatoric uncertainty.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        interp_img_in = gr.Image(label="Upload Retinal Fundus Photograph", type="pil")
                        with gr.Row():
                            model_select = gr.Dropdown(
                                choices=["Fundus-only Classifier (Model A)", "Multimodal Classifier (Model B)"],
                                value="Fundus-only Classifier (Model A)",
                                label="Target Classifier Model",
                            )
                            target_class_select = gr.Radio(
                                choices=["Auto (Predicted Class)", "0: Control / Non-pathology", "1: Macular Degeneration (MD)"],
                                value="Auto (Predicted Class)",
                                label="Class to Explain",
                            )
                        with gr.Row():
                            mc_passes_slider = gr.Slider(
                                minimum=2,
                                maximum=30,
                                value=15,
                                step=1,
                                label="MC Dropout Passes (T)",
                            )
                            unc_metric_select = gr.Dropdown(
                                choices=["predictive_entropy", "mutual_information", "expected_entropy", "variance"],
                                value="predictive_entropy",
                                label="Uncertainty Metric",
                            )
                            threshold_slider = gr.Slider(
                                minimum=0.05,
                                maximum=1.00,
                                value=0.50,
                                step=0.05,
                                label="High Uncertainty Threshold",
                            )
                            alpha_slider = gr.Slider(
                                minimum=0.10,
                                maximum=0.90,
                                value=0.45,
                                step=0.05,
                                label="Grad-CAM Overlay Alpha",
                            )
                        run_interp_btn = gr.Button("⚡ Analyze Interpretability & Uncertainty", variant="primary")

                    with gr.Column(scale=5):
                        with gr.Row():
                            interp_overlay_out = gr.Image(label="Grad-CAM Retinal Saliency Overlay", type="numpy")
                            interp_heatmap_out = gr.Image(label="Normalized Heatmap (JET)", type="numpy")
                        interp_report_out = gr.Markdown(value="Upload an image and run analysis to inspect Grad-CAM and uncertainty.")
                        interp_metrics_out = gr.JSON(label="Numerical Uncertainty Telemetry")

                run_interp_btn.click(
                    fn=run_interpretability_and_uncertainty,
                    inputs=[
                        interp_img_in,
                        model_select,
                        target_class_select,
                        mc_passes_slider,
                        unc_metric_select,
                        threshold_slider,
                        alpha_slider,
                    ],
                    outputs=[
                        interp_overlay_out,
                        interp_heatmap_out,
                        interp_report_out,
                        interp_metrics_out,
                    ],
                )

        gr.Markdown(
            """
            ---
            **Func2OCT Clinical Architecture** | Standardized Retinal Tolerances (2.0 - 3.5 DD, +/-15° temporal axis) | Nayar Modified Laplacian & 2D FFT Tenengrad Spectral Gradient
            """
        )

    return demo


# Top-level ASGI/WSGI/Serverless exports for hosting platforms (Vercel, Uvicorn, Hugging Face, etc.)
# Resolves: [PYTHON_ENTRYPOINT_NOT_FOUND] Found app.py but it does not export a top-level "app", "application", or "handler" variable.
demo = build_app()
app = demo.app
application = app
handler = app


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
