"""
app.py

Func2OCT - Real-Time Retinal Auto-Capture & Quality Control Web Studio.
Integrates live optic disc detection, foveal macula localization, geometric distance
validation, and Modified Laplacian / Tenengrad frequency-domain focus clarity scoring
into an interactive auto-capture feedback loop.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

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

# Persistent Auto-Capture Engine instance for stateful video streams
_global_auto_capture_engine = AutoCaptureEngine(
    min_clarity_score=0.50,
    min_od_confidence=0.30,
    min_macula_confidence=0.25,
    required_consecutive_frames=3,
)


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
