"""
app.py

Func2OCT - Real-Time Retinal Auto-Capture, Quality Control & Clinical Turing Test Web Studio.
Integrates:
1. Live optic disc detection, foveal macula localization, geometric distance
   validation, and Modified Laplacian / Tenengrad frequency-domain focus clarity scoring.
2. Clinical Turing Test Module for Synthetic vs Real Retinal OCT Discrimination (outputs/demo_assets/).
3. CDSCO Medical Device Investigational Use Disclaimers under Medical Device Rules (MDR) 2017, Rule 51.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import gradio as gr

# Ensure src directory and outputs/demo_assets are in Python path
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "outputs", "demo_assets"))

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

# Import Clinical Turing Test Benchmark Session
try:
    from clinical_turing_test import (
        ClinicalTuringTestSession,
        CDSCO_TURING_DISCLAIMER,
    )
except ImportError:
    from outputs.demo_assets.clinical_turing_test import (
        ClinicalTuringTestSession,
        CDSCO_TURING_DISCLAIMER,
    )

# ------------------------------------------------------------------------------
# Global Singletons
# ------------------------------------------------------------------------------
_global_auto_capture_engine = AutoCaptureEngine(
    min_clarity_score=0.50,
    min_od_confidence=0.30,
    min_macula_confidence=0.25,
    required_consecutive_frames=3,
)

_global_turing_session = ClinicalTuringTestSession()


# ------------------------------------------------------------------------------
# Regulatory CDSCO Medical Device Disclaimers
# ------------------------------------------------------------------------------
CDSCO_REGULATORY_BANNER_HTML = """
<div style="background-color: #fff8e1; border: 1px solid #ffe082; border-left: 6px solid #f57f17; padding: 14px 18px; margin-bottom: 18px; border-radius: 6px; color: #37474f; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
    <div style="display: flex; align-items: center; gap: 12px;">
      <span style="font-size: 26px;">⚠️</span>
      <div>
        <strong style="font-size: 14.5px; color: #b71c1c; letter-spacing: 0.5px;">
          CDSCO INVESTIGATIONAL MEDICAL DEVICE NOTICE | RULE 51, MEDICAL DEVICE RULES (MDR) 2017
        </strong>
        <div style="font-size: 12.5px; margin-top: 3px; color: #424242; line-height: 1.45;">
          <strong>FOR INVESTIGATIONAL & ACADEMIC RESEARCH USE ONLY.</strong> NOT CLEARED OR APPROVED BY THE CENTRAL DRUGS STANDARD CONTROL ORGANIZATION (CDSCO, GOVT. OF INDIA) FOR IN-VIVO CLINICAL DIAGNOSIS, THERAPEUTIC MONITORING, OR COMMERCIAL USE.
        </div>
      </div>
    </div>
    <span style="background-color: #f57f17; color: #ffffff; font-weight: 700; font-size: 11px; padding: 5px 12px; border-radius: 4px; letter-spacing: 0.8px; text-transform: uppercase;">
      Investigational SaMD
    </span>
  </div>
</div>
"""

CDSCO_FULL_DISCLAIMER_MARKDOWN = """
### 📜 Central Drugs Standard Control Organization (CDSCO) Statutory Disclaimers
**Governing Authority**: Directorate General of Health Services, Central Drugs Standard Control Organization (CDSCO), Ministry of Health & Family Welfare, Government of India.

1. **Investigational Medical Device Status (Rule 51, MDR 2017)**:
   - This software platform (**Func2OCT**) is classified as an **Investigational Software as a Medical Device (SaMD)**.
   - It is manufactured, distributed, and operated exclusively for computational academic research, clinical feasibility evaluation, and generative AI benchmark validation.
   - It has **not** received import licenses, manufacturing licenses, or regulatory clearance under the *Drugs and Cosmetics Act, 1940* or *Medical Device Rules, 2017*.

2. **Absolute Clinical & Diagnostic Boundaries**:
   - The synthetic Optical Coherence Tomography (OCT) cross-sections, 32-slice volumetric sweeps, automated quality control telemetry, and downstream disease likelihood scores are **computational approximations**.
   - Under no circumstances should outputs from this system be used as primary or standalone clinical evidence for the diagnosis, staging, triage, surgical planning, or therapeutic management of ophthalmic conditions (including Age-Related Macular Degeneration, Diabetic Retinopathy, Diabetic Macular Edema, or Glaucoma).
   - Clinical decisions must always be made by a **Registered Medical Practitioner (Ophthalmologist)** using cleared conventional optical imaging hardware and standard-of-care clinical evaluations.

3. **Physical & Generative Limits**:
   - Retinal fundus photography captures superficial 2D reflected optical spectra, whereas clinical OCT relies on low-coherence near-infrared interferometry to penetrate intra-retinal layers.
   - Cross-modal generative diffusion synthesis cannot recover anatomical micro-structures that are physically obscured by dense cataracts, vitreous hemorrhages, or severe media opacities.

4. **Institutional Ethics Committee (IEC) Oversight**:
   - In accordance with the *National Ethical Guidelines for Biomedical and Health Research Involving Human Participants* (ICMR, 2017), any institutional evaluation or clinician reader study utilizing this platform must operate under approved Institutional Ethics Committee (IEC) protocols with appropriate patient data de-identification and informed consent.
"""


# ------------------------------------------------------------------------------
# Auto-Capture & Quality Control Handlers
# ------------------------------------------------------------------------------
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
        frame = create_demo_fundus(blur_level=0.0)
    else:
        frame = np.array(input_image)

    decision: AutoCaptureDecision = _global_auto_capture_engine.process_frame(frame)

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


# ------------------------------------------------------------------------------
# Clinical Turing Test Handlers
# ------------------------------------------------------------------------------
def load_turing_case(case_id_request: Optional[str] = None) -> Tuple[
    Optional[np.ndarray], str, str, str, str, str, str
]:
    """Loads a blind test case for reader discrimination."""
    global _global_turing_session
    case_info = _global_turing_session.get_blind_case(case_id_request)
    if not case_info:
        return None, "No Cases", "N/A", "N/A", "", "No benchmark cases found in outputs/demo_assets/.", ""

    img_path = case_info["image_path"]
    if os.path.exists(img_path):
        img_arr = np.array(Image.open(img_path))
    else:
        img_arr = np.zeros((256, 256), dtype=np.uint8)

    case_id = case_info["case_id"]
    category = case_info["pathology_category"]
    difficulty = case_info["difficulty"]
    progress = f"Evaluated: {case_info['evaluated_so_far']} / {case_info['total_available_cases']} benchmark cases"
    initial_prompt = "🔍 **Examination Ready**: Review the OCT B-scan above and formulate your discrimination hypothesis below."

    return img_arr, case_id, category, difficulty, case_id, initial_prompt, progress


def submit_turing_evaluation(
    case_id: str,
    reader_choice: str,
    confidence_pct: float,
    realism_score: int,
    layer_clarity_score: int,
    observed_artifacts: List[str],
    reader_notes: str,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Evaluates reader submission, reveals ground truth, and computes live Turing metrics."""
    global _global_turing_session
    if not case_id or case_id == "No Cases":
        return "⚠️ Please load a test case first.", "", "", {}

    try:
        res = _global_turing_session.submit_evaluation(
            case_id=case_id,
            reader_choice=reader_choice,
            confidence_pct=float(confidence_pct),
            realism_score=int(realism_score),
            layer_clarity_score=int(layer_clarity_score),
            observed_artifacts=observed_artifacts,
            reader_notes=reader_notes,
        )
    except Exception as e:
        return f"⚠️ Evaluation error: {str(e)}", "", "", {}

    # Format clinical reveal text
    truth_icon = "🏥" if res["ground_truth"] == "Real OCT" else "🤖"
    verdict_md = (
        f"### {res['verdict_badge']}\n\n"
        f"- **Ground Truth**: {truth_icon} **{res['ground_truth']}**\n"
        f"- **Your Clinical Choice**: **{res['reader_choice']}**\n"
        f"- **Evaluation Assessment**: {res['verdict_summary']}\n\n"
        f"#### 🔬 Clinical Landmarks & Biomarkers:\n"
        f"- **Anatomical Bands Identified**: {', '.join(res['retinal_landmarks'])}\n"
        f"- **Histological Notes**: {res['clinical_pearls']}\n"
        f"- **Discriminative Artifact Cues**: {res['artifact_cues']}\n"
    )

    metrics = res["session_metrics"]
    scoreboard_md = (
        f"### 📊 Cumulative Clinical Turing Test Telemetry\n\n"
        f"| Metric | Current Session Score |\n"
        f"| :--- | :--- |\n"
        f"| **Total Cases Evaluated** | `{metrics['total_evaluated']}` |\n"
        f"| **Reader Accuracy** | `{metrics['reader_accuracy_pct']}%` |\n"
        f"| **AI Turing Fooling Rate** | `{metrics['turing_fooling_rate_pct']}%` (Synthetic judged Real) |\n"
        f"| **Real OCT Accuracy** | `{metrics['real_accuracy_pct']}%` ({metrics['real_count']} cases) |\n"
        f"| **Synthetic OCT Accuracy** | `{metrics['synthetic_accuracy_pct']}%` ({metrics['synthetic_count']} cases) |\n"
        f"| **Mean Realism Score** | `{metrics['mean_realism_score']} / 5.0` |\n"
        f"| **Mean Layer Definition** | `{metrics['mean_layer_clarity']} / 5.0` |\n\n"
        f"> **Turing Indistinguishability Status**:\n"
        f"> {metrics['turing_verdict']}\n"
    )

    return verdict_md, scoreboard_md, f"Case {case_id} Recorded", metrics


def reset_turing_session() -> Tuple[str, str, Dict[str, Any]]:
    """Resets the reader evaluation session logs."""
    global _global_turing_session
    _global_turing_session.reset_session()
    metrics = _global_turing_session.get_summary_metrics()
    scoreboard_md = "Session reset. Telemetry buffer cleared. Load a blind case to begin a new reader trial."
    return "Session cleared.", scoreboard_md, metrics


# ------------------------------------------------------------------------------
# Main Application Builder
# ------------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    """Build the Gradio Web Application."""
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Func2OCT - Auto-Capture, Quality Control & Clinical Turing Studio", theme=theme) as demo:
        # 1. CDSCO Regulatory Alert Banner (Persistent at Top of Interface)
        gr.HTML(CDSCO_REGULATORY_BANNER_HTML)

        gr.Markdown(
            """
            # 👁️ Func2OCT: Retinal Auto-Capture & Diagnostic QC Studio
            ### Real-Time Retinal Landmark Detection, Focus Clarity Scoring, & Clinical Turing Test Benchmark
            """
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1: Live Auto-Capture Simulator & Feedback Loop
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Tab 2: Full Diagnostic QC & Landmark Inspector
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Tab 3: Interactive Clinical Turing Test (Synthetic vs Real OCT)
            # ------------------------------------------------------------------
            with gr.TabItem("🩺 Clinical Turing Test (Synthetic vs Real OCT)"):
                gr.Markdown(
                    """
                    ### 🔬 Clinical Turing Test: Synthetic vs. Real OCT Discrimination Module
                    **Investigational Benchmark Protocol**: Evaluate your ability as an expert reader to discriminate 
                    between genuine in-vivo clinical OCT B-scans and cross-modal synthetic OCT B-scans synthesized via 
                    Latent Volumetric Diffusion Models (LVDM).
                    """
                )

                with gr.Row():
                    # Left Column: Blind Image Viewer & Acquisition Info
                    with gr.Column(scale=5):
                        turing_img_viewer = gr.Image(
                            label="Blinded OCT B-Scan Cross-Section (256x256)",
                            type="numpy",
                            interactive=False,
                        )
                        with gr.Row():
                            turing_case_id_display = gr.Textbox(label="Trial Case Identifier", interactive=False)
                            turing_difficulty_display = gr.Textbox(label="Difficulty Tier", interactive=False)
                            turing_pathology_display = gr.Textbox(label="Clinical Region / Pathology", interactive=False)

                        turing_progress_text = gr.Markdown("Click 'Load Next Blind Case' to begin reader evaluation.")
                        load_case_btn = gr.Button("🎲 Load Next Blind Case", variant="primary")

                    # Right Column: Clinician Discrimination & Likert Scoring
                    with gr.Column(scale=5):
                        hidden_case_id = gr.Textbox(visible=False)

                        reader_choice = gr.Radio(
                            choices=["🏥 Genuine Real OCT B-Scan", "🤖 Synthetic Generative OCT B-Scan"],
                            label="Your Clinical Discrimination Hypothesis",
                            value="🏥 Genuine Real OCT B-Scan",
                        )

                        confidence_slider = gr.Slider(
                            minimum=50.0,
                            maximum=100.0,
                            value=80.0,
                            step=5.0,
                            label="Diagnostic Confidence (%)",
                        )

                        with gr.Row():
                            realism_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=4,
                                step=1,
                                label="Anatomical Realism (1=Artificial, 5=Indistinguishable)",
                            )
                            layer_clarity_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=4,
                                step=1,
                                label="Layer Stratification (1=Diffuse, 5=Sharp ILM/RPE)",
                            )

                        observed_artifacts = gr.CheckboxGroup(
                            choices=[
                                "None / Anatomically Natural",
                                "Speckle Incoherence",
                                "Layer Boundary Discontinuity",
                                "Atypical Retinal Curvature",
                                "Hyperreflective Band Blur",
                                "Choroidal Vascular Regularity",
                            ],
                            value=["None / Anatomically Natural"],
                            label="Observed Morphological Cues / Artifacts",
                        )

                        reader_notes = gr.Textbox(
                            label="Clinical Impression & Rationale (Optional)",
                            placeholder="e.g., Continuous external limiting membrane and natural choroidal attenuation observed...",
                            lines=2,
                        )

                        with gr.Row():
                            submit_turing_btn = gr.Button("🎯 Submit Diagnosis & Reveal Ground Truth", variant="secondary")
                            reset_turing_btn = gr.Button("🔄 Reset Session", variant="stop")

                # Dynamic Reveal & Telemetry Section
                with gr.Row():
                    with gr.Column(scale=6):
                        turing_verdict_box = gr.Markdown(
                            value="*Awaiting submission: Select your diagnosis above and submit to reveal ground truth and histological analysis.*"
                        )
                    with gr.Column(scale=4):
                        turing_scoreboard_box = gr.Markdown(
                            value="*Reader session metrics will update after the first completed evaluation.*"
                        )
                        turing_metrics_json = gr.JSON(label="Telemetry Metrics", visible=False)

                # Turing Test Wiring
                load_case_btn.click(
                    fn=load_turing_case,
                    inputs=[],
                    outputs=[
                        turing_img_viewer,
                        turing_case_id_display,
                        turing_pathology_display,
                        turing_difficulty_display,
                        hidden_case_id,
                        turing_verdict_box,
                        turing_progress_text,
                    ],
                )

                submit_turing_btn.click(
                    fn=submit_turing_evaluation,
                    inputs=[
                        hidden_case_id,
                        reader_choice,
                        confidence_slider,
                        realism_slider,
                        layer_clarity_slider,
                        observed_artifacts,
                        reader_notes,
                    ],
                    outputs=[
                        turing_verdict_box,
                        turing_scoreboard_box,
                        turing_progress_text,
                        turing_metrics_json,
                    ],
                )

                reset_turing_btn.click(
                    fn=reset_turing_session,
                    inputs=[],
                    outputs=[
                        turing_verdict_box,
                        turing_scoreboard_box,
                        turing_metrics_json,
                    ],
                )

            # ------------------------------------------------------------------
            # Tab 4: Synthetic Calibration Generator
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Tab 5: CDSCO Statutory Disclaimers & Regulatory Framework
            # ------------------------------------------------------------------
            with gr.TabItem("📜 CDSCO Regulatory Disclaimers & SaMD Governance"):
                gr.Markdown(CDSCO_FULL_DISCLAIMER_MARKDOWN)

        gr.Markdown(
            """
            ---
            **Func2OCT Clinical Architecture** | Standardized Retinal Tolerances (2.0 - 3.5 DD, +/-15° temporal axis) | Nayar Modified Laplacian & 2D FFT Tenengrad Spectral Gradient  
            **Regulatory Classification**: Investigational Software as a Medical Device (SaMD) under CDSCO Medical Device Rules 2017 (Rule 51). Strictly for academic and research investigation.
            """
        )

    return demo


# ------------------------------------------------------------------------------
# Top-level ASGI/WSGI/Serverless exports for hosting platforms (Vercel, Uvicorn, HF)
# ------------------------------------------------------------------------------
demo = build_app()
app = demo.app
application = app
handler = app


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
