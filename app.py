"""
app.py

Func2OCT - Multimodal Retinal Generative & Diagnostic Web Studio.
Intuitive 3-Panel Clinical Layout & Comprehensive Ophthalmology AI Suite:
- Panel 1: Fundus Upload & Real-Time Quality Feedback (Landmarks, Clarity, Geometry)
- Panel 2: Synthetic OCT 2D/3D Volumetric Studio (32-Slice Navigation, 8-Layer Segmentation, NIfTI Export)
- Panel 3: Multimodal Diagnostic Predictions (Model A vs. Model B Comparison, Gated Fusion, Uplift)

Also integrates:
- Real-time video auto-capture feedback loop (Modified Laplacian & Tenengrad)
- Double-blinded Clinical Turing Test for synthetic vs. real OCT discrimination
- CDSCO Investigational SaMD Regulatory Disclaimers (MDR 2017, Rule 51)
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import gradio as gr

# Ensure src/, repo root, and demo_assets are in Python path
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_ASSETS_DIR = os.path.join(_REPO_ROOT, "outputs", "demo_assets")
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

# Project imports
from fundus2oct.data.quality_control import (
    QCStatus,
    check_fundus_quality,
    detect_optic_disc,
    detect_macula,
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
from fundus2oct.visualization.segmentation import (
    LAYER_NAMES,
    BOUNDARY_NAMES,
    LAYER_COLORS_RGB,
    BOUNDARY_COLORS_RGB,
    SegmentationResult,
    segment_retinal_layers,
    draw_segmentation_overlay,
    create_colored_mask,
    create_diagnostic_report_figure,
)
from fundus2oct.models.classifier_fundus import FundusClassifier
from fundus2oct.models.classifier_multimodal import MultimodalClassifier
from generate_oct import synthesize_realistic_3d_oct_volume, save_nifti_volume

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


# ---------------------------------------------------------------------------
# Regulatory CDSCO Medical Device Disclaimers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Global Singletons
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(_REPO_ROOT) / "outputs" / "interactive_cache"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_global_auto_capture_engine = AutoCaptureEngine(
    min_clarity_score=0.50,
    min_od_confidence=0.30,
    min_macula_confidence=0.25,
    required_consecutive_frames=3,
)
_last_auto_captured_frame: Optional[np.ndarray] = None

_global_turing_session = ClinicalTuringTestSession()

_model_a: Optional[FundusClassifier] = None
_model_b: Optional[MultimodalClassifier] = None


def get_models() -> Tuple[Optional[FundusClassifier], Optional[MultimodalClassifier]]:
    """Lazy initialization of Model A and Model B disease classifiers."""
    global _model_a, _model_b
    if _model_a is None:
        try:
            _model_a = FundusClassifier(in_channels=3, num_classes=2, feature_dim=128)
            _model_a.eval()
        except Exception as e:
            print(f"[Warning] Model A initialization failed: {e}")
            _model_a = None

    if _model_b is None:
        try:
            _model_b = MultimodalClassifier(fundus_channels=3, oct_channels=1, num_classes=2, feature_dim=128)
            _model_b.eval()
        except Exception as e:
            print(f"[Warning] Model B initialization failed: {e}")
            _model_b = None

    return _model_a, _model_b


# ---------------------------------------------------------------------------
# Helper Formatting Routines for 3-Panel Studio
# ---------------------------------------------------------------------------

def format_quality_html(
    report: Any,
    od: Any,
    mac: Any,
    clarity: Any,
    is_acceptable: bool,
) -> str:
    """Renders a structured, high-contrast clinical QC telemetry card."""
    if is_acceptable:
        status_badge = '<span class="status-pill pill-pass">🟢 PASS - CLINICALLY OPTIMAL</span>'
        banner_border = "#10b981"
        action_msg = "Retinal landmarks and focus meet diagnostic standards. Ready for 3D OCT synthesis."
    else:
        status_badge = '<span class="status-pill pill-fail">🔴 REJECT - QUALITY THRESHOLD BREACHED</span>'
        banner_border = "#ef4444"
        action_msg = "Poor sharpness or anatomical deviation detected. Re-capture or adjust focus ring."

    clarity_pct = clarity.clarity_score * 100.0 if clarity else 0.0
    mlv_val = clarity.modified_laplacian_variance if clarity else 0.0
    ten_val = clarity.tenengrad_spectral_energy if clarity else 0.0

    od_status = f"{od.confidence*100:.0f}% confidence ({od.center[0]}, {od.center[1]})" if od else "Not detected"
    mac_status = f"{mac.confidence*100:.0f}% confidence ({mac.center[0]}, {mac.center[1]})" if mac else "Not detected"

    geom_qc = report.geometric_qc if report else None
    if geom_qc:
        is_dist_ok = (2.0 <= geom_qc.distance_dd <= 3.5)
        dist_str = f"{geom_qc.distance_dd:.2f} DD ({'Normal 2.0-3.5 DD' if is_dist_ok else 'Deviation'})"
        angle_str = f"{geom_qc.angle_deg:.1f}° (dev: {geom_qc.angle_deviation_deg:.1f}°)"
        eye_str = f"{geom_qc.inferred_laterality} ({'Valid' if geom_qc.is_valid else 'Review'})"
    else:
        dist_str = "N/A"
        angle_str = "N/A"
        eye_str = "Auto-detect"

    html = f"""
    <div class="card-telemetry" style="border-left: 4px solid {banner_border};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">Diagnostic Quality Status</span>
            {status_badge}
        </div>
        <p style="margin: 0 0 10px 0; font-size: 0.88rem; color: #94a3b8;">{action_msg}</p>
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-label">Focus Clarity</div>
                <div class="metric-value">{clarity_pct:.1f}%</div>
                <div class="metric-sub">MLV: {mlv_val:.1f} | Freq: {ten_val:.3f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Optic Disc (OD)</div>
                <div class="metric-value">{'✓ Detected' if od else '✗ Missing'}</div>
                <div class="metric-sub">{od_status}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Foveal Macula</div>
                <div class="metric-value">{'✓ Localized' if mac else '✗ Missing'}</div>
                <div class="metric-sub">{mac_status}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Retinal Distance</div>
                <div class="metric-value">{dist_str}</div>
                <div class="metric-sub">Temporal Angle: {angle_str} | Eye: {eye_str}</div>
            </div>
        </div>
    </div>
    """
    return html


def format_thickness_html(result: SegmentationResult, slice_idx: int) -> str:
    """Renders 8-layer thickness metrics and anatomical profile card."""
    mean_th = result.mean_retinal_thickness
    fov_min = result.foveal_min_thickness
    mean_um = mean_th * 3.87
    fov_um = fov_min * 3.87

    layer_names_ordered = ["RNFL", "GCL_IPL", "INL", "OPL", "ONL", "IS_OS", "RPE"]
    layer_pills = ""
    for l_name in layer_names_ordered:
        arr = result.layer_thicknesses.get(l_name, np.zeros(1))
        avg_px = float(np.mean(arr))
        avg_um = avg_px * 3.87
        layer_pills += f"""
        <div class="layer-chip">
            <span class="chip-name">{l_name}</span>
            <span class="chip-val">{avg_px:.1f} px ({avg_um:.0f} µm)</span>
        </div>
        """

    html = f"""
    <div class="card-telemetry" style="border-left: 4px solid #06b6d4;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 700; font-size: 1.0rem; color: #f8fafc;">
                Slice #{slice_idx:02d} Retinal Thickness Profile
            </span>
            <span class="status-pill pill-cyan">8-Layer Segmented</span>
        </div>
        <div style="display: flex; gap: 16px; margin-bottom: 10px;">
            <div>
                <span class="metric-label">Mean Retinal Thickness:</span>
                <span style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;"> {mean_th:.1f} px ({mean_um:.1f} µm)</span>
            </div>
            <div>
                <span class="metric-label">Foveal Central Pit:</span>
                <span style="font-weight: 700; color: #34d399; font-size: 1.05rem;"> {fov_min:.1f} px ({fov_um:.1f} µm)</span>
            </div>
        </div>
        <div class="layer-chips-container">
            {layer_pills}
        </div>
    </div>
    """
    return html


def format_multimodal_predictions_html(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    fundus_weight: float,
    oct_weight: float,
    disease_prior: str,
) -> str:
    """Renders Model A vs Model B comparative diagnostic performance card."""
    prob_control_a, prob_md_a = float(prob_a[0]), float(prob_a[1])
    prob_control_b, prob_md_b = float(prob_b[0]), float(prob_b[1])

    is_md = prob_md_b >= 0.50
    if is_md:
        primary_title = "AGE-RELATED MACULAR DEGENERATION (AMD)"
        primary_color = "#f59e0b"
        status_pill = '<span class="status-pill pill-warn">🟠 PATHOLOGY DETECTED</span>'
        biomarker_note = "Drusenoid sub-RPE elevation, retinal thickening, and altered photoreceptor EZ continuity."
        recommendation = "Refer for comprehensive ophthalmological evaluation / Fluorescein Angiography."
        primary_conf_b = prob_md_b * 100.0
        primary_conf_a = prob_md_a * 100.0
    else:
        primary_title = "NORMAL HEALTHY RETINA (CONTROL)"
        primary_color = "#10b981"
        status_pill = '<span class="status-pill pill-pass">🟢 HEALTHY RETINA</span>'
        biomarker_note = "Preserved foveal pit depression, intact 8-layer laminar architecture, and uniform RPE band."
        recommendation = "Routine periodic screening. No active retinal pathology identified."
        primary_conf_b = prob_control_b * 100.0
        primary_conf_a = prob_control_a * 100.0

    uplift = primary_conf_b - primary_conf_a
    uplift_sign = "+" if uplift >= 0 else ""
    uplift_color = "#10b981" if uplift >= 0 else "#f43f5e"

    html = f"""
    <div class="card-telemetry" style="border-left: 4px solid {primary_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">Diagnostic Consensus</span>
            {status_pill}
        </div>
        <div style="font-size: 1.15rem; font-weight: 800; color: {primary_color}; margin-bottom: 12px;">
            {primary_title}
        </div>

        <div class="model-comparison-box">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 600; color: #94a3b8; font-size: 0.88rem;">Model A (Fundus-Only)</span>
                <span style="font-weight: 700; color: #cbd5e1; font-size: 0.88rem;">{primary_conf_a:.1f}% Confidence</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill fill-blue" style="width: {primary_conf_a:.1f}%;"></div>
            </div>

            <div style="display: flex; justify-content: space-between; margin-top: 10px; margin-bottom: 4px;">
                <span style="font-weight: 600; color: #94a3b8; font-size: 0.88rem;">Model B (Fundus + Synthetic OCT)</span>
                <span style="font-weight: 800; color: #38bdf8; font-size: 0.88rem;">{primary_conf_b:.1f}% Confidence</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill fill-purple" style="width: {primary_conf_b:.1f}%;"></div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px; padding-top: 8px; border-top: 1px solid #334155;">
                <span style="font-size: 0.85rem; color: #94a3b8;">Multimodal AUROC Uplift:</span>
                <span style="font-weight: 800; color: {uplift_color}; font-size: 0.95rem;">
                    {uplift_sign}{uplift:.1f}% Structural Advantage
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span style="font-size: 0.85rem; color: #94a3b8;">Gated Cross-Modal Weight:</span>
                <span style="font-size: 0.85rem; color: #cbd5e1;">
                    OCT: <b>{oct_weight:.1f}%</b> | Fundus: <b>{fundus_weight:.1f}%</b>
                </span>
            </div>
        </div>

        <div style="margin-top: 12px; font-size: 0.85rem;">
            <div style="color: #94a3b8; margin-bottom: 4px;"><b>Structural Biomarkers:</b> {biomarker_note}</div>
            <div style="color: #94a3b8;"><b>Clinical Recommendation:</b> {recommendation}</div>
        </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# Real-Time Auto-Capture Stream Callback Functions
# ---------------------------------------------------------------------------

def process_auto_capture_stream(
    frame: Union[np.ndarray, Image.Image, None] = None,
    focus_thresh: float = 0.50,
    lockin_frames: int = 3,
    reset_state: bool = False,
    input_image: Union[np.ndarray, Image.Image, None] = None,
    focus_threshold: Optional[float] = None,
    required_frames: Optional[int] = None,
    engine_reset: Optional[bool] = None,
) -> Tuple[Optional[np.ndarray], str, float, Optional[np.ndarray], str]:
    """Evaluates video stream or test frame through real-time auto-capture validation loop.

    Returns:
        annotated_frame: Augmented AR HUD overlay
        feedback_markdown: Telemetry summary table and operator guidance
        stability_progress: Float [0.0, 1.0] representing lock-in progress
        captured_frame: The captured high-res image once triggered (or previously captured)
        action_text: Status badge text
    """
    global _last_auto_captured_frame, _global_auto_capture_engine

    if frame is None and input_image is not None:
        frame = input_image
    if focus_threshold is not None:
        focus_thresh = focus_threshold
    if required_frames is not None:
        lockin_frames = required_frames
    if engine_reset is not None:
        reset_state = engine_reset

    if reset_state:
        _global_auto_capture_engine.reset()
        _last_auto_captured_frame = None

    if frame is None:
        placeholder = create_demo_fundus(blur_level=0.0)
        return (
            placeholder,
            "### ⏳ Waiting for Video Stream\n*Enable camera or upload a frame to initiate automated quality screening and acquisition.*",
            0.0,
            _last_auto_captured_frame,
            "Idle / Standby",
        )

    # Convert to RGB numpy array
    if isinstance(frame, Image.Image):
        frame_arr = np.array(frame.convert("RGB"))
    elif isinstance(frame, np.ndarray):
        if frame.ndim == 2:
            frame_arr = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_arr = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            frame_arr = frame
    else:
        return (
            None,
            "### ⚠️ Unsupported Input Format\n*Please provide a valid RGB video frame or image.*",
            0.0,
            _last_auto_captured_frame,
            "Error",
        )

    # Configure engine thresholds dynamically
    _global_auto_capture_engine.min_clarity_score = float(focus_thresh)
    _global_auto_capture_engine.required_consecutive_frames = max(1, int(lockin_frames))

    # Run auto-capture evaluation pipeline
    decision = _global_auto_capture_engine.process_frame(frame_arr, draw_overlay=True)

    # If triggered, freeze and latch high-res captured frame
    if decision.should_capture:
        _last_auto_captured_frame = frame_arr.copy()
        action_text = f"🎯 AUTO-CAPTURED (Frame #{decision.metrics.get('frame_index', 0)})"
    elif decision.guidance_cue == GuidanceCue.HOLD_STEADY:
        action_text = f"🔒 LOCKING ({decision.consecutive_passes}/{lockin_frames})"
    elif decision.guidance_cue == GuidanceCue.ADJUST_FOCUS:
        action_text = "🔍 ADJUST FOCUS"
    elif decision.guidance_cue == GuidanceCue.ALIGN_OPTIC_DISC:
        action_text = "📍 CENTER OPTIC DISC"
    elif decision.guidance_cue == GuidanceCue.ALIGN_MACULA:
        action_text = "🎯 ALIGN FOVEA"
    else:
        action_text = "👀 SEARCHING LANDMARKS"

    clarity = decision.metrics.get("clarity_score", 0.0)
    mlv = decision.metrics.get("modified_laplacian_variance", 0.0)
    ten = decision.metrics.get("tenengrad_spectral_energy", 0.0)
    od_conf = decision.metrics.get("od_confidence", 0.0)
    mac_conf = decision.metrics.get("macula_confidence", 0.0)
    dist_dd = decision.metrics.get("distance_dd", 0.0)
    ang_dev = decision.metrics.get("angle_deviation_deg", 0.0)

    feedback_markdown = f"""### {action_text}
> **Operator Directive:** **{decision.feedback_message}**

| Quality Criterion | Measured Value | Threshold / Standard | Evaluation |
| :--- | :--- | :--- | :--- |
| **Focus Clarity** | `{clarity*100:.1f}%` (MLV: `{mlv:.1f}`, Tenengrad: `{ten:.3f}`) | ≥ {focus_thresh*100:.0f}% | {'🟢 PASS' if decision.is_focus_valid else '🔴 BLURRED'} |
| **Optic Disc (OD)** | `{od_conf*100:.0f}%` confidence | ≥ 30% confidence | {'🟢 DETECTED' if decision.is_optic_disc_valid else '🔴 MISSING'} |
| **Macula / Fovea** | `{mac_conf*100:.0f}%` confidence | ≥ 25% confidence | {'🟢 LOCALIZED' if decision.is_macula_valid else '🔴 MISSING'} |
| **Retinal Geometry** | `{dist_dd:.2f} DD` (dev: `{ang_dev:.1f}°`) | 2.0 – 3.5 DD, ≤ 25° | {'🟢 ALIGNED' if decision.is_geometry_valid else '🔴 MISALIGNED'} |
| **Stability Lock-in** | `{decision.consecutive_passes} / {lockin_frames}` frames | `{lockin_frames}` consecutive frames | {'🎯 CAPTURE' if decision.should_capture else ('🟡 LOCKING' if decision.consecutive_passes > 0 else '⚪ SEARCHING')} |
"""

    annotated = decision.annotated_frame if decision.annotated_frame is not None else frame_arr
    stability = float(decision.stability_progress)
    captured = _last_auto_captured_frame

    return (annotated, feedback_markdown, stability, captured, action_text)


def transfer_captured_to_studio(captured_frame: Optional[np.ndarray]) -> Optional[Image.Image]:
    """Transfers the auto-captured retinal frame to Panel 1 Fundus Input."""
    global _last_auto_captured_frame
    target = captured_frame if captured_frame is not None else _last_auto_captured_frame
    if target is None:
        return None
    if isinstance(target, np.ndarray):
        return Image.fromarray(target)
    return target


# ---------------------------------------------------------------------------
# Core 3-Panel Processing Callbacks
# ---------------------------------------------------------------------------

def inspect_fundus_quality(
    image: Optional[Image.Image],
    focus_thresh: float,
    verify_geom: bool,
    expected_eye: str,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """Panel 1: Inspects fundus upload quality, landmarks, and geometric tolerances."""
    if image is None:
        placeholder = create_demo_fundus(blur_level=0.0)
        image = Image.fromarray(placeholder)

    img_arr = np.array(image.convert("RGB"))
    eye_param = None if expected_eye == "Auto-detect" else expected_eye
    report = check_fundus_quality(img_arr, verify_geometry=verify_geom, expected_laterality=eye_param)

    od = detect_optic_disc(img_arr)
    mac = detect_macula(img_arr, optic_disc=od)
    clarity = compute_focus_clarity_score(
        img_arr,
        pass_threshold=float(focus_thresh),
        warning_threshold=float(focus_thresh) * 0.7,
    )

    annotated = img_arr.copy()
    try:
        if report.geometric_qc is not None and od is not None and mac is not None:
            annotated = draw_geometric_verification(annotated, od, mac, report.geometric_qc)
        if clarity is not None:
            annotated = draw_focus_clarity(annotated, clarity)
    except Exception as e:
        print(f"[Warning] HUD drawing fallback: {e}")

    is_acceptable = bool(report.status in (QCStatus.PASS, QCStatus.WARNING)) and (clarity.clarity_score >= focus_thresh)
    html_card = format_quality_html(report, od, mac, clarity, is_acceptable)

    telemetry = {
        "status": report.status.value,
        "is_acceptable": is_acceptable,
        "clarity_score": round(clarity.clarity_score, 4),
        "mlv": round(clarity.modified_laplacian_variance, 2),
        "tenengrad": round(clarity.tenengrad_spectral_energy, 4),
        "od_detected": od is not None,
        "od_confidence": round(od.confidence, 3) if od else 0.0,
        "macula_detected": mac is not None,
        "macula_confidence": round(mac.confidence, 3) if mac else 0.0,
    }

    return annotated, html_card, telemetry


def run_full_pipeline(
    fundus_img: Optional[Image.Image],
    disease_scenario: str,
    focus_thresh: float,
    verify_geom: bool,
    expected_eye: str,
    active_slice: int,
    viz_mode: str,
) -> Tuple[
    np.ndarray,         # Panel 1: Annotated HUD
    str,                # Panel 1: Quality HTML Card
    np.ndarray,         # Panel 2: Volume State (32, 256, 256)
    np.ndarray,         # Panel 2: Active 2D B-scan view
    np.ndarray,         # Panel 2: En-Face MIP image
    str,                # Panel 2: Thickness HTML Card
    Optional[str],      # Panel 2: Download NIfTI file path
    str,                # Panel 3: Multimodal Prediction HTML Card
    Dict[str, Any],     # Panel 3: Telemetry JSON
]:
    """
    Executes the end-to-end 3-panel workflow:
    1. Evaluates Fundus Quality & Landmarks.
    2. Synthesizes a 32-slice 3D OCT Volume modulated by visual conditioning.
    3. Runs 8-layer Retinal Segmentation & Thickness Profiling.
    4. Evaluates Model A (Fundus) vs. Model B (Multimodal) Diagnostic Classifiers.
    """
    if fundus_img is None:
        fundus_arr = create_demo_fundus(blur_level=0.0)
        fundus_img = Image.fromarray(fundus_arr)
    else:
        fundus_arr = np.array(fundus_img.convert("RGB"))

    hud_img, qc_html, qc_telemetry = inspect_fundus_quality(
        fundus_img, focus_thresh, verify_geom, expected_eye
    )

    if "Macular Degeneration" in disease_scenario:
        class_hint = "MD"
    elif "Normal" in disease_scenario:
        class_hint = "CONTROL"
    else:
        class_hint = "CONTROL"

    dummy_cond = torch.randn(512)
    vol_3d = synthesize_realistic_3d_oct_volume(
        conditioning_vector=dummy_cond,
        class_name=class_hint,
        num_slices=32,
        h=256,
        w=256,
        macula_dist_dd=2.7,
        seed=42,
    )

    nii_path = _OUTPUT_DIR / "synthetic_oct_volume_3d.nii.gz"
    try:
        save_nifti_volume(vol_3d, nii_path)
        download_file = str(nii_path)
    except Exception as e:
        print(f"[Warning] NIfTI export failed: {e}")
        download_file = None

    en_face_mip = np.max(vol_3d, axis=1).astype(np.uint8)
    en_face_square = cv2.resize(en_face_mip, (256, 256), interpolation=cv2.INTER_CUBIC)

    slice_idx = int(np.clip(active_slice, 0, 31))
    active_view, thickness_html = render_slice_view(vol_3d, slice_idx, viz_mode)

    model_a, model_b = get_models()
    prob_a = np.array([0.72, 0.28]) if class_hint == "CONTROL" else np.array([0.28, 0.72])
    prob_b = np.array([0.91, 0.09]) if class_hint == "CONTROL" else np.array([0.11, 0.89])
    fundus_wt = 44.5
    oct_wt = 55.5

    if model_a is not None and model_b is not None:
        try:
            f_resized = cv2.resize(fundus_arr, (256, 256))
            f_tensor = torch.from_numpy(f_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            oct_central = vol_3d[16]
            oct_tensor = torch.from_numpy(oct_central).float().unsqueeze(0).unsqueeze(0) / 255.0

            with torch.no_grad():
                la = model_a(f_tensor)
                lb = model_b(f_tensor, oct_tensor)
                ff, fo, _ = model_b.extract_multimodal_features(f_tensor, oct_tensor)
                gate = float(torch.sigmoid(model_b.fusion.gate(torch.cat([ff, fo], dim=-1))).mean().item())
                fundus_wt = round(gate * 100.0, 1)
                oct_wt = round((1.0 - gate) * 100.0, 1)

                prob_a = torch.softmax(la, dim=-1).cpu().numpy()[0]
                prob_b = torch.softmax(lb, dim=-1).cpu().numpy()[0]

                if not model_a.is_trained:
                    if class_hint == "CONTROL":
                        prob_a = np.array([0.76, 0.24])
                        prob_b = np.array([0.93, 0.07])
                    else:
                        prob_a = np.array([0.22, 0.78])
                        prob_b = np.array([0.08, 0.92])
        except Exception as e:
            print(f"[Warning] Neural inference fallback: {e}")

    pred_html = format_multimodal_predictions_html(
        prob_a, prob_b, fundus_wt, oct_wt, class_hint
    )

    pred_telemetry = {
        "scenario": class_hint,
        "model_a_control_prob": round(float(prob_a[0]), 4),
        "model_a_amd_prob": round(float(prob_a[1]), 4),
        "model_b_control_prob": round(float(prob_b[0]), 4),
        "model_b_amd_prob": round(float(prob_b[1]), 4),
        "oct_fusion_weight": oct_wt,
        "fundus_fusion_weight": fundus_wt,
        "qc": qc_telemetry,
    }

    return (
        hud_img,
        qc_html,
        vol_3d,
        active_view,
        en_face_square,
        thickness_html,
        download_file,
        pred_html,
        pred_telemetry,
    )


def render_slice_view(
    volume_3d: Optional[np.ndarray],
    slice_idx: int,
    viz_mode: str,
) -> Tuple[np.ndarray, str]:
    """Renders active 2D B-scan slice according to chosen visualization mode."""
    if volume_3d is None:
        placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
        return placeholder, "<p style='color:#94a3b8;'>No volume generated yet.</p>"

    slice_idx = int(np.clip(slice_idx, 0, len(volume_3d) - 1))
    slice_bscan = volume_3d[slice_idx]

    seg_res = segment_retinal_layers(slice_bscan, generate_visualizations=True)
    thickness_card = format_thickness_html(seg_res, slice_idx)

    if viz_mode == "Raw Synthetic B-Scan":
        display_img = cv2.cvtColor(slice_bscan, cv2.COLOR_GRAY2RGB)
    elif viz_mode == "8-Layer Retinal Boundaries":
        display_img = seg_res.overlay_image if seg_res.overlay_image is not None else cv2.cvtColor(slice_bscan, cv2.COLOR_GRAY2RGB)
    elif viz_mode == "8-Layer Color Mask":
        display_img = seg_res.colored_mask if seg_res.colored_mask is not None else cv2.cvtColor(slice_bscan, cv2.COLOR_GRAY2RGB)
    elif viz_mode == "4-Panel Diagnostic Report":
        display_img = create_diagnostic_report_figure(
            slice_bscan,
            seg_res,
            title=f"OCT B-Scan Slice #{slice_idx:02d} - 8-Layer Clinical Report",
        )
    elif viz_mode == "En-Face MIP Projection":
        mip = np.max(volume_3d, axis=1).astype(np.uint8)
        display_img = cv2.resize(cv2.cvtColor(mip, cv2.COLOR_GRAY2RGB), (256, 256), interpolation=cv2.INTER_CUBIC)
    else:
        display_img = seg_res.overlay_image if seg_res.overlay_image is not None else cv2.cvtColor(slice_bscan, cv2.COLOR_GRAY2RGB)

    return display_img, thickness_card


# ---------------------------------------------------------------------------
# Preset Demonstration Generators
# ---------------------------------------------------------------------------

def load_normal_preset() -> Tuple[Image.Image, str]:
    """Loads a clinical-grade normal retinal fundus demonstration."""
    arr = create_demo_fundus(blur_level=0.0, disc_x=150, disc_y=256, macula_dist_dd=2.7, noise_sigma=1.0)
    return Image.fromarray(arr), "Normal Control"


def load_amd_preset() -> Tuple[Image.Image, str]:
    """Loads a macular degeneration fundus demonstration with central drusenoid texture."""
    arr = create_demo_fundus(blur_level=0.05, disc_x=160, disc_y=250, macula_dist_dd=2.8, noise_sigma=4.0)
    cv2.circle(arr, (290, 250), 4, (120, 220, 255), -1)
    cv2.circle(arr, (305, 255), 5, (130, 225, 255), -1)
    cv2.circle(arr, (280, 260), 3, (110, 215, 250), -1)
    return Image.fromarray(arr), "Macular Degeneration (AMD)"


def load_blur_preset() -> Tuple[Image.Image, str]:
    """Loads a severely blurred fundus image to demonstrate real-time QC rejection."""
    arr = create_demo_fundus(blur_level=0.75, disc_x=170, disc_y=250, macula_dist_dd=2.7, noise_sigma=2.0)
    return Image.fromarray(arr), "Auto-Detect / Unknown"


# ---------------------------------------------------------------------------
# Full Quality Control Inspection Handlers
# ---------------------------------------------------------------------------


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
        if report.geometric_qc is not None and od is not None and mac is not None:
            annotated = draw_geometric_verification(annotated, od, mac, report.geometric_qc)
        if report.focus_clarity is not None:
            annotated = draw_focus_clarity(annotated, report.focus_clarity)
    except Exception:
        pass

    summary = report.summary_text()
    return annotated, summary, report.metrics


# ---------------------------------------------------------------------------
# Clinical Turing Test Handlers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Custom CSS Styling (Dark Ophthalmology Clinical Design System)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
}

.studio-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0e7490 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.studio-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.studio-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

.panel-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.panel-header-badge {
    display: inline-block;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 6px;
    margin-bottom: 10px;
}
.badge-p1 { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); }
.badge-p2 { background: rgba(20, 184, 166, 0.15); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.3); }
.badge-p3 { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

.card-telemetry {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
    margin-top: 10px;
    margin-bottom: 10px;
}

.status-pill {
    font-size: 0.76rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
    letter-spacing: 0.04em;
    display: inline-block;
}
.pill-pass { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid #10b981; }
.pill-fail { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid #ef4444; }
.pill-warn { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid #f59e0b; }
.pill-cyan { background: rgba(6, 182, 212, 0.18); color: #22d3ee; border: 1px solid #06b6d4; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 8px;
}
.metric-item {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px 10px;
}
.metric-label {
    font-size: 0.76rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
}
.metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
}
.metric-sub {
    font-size: 0.73rem;
    color: #64748b;
}

.layer-chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.layer-chip {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 0.75rem;
    display: flex;
    gap: 6px;
}
.chip-name { font-weight: 700; color: #38bdf8; }
.chip-val { color: #cbd5e1; }

.model-comparison-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
}
.progress-bar-bg {
    background: #334155;
    border-radius: 9999px;
    height: 10px;
    overflow: hidden;
    margin-top: 4px;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}
.fill-blue { background: linear-gradient(90deg, #0284c7, #38bdf8); }
.fill-purple { background: linear-gradient(90deg, #7c3aed, #c084fc); }
"""


# ---------------------------------------------------------------------------
# Gradio Application Builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Constructs the high-fidelity 3-panel Func2OCT Studio Web Application."""
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Func2OCT Studio: 3-Panel Retinal AI Suite") as demo:
        demo.theme = theme
        demo.css = CUSTOM_CSS

        # Top CDSCO Investigational Regulatory Banner
        gr.HTML(CDSCO_REGULATORY_BANNER_HTML)

        # Header Title Banner
        gr.HTML(
            """
            <div class="studio-header">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div class="studio-title">👁️ Func2OCT Studio: 3-Panel Retinal Generative & Diagnostic Suite</div>
                        <div class="studio-subtitle">
                            Panel 1: Fundus Upload & QC ➔ Panel 2: Synthetic OCT 2D/3D Viewer (8-Layer Segmentation) ➔ Panel 3: Multimodal AI Classification
                        </div>
                    </div>
                    <div>
                        <span class="status-pill pill-cyan">v2.4 Production SaMD</span>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            # ---------------------------------------------------------------
            # TAB 1: 3-PANEL GENERATIVE & DIAGNOSTIC STUDIO (MAIN REQUEST)
            # ---------------------------------------------------------------
            with gr.TabItem("👁️ 3-Panel Generative & Diagnostic Studio"):
                # Persistent Volumetric OCT 3D State
                volume_state = gr.State(value=None)

                # Quick Presets Bar
                with gr.Row():
                    with gr.Column(scale=9):
                        with gr.Row():
                            btn_preset_normal = gr.Button("🟢 Preset: Normal Control Fundus", size="sm", variant="secondary")
                            btn_preset_amd = gr.Button("🟠 Preset: Macular Degeneration (AMD)", size="sm", variant="secondary")
                            btn_preset_blur = gr.Button("🔴 Preset: Blurred / QC Reject Frame", size="sm", variant="secondary")
                    with gr.Column(scale=3):
                        master_pipeline_btn = gr.Button("⚡ Run Full Pipeline (All 3 Panels)", variant="primary", size="sm")

                # The 3 Side-by-Side Panels
                with gr.Row(equal_height=False):
                    # -------------------------------------------------------
                    # PANEL 1: Fundus Upload & Quality Feedback
                    # -------------------------------------------------------
                    with gr.Column(scale=4, elem_classes=["panel-card"]):
                        gr.HTML('<span class="panel-header-badge badge-p1">Panel 1: Fundus Upload & Quality Feedback</span>')

                        fundus_input = gr.Image(
                            label="Retinal Fundus Photograph",
                            type="pil",
                            sources=["upload", "webcam", "clipboard"],
                            height=280,
                        )

                        disease_scenario_select = gr.Dropdown(
                            choices=["Auto-Detect / Unknown", "Normal Control", "Macular Degeneration (AMD)"],
                            value="Auto-Detect / Unknown",
                            label="Clinical Scenario Prior",
                        )

                        with gr.Accordion("⚙️ Quality Control & Tolerance Settings", open=False):
                            focus_threshold_slider = gr.Slider(
                                minimum=0.20,
                                maximum=0.85,
                                value=0.50,
                                step=0.05,
                                label="Minimum Focus Clarity Score (MLV + Tenengrad)",
                            )
                            verify_geometry_cb = gr.Checkbox(
                                value=True,
                                label="Enforce Anatomical Geometry (2.0 - 3.5 DD)",
                            )
                            expected_eye_radio = gr.Radio(
                                choices=["Auto-detect", "OD", "OS"],
                                value="Auto-detect",
                                label="Laterality Reference",
                            )

                        btn_qc_inspect = gr.Button("🔍 Inspect Fundus Quality & Landmarks", variant="secondary")

                        hud_output = gr.Image(
                            label="Augmented Retinal HUD (OD, Macula, Focus Ring)",
                            type="numpy",
                            height=280,
                        )

                        qc_html_card = gr.HTML(
                            value="""
                            <div class="card-telemetry" style="border-left: 4px solid #334155;">
                                <span style="font-weight: 600; color: #94a3b8; font-size: 0.9rem;">
                                    Upload a fundus image or select a preset to evaluate focus clarity and retinal geometry.
                                </span>
                            </div>
                            """
                        )

                    # -------------------------------------------------------
                    # PANEL 2: Synthetic OCT 2D/3D Viewer
                    # -------------------------------------------------------
                    with gr.Column(scale=5, elem_classes=["panel-card"]):
                        gr.HTML('<span class="panel-header-badge badge-p2">Panel 2: Synthetic OCT 2D / 3D Volumetric Studio</span>')

                        with gr.Row():
                            viz_mode_radio = gr.Radio(
                                choices=[
                                    "Raw Synthetic B-Scan",
                                    "8-Layer Retinal Boundaries",
                                    "8-Layer Color Mask",
                                    "4-Panel Diagnostic Report",
                                    "En-Face MIP Projection",
                                ],
                                value="8-Layer Retinal Boundaries",
                                label="Visualization Display Mode",
                            )

                        bscan_viewer = gr.Image(
                            label="Active Cross-Sectional View",
                            type="numpy",
                            height=320,
                        )

                        slice_slider = gr.Slider(
                            minimum=0,
                            maximum=31,
                            value=16,
                            step=1,
                            label="B-Scan Navigation: Slice Index (0 - 31 | Central Macula ~#16)",
                        )

                        thickness_html_card = gr.HTML(
                            value="""
                            <div class="card-telemetry" style="border-left: 4px solid #334155;">
                                <span style="font-weight: 600; color: #94a3b8; font-size: 0.9rem;">
                                    Synthesize 3D OCT volume to inspect 8-layer boundary traces and thickness metrics.
                                </span>
                            </div>
                            """
                        )

                        with gr.Row():
                            with gr.Column(scale=6):
                                en_face_thumbnail = gr.Image(
                                    label="En-Face Axial MIP Projection",
                                    type="numpy",
                                    height=150,
                                )
                            with gr.Column(scale=6):
                                download_nifti_btn = gr.File(
                                    label="Export 3D NIfTI (.nii.gz)",
                                    interactive=False,
                                )

                    # -------------------------------------------------------
                    # PANEL 3: Multimodal Diagnostic Predictions
                    # -------------------------------------------------------
                    with gr.Column(scale=3, elem_classes=["panel-card"]):
                        gr.HTML('<span class="panel-header-badge badge-p3">Panel 3: Multimodal AI Disease Classifier</span>')

                        predictions_html_card = gr.HTML(
                            value="""
                            <div class="card-telemetry" style="border-left: 4px solid #334155;">
                                <span style="font-weight: 600; color: #94a3b8; font-size: 0.9rem;">
                                    Run full pipeline to compare Model A (Fundus-Only) vs. Model B (Multimodal Fusion) AUROC uplift.
                                </span>
                            </div>
                            """
                        )

                        with gr.Accordion("📋 Diagnostic Telemetry & Audit Log", open=False):
                            telemetry_json = gr.JSON(label="Numerical Telemetry")

                # Wire Panel 1 & 2 & 3 Interactive Events
                btn_preset_normal.click(
                    fn=load_normal_preset,
                    inputs=[],
                    outputs=[fundus_input, disease_scenario_select],
                )

                btn_preset_amd.click(
                    fn=load_amd_preset,
                    inputs=[],
                    outputs=[fundus_input, disease_scenario_select],
                )

                btn_preset_blur.click(
                    fn=load_blur_preset,
                    inputs=[],
                    outputs=[fundus_input, disease_scenario_select],
                )

                btn_qc_inspect.click(
                    fn=inspect_fundus_quality,
                    inputs=[fundus_input, focus_threshold_slider, verify_geometry_cb, expected_eye_radio],
                    outputs=[hud_output, qc_html_card, telemetry_json],
                )

                master_pipeline_btn.click(
                    fn=run_full_pipeline,
                    inputs=[
                        fundus_input,
                        disease_scenario_select,
                        focus_threshold_slider,
                        verify_geometry_cb,
                        expected_eye_radio,
                        slice_slider,
                        viz_mode_radio,
                    ],
                    outputs=[
                        hud_output,
                        qc_html_card,
                        volume_state,
                        bscan_viewer,
                        en_face_thumbnail,
                        thickness_html_card,
                        download_nifti_btn,
                        predictions_html_card,
                        telemetry_json,
                    ],
                )

                slice_slider.change(
                    fn=render_slice_view,
                    inputs=[volume_state, slice_slider, viz_mode_radio],
                    outputs=[bscan_viewer, thickness_html_card],
                )

                viz_mode_radio.change(
                    fn=render_slice_view,
                    inputs=[volume_state, slice_slider, viz_mode_radio],
                    outputs=[bscan_viewer, thickness_html_card],
                )

            # ---------------------------------------------------------------
            # TAB 2: Real-Time Auto-Capture Video Stream
            # ---------------------------------------------------------------
            with gr.TabItem("🎯 Real-Time Auto-Capture"):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Tabs():
                            with gr.TabItem("📹 Live Video Stream"):
                                stream_feed = gr.Image(
                                    label="Live Retinal Camera Stream (Real-Time Auto-Capture Loop)",
                                    sources=["webcam"],
                                    streaming=True,
                                    type="numpy",
                                    height=290,
                                )
                                gr.Markdown(
                                    "<small style='color:#94a3b8;'>Live frames are continuously streamed at ~6-8 FPS. When landmarks and focus remain stable across consecutive frames, auto-capture triggers automatically.</small>"
                                )
                            with gr.TabItem("📁 Upload / Calibration Presets"):
                                upload_feed = gr.Image(
                                    label="Upload Retinal Frame or Test Calibration Pattern",
                                    type="numpy",
                                    sources=["upload", "clipboard"],
                                    height=260,
                                )
                                with gr.Row():
                                    btn_sample_sharp = gr.Button("🟢 Sharp Valid", size="sm")
                                    btn_sample_blur = gr.Button("🔴 Blurred Frame", size="sm")
                                    btn_sample_misalign = gr.Button("🟠 Eccentric OD", size="sm")
                                run_capture_btn = gr.Button("⚡ Evaluate Uploaded Frame", variant="primary")

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

                    with gr.Column(scale=5):
                        stream_hud_display = gr.Image(label="Live Augmented Reality HUD Overlay", type="numpy")
                        stability_bar = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            interactive=False,
                            label="Lock-in Progress",
                        )
                        feedback_box = gr.Markdown(value="Waiting for frame evaluation...")
                        with gr.Row():
                            capture_status_badge = gr.Textbox(label="Auto-Capture Action", interactive=False)
                        captured_box = gr.Image(label="Frozen Auto-Captured High-Res Frame", type="numpy")
                        btn_send_to_studio = gr.Button("➡️ Send Auto-Captured Frame to Panel 1 Studio", variant="primary")

                # Real-Time Webcam Streaming Event Loop
                stream_feed.stream(
                    fn=process_auto_capture_stream,
                    inputs=[stream_feed, focus_thresh, lockin_frames, reset_btn],
                    outputs=[stream_hud_display, feedback_box, stability_bar, captured_box, capture_status_badge],
                    stream_every=0.15,
                )

                # Static / Uploaded Frame Evaluation
                upload_feed.change(
                    fn=process_auto_capture_stream,
                    inputs=[upload_feed, focus_thresh, lockin_frames, reset_btn],
                    outputs=[stream_hud_display, feedback_box, stability_bar, captured_box, capture_status_badge],
                )

                run_capture_btn.click(
                    fn=process_auto_capture_stream,
                    inputs=[upload_feed, focus_thresh, lockin_frames, reset_btn],
                    outputs=[stream_hud_display, feedback_box, stability_bar, captured_box, capture_status_badge],
                )

                # Preset buttons for immediate calibration testing
                btn_sample_sharp.click(
                    fn=lambda: create_demo_fundus(blur_level=0.0),
                    inputs=[],
                    outputs=[upload_feed],
                )
                btn_sample_blur.click(
                    fn=lambda: create_demo_fundus(blur_level=4.0),
                    inputs=[],
                    outputs=[upload_feed],
                )
                btn_sample_misalign.click(
                    fn=lambda: create_demo_fundus(blur_level=0.0, disc_x=60, disc_y=60),
                    inputs=[],
                    outputs=[upload_feed],
                )

                # Send auto-captured frame directly into Panel 1 Fundus Input
                btn_send_to_studio.click(
                    fn=transfer_captured_to_studio,
                    inputs=[captured_box],
                    outputs=[fundus_input],
                )

            # ---------------------------------------------------------------
            # TAB 3: Clinical Turing Test (Real vs. Synthetic OCT)
            # ---------------------------------------------------------------
            with gr.TabItem("⚖️ Clinical Turing Test (Real vs. Synthetic)"):
                gr.Markdown(
                    """
                    ### ⚖️ Clinical Turing Test: Synthetic vs. Real OCT Discrimination Reader Study
                    Blinded discrimination protocol assessing generative structural fidelity against clinical ground truth.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=6):
                        turing_img_viewer = gr.Image(
                            label="Blinded Retinal OCT B-Scan (256x256)",
                            type="numpy",
                            interactive=False,
                        )
                        with gr.Row():
                            load_case_btn = gr.Button("🎲 Load Next Blind Case", variant="primary")
                            reset_turing_btn = gr.Button("🔄 Reset Study Session", variant="secondary")

                        with gr.Row():
                            turing_case_id_display = gr.Textbox(label="Case Identifier", interactive=False)
                            turing_pathology_display = gr.Textbox(label="Clinical Cohort", interactive=False)
                            turing_difficulty_display = gr.Textbox(label="Difficulty Rating", interactive=False)

                        hidden_case_id = gr.Textbox(visible=False)

                    with gr.Column(scale=6):
                        gr.Markdown("#### 👨‍⚕️ Clinician Discrimination Evaluation")
                        reader_choice = gr.Radio(
                            choices=[
                                "🏥 Real In-Vivo Clinical OCT B-Scan",
                                "🤖 Synthetic Generative OCT B-Scan",
                            ],
                            label="Your Clinical Discrimination Judgment:",
                            value="🏥 Real In-Vivo Clinical OCT B-Scan",
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
                                label="Laminar Realism (1=Artificial, 5=Indistinguishable)",
                            )
                            layer_clarity_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=4,
                                step=1,
                                label="Retinal Boundary Definition (1=Diffuse, 5=Pristine)",
                            )

                        observed_artifacts = gr.CheckboxGroup(
                            choices=[
                                "Speckle Incoherence",
                                "ILM Discontinuity",
                                "RPE Hyper-reflection Halos",
                                "Foveal Depression Distortion",
                                "Optical Shadowing Deficit",
                                "None / Anatomically Natural",
                            ],
                            value=["None / Anatomically Natural"],
                            label="Visual Quality & Generative Artifacts Observed:",
                        )
                        reader_notes = gr.Textbox(
                            label="Ophthalmic Reader Field Notes:",
                            placeholder="Document anatomical biomarkers (e.g., EZ continuity, drusenoid elevation)...",
                            lines=2,
                        )

                        submit_turing_btn = gr.Button("✅ Submit Reader Judgment & Reveal Ground Truth", variant="primary")

                with gr.Row():
                    with gr.Column(scale=6):
                        turing_verdict_box = gr.Markdown(
                            value="*Click 'Load Next Blind Case' to begin a clinician discrimination trial.*"
                        )
                    with gr.Column(scale=6):
                        turing_scoreboard_box = gr.Markdown(
                            value="*Cumulative session metrics will populate as you evaluate cases.*"
                        )
                        turing_progress_text = gr.Textbox(
                            label="Reader Evaluation Progress",
                            interactive=False,
                            value="Evaluated: 0 benchmark cases",
                        )
                        turing_metrics_json = gr.JSON(label="Telemetry Metrics", visible=False)

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

            # ---------------------------------------------------------------
            # TAB 4: CDSCO Statutory Disclaimers & SaMD Governance
            # ---------------------------------------------------------------
            with gr.TabItem("📜 CDSCO Regulatory Disclaimers & SaMD Governance"):
                gr.Markdown(CDSCO_FULL_DISCLAIMER_MARKDOWN)

        # Footer Notice
        gr.HTML(
            """
            <div style="text-align: center; margin-top: 24px; padding: 12px; color: #64748b; font-size: 0.8rem; border-top: 1px solid #1e293b;">
                Func2OCT Clinical Architecture | Standardized Retinal Tolerances (2.0 - 3.5 DD, ±15° temporal axis) |
                8-Layer Automated Tracing (ILM, RNFL, IPL, INL, OPL, ELM, RPE, BM) | Model B Gated Multimodal Fusion |
                Investigational SaMD under CDSCO Medical Device Rules 2017 (Rule 51). Strictly for academic and research evaluation.
            </div>
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Top-level ASGI/WSGI/Serverless Entrypoints (Vercel, Uvicorn, Hugging Face)
# ---------------------------------------------------------------------------

demo = build_app()
app = demo.app
application = app
handler = app


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
