"""
test_turing_and_app.py

Unit tests for Clinical Turing Test Module & CDSCO Disclaimers in app.py.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from outputs.demo_assets.clinical_turing_test import (
    ClinicalTuringTestSession,
    CDSCO_TURING_DISCLAIMER,
    TuringCase,
)
import app


def test_cdsco_disclaimers_present_in_app():
    """Verifies that CDSCO investigational medical device disclaimers are present in app.py."""
    banner_upper = app.CDSCO_REGULATORY_BANNER_HTML.upper()
    assert "CDSCO INVESTIGATIONAL MEDICAL DEVICE NOTICE" in banner_upper
    assert "RULE 51" in banner_upper
    assert "MEDICAL DEVICE RULES" in banner_upper
    assert "2017" in banner_upper
    assert "INVESTIGATIONAL SAMD" in banner_upper
    assert "Central Drugs Standard Control Organization" in app.CDSCO_FULL_DISCLAIMER_MARKDOWN
    assert "Rule 51" in app.CDSCO_FULL_DISCLAIMER_MARKDOWN
    assert "Institutional Ethics Committee" in app.CDSCO_FULL_DISCLAIMER_MARKDOWN


def test_clinical_turing_test_session_initialization():
    """Verifies that the Turing test session initializes and loads benchmark cases."""
    session = ClinicalTuringTestSession()
    assert len(session.cases) >= 6
    assert any(c.ground_truth == "Real OCT" for c in session.cases.values())
    assert any(c.ground_truth == "Synthetic OCT" for c in session.cases.values())


def test_blind_case_retrieval_and_evaluation_flow():
    """Verifies blinded case retrieval, discrimination evaluation, and metric calculation."""
    session = ClinicalTuringTestSession()
    session.reset_session()

    # 1. Blind case retrieval
    blind_case = session.get_blind_case()
    assert blind_case is not None
    assert "case_id" in blind_case
    assert "ground_truth" not in blind_case  # Must be strictly blinded!
    assert Path(blind_case["image_path"]).exists()

    # 2. Reader submits diagnosis (testing discrimination)
    case_id = blind_case["case_id"]
    res = session.submit_evaluation(
        case_id=case_id,
        reader_choice="Real OCT",
        confidence_pct=85.0,
        realism_score=4,
        layer_clarity_score=4,
        observed_artifacts=["None / Anatomically Natural"],
        reader_notes="Distinct external limiting membrane and intact RPE band.",
    )

    assert "verdict_badge" in res
    assert "ground_truth" in res
    assert res["ground_truth"] in ("Real OCT", "Synthetic OCT")
    assert isinstance(res["is_correct"], bool)
    assert isinstance(res["is_fooled"], bool)

    # 3. Verify session summary metrics
    metrics = session.get_summary_metrics()
    assert metrics["total_evaluated"] >= 1
    assert "reader_accuracy_pct" in metrics
    assert "turing_fooling_rate_pct" in metrics
    assert "confusion_matrix" in metrics


def test_app_turing_handlers():
    """Tests the Gradio callback handlers in app.py."""
    img, case_id, cat, diff, hid_id, prompt, progress = app.load_turing_case()
    assert img is not None
    assert case_id != "No Cases"
    assert hid_id == case_id

    verdict_md, scoreboard_md, prog_out, metrics = app.submit_turing_evaluation(
        case_id=case_id,
        reader_choice="🤖 Synthetic Generative OCT B-Scan",
        confidence_pct=75.0,
        realism_score=3,
        layer_clarity_score=4,
        observed_artifacts=["Speckle Incoherence"],
        reader_notes="Automated test submission",
    )
    assert "Ground Truth" in verdict_md
    assert "Cumulative Clinical Turing Test Telemetry" in scoreboard_md
    assert metrics["total_evaluated"] >= 1


def test_auto_capture_stream_none_input():
    """Verifies safe handling of None frame input in streaming loop."""
    hud, fb, stab, cap, status = app.process_auto_capture_stream(
        frame=None,
        focus_thresh=0.50,
        lockin_frames=3,
        reset_state=True,
    )
    assert hud is not None
    assert "Waiting for Video Stream" in fb
    assert stab == 0.0
    assert status == "Idle / Standby"


def test_auto_capture_stream_consecutive_trigger_and_transfer():
    """Verifies that 3 consecutive valid frames trigger auto-capture and allow transfer to studio."""
    sharp_frame = app.create_demo_fundus(blur_level=0.0)

    # Reset state
    app.process_auto_capture_stream(sharp_frame, focus_thresh=0.30, lockin_frames=3, reset_state=True)

    # Frame 1
    hud1, fb1, stab1, cap1, status1 = app.process_auto_capture_stream(
        sharp_frame, focus_thresh=0.30, lockin_frames=3, reset_state=False
    )
    assert stab1 >= 0.33
    assert "Focus Clarity" in fb1

    # Frame 2
    hud2, fb2, stab2, cap2, status2 = app.process_auto_capture_stream(
        sharp_frame, focus_thresh=0.30, lockin_frames=3, reset_state=False
    )
    assert stab2 >= 0.66

    # Frame 3 -> Should trigger capture
    hud3, fb3, stab3, cap3, status3 = app.process_auto_capture_stream(
        sharp_frame, focus_thresh=0.30, lockin_frames=3, reset_state=False
    )
    assert stab3 >= 1.0
    assert cap3 is not None
    assert "AUTO-CAPTURED" in status3

    # Transfer captured frame to Panel 1 Studio
    transferred = app.transfer_captured_to_studio(cap3)
    assert transferred is not None
    assert transferred.size == (512, 512)


def test_gtts_speech_synthesis():
    """Verifies that synthesize_speech generates a valid, non-empty MP3 audio file."""
    audio_path = app.synthesize_speech("నమస్కారం పరీక్ష", lang="te", filename_prefix="unit_test")
    assert audio_path is not None
    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 100


def test_telugu_voice_cues():
    """Verifies retrieval and synthesis of auto-capture guidance cues in localized Telugu."""
    text, audio_path = app.get_telugu_voice_cue("HOLD_STEADY")
    assert "స్థిరంగా" in text
    assert audio_path is not None
    assert os.path.exists(audio_path)

    text_cap, audio_cap = app.get_telugu_voice_cue("CAPTURE")
    assert "ఆటో-క్యాప్చర్" in text_cap
    assert audio_cap is not None
    assert os.path.exists(audio_cap)


def test_telugu_qc_patient_report():
    """Verifies localized Telugu audio report generation from QC inspection."""
    fundus = app.create_demo_fundus(blur_level=0.0)
    report = app.check_fundus_quality(fundus, verify_geometry=True, expected_laterality="OD")

    audio_path, te_card, en_card = app.format_telugu_qc_patient_report(
        report=report,
        patient_id="PID-TEST-99",
        expected_eye="OD",
        language="Telugu (తెలుగు)",
    )
    assert audio_path is not None
    assert os.path.exists(audio_path)
    assert "PID-TEST-99" in te_card
    assert "ఆప్టిక్ డిస్క్" in te_card
    assert "PID-TEST-99" in en_card


def test_accessible_audio_report_studio():
    """Verifies dedicated accessibility studio audio report and transcript generation."""
    audio_path, summary_md = app.generate_accessible_audio_report(
        patient_name="శ్రీనివాస రావు",
        eye_choice="కుడి కన్ను (Right Eye - OD)",
        condition_preset="ఉత్తమ నాణ్యత - సాధారణ రెటీనా (Normal Healthy Retina - Pass)",
        custom_findings="స్పష్టమైన నాళాల అమరిక.",
        language_choice="Telugu (తెలుగు)",
        speech_speed="సాధారణం (Normal)",
    )
    assert audio_path is not None
    assert os.path.exists(audio_path)
    assert "శ్రీనివాస రావు" in summary_md
    assert "ఆడియో సంభాషణ" in summary_md


def test_patient_diagnostic_audio_report():
    """Verifies multimodal diagnostic report voice synthesis."""
    audio_path, te_card, en_card = app.generate_patient_diagnostic_audio_report(
        disease_scenario="Macular Degeneration (AMD)",
        qc_telemetry={"clarity_score": 0.88},
        prob_b=np.array([0.15, 0.85]),
        patient_id="PID-AMD-01",
        expected_eye="OD",
        language="Telugu (తెలుగు)",
    )
    assert audio_path is not None
    assert os.path.exists(audio_path)
    assert "మాక్యులర్ డీజెనరేషన్" in te_card or "AMD" in te_card
    assert "PID-AMD-01" in te_card


def test_turing_telugu_narration():
    """Verifies Clinical Turing test ground truth reveal narration in Telugu."""
    session = ClinicalTuringTestSession()
    blind_case = session.get_blind_case()
    case_id = blind_case["case_id"]

    audio_path, transcript = app.narrate_turing_evaluation_telugu(
        case_id=case_id,
        reader_choice="🏥 Real In-Vivo Clinical OCT B-Scan",
    )
    assert audio_path is not None
    assert os.path.exists(audio_path)
    assert case_id in transcript
    assert "క్లినికల్ ట్యూరింగ్" in transcript


def test_generative_pipeline_integration():
    """Verifies end-to-end integration of EyeFound + LVDM + VAE in pipeline.py and app.py."""
    pipe = app.get_generative_pipeline()
    assert pipe is not None
    assert hasattr(pipe, "synthesize_3d_volume")

    # Verify synthesize_3d_volume produces standardized (32, 256, 256) uint8 volume
    demo_fundus = app.create_demo_fundus(blur_level=0.0)
    vol_3d = pipe.synthesize_3d_volume(
        fundus=demo_fundus,
        class_hint="DR",
        num_slices=32,
        seed=42,
    )
    assert isinstance(vol_3d, np.ndarray)
    assert vol_3d.shape == (32, 256, 256)
    assert vol_3d.dtype == np.uint8


def test_comparative_classification_and_auroc_ci():
    """Verifies 2D baseline and 3D multimodal classification connection, AUROC, and CIs."""
    # 1. Verify Wilson score CI computation
    low, high = app.compute_wilson_ci(0.85, n=100)
    assert 70.0 < low < 85.0
    assert 85.0 < high < 95.0

    # 2. Verify model loader retrieves both models
    model_a, model_b = app.get_models()
    assert model_a is not None
    assert model_b is not None

    # 3. Verify format_multimodal_predictions_html contains AUROC, CIs, and likelihoods
    prob_a = np.array([0.25, 0.75])
    prob_b = np.array([0.08, 0.92])
    html_card = app.format_multimodal_predictions_html(
        prob_a=prob_a,
        prob_b=prob_b,
        fundus_weight=42.0,
        oct_weight=58.0,
        disease_prior="MD",
        auroc_a=0.846,
        auroc_b=0.942,
    )
    assert "Model A: 2D Fundus Baseline" in html_card
    assert "Model B: 3D Multimodal" in html_card
    assert "AUROC: 0.846" in html_card
    assert "AUROC: 0.942" in html_card
    assert "95% CI" in html_card
    assert "Multimodal AUROC Uplift" in html_card

    # 4. Verify run_full_pipeline telemetry output format
    fundus_demo = app.create_demo_fundus(blur_level=0.0)
    res = app.run_full_pipeline(
        fundus_img=app.Image.fromarray(fundus_demo),
        disease_scenario="Normal Control",
        focus_thresh=0.50,
        verify_geom=True,
        expected_eye="OD",
        active_slice=16,
        viz_mode="Raw Synthetic B-Scan",
    )
    telemetry = res[-1]
    assert "model_a_2d_baseline" in telemetry
    assert "model_b_3d_multimodal" in telemetry
    assert "auroc_uplift" in telemetry
    assert telemetry["model_a_2d_baseline"]["auroc"] == 0.846
    assert telemetry["model_b_3d_multimodal"]["auroc"] == 0.942
