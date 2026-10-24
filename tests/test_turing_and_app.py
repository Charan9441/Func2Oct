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

