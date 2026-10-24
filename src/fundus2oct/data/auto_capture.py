"""
src/fundus2oct/data/auto_capture.py

Real-time Auto-Capture Feedback Loop for Retinal Fundus Photography.
Evaluates live camera video frames to trigger image capture only when:
1. Optic disc is detected with sufficient confidence and within frame bounds.
2. Macula / foveal center is detected within anatomical expectations.
3. Retinal geometry (distance in DD and angle) conforms to clinical standards.
4. Comprehensive focus clarity score (Modified Laplacian Variance + Frequency Tenengrad) meets threshold.
5. Consecutive multi-frame stability criterion is met to prevent motion/blink capture.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

from fundus2oct.data.quality_control import (
    QCStatus,
    OpticDiscResult,
    MaculaResult,
    GeometricVerificationResult,
    FocusClarityResult,
    detect_optic_disc,
    detect_macula,
    verify_optic_disc_macula_geometry,
    compute_focus_clarity_score,
    _ensure_uint8_image,
)


class GuidanceCue(str, Enum):
    """Clinical operator guidance cues during live acquisition."""
    SEARCHING = "SEARCHING_LANDMARKS"
    ALIGN_OPTIC_DISC = "ALIGN_OPTIC_DISC"
    ALIGN_MACULA = "ALIGN_MACULA"
    ADJUST_FOCUS = "ADJUST_FOCUS"
    HOLD_STEADY = "HOLD_STEADY"
    CAPTURE = "CAPTURE_TRIGGERED"


@dataclass
class AutoCaptureDecision:
    """Decision and state metrics from the real-time auto-capture feedback loop.

    Attributes:
        should_capture: True if all criteria are satisfied across stability window.
        guidance_cue: GuidanceCue enumeration for operator action.
        feedback_message: Human-readable instruction for live HUD/operator.
        consecutive_passes: Number of consecutive frames meeting quality standards.
        stability_progress: Normalized stability lock-in progress [0.0, 1.0].
        is_optic_disc_valid: Whether optic disc localization passed criteria.
        is_macula_valid: Whether macula localization passed criteria.
        is_geometry_valid: Whether anatomical distance & angle passed criteria.
        is_focus_valid: Whether focus clarity score passed threshold.
        optic_disc: Optional detected OpticDiscResult.
        macula: Optional detected MaculaResult.
        geometric_qc: Optional GeometricVerificationResult.
        focus_clarity: Optional FocusClarityResult.
        annotated_frame: Frame with augmented HUD guidance overlay.
        metrics: Dictionary of numerical scores for telemetry/logging.
    """
    should_capture: bool
    guidance_cue: GuidanceCue
    feedback_message: str
    consecutive_passes: int
    stability_progress: float
    is_optic_disc_valid: bool
    is_macula_valid: bool
    is_geometry_valid: bool
    is_focus_valid: bool
    optic_disc: Optional[OpticDiscResult] = None
    macula: Optional[MaculaResult] = None
    geometric_qc: Optional[GeometricVerificationResult] = None
    focus_clarity: Optional[FocusClarityResult] = None
    annotated_frame: Optional[np.ndarray] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class AutoCaptureEngine:
    """Real-time Auto-Capture feedback loop engine.

    Maintains temporal stability history across consecutive frames to trigger
    acquisition without motion blur or transient alignment dropouts.
    """

    def __init__(
        self,
        min_clarity_score: float = 0.50,
        min_od_confidence: float = 0.30,
        min_macula_confidence: float = 0.25,
        required_consecutive_frames: int = 3,
        distance_dd_range: Tuple[float, float] = (1.8, 4.0),
        max_angle_deviation_deg: float = 25.0,
        expected_laterality: Optional[str] = None,
        draw_hud_overlay: bool = True,
    ):
        """Initialize the auto-capture feedback loop parameters."""
        self.min_clarity_score = min_clarity_score
        self.min_od_confidence = min_od_confidence
        self.min_macula_confidence = min_macula_confidence
        self.required_consecutive_frames = max(1, required_consecutive_frames)
        self.distance_dd_range = distance_dd_range
        self.max_angle_deviation_deg = max_angle_deviation_deg
        self.expected_laterality = expected_laterality
        self.draw_hud_overlay = draw_hud_overlay

        # Temporal state tracking
        self.consecutive_passes = 0
        self.total_frames_processed = 0
        self.last_capture_frame_idx: Optional[int] = None

    def reset(self) -> None:
        """Reset temporal state buffer."""
        self.consecutive_passes = 0
        self.total_frames_processed = 0
        self.last_capture_frame_idx = None

    def process_frame(
        self,
        frame: Union[np.ndarray, Image.Image],
        draw_overlay: Optional[bool] = None,
    ) -> AutoCaptureDecision:
        """Evaluate a single video frame for auto-capture eligibility.

        Runs optic disc detection, macula localization, geometric distance and
        angle verification, and Modified Laplacian / Tenengrad focus scoring.
        """
        uint8_frame = _ensure_uint8_image(frame)
        h, w = uint8_frame.shape[:2]
        self.total_frames_processed += 1

        should_draw = self.draw_hud_overlay if draw_overlay is None else draw_overlay

        # 1. Focus Clarity Check (MLV + Tenengrad Frequency Analysis)
        focus_res = compute_focus_clarity_score(
            uint8_frame,
            pass_threshold=self.min_clarity_score,
            warning_threshold=self.min_clarity_score * 0.7,
        )
        is_focus_valid = bool(focus_res.clarity_score >= self.min_clarity_score)

        # 2. Optic Disc Detection
        try:
            od_res = detect_optic_disc(uint8_frame, method="hybrid")
            is_od_valid = bool(od_res.confidence >= self.min_od_confidence)
            # Verify OD is not severely clipped on outer border
            cx, cy = od_res.center
            r = od_res.radius
            margin = max(5, int(min(h, w) * 0.04))
            if (cx - r < margin) or (cx + r > w - margin) or (cy - r < margin) or (cy + r > h - margin):
                is_od_valid = False
        except Exception:
            od_res = None
            is_od_valid = False

        # 3. Macula Detection
        mac_res = None
        is_macula_valid = False
        if is_od_valid and od_res is not None:
            try:
                mac_res = detect_macula(uint8_frame, optic_disc=od_res)
                is_macula_valid = bool(mac_res.confidence >= self.min_macula_confidence)
                # Verify macula is within field of view
                mx, my = mac_res.center
                if mx < margin or mx > w - margin or my < margin or my > h - margin:
                    is_macula_valid = False
            except Exception:
                mac_res = None
                is_macula_valid = False

        # 4. Geometric Anatomical Verification (Distance in DD and Angle)
        geom_res = None
        is_geometry_valid = False
        if is_od_valid and is_macula_valid and od_res is not None and mac_res is not None:
            try:
                geom_res = verify_optic_disc_macula_geometry(
                    od_res,
                    mac_res,
                    distance_dd_fail_min=self.distance_dd_range[0],
                    distance_dd_fail_max=self.distance_dd_range[1],
                    fail_angle_deviation_deg=self.max_angle_deviation_deg,
                    expected_laterality=self.expected_laterality,
                )
                is_geometry_valid = bool(geom_res.is_valid and geom_res.status != QCStatus.FAIL)
            except Exception:
                geom_res = None
                is_geometry_valid = False

        # 5. Composite Evaluation and Guidance Decision
        all_passed = is_od_valid and is_macula_valid and is_geometry_valid and is_focus_valid

        if all_passed:
            self.consecutive_passes += 1
            stability_progress = min(1.0, float(self.consecutive_passes / self.required_consecutive_frames))

            if self.consecutive_passes >= self.required_consecutive_frames:
                should_capture = True
                cue = GuidanceCue.CAPTURE
                msg = f"AUTO-CAPTURE TRIGGERED ({self.consecutive_passes}/{self.required_consecutive_frames} stable frames)"
                self.last_capture_frame_idx = self.total_frames_processed
            else:
                should_capture = False
                cue = GuidanceCue.HOLD_STEADY
                msg = f"HOLD STEADY - Locking Focus ({self.consecutive_passes}/{self.required_consecutive_frames})"
        else:
            self.consecutive_passes = max(0, self.consecutive_passes - 1)
            stability_progress = float(self.consecutive_passes / self.required_consecutive_frames)
            should_capture = False

            if not is_od_valid:
                cue = GuidanceCue.ALIGN_OPTIC_DISC
                msg = "Center the optic disc in camera aperture"
            elif not is_macula_valid:
                cue = GuidanceCue.ALIGN_MACULA
                msg = "Bring macula and foveal center into field of view"
            elif not is_geometry_valid:
                cue = GuidanceCue.SEARCHING
                msg = "Anatomical landmark distance/angle outside clinical tolerances"
            elif not is_focus_valid:
                cue = GuidanceCue.ADJUST_FOCUS
                msg = f"Adjust focus ring (Clarity: {focus_res.clarity_score * 100:.0f}% < {self.min_clarity_score * 100:.0f}%)"
            else:
                cue = GuidanceCue.SEARCHING
                msg = "Searching for retinal landmarks"

        metrics = {
            "frame_index": self.total_frames_processed,
            "clarity_score": focus_res.clarity_score,
            "modified_laplacian_variance": focus_res.modified_laplacian_variance,
            "tenengrad_spectral_energy": focus_res.tenengrad_spectral_energy,
            "od_confidence": od_res.confidence if od_res else 0.0,
            "macula_confidence": mac_res.confidence if mac_res else 0.0,
            "distance_dd": geom_res.distance_dd if geom_res else 0.0,
            "angle_deviation_deg": geom_res.angle_deviation_deg if geom_res else 0.0,
            "consecutive_passes": self.consecutive_passes,
            "should_capture": should_capture,
        }

        # 6. Render HUD Overlay if requested
        annotated = None
        if should_draw:
            annotated = self._render_hud_overlay(
                uint8_frame,
                should_capture=should_capture,
                guidance_cue=cue,
                feedback_message=msg,
                consecutive_passes=self.consecutive_passes,
                stability_progress=stability_progress,
                is_focus_valid=is_focus_valid,
                od_res=od_res,
                mac_res=mac_res,
                geom_res=geom_res,
                focus_res=focus_res,
            )

        return AutoCaptureDecision(
            should_capture=should_capture,
            guidance_cue=cue,
            feedback_message=msg,
            consecutive_passes=self.consecutive_passes,
            stability_progress=stability_progress,
            is_optic_disc_valid=is_od_valid,
            is_macula_valid=is_macula_valid,
            is_geometry_valid=is_geometry_valid,
            is_focus_valid=is_focus_valid,
            optic_disc=od_res,
            macula=mac_res,
            geometric_qc=geom_res,
            focus_clarity=focus_res,
            annotated_frame=annotated,
            metrics=metrics,
        )

    def _render_hud_overlay(
        self,
        frame: np.ndarray,
        should_capture: bool,
        guidance_cue: GuidanceCue,
        feedback_message: str,
        consecutive_passes: int,
        stability_progress: float,
        is_focus_valid: bool,
        od_res: Optional[OpticDiscResult],
        mac_res: Optional[MaculaResult],
        geom_res: Optional[GeometricVerificationResult],
        focus_res: FocusClarityResult,
    ) -> np.ndarray:
        """Render high-contrast, clinical augmented reality HUD overlay."""
        canvas = frame.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        elif canvas.shape[2] == 1:
            canvas = cv2.cvtColor(canvas[:, :, 0], cv2.COLOR_GRAY2BGR)

        h, w = canvas.shape[:2]

        # 1. Draw Optic Disc reticle if available
        if od_res is not None:
            cx, cy = od_res.center
            r = od_res.radius
            od_color = (0, 255, 0) if od_res.confidence >= self.min_od_confidence else (0, 165, 255)
            cv2.circle(canvas, (cx, cy), r, od_color, 2)
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                f"OD ({od_res.confidence:.2f})",
                (cx - r, cy - r - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                od_color,
                1,
                lineType=cv2.LINE_AA,
            )

        # 2. Draw Macula reticle and crosshair if available
        if mac_res is not None:
            mx, my = mac_res.center
            mr = mac_res.radius
            mac_color = (255, 200, 0) if mac_res.confidence >= self.min_macula_confidence else (0, 140, 255)
            cv2.circle(canvas, (mx, my), mr, mac_color, 2)
            # Foveal targeting crosshair
            ch_len = max(6, int(mr * 0.4))
            cv2.line(canvas, (mx - ch_len, my), (mx + ch_len, my), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (mx, my - ch_len), (mx, my + ch_len), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "FOVEA",
                (mx - mr, my + mr + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                mac_color,
                1,
                lineType=cv2.LINE_AA,
            )

        # 3. Draw Connecting Vector Line & Anatomical Distance
        if od_res is not None and mac_res is not None:
            cv2.line(canvas, od_res.center, mac_res.center, (0, 255, 255), 1, cv2.LINE_AA)
            if geom_res is not None:
                mid_x = (od_res.center[0] + mac_res.center[0]) // 2
                mid_y = (od_res.center[1] + mac_res.center[1]) // 2 - 8
                geom_color = (0, 255, 0) if geom_res.is_valid else (0, 0, 255)
                cv2.putText(
                    canvas,
                    f"{geom_res.distance_dd:.2f} DD ({geom_res.inferred_laterality})",
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    geom_color,
                    1,
                    lineType=cv2.LINE_AA,
                )

        # 4. Top HUD Header Banner
        header_h = 58
        cv2.rectangle(canvas, (0, 0), (w, header_h), (25, 25, 25), -1)

        cue_colors = {
            GuidanceCue.CAPTURE: (0, 230, 0),       # Green
            GuidanceCue.HOLD_STEADY: (0, 255, 255),  # Yellow
            GuidanceCue.ADJUST_FOCUS: (0, 165, 255), # Orange
            GuidanceCue.ALIGN_OPTIC_DISC: (0, 140, 255),
            GuidanceCue.ALIGN_MACULA: (0, 140, 255),
            GuidanceCue.SEARCHING: (180, 180, 180),
        }
        cue_color = cue_colors.get(guidance_cue, (255, 255, 255))

        # Guidance text
        cv2.putText(
            canvas,
            f"GUIDANCE: {feedback_message}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            cue_color,
            2,
            lineType=cv2.LINE_AA,
        )

        # Status meters (Clarity & Lock-in Progress)
        clarity_pct = int(round(focus_res.clarity_score * 100))
        clarity_color = (0, 255, 0) if is_focus_valid else (0, 0, 255)
        meter_text = (
            f"Clarity: {clarity_pct}% (MLV: {focus_res.modified_laplacian_variance:.0f}) | "
            f"Lock-in: {consecutive_passes}/{self.required_consecutive_frames}"
        )
        cv2.putText(
            canvas,
            meter_text,
            (15, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            lineType=cv2.LINE_AA,
        )

        # Stability Progress Bar (Right side of header)
        bar_w = 140
        bar_h = 14
        bar_x = w - bar_w - 20
        bar_y = 18
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(round(bar_w * stability_progress))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), cue_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        # 5. Capture Flash Frame if capture was just triggered
        if should_capture:
            # Draw green bounding border around entire frame
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)
            cv2.putText(
                canvas,
                "*** CAPTURED ***",
                (w // 2 - 110, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

        return canvas


def evaluate_auto_capture_frame(
    frame: Union[np.ndarray, Image.Image],
    min_clarity_score: float = 0.50,
    required_consecutive_frames: int = 1,
) -> AutoCaptureDecision:
    """Convenience function to evaluate a single frame for auto-capture quality.

    Useful for stateless image assessment and batch screening.
    """
    engine = AutoCaptureEngine(
        min_clarity_score=min_clarity_score,
        required_consecutive_frames=required_consecutive_frames,
    )
    return engine.process_frame(frame)


def create_demo_fundus(
    blur_level: float = 0.0,
    disc_x: int = 160,
    disc_y: int = 250,
    macula_dist_dd: float = 2.7,
    noise_sigma: float = 3.0,
) -> np.ndarray:
    """Generate a realistic synthetic fundus photograph for auto-capture testing and calibration."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Circular camera aperture FOV mask
    fov_mask = np.zeros((h, w), dtype=np.uint8)
    fov_center = (w // 2, h // 2)
    fov_radius = int(min(h, w) * 0.46)
    cv2.circle(fov_mask, fov_center, fov_radius, 255, -1)

    # Retinal background
    img[fov_mask > 0] = [20, 50, 160]

    # Radial illumination gradient
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_center = np.sqrt((x_grid - fov_center[0]) ** 2 + (y_grid - fov_center[1]) ** 2)
    falloff = np.clip(1.0 - (dist_center / fov_radius) * 0.32, 0.35, 1.0)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * falloff, 0, 255).astype(np.uint8)

    # Optic Disc (bright circular feature)
    od_radius = 35
    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, (disc_x, disc_y), od_radius, 255, -1)
    od_blur = cv2.GaussianBlur(od_mask, (15, 15), 0) / 255.0
    for c, val in enumerate([70, 200, 245]):
        img[:, :, c] = np.clip(img[:, :, c] * (1.0 - od_blur) + val * od_blur, 0, 255).astype(np.uint8)

    # Macula (dark absorption valley)
    dir_sign = 1.0 if disc_x < w // 2 else -1.0
    nom_dist = int(macula_dist_dd * (2 * od_radius))
    mac_x = int(disc_x + dir_sign * nom_dist)
    mac_y = disc_y
    mac_radius = 35

    dist_m = np.sqrt((x_grid - mac_x) ** 2 + (y_grid - mac_y) ** 2)
    m_mask = np.clip(1.0 - dist_m / float(mac_radius), 0.0, 1.0)
    m_mask = cv2.GaussianBlur(m_mask.astype(np.float32), (15, 15), 0)
    img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - 0.45 * m_mask), 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - 0.35 * m_mask), 0, 255).astype(np.uint8)

    # Blood vessels branching outward
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        sx = int(disc_x + np.cos(angle) * (od_radius * 0.25))
        sy = int(disc_y + np.sin(angle) * (od_radius * 0.25))
        ex = int(disc_x + np.cos(angle) * fov_radius * 0.75)
        ey = int(disc_y + np.sin(angle) * fov_radius * 0.75)
        cv2.line(img, (sx, sy), (ex, ey), (10, 25, 90), thickness=2)

    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Apply synthetic focus ring blur if requested
    if blur_level > 0.1:
        k = int(round(blur_level * 18))
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), blur_level * 5.0)

    img[fov_mask == 0] = 0
    return img

