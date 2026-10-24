"""
src/fundus2oct/data/quality_control.py

Automated Quality Control (QC) screening for clinical retinal fundus photographs.
Evaluates blurriness, contrast dynamic range, illumination extremes, and resolution.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from fundus2oct.config import QualityControlConfig
from fundus2oct.data.preprocessing import safe_load_image


class QCStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class QCReport:
    status: QCStatus
    is_valid: bool
    blur_score: float
    contrast_std: float
    underexposed_pct: float
    overexposed_pct: float
    resolution: Tuple[int, int]
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary_text(self) -> str:
        """Formatted human-readable QC summary."""
        lines = [
            f"QC Status: {self.status.value}",
            f"Resolution: {self.resolution[0]}x{self.resolution[1]}",
            f"Sharpness (Laplacian Var): {self.blur_score:.1f}",
            f"Contrast (Intensity Std): {self.contrast_std:.1f}",
            f"Underexposed: {self.underexposed_pct:.1f}% | Overexposed: {self.overexposed_pct:.1f}%",
        ]
        if self.issues:
            lines.append("Warnings/Issues:")
            for issue in self.issues:
                lines.append(f"  • {issue}")
        else:
            lines.append("✓ Image passed all quality screening checks.")
        return "\n".join(lines)


def calculate_laplacian_variance(gray_array: np.ndarray) -> float:
    """
    Computes Laplacian variance as an indicator of focus/sharpness.
    Uses discrete 3x3 Laplacian convolution kernel.
    """
    if gray_array.ndim != 2:
        raise ValueError("Expected 2D grayscale array for Laplacian variance.")

    # 3x3 discrete Laplacian kernel
    # [ 0,  1,  0]
    # [ 1, -4,  1]
    # [ 0,  1,  0]
    img = gray_array.astype(np.float32)
    laplacian = (
        np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
        - 4.0 * img
    )
    # Clip borders to avoid roll wrap-around artifacts
    laplacian = laplacian[1:-1, 1:-1]
    return float(np.var(laplacian))


def check_fundus_quality(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    config: Optional[QualityControlConfig] = None,
) -> QCReport:
    """
    Screens retinal fundus image for clinical suitability and quality degradations.
    """
    cfg = config or QualityControlConfig()
    try:
        pil_img = safe_load_image(image_input, expected_channels=3)
    except Exception as e:
        return QCReport(
            status=QCStatus.FAIL,
            is_valid=False,
            blur_score=0.0,
            contrast_std=0.0,
            underexposed_pct=100.0,
            overexposed_pct=0.0,
            resolution=(0, 0),
            issues=[f"Corrupted or unreadable image file: {e}"],
        )

    w, h = pil_img.size
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)

    issues: List[str] = []
    is_fail = False

    # 1. Resolution check
    if w < cfg.min_resolution[0] or h < cfg.min_resolution[1]:
        issues.append(f"Low resolution: {w}x{h} (minimum required is {cfg.min_resolution[0]}x{cfg.min_resolution[1]})")
        is_fail = True

    # 2. Sharpness / Blurriness check
    blur_score = calculate_laplacian_variance(gray)
    if blur_score < cfg.blur_laplacian_threshold:
        issues.append(f"Image appears blurred/out-of-focus (sharpness: {blur_score:.1f} < threshold {cfg.blur_laplacian_threshold})")

    # 3. Contrast check
    contrast_std = float(np.std(gray))
    if contrast_std < cfg.min_contrast_std:
        issues.append(f"Low contrast dynamic range (std: {contrast_std:.1f} < threshold {cfg.min_contrast_std})")

    # 4. Illumination extremes
    total_pixels = gray.size
    under_count = np.sum(gray < 10)
    over_count = np.sum(gray > 245)

    under_pct = float(under_count / total_pixels * 100.0)
    over_pct = float(over_count / total_pixels * 100.0)

    if under_pct > cfg.underexposure_max_percent:
        issues.append(f"Excessive underexposure / dark regions ({under_pct:.1f}% > {cfg.underexposure_max_percent}%)")

    if over_pct > cfg.overexposure_max_percent:
        issues.append(f"Excessive glare or overexposure ({over_pct:.1f}% > {cfg.overexposure_max_percent}%)")

    # Determine status
    if is_fail:
        status = QCStatus.FAIL
        is_valid = False
    elif issues:
        status = QCStatus.WARNING
        is_valid = True
    else:
        status = QCStatus.PASS
        is_valid = True

    metrics = {
        "blur_score": blur_score,
        "contrast_std": contrast_std,
        "underexposed_pct": under_pct,
        "overexposed_pct": over_pct,
        "width": float(w),
        "height": float(h),
    }

    return QCReport(
        status=status,
        is_valid=is_valid,
        blur_score=blur_score,
        contrast_std=contrast_std,
        underexposed_pct=under_pct,
        overexposed_pct=over_pct,
        resolution=(w, h),
        issues=issues,
        metrics=metrics,
    )
