"""
src/fundus2oct/data/quality_control.py

Automated Quality Control (QC) screening for clinical retinal fundus photographs.
Evaluates blurriness, contrast dynamic range, illumination extremes, resolution,
optic disc detection, macula/foveal center localization, and geometric anatomical validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

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
    geometric_qc: Optional["GeometricVerificationResult"] = None

    def summary_text(self) -> str:
        """Formatted human-readable QC summary."""
        lines = [
            f"QC Status: {self.status.value}",
            f"Resolution: {self.resolution[0]}x{self.resolution[1]}",
            f"Sharpness (Laplacian Var): {self.blur_score:.1f}",
            f"Contrast (Intensity Std): {self.contrast_std:.1f}",
            f"Underexposed: {self.underexposed_pct:.1f}% | Overexposed: {self.overexposed_pct:.1f}%",
        ]
        if self.geometric_qc is not None:
            lines.append(
                f"Retinal Geometry: {self.geometric_qc.status.value} "
                f"(Distance: {self.geometric_qc.distance_dd:.2f} DD, "
                f"Angle Dev: {self.geometric_qc.angle_deviation_deg:.1f}°)"
            )
        if self.issues:
            lines.append("Warnings/Issues:")
            for issue in self.issues:
                lines.append(f"  • {issue}")
        else:
            lines.append("✓ Image passed all quality screening checks.")
        return "\n".join(lines)


@dataclass
class OpticDiscResult:
    """Dataclass storing the detected optic disc properties.

    Attributes:
        center: (x, y) coordinates of the optic disc center.
        radius: Radius in pixels of the detected optic disc boundary.
        bbox: Bounding box as (x, y, width, height).
        mask: Binary mask of the detected optic disc region (uint8, 0 or 255).
        contour: Contours of the detected optic disc, if available.
        confidence: Confidence score between 0.0 and 1.0.
        method: Method used for detection ('hough', 'threshold', or 'hybrid').
    """
    center: Tuple[int, int]
    radius: int
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    contour: Optional[np.ndarray] = None
    confidence: float = 0.0
    method: str = "threshold"


@dataclass
class MaculaResult:
    """Dataclass storing the detected macula and foveal center properties.

    Attributes:
        center: (x, y) coordinates of the detected foveal center.
        radius: Estimated radius in pixels of the macular boundary.
        bbox: Bounding box as (x, y, width, height).
        mask: Binary mask of the macular region (uint8, 0 or 255).
        od_to_fovea_distance: Distance in pixels from optic disc center to foveal center.
        od_to_fovea_distance_dd: Distance expressed in units of Optic Disc Diameters (DD).
        od_to_fovea_angle: Angle in degrees from OD center to foveal center.
        confidence: Confidence score between 0.0 and 1.0.
        method: Method used for detection ('anatomical_valley', 'fallback', etc.).
    """
    center: Tuple[int, int]
    radius: int
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    od_to_fovea_distance: float
    od_to_fovea_distance_dd: float
    od_to_fovea_angle: float
    confidence: float = 0.0
    method: str = "anatomical_valley"


@dataclass
class GeometricVerificationResult:
    """Result of geometric distance and angle verification between optic disc and macula.

    Attributes:
        is_valid: Whether the geometric layout satisfies anatomical criteria.
        status: QCStatus (PASS, WARNING, FAIL).
        distance_px: Euclidean distance in pixels between OD and macula center.
        distance_dd: Distance normalized by optic disc diameter (DD).
        angle_deg: Polar angle in degrees from OD center to fovea (-180° to 180°).
        angle_deviation_deg: Absolute deviation from expected horizontal temporal axis.
        inferred_laterality: Inferred eye laterality ('OD' for Right Eye, 'OS' for Left Eye).
        issues: List of warning or failure reasons.
        metrics: Dictionary of numerical metrics.
    """
    is_valid: bool
    status: QCStatus
    distance_px: float
    distance_dd: float
    angle_deg: float
    angle_deviation_deg: float
    inferred_laterality: str
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def calculate_laplacian_variance(gray_array: np.ndarray) -> float:
    """Computes Laplacian variance as an indicator of focus/sharpness."""
    if gray_array.ndim != 2:
        raise ValueError("Expected 2D grayscale array for Laplacian variance.")

    img = gray_array.astype(np.float32)
    laplacian = (
        np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
        - 4.0 * img
    )
    laplacian = laplacian[1:-1, 1:-1]
    return float(np.var(laplacian))


def _ensure_uint8_image(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Normalize and convert input image to uint8 [0, 255]."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray or PIL.Image, got {type(image)}")

    if image.dtype == np.uint8:
        return image.copy()

    img_float = image.astype(np.float32)
    if img_float.max() <= 1.0 and img_float.min() >= 0.0:
        return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

    min_val, max_val = img_float.min(), img_float.max()
    if max_val > min_val:
        norm = (img_float - min_val) / (max_val - min_val) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)
    return np.zeros(img_float.shape, dtype=np.uint8)


def _extract_bright_channel(image: np.ndarray) -> np.ndarray:
    """Extract channel with highest optic disc contrast (Red / luminance)."""
    uint8_img = _ensure_uint8_image(image)
    if uint8_img.ndim == 2:
        return uint8_img
    elif uint8_img.ndim == 3:
        gray = cv2.cvtColor(uint8_img, cv2.COLOR_BGR2GRAY)
        red_channel = uint8_img[:, :, 2] if uint8_img.shape[2] >= 3 else gray
        combined = cv2.addWeighted(red_channel, 0.7, gray, 0.3, 0)
        return combined
    else:
        raise ValueError(f"Unsupported image dimensions: {uint8_img.ndim}")


def _preprocess_optic_disc(
    image: np.ndarray,
    clahe_clip_limit: float = 2.5,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    blur_kernel_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Enhance contrast and reduce noise for optic disc detection."""
    channel = _extract_bright_channel(image)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    enhanced = clahe.apply(channel)
    ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
    return blurred, enhanced


def detect_optic_disc_threshold(
    image: np.ndarray,
    percentile_threshold: float = 95.0,
    min_radius_ratio: float = 0.03,
    max_radius_ratio: float = 0.20,
    circularity_threshold: float = 0.35,
) -> Optional[OpticDiscResult]:
    """Detect optic disc using intensity thresholding and morphological filtering."""
    uint8_img = _ensure_uint8_image(image)
    h, w = uint8_img.shape[:2]
    min_dim = min(h, w)
    min_radius = max(5, int(min_dim * min_radius_ratio))
    max_radius = max(min_radius + 5, int(min_dim * max_radius_ratio))

    blurred, _ = _preprocess_optic_disc(image)

    retinal_mask = blurred > max(15, int(np.percentile(blurred, 10)))
    if np.any(retinal_mask):
        valid_intensities = blurred[retinal_mask]
        thresh_val = float(np.percentile(valid_intensities, percentile_threshold))
    else:
        thresh_val = float(np.percentile(blurred, percentile_threshold))

    _, binary = cv2.threshold(blurred, int(thresh_val), 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    morph_ksize = max(5, int(min_dim * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    highest_score = -1.0

    min_area = np.pi * (min_radius ** 2) * 0.5
    max_area = np.pi * (max_radius ** 2) * 1.5

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        if hull_perimeter == 0:
            continue

        circularity = (4.0 * np.pi * hull_area) / (hull_perimeter ** 2)
        if circularity < circularity_threshold:
            continue

        M = cv2.moments(cnt)
        (enc_cx, enc_cy), _ = cv2.minEnclosingCircle(cnt)
        radius = int(round(np.sqrt(hull_area / np.pi)))

        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
        else:
            cx, cy = int(round(enc_cx)), int(round(enc_cy))

        if radius < min_radius or radius > max_radius:
            continue

        mask_test = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_test, (cx, cy), radius, 255, -1)
        mean_intensity = cv2.mean(blurred, mask=mask_test)[0]

        score = circularity * 0.4 + (mean_intensity / 255.0) * 0.6

        if score > highest_score:
            highest_score = score
            best_candidate = (cx, cy, radius, cnt, score)

    if best_candidate is None:
        smoothed = cv2.GaussianBlur(blurred, (45, 45), 0)
        _, _, _, max_loc = cv2.minMaxLoc(smoothed)
        cx, cy = max_loc
        estimated_radius = int(min_dim * 0.08)
        bbox = (
            max(0, cx - estimated_radius),
            max(0, cy - estimated_radius),
            min(w - max(0, cx - estimated_radius), 2 * estimated_radius),
            min(h - max(0, cy - estimated_radius), 2 * estimated_radius),
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), estimated_radius, 255, -1)
        return OpticDiscResult(
            center=(cx, cy),
            radius=estimated_radius,
            bbox=bbox,
            mask=mask,
            contour=None,
            confidence=0.35,
            method="threshold",
        )

    cx, cy, radius, cnt, score = best_candidate
    x, y, bw, bh = cv2.boundingRect(cnt)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    return OpticDiscResult(
        center=(cx, cy),
        radius=radius,
        bbox=(x, y, bw, bh),
        mask=mask,
        contour=cnt,
        confidence=min(1.0, float(score)),
        method="threshold",
    )


def detect_optic_disc_hough(
    image: np.ndarray,
    min_radius_ratio: float = 0.03,
    max_radius_ratio: float = 0.15,
    dp: float = 1.2,
    param1: float = 50.0,
    param2: float = 30.0,
) -> Optional[OpticDiscResult]:
    """Detect optic disc using the Hough Circle Transform."""
    uint8_img = _ensure_uint8_image(image)
    h, w = uint8_img.shape[:2]
    min_dim = min(h, w)
    min_radius = max(5, int(min_dim * min_radius_ratio))
    max_radius = max(min_radius + 5, int(min_dim * max_radius_ratio))
    min_dist = max(min_radius * 2, int(min_dim * 0.2))

    blurred, _ = _preprocess_optic_disc(image, blur_kernel_size=9)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    best_circle = None
    best_intensity = -1.0

    for (cx, cy, r) in circles:
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean_val = cv2.mean(blurred, mask=mask)[0]

        if mean_val > best_intensity:
            best_intensity = mean_val
            best_circle = (cx, cy, r)

    if best_circle is None:
        return None

    cx, cy, r = best_circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    bbox = (max(0, cx - r), max(0, cy - r), min(w - max(0, cx - r), 2 * r), min(h - max(0, cy - r), 2 * r))

    confidence = min(1.0, float(best_intensity / 255.0))

    return OpticDiscResult(
        center=(cx, cy),
        radius=r,
        bbox=bbox,
        mask=mask,
        contour=None,
        confidence=confidence,
        method="hough",
    )


def detect_optic_disc(
    image: np.ndarray,
    method: str = "hybrid",
    **kwargs: Any,
) -> OpticDiscResult:
    """Detect the optic disc center and boundary in a fundus image."""
    method_lower = method.lower()

    if method_lower == "threshold":
        result = detect_optic_disc_threshold(image, **kwargs)
        return result if result is not None else _create_fallback_optic_disc(image)

    elif method_lower == "hough":
        result = detect_optic_disc_hough(image, **kwargs)
        if result is not None:
            return result
        res_fallback = detect_optic_disc_threshold(image)
        if res_fallback is not None:
            res_fallback.method = "hough_fallback_to_threshold"
            return res_fallback
        return _create_fallback_optic_disc(image)

    elif method_lower == "hybrid":
        thresh_result = detect_optic_disc_threshold(image, **kwargs)
        hough_result = detect_optic_disc_hough(image, **kwargs)

        if thresh_result is not None and hough_result is not None:
            tx, ty = thresh_result.center
            hx, hy = hough_result.center
            dist = np.hypot(tx - hx, ty - hy)

            if dist <= max(thresh_result.radius, hough_result.radius):
                final_center = (int(round(0.5 * (tx + hx))), int(round(0.5 * (ty + hy))))
                final_radius = int(round(0.5 * (thresh_result.radius + hough_result.radius)))
                uint8_img = _ensure_uint8_image(image)
                h, w = uint8_img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, final_center, final_radius, 255, -1)
                bbox = (
                    max(0, final_center[0] - final_radius),
                    max(0, final_center[1] - final_radius),
                    min(w - max(0, final_center[0] - final_radius), 2 * final_radius),
                    min(h - max(0, final_center[1] - final_radius), 2 * final_radius),
                )
                return OpticDiscResult(
                    center=final_center,
                    radius=final_radius,
                    bbox=bbox,
                    mask=mask,
                    contour=thresh_result.contour,
                    confidence=min(1.0, (thresh_result.confidence + hough_result.confidence) * 0.6),
                    method="hybrid",
                )

            if hough_result.confidence > thresh_result.confidence:
                return hough_result
            return thresh_result

        if hough_result is not None:
            return hough_result
        if thresh_result is not None:
            return thresh_result

        return _create_fallback_optic_disc(image)

    else:
        raise ValueError(
            f"Unknown detection method '{method}'. Choose from 'hybrid', 'threshold', or 'hough'."
        )


def _create_fallback_optic_disc(image: np.ndarray) -> OpticDiscResult:
    """Fallback optic disc detection when all algorithmic criteria fail."""
    uint8_img = _ensure_uint8_image(image)
    h, w = uint8_img.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(5, int(min(h, w) * 0.08))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    bbox = (max(0, cx - r), max(0, cy - r), min(w - max(0, cx - r), 2 * r), min(h - max(0, cy - r), 2 * r))
    return OpticDiscResult(
        center=(cx, cy),
        radius=r,
        bbox=bbox,
        mask=mask,
        confidence=0.1,
        method="fallback",
    )


def assess_optic_disc_focus(
    image: np.ndarray,
    optic_disc: OpticDiscResult,
    margin_ratio: float = 0.5,
    focus_threshold: float = 100.0,
) -> Dict[str, Any]:
    """Assess whether the fundus image is in focus at the optic disc."""
    uint8_img = _ensure_uint8_image(image)
    h, w = uint8_img.shape[:2]

    if uint8_img.ndim == 3:
        gray = uint8_img[:, :, 1] if uint8_img.shape[2] >= 2 else cv2.cvtColor(uint8_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = uint8_img

    cx, cy = optic_disc.center
    eff_radius = int(round(optic_disc.radius * (1.0 + margin_ratio)))

    x1 = max(0, cx - eff_radius)
    y1 = max(0, cy - eff_radius)
    x2 = min(w, cx + eff_radius)
    y2 = min(h, cy + eff_radius)

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return {
            "is_in_focus": False,
            "laplacian_variance": 0.0,
            "tenengrad_score": 0.0,
            "sharpness_score": 0.0,
        }

    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    lap_var = float(laplacian.var())

    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))

    norm_score = float(1.0 / (1.0 + np.exp(-((lap_var - focus_threshold) / (focus_threshold * 0.5 + 1e-6)))))

    return {
        "is_in_focus": lap_var >= focus_threshold,
        "laplacian_variance": lap_var,
        "tenengrad_score": tenengrad,
        "sharpness_score": norm_score,
    }


def draw_optic_disc(
    image: np.ndarray,
    optic_disc: OpticDiscResult,
    color_center: Tuple[int, int, int] = (0, 0, 255),
    color_boundary: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    draw_bbox: bool = False,
) -> np.ndarray:
    """Draw the optic disc center and boundary on the fundus image."""
    uint8_img = _ensure_uint8_image(image)
    if uint8_img.ndim == 2:
        canvas = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
    elif uint8_img.shape[2] == 1:
        canvas = cv2.cvtColor(uint8_img[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        canvas = uint8_img.copy()

    cx, cy = optic_disc.center
    radius = optic_disc.radius

    cv2.circle(canvas, (cx, cy), radius, color_boundary, thickness)
    cv2.circle(canvas, (cx, cy), max(2, thickness + 1), color_center, -1)

    if optic_disc.contour is not None:
        cv2.drawContours(canvas, [optic_disc.contour], -1, (255, 255, 0), 1)

    if draw_bbox:
        bx, by, bw, bh = optic_disc.bbox
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 128, 0), thickness)

    return canvas


def _extract_macula_channel(image: np.ndarray) -> np.ndarray:
    """Extract Green channel with highest macular luteal pigment absorption."""
    uint8_img = _ensure_uint8_image(image)
    if uint8_img.ndim == 2:
        return uint8_img
    elif uint8_img.shape[2] >= 3:
        return uint8_img[:, :, 1]
    return uint8_img[:, :, 0]


def detect_macula(
    image: np.ndarray,
    optic_disc: Optional[OpticDiscResult] = None,
    distance_dd_range: Tuple[float, float] = (2.0, 3.5),
    nominal_angle_deg: float = 0.0,
    angle_std_deg: float = 12.0,
    temporal_direction: Optional[str] = None,
    morph_kernel_ratio: float = 0.25,
) -> MaculaResult:
    """Detect the macula and foveal center in a fundus image using anatomical priors and valley searching."""
    uint8_img = _ensure_uint8_image(image)
    h, w = uint8_img.shape[:2]
    min_dim = min(h, w)

    if optic_disc is None:
        optic_disc = detect_optic_disc(image)

    od_x, od_y = optic_disc.center
    od_radius = optic_disc.radius
    od_diameter = max(20, 2 * od_radius)

    if temporal_direction is not None:
        dir_lower = temporal_direction.lower()
        dir_sign = -1.0 if dir_lower == "left" else 1.0
    else:
        dir_sign = 1.0 if od_x < (w / 2.0) else -1.0

    min_dd, max_dd = distance_dd_range
    nominal_dd = 0.5 * (min_dd + max_dd)
    nominal_dist_px = nominal_dd * od_diameter

    rad_angle = np.radians(nominal_angle_deg)
    mu_x = float(od_x + dir_sign * nominal_dist_px * np.cos(rad_angle))
    mu_y = float(od_y + nominal_dist_px * np.sin(rad_angle))

    mu_x = np.clip(mu_x, od_diameter * 0.5, w - od_diameter * 0.5)
    mu_y = np.clip(mu_y, od_diameter * 0.5, h - od_diameter * 0.5)

    dist_span = (max_dd - min_dd) * od_diameter
    sigma_x = max(15.0, dist_span * 0.5)
    sigma_y = max(12.0, nominal_dist_px * np.tan(np.radians(angle_std_deg)))

    y_grid, x_grid = np.ogrid[:h, :w]
    spatial_prior = np.exp(
        -0.5 * (((x_grid - mu_x) / sigma_x) ** 2 + ((y_grid - mu_y) / sigma_y) ** 2)
    ).astype(np.float32)

    od_dist = np.sqrt((x_grid - od_x) ** 2 + (y_grid - od_y) ** 2)
    spatial_prior[od_dist < (od_radius * 1.5)] = 0.0

    bright_ch = _extract_bright_channel(image)
    fov_mask = bright_ch > max(15, int(np.percentile(bright_ch, 10)))
    spatial_prior[~fov_mask] = 0.0

    channel = _extract_macula_channel(image)

    morph_k = max(5, int(od_diameter * morph_kernel_ratio))
    if morph_k % 2 == 0:
        morph_k += 1
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, struct_elem)

    bg_k = max(21, int(min_dim * 0.15))
    if bg_k % 2 == 0:
        bg_k += 1
    bg = cv2.GaussianBlur(closed, (bg_k, bg_k), 0)

    valley_map = np.clip(bg.astype(np.float32) - closed.astype(np.float32), 0.0, 255.0)
    valley_map = cv2.GaussianBlur(valley_map, (11, 11), 0)

    response_map = valley_map * spatial_prior
    max_val = float(response_map.max())

    if max_val > 1e-4:
        _, _, _, (peak_x, peak_y) = cv2.minMaxLoc(response_map)

        r_win = max(5, int(od_diameter * 0.15))
        x1 = max(0, peak_x - r_win)
        y1 = max(0, peak_y - r_win)
        x2 = min(w, peak_x + r_win + 1)
        y2 = min(h, peak_y + r_win + 1)

        patch = response_map[y1:y2, x1:x2]
        M = cv2.moments(patch)
        if M["m00"] > 0:
            fovea_x = int(round(x1 + M["m10"] / M["m00"]))
            fovea_y = int(round(y1 + M["m01"] / M["m00"]))
        else:
            fovea_x, fovea_y = peak_x, peak_y

        confidence = float(np.clip(0.4 * (max_val / 30.0) + 0.6 * spatial_prior[fovea_y, fovea_x], 0.2, 1.0))
        method_used = "anatomical_valley"
    else:
        fovea_x = int(round(mu_x))
        fovea_y = int(round(mu_y))
        confidence = 0.25
        method_used = "anatomical_prior_fallback"

    fovea_x = max(0, min(w - 1, fovea_x))
    fovea_y = max(0, min(h - 1, fovea_y))

    macula_radius = max(8, int(round(od_radius * 1.0)))

    dist_px = float(np.hypot(fovea_x - od_x, fovea_y - od_y))
    dist_dd = float(dist_px / max(1.0, float(od_diameter)))
    angle_deg = float(np.degrees(np.arctan2(fovea_y - od_y, fovea_x - od_x)))

    bx = max(0, fovea_x - macula_radius)
    by = max(0, fovea_y - macula_radius)
    bw = min(w - bx, 2 * macula_radius)
    bh = min(h - by, 2 * macula_radius)
    bbox = (bx, by, bw, bh)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (fovea_x, fovea_y), macula_radius, 255, -1)

    return MaculaResult(
        center=(fovea_x, fovea_y),
        radius=macula_radius,
        bbox=bbox,
        mask=mask,
        od_to_fovea_distance=dist_px,
        od_to_fovea_distance_dd=dist_dd,
        od_to_fovea_angle=angle_deg,
        confidence=confidence,
        method=method_used,
    )


def draw_macula(
    image: np.ndarray,
    macula: MaculaResult,
    optic_disc: Optional[OpticDiscResult] = None,
    color_fovea: Tuple[int, int, int] = (255, 0, 0),
    color_macula: Tuple[int, int, int] = (0, 165, 255),
    color_line: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    draw_connection: bool = True,
) -> np.ndarray:
    """Draw the macula boundary, foveal center, and OD-to-fovea vector on the fundus image."""
    uint8_img = _ensure_uint8_image(image)
    if uint8_img.ndim == 2:
        canvas = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
    elif uint8_img.shape[2] == 1:
        canvas = cv2.cvtColor(uint8_img[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        canvas = uint8_img.copy()

    fx, fy = macula.center
    mr = macula.radius

    cv2.circle(canvas, (fx, fy), mr, color_macula, thickness)
    cv2.circle(canvas, (fx, fy), max(2, thickness + 1), color_fovea, -1)

    if draw_connection and optic_disc is not None:
        od_x, od_y = optic_disc.center
        cv2.line(canvas, (od_x, od_y), (fx, fy), color_line, max(1, thickness - 1), lineType=cv2.LINE_AA)

        mid_x = int(round(0.5 * (od_x + fx)))
        mid_y = int(round(0.5 * (od_y + fy))) - 8
        label = f"{macula.od_to_fovea_distance_dd:.2f} DD"
        cv2.putText(
            canvas,
            label,
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_line,
            1,
            lineType=cv2.LINE_AA,
        )

    return canvas


def verify_optic_disc_macula_geometry(
    optic_disc: Union[OpticDiscResult, Tuple[int, int], Dict[str, Any]],
    macula: Union[MaculaResult, Tuple[int, int], Dict[str, Any]],
    od_radius: Optional[float] = None,
    distance_dd_min: float = 2.0,
    distance_dd_max: float = 3.5,
    distance_dd_fail_min: float = 1.8,
    distance_dd_fail_max: float = 4.0,
    max_angle_deviation_deg: float = 15.0,
    fail_angle_deviation_deg: float = 30.0,
    expected_laterality: Optional[str] = None,
) -> GeometricVerificationResult:
    """Validate Euclidean distance and angle between optic disc and macula against standardized retinal anatomical tolerances.

    Clinical Standards:
        - Distance: In standard human retinal anatomy, the fovea centralis lies approximately
          2.5 to 3.0 disc diameters (DD) from the optic disc center. The standard normal
          tolerance is [2.0, 3.5] DD. Values in [1.8, 2.0) or (3.5, 4.0] are flagged as warnings,
          and values < 1.8 or > 4.0 DD represent anatomical failures.
        - Angle: The fovea is located temporally within +/-15 degrees of the horizontal meridian
          passing through the optic disc center. Deviations > 30 degrees or nasal positioning
          represent anatomical landmark failures.

    Args:
        optic_disc: OpticDiscResult, (x, y) tuple, or dict with 'center' and 'radius'.
        macula: MaculaResult, (x, y) tuple, or dict with 'center'.
        od_radius: Explicit optic disc radius in pixels (required if optic_disc is a tuple).
        distance_dd_min: Minimum normal distance in disc diameters (default 2.0).
        distance_dd_max: Maximum normal distance in disc diameters (default 3.5).
        distance_dd_fail_min: Minimum allowable threshold before failure (default 1.8).
        distance_dd_fail_max: Maximum allowable threshold before failure (default 4.0).
        max_angle_deviation_deg: Maximum allowable deviation from temporal horizontal (default 15.0°).
        fail_angle_deviation_deg: Angle deviation triggering FAIL status (default 30.0°).
        expected_laterality: Optional expected eye ('OD' for Right Eye, 'OS' for Left Eye).

    Returns:
        GeometricVerificationResult with status (PASS, WARNING, FAIL), metrics, and clinical issues.
    """
    # 1. Parse Optic Disc coordinates and radius
    if isinstance(optic_disc, OpticDiscResult):
        od_x, od_y = optic_disc.center
        r_od = float(optic_disc.radius)
    elif isinstance(optic_disc, dict):
        center = optic_disc.get("center", (0, 0))
        od_x, od_y = center[0], center[1]
        r_od = float(optic_disc.get("radius", od_radius or 35.0))
    elif isinstance(optic_disc, (tuple, list)):
        od_x, od_y = float(optic_disc[0]), float(optic_disc[1])
        if od_radius is None:
            raise ValueError("od_radius must be provided when optic_disc is given as a tuple.")
        r_od = float(od_radius)
    else:
        raise TypeError(f"Unsupported optic_disc type: {type(optic_disc)}")

    # 2. Parse Macula coordinates
    if isinstance(macula, MaculaResult):
        mx, my = macula.center
    elif isinstance(macula, dict):
        center = macula.get("center", (0, 0))
        mx, my = center[0], center[1]
    elif isinstance(macula, (tuple, list)):
        mx, my = float(macula[0]), float(macula[1])
    else:
        raise TypeError(f"Unsupported macula type: {type(macula)}")

    # 3. Compute Euclidean Distance
    dx = float(mx - od_x)
    dy = float(my - od_y)
    dist_px = float(np.hypot(dx, dy))

    od_diameter = max(1.0, 2.0 * r_od)
    dist_dd = float(dist_px / od_diameter)

    # 4. Compute Vector Angle and Inferred Laterality
    angle_rad = np.arctan2(dy, dx)
    angle_deg = float(np.degrees(angle_rad))  # [-180, 180]

    if dx >= 0:
        inferred_laterality = "OD"  # Right Eye: temporal side is to the right
        angle_deviation = float(abs(angle_deg))
    else:
        inferred_laterality = "OS"  # Left Eye: temporal side is to the left
        angle_deviation = float(abs(180.0 - abs(angle_deg)))

    issues: List[str] = []
    status = QCStatus.PASS

    # 5. Distance Validation Checks
    if dist_dd < distance_dd_fail_min:
        issues.append(
            f"Distance between OD and macula is critically short: {dist_dd:.2f} DD "
            f"(clinical minimum is {distance_dd_fail_min:.1f} DD)."
        )
        status = QCStatus.FAIL
    elif dist_dd > distance_dd_fail_max:
        issues.append(
            f"Distance between OD and macula is critically large: {dist_dd:.2f} DD "
            f"(clinical maximum is {distance_dd_fail_max:.1f} DD)."
        )
        status = QCStatus.FAIL
    elif dist_dd < distance_dd_min:
        issues.append(
            f"Distance between OD and macula is slightly short: {dist_dd:.2f} DD "
            f"(expected standard range is [{distance_dd_min:.1f}, {distance_dd_max:.1f}] DD)."
        )
        if status != QCStatus.FAIL:
            status = QCStatus.WARNING
    elif dist_dd > distance_dd_max:
        issues.append(
            f"Distance between OD and macula is slightly large: {dist_dd:.2f} DD "
            f"(expected standard range is [{distance_dd_min:.1f}, {distance_dd_max:.1f}] DD)."
        )
        if status != QCStatus.FAIL:
            status = QCStatus.WARNING

    # 6. Angle Validation Checks
    if angle_deviation > fail_angle_deviation_deg:
        issues.append(
            f"Optic disc-to-macula angular deviation ({angle_deviation:.1f}°) exceeds "
            f"anatomical failure limit ({fail_angle_deviation_deg:.1f}°)."
        )
        status = QCStatus.FAIL
    elif angle_deviation > max_angle_deviation_deg:
        issues.append(
            f"Optic disc-to-macula angular deviation ({angle_deviation:.1f}°) exceeds "
            f"standard normal tolerance ({max_angle_deviation_deg:.1f}°)."
        )
        if status != QCStatus.FAIL:
            status = QCStatus.WARNING

    # 7. Laterality Consistency Check
    if expected_laterality is not None:
        expected_norm = expected_laterality.upper().strip()
        if expected_norm in ("OD", "RIGHT", "R") and inferred_laterality != "OD":
            issues.append(
                f"Detected anatomical orientation indicates Left Eye (OS), "
                f"conflicting with expected laterality {expected_laterality}."
            )
            status = QCStatus.FAIL
        elif expected_norm in ("OS", "LEFT", "L") and inferred_laterality != "OS":
            issues.append(
                f"Detected anatomical orientation indicates Right Eye (OD), "
                f"conflicting with expected laterality {expected_laterality}."
            )
            status = QCStatus.FAIL

    metrics = {
        "distance_px": dist_px,
        "distance_dd": dist_dd,
        "angle_deg": angle_deg,
        "angle_deviation_deg": angle_deviation,
        "od_radius": r_od,
        "od_x": float(od_x),
        "od_y": float(od_y),
        "macula_x": float(mx),
        "macula_y": float(my),
    }

    return GeometricVerificationResult(
        is_valid=(status != QCStatus.FAIL),
        status=status,
        distance_px=dist_px,
        distance_dd=dist_dd,
        angle_deg=angle_deg,
        angle_deviation_deg=angle_deviation,
        inferred_laterality=inferred_laterality,
        issues=issues,
        metrics=metrics,
    )


def draw_geometric_verification(
    image: np.ndarray,
    optic_disc: OpticDiscResult,
    macula: MaculaResult,
    verification: GeometricVerificationResult,
    thickness: int = 2,
) -> np.ndarray:
    """Draw comprehensive retinal landmark verification overlays on fundus image."""
    canvas = draw_optic_disc(image, optic_disc, thickness=thickness)
    canvas = draw_macula(canvas, macula, optic_disc=optic_disc, thickness=thickness, draw_connection=True)

    status_colors = {
        QCStatus.PASS: (0, 200, 0),       # Green
        QCStatus.WARNING: (0, 215, 255),  # Yellow/Gold
        QCStatus.FAIL: (0, 0, 255),       # Red
    }
    badge_color = status_colors.get(verification.status, (255, 255, 255))

    banner_text = (
        f"GEOM QC: {verification.status.value} | "
        f"Dist: {verification.distance_dd:.2f} DD | "
        f"Dev: {verification.angle_deviation_deg:.1f} deg | "
        f"Eye: {verification.inferred_laterality}"
    )

    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (10, 10), (min(w - 10, 520), 45), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, 10), (min(w - 10, 520), 45), badge_color, 1)
    cv2.putText(
        canvas,
        banner_text,
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        badge_color,
        1,
        lineType=cv2.LINE_AA,
    )

    return canvas


def check_fundus_quality(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    config: Optional[QualityControlConfig] = None,
    verify_geometry: bool = False,
    expected_laterality: Optional[str] = None,
) -> QCReport:
    """Screens retinal fundus image for clinical suitability, quality degradations, and landmark geometry."""
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
    np_img = np.array(pil_img, dtype=np.uint8)

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

    # 5. Optional Retinal Landmark Geometric QC
    geom_result = None
    if verify_geometry:
        try:
            od = detect_optic_disc(np_img, method="hybrid")
            mac = detect_macula(np_img, optic_disc=od)
            geom_result = verify_optic_disc_macula_geometry(
                od,
                mac,
                expected_laterality=expected_laterality,
            )
            if not geom_result.is_valid:
                is_fail = True
            for g_issue in geom_result.issues:
                issues.append(f"[Geometric QC] {g_issue}")
        except Exception as ge:
            issues.append(f"[Geometric QC] Landmark detection failed: {ge}")
            if status != QCStatus.FAIL:
                status = QCStatus.WARNING

    # Determine status
    if is_fail or (geom_result is not None and geom_result.status == QCStatus.FAIL):
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
    if geom_result is not None:
        metrics.update({f"geom_{k}": v for k, v in geom_result.metrics.items()})

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
        geometric_qc=geom_result,
    )
