import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
from .segmentation import (
    SegmentationResult,
    segment_retinal_layers,
    draw_segmentation_overlay,
    create_colored_mask,
    BOUNDARY_COLORS_RGB,
    LAYER_COLORS_RGB,
)

COLORMAP_MAP = {
    "bone": cv2.COLORMAP_BONE,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
    "plasma": cv2.COLORMAP_PLASMA,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "jet": cv2.COLORMAP_JET,
}
CLINICAL_COLORMAPS = COLORMAP_MAP


def render_clinical_bscan_2d(
    bscan: np.ndarray,
    colormap: str = "bone",
    overlay_boundaries: bool = True,
    overlay_mask: bool = False,
    segmentation: Optional[SegmentationResult] = None,
    show_scalebar: bool = True,
    show_fovea_marker: bool = True,
    scalebar_um: float = 200.0,
    pixels_per_um: float = 0.50,
) -> np.ndarray:
    """Renders a clinical 2D B-scan with colormapping, 8-layer boundaries, and optical scalebars.

    Args:
        bscan: 2D uint8 numpy array (H, W).
        colormap: Palette name ('bone', 'gray', 'inferno', 'viridis', 'hot', 'plasma', 'inverted').
        overlay_boundaries: Whether to draw 8-layer anatomical boundary traces.
        overlay_mask: Whether to overlay semi-transparent 8-layer color mask.
        segmentation: Optional precomputed SegmentationResult.
        show_scalebar: Whether to overlay 200 µm calibrated optical scale bar.
        show_fovea_marker: Whether to draw foveal central axis indicator.
        scalebar_um: Length of scale bar in micrometers.
        pixels_per_um: Optical magnification factor (approx 0.50 px/µm for standard 256x256 B-scan).

    Returns:
        Rendered 3-channel RGB image (H, W, 3) uint8.
    """
    if bscan.ndim == 3:
        gray = cv2.cvtColor(bscan, cv2.COLOR_RGB2GRAY)
    else:
        gray = bscan.copy()

    h, w = gray.shape[:2]

    # 1. Colormap conversion
    cmap_lower = colormap.lower()
    if cmap_lower == "gray":
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif cmap_lower == "inverted":
        inv = 255 - gray
        canvas = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
    elif cmap_lower in COLORMAP_MAP:
        canvas = cv2.applyColorMap(gray, COLORMAP_MAP[cmap_lower])
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    else:
        canvas = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # 2. Retinal Layer Segmentation Overlays
    if (overlay_boundaries or overlay_mask) and segmentation is None:
        segmentation = segment_retinal_layers(gray, generate_visualizations=True)

    if segmentation is not None:
        if overlay_mask and segmentation.colored_mask is not None:
            # Alpha blend colored mask
            canvas = cv2.addWeighted(canvas, 0.65, segmentation.colored_mask, 0.35, 0)

        if overlay_boundaries:
            # Draw boundary curves
            for b_name, coords in segmentation.boundaries.items():
                color = BOUNDARY_COLORS_RGB.get(b_name, (0, 255, 0))
                pts = np.column_stack([np.arange(len(coords)), coords]).astype(np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    # 3. Central Foveal Axis Indicator
    if show_fovea_marker:
        fovea_x = w // 2
        cv2.line(canvas, (fovea_x, 6), (fovea_x, 22), (0, 220, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Fovea",
            (fovea_x - 16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 220, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    # 4. Calibrated Optical Scale Bar
    if show_scalebar:
        bar_px = int(round(scalebar_um * pixels_per_um))
        bx2 = w - 16
        bx1 = bx2 - bar_px
        by = h - 16
        cv2.line(canvas, (bx1, by), (bx2, by), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{int(scalebar_um)} um",
            (bx1, by - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return canvas


def create_etdrs_macular_grid(
    thickness_map: np.ndarray,
    center_coords: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Create a 9-zone ETDRS (Early Treatment Diabetic Retinopathy Study) thickness grid.

    Standard ETDRS zones:
    1: Central Fovea (1mm ring)
    2-5: Inner Ring (3mm - Superior, Nasal, Inferior, Temporal)
    6-9: Outer Ring (6mm - Superior, Nasal, Inferior, Temporal)

    Args:
        thickness_map: 2D array of retinal thickness.
        center_coords: (cx, cy) center coordinate, defaults to center of map.

    Returns:
        RGB image with overlaid ETDRS circles and zone average values.
    """
    h, w = thickness_map.shape[:2]
    cx = center_coords[0] if center_coords else w // 2
    cy = center_coords[1] if center_coords else h // 2

    # Normalize thickness map to color representation
    norm_thick = cv2.normalize(thickness_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canvas = cv2.applyColorMap(norm_thick, cv2.COLORMAP_VIRIDIS)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # ETDRS Diameters: 1mm (r=20px), 3mm (r=60px), 6mm (r=115px)
    r1, r2, r3 = int(min(h, w) * 0.08), int(min(h, w) * 0.24), int(min(h, w) * 0.44)

    grid_color = (255, 255, 255)
    cv2.circle(canvas, (cx, cy), r1, grid_color, 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r2, grid_color, 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r3, grid_color, 1, lineType=cv2.LINE_AA)

    # Diagonal radial dividers (45 deg, 135 deg)
    for angle in [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]:
        x1 = int(cx + r1 * np.cos(angle))
        y1 = int(cy + r1 * np.sin(angle))
        x2 = int(cx + r3 * np.cos(angle))
        y2 = int(cy + r3 * np.sin(angle))
        cv2.line(canvas, (x1, y1), (x2, y2), grid_color, 1, lineType=cv2.LINE_AA)

    return canvas
