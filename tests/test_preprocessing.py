"""
tests/test_preprocessing.py

Unit tests for image loading, FOV cropping, normalization,
and automated quality control (QC) screening.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from fundus2oct.data.preprocessing import (
    safe_load_image,
    extract_retinal_fov_bbox,
    preprocess_fundus,
    preprocess_oct_bscan,
    denormalize_to_numpy,
)
from fundus2oct.data.quality_control import (
    check_fundus_quality,
    calculate_laplacian_variance,
    QCStatus,
)


@pytest.fixture
def sample_fundus_image():
    """Generates a synthetic fundus-like image (circular aperture with black margins)."""
    h, w = 300, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (w // 2 - 20) ** 2
    # Red-orange retinal background
    img[circle_mask] = [200, 80, 40]
    # Add simulated optic disc
    disc_mask = (xx - 90) ** 2 + (yy - 150) ** 2 <= 20 ** 2
    img[disc_mask] = [250, 220, 150]
    return Image.fromarray(img)


@pytest.fixture
def sample_oct_bscan():
    """Generates a synthetic OCT-like B-scan (layered horizontal intensity bands)."""
    h, w = 200, 200
    arr = np.zeros((h, w), dtype=np.uint8)
    # Vitreous (dark) 0-50
    # Retinal layers (bright bands) 50-130
    arr[50:60, :] = 180   # ILM/RNFL
    arr[75:85, :] = 120   # IPL
    arr[100:110, :] = 160  # OPL
    arr[120:130, :] = 220  # IS/OS and RPE
    # Choroid (speckle) 130-170
    arr[130:170, :] = np.random.randint(40, 100, (40, w), dtype=np.uint8)
    return Image.fromarray(arr)


def test_safe_load_image(sample_fundus_image):
    """Verifies safe_load_image with PIL, numpy arrays, and channel conversions."""
    # Test PIL input
    loaded = safe_load_image(sample_fundus_image, expected_channels=3)
    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"

    # Test numpy array input
    np_arr = np.array(sample_fundus_image)
    loaded_np = safe_load_image(np_arr, expected_channels=3)
    assert loaded_np.size == sample_fundus_image.size

    # Test conversion to grayscale
    gray = safe_load_image(sample_fundus_image, expected_channels=1)
    assert gray.mode == "L"


def test_safe_load_image_corrupt():
    """Ensures corrupt/empty inputs raise clear ValueErrors."""
    empty_arr = np.array([])
    with pytest.raises(ValueError):
        safe_load_image(empty_arr)

    with pytest.raises(FileNotFoundError):
        safe_load_image("nonexistent_image_path.png")


def test_extract_retinal_fov_bbox(sample_fundus_image):
    """Verifies that FOV detection crops outer black margins correctly."""
    bbox = extract_retinal_fov_bbox(sample_fundus_image, threshold=15)
    xmin, ymin, xmax, ymax = bbox
    assert xmin > 0
    assert ymin > 0
    assert xmax < sample_fundus_image.width
    assert ymax < sample_fundus_image.height


def test_preprocess_fundus(sample_fundus_image):
    """Checks fundus preprocessing output shape, dtype, and range."""
    tensor = preprocess_fundus(
        sample_fundus_image,
        target_size=(256, 256),
        preserve_fov=True,
        normalize="bipolar",
    )
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_preprocess_oct_bscan(sample_oct_bscan):
    """Checks OCT B-scan preprocessing output shape and normalization."""
    tensor = preprocess_oct_bscan(
        sample_oct_bscan,
        target_size=(256, 256),
        normalize="bipolar",
    )
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_denormalize_to_numpy():
    """Verifies round-trip tensor denormalization to displayable uint8 array."""
    # Synthetic tensor in [-1, 1]
    tensor = torch.full((1, 64, 64), 0.0)  # midpoint should map to ~127
    arr = denormalize_to_numpy(tensor, normalize_mode="bipolar")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (64, 64)
    assert 126 <= arr[0, 0] <= 128


def test_quality_control(sample_fundus_image):
    """Verifies automated quality control returns PASS on clear synthetic fundus."""
    report = check_fundus_quality(sample_fundus_image)
    assert report.is_valid is True
    assert report.resolution == sample_fundus_image.size
    assert report.blur_score > 0.0
    assert report.status in (QCStatus.PASS, QCStatus.WARNING)


def test_quality_control_blur():
    """Verifies that an artificially blurred image is flagged with low sharpness."""
    # Uniform gray image has zero Laplacian variance (maximum blur)
    flat_img = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
    report = check_fundus_quality(flat_img)
    assert report.blur_score == 0.0
    assert any("blurred" in issue.lower() for issue in report.issues)


def test_geometric_verification_normal_pass():
    """Validates that a normal anatomical OD-macula layout yields PASS."""
    from fundus2oct.data.quality_control import (
        verify_optic_disc_macula_geometry,
        OpticDiscResult,
        MaculaResult,
        QCStatus,
    )
    # Right Eye (OD): Optic Disc at (150, 250), radius 30 (diameter 60)
    # Macula at (315, 255) -> dx = 165, dy = 5 -> distance = 165.08 px = 2.75 DD, angle = 1.74 deg
    od = OpticDiscResult(
        center=(150, 250),
        radius=30,
        bbox=(120, 220, 60, 60),
        mask=np.zeros((500, 500), dtype=np.uint8),
    )
    macula = MaculaResult(
        center=(315, 255),
        radius=30,
        bbox=(285, 225, 60, 60),
        mask=np.zeros((500, 500), dtype=np.uint8),
        od_to_fovea_distance=165.08,
        od_to_fovea_distance_dd=2.75,
        od_to_fovea_angle=1.74,
    )

    geom = verify_optic_disc_macula_geometry(od, macula, expected_laterality="OD")
    assert geom.is_valid is True
    assert geom.status == QCStatus.PASS
    assert geom.inferred_laterality == "OD"
    assert 2.5 <= geom.distance_dd <= 3.0
    assert geom.angle_deviation_deg < 5.0
    assert len(geom.issues) == 0


def test_geometric_verification_left_eye_pass():
    """Validates Left Eye (OS) geometry where macula is temporal (left of OD)."""
    from fundus2oct.data.quality_control import verify_optic_disc_macula_geometry, QCStatus

    # Left eye: OD at (350, 250), radius 30 (diameter 60). Macula at (185, 248) -> dx = -165, dy = -2
    od_pt = (350, 250)
    macula_pt = (185, 248)
    geom = verify_optic_disc_macula_geometry(
        od_pt,
        macula_pt,
        od_radius=30,
        expected_laterality="OS",
    )
    assert geom.is_valid is True
    assert geom.status == QCStatus.PASS
    assert geom.inferred_laterality == "OS"
    assert 2.7 <= geom.distance_dd <= 2.8
    assert geom.angle_deviation_deg < 2.0


def test_geometric_verification_distance_failures():
    """Validates tolerance failure when distance is anatomically impossible."""
    from fundus2oct.data.quality_control import verify_optic_disc_macula_geometry, QCStatus

    # Critically short distance: 1.2 DD (72 px with diameter 60)
    geom_short = verify_optic_disc_macula_geometry(
        (150, 250),
        (222, 250),
        od_radius=30,
    )
    assert geom_short.is_valid is False
    assert geom_short.status == QCStatus.FAIL
    assert any("critically short" in issue.lower() for issue in geom_short.issues)

    # Critically large distance: 4.8 DD (288 px with diameter 60)
    geom_long = verify_optic_disc_macula_geometry(
        (100, 250),
        (388, 250),
        od_radius=30,
    )
    assert geom_long.is_valid is False
    assert geom_long.status == QCStatus.FAIL
    assert any("critically large" in issue.lower() for issue in geom_long.issues)


def test_geometric_verification_distance_warning():
    """Validates WARNING status for borderline distance (e.g. 1.9 DD)."""
    from fundus2oct.data.quality_control import verify_optic_disc_macula_geometry, QCStatus

    # 1.9 DD (114 px with diameter 60) -> within [1.8, 2.0)
    geom = verify_optic_disc_macula_geometry(
        (150, 250),
        (264, 250),
        od_radius=30,
    )
    assert geom.is_valid is True
    assert geom.status == QCStatus.WARNING
    assert any("slightly short" in issue.lower() for issue in geom.issues)


def test_geometric_verification_angle_failure():
    """Validates failure when macula is placed far outside horizontal meridian."""
    from fundus2oct.data.quality_control import verify_optic_disc_macula_geometry, QCStatus

    # Macula placed directly below OD (dy = 160 px, dx = 0 px => angle = 90 deg)
    geom = verify_optic_disc_macula_geometry(
        (200, 100),
        (200, 260),
        od_radius=30,
    )
    assert geom.is_valid is False
    assert geom.status == QCStatus.FAIL
    assert any("angular deviation" in issue.lower() for issue in geom.issues)


def test_geometric_verification_laterality_conflict():
    """Validates failure when detected orientation conflicts with expected laterality."""
    from fundus2oct.data.quality_control import verify_optic_disc_macula_geometry, QCStatus

    # Macula is to the right of OD (Right Eye OD), but expected is Left Eye OS
    geom = verify_optic_disc_macula_geometry(
        (150, 250),
        (315, 250),
        od_radius=30,
        expected_laterality="OS",
    )
    assert geom.is_valid is False
    assert geom.status == QCStatus.FAIL
    assert any("conflicting with expected laterality" in issue.lower() for issue in geom.issues)


def test_draw_geometric_verification():
    """Verifies visualization overlay generation."""
    from fundus2oct.data.quality_control import (
        verify_optic_disc_macula_geometry,
        draw_geometric_verification,
        OpticDiscResult,
        MaculaResult,
    )
    canvas = np.full((400, 400, 3), 50, dtype=np.uint8)
    od = OpticDiscResult(
        center=(120, 200),
        radius=25,
        bbox=(95, 175, 50, 50),
        mask=np.zeros((400, 400), dtype=np.uint8),
    )
    macula = MaculaResult(
        center=(250, 205),
        radius=25,
        bbox=(225, 180, 50, 50),
        mask=np.zeros((400, 400), dtype=np.uint8),
        od_to_fovea_distance=130.1,
        od_to_fovea_distance_dd=2.6,
        od_to_fovea_angle=2.2,
    )
    geom = verify_optic_disc_macula_geometry(od, macula)
    annotated = draw_geometric_verification(canvas, od, macula, geom)
    assert annotated.shape == canvas.shape
    assert annotated.dtype == np.uint8

