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
