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
    center_crop_to_square,
    extract_retinal_fov_bbox,
    extract_central_bscans,
    preprocess_fundus,
    preprocess_oct_bscan,
    preprocess_oct_volume,
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



def test_compute_modified_laplacian_variance():
    """Validates that MLV differentiates sharp vs blurred patterns."""
    from fundus2oct.data.quality_control import compute_modified_laplacian_variance
    import cv2

    # Sharp image with synthetic edges
    sharp = np.zeros((200, 200), dtype=np.uint8)
    for i in range(20, 180, 20):
        cv2.line(sharp, (i, 20), (i, 180), 220, 2)
        cv2.line(sharp, (20, i), (180, i), 220, 2)

    blurred = cv2.GaussianBlur(sharp, (21, 21), 6.0)

    mlv_sharp_var, mlv_sharp_mean = compute_modified_laplacian_variance(sharp)
    mlv_blur_var, mlv_blur_mean = compute_modified_laplacian_variance(blurred)

    assert mlv_sharp_var > 100.0
    assert mlv_sharp_var > 10.0 * mlv_blur_var
    assert mlv_sharp_mean > mlv_blur_mean


def test_compute_frequency_tenengrad():
    """Validates frequency domain Tenengrad gradient analysis."""
    from fundus2oct.data.quality_control import compute_frequency_tenengrad
    import cv2

    sharp = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(sharp, (128, 128), 80, 200, -1)
    cv2.line(sharp, (50, 50), (200, 200), 255, 3)
    cv2.line(sharp, (70, 190), (190, 70), 255, 2)

    blurred = cv2.GaussianBlur(sharp, (19, 19), 5.0)

    t_spatial_sharp, hf_sharp, spectral_sharp = compute_frequency_tenengrad(sharp)
    t_spatial_blur, hf_blur, spectral_blur = compute_frequency_tenengrad(blurred)

    # Spatial Tenengrad
    assert t_spatial_sharp > t_spatial_blur
    # High frequency spectral energy ratio
    assert hf_sharp > hf_blur
    # Frequency-weighted spectral energy
    assert spectral_sharp > spectral_blur


def test_compute_focus_clarity_score(sample_fundus_image):
    """Validates composite focus clarity score on clinical/synthetic fundus."""
    from fundus2oct.data.quality_control import compute_focus_clarity_score, QCStatus
    import cv2

    fundus_arr = np.array(sample_fundus_image)
    # Sharp fundus
    sharp_result = compute_focus_clarity_score(fundus_arr)
    assert sharp_result.clarity_score >= 0.40
    assert sharp_result.status in (QCStatus.PASS, QCStatus.WARNING)
    assert sharp_result.modified_laplacian_variance > 0.0
    assert sharp_result.tenengrad_spatial > 0.0
    assert sharp_result.tenengrad_freq_ratio > 0.0

    # Heavily blurred fundus
    blurred_arr = cv2.GaussianBlur(fundus_arr, (25, 25), 8.0)
    blurred_result = compute_focus_clarity_score(blurred_arr)
    assert blurred_result.clarity_score < sharp_result.clarity_score
    assert blurred_result.status == QCStatus.FAIL
    assert blurred_result.is_in_focus is False
    assert any("out-of-focus" in iss.lower() or "blurred" in iss.lower() for iss in blurred_result.issues)


def test_draw_focus_clarity(sample_fundus_image):
    """Validates focus clarity overlay generation."""
    from fundus2oct.data.quality_control import compute_focus_clarity_score, draw_focus_clarity

    fundus_arr = np.array(sample_fundus_image)
    clarity_res = compute_focus_clarity_score(fundus_arr)
    annotated = draw_focus_clarity(fundus_arr, clarity_res)

    assert annotated.shape == fundus_arr.shape
    assert annotated.dtype == np.uint8

# ---------------------------------------------------------------------------
# Unit Tests: Center-crop to Square, Central 32 B-Scans, & 256x256 Resizing
# ---------------------------------------------------------------------------

def test_center_crop_to_square_pil():
    """Validates that PIL images (wide, tall, and square) are center-cropped to squares."""
    # 1. Wide image (2144 x 1424, real fundus aspect ratio)
    wide_img = Image.new("RGB", (2144, 1424), color=(200, 100, 50))
    cropped_wide = center_crop_to_square(wide_img)
    assert cropped_wide.size == (1424, 1424)

    # 2. Tall image (600 x 900)
    tall_img = Image.new("RGB", (600, 900), color=(100, 150, 200))
    cropped_tall = center_crop_to_square(tall_img)
    assert cropped_tall.size == (600, 600)

    # 3. Already square image (512 x 512)
    sq_img = Image.new("RGB", (512, 512), color=(50, 50, 50))
    cropped_sq = center_crop_to_square(sq_img)
    assert cropped_sq.size == (512, 512)


def test_center_crop_to_square_numpy():
    """Validates center cropping for 2D and 3D NumPy arrays."""
    # 2D array: (H=300, W=500)
    arr_2d = np.zeros((300, 500), dtype=np.uint8)
    cropped_2d = center_crop_to_square(arr_2d)
    assert cropped_2d.shape == (300, 300)

    # 3D HWC array: (H=400, W=600, C=3)
    arr_hwc = np.zeros((400, 600, 3), dtype=np.uint8)
    cropped_hwc = center_crop_to_square(arr_hwc)
    assert cropped_hwc.shape == (400, 400, 3)

    # 3D CHW array: (C=3, H=500, W=300)
    arr_chw = np.zeros((3, 500, 300), dtype=np.uint8)
    cropped_chw = center_crop_to_square(arr_chw)
    assert cropped_chw.shape == (3, 300, 300)


def test_center_crop_to_square_tensor():
    """Validates center cropping for PyTorch tensors."""
    # (C, H, W): (3, 200, 350)
    t_chw = torch.randn(3, 200, 350)
    cropped_t = center_crop_to_square(t_chw)
    assert cropped_t.shape == (3, 200, 200)

    # Batched (B, C, H, W): (4, 1, 400, 300)
    t_bchw = torch.randn(4, 1, 400, 300)
    cropped_bt = center_crop_to_square(t_bchw)
    assert cropped_bt.shape == (4, 1, 300, 300)


def test_extract_central_bscans_greater_than_32():
    """Validates central 32 B-scans extraction from volume with D > 32."""
    # Volume with 49 slices, H=100, W=100
    # Each slice labeled by its slice index
    vol = np.zeros((49, 100, 100), dtype=np.float32)
    for i in range(49):
        vol[i] = float(i)

    # Extract central 32 B-scans and resize spatially to (256, 256)
    out = extract_central_bscans(vol, num_scans=32, target_size=(256, 256))
    assert out.shape == (32, 256, 256)

    # Verify slices extracted are indeed the central ones: start = (49 - 32) // 2 = 8
    # Slice 0 of output corresponds to slice 8 of original
    expected_start_val = (49 - 32) // 2  # 8
    assert np.isclose(out[0, 0, 0], expected_start_val)
    assert np.isclose(out[-1, 0, 0], expected_start_val + 31)


def test_extract_central_bscans_equal_32():
    """Validates that a volume with exactly 32 B-scans preserves all slices."""
    vol = np.random.randn(32, 128, 128).astype(np.float32)
    out = extract_central_bscans(vol, num_scans=32, target_size=(256, 256))
    assert out.shape == (32, 256, 256)


def test_extract_central_bscans_less_than_32():
    """Validates symmetric edge padding when volume depth D < 32."""
    vol = np.zeros((16, 80, 80), dtype=np.float32)
    for i in range(16):
        vol[i] = float(i + 1)  # 1 to 16

    out = extract_central_bscans(vol, num_scans=32, target_size=(256, 256), pad_mode="edge")
    assert out.shape == (32, 256, 256)

    # Padded before: (32 - 16) // 2 = 8 slices of value 1.0
    assert np.isclose(out[0, 0, 0], 1.0)
    assert np.isclose(out[7, 0, 0], 1.0)
    # Original first slice at index 8
    assert np.isclose(out[8, 0, 0], 1.0)
    # Original last slice at index 23
    assert np.isclose(out[23, 0, 0], 16.0)
    # Padded after slices of value 16.0
    assert np.isclose(out[-1, 0, 0], 16.0)


def test_extract_central_bscans_tensor():
    """Validates central 32 extraction on PyTorch tensors."""
    t_vol = torch.randn(64, 150, 150)
    out = extract_central_bscans(t_vol, num_scans=32, target_size=(256, 256))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (32, 256, 256)


def test_extract_central_bscans_list_of_images():
    """Validates extracting central 32 B-scans from a list of 2D images."""
    img_list = [Image.new("L", (100, 100), color=i) for i in range(40)]
    out = extract_central_bscans(img_list, num_scans=32, target_size=(256, 256))
    assert isinstance(out, np.ndarray)
    assert out.shape == (32, 256, 256)


def test_preprocess_fundus_center_crop_and_resize_256():
    """Validates full fundus preprocessing: center-crop square + resize to 256x256."""
    # Wide image simulating typical fundus camera output (e.g., 2144 x 1424)
    wide_fundus = Image.new("RGB", (2144, 1424), color=(180, 80, 40))
    tensor = preprocess_fundus(
        wide_fundus,
        target_size=(256, 256),
        center_crop=True,
        normalize="bipolar",
    )
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert -1.0 <= tensor.min() <= 1.0
    assert -1.0 <= tensor.max() <= 1.0


def test_preprocess_oct_volume_central_32():
    """Validates full OCT volume preprocessing: central 32 B-scans + resize to (1, 32, 256, 256)."""
    raw_vol = np.random.randint(0, 256, (60, 200, 200), dtype=np.uint8)
    vol_tensor = preprocess_oct_volume(
        raw_vol,
        target_shape=(32, 256, 256),
        num_scans=32,
        normalize="bipolar",
    )
    assert isinstance(vol_tensor, torch.Tensor)
    assert vol_tensor.shape == (1, 32, 256, 256)
    assert vol_tensor.dtype == torch.float32
    assert vol_tensor.min() >= -1.0
    assert vol_tensor.max() <= 1.0


def test_auto_capture_engine_trigger_flow():
    """Validates real-time auto-capture feedback loop and stability trigger."""
    from fundus2oct.data.auto_capture import AutoCaptureEngine, GuidanceCue, create_demo_fundus

    engine = AutoCaptureEngine(
        min_clarity_score=0.40,
        required_consecutive_frames=3,
    )

    sharp_frame = create_demo_fundus(blur_level=0.0)

    # Frame 1: Passes, holding steady (1/3)
    dec1 = engine.process_frame(sharp_frame)
    assert dec1.is_focus_valid is True
    assert dec1.should_capture is False
    assert dec1.consecutive_passes == 1
    assert dec1.guidance_cue == GuidanceCue.HOLD_STEADY

    # Frame 2: Passes, holding steady (2/3)
    dec2 = engine.process_frame(sharp_frame)
    assert dec2.should_capture is False
    assert dec2.consecutive_passes == 2

    # Frame 3: Stability lock-in met -> TRIGGER CAPTURE!
    dec3 = engine.process_frame(sharp_frame)
    assert dec3.should_capture is True
    assert dec3.consecutive_passes == 3
    assert dec3.guidance_cue == GuidanceCue.CAPTURE
    assert dec3.annotated_frame is not None


def test_auto_capture_engine_blocks_on_blurred_frame():
    """Validates that auto-capture rejects blurred frames and instructs focus adjustment."""
    from fundus2oct.data.auto_capture import AutoCaptureEngine, GuidanceCue, create_demo_fundus

    engine = AutoCaptureEngine(min_clarity_score=0.45)
    blurred_frame = create_demo_fundus(blur_level=0.9)

    dec = engine.process_frame(blurred_frame)
    assert dec.should_capture is False
    assert dec.is_focus_valid is False
    assert dec.guidance_cue == GuidanceCue.ADJUST_FOCUS
    assert "focus" in dec.feedback_message.lower()


def test_evaluate_auto_capture_frame_convenience():
    """Validates the convenience wrapper evaluate_auto_capture_frame in quality_control."""
    from fundus2oct.data.quality_control import evaluate_auto_capture_frame
    from fundus2oct.data.auto_capture import create_demo_fundus

    frame = create_demo_fundus(blur_level=0.0)
    decision = evaluate_auto_capture_frame(frame, min_clarity_score=0.40, required_consecutive_frames=1)
    assert decision.should_capture is True
    assert decision.is_focus_valid is True


def test_apply_clahe_grayscale_and_color():
    """Validates CLAHE enhancement across grayscale and LAB color channels."""
    from fundus2oct.data.quality_control import apply_clahe

    # 1. Grayscale image
    gray = np.random.randint(50, 150, (128, 128), dtype=np.uint8)
    enhanced_gray = apply_clahe(gray, clip_limit=3.0, tile_grid_size=(8, 8))
    assert enhanced_gray.shape == gray.shape
    assert enhanced_gray.dtype == np.uint8
    assert enhanced_gray.std() >= gray.std()

    # 2. RGB image in LAB mode
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180  # Red
    rgb[:, :, 1] = 80   # Green
    rgb[:, :, 2] = 40   # Blue
    enhanced_rgb = apply_clahe(rgb, clip_limit=2.5, tile_grid_size=(8, 8), color_space="LAB")
    assert enhanced_rgb.shape == rgb.shape
    assert enhanced_rgb.dtype == np.uint8


def test_preprocess_optic_disc_clahe():
    """Validates optic disc bright channel extraction, CLAHE enhancement, and vessel suppression."""
    from fundus2oct.data.quality_control import preprocess_optic_disc_clahe
    from fundus2oct.data.auto_capture import create_demo_fundus

    fundus = create_demo_fundus(blur_level=0.0, disc_x=160, disc_y=250)
    blurred, clahe_enh, vessel_supp = preprocess_optic_disc_clahe(
        fundus,
        clahe_clip_limit=2.5,
        clahe_grid_size=(8, 8),
        suppress_vessels=True,
    )
    assert blurred.shape == fundus.shape[:2]
    assert clahe_enh.shape == fundus.shape[:2]
    assert vessel_supp.shape == fundus.shape[:2]
    # Disc region (around 160, 250) should be bright
    assert blurred[250, 160] > blurred[250, 340]


def test_preprocess_macula_clahe():
    """Validates macula green channel luteal extraction, CLAHE, and valley response map."""
    from fundus2oct.data.quality_control import preprocess_macula_clahe
    from fundus2oct.data.auto_capture import create_demo_fundus

    fundus = create_demo_fundus(blur_level=0.0, disc_x=160, disc_y=250, macula_dist_dd=2.7)
    valley_map, clahe_enh, closed_ch = preprocess_macula_clahe(
        fundus,
        clahe_clip_limit=2.5,
        clahe_grid_size=(8, 8),
        od_diameter=70,
    )
    assert valley_map.shape == fundus.shape[:2]
    assert clahe_enh.shape == fundus.shape[:2]
    assert closed_ch.shape == fundus.shape[:2]
    assert valley_map.max() > 5.0


def test_detect_optic_disc_with_clahe():
    """Validates accurate optic disc localization using CLAHE enhancement."""
    from fundus2oct.data.quality_control import detect_optic_disc
    from fundus2oct.data.auto_capture import create_demo_fundus

    fundus = create_demo_fundus(blur_level=0.0, disc_x=160, disc_y=250)
    od = detect_optic_disc(fundus, method="hybrid", use_clahe=True)

    assert od is not None
    assert od.confidence >= 0.70
    assert abs(od.center[0] - 160) <= 8
    assert abs(od.center[1] - 250) <= 8
    assert 25 <= od.radius <= 45


def test_detect_macula_with_clahe():
    """Validates foveal center detection with CLAHE and anatomical prior."""
    from fundus2oct.data.quality_control import detect_optic_disc, detect_macula
    from fundus2oct.data.auto_capture import create_demo_fundus

    # True OD at (160, 250), True macula at (349, 250) -> distance ~2.7 DD
    fundus = create_demo_fundus(blur_level=0.0, disc_x=160, disc_y=250, macula_dist_dd=2.7)
    od = detect_optic_disc(fundus, use_clahe=True)
    mac = detect_macula(fundus, optic_disc=od, use_clahe=True)

    assert mac is not None
    assert mac.confidence >= 0.40
    assert "clahe" in mac.method
    # Foveal center should be positioned temporally to the OD
    assert mac.center[0] > od.center[0]
    assert abs(mac.center[1] - od.center[1]) <= 15
    assert 2.2 <= mac.od_to_fovea_distance_dd <= 3.2


def test_detect_retinal_landmarks_full_pipeline():
    """Validates the coordinated all-in-one CLAHE landmark detection pipeline."""
    from fundus2oct.data.quality_control import detect_retinal_landmarks, QCStatus
    from fundus2oct.data.auto_capture import create_demo_fundus

    fundus = create_demo_fundus(blur_level=0.0, disc_x=160, disc_y=250, macula_dist_dd=2.7)
    res = detect_retinal_landmarks(fundus, use_clahe=True, expected_laterality="OD")

    assert res["is_valid"] is True
    assert res["optic_disc"].confidence >= 0.70
    assert res["macula"].confidence >= 0.40
    assert res["geometric_qc"] is not None
    assert res["geometric_qc"].status == QCStatus.PASS
    assert res["geometric_qc"].inferred_laterality == "OD"
    assert res["clahe_image"].shape == fundus.shape
    assert res["annotated_image"].shape == fundus.shape


def test_draw_clahe_enhancement_comparison():
    """Validates diagnostic 4-panel CLAHE comparison generation."""
    from fundus2oct.data.quality_control import draw_clahe_enhancement_comparison
    from fundus2oct.data.auto_capture import create_demo_fundus

    fundus = create_demo_fundus(blur_level=0.0)
    grid = draw_clahe_enhancement_comparison(fundus, target_size=(256, 256))

    # 2x2 grid of (256, 256) tiles -> (512, 512, 3)
    assert grid.shape == (512, 512, 3)
    assert grid.dtype == np.uint8





