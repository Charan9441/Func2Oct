"""
tests/test_segmentation.py

Unit and validation tests for 8-layer retinal OCT segmentation and visualization:
- Verifies detection of all 8 clinical boundaries (ILM, RNFL-GCL, IPL-INL, INL-OPL, OPL-ONL, ELM/ISOS, RPE, BM).
- Verifies non-crossing anatomical boundary constraints: ILM <= RNFL <= ... <= RPE <= BM.
- Confirms 8-layer region mask contains valid pixel labels in range 0..8.
- Validates retinal thickness profiling and diagnostic report rendering.
- Verifies 3D volume segmentation and export to outputs/segmentation_masks/.
"""

import os
import sys
from pathlib import Path

# Ensure repo root and src/ are in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pytest
import torch

from fundus2oct.visualization.segmentation import (
    BOUNDARY_NAMES,
    LAYER_NAMES,
    SegmentationResult,
    segment_retinal_layers,
    draw_segmentation_overlay,
    create_colored_mask,
    create_diagnostic_report_figure,
    segment_and_save_oct_volume,
)
from generate_oct import synthesize_realistic_3d_oct_volume


def test_segment_retinal_layers_shape_and_boundaries():
    """Verifies that 8-layer segmentation identifies all 8 boundaries and non-crossing constraints."""
    cond = torch.randn(512)
    vol = synthesize_realistic_3d_oct_volume(cond, class_name="CONTROL", num_slices=1, h=256, w=256)
    bscan = vol[0]

    result = segment_retinal_layers(bscan, generate_visualizations=True)

    # 1. Check all 8 boundaries are detected
    assert len(result.boundaries) == 8
    for name in BOUNDARY_NAMES:
        assert name in result.boundaries
        assert len(result.boundaries[name]) == 256
        assert np.isfinite(result.boundaries[name]).all()

    # 2. Check strict non-crossing anatomical constraints
    ilm = result.boundaries["ILM"]
    rnfl = result.boundaries["RNFL_GCL"]
    ipl = result.boundaries["IPL_INL"]
    inl = result.boundaries["INL_OPL"]
    opl = result.boundaries["OPL_ONL"]
    elm = result.boundaries["ELM_ISOS"]
    rpe = result.boundaries["RPE"]
    bm = result.boundaries["BM"]

    assert np.all(ilm <= rnfl)
    assert np.all(rnfl <= ipl)
    assert np.all(ipl <= inl)
    assert np.all(inl <= opl)
    assert np.all(opl <= elm)
    assert np.all(elm <= rpe)
    assert np.all(rpe <= bm)

    # 3. Check mask dimensions and classes
    assert result.mask.shape == (256, 256)
    assert result.mask.dtype == np.uint8
    unique_labels = np.unique(result.mask)
    assert np.all(unique_labels >= 0)
    assert np.all(unique_labels <= 8)


def test_retinal_thickness_metrics():
    """Verifies retinal layer thickness calculation and foveal pit detection."""
    cond = torch.randn(512)
    vol = synthesize_realistic_3d_oct_volume(cond, class_name="MD", num_slices=1, h=256, w=256)
    bscan = vol[0]

    result = segment_retinal_layers(bscan, generate_visualizations=False)

    assert result.mean_retinal_thickness > 15.0
    assert result.foveal_min_thickness > 8.0
    assert result.foveal_min_thickness <= float(np.max(result.total_retinal_thickness))

    for layer_name in ["RNFL", "GCL_IPL", "INL", "OPL", "ONL", "IS_OS", "RPE"]:
        assert layer_name in result.layer_thicknesses
        t = result.layer_thicknesses[layer_name]
        assert len(t) == 256
        assert np.all(t >= 0.0)


def test_segmentation_visualizations_and_report(tmp_path):
    """Verifies diagnostic report figure generation and RGB color-coded mask construction."""
    cond = torch.randn(512)
    vol = synthesize_realistic_3d_oct_volume(cond, class_name="CONTROL", num_slices=1, h=256, w=256)
    bscan = vol[0]

    result = segment_retinal_layers(bscan, generate_visualizations=True)

    assert result.overlay_image is not None
    assert result.overlay_image.shape == (256, 256, 3)

    assert result.colored_mask is not None
    assert result.colored_mask.shape == (256, 256, 3)

    report_fig = create_diagnostic_report_figure(bscan, result, title="Test Report")
    assert report_fig.ndim == 3
    assert report_fig.shape[2] == 3
    assert report_fig.shape[0] > 500
    assert report_fig.shape[1] > 500


def test_volume_segmentation_save(tmp_path):
    """Verifies segmenting a 3D volume stack and saving structured outputs."""
    cond = torch.randn(512)
    vol = synthesize_realistic_3d_oct_volume(cond, class_name="MD", num_slices=4, h=256, w=256)

    out = segment_and_save_oct_volume(
        oct_volume=vol,
        output_dir=tmp_path,
        prefix="test_vol",
        save_all_slices=True,
        save_diagnostic_report=True,
        export_3d_mask=True,
    )

    assert out["num_slices"] == 4
    assert Path(out["diagnostic_report"]).exists()
    assert Path(out["thickness_csv"]).exists()
    assert Path(out["mask_npy"]).exists()
    assert (tmp_path / "overlays").exists()
    assert (tmp_path / "masks").exists()
