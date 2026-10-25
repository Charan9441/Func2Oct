"""
tests/test_volume_3d.py

Unit tests for 3D interactive volume visualization, 2D clinical multi-planar reconstruction,
and retinal surface topography extraction.
"""

import numpy as np
import pytest
import plotly.graph_objects as go

from fundus2oct.visualization.volume_3d import (
    create_interactive_3d_volume,
    create_orthogonal_2d_views,
    extract_3d_retinal_surfaces,
    PLOTLY_COLORMAPS,
)
from fundus2oct.visualization.oct_viewer import (
    render_clinical_bscan_2d,
    create_etdrs_macular_grid,
    COLORMAP_MAP,
    CLINICAL_COLORMAPS,
)
from fundus2oct.visualization.segmentation import segment_retinal_layers


@pytest.fixture
def sample_oct_volume():
    """Generates a realistic synthetic OCT volume for fast testing (32, 256, 256)."""
    np.random.seed(42)
    d, h, w = 32, 256, 256
    vol = np.zeros((d, h, w), dtype=np.uint8)

    # Simulate realistic retinal stratification across slices
    for z in range(d):
        slice_offset = int(np.sin(z / 5.0) * 4.0)
        # Background vitreous (low noise)
        vol[z, :60, :] = np.random.normal(15, 5, (60, w)).clip(0, 255)
        # ILM & RNFL (hyper-reflective)
        vol[z, 60 + slice_offset:75 + slice_offset, :] = np.random.normal(160, 20, (15, w)).clip(0, 255)
        # Nuclear layers (hypo-reflective)
        vol[z, 75 + slice_offset:120 + slice_offset, :] = np.random.normal(65, 15, (45, w)).clip(0, 255)
        # RPE (hyper-reflective band)
        vol[z, 120 + slice_offset:140 + slice_offset, :] = np.random.normal(210, 25, (20, w)).clip(0, 255)
        # Choroid / Sclera (gradual falloff)
        vol[z, 140 + slice_offset:, :] = np.random.normal(40, 15, (h - 140 - slice_offset, w)).clip(0, 255)

    return vol


class TestVolume3D:
    def test_extract_3d_retinal_surfaces(self, sample_oct_volume):
        ilm, rpe, thickness = extract_3d_retinal_surfaces(sample_oct_volume)
        d, h, w = sample_oct_volume.shape
        assert ilm.shape == (d, w)
        assert rpe.shape == (d, w)
        assert thickness.shape == (d, w)
        assert np.all(rpe >= ilm), "RPE surface should be deeper than or equal to ILM"
        assert np.mean(thickness) > 0, "Mean retinal thickness should be strictly positive"

    @pytest.mark.parametrize("mode", [
        "3D Multi-Planar Orthogonal Slices (MPR)",
        "3D Retinal Surface Topography (Foveal Pit)",
        "3D Volumetric Isosurface Cloud",
        "Complete 3D Composite",
    ])
    def test_create_interactive_3d_volume_modes(self, sample_oct_volume, mode):
        fig = create_interactive_3d_volume(
            volume=sample_oct_volume,
            mode=mode,
            slice_y=16,
            slice_x=128,
            slice_z=128,
            colormap="bone",
            depth_scale=2.0,
            show_fovea_marker=True,
            show_bounding_box=True,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0, f"Figure for mode {mode} must contain visual traces"

    def test_create_interactive_3d_volume_colormaps(self, sample_oct_volume):
        for cmap in ["bone", "inferno", "viridis", "gray", "plasma"]:
            fig = create_interactive_3d_volume(
                volume=sample_oct_volume,
                mode="3D Multi-Planar Orthogonal Slices (MPR)",
                colormap=cmap,
            )
            assert isinstance(fig, go.Figure)

    def test_create_orthogonal_2d_views(self, sample_oct_volume):
        montage = create_orthogonal_2d_views(
            volume=sample_oct_volume,
            slice_y=16,
            slice_x=128,
            slice_z=128,
            colormap="bone",
            show_crosshairs=True,
        )
        assert isinstance(montage, np.ndarray)
        assert montage.shape == (256, 768, 3), "Montage must be (256, 768, 3) 3-panel horizontal composite"
        assert montage.dtype == np.uint8


class TestOCTViewer2D:
    def test_render_clinical_bscan_raw(self, sample_oct_volume):
        bscan = sample_oct_volume[16]
        img = render_clinical_bscan_2d(
            bscan,
            colormap="bone",
            overlay_boundaries=False,
            show_scalebar=True,
            show_fovea_marker=True,
        )
        assert isinstance(img, np.ndarray)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8

    def test_render_clinical_bscan_with_boundaries(self, sample_oct_volume):
        bscan = sample_oct_volume[16]
        seg = segment_retinal_layers(bscan, generate_visualizations=True)
        img = render_clinical_bscan_2d(
            bscan,
            colormap="inferno",
            overlay_boundaries=True,
            segmentation=seg,
            show_scalebar=True,
            show_fovea_marker=True,
        )
        assert isinstance(img, np.ndarray)
        assert img.shape == (256, 256, 3)

    @pytest.mark.parametrize("cmap", ["bone", "inferno", "viridis", "gray", "inverted", "hot", "rainbow"])
    def test_render_clinical_bscan_all_colormaps(self, sample_oct_volume, cmap):
        bscan = sample_oct_volume[16]
        img = render_clinical_bscan_2d(bscan, colormap=cmap)
        assert img.shape == (256, 256, 3)

    def test_create_etdrs_macular_grid(self, sample_oct_volume):
        _, _, thickness_map = extract_3d_retinal_surfaces(sample_oct_volume)
        grid_img = create_etdrs_macular_grid(thickness_map)
        assert isinstance(grid_img, np.ndarray)
        assert grid_img.shape == (32, 256, 3)
