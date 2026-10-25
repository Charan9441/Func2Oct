import numpy as np
import plotly.graph_objects as go
import cv2
from typing import Dict, List, Optional, Tuple, Union

# Clinical Colormap mapping for Plotly
PLOTLY_COLORMAPS = {
    "bone": [
        [0.0, "rgb(0, 0, 0)"],
        [0.2, "rgb(20, 28, 48)"],
        [0.4, "rgb(60, 85, 120)"],
        [0.7, "rgb(150, 180, 205)"],
        [1.0, "rgb(255, 255, 255)"],
    ],
    "inferno": "inferno",
    "viridis": "viridis",
    "gray": "gray",
    "plasma": "plasma",
    "hot": "hot",
}

def extract_3d_retinal_surfaces(
    volume: np.ndarray,
    smooth_sigma: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 3D Inner Limiting Membrane (ILM) and Retinal Pigment Epithelium (RPE) surfaces.
    
    Args:
        volume: 3D numpy array (D, H, W) of uint8 OCT data.
        smooth_sigma: Gaussian smoothing parameter.
        
    Returns:
        Tuple of (ilm_surface, rpe_surface, thickness_map) as 2D arrays of shape (D, W).
    """
    d, h, w = volume.shape
    ilm_grid = np.zeros((d, w), dtype=np.float32)
    rpe_grid = np.zeros((d, w), dtype=np.float32)

    for i in range(d):
        bscan = volume[i]
        # Smooth B-scan to reduce speckle noise
        smoothed = cv2.GaussianBlur(bscan, (9, 9), 1.5)
        # Vertical gradient (dI/dz) for boundary detection
        grad_z = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        for x in range(w):
            col = bscan[:, x]
            col_grad = grad_z[:, x]
            
            # ILM is first strong positive vertical gradient from vitreous into retina
            vitreous_margin = int(h * 0.12)
            ilm_cands = np.where(col_grad[vitreous_margin:int(h * 0.6)] > 8.0)[0]
            if len(ilm_cands) > 0:
                ilm_z = vitreous_margin + ilm_cands[0]
            else:
                ilm_z = int(h * 0.28)
            
            # RPE is the deepest intense hyper-reflective band
            search_start = max(ilm_z + 15, int(h * 0.40))
            search_end = min(h - 5, int(h * 0.85))
            if search_end > search_start:
                sub_col = col[search_start:search_end]
                rpe_offset = np.argmax(sub_col)
                rpe_z = search_start + rpe_offset
            else:
                rpe_z = min(h - 5, ilm_z + 45)

            ilm_grid[i, x] = float(ilm_z)
            rpe_grid[i, x] = float(rpe_z)

    # Spatial smoothing across volume (D, W)
    ilm_grid = cv2.GaussianBlur(ilm_grid, (5, 5), smooth_sigma)
    rpe_grid = cv2.GaussianBlur(rpe_grid, (5, 5), smooth_sigma)
    thickness_grid = np.clip(rpe_grid - ilm_grid, 10.0, 120.0)

    return ilm_grid, rpe_grid, thickness_grid


def create_interactive_3d_volume(
    volume: np.ndarray,
    mode: str = "3D Multi-Planar Orthogonal Slices",
    slice_y: int = 16,
    slice_x: int = 128,
    slice_z: int = 110,
    colormap: str = "bone",
    depth_scale: float = 0.60,
    downsample_factor: int = 2,
    show_bounding_box: bool = True,
    show_fovea_marker: bool = True,
) -> go.Figure:
    """Create a high-performance interactive 3D WebGL Plotly visualization of an OCT volume.

    Args:
        volume: 3D OCT volume (D, H, W) uint8.
        mode: 3D Visualization mode:
            - '3D Multi-Planar Orthogonal Slices': Interactive 3-plane orthoslices (MPR).
            - '3D Retinal Surface Topography (Foveal Pit)': 3D ILM/RPE elevation topography.
            - '3D Volumetric Isosurface Cloud': 3D cellular hyper-reflectivity shells.
            - 'Complete 3D Composite': Combined orthoslices and surface mesh.
        slice_y: Active B-scan slice index (0 to D-1).
        slice_x: Active sagittal cutplane position (0 to W-1).
        slice_z: Active axial en-face C-scan depth (0 to H-1).
        colormap: Color palette ('bone', 'inferno', 'viridis', 'gray', 'plasma').
        depth_scale: Z-axis physical stretch factor.
        downsample_factor: Subsampling factor for buttery smooth 60fps browser rendering.
        show_bounding_box: Whether to draw volumetric wireframe box.

    Returns:
        plotly.graph_objects.Figure ready for Gradio gr.Plot or browser display.
    """
    d, h, w = volume.shape
    slice_y = int(np.clip(slice_y, 0, d - 1))
    slice_x = int(np.clip(slice_x, 0, w - 1))
    slice_z = int(np.clip(slice_z, 0, h - 1))

    cmap_plotly = PLOTLY_COLORMAPS.get(colormap.lower(), PLOTLY_COLORMAPS["bone"])
    fig = go.Figure()

    # Physical coordinate dimensions:
    # X: Nasal-Temporal width [0, W] (e.g. 6.0 mm)
    # Y: Superior-Inferior scan axis [0, D] (e.g. 6.0 mm)
    # Z: Axial depth [0, H] (e.g. 2.0 mm)
    y_coords = np.linspace(0, w, d) # Scale Y to square field-of-view

    if mode in ("3D Multi-Planar Orthogonal Slices", "Complete 3D Composite"):
        # 1. Transverse B-Scan Slice (X-Z plane at Y = slice_y)
        bscan = volume[slice_y]
        bscan_res = cv2.resize(bscan, (w // downsample_factor, h // downsample_factor))
        xb = np.linspace(0, w, bscan_res.shape[1])
        zb = np.linspace(h * depth_scale, 0, bscan_res.shape[0]) # Invert Z so vitreous is top
        XB, ZB = np.meshgrid(xb, zb)
        YB = np.full_like(XB, y_coords[slice_y])

        fig.add_trace(go.Surface(
            x=XB,
            y=YB,
            z=ZB,
            surfacecolor=bscan_res,
            colorscale=cmap_plotly,
            showscale=False,
            name=f"Transverse B-Scan #{slice_y:02d}",
            hoverinfo="x+y+z",
            opacity=0.95,
        ))

        # 2. Sagittal B-Scan Cutplane (Y-Z plane at X = slice_x)
        sagittal = volume[:, :, slice_x] # (D, H)
        sagittal_res = cv2.resize(sagittal, (h // downsample_factor, d))
        ys = y_coords
        zs = np.linspace(h * depth_scale, 0, sagittal_res.shape[1])
        YS, ZS = np.meshgrid(ys, zs)
        XS = np.full_like(YS, slice_x)

        fig.add_trace(go.Surface(
            x=XS,
            y=YS,
            z=ZS,
            surfacecolor=sagittal_res.T,
            colorscale=cmap_plotly,
            showscale=False,
            name=f"Sagittal Cutplane (X={slice_x})",
            hoverinfo="x+y+z",
            opacity=0.88,
        ))

        # 3. Axial En-Face C-Scan Plane (X-Y plane at depth Z = slice_z)
        cscan = volume[:, slice_z, :] # (D, W)
        cscan_res = cv2.resize(cscan, (w // downsample_factor, d))
        xc = np.linspace(0, w, cscan_res.shape[1])
        yc = y_coords
        XC, YC = np.meshgrid(xc, yc)
        z_phys = (h - slice_z) * depth_scale
        ZC = np.full_like(XC, z_phys)

        fig.add_trace(go.Surface(
            x=XC,
            y=YC,
            z=ZC,
            surfacecolor=cscan_res,
            colorscale=cmap_plotly,
            showscale=False,
            name=f"Axial En-Face C-Scan (Z={slice_z})",
            hoverinfo="x+y+z",
            opacity=0.82,
        ))

    if mode in ("3D Retinal Surface Topography (Foveal Pit)", "Complete 3D Composite"):
        # 3D ILM & RPE Elevation Maps
        ilm_surf, rpe_surf, thick_map = extract_3d_retinal_surfaces(volume)
        
        # Subsample for rendering
        ilm_sub = cv2.resize(ilm_surf, (w // downsample_factor, d))
        thick_sub = cv2.resize(thick_map, (w // downsample_factor, d))
        
        xt = np.linspace(0, w, ilm_sub.shape[1])
        yt = y_coords
        XT, YT = np.meshgrid(xt, yt)
        ZT_ILM = (h - ilm_sub) * depth_scale # Invert so vitreous is top

        # ILM Surface showing Foveal Pit
        fig.add_trace(go.Surface(
            x=XT,
            y=YT,
            z=ZT_ILM,
            surfacecolor=thick_sub * 2.5, # Scaled to approx micrometers (~250 µm)
            colorscale="Viridis",
            colorbar=dict(
                title="Retinal Thickness (µm)",
                thickness=14,
                len=0.6,
                x=1.02,
            ),
            showscale=True,
            name="3D ILM Retinal Surface (Foveal Depression)",
            opacity=0.90,
            contours_z=dict(show=True, usecolormap=True, project_z=True),
        ))

    elif mode == "3D Volumetric Isosurface Cloud":
        # Volumetric Isosurface showing hyper-reflective cellular layers
        step = max(1, downsample_factor)
        vol_sub = volume[::1, ::step, ::step]
        vd, vh, vw = vol_sub.shape
        
        x_iso = np.tile(np.linspace(0, w, vw), vd * vh)
        y_iso = np.tile(np.repeat(y_coords, vw), vh)
        z_iso = np.repeat(np.linspace(h * depth_scale, 0, vh), vd * vw)

        fig.add_trace(go.Isosurface(
            x=x_iso,
            y=y_iso,
            z=z_iso,
            value=vol_sub.flatten(),
            isomin=110,
            isomax=240,
            surface_count=3,
            colorscale="Inferno",
            opacity=0.35,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="Hyper-Reflective Retinal Layers (RPE / RNFL / IS-OS)",
        ))

    # Foveal Depression Center 3D Marker
    if show_fovea_marker:
        fovea_x = w / 2.0
        fovea_y = y_coords[d // 2]
        fovea_z = h * depth_scale * 0.70
        fig.add_trace(go.Scatter3d(
            x=[fovea_x],
            y=[fovea_y],
            z=[fovea_z],
            mode="markers+text",
            marker=dict(size=5, color="#f59e0b", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            text=["Foveal Center (FAZ)"],
            textposition="top center",
            textfont=dict(color="#fbbf24", size=11),
            name="Foveal Center (FAZ)",
            showlegend=True,
        ))

    # Volumetric Bounding Box Wireframe
    if show_bounding_box:
        max_z = h * depth_scale
        box_lines_x = [0, w, w, 0, 0,   0, w, w, 0, 0,   0, 0,  w, w,  w, w,  0, 0]
        box_lines_y = [0, 0, w, w, 0,   0, 0, w, w, 0,   0, 0,  0, 0,  w, w,  w, w]
        box_lines_z = [0, 0, 0, 0, 0,   max_z, max_z, max_z, max_z, max_z,  0, max_z, 0, max_z, 0, max_z, 0, max_z]

        fig.add_trace(go.Scatter3d(
            x=box_lines_x,
            y=box_lines_y,
            z=box_lines_z,
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=2),
            hoverinfo="none",
            showlegend=False,
            name="Volume Boundary",
        ))

    # Layout styling
    fig.update_layout(
        title=dict(
            text=f"<b>Interactive 3D Retinal Studio</b> — {mode} [B-Scan #{slice_y:02d}]",
            font=dict(color="#f8fafc", size=15),
            x=0.03,
            y=0.96,
        ),
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                title="Nasal-Temporal (X) [px]",
                backgroundcolor="rgb(15, 23, 42)",
                gridcolor="rgba(51, 65, 85, 0.6)",
                showbackground=True,
            ),
            yaxis=dict(
                title="Superior-Inferior (Y) [px]",
                backgroundcolor="rgb(15, 23, 42)",
                gridcolor="rgba(51, 65, 85, 0.6)",
                showbackground=True,
            ),
            zaxis=dict(
                title="Retinal Depth (Z) [Optical]",
                backgroundcolor="rgb(15, 23, 42)",
                gridcolor="rgba(51, 65, 85, 0.6)",
                showbackground=True,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=depth_scale),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.1),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=35),
        paper_bgcolor="rgb(15, 23, 42)",
        plot_bgcolor="rgb(15, 23, 42)",
    )

    return fig


def create_orthogonal_2d_views(
    volume: np.ndarray,
    slice_y: int = 16,
    slice_x: int = 128,
    slice_z: int = 110,
    colormap: str = "bone",
    show_crosshairs: bool = True,
) -> np.ndarray:
    """Render a synchronized 3-panel 2D Multi-Planar Orthogonal montage (Axial En-Face, Transverse, Sagittal)."""
    d, h, w = volume.shape
    slice_y = int(np.clip(slice_y, 0, d - 1))
    slice_x = int(np.clip(slice_x, 0, w - 1))
    slice_z = int(np.clip(slice_z, 0, h - 1))
    panel_size = 256

    # 1. Transverse B-Scan (X-Z at slice_y)
    bscan = volume[slice_y].copy()
    bscan_sq = cv2.resize(bscan, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    # 2. Sagittal B-Scan (Y-Z at slice_x)
    sagittal = volume[:, :, slice_x].copy() # (D, H)
    sagittal_sq = cv2.resize(sagittal.T, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    # 3. Axial En-Face C-Scan (X-Y at slice_z)
    cscan = volume[:, slice_z, :].copy() # (D, W)
    cscan_sq = cv2.resize(cscan, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)

    # Colormap application
    cmap_lower = colormap.lower()
    if cmap_lower == "inferno":
        bscan_rgb = cv2.applyColorMap(bscan_sq, cv2.COLORMAP_INFERNO)
        sagittal_rgb = cv2.applyColorMap(sagittal_sq, cv2.COLORMAP_INFERNO)
        cscan_rgb = cv2.applyColorMap(cscan_sq, cv2.COLORMAP_INFERNO)
    elif cmap_lower == "viridis":
        bscan_rgb = cv2.applyColorMap(bscan_sq, cv2.COLORMAP_VIRIDIS)
        sagittal_rgb = cv2.applyColorMap(sagittal_sq, cv2.COLORMAP_VIRIDIS)
        cscan_rgb = cv2.applyColorMap(cscan_sq, cv2.COLORMAP_VIRIDIS)
    elif cmap_lower == "bone":
        bscan_rgb = cv2.applyColorMap(bscan_sq, cv2.COLORMAP_BONE)
        sagittal_rgb = cv2.applyColorMap(sagittal_sq, cv2.COLORMAP_BONE)
        cscan_rgb = cv2.applyColorMap(cscan_sq, cv2.COLORMAP_BONE)
    else:
        bscan_rgb = cv2.cvtColor(bscan_sq, cv2.COLOR_GRAY2RGB)
        sagittal_rgb = cv2.cvtColor(sagittal_sq, cv2.COLOR_GRAY2RGB)
        cscan_rgb = cv2.cvtColor(cscan_sq, cv2.COLOR_GRAY2RGB)

    # Convert to RGB if OpenCV colormap was used
    if cmap_lower in ("inferno", "viridis", "bone"):
        bscan_rgb = cv2.cvtColor(bscan_rgb, cv2.COLOR_BGR2RGB)
        sagittal_rgb = cv2.cvtColor(sagittal_rgb, cv2.COLOR_BGR2RGB)
        cscan_rgb = cv2.cvtColor(cscan_rgb, cv2.COLOR_BGR2RGB)

    # Draw Synchronized Crosshairs
    if show_crosshairs:
        crosshair_x = int((slice_x / max(1, w - 1)) * (panel_size - 1))
        crosshair_y = int((slice_y / max(1, d - 1)) * (panel_size - 1))
        crosshair_z = int((slice_z / max(1, h - 1)) * (panel_size - 1))

        # On Transverse B-Scan (X-Z): vertical line at X, horizontal line at Z
        cv2.line(bscan_rgb, (crosshair_x, 0), (crosshair_x, panel_size - 1), (0, 220, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(bscan_rgb, (0, crosshair_z), (panel_size - 1, crosshair_z), (255, 120, 0), 1, lineType=cv2.LINE_AA)

        # On Sagittal B-Scan (Y-Z): vertical line at Y, horizontal line at Z
        cv2.line(sagittal_rgb, (crosshair_y, 0), (crosshair_y, panel_size - 1), (0, 255, 100), 1, lineType=cv2.LINE_AA)
        cv2.line(sagittal_rgb, (0, crosshair_z), (panel_size - 1, crosshair_z), (255, 120, 0), 1, lineType=cv2.LINE_AA)

        # On Axial C-Scan (X-Y): horizontal line at Y, vertical line at X
        cv2.line(cscan_rgb, (0, crosshair_y), (panel_size - 1, crosshair_y), (0, 255, 100), 1, lineType=cv2.LINE_AA)
        cv2.line(cscan_rgb, (crosshair_x, 0), (crosshair_x, panel_size - 1), (0, 220, 255), 1, lineType=cv2.LINE_AA)

    # Add Titles
    _add_header(cscan_rgb, f"Axial En-Face C-Scan (Z={slice_z:03d})")
    _add_header(bscan_rgb, f"Transverse B-Scan (Slice #{slice_y:02d})")
    _add_header(sagittal_rgb, f"Sagittal B-Scan (X={slice_x:03d})")

    # Stack horizontally into a 3-panel orthogonal view
    montage = np.hstack([cscan_rgb, bscan_rgb, sagittal_rgb])
    return montage


def _add_header(img: np.ndarray, text: str) -> None:
    _, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 24), (15, 23, 42), -1)
    cv2.putText(img, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1, cv2.LINE_AA)
