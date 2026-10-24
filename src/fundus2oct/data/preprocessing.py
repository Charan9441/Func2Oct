"""
src/fundus2oct/data/preprocessing.py

Medical image preprocessing pipeline for 2D Retinal Fundus photographs
and Optical Coherence Tomography (OCT) B-scans/volumes.
Guarantees consistent normalization, structural preservation, and type conversions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def safe_load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    expected_channels: int = 3,
) -> Image.Image:
    """
    Safely loads an image from a filepath, numpy array, or PIL Image.
    Raises ValueError with descriptive diagnostics if the image is corrupt or empty.
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy image array is empty.")
        if image_input.ndim == 2:
            pil_img = Image.fromarray(image_input)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 1:
                pil_img = Image.fromarray(image_input[:, :, 0])
            elif image_input.shape[0] in (1, 3, 4):  # CHW format
                pil_img = Image.fromarray(np.transpose(image_input, (1, 2, 0)))
            else:
                pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {image_input.shape}")
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Image file is 0 bytes (empty): {path}")
        try:
            pil_img = Image.open(path)
            pil_img.load()  # Force load to catch truncated or corrupted files
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable image file {path}: {e}") from e
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if expected_channels == 3:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
    elif expected_channels == 1:
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")

    return pil_img


def extract_retinal_fov_bbox(
    pil_img: Image.Image,
    threshold: int = 15,
) -> Tuple[int, int, int, int]:
    """
    Detects the circular retinal field-of-view (FOV) within a fundus photograph
    by thresholding the green channel and returning bounding box (left, upper, right, lower).
    Falls back to the entire image if no distinct boundary is identified.
    """
    img_gray = np.array(pil_img.convert("L"))
    mask = img_gray > threshold
    if not np.any(mask):
        return (0, 0, pil_img.width, pil_img.height)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Ensure valid bounding box
    if xmax <= xmin or ymax <= ymin:
        return (0, 0, pil_img.width, pil_img.height)

    return (int(xmin), int(ymin), int(xmax + 1), int(ymax + 1))


def preprocess_fundus(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    preserve_fov: bool = True,
    fov_threshold: int = 15,
    normalize: str = "imagenet",
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for retinal fundus photographs:
    1. Safe load & RGB conversion.
    2. Optional FOV bounding box crop to eliminate outer black margins.
    3. High-quality antialiased bicubic resize to target_size.
    4. Tensor conversion [0.0, 1.0].
    5. Normalization ('imagenet', 'bipolar', or 'unit').

    Returns tensor of shape: (3, H, W)
    """
    img = safe_load_image(image_input, expected_channels=3)

    if preserve_fov:
        bbox = extract_retinal_fov_bbox(img, threshold=fov_threshold)
        # Only crop if the bbox captures a meaningful region (> 50% of area)
        orig_area = img.width * img.height
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area > 0.4 * orig_area:
            img = img.crop(bbox)

    # Resize to target dimensions
    img_resized = img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BICUBIC)

    # Convert to Tensor (3, H, W) float in [0.0, 1.0]
    tensor = TF.to_tensor(img_resized)

    # Normalization
    if normalize == "imagenet":
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif normalize == "bipolar":
        tensor = (tensor * 2.0) - 1.0  # [0, 1] -> [-1, 1]
    elif normalize == "unit":
        pass  # already in [0, 1]
    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    return tensor


def preprocess_oct_bscan(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    normalize: str = "bipolar",
    invert: bool = False,
) -> torch.Tensor:
    """
    Preprocessing pipeline for OCT B-scans:
    1. Safe load & grayscale conversion (1 channel).
    2. High-quality bilinear resize to target_size.
    3. Tensor conversion [0.0, 1.0].
    4. Optional optical inversion (inverted scans).
    5. Normalization ('bipolar' [-1, 1] or 'unit' [0, 1]).

    Returns tensor of shape: (1, H, W)
    """
    img = safe_load_image(image_input, expected_channels=1)
    img_resized = img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR)

    tensor = TF.to_tensor(img_resized)  # (1, H, W) in [0.0, 1.0]

    if invert:
        tensor = 1.0 - tensor

    if normalize == "bipolar":
        tensor = (tensor * 2.0) - 1.0  # [-1.0, 1.0]
    elif normalize == "unit":
        pass
    else:
        raise ValueError(f"Unknown normalization mode for OCT: {normalize}")

    return tensor


def preprocess_oct_volume(
    volume_input: Union[str, Path, np.ndarray],
    target_shape: Tuple[int, int, int] = (16, 256, 256),  # (D, H, W)
    normalize: str = "bipolar",
) -> torch.Tensor:
    """
    Preprocesses 3D OCT volume stacks (Depth, Height, Width).
    Supports NumPy arrays (.npy, .npz) or multi-slice arrays.

    Returns tensor of shape: (1, D, H, W)
    """
    if isinstance(volume_input, (str, Path)):
        p = Path(volume_input)
        if not p.exists():
            raise FileNotFoundError(f"OCT volume file not found: {p}")
        if p.suffix.lower() == ".npz":
            data = np.load(p)
            vol = data[list(data.keys())[0]]
        elif p.suffix.lower() == ".npy":
            vol = np.load(p)
        else:
            raise ValueError(f"Unsupported volume file extension: {p.suffix}")
    elif isinstance(volume_input, np.ndarray):
        vol = volume_input
    else:
        raise TypeError(f"Unsupported volume input type: {type(volume_input)}")

    if vol.ndim == 3:
        vol = np.expand_dims(vol, axis=0)  # (1, D, H, W)
    elif vol.ndim != 4:
        raise ValueError(f"Expected 3D or 4D volume array, got shape {vol.shape}")

    tensor = torch.from_numpy(vol).float()

    # Min-max scaling to [0, 1]
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
    else:
        tensor = torch.zeros_like(tensor)

    # Resize volume using trilinear interpolation
    if tensor.shape[1:] != target_shape:
        # F.interpolate expects (B, C, D, H, W)
        tensor = tensor.unsqueeze(0)  # (1, 1, D, H, W)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # back to (1, D, H, W)

    if normalize == "bipolar":
        tensor = (tensor * 2.0) - 1.0

    return tensor


def denormalize_to_numpy(
    tensor: torch.Tensor,
    normalize_mode: str = "bipolar",
) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor back into a displayable NumPy array uint8 [0, 255].
    Handles single-image (C, H, W) or batched (B, C, H, W).
    """
    t = tensor.detach().cpu().clone()

    if t.ndim == 4:
        t = t[0]  # Take first sample if batched

    if normalize_mode == "bipolar":
        t = (t + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    elif normalize_mode == "imagenet":
        # Undo ImageNet normalization
        for c in range(min(3, t.shape[0])):
            t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]

    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.numpy() * 255.0).astype(np.uint8)

    if arr.shape[0] == 1:
        return arr[0]  # (H, W) grayscale
    elif arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))  # (H, W, 3) RGB

    return arr
