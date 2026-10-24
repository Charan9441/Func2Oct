"""
src/fundus2oct/data/preprocessing.py

Medical image preprocessing pipeline for 2D Retinal Fundus photographs
and Optical Coherence Tomography (OCT) B-scans/volumes.
Guarantees consistent normalization, square center-cropping, central B-scan extraction,
resizing to 256x256, and tensor type conversions.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
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
            elif image_input.shape[0] in (1, 3, 4) and image_input.shape[2] not in (1, 3, 4):  # CHW format
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


def center_crop_to_square(
    image_input: Union[Image.Image, np.ndarray, torch.Tensor],
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Crops the center square region min(H, W) x min(H, W) of an image.
    Preserves true anatomical aspect ratio before resizing to square dimensions (e.g. 256x256).

    Supports:
    - PIL Image (width, height)
    - NumPy Array: (H, W), (H, W, C), or (C, H, W)
    - PyTorch Tensor: (..., H, W)
    """
    if isinstance(image_input, Image.Image):
        w, h = image_input.size
        if w == h:
            return image_input
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return image_input.crop((left, top, right, bottom))

    elif isinstance(image_input, np.ndarray):
        if image_input.ndim < 2:
            raise ValueError(f"NumPy array must have at least 2 dimensions, got {image_input.ndim}")

        if image_input.ndim == 2:
            h, w = image_input.shape
            crop_size = min(h, w)
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            return image_input[top : top + crop_size, left : left + crop_size]

        elif image_input.ndim == 3:
            # Check if (C, H, W) with C in (1, 3, 4) and spatial dims larger
            if image_input.shape[0] in (1, 3, 4) and image_input.shape[0] < min(image_input.shape[1], image_input.shape[2]):
                c, h, w = image_input.shape
                crop_size = min(h, w)
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
                return image_input[:, top : top + crop_size, left : left + crop_size]
            else:
                # (H, W, C)
                h, w = image_input.shape[:2]
                crop_size = min(h, w)
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
                return image_input[top : top + crop_size, left : left + crop_size, :]
        else:
            # Higher-dimensional array with last two as spatial dims
            h, w = image_input.shape[-2:]
            crop_size = min(h, w)
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            return image_input[..., top : top + crop_size, left : left + crop_size]

    elif isinstance(image_input, torch.Tensor):
        if image_input.ndim < 2:
            raise ValueError(f"Tensor must have at least 2 spatial dimensions, got {image_input.ndim}")
        h, w = image_input.shape[-2], image_input.shape[-1]
        if h == w:
            return image_input
        crop_size = min(h, w)
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        return image_input[..., top : top + crop_size, left : left + crop_size]

    else:
        raise TypeError(f"Unsupported input type for center_crop_to_square: {type(image_input)}")


def extract_retinal_fov_bbox(
    pil_img: Image.Image,
    threshold: int = 15,
) -> Tuple[int, int, int, int]:
    """
    Detects the circular retinal field-of-view (FOV) within a fundus photograph
    by thresholding the grayscale intensity and returning bounding box (left, upper, right, lower).
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


def extract_central_bscans(
    volume_input: Union[np.ndarray, torch.Tensor, List[Union[Image.Image, np.ndarray]]],
    num_scans: int = 32,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    pad_mode: str = "edge",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extracts the central `num_scans` (default 32) B-scans from an OCT volume along the depth/slice axis.
    
    Behavior:
    - If volume depth D >= num_scans: Symmetrically slices the central `num_scans` slices.
    - If volume depth D < num_scans: Symmetrically pads edge slices to reach exactly `num_scans`.
    - If target_size is provided: Resizes each B-scan spatially to target_size (e.g. 256x256).

    Args:
        volume_input: 3D/4D numpy array (D, H, W) or (D, H, W, C), or torch.Tensor (D, H, W) or (C, D, H, W),
                      or list of 2D B-scan slices.
        num_scans: Target number of central B-scans to extract (default: 32).
        target_size: Spatial dimensions (H, W) to resize each B-scan to.
        pad_mode: Padding strategy if D < num_scans ('edge' repeats boundary slices, 'zeros' pads with 0).

    Returns:
        NumPy array (D=num_scans, H, W) or PyTorch tensor (D=num_scans, H, W).
    """
    if isinstance(volume_input, list):
        # List of individual B-scan images or arrays
        slices = []
        for s in volume_input:
            if isinstance(s, Image.Image):
                slices.append(np.array(s.convert("L")))
            elif isinstance(s, np.ndarray):
                slices.append(s if s.ndim == 2 else s[:, :, 0])
            else:
                slices.append(np.array(s))
        vol = np.stack(slices, axis=0)  # (D, H, W)
        is_tensor = False
    elif isinstance(volume_input, torch.Tensor):
        vol = volume_input
        is_tensor = True
    elif isinstance(volume_input, np.ndarray):
        vol = volume_input
        is_tensor = False
    else:
        raise TypeError(f"Unsupported volume input type: {type(volume_input)}")

    # Determine slice axis
    # Standard format: (D, H, W) or (1, D, H, W) or (C, D, H, W)
    if is_tensor:
        if vol.ndim == 3:
            # (D, H, W)
            d = vol.shape[0]
            if d >= num_scans:
                start = (d - num_scans) // 2
                out = vol[start : start + num_scans]
            else:
                pad_total = num_scans - d
                pad_pre = pad_total // 2
                pad_post = pad_total - pad_pre
                pre_slice = vol[:1].repeat(pad_pre, 1, 1) if pad_mode == "edge" else torch.zeros(pad_pre, vol.shape[1], vol.shape[2], dtype=vol.dtype, device=vol.device)
                post_slice = vol[-1:].repeat(pad_post, 1, 1) if pad_mode == "edge" else torch.zeros(pad_post, vol.shape[1], vol.shape[2], dtype=vol.dtype, device=vol.device)
                out = torch.cat([pre_slice, vol, post_slice], dim=0)

            # Spatial resize to target_size if needed
            if target_size is not None and (out.shape[1], out.shape[2]) != target_size:
                # F.interpolate on (D, 1, H, W)
                out = out.unsqueeze(1)
                out = torch.nn.functional.interpolate(
                    out.float(),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).to(vol.dtype)
            return out

        elif vol.ndim == 4:
            # Check if (C, D, H, W) or (D, H, W, C)
            if vol.shape[0] in (1, 3):
                # (C, D, H, W)
                c, d, h, w = vol.shape
                if d >= num_scans:
                    start = (d - num_scans) // 2
                    out = vol[:, start : start + num_scans]
                else:
                    pad_total = num_scans - d
                    pad_pre = pad_total // 2
                    pad_post = pad_total - pad_pre
                    pre = vol[:, :1].repeat(1, pad_pre, 1, 1) if pad_mode == "edge" else torch.zeros(c, pad_pre, h, w, dtype=vol.dtype, device=vol.device)
                    post = vol[:, -1:].repeat(1, pad_post, 1, 1) if pad_mode == "edge" else torch.zeros(c, pad_post, h, w, dtype=vol.dtype, device=vol.device)
                    out = torch.cat([pre, vol, post], dim=1)

                if target_size is not None and (out.shape[2], out.shape[3]) != target_size:
                    out = torch.nn.functional.interpolate(
                        out.float(),
                        size=(num_scans, target_size[0], target_size[1]),
                        mode="trilinear",
                        align_corners=False,
                    ).to(vol.dtype)
                return out
            else:
                # (D, H, W, C) -> transpose to (C, D, H, W)
                transposed = vol.permute(3, 0, 1, 2)
                res = extract_central_bscans(transposed, num_scans=num_scans, target_size=target_size, pad_mode=pad_mode)
                return res.permute(1, 2, 3, 0)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {vol.shape}")

    else:
        # NumPy array
        if vol.ndim == 3:
            # (D, H, W)
            d = vol.shape[0]
            if d >= num_scans:
                start = (d - num_scans) // 2
                out = vol[start : start + num_scans]
            else:
                pad_total = num_scans - d
                pad_pre = pad_total // 2
                pad_post = pad_total - pad_pre
                pre_slice = np.repeat(vol[:1], pad_pre, axis=0) if pad_mode == "edge" else np.zeros((pad_pre, vol.shape[1], vol.shape[2]), dtype=vol.dtype)
                post_slice = np.repeat(vol[-1:], pad_post, axis=0) if pad_mode == "edge" else np.zeros((pad_post, vol.shape[1], vol.shape[2]), dtype=vol.dtype)
                out = np.concatenate([pre_slice, vol, post_slice], axis=0)

            # Spatial resize to target_size if needed
            if target_size is not None and (out.shape[1], out.shape[2]) != target_size:
                resized_slices = []
                for idx in range(num_scans):
                    slice_img = Image.fromarray(out[idx])
                    slice_res = slice_img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR)
                    resized_slices.append(np.array(slice_res))
                out = np.stack(resized_slices, axis=0)
            return out

        elif vol.ndim == 4:
            # (D, H, W, C)
            d = vol.shape[0]
            if d >= num_scans:
                start = (d - num_scans) // 2
                out = vol[start : start + num_scans]
            else:
                pad_total = num_scans - d
                pad_pre = pad_total // 2
                pad_post = pad_total - pad_pre
                pre_slice = np.repeat(vol[:1], pad_pre, axis=0) if pad_mode == "edge" else np.zeros((pad_pre, vol.shape[1], vol.shape[2], vol.shape[3]), dtype=vol.dtype)
                post_slice = np.repeat(vol[-1:], pad_post, axis=0) if pad_mode == "edge" else np.zeros((pad_post, vol.shape[1], vol.shape[2], vol.shape[3]), dtype=vol.dtype)
                out = np.concatenate([pre_slice, vol, post_slice], axis=0)

            if target_size is not None and (out.shape[1], out.shape[2]) != target_size:
                resized_slices = []
                for idx in range(num_scans):
                    slice_img = Image.fromarray(out[idx])
                    slice_res = slice_img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BILINEAR)
                    resized_slices.append(np.array(slice_res))
                out = np.stack(resized_slices, axis=0)
            return out
        else:
            raise ValueError(f"Expected 3D or 4D numpy array, got shape {vol.shape}")


def preprocess_fundus(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    center_crop: bool = True,
    preserve_fov: bool = False,
    fov_threshold: int = 15,
    normalize: str = "imagenet",
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for retinal fundus photographs:
    1. Safe load & RGB conversion (3 channels).
    2. Center-crop to square (min(H, W) x min(H, W)) to avoid aspect ratio distortion.
    3. Optional circular FOV margin refinement.
    4. High-quality antialiased bicubic resize to target_size (256, 256).
    5. Convert to PyTorch float32 tensor in range [0.0, 1.0].
    6. Normalize ('imagenet', 'bipolar' [-1, 1], or 'unit' [0, 1]).

    Returns tensor of shape: (3, target_size[0], target_size[1]) -> (3, 256, 256)
    """
    img = safe_load_image(image_input, expected_channels=3)

    # 1. Center crop to square
    if center_crop:
        img = center_crop_to_square(img)

    # 2. Optional FOV bounding box crop
    if preserve_fov:
        bbox = extract_retinal_fov_bbox(img, threshold=fov_threshold)
        orig_area = img.width * img.height
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area > 0.4 * orig_area:
            img = img.crop(bbox)
            if center_crop:
                img = center_crop_to_square(img)

    # 3. High quality resize to target_size (width, height)
    img_resized = img.resize((target_size[1], target_size[0]), resample=Image.Resampling.BICUBIC)

    # 4. Convert to Tensor (3, H, W) float in [0.0, 1.0]
    tensor = TF.to_tensor(img_resized)

    # 5. Normalization
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
    Preprocessing pipeline for single 2D OCT B-scans:
    1. Safe load & grayscale conversion (1 channel).
    2. High-quality bilinear resize to target_size (256, 256).
    3. Tensor conversion [0.0, 1.0].
    4. Optional optical inversion (inverted scans).
    5. Normalization ('bipolar' [-1, 1] or 'unit' [0, 1]).

    Returns tensor of shape: (1, target_size[0], target_size[1]) -> (1, 256, 256)
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
    volume_input: Union[str, Path, np.ndarray, torch.Tensor, List[Union[Image.Image, np.ndarray]]],
    target_shape: Tuple[int, int, int] = (32, 256, 256),  # (D, H, W)
    num_scans: int = 32,
    normalize: str = "bipolar",
) -> torch.Tensor:
    """
    Preprocesses 3D OCT volume stacks:
    1. Loads from file (.npz, .npy) or accepts array / tensor / list of B-scans.
    2. Extracts the central `num_scans` (default: 32) B-scans.
    3. Resizes each B-scan spatially to target_shape[1:] -> (256, 256).
    4. Normalizes intensity to 'bipolar' [-1, 1] or 'unit' [0, 1].

    Returns tensor of shape: (1, num_scans, target_shape[1], target_shape[2]) -> (1, 32, 256, 256)
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
    else:
        vol = volume_input

    # Extract central B-scans and resize spatially to (256, 256)
    spatial_size = (target_shape[1], target_shape[2]) if len(target_shape) == 3 else (256, 256)
    target_d = num_scans or (target_shape[0] if len(target_shape) == 3 else 32)

    extracted = extract_central_bscans(vol, num_scans=target_d, target_size=spatial_size)

    if isinstance(extracted, np.ndarray):
        tensor = torch.from_numpy(extracted).float()
    else:
        tensor = extracted.float()

    # Ensure shape has channel dimension: (1, D, H, W)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # (1, D, H, W)
    elif tensor.ndim == 4 and tensor.shape[0] != 1 and tensor.shape[1] == target_d:
        # e.g. (D, H, W, C) -> handled earlier or transpose
        pass

    # Min-max scaling to [0, 1] if not already bounded
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
    else:
        tensor = torch.zeros_like(tensor)

    if normalize == "bipolar":
        tensor = (tensor * 2.0) - 1.0  # [0, 1] -> [-1, 1]

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
