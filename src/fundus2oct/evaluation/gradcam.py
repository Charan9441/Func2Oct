"""
src/fundus2oct/evaluation/gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) Interpretability Module.
Provides visual explanations for decisions made by convolutional retinal disease classifiers
(FundusClassifier Model A and MultimodalClassifier Model B).

References:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundus2oct.data.preprocessing import denormalize_to_numpy


@dataclass
class GradCAMResult:
    """
    Encapsulates the output of a Grad-CAM computation.

    Attributes:
        heatmap: 2D NumPy array (H, W) in range [0.0, 1.0] representing normalized feature importance.
        raw_heatmap: Unnormalized 2D NumPy array before min-max scaling.
        target_class: Integer index of the class explained.
        predicted_class: Integer index of the model's top predicted class.
        class_probabilities: 1D NumPy array of softmax class probabilities.
        target_layer_name: String identifier of the convolutional layer evaluated.
        modality_explained: Input modality targeted ("fundus", "oct", or "unified").
        metadata: Additional diagnostic context.
    """
    heatmap: np.ndarray
    raw_heatmap: np.ndarray
    target_class: int
    predicted_class: int
    class_probabilities: np.ndarray
    target_layer_name: str
    modality_explained: str = "fundus"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def overlay(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
        normalize_mode: str = "imagenet",
    ) -> np.ndarray:
        """
        Overlays the Grad-CAM heatmap onto the original image.

        Args:
            image: Original input image (PIL Image, uint8 ndarray, or normalized tensor).
            alpha: Weight of the heatmap overlay (0.0 to 1.0).
            colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).
            normalize_mode: Denormalization mode if tensor is provided ("imagenet" or "bipolar").

        Returns:
            RGB NumPy uint8 array of shape (H, W, 3).
        """
        return overlay_heatmap_on_image(
            image=image,
            heatmap=self.heatmap,
            alpha=alpha,
            colormap=colormap,
            normalize_mode=normalize_mode,
        )

    def save(
        self,
        output_dir: Union[str, Path],
        filename_prefix: str = "gradcam",
        original_image: Optional[Union[np.ndarray, Image.Image, torch.Tensor]] = None,
        alpha: float = 0.45,
    ) -> Dict[str, Path]:
        """Saves heatmap and optional overlay to disk."""
        return save_gradcam_artifacts(
            result=self,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            original_image=original_image,
            alpha=alpha,
        )


def find_target_layer(
    model: nn.Module,
    target_layer: Optional[Union[nn.Module, str]] = None,
    modality: str = "fundus",
) -> Tuple[nn.Module, str]:
    """
    Identifies or resolves the target convolutional layer for Grad-CAM.

    Args:
        model: PyTorch classifier module.
        target_layer: Explicit layer module or named submodule string (e.g. "layer4", "fundus_l4").
        modality: Modality to target for multimodal classifiers ("fundus" or "oct").

    Returns:
        Tuple of (target_module, target_layer_name).

    Raises:
        ValueError: If specified layer is missing or no suitable conv layer can be located.
    """
    if target_layer is not None:
        if isinstance(target_layer, nn.Module):
            # Find name of the explicit module
            name = "custom_target_module"
            for n, m in model.named_modules():
                if m is target_layer:
                    name = n
                    break
            return target_layer, name
        elif isinstance(target_layer, str):
            try:
                mod = model.get_submodule(target_layer)
                return mod, target_layer
            except (AttributeError, KeyError):
                # Fallback recursive attribute traversal
                curr = model
                for part in target_layer.split("."):
                    if not hasattr(curr, part):
                        raise ValueError(f"Target layer '{target_layer}' not found in model hierarchy.")
                    curr = getattr(curr, part)
                if not isinstance(curr, nn.Module):
                    raise ValueError(f"Resolved attribute '{target_layer}' is not an nn.Module.")
                return curr, target_layer

    # Model-specific smart defaults
    class_name = model.__class__.__name__

    if class_name == "FundusClassifier":
        if hasattr(model, "layer4"):
            return model.layer4, "layer4"

    elif class_name == "MultimodalClassifier":
        if modality.lower() == "oct":
            if hasattr(model, "oct_l4"):
                return model.oct_l4, "oct_l4"
        else:
            if hasattr(model, "fundus_l4"):
                return model.fundus_l4, "fundus_l4"

    # Generic automatic discovery: find the last Conv2d or container with Conv2d
    last_conv_name: Optional[str] = None
    last_conv_module: Optional[nn.Module] = None

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            last_conv_name = name
            last_conv_module = module

    if last_conv_module is not None and last_conv_name is not None:
        return last_conv_module, last_conv_name

    raise ValueError(
        f"Unable to find a convolutional layer for Grad-CAM in model '{class_name}'. "
        "Please provide an explicit 'target_layer'."
    )


class GradCAM:
    """
    Grad-CAM Explainer for 2D Retinal Fundus & Multimodal Classifiers.

    Computes class-discriminative localization maps highlighting the anatomical
    regions of the retina that most influence the classifier's diagnosis.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        modality: str = "fundus",
    ):
        """
        Args:
            model: PyTorch classifier (e.g. FundusClassifier or MultimodalClassifier).
            target_layer: Optional explicit target module or module name.
            modality: Modality being explained ("fundus" or "oct").
        """
        self.model = model
        self.modality = modality
        self.target_module, self.target_layer_name = find_target_layer(
            model=model,
            target_layer=target_layer,
            modality=modality,
        )

    def generate(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        target_class: Optional[int] = None,
    ) -> GradCAMResult:
        """
        Generates a Grad-CAM localization map for the given input and target class.

        Args:
            inputs: Model input tensor or tuple/dict of tensors:
                    - For FundusClassifier: Tensor of shape (1, 3, H, W) or (3, H, W)
                    - For MultimodalClassifier: Tuple (fundus_tensor, oct_tensor) or
                      dict {"fundus": ..., "oct_img": ...}
            target_class: Class index to compute gradients for. If None, uses model's predicted class.

        Returns:
            GradCAMResult containing normalized heatmap and classification metadata.
        """
        was_training = self.model.training
        self.model.eval()

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        # Hook to capture forward activations and register backward tensor hook
        def forward_hook(module: nn.Module, input_t: Any, output_t: Any):
            # If output is a tuple, take the first tensor
            feat = output_t[0] if isinstance(output_t, (tuple, list)) else output_t
            activations.append(feat)

            def tensor_backward_hook(grad: torch.Tensor):
                gradients.append(grad)

            feat.register_hook(tensor_backward_hook)

        hook_handle = self.target_module.register_forward_hook(forward_hook)

        try:
            self.model.zero_grad(set_to_none=True)

            # Standardize inputs into proper batch shape
            forward_args, forward_kwargs, spatial_hw = self._prepare_forward_inputs(inputs)

            # Forward pass
            logits = self.model(*forward_args, **forward_kwargs)
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Model forward output must be a torch.Tensor, got {type(logits)}")

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            # Softmax probabilities
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            predicted_class = int(np.argmax(probs))

            # Select target class
            resolved_class = predicted_class if target_class is None else int(target_class)
            if resolved_class < 0 or resolved_class >= logits.shape[-1]:
                raise ValueError(
                    f"Target class {resolved_class} is out of bounds for output with {logits.shape[-1]} classes."
                )

            # Target score
            score = logits[0, resolved_class]

            # Backward pass to calculate gradients w.r.t. target layer activations
            score.backward(retain_graph=False)

            if len(activations) == 0 or len(gradients) == 0:
                raise RuntimeError(
                    f"Grad-CAM hooks failed to capture activations/gradients from layer '{self.target_layer_name}'."
                )

            act = activations[0]  # (B, C, H', W')
            grad = gradients[0]   # (B, C, H', W')

            # Channel-wise pooling of gradients: alpha_k = (1/Z) * sum_i sum_j (d y^c / d A_ij^k)
            weights = torch.mean(grad, dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)

            # Linear combination of weighted forward activation maps
            cam = torch.sum(weights * act, dim=1, keepdim=True)  # (B, 1, H', W')

            # Apply ReLU to retain only features with positive influence on the target class
            cam = F.relu(cam)

            # Resize to input spatial dimensions (H, W)
            target_h, target_w = spatial_hw
            if cam.shape[-2:] != (target_h, target_w):
                cam = F.interpolate(
                    cam,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

            raw_heatmap = cam.squeeze().detach().cpu().numpy()

            # Safe normalization to [0.0, 1.0] without division by zero or NaNs
            min_val = float(raw_heatmap.min())
            max_val = float(raw_heatmap.max())
            if max_val > min_val:
                norm_heatmap = (raw_heatmap - min_val) / (max_val - min_val)
            else:
                norm_heatmap = np.zeros_like(raw_heatmap, dtype=np.float32)

            norm_heatmap = np.clip(norm_heatmap, 0.0, 1.0).astype(np.float32)

            return GradCAMResult(
                heatmap=norm_heatmap,
                raw_heatmap=raw_heatmap,
                target_class=resolved_class,
                predicted_class=predicted_class,
                class_probabilities=probs,
                target_layer_name=self.target_layer_name,
                modality_explained=self.modality,
                metadata={
                    "model_class": self.model.__class__.__name__,
                    "target_layer": self.target_layer_name,
                    "target_score": float(score.detach().cpu().item()),
                    "spatial_shape": (target_h, target_w),
                },
            )

        finally:
            hook_handle.remove()
            activations.clear()
            gradients.clear()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def _prepare_forward_inputs(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], Tuple[int, int]]:
        """Validates and shapes input arguments for the forward pass."""
        forward_args: List[torch.Tensor] = []
        forward_kwargs: Dict[str, torch.Tensor] = {}

        if isinstance(inputs, torch.Tensor):
            t = inputs
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.ndim != 4:
                raise ValueError(f"Expected 3D or 4D tensor for single-image model, got shape {inputs.shape}")
            forward_args.append(t)
            spatial_hw = (t.shape[-2], t.shape[-1])

        elif isinstance(inputs, (tuple, list)):
            if len(inputs) == 0:
                raise ValueError("Inputs tuple cannot be empty.")
            first_t = None
            for item in inputs:
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"All elements of inputs tuple must be torch.Tensor, got {type(item)}")
                t = item.unsqueeze(0) if item.ndim == 3 else item
                forward_args.append(t)
                if first_t is None:
                    first_t = t
            spatial_hw = (first_t.shape[-2], first_t.shape[-1])  # type: ignore

        elif isinstance(inputs, dict):
            first_t = None
            for k, item in inputs.items():
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"Dictionary input '{k}' must be a torch.Tensor, got {type(item)}")
                t = item.unsqueeze(0) if item.ndim == 3 else item
                forward_kwargs[k] = t
                if first_t is None:
                    first_t = t
            spatial_hw = (first_t.shape[-2], first_t.shape[-1])  # type: ignore

        else:
            raise TypeError(f"Unsupported inputs format: {type(inputs)}")

        return forward_args, forward_kwargs, spatial_hw


def overlay_heatmap_on_image(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    normalize_mode: str = "imagenet",
) -> np.ndarray:
    """
    Overlays a normalized Grad-CAM heatmap onto a fundus image with medical clarity.

    Args:
        image: Original fundus image (PIL Image, uint8 ndarray, or PyTorch tensor).
        heatmap: 2D float32 array in [0.0, 1.0].
        alpha: Blend opacity for the heatmap (0.0 = only original image, 1.0 = only heatmap).
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
        normalize_mode: Normalization mode to undo if tensor is provided ("imagenet" or "bipolar").

    Returns:
        RGB uint8 NumPy array of shape (H, W, 3).
    """
    # 1. Standardize base image to uint8 RGB (H, W, 3)
    if isinstance(image, torch.Tensor):
        img_np = denormalize_to_numpy(image, normalize_mode=normalize_mode)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image.copy()
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 3 and img_np.shape[0] in (1, 3):  # CHW format
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    h, w = img_np.shape[:2]

    # 2. Resize heatmap to match image dimensions if needed
    if heatmap.shape != (h, w):
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    # 3. Apply color map to heatmap
    heatmap_uint8 = np.uint8(255.0 * np.clip(heatmap_resized, 0.0, 1.0))
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # 4. Alpha blend
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = cv2.addWeighted(img_np, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
    return blended


def save_gradcam_artifacts(
    result: GradCAMResult,
    output_dir: Union[str, Path],
    filename_prefix: str = "gradcam",
    original_image: Optional[Union[np.ndarray, Image.Image, torch.Tensor]] = None,
    alpha: float = 0.45,
) -> Dict[str, Path]:
    """
    Saves raw heatmap and blended overlay images to the specified directory.

    Returns:
        Dictionary of saved filepaths: {"heatmap": Path, "overlay": Optional[Path]}.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}

    # 1. Save normalized colormapped heatmap
    heatmap_path = out_dir / f"{filename_prefix}_heatmap.png"
    heatmap_uint8 = np.uint8(255.0 * np.clip(result.heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(str(heatmap_path), heatmap_colored)
    saved_paths["heatmap"] = heatmap_path

    # 2. Save blended overlay if base image was provided
    if original_image is not None:
        overlay_path = out_dir / f"{filename_prefix}_overlay.png"
        overlay_rgb = result.overlay(original_image, alpha=alpha)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        saved_paths["overlay"] = overlay_path

    return saved_paths


def compute_gradcam(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    target_class: Optional[int] = None,
    target_layer: Optional[Union[nn.Module, str]] = None,
    modality: str = "fundus",
) -> GradCAMResult:
    """
    Convenience functional interface to compute Grad-CAM for a given model and input.
    """
    explainer = GradCAM(model=model, target_layer=target_layer, modality=modality)
    return explainer.generate(inputs=inputs, target_class=target_class)
