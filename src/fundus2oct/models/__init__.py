"""
src/fundus2oct/models/__init__.py
"""

from .fundus_encoder import FundusEncoder, build_fundus_encoder
from .oct_vae import (
    CheckpointError,
    OCTEncoder2D,
    OCTDecoder2D,
    OCTVAE2D,
    OCTAutoencoder2D,
    OCTVAE3D,
    build_oct_vae,
    load_vae_checkpoint,
)
from .conditional_diffusion import (
    ConditionalLVDM3D,
    ResBlock3D,
    UNet3D,
    build_lvdm,
    load_diffusion_checkpoint,
)
from .pipeline import (
    Fundus2OCTPipeline,
    GeneratorNotReadyError,
)
from .classifier_fundus import (
    FundusClassifier,
    EyeFoundFundusClassifier,
    build_fundus_classifier,
)
from .classifier_multimodal import (
    MultimodalClassifier,
    Multimodal3DClassifier,
    EyeFoundOCT3DClassifier,
    ConvBlock3D,
    OCT3DEncoder,
    MultimodalFusion,
    build_multimodal_classifier,
)

__all__ = [
    "FundusEncoder",
    "build_fundus_encoder",
    "CheckpointError",
    "OCTEncoder2D",
    "OCTDecoder2D",
    "OCTVAE2D",
    "OCTAutoencoder2D",
    "OCTVAE3D",
    "build_oct_vae",
    "load_vae_checkpoint",
    "ConditionalLVDM3D",
    "ResBlock3D",
    "UNet3D",
    "build_lvdm",
    "load_diffusion_checkpoint",
    "Fundus2OCTPipeline",
    "GeneratorNotReadyError",
    "FundusClassifier",
    "EyeFoundFundusClassifier",
    "build_fundus_classifier",
    "MultimodalClassifier",
    "Multimodal3DClassifier",
    "EyeFoundOCT3DClassifier",
    "ConvBlock3D",
    "OCT3DEncoder",
    "MultimodalFusion",
    "build_multimodal_classifier",
]

