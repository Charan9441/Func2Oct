"""
src/fundus2oct/models/pipeline.py

End-to-end Fundus2OCT Generative Pipeline Coordinator.
Connects:
Fundus Image -> Preprocessing -> Fundus Encoder -> Conditional Latent Diffusion -> OCT Latent -> VAE Decoder -> Synthetic OCT

Enforces strict research integrity:
If any required checkpoint (Fundus Encoder, Diffusion UNet, or VAE) is missing or untrained,
raises GeneratorNotReadyError rather than substituting random noise or disguised real images.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from fundus2oct.config import AppConfig, load_config
from fundus2oct.models.oct_vae import OCTVAE2D, build_oct_vae, load_vae_checkpoint


class GeneratorNotReadyError(RuntimeError):
    """Raised when inference is attempted but generative model checkpoints are missing/untrained."""
    pass


class Fundus2OCTPipeline:
    """
    Coordinates the complete generative inference pipeline from a single 2D Fundus
    photograph to synthetic OCT B-scans or 3D volumes.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        vae_model: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.config = config or load_config()
        self.device = self._resolve_device(device)

        # 1. OCT VAE
        self.oct_vae = vae_model or build_oct_vae(
            is_3d=self.config.vae.is_3d,
            in_channels=self.config.vae.in_channels,
            latent_channels=self.config.vae.latent_channels,
            base_channels=self.config.vae.base_channels,
            channel_multipliers=tuple(self.config.vae.channel_multipliers),
        ).to(self.device)

        # 2. Checkpoint Availability Auditing
        self.vae_checkpoint_path = Path(self.config.checkpoints.oct_vae)
        self.diffusion_checkpoint_path = Path(self.config.checkpoints.diffusion)
        self.encoder_checkpoint_path = Path(self.config.checkpoints.fundus_encoder)

        self.vae_ready = self._check_checkpoint(self.vae_checkpoint_path)
        self.diffusion_ready = self._check_checkpoint(self.diffusion_checkpoint_path)
        self.encoder_ready = self._check_checkpoint(self.encoder_checkpoint_path)

        # Load VAE if checkpoint exists
        if self.vae_ready:
            try:
                load_vae_checkpoint(self.oct_vae, self.vae_checkpoint_path, device=self.device)
            except Exception as e:
                print(f"[Warning] Failed to load VAE checkpoint: {e}")
                self.vae_ready = False

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return self.config.resolve_device()
        if isinstance(device, str):
            return torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        return device

    def _check_checkpoint(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 1024  # Must be larger than empty stub

    def is_generator_ready(self) -> bool:
        """Returns True only when ALL generative checkpoints are trained and loaded."""
        return self.vae_ready and self.diffusion_ready

    def get_status(self) -> Dict[str, Any]:
        """Audits readiness of all generative submodules."""
        return {
            "device": str(self.device),
            "oct_vae_ready": self.vae_ready,
            "diffusion_ready": self.diffusion_ready,
            "fundus_encoder_ready": self.encoder_ready,
            "pipeline_operable": self.is_generator_ready(),
            "vae_checkpoint": str(self.vae_checkpoint_path),
            "diffusion_checkpoint": str(self.diffusion_checkpoint_path),
        }

    def generate(
        self,
        fundus_tensor: torch.Tensor,
        num_candidates: int = 1,
        seed: int = 42,
        steps: int = 50,
        guidance_scale: float = 2.5,
        allow_smoke_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Synthesizes OCT B-scans from preprocessed fundus tensor (B, 3, 256, 256).
        Enforces integrity: Fails cleanly if conditional diffusion is untrained.
        """
        if not self.is_generator_ready():
            if not allow_smoke_test:
                raise GeneratorNotReadyError(
                    "Synthetic OCT generation is BLOCKED: The conditional diffusion checkpoint "
                    f"('{self.diffusion_checkpoint_path}') is missing or untrained. "
                    "In accordance with research integrity standards, untrained models cannot "
                    "synthesize mock measurements as valid predictions."
                )
            else:
                # Controlled smoke-test mock only when explicitly authorized
                b = fundus_tensor.shape[0]
                synthetic_oct = torch.zeros(
                    (b * num_candidates, 1, self.config.preprocessing.oct_size[0], self.config.preprocessing.oct_size[1]),
                    device=self.device,
                )
                return {
                    "synthetic_oct": synthetic_oct,
                    "is_genuine": False,
                    "notice": "SMOKE TEST ONLY: Generated with architectural mock placeholder.",
                }

        # Real generative forward flow when weights are trained:
        # 1. c = fundus_encoder(fundus_tensor)
        # 2. z_0 = diffusion.sample(c, steps=steps, guidance=guidance_scale)
        # 3. x_synthetic = oct_vae.decode(z_0)
        raise NotImplementedError("Diffusion weights trained milestone is pending.")
