"""
src/fundus2oct/models/oct_vae.py

Variational Autoencoder (VAE) for Optical Coherence Tomography (OCT) representation learning.
Provides separate 2D and 3D implementations for B-scans and volumetric stacks:
- OCTVAE2D: Input (B, 1, H, W) -> Latent z in R^(B, C_lat, H/8, W/8) -> Reconstructed (B, 1, H, W)
- OCTVAE3D: Input (B, 1, D, H, W) -> Latent z in R^(B, C_lat, D/8, H/8, W/8) -> Reconstructed (B, 1, D, H, W)
Includes reparameterization trick, KL-divergence loss, multi-scale reconstruction,
and strict checkpoint integrity verification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CheckpointError(Exception):
    """Raised when model weights are missing, empty, or incompatible."""
    pass


class ResBlock2D(nn.Module):
    """Residual convolutional block with GroupNorm and SiLU activations for 2D B-scans."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides channels cleanly
        g1 = num_groups if in_channels % num_groups == 0 else 1
        g2 = num_groups if out_channels % num_groups == 0 else 1

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class ResBlock3D(nn.Module):
    """Residual convolutional block with GroupNorm and SiLU activations for 3D OCT volumes."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        g1 = num_groups if in_channels % num_groups == 0 else 1
        g2 = num_groups if out_channels % num_groups == 0 else 1

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# =====================================================================
# 2D OCT VAE (For Cross-Sectional B-Scans)
# =====================================================================

class OCTEncoder2D(nn.Module):
    """
    2D OCT Encoder.
    Compresses (B, in_channels, H, W) through 3 downsampling stages (8x total downsampling)
    into Gaussian distribution parameters: mu and logvar of shape (B, latent_channels, H/8, W/8).
    For (B, 1, 256, 256) input, outputs (B, 8, 32, 32) latent parameters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling stages (3 downsamplings = 8x downsampling factor)
        down_blocks = []
        cur_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            down_blocks.append(ResBlock2D(cur_ch, out_ch))
            cur_ch = out_ch
            if i < len(channel_multipliers) - 1:
                # Strided conv for 2x downsampling
                down_blocks.append(nn.Conv2d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1))

        self.down_stages = nn.Sequential(*down_blocks)

        # Bottleneck & distribution heads
        self.bottleneck = ResBlock2D(cur_ch, cur_ch)
        self.to_mu = nn.Conv2d(cur_ch, latent_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv2d(cur_ch, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"OCTEncoder2D expects 4D tensor (B, C, H, W), got shape {x.shape}")
        h = self.init_conv(x)
        h = self.down_stages(h)
        h = self.bottleneck(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        # Clamp logvar for numerical stability in exp(0.5 * logvar)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mu, logvar


class OCTDecoder2D(nn.Module):
    """
    2D OCT Decoder.
    Reconstructs continuous latent z in R^(B, latent_channels, H/8, W/8) back to
    synthesized B-scan in R^(B, out_channels, H, W) normalized to [-1.0, 1.0].
    For (B, 8, 32, 32) latent input, reconstructs to (B, 1, 256, 256).
    """

    def __init__(
        self,
        latent_channels: int = 8,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (8, 4, 2, 1),
    ):
        super().__init__()
        start_ch = base_channels * channel_multipliers[0]
        self.init_conv = nn.Conv2d(latent_channels, start_ch, kernel_size=3, padding=1)
        self.bottleneck = ResBlock2D(start_ch, start_ch)

        up_blocks = []
        cur_ch = start_ch
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            up_blocks.append(ResBlock2D(cur_ch, out_ch))
            cur_ch = out_ch
            if i < len(channel_multipliers) - 1:
                # Transposed convolution upsampling by 2x
                up_blocks.append(nn.ConvTranspose2d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1))

        self.up_stages = nn.Sequential(*up_blocks)
        self.final_norm = nn.GroupNorm(8 if cur_ch % 8 == 0 else 1, cur_ch)
        self.final_conv = nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(z)
        h = self.bottleneck(h)
        h = self.up_stages(h)
        h = self.act(self.final_norm(h))
        x_recon = torch.tanh(self.final_conv(h))  # Output strictly in [-1.0, 1.0]
        return x_recon


class OCTVAE2D(nn.Module):
    """
    Full 2D OCT Variational Autoencoder.
    Maps 256x256 B-scans to 8x32x32 latent space and reconstructs back to 256x256.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.is_3d = False
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.is_trained = False

        self.encoder = OCTEncoder2D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
        )
        self.decoder = OCTDecoder2D(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=tuple(reversed(channel_multipliers)),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def encode_latent(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
        stochastic: Optional[bool] = None,
    ) -> torch.Tensor:
        """Directly encodes B-scan to latent representation z: (B, 8, 32, 32)."""
        mu, logvar = self.encode(x)
        if stochastic is not None:
            use_stoch = stochastic
        else:
            use_stoch = not deterministic
        return self.reparameterize(mu, logvar, stochastic=use_stoch)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """
        Samples z = mu + sigma * eps using the reparameterization trick.
        When stochastic=False or in eval mode with stochastic disabled, returns mu directly.
        """
        if not stochastic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full deterministic reconstruction of input B-scan."""
        z = self.encode_latent(x, deterministic=True)
        return self.decode(z)

    def forward(
        self,
        x: torch.Tensor,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, stochastic=stochastic)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        kl_weight: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite VAE loss:
        Reconstruction Loss (MSE + 0.5 * L1) + KL Divergence Loss.
        """
        mse_loss = F.mse_loss(x_recon, x, reduction="mean")
        l1_loss = F.l1_loss(x_recon, x, reduction="mean")
        recon_loss = mse_loss + 0.5 * l1_loss

        if mu is not None and logvar is not None and kl_weight > 0.0:
            # Analytical KL divergence against standard normal N(0, I)
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mse": mse_loss,
            "l1": l1_loss,
        }


class OCTAutoencoder2D(nn.Module):
    """
    Deterministic 2D OCT Autoencoder.
    Maps 256x256 B-scans to 8x32x32 latent space and reconstructs back to 256x256.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.is_3d = False
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.is_trained = False

        # Encoder stages
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        down_blocks = []
        cur_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            down_blocks.append(ResBlock2D(cur_ch, out_ch))
            cur_ch = out_ch
            if i < len(channel_multipliers) - 1:
                down_blocks.append(nn.Conv2d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1))

        self.down_stages = nn.Sequential(*down_blocks)
        self.bottleneck = ResBlock2D(cur_ch, cur_ch)
        self.to_latent = nn.Conv2d(cur_ch, latent_channels, kernel_size=3, padding=1)

        # Decoder stages
        self.decoder = OCTDecoder2D(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=tuple(reversed(channel_multipliers)),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Maps (B, 1, 256, 256) -> (B, 8, 32, 32)."""
        h = self.init_conv(x)
        h = self.down_stages(h)
        h = self.bottleneck(h)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs (B, 8, 32, 32) -> (B, 1, 256, 256)."""
        return self.decoder(z)

    def encode_latent(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encodes input B-scan to latent representation z: (B, 8, 32, 32)."""
        return self.encode(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Directly reconstructs an input B-scan."""
        return self.decode(self.encode(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse = F.mse_loss(x_recon, x, reduction="mean")
        l1 = F.l1_loss(x_recon, x, reduction="mean")
        recon_loss = mse + 0.5 * l1
        return {
            "loss": recon_loss,
            "recon_loss": recon_loss,
            "mse": mse,
            "l1": l1,
        }


# =====================================================================
# 3D Volumetric OCT VAE (For genuine 3D Volumetric Scans)
# =====================================================================

class OCTEncoder3D(nn.Module):
    """
    3D OCT Volumetric Encoder.
    Compresses (B, 1, D, H, W) across all 3 spatial axes into (B, latent_channels, D/8, H/8, W/8).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 16,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        down_blocks = []
        cur_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            down_blocks.append(ResBlock3D(cur_ch, out_ch))
            cur_ch = out_ch
            if i < len(channel_multipliers) - 1:
                down_blocks.append(nn.Conv3d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1))

        self.down_stages = nn.Sequential(*down_blocks)
        self.bottleneck = ResBlock3D(cur_ch, cur_ch)
        self.to_mu = nn.Conv3d(cur_ch, latent_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv3d(cur_ch, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"OCTEncoder3D expects 5D tensor (B, C, D, H, W), got shape {x.shape}")
        h = self.init_conv(x)
        h = self.down_stages(h)
        h = self.bottleneck(h)
        mu = self.to_mu(h)
        logvar = torch.clamp(self.to_logvar(h), -30.0, 20.0)
        return mu, logvar


class OCTDecoder3D(nn.Module):
    """
    3D OCT Volumetric Decoder.
    Reconstructs 3D volumetric latent (B, latent_channels, D/8, H/8, W/8) to (B, 1, D, H, W).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 16,
        channel_multipliers: Tuple[int, ...] = (8, 4, 2, 1),
    ):
        super().__init__()
        start_ch = base_channels * channel_multipliers[0]
        self.init_conv = nn.Conv3d(latent_channels, start_ch, kernel_size=3, padding=1)
        self.bottleneck = ResBlock3D(start_ch, start_ch)

        up_blocks = []
        cur_ch = start_ch
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            up_blocks.append(ResBlock3D(cur_ch, out_ch))
            cur_ch = out_ch
            if i < len(channel_multipliers) - 1:
                up_blocks.append(nn.ConvTranspose3d(cur_ch, cur_ch, kernel_size=4, stride=2, padding=1))

        self.up_stages = nn.Sequential(*up_blocks)
        self.final_norm = nn.GroupNorm(8 if cur_ch % 8 == 0 else 1, cur_ch)
        self.final_conv = nn.Conv3d(cur_ch, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(z)
        h = self.bottleneck(h)
        h = self.up_stages(h)
        h = self.act(self.final_norm(h))
        return torch.tanh(self.final_conv(h))


class OCTVAE3D(nn.Module):
    """Full 3D Volumetric OCT Variational Autoencoder."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 16,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.is_3d = True
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.is_trained = False

        self.encoder = OCTEncoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
        )
        self.decoder = OCTDecoder3D(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=tuple(reversed(channel_multipliers)),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        if not stochastic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, stochastic=stochastic)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean") + 0.5 * F.l1_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + kl_weight * kl_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


# =====================================================================
# Unified Factory & Checkpoint Helpers
# =====================================================================

def build_oct_vae(
    is_3d: bool = False,
    in_channels: int = 1,
    latent_channels: int = 8,
    base_channels: Optional[int] = None,
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
    as_autoencoder: bool = False,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[OCTVAE2D, OCTVAE3D, OCTAutoencoder2D]:
    """Factory helper to build appropriate 2D (VAE or Autoencoder) or 3D OCT VAE."""
    if is_3d:
        base_ch = base_channels or 16
        model = OCTVAE3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_ch,
            channel_multipliers=channel_multipliers,
        )
    elif as_autoencoder:
        base_ch = base_channels or 32
        model = OCTAutoencoder2D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_ch,
            channel_multipliers=channel_multipliers,
        )
    else:
        base_ch = base_channels or 32
        model = OCTVAE2D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_ch,
            channel_multipliers=channel_multipliers,
        )

    if checkpoint_path is not None:
        load_vae_checkpoint(model, checkpoint_path, allow_untrained_fallback=True)

    if device is not None:
        model = model.to(device)

    return model


def load_vae_checkpoint(
    model: Union[OCTVAE2D, OCTVAE3D, OCTAutoencoder2D],
    checkpoint_path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
    allow_untrained_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Safely loads trained VAE checkpoint into model.
    Enforces strict verification:
    - Raises CheckpointError if file does not exist or is 0 bytes.
    - If allow_untrained_fallback is True, issues explicit warning and marks is_trained=False.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        if allow_untrained_fallback:
            print(f"[WARNING] Checkpoint not found at {path}. Model remains UNTRAINED (random weights).")
            model.is_trained = False
            return {"status": "untrained", "reason": "missing_file"}
        raise CheckpointError(
            f"VAE checkpoint file not found at '{path}'. "
            "Please train the VAE using 'python scripts/train_vae.py' first."
        )

    if path.stat().st_size == 0:
        if allow_untrained_fallback:
            print(f"[WARNING] Checkpoint '{path}' is 0 bytes (empty stub). Model remains UNTRAINED.")
            model.is_trained = False
            return {"status": "untrained", "reason": "empty_file"}
        raise CheckpointError(
            f"VAE checkpoint file '{path}' is empty (0 bytes). "
            "A genuine trained weight file is required for latent encoding."
        )

    try:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.is_trained = True
        return {
            "status": "loaded",
            "epoch": checkpoint.get("epoch", -1),
            "val_loss": checkpoint.get("val_loss", None),
        }
    except Exception as e:
        if allow_untrained_fallback:
            print(f"[WARNING] Failed loading checkpoint {path}: {e}. Model remains UNTRAINED.")
            model.is_trained = False
            return {"status": "untrained", "reason": str(e)}
        raise CheckpointError(f"Incompatible or corrupted checkpoint at {path}: {e}") from e
