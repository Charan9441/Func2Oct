"""
src/fundus2oct/models/conditional_diffusion.py

Conditional Latent Video/Volume Diffusion Model (LVDM) for Retinal OCT synthesis.
Features:
- 3D UNet architecture with 4 residual blocks (res1, res2, res3, res4).
- Conditioned on EyeFound retinal fundus foundation model embeddings (512-dim).
- Spatial-temporal 3D convolution processing across B-scan volume depth and cross-sectional planes.
- Classifier-Free Guidance (CFG) conditioning with unconditional dropout.
- DDPM and fast DDIM sampling algorithms.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Sinusoidal Positional & Conditioning Embeddings
# =====================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =====================================================================
# 2. 3D Residual Block (Spatial-Temporal 3D Convolution with AdaGN)
# =====================================================================

class ResBlock3D(nn.Module):
    """
    3D Residual Block for Volume / Latent Video Diffusion.
    Modulates feature maps using combined timestep and EyeFound fundus conditioning embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # GroupNorm requires channels to be divisible by num_groups
        ng1 = min(num_groups, in_channels)
        while in_channels % ng1 != 0 and ng1 > 1:
            ng1 -= 1

        ng2 = min(num_groups, out_channels)
        while out_channels % ng2 != 0 and ng2 > 1:
            ng2 -= 1

        self.norm1 = nn.GroupNorm(ng1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Conditioning projection to scale and shift
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2),
        )

        self.norm2 = nn.GroupNorm(ng2, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D feature tensor of shape (B, C, D, H, W)
            emb: Conditioning embedding of shape (B, emb_dim)
        Returns:
            Output feature tensor of shape (B, out_channels, D, H, W)
        """
        h = self.conv1(self.act1(self.norm1(x)))

        # Adaptive GroupNorm modulation: scale and shift
        emb_out = self.emb_proj(emb)  # (B, 2 * out_channels)
        scale, shift = emb_out.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(self.act2(h)))

        return h + self.shortcut(x)


# =====================================================================
# 3. 3D UNet Architecture with 4 Residual Blocks
# =====================================================================

class UNet3D(nn.Module):
    """
    3D UNet for Conditional Latent Video/Volume Diffusion.
    Contains exactly 4 primary residual blocks:
    - res1: Downstage ResBlock 1 (in_channels -> base_channels)
    - res2: Downstage ResBlock 2 (base_channels -> base_channels * 2)
    - res3: Bottleneck / Middle ResBlock 3 (base_channels * 2 -> base_channels * 2)
    - res4: Upstage ResBlock 4 (concatenated skip -> base_channels)
    Conditioned on EyeFound fundus embeddings (512-dim).
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: Optional[int] = None,
        base_channels: int = 64,
        cond_dim: int = 512,
        time_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.time_dim = time_dim

        # 1. Timestep and conditioning embedding projections
        self.time_pos_emb = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # 2. Input convolution
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        # 3. Exactly 4 Residual Blocks across the UNet
        # Block 1 (Down Stage 1):
        self.res1 = ResBlock3D(base_channels, base_channels, emb_dim=time_dim, dropout=dropout)

        # Downsampling stage 1: spatial stride 2, preserving depth
        self.down1 = nn.Conv3d(base_channels, base_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Block 2 (Down Stage 2):
        self.res2 = ResBlock3D(base_channels, base_channels * 2, emb_dim=time_dim, dropout=dropout)

        # Downsampling stage 2: spatial stride 2, preserving depth
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Block 3 (Middle / Bottleneck Stage):
        self.res3 = ResBlock3D(base_channels * 2, base_channels * 2, emb_dim=time_dim, dropout=dropout)

        # Upsampling stage 1: 2x spatial upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        )

        # Block 4 (Up Stage):
        # Receives concatenated skip connection from res2 output (base_channels * 2 + base_channels * 2)
        self.res4 = ResBlock3D(base_channels * 4, base_channels, emb_dim=time_dim, dropout=dropout)

        # Upsampling stage 2: 2x spatial upsampling
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        # Final post-skip block and output projection
        self.post_res = ResBlock3D(base_channels * 2, base_channels, emb_dim=time_dim, dropout=dropout)
        self.final_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv3d(base_channels, self.out_channels, kernel_size=3, padding=1)

        # Reference list containing all 4 core residual blocks
        self.res_blocks = nn.ModuleList([self.res1, self.res2, self.res3, self.res4])

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy latent volume of shape (B, C, D, H, W)
            t: Timesteps tensor of shape (B,)
            cond: EyeFound fundus conditioning embedding of shape (B, cond_dim)
        Returns:
            Predicted noise tensor epsilon_theta of shape (B, out_channels, D, H, W)
        """
        # Embed timestep and condition, then fuse
        t_emb = self.time_mlp(self.time_pos_emb(t))
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb  # (B, time_dim)

        # Initial projection
        h = self.init_conv(x)  # (B, base_channels, D, H, W)

        # Down Stage 1: ResBlock 1
        h1 = self.res1(h, emb)
        d1 = self.down1(h1)

        # Down Stage 2: ResBlock 2
        h2 = self.res2(d1, emb)
        d2 = self.down2(h2)

        # Bottleneck: ResBlock 3
        mid = self.res3(d2, emb)

        # Up Stage 1: Upsample + Concatenate skip from h2
        u1 = self.up1(mid)
        # Handle potential shape rounding differences
        if u1.shape[2:] != h2.shape[2:]:
            u1 = F.interpolate(u1, size=h2.shape[2:], mode="trilinear", align_corners=False)
        u1 = torch.cat([u1, h2], dim=1)

        # Up Stage 1: ResBlock 4
        h4 = self.res4(u1, emb)

        # Up Stage 2: Upsample + Concatenate skip from h1
        u2 = self.up2(h4)
        if u2.shape[2:] != h1.shape[2:]:
            u2 = F.interpolate(u2, size=h1.shape[2:], mode="trilinear", align_corners=False)
        u2 = torch.cat([u2, h1], dim=1)

        # Final stage & projection
        out = self.post_res(u2, emb)
        out = self.final_conv(self.final_act(self.final_norm(out)))

        return out


# =====================================================================
# 4. Conditional Latent Video/Volume Diffusion Model (LVDM)
# =====================================================================

class ConditionalLVDM3D(nn.Module):
    """
    Conditional Latent Video/Volume Diffusion Model (LVDM).
    Generates synthetic 3D OCT volumes (or sequence of B-scans) conditioned on
    EyeFound fundus image embeddings.
    """

    def __init__(
        self,
        unet: Optional[UNet3D] = None,
        latent_channels: int = 8,
        cond_dim: int = 512,
        base_channels: int = 64,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        uncond_prob: float = 0.1,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.uncond_prob = uncond_prob

        # UNet backbone
        if unet is not None:
            self.unet = unet
        else:
            self.unet = UNet3D(
                in_channels=latent_channels,
                out_channels=latent_channels,
                base_channels=base_channels,
                cond_dim=cond_dim,
            )

        # Learned unconditional null embedding for Classifier-Free Guidance (CFG)
        self.null_cond = nn.Parameter(torch.randn(1, cond_dim) * 0.02)

        # Build beta schedule and diffusion coefficients
        self._register_diffusion_schedule(beta_schedule, beta_start, beta_end, timesteps)

    def _register_diffusion_schedule(
        self,
        schedule: str,
        beta_start: float,
        beta_end: float,
        timesteps: int,
    ) -> None:
        """Calculates and registers closed-form Gaussian diffusion coefficients."""
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            # Nichol & Dhariwal cosine schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = torch.clip(1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Closed-form q(x_t | x_0) forward diffusion parameters
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples noisy latent x_t at timestep t:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

        x_noisy = sqrt_alpha * x_start + sqrt_one_minus * noise
        return x_noisy, noise

    def forward(
        self,
        x_0: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        apply_dropout: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass:
        Computes LVDM noise prediction MSE loss conditioned on EyeFound fundus embeddings.
        Args:
            x_0: Clean ground truth latent OCT volume of shape (B, C, D, H, W)
                 (or (B, C, H, W), which is automatically unsqueezed to D=1)
            cond: EyeFound fundus embedding of shape (B, cond_dim)
            t: Optional timesteps (randomly sampled if None)
            noise: Optional Gaussian noise target
            apply_dropout: Whether to randomly drop conditioning for CFG
        """
        if x_0.dim() == 4:
            # (B, C, H, W) -> (B, C, 1, H, W)
            x_0 = x_0.unsqueeze(2)

        B = x_0.shape[0]
        device = x_0.device

        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # Add Gaussian noise
        x_t, target_noise = self.q_sample(x_0, t, noise)

        # Classifier-free guidance dropout during training
        eff_cond = cond
        if apply_dropout and self.uncond_prob > 0.0 and self.training:
            mask = torch.rand(B, device=device) < self.uncond_prob
            if mask.any():
                null_expanded = self.null_cond.expand(B, -1)
                eff_cond = torch.where(mask.unsqueeze(1), null_expanded, cond)

        # Predict noise using 3D UNet
        predicted_noise = self.unet(x_t, t, eff_cond)

        # MSE noise prediction loss
        loss = F.mse_loss(predicted_noise, target_noise)

        return {
            "loss": loss,
            "mse_loss": loss,
            "predicted_noise": predicted_noise,
            "target_noise": target_noise,
            "x_noisy": x_t,
            "timesteps": t,
        }

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t_val: int,
        cond: torch.Tensor,
        guidance_scale: float = 2.5,
    ) -> torch.Tensor:
        """Single reverse diffusion step from x_t to x_{t-1} with CFG."""
        B = x_t.shape[0]
        device = x_t.device
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Classifier-Free Guidance prediction
        if guidance_scale > 1.0:
            null_cond = self.null_cond.expand(B, -1)
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            c_in = torch.cat([cond, null_cond], dim=0)

            eps_both = self.unet(x_in, t_in, c_in)
            eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = self.unet(x_t, t_batch, cond)

        # Compute mean
        beta = self.betas[t_val]
        alpha = self.alphas[t_val]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_val]

        pred_x0 = (x_t - (beta / sqrt_one_minus) * eps) / torch.sqrt(alpha)

        if t_val > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t_val])
            return pred_x0 + variance * noise
        return pred_x0

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        guidance_scale: float = 2.5,
        steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Samples full synthetic OCT latent volume from Gaussian prior N(0, I)
        conditioned on EyeFound embeddings.
        Args:
            cond: EyeFound fundus embeddings of shape (B, cond_dim)
            shape: Latent volume shape (B, C, D, H, W)
            guidance_scale: CFG scale factor
            steps: Optional sub-sampled step count (defaults to full timesteps or DDIM)
        """
        if device is None:
            device = cond.device

        # If shape is 4D (B, C, H, W), unsqueeze to 5D (B, C, 1, H, W)
        is_2d_shape = len(shape) == 4
        if is_2d_shape:
            shape = (shape[0], shape[1], 1, shape[2], shape[3])

        B = cond.shape[0]
        img = torch.randn(shape, device=device)

        if steps is not None and steps < self.timesteps:
            # Use DDIM sampling for fast generation
            return self.sample_ddim(cond, shape, steps=steps, guidance_scale=guidance_scale, device=device)

        # Full ancestral DDPM sampling
        for t_val in reversed(range(self.timesteps)):
            img = self.p_sample(img, t_val, cond, guidance_scale=guidance_scale)

        if is_2d_shape:
            img = img.squeeze(2)
        return img

    @torch.no_grad()
    def sample_ddim(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 2.5,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Accelerated Denoising Diffusion Implicit Models (DDIM) deterministic/stochastic sampling.
        """
        if device is None:
            device = cond.device

        is_2d_shape = len(shape) == 4
        if is_2d_shape:
            shape = (shape[0], shape[1], 1, shape[2], shape[3])

        B = cond.shape[0]
        x = torch.randn(shape, device=device)

        # Generate evenly spaced timesteps
        times = torch.linspace(0, self.timesteps - 1, steps=steps, dtype=torch.long, device=device).flip(0)

        for i, t in enumerate(times):
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)

            # CFG prediction
            if guidance_scale > 1.0:
                null_cond = self.null_cond.expand(B, -1)
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)
                c_in = torch.cat([cond, null_cond], dim=0)

                eps_both = self.unet(x_in, t_in, c_in)
                eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.unet(x, t_batch, cond)

            alpha_bar = self.alphas_cumprod[t.item()]
            if i + 1 < len(times):
                t_next = times[i + 1].item()
                alpha_bar_prev = self.alphas_cumprod[t_next]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

            # Direction pointing to x_t
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                * torch.sqrt(1.0 - alpha_bar / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * eps
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if eta > 0.0:
                x_prev = x_prev + sigma * torch.randn_like(x)

            x = x_prev

        if is_2d_shape:
            x = x.squeeze(2)
        return x


# =====================================================================
# 5. Factory Functions & Model Loading
# =====================================================================

def build_lvdm(
    latent_channels: int = 8,
    cond_dim: int = 512,
    base_channels: int = 64,
    timesteps: int = 1000,
    beta_schedule: str = "linear",
    dropout: float = 0.0,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> ConditionalLVDM3D:
    """Convenience factory function creating a ConditionalLVDM3D instance."""
    unet = UNet3D(
        in_channels=latent_channels,
        out_channels=latent_channels,
        base_channels=base_channels,
        cond_dim=cond_dim,
        dropout=dropout,
    )
    model = ConditionalLVDM3D(
        unet=unet,
        latent_channels=latent_channels,
        cond_dim=cond_dim,
        base_channels=base_channels,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
    )

    if checkpoint_path is not None:
        load_diffusion_checkpoint(model, checkpoint_path)

    if device is not None:
        model = model.to(device)

    return model


def load_diffusion_checkpoint(
    model: ConditionalLVDM3D,
    checkpoint_path: Union[str, Path],
    strict: bool = False,
) -> bool:
    """Loads weights from checkpoint file if available."""
    path = Path(checkpoint_path)
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        data = torch.load(path, map_location="cpu")
        state_dict = data.get("state_dict", data.get("model", data))
        model.load_state_dict(state_dict, strict=strict)
        return True
    except Exception as e:
        print(f"[Warning] Failed loading diffusion checkpoint from {path}: {e}")
        return False


# Aliases for cross-module compatibility
build_conditional_diffusion = build_lvdm
LVDMDiffusion = ConditionalLVDM3D

