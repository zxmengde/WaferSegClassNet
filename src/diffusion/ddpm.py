# -*- coding: utf-8 -*-
"""
简化版 DDPM 实现
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """带时间嵌入的残差块"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.res_conv = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.res_conv(x)


class Downsample(nn.Module):
    """下采样"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet(nn.Module):
    """小型 UNet"""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes

        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.class_embedding = None
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, time_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_channels = []

        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(in_ch, out_ch, time_dim, dropout))
                in_ch = out_ch
                self.skip_channels.append(out_ch)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(in_ch))

        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim, dropout)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim, dropout)

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.ups.append(ResBlock(in_ch + out_ch, out_ch, time_dim, dropout))
                in_ch = out_ch
            if i != len(channel_mults) - 1:
                self.ups.append(Upsample(in_ch))

        self.final_norm = _group_norm(in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        if self.class_embedding is not None and labels is not None:
            t_emb = t_emb + self.class_embedding(labels)

        h = self.init_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:
                h = layer(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h


def _extract(buffer: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    out = buffer.gather(0, t)
    return out.view(-1, *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM 前向/反向过程"""

    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas * x_start + sqrt_one_minus * noise

    def p_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, labels)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        betas_t = _extract(self.betas, t, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip = _extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip * (x - betas_t * model(x, t, labels) / sqrt_one_minus)

        if (t == 0).all():
            return model_mean

        posterior_var = _extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        image_size: int,
        batch_size: int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.randn(batch_size, 1, image_size, image_size, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, labels)
        return x
