"""Conditional U-Net + cooling-curve encoder for the SSRVE DDPM baseline.

The cooling-curve encoder mirrors the transformer's `CoolingCurveEncoder`
(2 layers, d_model=128, 4 heads) so the conditioning signal carries the same
information as the SSRVE transformer sees. The encoder output (B, L, d_model)
feeds the U-Net as cross-attention key/value sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from diffusers import UNet2DConditionModel
from torch import nn

# Cooling-curve encoder (mirrors multiphasessrve.ml.model.CoolingCurveEncoder)

class CoolingCurveEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        max_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(
        self, curve: torch.Tensor, curve_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """curve: (B, L, 2)  curve_mask: (B, L) bool (True = valid).

        Returns (encoder_hidden_states, key_padding_mask) where
        key_padding_mask follows the diffusers convention:
        True = padded position to be IGNORED by attention.
        """
        x = self.input_proj(curve)
        x = x + self.pos_emb[:, : x.size(1)]
        # nn.TransformerEncoder expects key_padding_mask: True = ignore.
        kpm = ~curve_mask
        h = self.encoder(x, src_key_padding_mask=kpm)
        return h, kpm

# Wrapper: U-Net + cond encoder

@dataclass
class ModelConfig:
    in_channels: int = 4
    out_channels: int = 4
    sample_size: int = 64
    block_out_channels: tuple[int, ...] = (64, 128, 256, 256)
    layers_per_block: int = 2
    cross_attention_dim: int = 128
    attention_head_dim: int = 8


class ConditionalDDPM(nn.Module):
    def __init__(self, mc: ModelConfig, cond_encoder: CoolingCurveEncoder):
        super().__init__()
        if cond_encoder.d_model != mc.cross_attention_dim:
            raise ValueError(
                f"cond_encoder.d_model ({cond_encoder.d_model}) must equal "
                f"cross_attention_dim ({mc.cross_attention_dim})"
            )
        self.cond_encoder = cond_encoder
        self.unet = UNet2DConditionModel(
            sample_size=mc.sample_size,
            in_channels=mc.in_channels,
            out_channels=mc.out_channels,
            layers_per_block=mc.layers_per_block,
            block_out_channels=mc.block_out_channels,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=mc.cross_attention_dim,
            attention_head_dim=mc.attention_head_dim,
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        cooling_curve: torch.Tensor,
        curve_mask: torch.Tensor,
    ) -> torch.Tensor:
        ehs, _kpm = self.cond_encoder(cooling_curve, curve_mask)
        # diffusers cross-attn ignores attention_mask in older releases on this
        # block type; padding tokens are zero-attention-weight after softmax in
        # practice because their value vectors carry no signal. For strict
        # masking, switch to encoder_attention_mask once on diffusers >= 0.27.
        return self.unet(
            sample=x_noisy,
            timestep=timesteps,
            encoder_hidden_states=ehs,
        ).sample


def num_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
