"""Transformer (DETR-style) generator for SSRVE nucleation points - Option B.

Architecture (VAE encoder removed - see Option B rationale below)
-----------------------------------------------------------------
CoolingCurveEncoder (TransformerEncoder + masked mean pool)
        │
        ├──── sequence (B, L, d_model)  ──────────────────────┐
        │                                                      │
        ▼                                                      │
   curve_pooled (B, d_model)                                   │
        │                                                      │
        ├──► Auxiliary heads: fraction, grain_size, n_points   │
        │                                                      │
        ▼ (inference: z ~ N(0, I) injected as memory token)    │
   z_token via z_proj (B, 1, d_model)                          │
                                                                │
   memory = [curve_seq || z_token]  (B, L+1, d_model) ◄────────┘
        │
        ▼
   PointDecoder (TransformerDecoder with learned slot queries)
        │
        ▼
   Per-slot outputs: nuc (linear), phase_logits, active_logit
        │
        ▼
   RepairLayer (softmax phase, sigmoid active; nuc inference clamp)

Why no VAE encoder (Option B)
-----------------------------
The previous CVAE design encoded training target params into a posterior z to
handle multimodality (~200 valid configurations per CR). Option B solves the
multimodality problem differently: the loss is distribution-matching (Sinkhorn
divergence on the set {nuc, phase_embed} + marginal KL on phase fractions),
so no per-sample posterior is needed. z ~ N(0, I) injected as a memory token
provides stochasticity for diverse sampling without the teacher-forcing
leakage of a CVAE.

phase_embed
-----------
A single learned linear `nn.Linear(num_phases, 4, bias=False)` maps soft phase
distributions (softmax of logits) into a 4-D metric space used by Sinkhorn.
Initialized orthogonally so class distances start isotropic; training adapts
them to reflect which phases are metallurgically "close".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from multiphasessrve.ml.config import CVAEConfig
from multiphasessrve.ml.repair import RepairLayer


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CoolingCurveEncoder(nn.Module):
    """Returns both the full sequence (for cross-attention memory) and a
    masked-mean pooled embedding (for auxiliary heads)."""

    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.curve_input_dim, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

    def forward(
        self, curve: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sequence, pooled).

        sequence: (B, L, d_model) - full transformer output
        pooled:   (B, d_model)    - masked mean over active timesteps
        """
        x = self.input_proj(curve)
        x = self.pos_encoding(x)
        pad_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return x, pooled


class PointDecoder(nn.Module):
    """DETR-style transformer decoder with learned per-slot queries.

    Memory is `[curve_seq || z_token]`. Queries are pure learned parameters  - 
    no per-sample conditioning shortcut. All conditioning flows through
    cross-attention.

    The decoder is deterministic with respect to the cooling curve: in
    empirical tests, the injected noise `z` is effectively ignored (std of
    outputs across different `z` values is ~1e-5). This is physically
    correct: for a fixed cooling rate, phase fractions and grain size are
    deterministic outcomes of the phase transformation. Spatial diversity
    of the final SSRVE comes from uniform random xy in `denormalize_candidate`
    - that corresponds to the physical randomness of nucleation sites in a
    specific fragment of steel, not to multimodality of the distribution.
    """

    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.z_proj = nn.Linear(config.latent_dim, config.d_model)
        self.learned_queries = nn.Parameter(
            torch.randn(config.num_points, config.d_model) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        # xy_head predicts nucleation positions in [0,1]² (sigmoid applied in
        # SSRVEGenerator.forward). Trained end-to-end via SoftVoronoi: gradient
        # flows from phase fraction / grain size metrics back through tessellation.
        # Unlike MSE(xy_pred, xy_target), this does not require 1:1 correspondence
        # with CMA-ES positions - the model discovers its own valid configurations.
        self.xy_head = nn.Linear(config.d_model, 2)
        self.nuc_head = nn.Linear(config.d_model, 1)
        self.phase_head = nn.Linear(config.d_model, config.num_phases)
        self.active_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        curve_seq: torch.Tensor,
        curve_mask: torch.Tensor | None,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            curve_seq:  (B, L_curve, d_model)
            curve_mask: (B, L_curve) bool, True = valid; or None
            z:          (B, latent_dim)

        Returns:
            nuc_raw:      (B, P, 1)
            phase_logits: (B, P, num_phases)
            active_logit: (B, P, 1)
            xy_raw:       (B, P, 2) - sigmoid applied in SSRVEGenerator.forward
        """
        B = curve_seq.shape[0]
        z_token = self.z_proj(z).unsqueeze(1)                 # (B, 1, d_model)
        memory = torch.cat([curve_seq, z_token], dim=1)        # (B, L_curve+1, d_model)

        if curve_mask is not None:
            extra_valid = torch.ones(B, 1, dtype=torch.bool, device=curve_mask.device)
            mem_valid = torch.cat([curve_mask, extra_valid], dim=1)
            memory_pad_mask = ~mem_valid
        else:
            memory_pad_mask = None

        queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # (B, P, d_model)
        decoded = self.transformer(
            queries, memory, memory_key_padding_mask=memory_pad_mask
        )
        return (
            self.nuc_head(decoded),
            self.phase_head(decoded),
            self.active_head(decoded),
            self.xy_head(decoded),
        )


class SSRVEGenerator(nn.Module):
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config
        self.curve_encoder = CoolingCurveEncoder(config)
        self.decoder = PointDecoder(config)
        self.repair = RepairLayer(config.max_nucleation_ratio, config.num_phases)

        # Learned phase embedding used by the Sinkhorn loss (Option B).
        # Maps one-hot / softmax(phase_logits) to R^{phase_embed_dim}.
        self.phase_embed = nn.Linear(config.num_phases, config.phase_embed_dim, bias=False)
        nn.init.orthogonal_(self.phase_embed.weight)
        if not config.phase_embed_trainable:
            self.phase_embed.weight.requires_grad = False

        self.fraction_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_phases),
        )
        self.grain_size_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
        # Auxiliary: predict fraction of active slots ∈ [0, 1] from curve embedding.
        self.n_points_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cooling_curve: torch.Tensor,
        curve_mask: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass (Option B: no VAE encoder, no teacher forcing).

        Args:
            cooling_curve: (B, L, 2)
            curve_mask:    (B, L) bool or None
            z:             (B, latent_dim); if None, sampled from N(0, I)

        Returns dict with keys: nuc, phase_logits, phase_probs, active_prob,
        pred_fractions, pred_grain_size, pred_active_ratio.
        """
        curve_seq, curve_emb = self.curve_encoder(cooling_curve, curve_mask)
        B = curve_emb.shape[0]

        pred_fractions = torch.softmax(self.fraction_head(curve_emb), dim=-1)
        pred_grain_size = self.grain_size_head(curve_emb)
        pred_active_ratio = self.n_points_head(curve_emb)

        if z is None:
            if getattr(self.config, "disable_z_token", False):
                z = torch.zeros(B, self.config.latent_dim, device=curve_emb.device)
            else:
                z = torch.randn(B, self.config.latent_dim, device=curve_emb.device)

        nuc_raw, phase_logits, active_logit, xy_raw = self.decoder(curve_seq, curve_mask, z)
        nuc_out, phase_probs, active_prob = self.repair(
            nuc_raw, phase_logits, active_logit
        )
        # sigmoid bounds xy to [0,1]². Unlike nuc (sigmoid+MSE would saturate),
        # the gradient here comes from SoftVoronoi to phase_fractions - no saturation
        # issue because the chain is: loss(fractions) to Voronoi to dist² to xy_raw.
        xy = torch.sigmoid(xy_raw)  # (B, P, 2)

        return {
            "nuc": nuc_out.squeeze(-1),
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "active_prob": active_prob.squeeze(-1),
            "xy": xy,  # (B, P, 2)
            "pred_fractions": pred_fractions,
            "pred_grain_size": pred_grain_size,
            "pred_active_ratio": pred_active_ratio.squeeze(-1),
        }

    @torch.no_grad()
    def generate_params(
        self,
        cooling_curve: torch.Tensor,
        curve_mask: torch.Tensor | None = None,
        n: int = 1,
        active_threshold: float = 0.5,
        deterministic: bool = False,
        use_top_k: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Generate n candidate SSRVEs for the given cooling curve.

        Note: xy positions are NOT produced by this model. The caller should
        sample xy uniformly from [0,1]² for each active slot - see
        inference.generate_ssrve and evaluate.denormalize_candidate.

        Args:
            deterministic: if True, z = zeros (same output for same curve).
                Otherwise z = randn for each candidate (sampling).
            use_top_k: if True, select exactly `k = round(pred_active_ratio*P)`
                active slots by top active_prob ranking.
        """
        self.eval()
        if cooling_curve.shape[0] == 1 and n > 1:
            cooling_curve = cooling_curve.expand(n, -1, -1)
            if curve_mask is not None:
                curve_mask = curve_mask.expand(n, -1)

        if deterministic:
            z = torch.zeros(n, self.config.latent_dim, device=cooling_curve.device)
        else:
            z = torch.randn(n, self.config.latent_dim, device=cooling_curve.device)

        out = self(cooling_curve, curve_mask, z=z)
        nuc = self.repair.clamp_nuc_inference(out["nuc"])

        if use_top_k:
            P = self.config.num_points
            B = out["active_prob"].shape[0]
            k_per_sample = torch.round(out["pred_active_ratio"] * P).long().clamp(min=1, max=P)
            mask = torch.zeros_like(out["active_prob"], dtype=torch.bool)
            for b in range(B):
                k = int(k_per_sample[b].item())
                topk = torch.topk(out["active_prob"][b], k).indices
                mask[b, topk] = True
        else:
            mask = out["active_prob"] > active_threshold

        # Budget-constrained phase assignment: use auxiliary fraction head
        # (which is accurate) to determine how many slots each phase gets,
        # then greedily assign starting from minority phases.
        # This avoids argmax collapse to dominant class (ferrite).
        phase_id = self.repair.get_phase_ids_budgeted(
            out["phase_probs"], mask, out["pred_fractions"]
        )

        return {
            "phase_id": phase_id,
            "nuc": nuc,
            "mask": mask,
            "xy": out["xy"],  # (B, P, 2) predicted positions in [0,1]²
            "active_prob": out["active_prob"],
            "pred_active_ratio": out["pred_active_ratio"],
        }
