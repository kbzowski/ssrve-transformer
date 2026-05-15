"""Differentiable Soft Voronoi tessellation for end-to-end SSRVE training.

Approximates a Voronoi tessellation using softmax-weighted pixel
assignments, so gradients from phase-fraction and grain-size metrics
can flow back to the predicted nucleation positions xy.

Why the soft surrogate is needed
--------------------------------
MSE(xy_pred, xy_target) does not work because the CMA-ES training data
contains around 200 equally good spatial configurations per CR; there
is no single "correct" xy target. Soft Voronoi replaces per-slot xy
supervision with aggregate metrics: the model is trained to produce
configurations that yield correct phase fractions and grain sizes
AFTER tessellation, which is what we ultimately care about.

Two tessellation modes
----------------------
1) Classic soft Voronoi: log_w = -dist^2 / T. Fast, convex cells, no
   periodic BC, no nucleation delay. Matches the historical sv5 setup.

2) Apollonius (additively-weighted Voronoi):
   log_w = -(dist + delay) / T with toroidal (periodic) distance and
   delay as a function of nuc_ratio. In the limit T to 0 this matches
   the Johnson-Mehl cellular automaton in `ssrve_algs.py` for unit
   growth velocity, closing the train-inference gap for CR > 20.

Mode 2 is enabled by `sv_periodic=True` and `sv_use_nuc_weights=True`
in the config.

Temperature schedule
--------------------
High T (e.g. 0.05): grains claim smooth overlapping regions; large
informative gradients, the model can make big moves.
Low T (e.g. 0.005): sharp boundaries approaching the true Voronoi;
fine-grained refinement of positions.

Physical mapping
----------------
xy in [0, 1]^2 represents a real_size by real_size micrometre domain.
Grid pixel size: (real_size / G) micrometres. For real_size = 50 and
G = 64 this is about 0.78 micrometres per pixel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def compute_nuc_delay(
    nuc_ratio: torch.Tensor,
    scale: float,
    mode: str = "sqrt",
    const_value: float | None = None,
) -> torch.Tensor:
    """Map nuc_ratio in [0, 0.3] to a delay in distance units.

    Modes (A3 ablation):
    - "sqrt"   (baseline): delay = scale * sqrt(nuc_ratio)
    - "linear" (A3c):      delay = scale * nuc_ratio
    - "const"  (A3b):      delay = scale * sqrt(const_value), per slot constant

    Physical motivation for the sqrt mapping: in JM-CA with many active
    nuclei the fill ratio grows as t^2, so the time to cross a given
    nuc_ratio threshold scales as sqrt(nuc_ratio).
    """
    nr = nuc_ratio.clamp(min=0.0)
    if mode == "sqrt":
        return scale * torch.sqrt(nr + 1e-12)
    if mode == "linear":
        return scale * nr
    if mode == "const":
        if const_value is None:
            raise ValueError("apollonius_mode='const' requires const_value (A3b w_bar).")
        return scale * math.sqrt(max(const_value, 0.0) + 1e-12) * torch.ones_like(nr)
    raise ValueError(f"unknown apollonius_mode={mode!r}")


def sample_gumbel(shape: tuple[int, ...], device: torch.device, eps: float = 1e-10) -> torch.Tensor:
    """Sample from standard Gumbel(0, 1) for Gumbel-Softmax noise."""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


class SoftVoronoi(nn.Module):
    """Softmax-based differentiable Voronoi tessellation.

    Computes per-pixel soft assignments, phase fractions, and ferrite grain
    size from predicted nucleation centers. All operations are differentiable
    w.r.t. xy (nucleation positions), phase_probs, and (optionally) nuc_ratio.

    Args:
        grid_size: resolution G; uses G×G pixel grid. Higher = more accurate
                   but more memory (O(B·P·G²)).
        real_size: physical domain size in μm (for grain diameter calculation).
        num_phases: number of phase classes.
        periodic: if True, uses toroidal distance (periodic BC); matches
                  the JM-CA in `ssrve_algs.handle_periodic_boundaries`.
        use_nuc_weights: if True, Apollonius diagram with `nuc_delay`
                         added to the (Euclidean, not squared) distance.
                         Requires `nuc_ratio` to be passed to forward().
        nuc_weight_scale: scale parameter for `compute_nuc_delay`.
    """

    def __init__(
        self,
        grid_size: int,
        real_size: float,
        num_phases: int,
        periodic: bool = True,
        use_nuc_weights: bool = True,
        nuc_weight_scale: float = 0.5,
        apollonius_mode: str = "sqrt",
        nuc_ratio_const: float | None = None,
    ):
        super().__init__()
        self.G = grid_size
        self.real_size = real_size
        self.num_phases = num_phases
        self.periodic = periodic
        self.use_nuc_weights = use_nuc_weights
        self.nuc_weight_scale = nuc_weight_scale
        self.apollonius_mode = apollonius_mode
        self.nuc_ratio_const = nuc_ratio_const

        # Precompute pixel centre positions in [0,1]²: (G², 2).
        gy = torch.linspace(0.0, 1.0, grid_size)
        gx = torch.linspace(0.0, 1.0, grid_size)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (G², 2)
        self.register_buffer("grid", grid)

    def _pairwise_diff(self, xy: torch.Tensor) -> torch.Tensor:
        """Per-pair differences (xy - grid) with optional toroidal wrap.

        Args:
            xy: (B, P, 2)
        Returns:
            diff: (B, P, G², 2) in [-0.5, 0.5] if periodic, else unbounded.
        """
        diff = xy.unsqueeze(2) - self.grid.unsqueeze(0).unsqueeze(0)  # (B, P, G², 2)
        if self.periodic:
            # Torus wrap: delta in [-0.5, 0.5]. `delta - round(delta)` is
            # differentiable everywhere except at the jump points (zero
            # measure, so the gradient is unaffected).
            diff = diff - torch.round(diff)
        return diff

    def forward(
        self,
        xy: torch.Tensor,                              # (B, P, 2) positions in [0,1]²
        phase_probs: torch.Tensor,                      # (B, P, C)
        active_prob: torch.Tensor,                      # (B, P)
        temperature: float,
        nuc_ratio: torch.Tensor | None = None,          # (B, P) nucleation ratios
        gumbel_ste: bool = False,                       # enable Gumbel-Softmax Straight-Through
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute phase fractions and ferrite grain size via soft Voronoi.

        Args:
            xy:           (B, P, 2) predicted nucleation positions
            phase_probs:  (B, P, C) phase probabilities (sum to 1 over C)
            active_prob:  (B, P) active slot probability
            temperature:  softmax temperature.
            nuc_ratio:    (B, P) nucleation ratios ∈ [0, max_nuc_ratio]. Required
                          when use_nuc_weights=True (Apollonius mode).
            gumbel_ste:   if True, forward uses hard argmax with Gumbel noise,
                          backward propagates soft gradients (STE).

        Returns:
            phase_fractions:  (B, C) area fractions per phase (sum to ~1)
            ferrite_gs_um:    (B,) mean ferrite grain diameter in μm
            pixel_phase_2d:   (B, G, G, C) soft per-pixel phase assignment
        """
        B, P, _ = xy.shape
        G2 = self.G * self.G
        pixel_area_um2 = (self.real_size / self.G) ** 2
        eps = 1e-8

        diff = self._pairwise_diff(xy)                              # (B, P, G², 2)

        if self.use_nuc_weights:
            if nuc_ratio is None:
                raise ValueError(
                    "SoftVoronoi(use_nuc_weights=True) requires nuc_ratio; "
                    "pass output['nuc'] from the model."
                )
            # Apollonius: additive weights on Euclidean distance.
            # Winner per pixel: argmin_p (d(x, p) + delay_p), d = sqrt(dx^2 + dy^2).
            dist = torch.sqrt(diff.pow(2).sum(-1) + eps)             # (B, P, G²)
            nuc_delay = compute_nuc_delay(
                nuc_ratio,
                self.nuc_weight_scale,
                mode=self.apollonius_mode,
                const_value=self.nuc_ratio_const,
            )  # (B, P)
            effective = dist + nuc_delay.unsqueeze(-1)               # (B, P, G²)
            # Log-space weights: -effective/T + log(active).
            log_w = -effective / temperature + torch.log(active_prob.unsqueeze(-1) + eps)
        else:
            # Classic soft Voronoi: squared distance (historical sv5 behavior).
            dist2 = diff.pow(2).sum(-1)                              # (B, P, G²)
            log_w = -dist2 / temperature + torch.log(active_prob.unsqueeze(-1) + eps)

        if gumbel_ste:
            # Gumbel-Softmax straight-through (Jang 2017 + decoupled ST).
            # Forward: hard one-hot from (log_w + Gumbel). Backward: gradient
            # through soft(log_w / T_bwd). The expression
            # `soft + (hard - soft).detach()` gives forward = hard while
            # routing the gradient through soft.
            gumbel = sample_gumbel((B, P, G2), xy.device)
            logits_noisy = log_w + gumbel
            soft_assign = torch.softmax(logits_noisy, dim=1)          # (B, P, G²)
            winner = torch.argmax(logits_noisy, dim=1, keepdim=True)  # (B, 1, G²)
            hard_assign = torch.zeros_like(soft_assign).scatter_(1, winner, 1.0)
            soft_assign = hard_assign + soft_assign - soft_assign.detach()
        else:
            soft_assign = torch.softmax(log_w, dim=1)                 # (B, P, G²)

        # Phase fractions: pixel-wise soft phase distribution, averaged over pixels.
        pixel_phase = torch.einsum("bpg,bpc->bgc", soft_assign, phase_probs)  # (B, G², C)
        phase_fractions = pixel_phase.sum(dim=1) / G2                          # (B, C)

        # Grain areas in pixels and μm².
        grain_area_px = soft_assign.sum(dim=-1)                               # (B, P)
        grain_area_um2 = grain_area_px * pixel_area_um2                        # (B, P)

        # Equivalent circle diameter per grain: d = 2√(area/π).
        grain_diam_um = 2.0 * torch.sqrt(grain_area_um2 / math.pi + eps)      # (B, P)

        # Mean ferrite grain diameter (phase index 0 = ferrite).
        ferrite_prob = phase_probs[:, :, 0]                                    # (B, P)
        w = ferrite_prob * active_prob                                          # (B, P)
        ferrite_gs_um = (grain_diam_um * w).sum(dim=1) / w.sum(dim=1).clamp(min=eps)

        pixel_phase_2d = pixel_phase.reshape(B, self.G, self.G, -1)            # (B, G, G, C)
        return phase_fractions, ferrite_gs_um, pixel_phase_2d


def sv_temperature(epoch: int, config) -> float:
    """Linear temperature decay: sv_temp_start to sv_temp_end over sv_temp_warmup_epochs."""
    if epoch >= config.sv_temp_warmup_epochs:
        return config.sv_temp_end
    t = epoch / max(config.sv_temp_warmup_epochs, 1)
    return config.sv_temp_start + (config.sv_temp_end - config.sv_temp_start) * t


def sv_gumbel_active(epoch: int, config) -> bool:
    """Gumbel-STE activates only after sv_gumbel_start_epoch (if sv_gumbel_ste=True).

    Train with soft gradients in the early epochs (good global
    optimisation), then switch to hard assignments late for fine-tuning
    against the discrete Johnson-Mehl CA.
    """
    return bool(getattr(config, "sv_gumbel_ste", False)) and epoch >= getattr(config, "sv_gumbel_start_epoch", 100)
