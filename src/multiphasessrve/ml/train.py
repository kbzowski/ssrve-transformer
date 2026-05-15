"""Training loop for SSRVE generator (Option B: Sinkhorn + marginal KL).

Loss components
---------------
- l_sinkhorn   : weighted Sinkhorn divergence between predicted and target
                 point sets in feature space (feat_w_nuc * nuc, phase_embed).
                 Point masses: α = pred_active_prob, β = active_mask.
- l_marg_kl    : KL divergence between average predicted phase distribution
                 (over active slots) and target phase fractions. Forces
                 global proportions to match; complements Sinkhorn which
                 controls local structure.
- l_active     : BCE on per-slot active probability
- l_n_active   : MSE on auxiliary n_active head (from curve embedding)
- l_frac       : MSE on phase fractions head (from curve embedding)  - 
                 auxiliary regularizer, complementary to marginal KL
- l_gs         : MSE on grain size head (from curve embedding)

Rationale for Option B
----------------------
Per-slot CE/MSE supervision (Option H) is fundamentally mismatched with the
evaluation metric (aggregate phase fractions + grain size after Voronoi growth).
200 CMA-ES seeds per CR produce ~equivalent microstructures, so slot-level
targets are arbitrary. Sinkhorn divergence measures distance between point
**distributions** directly, matching the aggregate nature of evaluation.
Marginal KL adds a hard constraint on global phase proportions.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from multiphasessrve.ml.config import CVAEConfig
from multiphasessrve.ml.dataset import SSRVEDataset, build_cache, collate_fn
from multiphasessrve.ml.model import SSRVEGenerator
from multiphasessrve.ml.morphology import MorphologyLoss, build_morphology_targets
from multiphasessrve.ml.soft_voronoi import SoftVoronoi, sv_temperature, sv_gumbel_active

# MLflow helpers

def _init_mlflow(enabled: bool) -> bool:
    """Configure tracking URI + experiment. Respects env overrides."""
    if not enabled:
        return False
    # MLflow prints run URLs with a non-ASCII character that crashes Windows cp1252 stdout.
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(
        os.environ.get("MLFLOW_EXPERIMENT", "ssrve-option-b-v2")
    )
    return True


def _mlflow_params(config: CVAEConfig) -> dict[str, Any]:
    """Flatten dataclass config to MLflow-compatible scalars."""
    raw = asdict(config)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = "null"
        elif isinstance(v, (tuple, list)):
            out[k] = ",".join(str(x) for x in v)
        else:
            out[k] = v
    return out

# Metadata helpers (tags)

def _git_meta() -> dict[str, str]:
    """Best-effort git metadata. Silent fallback when outside a repo."""
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return ""
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "git.commit": commit or "unknown",
        "git.branch": branch or "unknown",
        "git.dirty": "true" if status else "false",
    }


def _cr_range_tag(config: CVAEConfig) -> str:
    lo, hi = config.cr_min, config.cr_max
    if lo is None and hi is None:
        return "all"
    return f"{lo if lo is not None else '-inf'}..{hi if hi is not None else 'inf'}"


def _hardware_tag() -> dict[str, str]:
    tags = {"hardware.torch": torch.__version__}
    if torch.cuda.is_available():
        tags["hardware.gpu"] = torch.cuda.get_device_name(0)
        tags["hardware.cuda"] = torch.version.cuda or "unknown"
    else:
        tags["hardware.gpu"] = "cpu"
    return tags

# Metric helpers (grad/weight norms, system, loss decomposition, physics)

_MODULE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("encoder", ("curve_encoder",)),
    ("decoder", ("decoder",)),
    ("heads", ("fraction_head", "grain_size_head", "n_points_head", "phase_embed")),
)


def _group_norms(model: "SSRVEGenerator", mode: str) -> dict[str, float]:
    """L2 norm per submodule group. mode ∈ {'grad', 'weight'}."""
    out: dict[str, float] = {}
    for group_name, members in _MODULE_GROUPS:
        sq = 0.0
        for name in members:
            sub = getattr(model, name, None)
            if sub is None:
                continue
            for p in sub.parameters():
                if mode == "grad":
                    if p.grad is None:
                        continue
                    sq += float(p.grad.detach().pow(2).sum().item())
                else:  # weight
                    sq += float(p.detach().pow(2).sum().item())
        out[f"{mode}/norm_{group_name}"] = math.sqrt(sq)
    return out


def _gpu_mem(device: str) -> dict[str, float]:
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return {}
    return {
        "system/gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "system/gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


_LOSS_TO_LAMBDA: tuple[tuple[str, str], ...] = (
    ("sinkhorn", "lambda_sinkhorn"),
    ("marg_kl", "lambda_marg_kl"),
    ("active_bce", "lambda_active"),
    ("n_active", "lambda_n_active"),
    ("fractions", "lambda_fractions"),
    ("grain_size", "lambda_grain_size"),
    ("sv_frac", "lambda_sv_frac"),
    ("sv_gs", "lambda_sv_gs"),
    ("coverage", "lambda_coverage"),
    ("s2", "lambda_s2"),
    ("boundary", "lambda_boundary"),
    ("pair_dist", "lambda_pair_dist"),
)


def _decompose_losses(metrics: dict[str, float], config: CVAEConfig) -> dict[str, float]:
    """Add weighted values (λ·L) and fractions (λ·L / total) per loss component."""
    total = metrics.get("loss/total", 0.0)
    out: dict[str, float] = {}
    for name, lam_attr in _LOSS_TO_LAMBDA:
        raw_key = f"loss/{name}"
        if raw_key not in metrics:
            continue
        lam = float(getattr(config, lam_attr))
        w = lam * metrics[raw_key]
        out[f"loss/weighted/{name}"] = w
        out[f"loss/frac/{name}"] = (w / total) if total > 0 else 0.0
    return out


@torch.no_grad()
def _physics_metrics(
    model: "SSRVEGenerator",
    loader: DataLoader,
    config: CVAEConfig,
) -> dict[str, float]:
    """Semantic validation metrics (no_grad, deterministic z).

    Computed per-sample then aggregated so per-phase and per-CR buckets are
    population-weighted (each sample contributes 1/N regardless of batch size).
    """
    device = config.device
    phase_names = ["ferrite", "pearlite", "bainite", "martensite"][: config.num_phases]

    # Accumulators
    frac_abs_err_sum = torch.zeros(config.num_phases)
    frac_max_err_samples = []
    gs_rel_err_sum = 0.0
    n_active_err_sum = 0.0
    spatial_min_dist_sum = 0.0
    spatial_nn_cv_sum = 0.0
    n_total = 0

    # Per-CR-bucket accumulators (low <20, mid 20-60, high >60)
    bucket_n_active_sum: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    bucket_frac_err_sum: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    bucket_gs_err_sum: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    bucket_count: dict[str, int] = {"low": 0, "mid": 0, "high": 0}

    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["cooling_curve"].shape[0]
        z = torch.zeros(B, config.latent_dim, device=device)
        out = model(
            cooling_curve=batch["cooling_curve"],
            curve_mask=batch["curve_mask"],
            z=z,
        )

        # Phase fraction errors (per-sample, per-phase)
        pf = out["pred_fractions"]                                     # (B, C)
        tf = batch["target_fractions"]                                 # (B, C)
        per_phase_abs = (pf - tf).abs().detach().cpu()                 # (B, C)
        frac_abs_err_sum += per_phase_abs.sum(dim=0)
        frac_max_err_samples.append(per_phase_abs.max(dim=-1).values)  # (B,)

        # Grain size relative error
        tgs = batch["target_grain_size"].squeeze(-1).detach().cpu()
        pgs_norm = out["pred_grain_size"].squeeze(-1).detach().cpu()
        # pred_grain_size is normalized to [0,1] via max_grain_size
        pgs = pgs_norm * config.max_grain_size
        denom = tgs.clamp(min=1e-6)
        gs_rel = ((pgs - tgs).abs() / denom)
        gs_rel_err_sum += float(gs_rel.sum().item())

        # n_active error (discrete, rounded)
        P = out["active_prob"].shape[1]
        pred_n_active = (out["pred_active_ratio"].detach().cpu() * P).round()
        tgt_n_active = batch["n_active"].squeeze(-1).detach().cpu()
        n_active_abs = (pred_n_active - tgt_n_active).abs()
        n_active_err_sum += float(n_active_abs.sum().item())

        # Spatial diagnostics on xy (active slots only per sample)
        xy = out["xy"].detach().cpu()                                  # (B, P, 2)
        active = out["active_prob"].detach().cpu() > 0.5               # (B, P)
        per_sample_min_dist = torch.zeros(B)
        per_sample_nn_cv = torch.zeros(B)
        for i in range(B):
            pts = xy[i][active[i]]
            if pts.shape[0] >= 2:
                d = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0))[0]
                d = d + torch.eye(d.shape[0]) * 1e6  # mask self
                nn = d.min(dim=-1).values
                per_sample_min_dist[i] = float(nn.min().item())
                per_sample_nn_cv[i] = float((nn.std() / nn.mean().clamp(min=1e-8)).item())
            else:
                per_sample_min_dist[i] = 0.0
                per_sample_nn_cv[i] = 0.0
        spatial_min_dist_sum += float(per_sample_min_dist.sum().item())
        spatial_nn_cv_sum += float(per_sample_nn_cv.sum().item())

        # Per-CR bucketing
        cr_norm = batch["cr_norm"].squeeze(-1).detach().cpu().numpy()
        cr_raw = np.exp(cr_norm * config.cr_log_std + config.cr_log_mean)
        per_sample_frac_err = per_phase_abs.mean(dim=-1)                # (B,)
        for i in range(B):
            cr_i = float(cr_raw[i])
            if cr_i < 20:
                bkt = "low"
            elif cr_i < 60:
                bkt = "mid"
            else:
                bkt = "high"
            bucket_count[bkt] += 1
            bucket_n_active_sum[bkt] += float(n_active_abs[i].item())
            bucket_frac_err_sum[bkt] += float(per_sample_frac_err[i].item())
            bucket_gs_err_sum[bkt] += float(gs_rel[i].item())

        n_total += B

    n = max(n_total, 1)
    out_metrics: dict[str, float] = {
        "phys/mean_frac_err": float(frac_abs_err_sum.sum().item()) / (n * config.num_phases),
        "phys/max_frac_err": float(torch.cat(frac_max_err_samples).max().item()),
        "phys/gs_rel_err": gs_rel_err_sum / n,
        "phys/n_active_err": n_active_err_sum / n,
        "phys/spatial_min_dist": spatial_min_dist_sum / n,
        "phys/spatial_nn_cv": spatial_nn_cv_sum / n,
    }
    for k, phase_name in enumerate(phase_names):
        out_metrics[f"phys/frac_err_{phase_name}"] = float(frac_abs_err_sum[k].item()) / n
    for bkt in ("low", "mid", "high"):
        c = max(bucket_count[bkt], 1)
        out_metrics[f"phys/n_active_err_cr_{bkt}"] = bucket_n_active_sum[bkt] / c
        out_metrics[f"phys/frac_err_cr_{bkt}"] = bucket_frac_err_sum[bkt] / c
        out_metrics[f"phys/gs_rel_err_cr_{bkt}"] = bucket_gs_err_sum[bkt] / c
        out_metrics[f"phys/count_cr_{bkt}"] = float(bucket_count[bkt])
    return out_metrics

# Blur schedule (linear warmup from start to end over warmup_epochs)

def _blur_schedule(epoch: int, config: CVAEConfig) -> float:
    if epoch >= config.sinkhorn_blur_warmup_epochs:
        return config.sinkhorn_blur_end
    t = epoch / max(config.sinkhorn_blur_warmup_epochs, 1)
    return config.sinkhorn_blur_start + (config.sinkhorn_blur_end - config.sinkhorn_blur_start) * t

# Loss (Option B: Sinkhorn + marginal KL)

def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model: SSRVEGenerator,
    config: CVAEConfig,
    epoch: int,
    soft_voronoi: SoftVoronoi | None = None,
    morphology_loss: MorphologyLoss | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss + metrics (Option B).

    Distribution matching via weighted Sinkhorn on point sets, plus marginal
    phase KL. No per-slot supervision on nuc/phase.
    """
    pred_nuc = output["nuc"]                     # (B, P)
    pred_phase_logits = output["phase_logits"]   # (B, P, num_phases)
    pred_active = output["active_prob"]          # (B, P)
    pred_active_ratio = output["pred_active_ratio"]  # (B,)

    target_phase = batch["phase_class"]          # (B, P), 1-indexed
    target_nuc = batch["nuc"]                    # (B, P)
    target_mask = batch["active_mask"]           # (B, P) bool
    target_n_active = batch["n_active"].squeeze(-1)

    B, P = pred_nuc.shape
    device = pred_nuc.device
    num_phases = config.num_phases

    # --- Sinkhorn feature vectors ----------------------------------------
    # Pred side: phase features from softmax(logits) through learned embedding.
    pred_phase_probs = F.softmax(pred_phase_logits, dim=-1)               # (B, P, C)
    pred_phase_feat = model.phase_embed(pred_phase_probs)                  # (B, P, k)

    # Target side: one-hot to embedding (same projection).
    tgt_idx = (target_phase - 1).clamp(min=0, max=num_phases - 1)
    tgt_onehot = F.one_hot(tgt_idx, num_phases).float()                    # (B, P, C)
    tgt_phase_feat = model.phase_embed(tgt_onehot)                         # (B, P, k)

    w_nuc = config.feat_w_nuc
    pred_feat = torch.cat([w_nuc * pred_nuc.unsqueeze(-1), pred_phase_feat], dim=-1)
    tgt_feat = torch.cat([w_nuc * target_nuc.unsqueeze(-1), tgt_phase_feat], dim=-1)

    # Masses: normalize so each sample's measure sums to 1.
    eps = 1e-8
    alpha = pred_active / pred_active.sum(dim=-1, keepdim=True).clamp(min=eps)
    tgt_mask_f = target_mask.float()
    beta = tgt_mask_f / tgt_mask_f.sum(dim=-1, keepdim=True).clamp(min=eps)

    sinkhorn_loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=_blur_schedule(epoch, config),
        scaling=config.sinkhorn_scaling,
    )
    # SamplesLoss returns (B,) - per-sample divergence; mean over batch.
    l_sinkhorn = sinkhorn_loss(alpha, pred_feat, beta, tgt_feat).mean()

    # --- Marginal phase KL -----------------------------------------------
    # Average predicted phase distribution over active slots (soft mask via β).
    # β is the normalized target active mask; use pred_active_prob-normalized
    # α instead so the gradient flows through active head.
    pred_marg = (pred_phase_probs * alpha.unsqueeze(-1)).sum(dim=1)         # (B, C)
    pred_marg = pred_marg / pred_marg.sum(dim=-1, keepdim=True).clamp(min=eps)
    target_frac = batch["target_fractions"]                                  # (B, C)
    target_frac_norm = target_frac / target_frac.sum(dim=-1, keepdim=True).clamp(min=eps)
    # kl_div(log_p, q) computes KL(q || p). We want KL(target || pred).
    l_marg_kl = F.kl_div(
        torch.log(pred_marg.clamp(min=eps)),
        target_frac_norm,
        reduction="batchmean",
    )

    # --- Auxiliary losses (unchanged from Option H) ----------------------
    l_active = F.binary_cross_entropy(pred_active, tgt_mask_f, reduction="mean")

    target_active_ratio = target_n_active / float(P)
    l_n_active = F.mse_loss(pred_active_ratio, target_active_ratio)

    l_frac = F.mse_loss(output["pred_fractions"], target_frac)
    target_gs_norm = batch["target_grain_size"] / config.max_grain_size
    l_gs = F.mse_loss(output["pred_grain_size"], target_gs_norm)

    # --- Per-phase uniform coverage loss (prevent phase clustering) ----------
    # Each phase's nuclei are pushed toward uniform spatial coverage of [0,1]²
    # independently. Soft weight for phase k: phase_probs[:,: ,k] * active_prob.
    # Contribution of each phase is weighted by its target fraction - absent
    # phases (target_frac=0) contribute zero, so CR=1 samples only see ferrite
    # and pearlite coverage, CR=70 samples see ferrite and martensite, etc.
    # This replaces the single combined coverage loss which allowed minority
    # phases to cluster at edges while ferrite was uniformly distributed.
    l_coverage = torch.zeros(1, device=device).squeeze()
    if config.lambda_coverage > 0 and soft_voronoi is not None and "xy" in output:
        G_ref = config.coverage_ref_size
        g = torch.linspace(0.0, 1.0, G_ref, device=device)
        gx, gy = torch.meshgrid(g, g, indexing="ij")
        ref_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)          # (G²,  2)
        ref_xy = ref_xy.unsqueeze(0).expand(B, -1, -1).contiguous()         # (B, G², 2)
        ref_weights = torch.ones(B, G_ref ** 2, device=device) / (G_ref ** 2)
        coverage_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.1, scaling=0.9)
        phase_cov_terms = []
        for k in range(num_phases):
            # Soft mass: probability that slot belongs to phase k and is active
            w_k = pred_phase_probs[:, :, k] * pred_active                   # (B, P)
            w_k_norm = w_k / w_k.sum(dim=-1, keepdim=True).clamp(min=eps)   # (B, P)
            l_k = coverage_fn(w_k_norm, output["xy"], ref_weights, ref_xy)  # (B,)
            # Per-sample weight = expected fraction of phase k in that sample
            l_k_weighted = (l_k * target_frac_norm[:, k]).mean()
            phase_cov_terms.append(l_k_weighted)
        l_coverage = torch.stack(phase_cov_terms).sum()

    # --- Soft Voronoi losses (xy end-to-end training) --------------------
    # Gradient flows: MSE(sv_fractions, target) to SoftVoronoi to dist² to xy.
    # Permutation-invariant by design - no per-slot xy correspondence needed.
    l_sv_frac = torch.zeros(1, device=device).squeeze()
    l_sv_gs = torch.zeros(1, device=device).squeeze()
    l_s2 = torch.zeros(1, device=device).squeeze()
    l_boundary = torch.zeros(1, device=device).squeeze()
    l_pair = torch.zeros(1, device=device).squeeze()
    sv_temp_val = 0.0
    pixel_phase_2d = None
    if soft_voronoi is not None and "xy" in output:
        sv_temp_val = sv_temperature(epoch, config)
        sv_fractions, sv_gs_um, pixel_phase_2d = soft_voronoi(
            output["xy"],         # (B, P, 2)
            pred_phase_probs,     # (B, P, C) - already computed above
            pred_active,          # (B, P)
            sv_temp_val,
            nuc_ratio=output["nuc"] if soft_voronoi.use_nuc_weights else None,
            gumbel_ste=sv_gumbel_active(epoch, config),
        )
        l_sv_frac = F.mse_loss(sv_fractions, target_frac)
        # Normalize grain size to [0,1] scale (consistent with auxiliary l_gs).
        l_sv_gs = F.mse_loss(
            sv_gs_um / config.max_grain_size,
            batch["target_grain_size"].squeeze(-1) / config.max_grain_size,
        )

    # --- Morphological losses (S₂, boundary ratio, pair distribution) ------
    # Only active when morphology targets are present in the batch AND
    # pixel_phase_2d is available from SoftVoronoi.
    has_morpho = (
        morphology_loss is not None
        and pixel_phase_2d is not None
        and "morpho_s2" in batch
        and "morpho_br" in batch
        and "morpho_pair_same" in batch
    )
    if has_morpho:
        l_s2, l_boundary, l_pair = morphology_loss(
            pixel_phase_2d,
            output["xy"],
            pred_phase_probs,
            pred_active,
            batch["morpho_s2"],                 # (B, C, n_bins_s2)
            batch["morpho_br"],                 # (B, C)
            batch["morpho_pair_same"],          # (B, C, n_bins_pair)
            target_frac_norm,                   # (B, C) for phase weighting
        )

    # --- CCT fraction prior (Etap XVII Stage 3) ----------------------------
    # One-sided inverse-frequency penalty on decoder phase allocation.
    # Penalizes underestimation of minority phases (bainite, pearlite) via
    # deficit = clamp(target - pred_marg, min=0) weighted by 1/mean_target_frac.
    l_cct = torch.zeros(1, device=device).squeeze()
    if config.lambda_cct > 0:
        deficit = torch.clamp(target_frac - pred_marg, min=0.0)  # (B, C)
        phase_mean_frac = target_frac.mean(dim=0).clamp(min=1e-4)  # (C,)
        inv_weights = 1.0 / phase_mean_frac  # minority phases weighted higher
        inv_weights = inv_weights / inv_weights.mean()  # normalize to mean=1
        l_cct = (deficit ** 2 * inv_weights.unsqueeze(0)).mean()

    # --- Phase separation regularizer (Etap XVII Stage 2) -----------------
    # Hinge loss na off-diagonal pairwise distances phase_embed.weight.
    # Aktywny gdy lambda_phase_sep > 0 i phase_embed jest trainable.
    l_phase_sep = torch.zeros(1, device=device).squeeze()
    if config.lambda_phase_sep > 0 and config.phase_embed_trainable:
        W = model.phase_embed.weight  # (phase_embed_dim, num_phases)
        pdist = torch.cdist(W.T, W.T)  # (num_phases, num_phases)
        off_diag = ~torch.eye(num_phases, dtype=torch.bool, device=device)
        hinge = torch.clamp(config.phase_sep_delta - pdist[off_diag], min=0.0)
        l_phase_sep = hinge.mean()

    total = (
        config.lambda_sinkhorn * l_sinkhorn
        + config.lambda_marg_kl * l_marg_kl
        + config.lambda_active * l_active
        + config.lambda_n_active * l_n_active
        + config.lambda_fractions * l_frac
        + config.lambda_grain_size * l_gs
        + config.lambda_sv_frac * l_sv_frac
        + config.lambda_sv_gs * l_sv_gs
        + config.lambda_coverage * l_coverage
        + config.lambda_s2 * l_s2
        + config.lambda_boundary * l_boundary
        + config.lambda_pair_dist * l_pair
        + config.lambda_phase_sep * l_phase_sep
        + config.lambda_cct * l_cct
    )

    return total, {
        "loss/total": total.item(),
        "loss/sinkhorn": l_sinkhorn.item(),
        "loss/marg_kl": l_marg_kl.item(),
        "loss/active_bce": l_active.item(),
        "loss/n_active": l_n_active.item(),
        "loss/fractions": l_frac.item(),
        "loss/grain_size": l_gs.item(),
        "loss/sv_frac": l_sv_frac.item(),
        "loss/sv_gs": l_sv_gs.item(),
        "loss/coverage": l_coverage.item(),
        "loss/s2": l_s2.item(),
        "loss/boundary": l_boundary.item(),
        "loss/pair_dist": l_pair.item(),
        "loss/phase_sep": l_phase_sep.item(),
        "loss/cct": l_cct.item(),
        "blur": _blur_schedule(epoch, config),
        "sv_temp": sv_temp_val,
    }

# Inference metrics (no teacher forcing - the real sanity check)

@torch.no_grad()
def compute_inference_metrics(
    model: SSRVEGenerator,
    loader: DataLoader,
    config: CVAEConfig,
    epoch: int,
    deterministic: bool = True,
) -> dict[str, float]:
    """Run forward and measure losses - identical to train-time path now
    that there is no teacher forcing (Option B). Kept as a separate entry
    point for explicit deterministic z (zeros) vs random z.

    The `epoch` argument MUST match the current training epoch so that the
    Sinkhorn blur schedule applied here matches the train-time blur. Earlier
    code used a constant epoch=0 here, which silently compared train losses
    after blur warmup against inference losses still at the warmup-start blur.
    """
    model.eval()
    accum: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(config.device) for k, v in batch.items()}
        B = batch["cooling_curve"].shape[0]
        if deterministic:
            z = torch.zeros(B, config.latent_dim, device=config.device)
        else:
            z = torch.randn(B, config.latent_dim, device=config.device)

        out = model(
            cooling_curve=batch["cooling_curve"],
            curve_mask=batch["curve_mask"],
            z=z,
        )
        _, metrics = compute_loss(out, batch, model, config, epoch=epoch)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in accum.items()}

# Train / val loops

def _epoch_pass(model, loader, config, epoch, optimizer=None, deterministic_z: bool = False, soft_voronoi: SoftVoronoi | None = None, morphology_loss: MorphologyLoss | None = None):
    """Run one full pass over `loader`.

    Args:
        deterministic_z: if True, pass z=zeros to forward (instead of letting it
            sample randn). Used for val/holdout/eval to remove sample variance
            from val_loss-driven early stopping. Train pass keeps stochastic z.
    """
    is_train = optimizer is not None
    model.train(is_train)
    accum: dict[str, float] = {}
    n = 0
    clip_max = 1.0
    n_clipped = 0
    for batch in loader:
        batch = {k: v.to(config.device) for k, v in batch.items()}
        if deterministic_z:
            B = batch["cooling_curve"].shape[0]
            z = torch.zeros(B, config.latent_dim, device=config.device)
        else:
            z = None
        with torch.set_grad_enabled(is_train):
            out = model(
                cooling_curve=batch["cooling_curve"],
                curve_mask=batch["curve_mask"],
                z=z,
            )
            loss, metrics = compute_loss(out, batch, model, config, epoch, soft_voronoi=soft_voronoi, morphology_loss=morphology_loss)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # Per-submodule grad norms BEFORE clipping (captures real magnitude).
            group_grad = _group_norms(model, mode="grad")
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max)
            total_norm_val = float(total_norm.item()) if hasattr(total_norm, "item") else float(total_norm)
            if total_norm_val >= clip_max:
                n_clipped += 1
            metrics["grad/norm_total"] = total_norm_val
            metrics["grad/log10_norm_total"] = math.log10(total_norm_val + 1e-12)
            metrics.update(group_grad)
            for k, v in group_grad.items():
                # grad/norm_encoder to grad/log10_norm_encoder
                metrics[k.replace("/norm_", "/log10_norm_")] = math.log10(v + 1e-12)
            optimizer.step()
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1
    result = {k: v / max(n, 1) for k, v in accum.items()}
    if is_train:
        result["grad/clipped_fraction"] = n_clipped / max(n, 1)
    return result


def train(
    config: CVAEConfig | None = None,
    checkpoint_name: str = "best_model.pt",
    use_mlflow: bool = True,
):
    if config is None:
        config = CVAEConfig()

    # Reproducibility - seed torch / numpy / random before any model init or data load.
    seed = getattr(config, "random_seed", None)
    if seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed} (deterministic init)", flush=True)
    else:
        print("Random seed: None (non-deterministic)", flush=True)

    mlflow_enabled = _init_mlflow(use_mlflow)

    print("Building/loading cache...", flush=True)
    build_cache(config)

    train_ds = SSRVEDataset(config, split="train")
    val_ds = SSRVEDataset(config, split="val")
    holdout_ds = SSRVEDataset(config, split="holdout")
    print(
        f"Train: {len(train_ds)}, Val: {len(val_ds)}, Holdout: {len(holdout_ds)}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_ds.get_sampler(),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    model = SSRVEGenerator(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters", flush=True)

    soft_voronoi = SoftVoronoi(
        grid_size=config.sv_grid_size,
        real_size=config.real_size,
        num_phases=config.num_phases,
        periodic=config.sv_periodic,
        use_nuc_weights=config.sv_use_nuc_weights,
        nuc_weight_scale=config.sv_nuc_weight_scale,
        apollonius_mode=getattr(config, "sv_apollonius_mode", "sqrt"),
        nuc_ratio_const=getattr(config, "sv_nuc_ratio_const", None),
    ).to(config.device)
    mode = "Apollonius (periodic, nuc-weighted)" if config.sv_use_nuc_weights else "classic"
    print(
        f"SoftVoronoi [{mode}]: G={config.sv_grid_size}, "
        f"T: {config.sv_temp_start}->{config.sv_temp_end} over {config.sv_temp_warmup_epochs} epochs",
        flush=True,
    )
    if config.sv_gumbel_ste:
        print(f"  Gumbel-STE activates at epoch {config.sv_gumbel_start_epoch}", flush=True)

    # Build morphological targets (one-time, ~15 min; skipped if already cached)
    print("Building morphology targets...", flush=True)
    build_morphology_targets(config.dataset_dir, config, config.cache_dir, num_workers=1)
    morpho_loss = MorphologyLoss(config).to(config.device)
    print(
        f"MorphologyLoss active: lambda_s2={config.lambda_s2}, "
        f"lambda_boundary={config.lambda_boundary}, lambda_pair={config.lambda_pair_dist}",
        flush=True,
    )

    # Reload datasets now that morphology_targets.npz exists
    train_ds = SSRVEDataset(config, split="train")
    val_ds = SSRVEDataset(config, split="val")
    holdout_ds = SSRVEDataset(config, split="holdout")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_ds.get_sampler(),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    best_val_loss = float("inf")
    best_epoch = -1
    best_phys: dict[str, float] = {}  # snapshot of phys/* at best-val-loss epoch
    best_mean_frac_err = float("inf")
    best_gs_rel_err = float("inf")
    patience_counter = 0
    holdout_every = 10

    run_ctx = mlflow.start_run(run_name=checkpoint_name) if mlflow_enabled else nullcontext()
    with run_ctx:
        if mlflow_enabled:
            mlflow.log_params(_mlflow_params(config))
            mlflow.log_param("n_params", n_params)
            mlflow.log_param("checkpoint_name", checkpoint_name)
            # Run metadata tags (descriptive only, do not affect results).
            mlflow.set_tags({
                **_git_meta(),
                **_hardware_tag(),
                "cr.range": _cr_range_tag(config),
                "arch.variant": os.environ.get("MLFLOW_ARCH_VARIANT", checkpoint_name),
                "dataset.size_train": str(len(train_ds)),
                "dataset.size_val": str(len(val_ds)),
                "dataset.size_holdout": str(len(holdout_ds)),
                "dataset.hash": getattr(train_ds, "dataset_hash", "unknown"),
            })

        for epoch in range(config.epochs):
            t_epoch_start = time.time()
            if torch.cuda.is_available() and str(config.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            train_m = _epoch_pass(model, train_loader, config, epoch, optimizer=optimizer, soft_voronoi=soft_voronoi, morphology_loss=morpho_loss)
            val_m = _epoch_pass(model, val_loader, config, epoch, deterministic_z=True, soft_voronoi=soft_voronoi, morphology_loss=morpho_loss)
            scheduler.step()

            for k, v in train_m.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            # Loss decomposition: lambda * L and relative share, for lambda tuning diagnostics.
            train_decomp = _decompose_losses(train_m, config)
            val_decomp = _decompose_losses(val_m, config)

            # Physics/semantic metrics - realne KPI (per-faza, per-CR bucket, spatial).
            phys_m = _physics_metrics(model, val_loader, config)

            epoch_time = time.time() - t_epoch_start
            system_m: dict[str, float] = {
                "system/epoch_time_s": epoch_time,
                "system/samples_per_sec": len(train_ds) / max(epoch_time, 1e-6),
            }
            system_m.update(_gpu_mem(config.device))

            if mlflow_enabled:
                mlflow.log_metrics({f"train/{k}": v for k, v in train_m.items()}, step=epoch)
                mlflow.log_metrics({f"val/{k}": v for k, v in val_m.items()}, step=epoch)
                mlflow.log_metrics({f"train/{k}": v for k, v in train_decomp.items()}, step=epoch)
                mlflow.log_metrics({f"val/{k}": v for k, v in val_decomp.items()}, step=epoch)
                mlflow.log_metrics(phys_m, step=epoch)
                mlflow.log_metrics(system_m, step=epoch)
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)
                # Weight norms every 10 epochs (cheap, tracks drift over time).
                if epoch % 10 == 0 or epoch == config.epochs - 1:
                    mlflow.log_metrics(_group_norms(model, mode="weight"), step=epoch)

            if epoch % holdout_every == 0 or epoch == config.epochs - 1:
                holdout_m = _epoch_pass(model, holdout_loader, config, epoch, deterministic_z=True, soft_voronoi=soft_voronoi, morphology_loss=morpho_loss)
                for k, v in holdout_m.items():
                    writer.add_scalar(f"holdout/{k}", v, epoch)
                if mlflow_enabled:
                    mlflow.log_metrics({f"holdout/{k}": v for k, v in holdout_m.items()}, step=epoch)

            if epoch % config.inference_eval_every == 0 or epoch == config.epochs - 1:
                inf_m = compute_inference_metrics(model, val_loader, config, epoch=epoch, deterministic=True)
                for k, v in inf_m.items():
                    writer.add_scalar(f"inference/{k}", v, epoch)
                if mlflow_enabled:
                    mlflow.log_metrics({f"inference/{k}": v for k, v in inf_m.items()}, step=epoch)
            else:
                inf_m = None

            val_loss = val_m["loss/total"]

            if epoch % 10 == 0 or epoch == config.epochs - 1:
                inf_p = (
                    f"inf_sink={inf_m['loss/sinkhorn']:.4f} "
                    f"inf_kl={inf_m['loss/marg_kl']:.4f} "
                    if inf_m
                    else ""
                )
                print(
                    f"Epoch {epoch:3d} | "
                    f"sink={train_m['loss/sinkhorn']:.4f} "
                    f"kl={train_m['loss/marg_kl']:.4f} "
                    f"act={train_m['loss/active_bce']:.3f} "
                    f"sv_f={train_m['loss/sv_frac']:.4f} "
                    f"sv_gs={train_m['loss/sv_gs']:.4f} "
                    f"cov={train_m['loss/coverage']:.4f} "
                    f"s2={train_m['loss/s2']:.4f} "
                    f"br={train_m['loss/boundary']:.4f} "
                    f"pd={train_m['loss/pair_dist']:.4f} "
                    f"T={train_m['sv_temp']:.4f} "
                    f"{inf_p}"
                    f"val={val_loss:.4f}",
                    flush=True,
                )

            # Track best-observed physics metrics (min across epochs),
            # separately from best_val_loss because they are an independent
            # quality signal.
            if phys_m.get("phys/mean_frac_err", float("inf")) < best_mean_frac_err:
                best_mean_frac_err = phys_m["phys/mean_frac_err"]
            if phys_m.get("phys/gs_rel_err", float("inf")) < best_gs_rel_err:
                best_gs_rel_err = phys_m["phys/gs_rel_err"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_phys = dict(phys_m)  # snapshot at best epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                    },
                    ckpt_dir / checkpoint_name,
                )
                if mlflow_enabled:
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                    mlflow.log_metric("best_epoch", epoch, step=epoch)
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

        writer.close()
        if mlflow_enabled:
            # Summary aliases: single numbers representing the result of the
            # whole run. Visible in the MLflow run list without opening per-step charts.
            summary: dict[str, float] = {
                "summary/best_val_loss": best_val_loss,
                "summary/best_epoch": float(best_epoch),
                "summary/best_mean_frac_err": best_mean_frac_err,
                "summary/best_gs_rel_err": best_gs_rel_err,
            }
            # Snapshot all phys/* metrics from the epoch where best_val_loss was reached.
            for k, v in best_phys.items():
                summary[k.replace("phys/", "summary/at_best/")] = v
            mlflow.log_metrics(summary)
            ckpt_path = ckpt_dir / checkpoint_name
            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
        print(f"Training complete. Best val loss: {best_val_loss:.4f}", flush=True)
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augmentation", action="store_true", help="enable rot/flip/translate augmentation")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--inference-every", type=int, default=20, help="inference metrics frequency")
    parser.add_argument("--cr-min", type=float, default=None, help="drop samples with cooling_rate below this value")
    parser.add_argument("--cr-max", type=float, default=None, help="drop samples with cooling_rate above this value")
    parser.add_argument("--num-points", type=int, default=None, help="fixed num_points (truncates dataset samples to first N points)")
    parser.add_argument("--frozen-phase-embed", action="store_true",
                        help="freeze phase_embed at orthogonal init (Option B+ ablation)")
    parser.add_argument("--lambda-s2", type=float, default=None, help="S2 two-point loss weight")
    parser.add_argument("--lambda-boundary", type=float, default=None, help="boundary ratio loss weight")
    parser.add_argument("--lambda-pair-dist", type=float, default=None, help="pair distribution loss weight")
    parser.add_argument("--lambda-phase-sep", type=float, default=None,
                        help="phase_embed separation regularizer weight (Etap XVII Stage 2)")
    parser.add_argument("--phase-sep-delta", type=float, default=None,
                        help="minimal pairwise dist target for phase_embed (default 0.5)")
    parser.add_argument("--lambda-cct", type=float, default=None,
                        help="CCT fraction prior weight: one-sided inv-freq penalty on pred_marg (Etap XVII Stage 3)")
    # ---- Ablation flags (experiments/ablation_baseline/) ----
    parser.add_argument("--lambda-sinkhorn", type=float, default=None)
    parser.add_argument("--lambda-marg-kl", type=float, default=None)
    parser.add_argument("--lambda-active", type=float, default=None)
    parser.add_argument("--lambda-n-active", type=float, default=None)
    parser.add_argument("--lambda-fractions", type=float, default=None)
    parser.add_argument("--lambda-grain-size", type=float, default=None)
    parser.add_argument("--lambda-sv-frac", type=float, default=None)
    parser.add_argument("--lambda-sv-gs", type=float, default=None)
    parser.add_argument("--lambda-coverage", type=float, default=None)
    parser.add_argument("--phase-embed-dim", type=int, default=None)
    parser.add_argument("--sv-temp-end", type=float, default=None)
    parser.add_argument("--sinkhorn-blur-end", type=float, default=None)
    parser.add_argument("--sv-use-nuc-weights", type=str, default=None, choices=["true", "false"],
                        help="A3a: 'false' to classic Voronoi (no Apollonius weighting)")
    parser.add_argument("--sv-apollonius-mode", type=str, default=None, choices=["sqrt", "linear", "const"],
                        help="A3b/A3c: delay functional form (baseline 'sqrt')")
    parser.add_argument("--sv-nuc-ratio-const", type=float, default=None,
                        help="A3b: constant nuc_ratio value (w̄ from L0 val set)")
    parser.add_argument("--disable-z-token", action="store_true",
                        help="A1: force z=0 (inference only ignores z anyway; this ablates training-time noise)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default 42 from config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="override config.checkpoint_dir (per-ablation isolation)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="override config.log_dir")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt",
                        help="filename for best checkpoint (e.g. best_model_sv5.pt)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="disable MLflow tracking (default: enabled, http://localhost:5000)")
    parser.add_argument("--arch-variant", type=str, default=None,
                        help="label for this architecture variant (MLflow tag; overrides MLFLOW_ARCH_VARIANT)")
    # Architecture hyperparameter overrides (for arch ablation)
    parser.add_argument("--d-model", type=int, default=None, help="transformer d_model")
    parser.add_argument("--n-heads", type=int, default=None, help="attention heads (must divide d_model)")
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--n-decoder-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None, help="noise z vector size")
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()
    if args.arch_variant:
        os.environ["MLFLOW_ARCH_VARIANT"] = args.arch_variant
    cfg_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        enable_augmentation=args.augmentation,
        early_stopping_patience=args.patience,
        inference_eval_every=args.inference_every,
        cr_min=args.cr_min,
        cr_max=args.cr_max,
    )
    if args.num_points is not None:
        cfg_kwargs["num_points"] = args.num_points
    if args.lambda_s2 is not None:
        cfg_kwargs["lambda_s2"] = args.lambda_s2
    if args.lambda_boundary is not None:
        cfg_kwargs["lambda_boundary"] = args.lambda_boundary
    if args.lambda_pair_dist is not None:
        cfg_kwargs["lambda_pair_dist"] = args.lambda_pair_dist
    if args.lambda_phase_sep is not None:
        cfg_kwargs["lambda_phase_sep"] = args.lambda_phase_sep
    if args.phase_sep_delta is not None:
        cfg_kwargs["phase_sep_delta"] = args.phase_sep_delta
    if args.lambda_cct is not None:
        cfg_kwargs["lambda_cct"] = args.lambda_cct
    # Ablation overrides
    for _name in (
        "lambda_sinkhorn", "lambda_marg_kl", "lambda_active", "lambda_n_active",
        "lambda_fractions", "lambda_grain_size", "lambda_sv_frac", "lambda_sv_gs",
        "lambda_coverage", "phase_embed_dim", "sv_temp_end", "sinkhorn_blur_end",
        "sv_apollonius_mode", "sv_nuc_ratio_const",
        "d_model", "n_heads", "n_encoder_layers", "n_decoder_layers",
        "dim_feedforward", "latent_dim", "dropout",
    ):
        _val = getattr(args, _name, None)
        if _val is not None:
            cfg_kwargs[_name] = _val
    if args.sv_use_nuc_weights is not None:
        cfg_kwargs["sv_use_nuc_weights"] = (args.sv_use_nuc_weights == "true")
    if args.disable_z_token:
        cfg_kwargs["disable_z_token"] = True
    if args.checkpoint_dir is not None:
        cfg_kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        cfg_kwargs["log_dir"] = args.log_dir
    if args.seed is not None:
        cfg_kwargs["random_seed"] = args.seed
    cfg_kwargs["phase_embed_trainable"] = not args.frozen_phase_embed
    config = CVAEConfig(**cfg_kwargs)
    train(config, checkpoint_name=args.checkpoint_name, use_mlflow=not args.no_mlflow)
