"""Inference API for generating SSRVEs from a trained model.

Two entry points:

1. ``predict(model, cr, config)``: deterministic prediction of phase
   composition and grain size from the auxiliary heads. Does not require
   Voronoi growth; returns a single dict of scalar values.

2. ``generate_ssrves(model, cr, config, n, out_dir)``: generates N
   distinct SSRVE microstructures (random xy positions), writes PNG
   images, per-sample starting-point CSVs, and a summary parameter
   table. Each microstructure has a different spatial arrangement of
   grains but global statistics (fractions, mean grain size) close to
   the deterministic prediction from the auxiliary heads.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from multiphasessrve.ml.config import CVAEConfig
from multiphasessrve.ml.evaluate import (
    PHASES,
    PHASE_NAMES,
    denormalize_candidate,
    generate_and_evaluate,
    load_model,
)
from multiphasessrve.ml.model import SSRVEGenerator
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.analysis.analysis import SSRVEAnalysis


def _curve_to_padded_tensors(
    curve: list[tuple[float, float]], config: CVAEConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (t, T) list to padded (1, curve_max_len, 2) tensors."""
    norm = config.normalize_curve(curve)
    L = config.curve_max_len
    pad_curve = np.zeros((1, L, 2), dtype=np.float32)
    pad_mask = np.zeros((1, L), dtype=np.bool_)
    for i, (t, T) in enumerate(norm[:L]):
        pad_curve[0, i] = (t, T)
        pad_mask[0, i] = True
    return (
        torch.from_numpy(pad_curve).to(config.device),
        torch.from_numpy(pad_mask).to(config.device),
    )


@torch.no_grad()
def generate_ssrve(
    model: SSRVEGenerator,
    cooling_rate: float,
    config: CVAEConfig,
    n_candidates: int = 10,
) -> Ssrve:
    """Generate best SSRVE for given cooling rate.

    Generates n_candidates, evaluates each through growth + analysis,
    and returns the one with lowest combined score.
    """
    results = generate_and_evaluate(model, cooling_rate, config, n_candidates)
    best = results[0]
    best_idx = best["candidate_idx"]

    # Re-generate to get raw params for the best candidate
    curve = config.cooling_curve_from_cr(cooling_rate)
    curve_t, mask_t = _curve_to_padded_tensors(curve, config)
    gen = model.generate_params(curve_t, mask_t, n=n_candidates)

    starting_points = denormalize_candidate(
        gen["phase_id"][best_idx].cpu().numpy(),
        gen["nuc"][best_idx].cpu().numpy(),
        gen["mask"][best_idx].cpu().numpy(),
        config,
    )
    ssrve = Ssrve(
        rows=config.grid_size,
        cols=config.grid_size,
        starting_points=starting_points,
        phases=PHASES,
    )
    ssrve.create_ssrve(enable_merging=False)
    return ssrve


@torch.no_grad()
def generate_from_curve(
    model: SSRVEGenerator,
    cooling_curve: list[tuple[float, float]],
    config: CVAEConfig,
    n_candidates: int = 10,
) -> list[dict]:
    """Generate raw params from arbitrary cooling curve (no Ssrve growth).

    Args:
        model: trained model
        cooling_curve: list of (time_seconds, temperature_celsius) points
        config: model config
        n_candidates: number of candidates

    Returns:
        List of dicts with xy/phase_id/nuc/mask + auxiliary head predictions.
    """
    model.eval()
    curve_t, mask_t = _curve_to_padded_tensors(cooling_curve, config)
    gen = model.generate_params(curve_t, mask_t, n=n_candidates)

    # Auxiliary heads come from the underlying forward; re-run to get them.
    curve_batch = curve_t.expand(n_candidates, -1, -1)
    mask_batch = mask_t.expand(n_candidates, -1)
    out = model(curve_batch, mask_batch)
    pred_fractions = out["pred_fractions"].cpu().numpy()
    pred_grain_size = out["pred_grain_size"].cpu().numpy()

    phase_id = gen["phase_id"].cpu().numpy()
    nuc = gen["nuc"].cpu().numpy()
    mask = gen["mask"].cpu().numpy()

    results = []
    for i in range(n_candidates):
        results.append(
            {
                "phase_id": phase_id[i],
                "nuc": nuc[i],
                "mask": mask[i],
                "n_active": int(mask[i].sum()),
                "predicted_fractions": {
                    p: float(pred_fractions[i, j]) for j, p in enumerate(PHASE_NAMES)
                },
                "predicted_grain_size": float(pred_grain_size[i, 0]) * config.max_grain_size,
            }
        )

    return results

# High-level API

@torch.no_grad()
def predict(
    model: SSRVEGenerator,
    cooling_rate: float,
    config: CVAEConfig,
) -> dict:
    """Deterministic prediction of phase composition and grain size.

    Reads the auxiliary heads (fraction_head, grain_size_head,
    n_points_head) directly from the cooling-curve embedding. Does not
    require Voronoi growth; output is fully deterministic given CR.

    Args:
        model: trained model (e.g. from load_model).
        cooling_rate: cooling rate in deg C / s.
        config: model config.

    Returns:
        Dict with keys:
        - cooling_rate (float)
        - ferrite_fraction, bainite_fraction, martensite_fraction, pearlite_fraction (float)
        - ferrite_grain_size_um (float, mean ferrite grain diameter in micrometres)
        - n_active (int, predicted number of active nucleation slots)
    """
    model.eval()
    curve = config.cooling_curve_from_cr(cooling_rate)
    curve_t, mask_t = _curve_to_padded_tensors(curve, config)

    _, curve_emb = model.curve_encoder(curve_t, mask_t)
    fractions = torch.softmax(model.fraction_head(curve_emb), dim=-1)[0].cpu().numpy()
    gs_norm = model.grain_size_head(curve_emb)[0, 0].cpu().item()
    n_ratio = model.n_points_head(curve_emb)[0, 0].cpu().item()

    return {
        "cooling_rate": cooling_rate,
        "ferrite_fraction": float(fractions[0]),
        "bainite_fraction": float(fractions[1]),
        "martensite_fraction": float(fractions[2]),
        "pearlite_fraction": float(fractions[3]),
        "ferrite_grain_size_um": float(gs_norm * config.max_grain_size),
        "n_active": int(round(n_ratio * config.num_points)),
    }


def generate_ssrves(
    model: SSRVEGenerator,
    cooling_rate: float,
    config: CVAEConfig,
    n: int = 10,
    out_dir: str | Path | None = None,
    seed: int = 42,
    n_candidates: int = 20,
) -> list[dict]:
    """Generate N SSRVE microstructures for a given cooling rate.

    Produces n_candidates candidates and returns the top N (ranked by
    mean_fraction_error + 0.5 * gs_error_norm). Each kept microstructure
    has a different spatial arrangement of grains; global statistics
    (fractions, mean grain size) stay close to the deterministic head
    prediction.

    If out_dir is given, writes:
    - ssrve_00.png ... ssrve_{N-1}.png (microstructure images)
    - starting_points_00.csv ... (per-SSRVE seed positions)
    - parameters.csv (per-sample summary statistics)
    - prediction.csv (deterministic prediction from the heads)

    Args:
        model: trained model.
        cooling_rate: cooling rate in deg C / s.
        config: model config.
        n: number of microstructures to return.
        out_dir: output directory (None disables file writing).
        seed: random seed for reproducibility.
        n_candidates: number of candidates to draw before selection (>= n).

    Returns:
        List of N dicts, each with keys:
        - ssrve: Ssrve instance after growth
        - starting_points: list of (x, y, phase, compat, threshold, nuc)
        - predicted_fractions: dict {phase_name: float}
        - predicted_grain_size: float (um)
        - mean_fraction_error: float
        - n_active: int
    """
    model.eval()
    if n_candidates < n:
        n_candidates = n

    head_pred = predict(model, cooling_rate, config)

    candidates = generate_and_evaluate(model, cooling_rate, config, n_candidates=n_candidates)
    selected = candidates[:n]

    # Raw model parameters for SSRVE construction
    curve = config.cooling_curve_from_cr(cooling_rate)
    curve_t, mask_t = _curve_to_padded_tensors(curve, config)
    gen = model.generate_params(curve_t, mask_t, n=n_candidates)
    phase_id_all = gen["phase_id"].cpu().numpy()
    nuc_all = gen["nuc"].cpu().numpy()
    mask_all = gen["mask"].cpu().numpy()
    xy_pred_all = gen["xy"].cpu().numpy() if "xy" in gen else None  # (N, P, 2) or None

    rng = np.random.default_rng(seed)
    results = []
    for rank, cand in enumerate(selected):
        idx = cand["candidate_idx"]
        sp = denormalize_candidate(
            phase_id_all[idx], nuc_all[idx], mask_all[idx].astype(bool), config, rng=rng,
            xy_predicted=xy_pred_all[idx] if xy_pred_all is not None else None,
        )
        ssrve = Ssrve(
            rows=config.grid_size,
            cols=config.grid_size,
            starting_points=sp,
            phases=PHASES,
        )
        ssrve.create_ssrve(enable_merging=False)

        analyzer = SSRVEAnalysis(ssrve)
        pred_fractions = analyzer.analyze("phase_fractions").values
        try:
            gs = analyzer.analyze("grain_size", phase_id=1, real_size=config.real_size).values.get(
                "mean_diameter", 0.0)
        except Exception:
            gs = 0.0

        result = {
            "ssrve": ssrve,
            "starting_points": sp,
            "predicted_fractions": pred_fractions,
            "predicted_grain_size": float(gs),
            "mean_fraction_error": cand["mean_fraction_error"],
            "n_active": len(sp),
        }
        results.append(result)

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Deterministic head prediction
        with open(out_path / "prediction.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(head_pred.keys()))
            w.writeheader()
            w.writerow(head_pred)

        # Per-sample summary
        rows = []
        for rank, r in enumerate(results):
            pf = r["predicted_fractions"]
            row = {
                "sample_idx": rank,
                "cooling_rate": cooling_rate,
                "n_active": r["n_active"],
                "mean_fraction_error": r["mean_fraction_error"],
                "image": f"ssrve_{rank:02d}.png",
                "ferrite_fraction": pf.get("ferrite", 0.0),
                "bainite_fraction": pf.get("bainite", 0.0),
                "martensite_fraction": pf.get("martensite", 0.0),
                "pearlite_fraction": pf.get("pearlite", 0.0),
                "ferrite_grain_size_um": r["predicted_grain_size"],
            }
            rows.append(row)

        with open(out_path / "parameters.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Microstructure images + per-sample starting-point CSVs
        for rank, r in enumerate(results):
            r["ssrve"].save_image(str(out_path / f"ssrve_{rank:02d}.png"))
            with open(out_path / f"starting_points_{rank:02d}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x_grid", "y_grid", "phase_id", "compatibility", "threshold", "nuc_ratio"])
                for sp in r["starting_points"]:
                    w.writerow(sp)

    return results
