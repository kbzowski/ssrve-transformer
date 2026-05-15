"""End-to-end evaluation: generate SSRVE params to growth to analysis to compare."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from multiphasessrve.ml.config import CVAEConfig
from multiphasessrve.ml.model import SSRVEGenerator
from multiphasessrve.ml.placement import get_placement_function
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.analysis.analysis import SSRVEAnalysis
from multiphasessrve.types.phase import PhaseModel, PhasesModel


PHASES = PhasesModel(
    root=[
        PhaseModel(id=1, color=(29, 143, 255), name="ferrite"),
        PhaseModel(id=2, color=(172, 255, 46), name="bainite"),
        PhaseModel(id=3, color=(255, 0, 0), name="martensite"),
        PhaseModel(id=4, color=(50, 50, 50), name="pearlite"),
    ]
)

PHASE_NAMES = ("ferrite", "bainite", "martensite", "pearlite")


def denormalize_candidate(
    phase_id: np.ndarray,
    nuc: np.ndarray,
    mask: np.ndarray,
    config: CVAEConfig,
    rng: np.random.Generator | None = None,
    placement: str = "uniform",
    xy_predicted: np.ndarray | None = None,
) -> list[tuple[int, int, int, float, float, float]]:
    """Convert per-point arrays to a starting_points list for Ssrve.

    xy positions come from one of two sources (xy_predicted takes priority):
    1. xy_predicted (P, 2): direct model prediction (xy_head via SoftVoronoi),
       used when the model was trained end-to-end with Soft Voronoi.
    2. placement strategy: fallback that draws xy positions according to
       the named strategy (phase_poisson, uniform, ...) when xy_predicted
       is None.

    compat and threshold are not optimised in this pipeline; constants from
    the config are used and ssrve growth treats them as inert.

    Args:
        phase_id:     (P,) in 1..num_phases
        nuc:          (P,) clamped to [0, max_nucleation_ratio]
        mask:         (P,) bool, True = active slot
        rng:          optional numpy Generator for reproducible xy sampling
        placement:    fallback strategy name ("phase_poisson" | "uniform")
        xy_predicted: (P, 2) optional model-predicted positions in [0, 1]^2;
                      used in place of the placement strategy if given

    Returns:
        list of (x_int, y_int, phase_id, compat, threshold, nuc_ratio)
    """
    active_idx = np.nonzero(mask)[0]
    if active_idx.size == 0:
        return []

    if rng is None:
        rng = np.random.default_rng()

    # SSRVE growth requires at least one grain with nucleation_ratio=0.0
    nuc_active = nuc[active_idx]
    min_local = int(np.argmin(nuc_active))
    forced_zero = active_idx[min_local]

    if xy_predicted is not None:
        # Clamp model predictions to [0, 1] in case of small over/undershoot.
        xy_normalized = np.clip(xy_predicted[active_idx], 0.0, 1.0)
    else:
        placement_fn = get_placement_function(placement)
        active_phase = phase_id[active_idx]
        xy_normalized = placement_fn(active_idx.size, active_phase, rng)

    starting_points = []
    grid_max = config.grid_size - 1
    for local_i, i in enumerate(active_idx):
        x = int(xy_normalized[local_i, 0] * grid_max)
        y = int(xy_normalized[local_i, 1] * grid_max)
        phase = int(phase_id[i])
        nuc_ratio = 0.0 if i == forced_zero else float(nuc[i])
        starting_points.append(
            (x, y, phase, config.constant_compatibility, config.constant_threshold, nuc_ratio)
        )
    return starting_points


def _build_ssrve(starting_points, config: CVAEConfig) -> Ssrve:
    ssrve = Ssrve(
        rows=config.grid_size,
        cols=config.grid_size,
        starting_points=starting_points,
        phases=PHASES,
    )
    ssrve.create_ssrve(enable_merging=False)
    return ssrve


def evaluate_candidate(
    phase_id: np.ndarray,
    nuc: np.ndarray,
    mask: np.ndarray,
    target_fractions: dict[str, float],
    target_grain_size: float,
    config: CVAEConfig,
    rng: np.random.Generator | None = None,
    xy_predicted: np.ndarray | None = None,
) -> dict:
    """Run single candidate through growth + analysis."""
    starting_points = denormalize_candidate(phase_id, nuc, mask, config, rng, xy_predicted=xy_predicted)
    if not starting_points:
        return {
            "predicted_fractions": {p: 0.0 for p in PHASE_NAMES},
            "predicted_grain_size": 0.0,
            "fraction_errors": {p: target_fractions.get(p, 0.0) for p in PHASE_NAMES},
            "grain_size_error": target_grain_size,
            "mean_fraction_error": float("inf"),
            "n_active": 0,
        }

    ssrve = _build_ssrve(starting_points, config)
    analyzer = SSRVEAnalysis(ssrve)
    pred_fractions = analyzer.analyze("phase_fractions").values

    try:
        pred_gs_result = analyzer.analyze("grain_size", phase_id=1, real_size=config.real_size)
        pred_grain_size = pred_gs_result.values.get("mean_diameter", 0.0)
    except Exception:
        pred_grain_size = 0.0

    frac_errors = {
        phase: abs(target_fractions.get(phase, 0.0) - pred_fractions.get(phase, 0.0))
        for phase in PHASE_NAMES
    }

    return {
        "predicted_fractions": pred_fractions,
        "predicted_grain_size": pred_grain_size,
        "fraction_errors": frac_errors,
        "grain_size_error": abs(target_grain_size - pred_grain_size),
        "mean_fraction_error": float(np.mean(list(frac_errors.values()))),
        "n_active": len(starting_points),
    }


def _load_targets_from_dataset(config: CVAEConfig) -> list[dict]:
    """Read targets from one manifest row per CR (target_*/manifest.csv)."""
    dataset_root = Path(config.dataset_dir)
    out = []
    for tdir in sorted(dataset_root.glob("target_*")):
        mp = tdir / "manifest.csv"
        if not mp.exists():
            continue
        with open(mp) as f:
            row = next(csv.DictReader(f), None)
        if row is None:
            continue
        out.append(
            {
                "cooling_rate": float(row["cooling_rate"]),
                "ferrite_fraction": float(row["target_ferrite_fraction"]),
                "bainite_fraction": float(row["target_bainite_fraction"]),
                "martensite_fraction": float(row["target_martensite_fraction"]),
                "pearlite_fraction": float(row["target_pearlite_fraction"]),
                "ferrite_grain_size": float(row["target_ferrite_grain_size"]),
            }
        )
    return out


@torch.no_grad()
def generate_and_evaluate(
    model: SSRVEGenerator,
    cooling_rate: float,
    config: CVAEConfig,
    n_candidates: int = 10,
) -> list[dict]:
    """Generate N candidates for given CR and evaluate each."""
    model.eval()

    curve = config.normalize_curve(config.cooling_curve_from_cr(cooling_rate))
    L = len(curve)
    P_max = config.curve_max_len
    pad_curve = np.zeros((1, P_max, 2), dtype=np.float32)
    pad_mask = np.zeros((1, P_max), dtype=np.bool_)
    for i, (t, T) in enumerate(curve[:P_max]):
        pad_curve[0, i] = (t, T)
        pad_mask[0, i] = True
    curve_t = torch.from_numpy(pad_curve).to(config.device)
    mask_t = torch.from_numpy(pad_mask).to(config.device)

    gen = model.generate_params(curve_t, mask_t, n=n_candidates)
    phase_id = gen["phase_id"].cpu().numpy()
    nuc = gen["nuc"].cpu().numpy()
    mask = gen["mask"].cpu().numpy()
    xy_pred = gen["xy"].cpu().numpy() if "xy" in gen else None  # (N, P, 2) or None

    targets = _load_targets_from_dataset(config)
    closest = min(targets, key=lambda t: abs(t["cooling_rate"] - cooling_rate))
    target_fractions = {p: closest[f"{p}_fraction"] for p in PHASE_NAMES}
    target_grain_size = closest["ferrite_grain_size"]

    rng = np.random.default_rng()
    results = []
    for i in range(n_candidates):
        result = evaluate_candidate(
            phase_id[i], nuc[i], mask[i],
            target_fractions, target_grain_size, config,
            rng=rng,
            xy_predicted=xy_pred[i] if xy_pred is not None else None,
        )
        result["candidate_idx"] = i
        result["cooling_rate"] = cooling_rate
        results.append(result)

    for r in results:
        gs_err_norm = r["grain_size_error"] / max(target_grain_size, 1.0)
        r["combined_score"] = r["mean_fraction_error"] + 0.5 * gs_err_norm
    results.sort(key=lambda r: r["combined_score"])
    return results


def load_model(checkpoint_path: str, config: CVAEConfig | None = None) -> SSRVEGenerator:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config is None:
        config = ckpt.get("config", CVAEConfig())
    model = SSRVEGenerator(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()
    return model
