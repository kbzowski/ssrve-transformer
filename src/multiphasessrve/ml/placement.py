"""Placement strategies for nucleation xy positions.

The ML pipeline emits a deterministic set of (phase_id, nuc_ratio) tuples
from the network; the xy positions on the 500x500 grid are chosen at
inference (in `denormalize_candidate`, `evaluate.py`). This module
provides the placement strategies that pick those positions.

Two strategies are exposed:

1. `uniform`: independent uniform sampling. Baseline.

2. `phase_poisson` (default): phase-aware Poisson-disk sampling, with a
   minimum distance r_min between nuclei of the same phase that scales
   as 1 / sqrt(N_phase). Minority phases get a larger r_min; inter-phase
   distances are unconstrained. Empirically removes the same-phase
   clustering artefact of uniform sampling and reduces the per-CR
   phase-fraction bias substantially (about 3x at CR=30 and an order of
   magnitude at CR=70 on our dataset).
"""
from __future__ import annotations

import numpy as np


def place_uniform(
    n_active: int,
    phase_id: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random sampling on the unit square. Baseline.

    Args:
        n_active: number of active slots to place.
        phase_id: (n_active,) 1-indexed phase ids (unused by this strategy).
        rng: random generator.

    Returns:
        (n_active, 2) array of positions in [0, 1]^2.
    """
    return rng.uniform(0.0, 1.0, size=(n_active, 2))


def place_phase_poisson(
    n_active: int,
    phase_id: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> np.ndarray:
    """Phase-aware Poisson-disk sampling.

    For each phase, enforces a minimum distance between same-phase nuclei
    proportional to 1 / sqrt(N_phase). Minority phases get a larger
    r_min, denser phases a smaller one; inter-phase distances are
    unconstrained.

    Algorithm:
      1. Sort phases by ascending count (rarer phases placed first).
      2. For each phase compute r_min = 0.6 / sqrt(N_phase),
         clamped to [0.04, 0.15].
      3. For each slot of that phase try up to max_attempts times to find
         a position at least r_min from existing same-phase points.
      4. If unsuccessful after max_attempts attempts, fall back to a
         uniform draw.

    Args:
        n_active: number of active slots.
        phase_id: (n_active,) 1-indexed phase ids.
        rng: random generator.
        max_attempts: per-slot rejection-sampling budget.

    Returns:
        (n_active, 2) array of positions in [0, 1]^2.
    """
    xy = np.zeros((n_active, 2), dtype=np.float64)

    counts: dict[int, int] = {}
    for ph in phase_id[:n_active]:
        counts[int(ph)] = counts.get(int(ph), 0) + 1

    # Place rarer phases first (they need more space).
    order_phases = sorted(counts, key=lambda p: counts[p])

    existing_per_phase: dict[int, list[np.ndarray]] = {p: [] for p in counts}

    for ph in order_phases:
        k = counts[ph]
        r_min_phase = 0.6 / np.sqrt(max(k, 1))
        r_min_phase = float(np.clip(r_min_phase, 0.04, 0.15))
        r_min_sq = r_min_phase * r_min_phase

        slot_ids = np.where(phase_id[:n_active] == ph)[0]
        for slot in slot_ids:
            existing = existing_per_phase[ph]
            placed = False
            for _ in range(max_attempts):
                q = rng.uniform(0.0, 1.0, size=2)
                ok = True
                for p in existing:
                    dx = q[0] - p[0]
                    dy = q[1] - p[1]
                    if dx * dx + dy * dy < r_min_sq:
                        ok = False
                        break
                if ok:
                    xy[slot] = q
                    existing.append(q)
                    placed = True
                    break
            if not placed:
                xy[slot] = rng.uniform(0.0, 1.0, size=2)
                existing.append(xy[slot])

    return xy


PLACEMENT_METHODS = {
    "uniform": place_uniform,
    "phase_poisson": place_phase_poisson,
}


def get_placement_function(name: str):
    """Look up a placement function by name."""
    if name not in PLACEMENT_METHODS:
        raise ValueError(
            f"Unknown placement method: {name!r}. "
            f"Available: {list(PLACEMENT_METHODS.keys())}"
        )
    return PLACEMENT_METHODS[name]
