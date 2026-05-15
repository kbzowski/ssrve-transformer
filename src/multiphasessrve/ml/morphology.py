"""Morphological descriptors for SSRVE microstructures.

Three complementary descriptors:

1. S₂ (two-point statistics) - per-phase autocorrelation function.
   S₂_k(r) = P(pixel at x is phase k AND pixel at x+r is phase k).
   Captures spatial scale (grain size) and distribution of each phase.
   Computed via FFT: S₂(r) = IFFT(|FFT(φ_k)|²) / G².

2. Boundary ratio - per-phase fraction of pixels adjacent to a different phase.
   Captures grain boundary density (inversely related to grain size).

3. Pair distribution - histogram of inter-nuclei distances, per phase pair.
   Same-phase (g_kk): captures nuclei clustering / anti-clustering.
   Cross-phase (g_kj): captures phase co-localization.

Targets are computed once from the CMA-ES dataset PKL files and stored in
cache/morphology_targets.npz. During training, differentiable versions are
computed from SoftVoronoi's pixel_phase_2d (S₂, boundary) and from
predicted xy positions (pair distribution).

Distance convention: normalized to [0, 1]² domain in all functions.
r_max = 0.5 ≈ half-domain (25 μm for real_size=50 μm).
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from multiphasessrve.ml.config import CVAEConfig

# ---- bin definitions (shared between numpy and torch paths) ----------------

S2_N_BINS: int = 32
PAIR_N_BINS: int = 16
R_MAX_NORM: float = 0.5   # half-domain in normalized [0,1]² coords
PAIR_MIN_NUCLEI: int = 2  # skip pair distribution if fewer nuclei for a phase


def _s2_bin_edges(n_bins: int = S2_N_BINS, r_max: float = R_MAX_NORM) -> np.ndarray:
    """Sqrt-spaced bin edges in [0, r_max] - finer resolution near r=0."""
    return np.sqrt(np.linspace(0.0, r_max ** 2, n_bins + 1))


def _pair_bin_edges(n_bins: int = PAIR_N_BINS, r_max: float = R_MAX_NORM) -> np.ndarray:
    """Linear bin edges for pair distribution."""
    return np.linspace(0.0, r_max, n_bins + 1)

# Numpy routines - used in build_morphology_targets (dataset cache build)

def _s2_numpy(
    phase_arr: np.ndarray,
    num_phases: int,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Radially-averaged S₂ for each phase via FFT autocorrelation.

    Args:
        phase_arr: (G, G) int array of phase IDs (1-indexed).
        num_phases: number of phases C.
        bin_edges:  (n_bins+1,) normalized bin edges.

    Returns:
        s2: (C, n_bins) float32
    """
    G = phase_arr.shape[0]
    n_bins = len(bin_edges) - 1
    s2 = np.zeros((num_phases, n_bins), dtype=np.float32)

    # Distance from fftshift center, normalized by G
    cy, cx = G // 2, G // 2
    y_idx, x_idx = np.mgrid[0:G, 0:G]
    r_norm = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / G

    for k in range(num_phases):
        phi = (phase_arr == (k + 1)).astype(np.float32)
        F = np.fft.fft2(phi)
        s2_2d = np.real(np.fft.ifft2(F * np.conj(F))) / (G * G)
        s2_2d = np.fft.fftshift(s2_2d)
        for i in range(n_bins):
            ring = (r_norm >= bin_edges[i]) & (r_norm < bin_edges[i + 1])
            if ring.sum() > 0:
                s2[k, i] = float(s2_2d[ring].mean())

    return s2


def _boundary_ratio_numpy(phase_arr: np.ndarray, num_phases: int) -> np.ndarray:
    """Per-phase fraction of pixels adjacent (4-connectivity) to a different phase.

    Returns:
        br: (C,) float32 in [0, 1]
    """
    br = np.zeros(num_phases, dtype=np.float32)
    for k in range(num_phases):
        phi = phase_arr == (k + 1)
        n_phase = int(phi.sum())
        if n_phase == 0:
            continue
        boundary = np.zeros(phi.shape, dtype=bool)
        boundary[:-1, :] |= phi[:-1, :] & ~phi[1:, :]
        boundary[1:, :] |= phi[1:, :] & ~phi[:-1, :]
        boundary[:, :-1] |= phi[:, :-1] & ~phi[:, 1:]
        boundary[:, 1:] |= phi[:, 1:] & ~phi[:, :-1]
        boundary &= phi
        br[k] = float(boundary.sum()) / n_phase
    return br


def _pair_dist_numpy(
    xy_norm: np.ndarray,
    phase_ids: np.ndarray,
    num_phases: int,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized pair distribution histograms from nuclei positions.

    Args:
        xy_norm:   (N, 2) positions in [0, 1]²
        phase_ids: (N,) int, values 1..num_phases
        bin_edges: (n_bins+1,) linear bin edges

    Returns:
        same_phase:  (C, n_bins) - normalized histogram for same-phase pairs
        cross_phase: (C, C, n_bins) - cross-phase (k≠j only, k to j row)
    """
    N = len(xy_norm)
    n_bins = len(bin_edges) - 1
    same = np.zeros((num_phases, n_bins), dtype=np.float32)
    cross = np.zeros((num_phases, num_phases, n_bins), dtype=np.float32)

    if N < 2:
        return same, cross

    diff = xy_norm[:, None, :] - xy_norm[None, :, :]  # (N, N, 2)
    dists = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)  # (N, N)
    r_max = float(bin_edges[-1])

    for k in range(num_phases):
        idx_k = np.where(phase_ids == (k + 1))[0]
        n_k = len(idx_k)

        if n_k >= PAIR_MIN_NUCLEI:
            d_kk = dists[np.ix_(idx_k, idx_k)]
            # Upper triangle only (undirected pairs)
            upper = np.triu(np.ones((n_k, n_k), dtype=bool), k=1)
            vals = d_kk[upper]
            vals = vals[vals <= r_max]
            if len(vals) > 0:
                counts, _ = np.histogram(vals, bins=bin_edges)
                same[k] = counts / max(int(counts.sum()), 1)

        for j in range(num_phases):
            if j == k:
                continue
            idx_j = np.where(phase_ids == (j + 1))[0]
            n_j = len(idx_j)
            if n_k < 1 or n_j < 1:
                continue
            vals = dists[np.ix_(idx_k, idx_j)].flatten()
            vals = vals[vals <= r_max]
            if len(vals) > 0:
                counts, _ = np.histogram(vals, bins=bin_edges)
                cross[k, j] = counts / max(int(counts.sum()), 1)

    return same, cross


def _process_pkl(
    pkl_path: Path,
    num_phases: int,
    s2_bins: np.ndarray,
    pair_bins: np.ndarray,
) -> dict | None:
    """Load one SSRVE PKL and compute all three descriptors."""
    try:
        with open(pkl_path, "rb") as f:
            ssrve = pickle.load(f)
        phase_arr = ssrve.get_phase_array()

        s2 = _s2_numpy(phase_arr, num_phases, s2_bins)
        br = _boundary_ratio_numpy(phase_arr, num_phases)

        sp = ssrve.starting_points_array  # (N, 6)
        xy_norm = sp[:, :2] / max(phase_arr.shape[0] - 1, 1)
        phase_ids = sp[:, 2].astype(int)
        same, cross = _pair_dist_numpy(xy_norm, phase_ids, num_phases, pair_bins)

        return {"s2": s2, "br": br, "pair_same": same, "pair_cross": cross}
    except Exception:
        return None


def build_morphology_targets(
    dataset_root: Path | str,
    config: CVAEConfig,
    cache_dir: Path | str,
    num_workers: int = 4,
    force: bool = False,
) -> Path:
    """Compute per-CR morphological targets and save to cache/morphology_targets.npz.

    Processes all PKL files in dataset_root/target_*/ssrve_*/.
    Results are averaged per target_idx (CR) and stored as arrays indexed
    by target_idx (0..n_targets-1).

    Args:
        dataset_root: path containing target_NN directories
        config:       CVAEConfig (num_phases, dataset_dir used as fallback)
        cache_dir:    directory where morphology_targets.npz is written
        num_workers:  parallel workers for PKL loading (0 = single-threaded)
        force:        rebuild even if npz already exists

    Returns:
        Path to the written morphology_targets.npz file.
    """
    dataset_root = Path(dataset_root)
    cache_dir = Path(cache_dir)
    out_path = cache_dir / "morphology_targets.npz"

    if out_path.exists() and not force:
        print(f"Morphology targets up-to-date: {out_path}", flush=True)
        return out_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    s2_bins = _s2_bin_edges(S2_N_BINS, R_MAX_NORM)
    pair_bins = _pair_bin_edges(PAIR_N_BINS, R_MAX_NORM)

    # Discover target dirs
    target_dirs = sorted(d for d in dataset_root.glob("target_*") if d.is_dir())
    if not target_dirs:
        raise FileNotFoundError(f"No target_* dirs in {dataset_root}")

    # Aggregate per target_idx
    n_targets = len(target_dirs)
    C = config.num_phases

    s2_agg = np.zeros((n_targets, C, S2_N_BINS), dtype=np.float32)
    br_agg = np.zeros((n_targets, C), dtype=np.float32)
    pair_same_agg = np.zeros((n_targets, C, PAIR_N_BINS), dtype=np.float32)
    pair_cross_agg = np.zeros((n_targets, C, C, PAIR_N_BINS), dtype=np.float32)
    counts_agg = np.zeros(n_targets, dtype=np.int32)
    target_indices = np.array(
        [int(d.name.replace("target_", "")) for d in target_dirs], dtype=np.int32
    )

    for ti, tdir in enumerate(target_dirs):
        pkl_paths = sorted(tdir.glob("ssrve_*/SSRVE.pkl"))
        if not pkl_paths:
            continue

        n = len(pkl_paths)
        print(f"  target_{target_indices[ti]:02d}: {n} PKL files...", flush=True)

        results = []
        if num_workers <= 1 or n < 4:
            for p in pkl_paths:
                r = _process_pkl(p, C, s2_bins, pair_bins)
                if r is not None:
                    results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futs = {ex.submit(_process_pkl, p, C, s2_bins, pair_bins): p for p in pkl_paths}
                for fut in as_completed(futs):
                    r = fut.result()
                    if r is not None:
                        results.append(r)

        if not results:
            continue

        s2_agg[ti] = np.mean([r["s2"] for r in results], axis=0)
        br_agg[ti] = np.mean([r["br"] for r in results], axis=0)
        pair_same_agg[ti] = np.mean([r["pair_same"] for r in results], axis=0)
        pair_cross_agg[ti] = np.mean([r["pair_cross"] for r in results], axis=0)
        counts_agg[ti] = len(results)
        print(f"    -> aggregated {len(results)} samples", flush=True)

    np.savez(
        out_path,
        s2=s2_agg,
        boundary_ratio=br_agg,
        pair_same=pair_same_agg,
        pair_cross=pair_cross_agg,
        target_indices=target_indices,
        n_samples=counts_agg,
        s2_bin_edges=s2_bins,
        pair_bin_edges=pair_bins,
    )
    print(f"Morphology targets saved: {out_path}", flush=True)
    return out_path


def load_morphology_targets(cache_dir: Path | str) -> dict[int, dict[str, np.ndarray]]:
    """Load morphology_targets.npz and return lookup dict {target_idx: dict}.

    Each inner dict has keys: s2, boundary_ratio, pair_same, pair_cross.
    """
    path = Path(cache_dir) / "morphology_targets.npz"
    if not path.exists():
        raise FileNotFoundError(f"Morphology targets not found: {path}. Run build_morphology_targets first.")
    data = np.load(path)
    target_indices = data["target_indices"]
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for i, tidx in enumerate(target_indices):
        lookup[int(tidx)] = {
            "s2": data["s2"][i],                       # (C, n_bins_s2)
            "boundary_ratio": data["boundary_ratio"][i],  # (C,)
            "pair_same": data["pair_same"][i],          # (C, n_bins_pair)
            "pair_cross": data["pair_cross"][i],        # (C, C, n_bins_pair)
        }
    return lookup

# Torch differentiable losses - used in train.py compute_loss

class MorphologyLoss(nn.Module):
    """Differentiable morphological losses from SoftVoronoi pixel_phase_2d and xy.

    Registered buffers:
        s2_bin_edges:    (n_bins_s2+1,) sqrt-spaced
        pair_bin_edges:  (n_bins_pair+1,) linear
        pair_bin_sigmas: (n_bins_pair,) soft histogram bandwidth per bin

    Usage in training loop:
        morpho = MorphologyLoss(config).to(device)
        ...
        l_s2, l_br, l_pair = morpho(pixel_phase_2d, xy, phase_probs,
                                     active_prob, target_s2, target_br,
                                     target_pair_same, target_frac)
    """

    def __init__(self, config: CVAEConfig):
        super().__init__()
        s2_edges = torch.tensor(_s2_bin_edges(S2_N_BINS, R_MAX_NORM), dtype=torch.float32)
        pair_edges = torch.tensor(_pair_bin_edges(PAIR_N_BINS, R_MAX_NORM), dtype=torch.float32)
        # Sigma per bin = bin width (ensures ~1 bin overlap, smooth histogram)
        pair_sigma = (pair_edges[1:] - pair_edges[:-1])  # (n_bins_pair,)

        self.register_buffer("s2_bin_edges", s2_edges)
        self.register_buffer("pair_bin_edges", pair_edges)
        self.register_buffer("pair_bin_sigmas", pair_sigma)

        self.num_phases = config.num_phases
        G = config.sv_grid_size
        # Precompute normalized distance map for S₂ radial average
        cy, cx = G // 2, G // 2
        y_idx, x_idx = torch.meshgrid(
            torch.arange(G, dtype=torch.float32),
            torch.arange(G, dtype=torch.float32),
            indexing="ij",
        )
        r_map = torch.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / G  # (G, G)
        self.register_buffer("r_map", r_map)

    def _compute_s2(self, pixel_phase_2d: torch.Tensor) -> torch.Tensor:
        """S₂ from SoftVoronoi pixel_phase.

        Args:
            pixel_phase_2d: (B, G, G, C)
        Returns:
            s2: (B, C, n_bins_s2)
        """
        B, G, _, C = pixel_phase_2d.shape
        n_bins = len(self.s2_bin_edges) - 1
        s2 = torch.zeros(B, C, n_bins, device=pixel_phase_2d.device)

        for k in range(C):
            phi = pixel_phase_2d[:, :, :, k]  # (B, G, G)
            F = torch.fft.fft2(phi)
            s2_2d = torch.fft.ifft2(F * torch.conj(F)).real / (G * G)  # (B, G, G)
            s2_2d = torch.fft.fftshift(s2_2d, dim=(-2, -1))

            for i in range(n_bins):
                lo = float(self.s2_bin_edges[i].item())
                hi = float(self.s2_bin_edges[i + 1].item())
                ring = (self.r_map >= lo) & (self.r_map < hi)  # (G, G) bool
                if ring.sum() > 0:
                    s2[:, k, i] = s2_2d[:, ring].mean(dim=-1)

        return s2

    def _compute_boundary(self, pixel_phase_2d: torch.Tensor) -> torch.Tensor:
        """Soft boundary ratio from pixel_phase.

        Args:
            pixel_phase_2d: (B, G, G, C)
        Returns:
            br: (B, C)
        """
        B, G, _, C = pixel_phase_2d.shape
        eps = 1e-8
        br = torch.zeros(B, C, device=pixel_phase_2d.device)
        for k in range(C):
            phi = pixel_phase_2d[:, :, :, k]  # (B, G, G)
            # 4-connectivity boundary signal (soft - gradient flows through phi)
            diff_y = (phi[:, 1:, :] - phi[:, :-1, :]).abs()   # (B, G-1, G)
            diff_x = (phi[:, :, 1:] - phi[:, :, :-1]).abs()   # (B, G, G-1)
            boundary_sum = diff_y.sum(dim=(-2, -1)) + diff_x.sum(dim=(-2, -1))  # (B,)
            phase_area = phi.sum(dim=(-2, -1))                                    # (B,)
            br[:, k] = boundary_sum / phase_area.clamp(min=eps)
        return br

    def _compute_pair_dist(
        self,
        xy: torch.Tensor,
        phase_probs: torch.Tensor,
        active_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Soft pair distribution histogram per same-phase pair.

        Uses Gaussian soft histogram so gradient flows through distances.

        Args:
            xy:           (B, P, 2)
            phase_probs:  (B, P, C)
            active_prob:  (B, P)

        Returns:
            pair_same: (B, C, n_bins_pair)
        """
        B, P, C = phase_probs.shape
        n_bins = len(self.pair_bin_edges) - 1
        bin_centers = 0.5 * (self.pair_bin_edges[:-1] + self.pair_bin_edges[1:])  # (n_bins,)
        sigmas = self.pair_bin_sigmas  # (n_bins,)

        # Pairwise distances: (B, P, P)
        diff = xy.unsqueeze(2) - xy.unsqueeze(1)
        dists = torch.sqrt((diff ** 2).sum(-1) + 1e-12)  # (B, P, P)

        # Mask upper triangle (undirected pairs, diagonal=1 to exclude self)
        upper = torch.ones(P, P, dtype=torch.bool, device=xy.device).triu(diagonal=1)

        pair_same = torch.zeros(B, C, n_bins, device=xy.device)
        eps = 1e-8

        for k in range(C):
            w_k = phase_probs[:, :, k] * active_prob  # (B, P)
            # Weight of pair (i, j): w_k_i × w_k_j
            w_kk = w_k.unsqueeze(2) * w_k.unsqueeze(1)  # (B, P, P)
            w_kk = w_kk * upper.unsqueeze(0)            # upper triangle only

            # Soft histogram via Gaussian kernel: (B, P, P, n_bins)
            dists_exp = dists.unsqueeze(-1)                  # (B, P, P, 1)
            centers = bin_centers.view(1, 1, 1, n_bins)      # (1, 1, 1, n_bins)
            sigs = sigmas.view(1, 1, 1, n_bins)
            gauss = torch.exp(-0.5 * ((dists_exp - centers) / sigs) ** 2)  # (B, P, P, n_bins)

            # Weighted sum over all pairs: (B, n_bins)
            weighted = (w_kk.unsqueeze(-1) * gauss).sum(dim=(1, 2))  # (B, n_bins)
            # Normalize to sum = 1
            total = weighted.sum(dim=-1, keepdim=True).clamp(min=eps)
            pair_same[:, k, :] = weighted / total

        return pair_same

    def forward(
        self,
        pixel_phase_2d: torch.Tensor,
        xy: torch.Tensor,
        phase_probs: torch.Tensor,
        active_prob: torch.Tensor,
        target_s2: torch.Tensor,
        target_br: torch.Tensor,
        target_pair_same: torch.Tensor,
        target_frac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute morphological losses.

        Losses are weighted by target phase fractions: absent phases (target=0)
        contribute zero, rare phases contribute proportionally less than
        dominant ones. This prevents the model wasting capacity on absent phases.

        Args:
            pixel_phase_2d:   (B, G, G, C) from SoftVoronoi
            xy:               (B, P, 2) predicted nucleation positions
            phase_probs:      (B, P, C) phase probabilities
            active_prob:      (B, P)
            target_s2:        (B, C, n_bins_s2)
            target_br:        (B, C)
            target_pair_same: (B, C, n_bins_pair)
            target_frac:      (B, C) target phase fractions (for weighting)

        Returns:
            l_s2, l_boundary, l_pair_same - scalar losses
        """
        eps = 1e-8
        # Phase-fraction weights: (B, C) normalized, rare phases count less
        w_phase = target_frac / target_frac.sum(dim=-1, keepdim=True).clamp(min=eps)  # (B, C)

        # --- S₂ loss ---
        pred_s2 = self._compute_s2(pixel_phase_2d)  # (B, C, n_bins_s2)
        err_s2 = (pred_s2 - target_s2).pow(2)        # (B, C, n_bins_s2)
        # Weight each phase by its fraction; average over bins and batch
        l_s2 = (err_s2.mean(dim=-1) * w_phase).sum(dim=-1).mean()

        # --- Boundary ratio loss ---
        pred_br = self._compute_boundary(pixel_phase_2d)  # (B, C)
        err_br = (pred_br - target_br).pow(2)
        l_br = (err_br * w_phase).sum(dim=-1).mean()

        # --- Pair distribution loss (same-phase only) ---
        pred_pair = self._compute_pair_dist(xy, phase_probs, active_prob)  # (B, C, n_bins)
        err_pair = (pred_pair - target_pair_same).pow(2)
        l_pair = (err_pair.mean(dim=-1) * w_phase).sum(dim=-1).mean()

        return l_s2, l_br, l_pair
