"""Read-only adapter to the SSRVE dataset for the DDPM baseline.

Two modes:
1. **Index build** (``build_split_index``): reuses manifest parsing + split logic
   from ``multiphasessrve.ml.dataset``. Used by ``prepare_data.py`` ONCE to
   produce ``cache/ddpm_grids.h5``.
2. **Training** (``DDPMSsrveDataset``, ``make_dataloaders``): reads only the
   pre-built HDF5 cache. No dependency on the main package at this point.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Optional path import (only needed by prepare_data.py / build_split_index).
# Training does NOT need this - it only reads ddpm_grids.h5.

def _ensure_main_pkg_importable():
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

# Index construction (used by prepare_data.py)

@dataclass
class SsrveSample:
    target_idx: int
    ssrve_id: int
    cooling_rate: float
    target_fractions: np.ndarray
    target_grain_size: float
    pkl_path: Path
    cooling_curve: np.ndarray
    curve_mask: np.ndarray


def build_split_index(cfg, dataset_root: Path) -> dict[str, list[SsrveSample]]:
    """Reproduce the transformer's stratified split with PKL paths attached.

    ``cfg`` is a ``multiphasessrve.ml.config.CVAEConfig``. Importing it lazily
    keeps this function out of the training-time dependency graph.
    """
    _ensure_main_pkg_importable()
    from multiphasessrve.ml.dataset import (
        _best_of, _filter_rows, _normalize_curve_padded, _read_manifest, _stratified_split,
    )

    target_dirs = sorted(d for d in dataset_root.glob("target_*") if d.is_dir())
    if not target_dirs:
        raise FileNotFoundError(f"No target_* under {dataset_root}")

    holdout = set(cfg.holdout_cr_indices)
    pool, holdout_rows = [], []
    for tdir in target_dirs:
        target_idx = int(tdir.name.replace("target_", ""))
        rows = _read_manifest(tdir / "manifest.csv", target_idx)
        rows = _filter_rows(rows, cfg.filter_max_frac_err, cfg.filter_max_gs_err_pct)
        if cfg.cr_min is not None:
            rows = [r for r in rows if r.cooling_rate >= cfg.cr_min]
        if cfg.cr_max is not None:
            rows = [r for r in rows if r.cooling_rate <= cfg.cr_max]
        if not rows:
            continue
        rows = _best_of(rows, cfg.best_of_per_cr)
        (holdout_rows if target_idx in holdout else pool).extend(rows)

    splits = _stratified_split(pool, cfg.split_ratios, cfg.split_seed)
    splits["holdout"] = holdout_rows

    out: dict[str, list[SsrveSample]] = {}
    for name, rows in splits.items():
        samples: list[SsrveSample] = []
        for r in rows:
            curve, cm = _normalize_curve_padded(cfg, cfg.cooling_curve_from_cr(r.cooling_rate))
            samples.append(SsrveSample(
                target_idx=r.target_idx,
                ssrve_id=r.ssrve_id,
                cooling_rate=r.cooling_rate,
                target_fractions=r.target_fractions.astype(np.float32),
                target_grain_size=float(r.target_grain_size),
                pkl_path=dataset_root / f"target_{r.target_idx:02d}" / f"ssrve_{r.ssrve_id:04d}" / "SSRVE.pkl",
                cooling_curve=curve,
                curve_mask=cm,
            ))
        out[name] = samples
    return out

# Phase-grid rasterization (only called by prepare_data.py)

def rasterize_phase_grid(pkl_path: Path, num_phases: int) -> np.ndarray:
    """Load Ssrve pickle and return uint8 phase grid at native resolution (1-indexed)."""
    _ensure_main_pkg_importable()
    with open(pkl_path, "rb") as f:
        ssrve = pickle.load(f)
    grain_arr = ssrve.arr
    out = np.zeros_like(grain_arr, dtype=np.uint8)
    for gid, prop in ssrve.grain_properties.items():
        out[grain_arr == gid] = prop.phase_id
    if out.max() > num_phases:
        raise ValueError(f"phase_id {out.max()} exceeds num_phases={num_phases}")
    return out


def downsample_nearest(grid: np.ndarray, target: int) -> np.ndarray:
    h, w = grid.shape
    iy = (np.arange(target) * h / target).astype(np.int64)
    ix = (np.arange(target) * w / target).astype(np.int64)
    return grid[iy[:, None], ix[None, :]]

# Training-time dataset (reads only the HDF5 cache)

def to_one_hot_signed(grid: np.ndarray, num_phases: int) -> np.ndarray:
    """uint8 phase ids in {1..num_phases} -> float32 (C, H, W) in {-1, +1}."""
    out = np.full((num_phases, *grid.shape), -1.0, dtype=np.float32)
    for c in range(num_phases):
        out[c] = np.where(grid == c + 1, 1.0, -1.0)
    return out


class DDPMSsrveDataset(Dataset):
    """Reads pre-rasterized grids + cooling-curve conditioning from one HDF5 group."""

    def __init__(self, cache_path: Path, split: str, num_phases: int):
        self.cache_path = Path(cache_path)
        self.split = split
        self.num_phases = num_phases
        # Open lazily per-worker; just probe length here.
        with h5py.File(self.cache_path, "r") as f:
            self._len = f[split]["grids"].shape[0]
            self.grid_size = int(f.attrs["grid_size"])
        self._h: h5py.File | None = None

    def _open(self):
        if self._h is None:
            self._h = h5py.File(self.cache_path, "r")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._open()
        g = self._h[self.split]
        grid = g["grids"][idx]
        x = to_one_hot_signed(grid, self.num_phases)
        return {
            "x": torch.from_numpy(x),
            "cooling_curve": torch.from_numpy(g["cooling_curve"][idx]),
            "curve_mask": torch.from_numpy(g["curve_mask"][idx]),
            "cr": torch.tensor(g["cr"][idx], dtype=torch.float32),
            "target_fractions": torch.from_numpy(g["target_fractions"][idx]),
            "target_grain_size": torch.tensor(g["target_grain_size"][idx], dtype=torch.float32),
            "target_idx": torch.tensor(g["target_idx"][idx], dtype=torch.long),
        }


def make_dataloaders(
    cache_path: Path,
    num_phases: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders = {}
    with h5py.File(cache_path, "r") as f:
        split_names = list(f.keys())
    for name in split_names:
        ds = DDPMSsrveDataset(cache_path, name, num_phases)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(name == "train"),
            persistent_workers=(num_workers > 0),
        )
    return loaders
