"""SSRVE dataset with HDF5 cache, stratified per-CR splits, and augmentation.

Pipeline
--------
1. Read manifest.csv per target_NN (29 targets, one per CR).
2. Filter samples by max_frac_err and |GS_err_pct| thresholds.
3. Best-of selection: top `best_of_per_cr` per CR sorted by combined score.
4. Stratified per-CR split to train/val/test, plus a holdout-CR set (OOD eval).
5. Parse individuals.csv per selected sample to 7-feature thermometer  to 
   4-feature (x, y, phase_id, nuc_ratio) padded to num_points with active_mask.
6. Persist all four splits to HDF5 (cache/{train,val,test,holdout}.h5).
7. SSRVEDataset reads from HDF5 in __getitem__ with optional augmentation.

The cache stores normalization stats and dataset_hash as HDF5 attrs;
trainer reads them from there (not from CVAEConfig) to avoid drift.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, WeightedRandomSampler

from multiphasessrve.ml.config import CVAEConfig
from multiphasessrve.ml.morphology import load_morphology_targets

# Constants for manifest parsing

PHASE_KEYS = ("ferrite", "bainite", "martensite", "pearlite")  # order = phase_id 1..4
SPLIT_NAMES = ("train", "val", "test", "holdout")

# Manifest parsing & filtering

@dataclass
class ManifestRow:
    target_idx: int
    ssrve_id: int
    cooling_rate: float
    target_fractions: np.ndarray  # shape (4,) - F, B, M, P (phase_id order)
    target_grain_size: float
    pred_fractions: np.ndarray  # shape (4,)
    pred_grain_size: float
    pkl_path: str
    max_frac_err: float
    gs_err_pct: float
    score: float  # combined score for best-of selection


def _frac_array(row: dict, prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_{p}_fraction"]) for p in PHASE_KEYS], dtype=np.float64)


def _read_manifest(manifest_path: Path, target_idx: int) -> list[ManifestRow]:
    """Read one manifest.csv and compute filter metrics per row."""
    if not manifest_path.exists():
        return []
    out: list[ManifestRow] = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                target_f = _frac_array(row, "target")
                pred_f = _frac_array(row, "predicted")
                target_gs = float(row["target_ferrite_grain_size"])
                pred_gs = float(row["predicted_ferrite_grain_size"])
            except (ValueError, KeyError):
                continue

            max_frac_err = float(np.max(np.abs(target_f - pred_f)))
            gs_err_pct = ((pred_gs - target_gs) / target_gs * 100.0) if target_gs > 0 else math.inf
            score = max_frac_err + abs(gs_err_pct) / 100.0

            out.append(
                ManifestRow(
                    target_idx=target_idx,
                    ssrve_id=int(row["ssrve_id"]),
                    cooling_rate=float(row["cooling_rate"]),
                    target_fractions=target_f,
                    target_grain_size=target_gs,
                    pred_fractions=pred_f,
                    pred_grain_size=pred_gs,
                    pkl_path=row["pkl_path"],
                    max_frac_err=max_frac_err,
                    gs_err_pct=gs_err_pct,
                    score=score,
                )
            )
    return out


def _filter_rows(
    rows: list[ManifestRow], max_frac_err: float, max_gs_err_pct: float
) -> list[ManifestRow]:
    return [
        r
        for r in rows
        if r.max_frac_err <= max_frac_err and abs(r.gs_err_pct) <= max_gs_err_pct
    ]


def _best_of(rows: list[ManifestRow], k: int) -> list[ManifestRow]:
    return sorted(rows, key=lambda r: r.score)[:k]

# Individuals.csv to 4-feature points

def _anchor_y_positions(num_slots: int) -> np.ndarray:
    """Fixed y-anchor positions for the `num_slots` decoder slots, evenly
    spaced in [0, 1]. Slot i is responsible for points near y = i/(P-1)."""
    return np.linspace(0.0, 1.0, num_slots, dtype=np.float64)


def _anchor_assign_points(
    pts: np.ndarray, num_slots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign active points to decoder slots by nearest y-anchor.

    Rationale: the previous canonical (y, x) sort made slot_i mean "i-th
    point from top in this sample" - a meaning that depends on n_active
    and therefore differs between cooling rates. For CR=0.5 (n=19) slot 18
    is the bottommost point at y≈0.95; for CR=10 (n=54) slot 18 is a
    middle point at y≈0.35. Learning a single query for such contradictory
    targets is near-impossible and causes the decoder to average (mode
    collapse).

    Anchor assignment fixes this by giving each slot a stable absolute
    y-region: slot i owns points near y_i = i / (num_slots - 1). Slots
    whose region contains no point are simply inactive (mask=False). This
    makes slot semantics CR-independent.

    Collisions (two points close to the same anchor) are resolved by
    rectangular Hungarian assignment on the y-distance cost matrix, so
    every target point gets a unique slot.

    Args:
        pts:        (n_active, 4) - [x, y, phase_id, nuc_ratio]
        num_slots:  number of decoder slots (P)

    Returns:
        slotted:    (num_slots, 4) - xy/phase/nuc per slot, zeros where inactive
        active:     (num_slots,) bool - True where a point was assigned
    """
    slotted = np.zeros((num_slots, 4), dtype=np.float32)
    active = np.zeros((num_slots,), dtype=bool)
    n_active = pts.shape[0]
    if n_active == 0:
        return slotted, active

    anchors = _anchor_y_positions(num_slots)
    # Cost matrix: rows = anchors (slots), cols = targets. Minimise |y_anchor - y_target|.
    cost = np.abs(anchors[:, None] - pts[:, 1][None, :])  # (P, n_active)
    # Rectangular assignment: returns (min(P, n_active)) matched pairs.
    row_ind, col_ind = linear_sum_assignment(cost)
    for slot_idx, tgt_idx in zip(row_ind, col_ind):
        slotted[slot_idx] = pts[tgt_idx]
        active[slot_idx] = True
    return slotted, active


def _parse_thermometer_individual(
    csv_path: Path, max_nucleation_ratio: float
) -> np.ndarray | None:
    """Parse best individual from optimizer output.

    individuals.csv layout: id, gen, mrse, then n_pts × 7 thermometer values
    [x, y, p1, p2, p3, p4, nuc_ratio]. Selects row with min mrse.

    Returns:
        (n_pts, 4) float32: [x_norm, y_norm, phase_id (1..num_phases), nuc_ratio_clamped]
    """
    try:
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        if not rows:
            return None
        best = min(rows, key=lambda r: float(r[2]))
        if float(best[2]) > 1e10:
            return None
        n_thermo_cols = len(best) - 3
        if n_thermo_cols % 7 != 0:
            return None
        n_pts = n_thermo_cols // 7
        thermo = np.array([float(v) for v in best[3 : 3 + n_pts * 7]], dtype=np.float32)
        thermo = thermo.reshape(n_pts, 7)

        x = thermo[:, 0]
        y = thermo[:, 1]
        phase_logits = thermo[:, 2:6]
        phase_id = phase_logits.argmax(axis=1) + 1  # 1..num_phases
        nuc = np.clip(thermo[:, 6], 0.0, max_nucleation_ratio)

        out = np.stack([x, y, phase_id.astype(np.float32), nuc], axis=1)
        return out  # (n_pts, 4)
    except (ValueError, IndexError, FileNotFoundError):
        return None

# Splits and dataset hashing

def _dataset_hash(manifest_paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(manifest_paths):
        if p.exists():
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def _stratified_split(
    rows: list[ManifestRow],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[ManifestRow]]:
    """Stratify per CR (target_idx). Each CR is split independently into 3 buckets."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[ManifestRow]] = {"train": [], "val": [], "test": []}
    by_cr: dict[int, list[ManifestRow]] = {}
    for r in rows:
        by_cr.setdefault(r.target_idx, []).append(r)

    train_r, val_r, _ = ratios
    for cr_rows in by_cr.values():
        idxs = np.arange(len(cr_rows))
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        out["train"].extend(cr_rows[i] for i in idxs[:n_train])
        out["val"].extend(cr_rows[i] for i in idxs[n_train : n_train + n_val])
        out["test"].extend(cr_rows[i] for i in idxs[n_train + n_val :])
    return out

# Cache builder

def _normalize_curve_padded(
    config: CVAEConfig, curve: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (curve_padded, curve_mask) with shape (curve_max_len, 2) / (curve_max_len,)."""
    norm = config.normalize_curve(curve)
    L = len(norm)
    pad = np.zeros((config.curve_max_len, 2), dtype=np.float32)
    mask = np.zeros((config.curve_max_len,), dtype=np.bool_)
    for i, (t, T) in enumerate(norm[: config.curve_max_len]):
        pad[i] = (t, T)
        mask[i] = True
    return pad, mask


def _pack_split(
    rows: list[ManifestRow],
    config: CVAEConfig,
    dataset_root: Path,
) -> dict[str, np.ndarray] | None:
    """Parse all rows in a split into stacked numpy arrays. Returns None if empty."""
    n = len(rows)
    if n == 0:
        return None

    P = config.num_points
    xy = np.zeros((n, P, 2), dtype=np.float32)
    phase = np.zeros((n, P), dtype=np.int8)
    nuc = np.zeros((n, P), dtype=np.float32)
    mask = np.zeros((n, P), dtype=np.bool_)

    curve = np.zeros((n, config.curve_max_len, 2), dtype=np.float32)
    curve_mask = np.zeros((n, config.curve_max_len), dtype=np.bool_)

    target_fractions = np.zeros((n, 4), dtype=np.float32)
    target_gs = np.zeros((n,), dtype=np.float32)
    cr = np.zeros((n,), dtype=np.float32)
    n_active = np.zeros((n,), dtype=np.int16)
    target_idx_arr = np.zeros((n,), dtype=np.int16)

    write_i = 0
    skipped = 0
    for r in rows:
        ssrve_dir = dataset_root / f"target_{r.target_idx:02d}" / f"ssrve_{r.ssrve_id:04d}"
        ind_csv = ssrve_dir / "individuals.csv"
        pts = _parse_thermometer_individual(ind_csv, config.max_nucleation_ratio)
        if pts is None:
            skipped += 1
            continue
        n_pts = pts.shape[0]
        if n_pts > P:
            pts = pts[:P]
            n_pts = P

        # Keep points in their natural (optimizer) order, no sort or anchor
        # assignment. Hungarian matching in compute_loss handles permutation
        # invariance during training.
        xy[write_i, :n_pts] = pts[:, :2]
        phase[write_i, :n_pts] = pts[:, 2].astype(np.int8)
        nuc[write_i, :n_pts] = pts[:, 3]
        mask[write_i, :n_pts] = True

        c, cm = _normalize_curve_padded(config, config.cooling_curve_from_cr(r.cooling_rate))
        curve[write_i] = c
        curve_mask[write_i] = cm

        target_fractions[write_i] = r.target_fractions.astype(np.float32)
        target_gs[write_i] = r.target_grain_size
        cr[write_i] = r.cooling_rate
        n_active[write_i] = n_pts
        target_idx_arr[write_i] = r.target_idx
        write_i += 1

    if write_i == 0:
        return None

    return {
        "xy": xy[:write_i],
        "phase": phase[:write_i],
        "nuc": nuc[:write_i],
        "mask": mask[:write_i],
        "cooling_curve": curve[:write_i],
        "curve_mask": curve_mask[:write_i],
        "target_fractions": target_fractions[:write_i],
        "target_grain_size": target_gs[:write_i],
        "cr": cr[:write_i],
        "n_active": n_active[:write_i],
        "target_idx": target_idx_arr[:write_i],
        "_skipped": np.array([skipped], dtype=np.int32),
    }


def _compute_norm_stats(train_pack: dict[str, np.ndarray]) -> dict[str, float]:
    cr = train_pack["cr"].astype(np.float64)
    log_cr = np.log(cr)
    gs = train_pack["target_grain_size"].astype(np.float64)
    return {
        "cr_log_mean": float(log_cr.mean()),
        "cr_log_std": float(log_cr.std() + 1e-8),
        "gs_mean": float(gs.mean()),
        "gs_std": float(gs.std() + 1e-8),
    }


def build_cache(config: CVAEConfig, force: bool = False) -> dict[str, int]:
    """Build HDF5 cache for all splits.

    Returns dict {split_name: n_samples} (after filtering, best-of, parsing).
    """
    dataset_root = Path(config.dataset_dir)
    cache_root = Path(config.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Discover targets
    target_dirs = sorted(d for d in dataset_root.glob("target_*") if d.is_dir())
    if not target_dirs:
        raise FileNotFoundError(f"No target_* dirs in {dataset_root}")

    manifest_paths = [d / "manifest.csv" for d in target_dirs]
    ds_hash = _dataset_hash(manifest_paths)
    # Append cr range + sort version to hash so cache rebuilds when filters change
    cache_version_suffix = f"_cr[{config.cr_min}:{config.cr_max}]_P{config.num_points}_rawV1"
    ds_hash = ds_hash + cache_version_suffix

    # Skip rebuild if cache up-to-date
    if not force:
        meta_file = cache_root / "train.h5"
        if meta_file.exists():
            try:
                with h5py.File(meta_file, "r") as f:
                    if f.attrs.get("dataset_hash") == ds_hash:
                        sizes = {}
                        for split in SPLIT_NAMES:
                            p = cache_root / f"{split}.h5"
                            if p.exists():
                                with h5py.File(p, "r") as f2:
                                    sizes[split] = int(f2["xy"].shape[0])
                            else:
                                sizes[split] = 0
                        print(f"Cache up-to-date (hash={ds_hash}). Sizes: {sizes}", flush=True)
                        return sizes
            except (OSError, KeyError):
                pass

    # Read all manifests
    holdout = set(config.holdout_cr_indices)
    pool_rows: list[ManifestRow] = []
    holdout_rows: list[ManifestRow] = []
    dropped_by_cr_range = 0

    for tdir in target_dirs:
        target_idx = int(tdir.name.replace("target_", ""))
        rows = _read_manifest(tdir / "manifest.csv", target_idx)
        rows = _filter_rows(rows, config.filter_max_frac_err, config.filter_max_gs_err_pct)

        # Optional CR range restriction
        if config.cr_min is not None:
            before = len(rows)
            rows = [r for r in rows if r.cooling_rate >= config.cr_min]
            dropped_by_cr_range += before - len(rows)
        if config.cr_max is not None:
            before = len(rows)
            rows = [r for r in rows if r.cooling_rate <= config.cr_max]
            dropped_by_cr_range += before - len(rows)
        if not rows:
            continue

        rows = _best_of(rows, config.best_of_per_cr)
        if target_idx in holdout:
            holdout_rows.extend(rows)
        else:
            pool_rows.extend(rows)

    cr_range_msg = ""
    if config.cr_min is not None or config.cr_max is not None:
        cr_range_msg = (
            f" (CR in "
            f"[{config.cr_min if config.cr_min is not None else '-inf'}, "
            f"{config.cr_max if config.cr_max is not None else 'inf'}]; "
            f"dropped {dropped_by_cr_range} out-of-range rows)"
        )
    print(f"After filter+best-of: pool={len(pool_rows)}, holdout={len(holdout_rows)}{cr_range_msg}", flush=True)

    splits = _stratified_split(pool_rows, config.split_ratios, config.split_seed)
    splits["holdout"] = holdout_rows

    sizes: dict[str, int] = {}
    train_pack = _pack_split(splits["train"], config, dataset_root)
    if train_pack is None:
        raise RuntimeError("Train split empty after parsing")
    norm_stats = _compute_norm_stats(train_pack)

    for split in SPLIT_NAMES:
        rows_split = splits[split]
        if split == "train":
            pack = train_pack
        else:
            pack = _pack_split(rows_split, config, dataset_root)
        if pack is None:
            sizes[split] = 0
            continue

        # cr_norm using train stats
        log_cr = np.log(pack["cr"].astype(np.float64))
        cr_norm = ((log_cr - norm_stats["cr_log_mean"]) / norm_stats["cr_log_std"]).astype(np.float32)

        out_path = cache_root / f"{split}.h5"
        with h5py.File(out_path, "w") as f:
            for k, v in pack.items():
                if k == "_skipped":
                    f.attrs["skipped_parse_failures"] = int(v[0])
                    continue
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            f.create_dataset("cr_norm", data=cr_norm, compression="gzip", compression_opts=4)
            f.attrs["dataset_hash"] = ds_hash
            f.attrs["split_seed"] = config.split_seed
            f.attrs["num_points"] = config.num_points
            f.attrs["num_phases"] = config.num_phases
            f.attrs["max_nucleation_ratio"] = config.max_nucleation_ratio
            f.attrs["curve_max_len"] = config.curve_max_len
            f.attrs["max_grain_size"] = config.max_grain_size
            for k, v in norm_stats.items():
                f.attrs[k] = v

        sizes[split] = pack["xy"].shape[0]
        print(
            f"  {split:8s}: {sizes[split]:5d} samples (skipped {int(pack['_skipped'][0])} parse failures)",
            flush=True,
        )

    return sizes

# Augmentation

def _augment_xy(
    xy: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Random rot90 + flip + periodic translate. xy shape (P, 2).

    Phase, nuc and mask are invariant. Slot order is preserved - Hungarian
    matching in compute_loss is permutation-invariant by construction, so
    we don't need to re-sort anything after augmentation.
    """
    k = int(rng.integers(0, 4))
    out = xy.copy()
    if k == 1:
        out = np.stack([1.0 - xy[:, 1], xy[:, 0]], axis=1)
    elif k == 2:
        out = np.stack([1.0 - xy[:, 0], 1.0 - xy[:, 1]], axis=1)
    elif k == 3:
        out = np.stack([xy[:, 1], 1.0 - xy[:, 0]], axis=1)

    if rng.random() < 0.5:
        out[:, 0] = 1.0 - out[:, 0]
    if rng.random() < 0.5:
        out[:, 1] = 1.0 - out[:, 1]

    dx = float(rng.random())
    dy = float(rng.random())
    out[:, 0] = (out[:, 0] + dx) % 1.0
    out[:, 1] = (out[:, 1] + dy) % 1.0

    out[~mask] = 0.0
    return out

# Dataset class

class SSRVEDataset(Dataset):
    """HDF5-backed dataset of SSRVE nucleation parameters.

    Reads cache/{split}.h5 produced by build_cache(). Augmentation is applied
    in __getitem__ when split == "train" and config.enable_augmentation is True.
    """

    def __init__(self, config: CVAEConfig, split: str = "train"):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")
        self.config = config
        self.split = split
        cache_path = Path(config.cache_dir) / f"{split}.h5"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}. Run dataset.build_cache(config) first."
            )

        # Load all into RAM (cache is small enough; HDF5 random access is slow on Windows).
        with h5py.File(cache_path, "r") as f:
            self.xy = f["xy"][:]
            self.phase = f["phase"][:]
            self.nuc = f["nuc"][:]
            self.mask = f["mask"][:]
            self.cooling_curve = f["cooling_curve"][:]
            self.curve_mask = f["curve_mask"][:]
            self.target_fractions = f["target_fractions"][:]
            self.target_grain_size = f["target_grain_size"][:]
            self.cr = f["cr"][:]
            self.cr_norm = f["cr_norm"][:]
            self.n_active = f["n_active"][:]
            self.target_idx = f["target_idx"][:]
            self.norm_stats = {
                "cr_log_mean": float(f.attrs["cr_log_mean"]),
                "cr_log_std": float(f.attrs["cr_log_std"]),
                "gs_mean": float(f.attrs["gs_mean"]),
                "gs_std": float(f.attrs["gs_std"]),
            }
            self.dataset_hash = str(f.attrs["dataset_hash"])

        # Per-target sample counts (for weighted sampler)
        unique, counts = np.unique(self.target_idx, return_counts=True)
        self._counts_by_target: dict[int, int] = {int(u): int(c) for u, c in zip(unique, counts)}

        # Per-phase class weights for imbalanced cross-entropy.
        # Computed from active slots only; formula: 1 / sqrt(freq), normalized
        # so the mean weight equals 1 (keeps overall phase loss magnitude
        # comparable to the unweighted case).
        active_phases = self.phase[self.mask.astype(bool)]  # (total_active,) int8, values 1..num_phases
        counts_per_phase = np.bincount(
            active_phases.astype(np.int64), minlength=config.num_phases + 1
        )[1 : config.num_phases + 1]
        freqs = counts_per_phase / max(int(counts_per_phase.sum()), 1)
        raw_weights = 1.0 / np.sqrt(np.maximum(freqs, 1e-6))
        if raw_weights.mean() > 0:
            raw_weights = raw_weights / raw_weights.mean()
        self.class_weights = torch.tensor(raw_weights, dtype=torch.float32)
        self.class_counts = counts_per_phase.astype(np.int64)

        self._rng = np.random.default_rng(config.split_seed + hash(split) % 2**31)

        # Morphological targets (optional - loaded if file exists)
        self._morpho: dict[int, dict[str, np.ndarray]] | None = None
        try:
            self._morpho = load_morphology_targets(config.cache_dir)
        except FileNotFoundError:
            pass  # morphology targets not built yet - skip silently

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        xy = self.xy[idx]  # (P, 2)
        mask = self.mask[idx]  # (P,)
        phase = self.phase[idx]  # (P,) int8
        nuc = self.nuc[idx]  # (P,)

        if self.split == "train" and self.config.enable_augmentation:
            xy = _augment_xy(xy, mask, self._rng)

        tidx = int(self.target_idx[idx])
        item = {
            "xy": torch.from_numpy(xy.astype(np.float32)),
            "phase_class": torch.from_numpy(phase.astype(np.int64)),  # 1..num_phases
            "nuc": torch.from_numpy(nuc.astype(np.float32)),
            "active_mask": torch.from_numpy(mask.astype(np.bool_)),
            "cooling_curve": torch.from_numpy(self.cooling_curve[idx].astype(np.float32)),
            "curve_mask": torch.from_numpy(self.curve_mask[idx].astype(np.bool_)),
            "target_fractions": torch.from_numpy(self.target_fractions[idx].astype(np.float32)),
            "target_grain_size": torch.tensor(
                [self.target_grain_size[idx]], dtype=torch.float32
            ),
            "cr_norm": torch.tensor([self.cr_norm[idx]], dtype=torch.float32),
            "n_active": torch.tensor([self.n_active[idx]], dtype=torch.float32),
            "target_idx": torch.tensor(tidx, dtype=torch.long),
        }

        # Morphological targets (per-CR lookup by target_idx)
        if self._morpho is not None and tidx in self._morpho:
            m = self._morpho[tidx]
            item["morpho_s2"] = torch.from_numpy(m["s2"].astype(np.float32))
            item["morpho_br"] = torch.from_numpy(m["boundary_ratio"].astype(np.float32))
            item["morpho_pair_same"] = torch.from_numpy(m["pair_same"].astype(np.float32))

        return item

    def get_sampler(self) -> WeightedRandomSampler:
        """Weighted sampler with weight = 1 / n_per_cr ** sampler_weight_pow."""
        pow_ = self.config.sampler_weight_pow
        weights = np.array(
            [1.0 / (self._counts_by_target[int(t)] ** pow_) for t in self.target_idx],
            dtype=np.float64,
        )
        return WeightedRandomSampler(
            weights.tolist(), num_samples=len(self), replacement=True
        )


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Default collate: every field has fixed shape (cache is pre-padded)."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
