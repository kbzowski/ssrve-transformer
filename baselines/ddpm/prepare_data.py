"""One-shot pre-rasterization of SSRVE.pkl files into a self-contained HDF5 cache.

Run ONCE using the main `multiphasessrve` env (which has numba):

    cd D:/PRACA/MultiPhaseSSRVE
    uv run python baselines/ddpm/prepare_data.py

After this the DDPM training loop has no dependency on the main package
beyond the produced ``cache/ddpm_grids.h5`` file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from multiphasessrve.ml.config import CVAEConfig  # noqa: E402

from data import build_split_index, downsample_nearest, rasterize_phase_grid  # noqa: E402


GRID_SIZE = 64
NUM_PHASES = 4


def main():
    cvae = CVAEConfig(
        dataset_dir=str((Path(__file__).parent / "../../dataset").resolve()),
        cache_dir=str((Path(__file__).parent / "../../cache").resolve()),
    )
    splits = build_split_index(cvae, Path(cvae.dataset_dir))

    out_path = Path(cvae.cache_dir) / "ddpm_grids.h5"
    with h5py.File(out_path, "w") as f:
        f.attrs["grid_size"] = GRID_SIZE
        f.attrs["num_phases"] = NUM_PHASES
        f.attrs["curve_max_len"] = cvae.curve_max_len
        for split, samples in splits.items():
            n = len(samples)
            print(f"\n[{split}] rasterizing {n} grids -> {out_path}")
            grp = f.create_group(split)
            grids = grp.create_dataset(
                "grids", (n, GRID_SIZE, GRID_SIZE), dtype=np.uint8,
                compression="gzip", compression_opts=4,
            )
            curves = grp.create_dataset(
                "cooling_curve", (n, cvae.curve_max_len, 2), dtype=np.float32,
                compression="gzip", compression_opts=4,
            )
            curve_masks = grp.create_dataset(
                "curve_mask", (n, cvae.curve_max_len), dtype=np.bool_,
                compression="gzip", compression_opts=4,
            )
            crs = grp.create_dataset("cr", (n,), dtype=np.float32)
            target_idx = grp.create_dataset("target_idx", (n,), dtype=np.int16)
            target_fracs = grp.create_dataset("target_fractions", (n, NUM_PHASES), dtype=np.float32)
            target_gs = grp.create_dataset("target_grain_size", (n,), dtype=np.float32)

            for i, s in enumerate(tqdm(samples, desc=split)):
                full = rasterize_phase_grid(s.pkl_path, NUM_PHASES)
                grids[i] = downsample_nearest(full, GRID_SIZE)
                curves[i] = s.cooling_curve
                curve_masks[i] = s.curve_mask
                crs[i] = s.cooling_rate
                target_idx[i] = s.target_idx
                target_fracs[i] = s.target_fractions
                target_gs[i] = s.target_grain_size
    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    main()
