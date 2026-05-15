"""Evaluation: phase-fraction error, ferrite grain-size error, sample-time per CR.

Mirrors the metrics used in the transformer evaluation so the two reports
are directly comparable. Writes a Markdown report to --out.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.measure import label

from sample import load_model, sample_for_cr  # noqa: E402


def _load_targets_from_cache(cache_path: Path, split: str) -> list[dict]:
    """Read per-sample (cr, target_fractions, target_grain_size) from prepared cache."""
    with h5py.File(cache_path, "r") as f:
        g = f[split]
        crs = g["cr"][:]
        fracs = g["target_fractions"][:]
        gss = g["target_grain_size"][:]
    out = []
    for i in range(len(crs)):
        out.append({
            "cooling_rate": float(crs[i]),
            "target_fractions": fracs[i],
            "target_grain_size": float(gss[i]),
        })
    return out

# Per-grid metrics

def phase_fractions(grid: np.ndarray, num_phases: int) -> np.ndarray:
    counts = np.bincount(grid.flatten(), minlength=num_phases + 1)
    return (counts[1 : num_phases + 1] / grid.size).astype(np.float64)


def ferrite_grain_size_um(grid: np.ndarray, real_size_um: float) -> float:
    """Mean ferrite (phase_id=1) grain area -> equivalent diameter in micrometres.

    Connected components on the ferrite mask. Returns NaN if no ferrite present.
    """
    mask = grid == 1
    if not mask.any():
        return float("nan")
    labels = label(mask, connectivity=1)
    if labels.max() == 0:
        return float("nan")
    sizes_px = np.bincount(labels.flatten())[1:]
    px_per_um = grid.shape[0] / real_size_um
    mean_area_um2 = sizes_px.mean() / (px_per_um ** 2)
    return float(2.0 * np.sqrt(mean_area_um2 / np.pi))

# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_per_cr", type=int, default=8,
                    help="Generated samples per CR (averaged for metrics)")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--cr_min", type=float, default=None, help="Filter CRs >= cr_min")
    ap.add_argument("--cr_max", type=float, default=None, help="Filter CRs <= cr_max")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ycfg, cvae_cfg = load_model(args.ckpt, device)
    real_size_um = cvae_cfg.real_size

    repo_root = Path(__file__).resolve().parents[2]
    cache_path = repo_root / "cache" / "ddpm_grids.h5"
    samples = _load_targets_from_cache(cache_path, args.split)

    # Group target stats per CR
    targets_by_cr: dict[float, dict] = {}
    for s in samples:
        cr = s["cooling_rate"]
        if cr not in targets_by_cr:
            targets_by_cr[cr] = {
                "fractions": s["target_fractions"].astype(np.float64),
                "grain_size": s["target_grain_size"],
                "n": 1,
            }
        else:
            targets_by_cr[cr]["n"] += 1

    if args.cr_min is not None or args.cr_max is not None:
        keep = {
            cr for cr in targets_by_cr
            if (args.cr_min is None or cr >= args.cr_min)
            and (args.cr_max is None or cr <= args.cr_max)
        }
        targets_by_cr = {k: v for k, v in targets_by_cr.items() if k in keep}
        print(f"Filtered to CR range [{args.cr_min}, {args.cr_max}]: {len(targets_by_cr)} CRs", flush=True)

    rows = []
    total_sample_time = 0.0
    for cr in sorted(targets_by_cr.keys()):
        target = targets_by_cr[cr]
        t0 = time.time()
        grids = sample_for_cr(model, ycfg, cvae_cfg, cr, args.n_per_cr, device, args.steps)
        sample_time = (time.time() - t0) / args.n_per_cr
        total_sample_time += sample_time * args.n_per_cr

        # Per-sample metrics, averaged
        fracs = np.stack([phase_fractions(g, ycfg["data"]["num_phases"]) for g in grids])
        gs = [ferrite_grain_size_um(g, real_size_um) for g in grids]

        mean_frac = fracs.mean(axis=0)
        max_frac_err = float(np.max(np.abs(mean_frac - target["fractions"])))
        gs_valid = [v for v in gs if not np.isnan(v)]
        mean_gs = float(np.mean(gs_valid)) if gs_valid else float("nan")
        gs_err_pct = (
            (mean_gs - target["grain_size"]) / target["grain_size"] * 100.0
            if target["grain_size"] > 0 and not np.isnan(mean_gs)
            else float("nan")
        )

        rows.append({
            "cr": cr,
            "n_dataset_samples": target["n"],
            "n_generated": args.n_per_cr,
            "target_F": target["fractions"][0],
            "target_B": target["fractions"][1],
            "target_M": target["fractions"][2],
            "target_P": target["fractions"][3],
            "pred_F": float(mean_frac[0]),
            "pred_B": float(mean_frac[1]),
            "pred_M": float(mean_frac[2]),
            "pred_P": float(mean_frac[3]),
            "max_frac_err": max_frac_err,
            "target_gs_um": target["grain_size"],
            "pred_gs_um": mean_gs,
            "gs_err_pct": gs_err_pct,
            "sample_time_sec": sample_time,
        })
        print(
            f"CR={cr:7.2f}  max_frac_err={max_frac_err:.4f}  "
            f"gs_err={gs_err_pct:+.1f}%  sample_time={sample_time:.2f}s",
            flush=True,
        )

    # Aggregate
    mean_max_frac_err = statistics.mean(r["max_frac_err"] for r in rows)
    valid_gs = [abs(r["gs_err_pct"]) for r in rows if not np.isnan(r["gs_err_pct"])]
    mean_abs_gs_err_pct = statistics.mean(valid_gs) if valid_gs else float("nan")
    mean_sample_time = statistics.mean(r["sample_time_sec"] for r in rows)

    # Markdown report ------------------------------------------------------
    lines = [
        "# DDPM Baseline Evaluation",
        "",
        f"- **Checkpoint:** `{args.ckpt}`",
        f"- **Split:** `{args.split}` ({len(samples)} dataset samples across {len(rows)} CRs)",
        f"- **Generated samples per CR:** {args.n_per_cr}",
        f"- **DDPM inference steps:** {args.steps}",
        f"- **Grid resolution:** {ycfg['data']['grid_size']}² (real size {real_size_um}μm)",
        "",
        "## Aggregate metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean max phase-fraction error | **{mean_max_frac_err:.4f}** |",
        f"| Mean abs ferrite grain-size error (%) | **{mean_abs_gs_err_pct:.2f}** |",
        f"| Mean sample time per microstructure (s) | **{mean_sample_time:.2f}** |",
        "",
        "## Per-CR breakdown",
        "",
        "| CR | target F/B/M/P | pred F/B/M/P | max_frac_err | target_gs (μm) | pred_gs (μm) | gs_err (%) | sample_time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cr']:.2f} "
            f"| {r['target_F']:.2f}/{r['target_B']:.2f}/{r['target_M']:.2f}/{r['target_P']:.2f} "
            f"| {r['pred_F']:.2f}/{r['pred_B']:.2f}/{r['pred_M']:.2f}/{r['pred_P']:.2f} "
            f"| {r['max_frac_err']:.4f} "
            f"| {r['target_gs_um']:.2f} "
            f"| {r['pred_gs_um']:.2f} "
            f"| {r['gs_err_pct']:+.1f} "
            f"| {r['sample_time_sec']:.2f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Phase fractions averaged over generated samples per CR.",
        "- Ferrite grain size = equivalent-circle diameter from connected-component areas (skimage.measure.label, 4-connectivity).",
        "- Sample time excludes data loading / warm-up; pure DDPM denoising loop.",
        "- For the head-to-head comparison vs the transformer + pymoo, see `baseline_comparison_pymoo_vs_ml_2026-04-25.md`.",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
