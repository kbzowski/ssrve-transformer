"""Generate one or more SSRVE phase grids from a trained DDPM.

CLI:
    uv run python sample.py --ckpt checkpoints/best.pt --cr 10.0 --n 4 --out out.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler

from model import ConditionalDDPM, CoolingCurveEncoder, ModelConfig


@dataclass
class CurveConfig:
    """Subset of CurveConfig needed at inference. Defaults match the transformer."""
    t_start: float = 830.0
    t_ms: float = 400.0
    max_time: float = 1300.0
    curve_num_samples: int = 16
    real_size: float = 50.0  # μm

    def cooling_curve_from_cr(self, cr: float) -> list[tuple[float, float]]:
        t_end = (self.t_start - self.t_ms) / cr
        return [
            (i / (self.curve_num_samples - 1) * t_end,
             self.t_start - cr * (i / (self.curve_num_samples - 1) * t_end))
            for i in range(self.curve_num_samples)
        ]

    def normalize_curve(self, curve):
        return [(t / self.max_time, T / self.t_start) for t, T in curve]

# Loader

def load_model(ckpt_path: Path, device: str) -> tuple[ConditionalDDPM, dict, CurveConfig]:
    ckpt = torch.load(ckpt_path, map_location=device)
    ycfg = ckpt["config"]
    mc_yaml = ycfg["model"]
    cond = CoolingCurveEncoder(
        input_dim=ycfg["data"]["curve_input_dim"],
        d_model=mc_yaml["cond_encoder"]["d_model"],
        n_heads=mc_yaml["cond_encoder"]["n_heads"],
        n_layers=mc_yaml["cond_encoder"]["n_layers"],
        dim_feedforward=mc_yaml["cond_encoder"]["dim_feedforward"],
        max_len=ycfg["data"]["curve_max_len"],
    )
    mc = ModelConfig(
        in_channels=mc_yaml["in_channels"],
        out_channels=mc_yaml["out_channels"],
        sample_size=ycfg["data"]["grid_size"],
        block_out_channels=tuple(mc_yaml["block_out_channels"]),
        layers_per_block=mc_yaml["layers_per_block"],
        cross_attention_dim=mc_yaml["cross_attention_dim"],
        attention_head_dim=mc_yaml["attention_head_dim"],
    )
    model = ConditionalDDPM(mc, cond).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ycfg, CurveConfig()

# Sampling

@torch.no_grad()
def sample_for_cr(
    model: ConditionalDDPM,
    ycfg: dict,
    cvae_cfg: CurveConfig,
    cr: float,
    n: int,
    device: str,
    num_inference_steps: int = 250,
) -> np.ndarray:
    """Return (n, G, G) integer phase grid in {1..num_phases}."""
    G = ycfg["data"]["grid_size"]
    C = ycfg["data"]["num_phases"]
    L = ycfg["data"]["curve_max_len"]

    curve = cvae_cfg.normalize_curve(cvae_cfg.cooling_curve_from_cr(cr))
    pad = np.zeros((L, 2), dtype=np.float32)
    mask = np.zeros((L,), dtype=bool)
    for i, (t, T) in enumerate(curve[:L]):
        pad[i] = (t, T)
        mask[i] = True
    cooling_curve = torch.from_numpy(pad).unsqueeze(0).expand(n, -1, -1).to(device)
    curve_mask = torch.from_numpy(mask).unsqueeze(0).expand(n, -1).to(device)

    sched = DDPMScheduler(
        num_train_timesteps=ycfg["diffusion"]["num_train_timesteps"],
        beta_schedule=ycfg["diffusion"]["beta_schedule"],
        prediction_type=ycfg["diffusion"]["prediction_type"],
    )
    sched.set_timesteps(num_inference_steps)

    x = torch.randn(n, C, G, G, device=device)
    for t in sched.timesteps:
        eps = model(x, t.expand(n).to(device), cooling_curve, curve_mask)
        x = sched.step(eps, t, x).prev_sample

    # x in roughly [-1, 1]; recover one-hot by argmax over channels.
    phase = x.argmax(dim=1).cpu().numpy() + 1  # back to 1-indexed phase ids
    return phase.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--cr", required=True, type=float)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", type=Path, default=Path("sample.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ycfg, cvae_cfg = load_model(args.ckpt, device)
    grids = sample_for_cr(model, ycfg, cvae_cfg, args.cr, args.n, device, args.steps)

    fig, axes = plt.subplots(1, args.n, figsize=(3 * args.n, 3))
    if args.n == 1:
        axes = [axes]
    for ax, g in zip(axes, grids):
        ax.imshow(g, cmap="tab10", vmin=1, vmax=ycfg["data"]["num_phases"])
        ax.set_title(f"CR={args.cr}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
