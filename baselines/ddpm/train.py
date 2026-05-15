"""DDPM training loop with MLflow logging.

Single source of truth for metrics is MLflow (per project rule
``feedback_mlflow_single_source.md``). No CSV/JSON sidecars are written.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data import DDPMSsrveDataset, make_dataloaders  # noqa: E402
from model import ConditionalDDPM, CoolingCurveEncoder, ModelConfig, num_parameters  # noqa: E402

# EMA

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(pm.detach(), alpha=1.0 - self.decay)

# Helpers

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

# One pass (epoch fragment)

def train_one_epoch(
    model: ConditionalDDPM,
    scheduler: DDPMScheduler,
    loader,
    optimizer,
    scaler,
    ema: EMA | None,
    device: str,
    epoch: int,
    use_amp: bool,
) -> float:
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"epoch {epoch:03d}", leave=False)
    for batch in pbar:
        batch = to_device(batch, device)
        x0 = batch["x"]
        bsz = x0.size(0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        noise = torch.randn_like(x0)
        x_noisy = scheduler.add_noise(x0, noise, t)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp, dtype=torch.float16):
            pred = model(x_noisy, t, batch["cooling_curve"], batch["curve_mask"])
            loss = F.mse_loss(pred, noise)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses))


@torch.no_grad()
def validate(
    model: ConditionalDDPM,
    scheduler: DDPMScheduler,
    loader,
    device: str,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        x0 = batch["x"]
        bsz = x0.size(0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        noise = torch.randn_like(x0)
        x_noisy = scheduler.add_noise(x0, noise, t)
        pred = model(x_noisy, t, batch["cooling_curve"], batch["curve_mask"])
        losses.append(F.mse_loss(pred, noise).item())
    return float(np.mean(losses)) if losses else float("nan")

# Entrypoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        ycfg = yaml.safe_load(f)

    seed_all(ycfg["training"]["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (ycfg["training"]["mixed_precision"] == "fp16") and (device == "cuda")

    # Data ------------------------------------------------------------------
    cache_path = (Path(__file__).parent / ycfg["data"]["cache_dir"] / "ddpm_grids.h5").resolve()
    if not cache_path.exists():
        raise FileNotFoundError(
            f"DDPM grid cache not found: {cache_path}\n"
            f"Run `uv run python prepare_data.py` from the repo root first."
        )
    loaders = make_dataloaders(
        cache_path=cache_path,
        num_phases=ycfg["data"]["num_phases"],
        batch_size=ycfg["training"]["batch_size"],
        num_workers=ycfg["training"]["num_workers"],
    )
    splits = {k: v.dataset for k, v in loaders.items()}
    print(
        "Splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()),
        flush=True,
    )

    # Model -----------------------------------------------------------------
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
    print(f"Params: {num_parameters(model):,}", flush=True)

    scheduler = DDPMScheduler(
        num_train_timesteps=ycfg["diffusion"]["num_train_timesteps"],
        beta_schedule=ycfg["diffusion"]["beta_schedule"],
        prediction_type=ycfg["diffusion"]["prediction_type"],
    )

    # Optim -----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ycfg["training"]["lr"],
        weight_decay=ycfg["training"]["weight_decay"],
    )
    scaler = GradScaler(enabled=use_amp)
    ema = EMA(model, ycfg["training"]["ema_decay"])

    # MLflow ----------------------------------------------------------------
    mlflow.set_tracking_uri(ycfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(ycfg["mlflow"]["experiment_name"])
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val = float("inf")

    with mlflow.start_run(run_name=ycfg["mlflow"]["run_name"]):
        mlflow.log_params({
            "n_params": num_parameters(model),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "n_holdout": len(splits["holdout"]),
            "grid_size": ycfg["data"]["grid_size"],
            "num_phases": ycfg["data"]["num_phases"],
            "batch_size": ycfg["training"]["batch_size"],
            "lr": ycfg["training"]["lr"],
            "epochs": ycfg["training"]["epochs"],
            "ema_decay": ycfg["training"]["ema_decay"],
            "diffusion_steps": ycfg["diffusion"]["num_train_timesteps"],
            "beta_schedule": ycfg["diffusion"]["beta_schedule"],
            "prediction_type": ycfg["diffusion"]["prediction_type"],
            "block_out_channels": str(mc_yaml["block_out_channels"]),
            "cross_attention_dim": mc_yaml["cross_attention_dim"],
            "amp": use_amp,
        })
        mlflow.log_artifact(str(cfg_path))

        for epoch in range(1, ycfg["training"]["epochs"] + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, scheduler, loaders["train"], optimizer, scaler, ema,
                device, epoch, use_amp,
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_time_sec", time.time() - t0, step=epoch)

            if epoch % ycfg["training"]["val_every_epochs"] == 0 or epoch == 1:
                val_loss = validate(ema.shadow, scheduler, loaders["val"], device)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": ema.shadow.state_dict(),
                            "config": ycfg,
                            "val_loss": val_loss,
                        },
                        ckpt_dir / "best.pt",
                    )
                    mlflow.log_metric("best_val_loss", best_val, step=epoch)
                print(
                    f"epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"best={best_val:.4f}  ({time.time()-t0:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"epoch {epoch:03d}  train={train_loss:.4f}  ({time.time()-t0:.1f}s)",
                    flush=True,
                )

        # Always keep last checkpoint as well
        torch.save(
            {
                "epoch": ycfg["training"]["epochs"],
                "model_state": ema.shadow.state_dict(),
                "config": ycfg,
                "val_loss": float("nan"),
            },
            ckpt_dir / "last.pt",
        )


if __name__ == "__main__":
    main()
