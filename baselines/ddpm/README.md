# DDPM baseline

Class-conditional pixel-space DDPM trained on the same dataset as the
transformer, used as the deep-generative baseline in the paper.

## Architecture

- `diffusers.UNet2DConditionModel`, 4 in/out channels (one-hot phase grids).
- Cooling-curve TransformerEncoder mirrors the main model's
  `CoolingCurveEncoder` and feeds cross-attention.
- 1000-step DDPM with `squaredcos_cap_v2` betas, epsilon prediction.
- Trained at 64x64; SSRVE pickles are nearest-neighbour resized from 500x500.

## Run

```bash
cd baselines/ddpm
uv sync

uv run python train.py --config config.yaml

uv run python sample.py --ckpt checkpoints/best.pt --cr 10.0 --out sample.png

uv run python evaluate.py --ckpt checkpoints/best.pt --out ddpm_eval.md
```

Same seed, same splits, and same HDF5 cache as the transformer. Logged to
MLflow as experiment `ssrve-baseline-ddpm`.

## Notes

The DDPM is kept in its own subdirectory with its own `pyproject.toml`
because `diffusers` and `accelerate` are heavy dependencies the main
package does not need.

See the paper for head-to-head results and the fairness caveats around
grid resolution and metric definitions.
