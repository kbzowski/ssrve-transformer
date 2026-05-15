# SSRVE Transformer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20202883.svg)](https://doi.org/10.5281/zenodo.20202883)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the paper *Generation of statistically similar representative volume
elements for multi-phase steels using an encoder-decoder transformer with
differentiable soft-Voronoi tessellation* (Bzowski & Rauch, Computational
Materials Science, 2026, DOI: TBA).

A transformer reads a cooling curve and emits a small set of nucleation
parameters; a differentiable Apollonius soft-Voronoi head turns that set
into a periodic phase map during training, and a Johnson-Mehl cellular
automaton turns it into the final 500x500 SSRVE at inference. The whole
inference path runs in roughly 100 ms per microstructure.

## Install

```bash
uv sync
```

(Project uses [uv](https://docs.astral.sh/uv/))

## Generate one SSRVE

```bash
uv run python -m multiphasessrve.ml.generate --cr 17 --out ssrve_cr17.png
```

## Train from scratch

```bash
docker compose up -d                              # MLflow stack
uv run python -m multiphasessrve.ml.dataset      # build HDF5 cache (~10 min)
uv run python -m multiphasessrve.ml.train         # train H2b_A1
```

Runs are tracked in MLflow under experiment `ssrve-option-b-v2`. A frozen
snapshot of all 21 ablation runs lives in `mlflow_export/runs/`.

## Heavy artefacts

The raw SSRVE dataset, the HDF5 training cache, and the trained checkpoints
are hosted separately on Zenodo
([10.5281/zenodo.20202883](https://doi.org/10.5281/zenodo.20202883))
because they are too large to ship with git.

## Layout

```
src/multiphasessrve/ml/       ML pipeline (transformer + soft-Voronoi decoder)
src/multiphasessrve/analysis/ phase-fraction and grain-size analysers used at evaluation
src/multiphasessrve/types/    result types (scalar / array / dict / histogram)
src/multiphasessrve/ssrve.py  Ssrve container; Johnson-Mehl cellular-automaton growth
src/multiphasessrve/ssrve_algs.py  Numba-jit grid kernels used by Ssrve
baselines/ddpm/               pixel-space DDPM baseline
mlflow_export/                JSON snapshot of training runs
RULES.md                      grain conventions, periodic BC
```

## Cite

```bibtex
@article{bzowski2026ssrvetransformer,
  author  = {Bzowski, Krzysztof and Rauch, {\L}ukasz},
  title   = {Generation of statistically similar representative volume elements
             for multi-phase steels using an encoder-decoder transformer with
             differentiable soft-Voronoi tessellation},
  journal = {Computational Materials Science},
  year    = {2026},
  doi     = {TBA}
}
```

## License

MIT.

## Funding

NCN grant 2021/43/B/ST8/01710 (Polish National Science Centre) and AGH
Faculty of Metals Engineering and Industrial Computer Science Grant for
Young Scientists (16.16.110.663, task 6).
