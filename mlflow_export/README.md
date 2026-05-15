# MLflow run export

Static JSON snapshot of the `ssrve-option-b-v2` MLflow experiment. Allows inspect hyperparameters, metrics, and run lineage of every reported model without needing to spin up the MLflow server.

Each run lives in `runs/<run_id>.json` and contains params, final metric values, per-step metric history, run timestamps, and artefact paths (artefacts themselves are on Zenodo). `index.json` maps run IDs to filenames and lists headline metrics.

## Regenerate

MLflow server must be up.

```bash
docker compose up -d

uv run python mlflow_export/export.py \
    --experiment ssrve-option-b-v2 \
    --out mlflow_export/runs
```

Override the tracking URI with `--tracking-uri http://...` if not using
the default `docker-compose.yml` stack.

The current snapshot was frozen on the day the camera-ready manuscript
was prepared. To pick up later runs, re-run the command.
