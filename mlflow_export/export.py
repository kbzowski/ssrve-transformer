"""Dump an MLflow experiment as static JSON.

One file per run plus index.json. Artefacts are not copied (they go on Zenodo);
only their paths are recorded.

    uv run python mlflow_export/export.py --experiment ssrve-option-b-v2 --out mlflow_export/runs
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


def _ts_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).isoformat()


def _serialize_run(client: MlflowClient, run: Run) -> dict:
    info = run.info
    data = run.data
    history: dict[str, list[dict]] = {}
    for metric_key in data.metrics:
        steps = client.get_metric_history(info.run_id, metric_key)
        history[metric_key] = [
            {"step": m.step, "value": m.value, "timestamp_ms": m.timestamp}
            for m in steps
        ]
    artefacts = []
    try:
        for a in client.list_artifacts(info.run_id):
            artefacts.append(
                {"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size}
            )
    except Exception:
        pass
    return {
        "run_id": info.run_id,
        "experiment_id": info.experiment_id,
        "run_name": data.tags.get("mlflow.runName", ""),
        "status": info.status,
        "start_time": _ts_to_iso(info.start_time),
        "end_time": _ts_to_iso(info.end_time),
        "duration_seconds": (
            (info.end_time - info.start_time) / 1000.0
            if info.end_time and info.start_time
            else None
        ),
        "user_id": info.user_id,
        "params": dict(data.params),
        "metrics_final": dict(data.metrics),
        "metrics_history": history,
        "tags": dict(data.tags),
        "artefacts": artefacts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment {args.experiment!r} not found at {args.tracking_uri}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id], max_results=10_000
    )
    print(f"Exporting {len(runs)} runs from experiment {args.experiment!r}")

    index = {
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
        "tracking_uri": args.tracking_uri,
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": [],
    }
    for run in runs:
        record = _serialize_run(client, run)
        out_file = out / f"{record['run_id']}.json"
        out_file.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        index["runs"].append(
            {
                "run_id": record["run_id"],
                "run_name": record["run_name"],
                "status": record["status"],
                "start_time": record["start_time"],
                "duration_seconds": record["duration_seconds"],
                "headline_metrics": {
                    k: v
                    for k, v in record["metrics_final"].items()
                    if k in {"val_loss", "best_val_loss", "train_loss", "epoch_time_sec"}
                },
                "file": out_file.name,
            }
        )

    (out.parent / "index.json").write_text(
        json.dumps(index, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {len(runs)} runs to {out} + index.json")


if __name__ == "__main__":
    main()
