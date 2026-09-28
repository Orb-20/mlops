"""MLflow helpers: tracking URI setup, lineage tags, and an availability check.

A failure to reach MLflow raises `TrackingUnavailable`; callers must not pretend a run was
logged or a model registered when it was not (spec §47).
"""

from __future__ import annotations

import platform
import sys

import mlflow

from src.config.settings import get_env, project_path
from src.data.versioning import dataset_version, git_commit


class TrackingUnavailable(RuntimeError):
    pass


def setup_mlflow(experiment: str) -> str:
    uri = get_env().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    try:
        exp = mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001 - mlflow raises many types on connection errors
        raise TrackingUnavailable(f"MLflow at {uri} unavailable: {exc}") from exc
    return exp.experiment_id


def lineage_tags() -> dict[str, str]:
    import torch
    import transformers

    return {
        "dataset_version": dataset_version(project_path("data", "processed")),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": sys.platform,
    }
