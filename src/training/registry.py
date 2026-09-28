"""MLflow Model Registry: the single source of truth for model lineage and lifecycle.

Registered model: `sentiment-toxicity-model`. Lifecycle is expressed with
  * aliases  : `candidate`, `staging`, `production`, `rejected` (latest version in that state)
  * tag      : `lifecycle` on every version (candidate | staging | production | rejected | archived)
Model versions are never identified by filenames.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

from src.config.settings import get_config, get_env, project_path

log = logging.getLogger(__name__)

ALIASES = ("candidate", "staging", "production", "rejected")


class RegistryError(RuntimeError):
    pass


class SentimentToxicityPyfunc(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so the artifact is a proper MLflow Model (loadable via mlflow.pyfunc)."""

    def load_context(self, context):
        from src.nlp.inference import Predictor

        self.predictor = Predictor(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input, params=None):
        import pandas as pd

        texts = list(model_input["text"] if hasattr(model_input, "columns") else model_input)
        out = self.predictor.run(texts)
        return pd.DataFrame([p.model_dump(exclude={"sentiment_probs"}) for p in out.predictions])


def registered_name() -> str:
    return get_config()["mlflow"]["registered_model"]


def client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(get_env().mlflow_tracking_uri)
    return mlflow.MlflowClient()


def log_and_register(model_dir: Path, run_id: str, tags: dict[str, str]) -> str:
    """Log the model dir as a pyfunc model inside the active run and register it.

    Returns the new registered version (string). Sets alias `candidate`.
    Raises RegistryError on any MLflow failure (never pretends registration happened).
    """
    name = registered_name()
    try:
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=SentimentToxicityPyfunc(),
            artifacts={"model_dir": str(model_dir)},
            code_paths=[str(project_path("src"))],
            pip_requirements=["torch", "transformers", "pandas", "numpy", "pydantic",
                              "pyyaml", "pydantic-settings", "langdetect"],
        )
        mv = mlflow.register_model(info.model_uri, name,
                                   tags={**tags, "run_id": run_id, "lifecycle": "candidate"})
        c = client()
        c.set_registered_model_alias(name, "candidate", mv.version)
    except MlflowException as exc:
        raise RegistryError(f"MLflow registration failed: {exc}") from exc
    log.info("Registered %s version %s (alias candidate)", name, mv.version)
    return str(mv.version)


def get_version_by_alias(alias: str):
    try:
        return client().get_model_version_by_alias(registered_name(), alias)
    except MlflowException:
        return None


def set_lifecycle(version: str, state: str) -> None:
    c, name = client(), registered_name()
    c.set_model_version_tag(name, version, "lifecycle", state)
    if state in ALIASES:
        c.set_registered_model_alias(name, state, version)


def promote_to_production(version: str) -> str | None:
    """Move `production` alias to `version`; previous production is tagged archived.
    Returns previous production version (or None)."""
    prev = get_version_by_alias("production")
    c, name = client(), registered_name()
    if prev is not None and prev.version != version:
        c.set_model_version_tag(name, prev.version, "lifecycle", "archived")
    set_lifecycle(version, "production")
    cand = get_version_by_alias("candidate")
    if cand is not None and cand.version == version:
        c.delete_registered_model_alias(name, "candidate")
    return prev.version if prev else None


def reject(version: str, reason: str) -> None:
    c, name = client(), registered_name()
    set_lifecycle(version, "rejected")
    c.set_model_version_tag(name, version, "rejection_reason", reason[:500])
    cand = get_version_by_alias("candidate")
    if cand is not None and cand.version == version:
        c.delete_registered_model_alias(name, "candidate")


def list_versions() -> list[dict]:
    c, name = client(), registered_name()
    try:
        versions = c.search_model_versions(f"name='{name}'")
    except MlflowException:
        return []
    try:
        aliases = c.get_registered_model(name).aliases
    except MlflowException:
        aliases = {}
    by_version: dict[str, list[str]] = {}
    for a, v in aliases.items():
        by_version.setdefault(str(v), []).append(a)
    return sorted(
        [{"version": v.version, "run_id": v.run_id, "created": v.creation_timestamp,
          "aliases": by_version.get(str(v.version), []), "tags": dict(v.tags)}
         for v in versions],
        key=lambda d: int(d["version"]), reverse=True,
    )


def download_model(version: str) -> Path:
    """Materialise a registered version into artifacts/model_cache/v<version>/ (idempotent)."""
    dest = project_path("artifacts", "model_cache", f"v{version}")
    if (dest / "metadata.json").exists():
        return dest
    uri = f"models:/{registered_name()}/{version}"
    try:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
    except MlflowException as exc:
        raise RegistryError(f"cannot download {uri}: {exc}") from exc
    src = next(local.rglob("metadata.json"), None)
    if src is None:
        raise RegistryError(f"{uri} has no model artifact (metadata.json missing)")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src.parent, dest)
    return dest
