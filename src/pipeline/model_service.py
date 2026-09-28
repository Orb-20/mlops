"""Process-wide access to the current production model (loaded once, reloaded on promotion)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from src.nlp.inference import Predictor
from src.training import registry

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class ModelInfo:
    version: str
    run_id: str | None
    dataset_version: str | None
    git_commit: str | None
    created: int | None
    lifecycle: str | None


class ModelService:
    """Caches the production Predictor; re-checks the `production` alias at most every
    `check_seconds`, so a promotion is picked up without restarting the API."""

    def __init__(self, check_seconds: float = 30.0):
        self._lock = threading.Lock()
        self._predictor: Predictor | None = None
        self._info: ModelInfo | None = None
        self._checked = 0.0
        self.check_seconds = check_seconds

    def _resolve(self) -> ModelInfo:
        try:
            mv = registry.get_version_by_alias("production")
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(f"MLflow registry unavailable: {exc}") from exc
        if mv is None:
            raise ModelUnavailable("No production model registered. Run "
                                   "`python -m src.training.train` then "
                                   "`python -m src.evaluation.promotion --alias candidate`.")
        t = mv.tags
        return ModelInfo(str(mv.version), mv.run_id, t.get("dataset_version"),
                         t.get("git_commit"), mv.creation_timestamp, t.get("lifecycle"))

    def get(self) -> tuple[Predictor, ModelInfo]:
        with self._lock:
            now = time.monotonic()
            if self._predictor is None or now - self._checked > self.check_seconds:
                info = self._resolve()
                self._checked = now
                if self._info is None or info.version != self._info.version:
                    log.info("Loading production model v%s", info.version)
                    path = registry.download_model(info.version)
                    self._predictor = Predictor(path, version=info.version)
                    self._info = info
            assert self._predictor is not None and self._info is not None
            return self._predictor, self._info

    def info(self) -> ModelInfo:
        return self._resolve()

    def invalidate(self) -> None:
        with self._lock:
            self._checked = 0.0


_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
