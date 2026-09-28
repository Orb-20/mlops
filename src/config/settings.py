"""Central configuration: YAML files in configs/ + environment variables (.env).

All tunables live in configs/*.yaml; secrets and deployment-specific URLs come from env vars.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class EnvSettings(BaseSettings):
    """Environment-driven settings. Secrets never get defaults that look real."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    youtube_api_key: str = ""
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'local.db').as_posix()}"
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    environment: str = "local"
    api_url: str = "http://localhost:8000"

    @property
    def has_youtube_key(self) -> bool:
        return bool(self.youtube_api_key.strip())


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache
def get_env() -> EnvSettings:
    return EnvSettings()


@lru_cache
def get_config() -> dict[str, Any]:
    """Merged view of all YAML config files."""
    cfg: dict[str, Any] = {}
    for name in ("config.yaml", "model.yaml", "drift.yaml", "gates.yaml"):
        cfg.update(_load_yaml(name))
    return cfg


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def set_seed(seed: int | None = None) -> int:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    seed = get_config()["project"]["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed
