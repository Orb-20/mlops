"""Reference window: what 'normal' input looks like for a given production model.

Reference texts = seeded sample of the held-out *validation* splits (the training distribution,
never the test split). They are embedded and predicted by the model version being monitored,
because embeddings from different encoders are not comparable. Cached per model version at
artifacts/reference/v<version>.npz.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_config, project_path
from src.monitoring.decision import WindowData
from src.nlp.language import detect_many

log = logging.getLogger(__name__)


def reference_texts(n: int | None = None, seed: int | None = None, offset: int = 0) -> list[str]:
    """Deterministic sample of validation texts, half sentiment / half toxicity.
    `offset` selects a disjoint sample (used to build a 'stable' comparison window)."""
    cfg = get_config()
    n = n or cfg["drift"]["reference_size"]
    seed = cfg["project"]["seed"] if seed is None else seed
    d = project_path("data", "processed")
    parts = []
    for task in ("sentiment", "toxicity"):
        df = pd.read_parquet(d / f"{task}_val.parquet").sample(frac=1.0, random_state=seed)
        k = n // 2
        parts.append(df["text"].iloc[offset * k : (offset + 1) * k])
    return pd.concat(parts).tolist()


def window_from_texts(predictor, texts: list[str]) -> WindowData:
    out = predictor.run(texts)
    langs = [r.language for r in detect_many(texts)]
    return WindowData(
        out.embeddings,
        langs,
        [p.sentiment for p in out.predictions],
        [p.toxicity_label for p in out.predictions],
        list(texts),
    )


def texts_for_model(predictor) -> list[str]:
    """A model may ship its own reference (after drift-triggered retraining the reference is
    reset to historical + recent data). Otherwise use the default validation sample."""
    path = Path(getattr(predictor, "model_dir", "")) / "reference_texts.json"
    if path.is_file():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return reference_texts()


def _path(version: str) -> Path:
    return project_path("artifacts", "reference", f"v{version}.npz")


def save_window(w: WindowData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=w.embeddings.astype(np.float16),
        languages=np.array(w.languages),
        sentiments=np.array(w.sentiments),
        toxicity=np.array(w.toxicity_labels),
        texts=np.array(w.texts),
    )


def load_window_file(path: Path) -> WindowData:
    z = np.load(path, allow_pickle=False)
    return WindowData(
        z["embeddings"].astype(np.float32),
        z["languages"].tolist(),
        z["sentiments"].tolist(),
        z["toxicity"].tolist(),
        z["texts"].tolist(),
    )


def get_reference(predictor, version: str, rebuild: bool = False) -> WindowData:
    path = _path(version)
    if path.exists() and not rebuild:
        return load_window_file(path)
    log.info("Building reference window for model v%s", version)
    w = window_from_texts(predictor, texts_for_model(predictor))
    save_window(w, path)
    return w
