"""Evaluate a saved model artifact on the held-out labelled test sets.

Ground-truth metrics come ONLY from labelled data (processed test splits, drift test split,
regression suite). Live YouTube comments are unlabelled and never used here (spec §33).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.settings import get_config, project_path
from src.nlp.inference import Predictor
from src.training.metrics import sentiment_metrics, toxicity_metrics

log = logging.getLogger(__name__)


def default_test_sets() -> dict[str, pd.DataFrame]:
    d = project_path("data", "processed")
    return {t: pd.read_parquet(d / f"{t}_test.parquet") for t in ("sentiment", "toxicity")}


def evaluate_predictor(
    predictor: Predictor, test_sets: dict[str, pd.DataFrame], prefix: str = ""
) -> dict:
    # each model is evaluated at its own decision threshold (calibrated on validation data,
    # or the configured default for models without one)
    thr = getattr(predictor, "threshold", get_config()["inference"]["toxicity_threshold"])
    out: dict = {}
    if "sentiment" in test_sets:
        df = test_sets["sentiment"]
        probs = predictor.run(df["text"].tolist()).sentiment_probs
        out.update(sentiment_metrics(df["label"], probs.argmax(1)))
    if "toxicity" in test_sets:
        df = test_sets["toxicity"]
        scores = predictor.run(df["text"].tolist()).toxicity_scores
        out.update(toxicity_metrics(df["label"], scores, thr))
    return {f"{prefix}{k}": v for k, v in out.items()}


def evaluate_model_dir(
    model_dir: Path, test_sets: dict[str, pd.DataFrame] | None = None, version: str = "local"
) -> dict:
    predictor = Predictor(model_dir, version=version)
    return evaluate_predictor(predictor, test_sets or default_test_sets())
