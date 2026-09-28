"""Phase 3 baseline: TF-IDF + Logistic Regression per task, logged to MLflow.

Usage: python -m src.training.baseline
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.settings import get_config, project_path, set_seed
from src.training.metrics import scalar_metrics, sentiment_metrics, toxicity_metrics
from src.training.tracking import lineage_tags, setup_mlflow

log = logging.getLogger(__name__)


def _load(task: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(project_path("data", "processed", f"{task}_{split}.parquet"))


def _pipeline(cfg: dict, seed: int, balanced: bool) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg["tfidf_max_features"],
                    ngram_range=tuple(cfg["ngram_range"]),
                    sublinear_tf=True,
                    lowercase=True,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg["C"],
                    max_iter=2000,
                    random_state=seed,
                    class_weight="balanced" if balanced else None,
                ),
            ),
        ]
    )


def run_baseline() -> dict:
    cfg = get_config()
    seed = set_seed()
    bcfg = cfg["baseline"]
    setup_mlflow(cfg["mlflow"]["training_experiment"])

    results: dict = {}
    with mlflow.start_run(run_name="baseline-tfidf-logreg") as run:
        mlflow.set_tags({**lineage_tags(), "model_family": "baseline", "stage": "baseline"})
        mlflow.log_params({f"baseline_{k}": v for k, v in bcfg.items()} | {"seed": seed})
        t0 = time.time()

        sent = _pipeline(bcfg, seed, balanced=False)
        tr = _load("sentiment", "train")
        sent.fit(tr["text"], tr["label"])
        te = _load("sentiment", "test")
        results.update(sentiment_metrics(te["label"], sent.predict(te["text"])))

        tox = _pipeline(bcfg, seed, balanced=True)
        tr = _load("toxicity", "train")
        tox.fit(tr["text"], tr["label"])
        te = _load("toxicity", "test")
        results.update(
            toxicity_metrics(
                te["label"],
                tox.predict_proba(te["text"])[:, 1],
                cfg["inference"]["toxicity_threshold"],
            )
        )

        mlflow.log_metric("training_seconds", time.time() - t0)
        mlflow.log_metrics({f"test_{k}": v for k, v in scalar_metrics(results).items()})
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(sent, Path(tmp) / "sentiment_tfidf_lr.joblib")
            joblib.dump(tox, Path(tmp) / "toxicity_tfidf_lr.joblib")
            (Path(tmp) / "metrics.json").write_text(json.dumps(results, indent=2))
            mlflow.log_artifacts(tmp, artifact_path="baseline")
        results["run_id"] = run.info.run_id

    out = project_path("reports", "evaluation", "baseline_metrics.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(
        "Baseline run %s: sentiment macro-F1=%.4f toxicity F1=%.4f",
        results["run_id"],
        results["sentiment_macro_f1"],
        results["toxicity_f1"],
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    run_baseline()
