"""Phase 4/5 training pipeline:
load versioned data → validate → (cap) → tokenize → train → validate → test → save →
log to MLflow → register as `candidate`.

Usage:
    python -m src.training.train                      # train + register candidate
    python -m src.training.train --epochs 1 --max-train 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config.settings import get_config, project_path, set_seed
from src.data.schemas import SENTIMENT_LABELS, TOXICITY_LABELS
from src.data.validation import validate_labeled_frame
from src.evaluation.evaluate import default_test_sets, evaluate_model_dir
from src.nlp.inference import Predictor, pick_device
from src.nlp.model import MultiTaskModel
from src.training.dataset import load_split
from src.training.metrics import calibrate_threshold, scalar_metrics
from src.training.registry import log_and_register
from src.training.tracking import lineage_tags, setup_mlflow
from src.training.trainer import TrainConfig, train

log = logging.getLogger(__name__)
N_CLASSES = {"sentiment": len(SENTIMENT_LABELS), "toxicity": len(TOXICITY_LABELS)}


@dataclass
class TrainResult:
    run_id: str
    version: str | None
    model_dir: str
    test_metrics: dict
    extra_metrics: dict


def _cap(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_train_config(overrides: dict | None = None) -> TrainConfig:
    cfg = get_config()
    t = {**cfg["training"], **(overrides or {})}
    return TrainConfig(
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        eval_batch_size=int(t["eval_batch_size"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        warmup_ratio=float(t["warmup_ratio"]),
        toxicity_loss_weight=float(t["toxicity_loss_weight"]),
        max_length=int(cfg["model"]["max_length"]),
        fp16=bool(t["fp16"]),
        seed=int(cfg["project"]["seed"]),
    )


def train_model(
    experiment: str | None = None,
    run_name: str = "multitask-distilbert",
    extra_train: dict[str, pd.DataFrame] | None = None,
    extra_eval: dict[str, dict[str, pd.DataFrame]] | None = None,
    overrides: dict | None = None,
    tags: dict[str, str] | None = None,
    register: bool = True,
    extra_dataset_version: str | None = None,
    reference_texts: list[str] | None = None,
    corrupt_labels: float = 0.0,
    calibrate: bool | None = None,
) -> TrainResult:
    """Train one model end-to-end.

    extra_train          rows added to each task's train split (e.g. labelled drift data)
    extra_eval           {prefix: test_sets} evaluated in addition to the standard test sets
    extra_dataset_version content hash of the extra data, folded into `dataset_version`
    reference_texts      saved with the model as its drift reference (reference_texts.json)
    corrupt_labels       FAULT INJECTION for the safety-gate demo: fraction of training labels
                         replaced with random wrong labels (simulates a broken annotation batch)
    calibrate            pick the toxicity threshold on validation data (default: config)
    """
    cfg = get_config()
    if calibrate is None:
        calibrate = bool(cfg["inference"].get("calibrate_threshold", False))
    seed = set_seed()
    overrides = overrides or {}
    tcfg = build_train_config(overrides)
    max_per_task = overrides.get("max_train_per_task", cfg["training"]["max_train_per_task"])

    train_data, val_data = {}, {}
    for task in ("sentiment", "toxicity"):
        tr = _cap(load_split(task, "train"), max_per_task, seed)
        if extra_train and task in extra_train:
            tr = pd.concat([tr, extra_train[task][["text", "label"]]], ignore_index=True)
            tr = tr.drop_duplicates("text").reset_index(drop=True)
        va = load_split(task, "val")
        for name, df in (("train", tr), ("val", va)):
            validate_labeled_frame(
                df, f"{task}/{name}", N_CLASSES[task], min_rows=10
            ).raise_if_failed()
        if corrupt_labels > 0:
            tr = _corrupt(tr, corrupt_labels, N_CLASSES[task], seed)
        train_data[task], val_data[task] = tr, va

    device = pick_device()
    pretrained = cfg["model"]["pretrained_name"]
    setup_mlflow(experiment or cfg["mlflow"]["training_experiment"])
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        lineage = lineage_tags()
        if extra_dataset_version:
            lineage["dataset_version"] = f"{lineage['dataset_version']}+{extra_dataset_version}"
        mlflow.set_tags({**lineage, "model_family": "multitask-distilbert", **(tags or {})})
        mlflow.log_params(
            {
                **{k: v for k, v in asdict(tcfg).items()},
                "pretrained_name": pretrained,
                "tokenizer": pretrained,
                "device": device.type,
                "max_train_per_task": max_per_task,
                "dropout": cfg["model"]["dropout"],
                **{f"n_train_{t}": len(df) for t, df in train_data.items()},
                "extra_train_rows": sum(len(v) for v in (extra_train or {}).values()),
                "corrupt_labels": corrupt_labels,
            }
        )
        log.info("Training on %s: %s", device, {t: len(df) for t, df in train_data.items()})

        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        model = MultiTaskModel.from_pretrained_encoder(pretrained, cfg["model"]["dropout"])
        result = train(
            model,
            tokenizer,
            train_data,
            val_data,
            tcfg,
            device,
            on_epoch_end=lambda rec: mlflow.log_metrics(
                {k: v for k, v in rec.items() if k != "epoch"}, step=rec["epoch"]
            ),
        )
        mlflow.log_metric("training_seconds", result["training_seconds"])

        out_dir = project_path("artifacts", "runs", run_id, "model")
        model.save(
            out_dir,
            tokenizer,
            {
                "pretrained_name": pretrained,
                "max_length": tcfg.max_length,
                "pooling": cfg["model"]["embedding_pooling"],
                "run_id": run_id,
                "dataset_version": lineage["dataset_version"],
                "git_commit": lineage["git_commit"],
            },
        )
        if reference_texts:
            (out_dir / "reference_texts.json").write_text(
                json.dumps(reference_texts, ensure_ascii=False), encoding="utf-8"
            )
        del model
        if calibrate:
            # per-model toxicity threshold, chosen on the VALIDATION split only
            val_scores = Predictor(out_dir).run(val_data["toxicity"]["text"].tolist())
            threshold = calibrate_threshold(
                val_data["toxicity"]["label"], val_scores.toxicity_scores
            )
            meta_path = out_dir / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["toxicity_threshold"] = threshold
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            mlflow.log_param("toxicity_threshold", threshold)
            log.info("Calibrated toxicity threshold on validation set: %.2f", threshold)
        # evaluate from the SAVED artifact, which also proves the artifact loads (gate 4)
        test = evaluate_model_dir(out_dir, default_test_sets())
        mlflow.log_metrics({f"test_{k}": v for k, v in scalar_metrics(test).items()})
        extra: dict = {}
        for prefix, sets in (extra_eval or {}).items():
            m = evaluate_model_dir(out_dir, sets)
            extra[prefix] = m
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in scalar_metrics(m).items()})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "evaluation.json"
            p.write_text(
                json.dumps({"test": test, **extra, "history": result["history"]}, indent=2)
            )
            mlflow.log_artifact(str(p))

        version = None
        if register:
            version = log_and_register(
                out_dir,
                run_id,
                {
                    "dataset_version": lineage["dataset_version"],
                    "git_commit": lineage["git_commit"],
                    **{k: str(v) for k, v in (tags or {}).items()},
                },
            )
            mlflow.set_tag("registered_version", version)

    log.info(
        "Run %s done: sentiment macro-F1=%.4f toxicity F1=%.4f recall=%.4f version=%s",
        run_id,
        test["sentiment_macro_f1"],
        test["toxicity_f1"],
        test["toxicity_recall"],
        version,
    )
    return TrainResult(run_id, version, str(out_dir), test, extra)


def _corrupt(df: pd.DataFrame, rate: float, n_classes: int, seed: int) -> pd.DataFrame:
    """Replace `rate` of labels with a different random class (seeded)."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    idx = rng.choice(len(df), int(len(df) * rate), replace=False)
    shift = rng.integers(1, n_classes, size=len(idx))
    df.loc[df.index[idx], "label"] = (df["label"].to_numpy()[idx] + shift) % n_classes
    return df


def cleanup_run_artifacts(run_id: str) -> None:
    shutil.rmtree(project_path("artifacts", "runs", run_id), ignore_errors=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train", type=int, help="cap rows per task")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.max_train:
        overrides["max_train_per_task"] = args.max_train
    res = train_model(overrides=overrides, register=not args.no_register)
    print(
        json.dumps(
            {
                "run_id": res.run_id,
                "version": res.version,
                "test": scalar_metrics(res.test_metrics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
