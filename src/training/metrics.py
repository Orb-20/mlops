"""Task metrics (spec §18). Toxicity is imbalanced, so accuracy is never reported alone."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def sentiment_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0
    )
    return {
        "sentiment_accuracy": float(accuracy_score(y_true, y_pred)),
        "sentiment_macro_precision": float(p),
        "sentiment_macro_recall": float(r),
        "sentiment_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "sentiment_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2], zero_division=0)
        ),
        "sentiment_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def toxicity_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """y_score = P(toxic)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    out = {
        "toxicity_precision": float(p),
        "toxicity_recall": float(r),
        "toxicity_f1": float(f),
        "toxicity_accuracy": float(accuracy_score(y_true, y_pred)),
        "toxicity_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["toxicity_roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["toxicity_pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def calibrate_threshold(
    y_true, y_score, lo: float = 0.05, hi: float = 0.95, step: float = 0.01
) -> float:
    """F1-optimal decision threshold on a VALIDATION set (never the test set).
    Ties resolve to the threshold closest to 0.5."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        tie = abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
        if f1 > best_f1 + 1e-12 or tie:
            best_t, best_f1 = float(round(t, 2)), f1
    return best_t


def scalar_metrics(metrics: dict) -> dict[str, float]:
    """Drop non-scalar entries (confusion matrices) for MLflow log_metrics."""
    return {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
