"""Evidently (0.7.x API, verified against the installed package) column-level drift report.

Why the split with src/monitoring/drift.py: the primary embedding signal is our calibrated
MMD^2 permutation test on 768-d embeddings (thresholds documented in configs/drift.yaml).
Evidently runs alongside it on the tabular view of each window - detected language,
predicted sentiment, predicted toxicity label, and comment length - and produces the
per-column statistical tests plus an HTML report for humans. Its result is recorded as
supporting evidence; it does not vote in the DriftDecisionEngine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.monitoring.decision import WindowData

log = logging.getLogger(__name__)

CATEGORICAL = ["language", "sentiment", "toxicity_label"]
NUMERICAL = ["text_length"]


def window_frame(w: WindowData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "language": w.languages,
            "sentiment": w.sentiments,
            "toxicity_label": w.toxicity_labels,
            "text_length": [len(t) for t in w.texts] if w.texts else [0] * len(w),
        }
    )


def run_evidently(
    reference: WindowData, current: WindowData, html_path: Path | None = None
) -> dict:
    """Return a compact summary; save the full HTML report if `html_path` is given.
    Returns {"error": ...} instead of raising, so monitoring never fails because of it."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:  # pragma: no cover
        return {"error": f"evidently unavailable: {exc}"}
    try:
        dd = DataDefinition(categorical_columns=CATEGORICAL, numerical_columns=NUMERICAL)
        ref = Dataset.from_pandas(window_frame(reference), data_definition=dd)
        cur = Dataset.from_pandas(window_frame(current), data_definition=dd)
        snap = Report([DataDriftPreset()]).run(cur, ref)
        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            snap.save_html(str(html_path))
        out: dict = {"columns": {}}
        for m in snap.dict()["metrics"]:
            name = str(m.get("metric_name", ""))
            val = m.get("value")
            if name.startswith("DriftedColumnsCount"):
                out["drifted_columns"] = val.get("count") if isinstance(val, dict) else val
                out["drifted_share"] = val.get("share") if isinstance(val, dict) else None
            elif name.startswith("ValueDrift"):
                col = name.split("column=")[1].split(",")[0]
                method = name.split("method=")[1].split(",")[0]
                out["columns"][col] = {"method": method, "p_value": float(val)}
        if html_path is not None:
            out["html"] = str(html_path)
        return out
    except Exception as exc:  # noqa: BLE001 - supplementary signal must not break monitoring
        log.warning("Evidently report failed: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
