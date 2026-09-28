"""Curated regression suite (spec §36).

Expected labels are the *intended* labels with a written rationale, not objective truth.
Ambiguous cases list several acceptable labels separated by '|'.

Usage:
    python -m src.evaluation.regression --validate          # schema check (DVC stage)
    python -m src.evaluation.regression --model-dir <dir>   # run suite against a model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import get_config, project_path
from src.data.schemas import SENTIMENT_LABELS, TOXICITY_LABELS

log = logging.getLogger(__name__)
REQUIRED = ["id", "text", "expected_sentiment", "expected_toxicity", "category", "rationale"]


def cases_path() -> Path:
    return project_path("data", "regression", "regression_cases.csv")


def load_cases(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or cases_path(), dtype=str, keep_default_na=False)


def validate_cases(df: pd.DataFrame) -> list[str]:
    problems = []
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        return [f"missing columns {sorted(missing)}"]
    if not 30 <= len(df) <= 60:
        problems.append(f"expected 30-60 cases, got {len(df)}")
    if df["id"].duplicated().any():
        problems.append("duplicate ids")
    for row in df.itertuples():
        for col, allowed in (("expected_sentiment", SENTIMENT_LABELS),
                             ("expected_toxicity", TOXICITY_LABELS)):
            bad = set(getattr(row, col).split("|")) - set(allowed)
            if bad:
                problems.append(f"{row.id}: invalid {col} {sorted(bad)}")
        if not row.rationale.strip():
            problems.append(f"{row.id}: missing rationale")
    critical = set(get_config()["gates"]["regression"].get("critical_cases", []))
    if critical - set(df["id"]):
        problems.append(f"critical ids not in suite: {sorted(critical - set(df['id']))}")
    return problems


@dataclass
class RegressionResult:
    total: int
    failed: int
    failed_ids: list[str]
    critical_failed_ids: list[str]
    max_failed_cases: int
    passed: bool
    by_category: dict[str, dict[str, int]]
    cases: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def run_suite(predictor, df: pd.DataFrame | None = None) -> RegressionResult:
    df = load_cases() if df is None else df
    gcfg = get_config()["gates"]["regression"]
    critical = set(gcfg.get("critical_cases", []))
    preds = predictor.run(df["text"].tolist()).predictions
    cases, failed_ids = [], []
    by_cat: dict[str, dict[str, int]] = {}
    for row, p in zip(df.itertuples(), preds, strict=True):
        ok_s = p.sentiment in row.expected_sentiment.split("|")
        ok_t = p.toxicity_label in row.expected_toxicity.split("|")
        ok = ok_s and ok_t
        if not ok:
            failed_ids.append(row.id)
        cat = by_cat.setdefault(row.category, {"total": 0, "failed": 0})
        cat["total"] += 1
        cat["failed"] += int(not ok)
        cases.append({"id": row.id, "text": row.text, "category": row.category,
                      "expected_sentiment": row.expected_sentiment,
                      "predicted_sentiment": p.sentiment,
                      "expected_toxicity": row.expected_toxicity,
                      "predicted_toxicity": p.toxicity_label,
                      "toxicity_score": p.toxicity_score, "passed": ok,
                      "critical": row.id in critical})
    crit_failed = [i for i in failed_ids if i in critical]
    passed = len(failed_ids) <= gcfg["max_failed_cases"] and not crit_failed
    return RegressionResult(len(df), len(failed_ids), failed_ids, crit_failed,
                            gcfg["max_failed_cases"], passed, by_cat, cases)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--model-dir", type=Path)
    args = ap.parse_args()
    df = load_cases()
    problems = validate_cases(df)
    if problems:
        log.error("Regression suite invalid: %s", problems)
        return 1
    log.info("Regression suite valid: %d cases, %d categories", len(df), df["category"].nunique())
    if args.model_dir:
        from src.nlp.inference import Predictor

        res = run_suite(Predictor(args.model_dir), df)
        print(json.dumps({k: v for k, v in res.to_dict().items() if k != "cases"}, indent=2))
        return 0 if res.passed else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
