import pandas as pd

from src.evaluation.regression import load_cases, run_suite, validate_cases
from tests.fixtures.fakes import RulePredictor


def test_curated_suite_is_valid():
    df = load_cases()
    assert validate_cases(df) == []
    assert 30 <= len(df) <= 60
    required = {"sarcasm", "negation", "slang", "abbreviation", "mixed_language", "profanity",
                "neutral_profanity", "positive_toxicity_edge", "negative_non_toxic",
                "toxicity_non_negative", "ambiguous"}
    assert required <= set(df["category"])


def test_validation_catches_bad_labels():
    df = load_cases().copy()
    df.loc[0, "expected_sentiment"] = "happy"
    assert any("invalid expected_sentiment" in p for p in validate_cases(df))


def test_run_suite_counts_and_accepts_alternatives():
    df = pd.DataFrame([
        {"id": "a", "text": "I love it", "expected_sentiment": "positive",
         "expected_toxicity": "safe", "category": "x", "rationale": "r"},
        {"id": "b", "text": "meh", "expected_sentiment": "neutral|negative",
         "expected_toxicity": "safe", "category": "x", "rationale": "r"},
        {"id": "c", "text": "you idiot", "expected_sentiment": "positive",
         "expected_toxicity": "safe", "category": "y", "rationale": "r"},
    ])
    res = run_suite(RulePredictor(), df)
    assert res.total == 3 and res.failed == 1 and res.failed_ids == ["c"]
    assert res.by_category == {"x": {"total": 2, "failed": 0}, "y": {"total": 1, "failed": 1}}


def test_critical_failure_fails_suite_even_within_tolerance(monkeypatch):
    from src.config import settings

    cfg = settings.get_config()
    monkeypatch.setitem(cfg["gates"]["regression"], "critical_cases", ["c"])
    monkeypatch.setitem(cfg["gates"]["regression"], "max_failed_cases", 10)
    df = pd.DataFrame([{"id": "c", "text": "hello", "expected_sentiment": "neutral",
                        "expected_toxicity": "toxic", "category": "z", "rationale": "r"}])
    res = run_suite(RulePredictor(), df)
    assert not res.passed and res.critical_failed_ids == ["c"]
