import numpy as np
import pytest

from src.monitoring.decision import DriftDecisionEngine, WindowData
from src.monitoring.drift import (
    categorical_drift,
    embedding_drift,
    mmd_rbf,
    proportions,
    total_variation,
)
from src.monitoring.vocabulary import vocabulary_report

CFG = {
    "minimum_samples": 50, "warning_ratio": 0.7, "overall_threshold": 0.5,
    "embedding": {"enabled": True, "threshold": 0.05, "p_value": 0.05, "permutations": 100,
                  "max_samples": 200, "weight": 0.7},
    "language": {"enabled": True, "threshold": 0.1, "weight": 0.15},
    "prediction": {"enabled": True, "threshold": 0.1, "weight": 0.15},
    "vocabulary": {"enabled": True, "top_k": 10, "min_count": 2},
}


def gauss(n, shift=0.0, seed=0, d=16):
    return np.random.default_rng(seed).normal(shift, 1.0, size=(n, d))


def test_mmd_same_distribution_not_significant():
    mmd2, p, _ = mmd_rbf(gauss(150, seed=1), gauss(150, seed=2), permutations=100)
    assert mmd2 < 0.02 and p > 0.05


def test_mmd_shifted_distribution_significant():
    mmd2, p, _ = mmd_rbf(gauss(150, seed=1), gauss(150, shift=1.0, seed=2), permutations=100)
    assert mmd2 > 0.1 and p < 0.05


def test_mmd_deterministic():
    a, b = gauss(80, seed=3), gauss(80, shift=0.5, seed=4)
    assert mmd_rbf(a, b, 50, seed=7) == mmd_rbf(a, b, 50, seed=7)


def test_mmd_needs_samples():
    with pytest.raises(ValueError):
        mmd_rbf(gauss(1), gauss(5))


def test_embedding_drift_statuses():
    stable = embedding_drift(gauss(150, seed=1), gauss(150, seed=2), CFG["embedding"], 0.7)
    drift = embedding_drift(gauss(150, seed=1), gauss(150, 1.0, seed=2), CFG["embedding"], 0.7)
    assert stable.status == "stable" and not stable.drifted
    assert drift.status == "drift" and drift.drifted


def test_total_variation():
    assert total_variation({"a": 1.0}, {"b": 1.0}) == 1.0
    assert total_variation({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert proportions(["a", "a", "b", "c"], ["a", "b"]) == {"a": 0.5, "b": 0.25}


def test_categorical_drift_takes_max_subsignal():
    r = categorical_drift("prediction",
                          {"s": {"x": 1.0, "y": 0.0}, "t": {"x": 0.5, "y": 0.5}},
                          {"s": {"x": 1.0, "y": 0.0}, "t": {"x": 0.3, "y": 0.7}},
                          {"threshold": 0.1}, 0.7)
    assert r.score == 0.2 and r.drifted


def _window(emb, lang="en", sent="positive", tox="safe", texts=None):
    n = len(emb)
    return WindowData(emb, [lang] * n, [sent] * n, [tox] * n, texts or ["good video"] * n)


class TestDecisionEngine:
    def test_insufficient_samples_never_triggers(self):
        d = DriftDecisionEngine(CFG, seed=0).evaluate(_window(gauss(100)),
                                                     _window(gauss(10, 3.0)))
        assert d.overall_status == "insufficient_data"
        assert not d.drift_detected and not d.retraining_recommended

    def test_stable(self):
        d = DriftDecisionEngine(CFG, seed=0).evaluate(_window(gauss(150, seed=1)),
                                                     _window(gauss(150, seed=2)))
        assert d.overall_status == "stable" and not d.drift_detected and d.reasons == []

    def test_embedding_drift_alone_triggers(self):
        d = DriftDecisionEngine(CFG, seed=0).evaluate(_window(gauss(150, seed=1)),
                                                     _window(gauss(150, 1.0, seed=2)))
        assert d.drift_detected and d.reasons == ["embedding_drift"]
        assert d.severity == "medium" and d.retraining_recommended

    def test_auxiliary_signals_alone_only_warn(self):
        """Language + prediction shift without embedding drift must not trigger retraining."""
        ref = _window(gauss(150, seed=1))
        cur = _window(gauss(150, seed=2), lang="hinglish", sent="negative", tox="toxic")
        d = DriftDecisionEngine(CFG, seed=0).evaluate(ref, cur)
        assert not d.drift_detected and d.overall_status == "warning"
        assert set(d.reasons) == {"language_distribution_shift", "prediction_distribution_shift"}

    def test_all_signals_high_severity(self):
        ref = _window(gauss(150, seed=1))
        cur = _window(gauss(150, 1.0, seed=2), lang="hinglish", sent="negative")
        d = DriftDecisionEngine(CFG, seed=0).evaluate(ref, cur)
        assert d.drift_detected and d.severity == "high" and len(d.reasons) == 3
        assert set(d.to_dict()) >= {"drift_detected", "severity", "reasons", "scores",
                                    "thresholds", "sample_count"}


def test_vocabulary_report_new_terms():
    ref = ["this movie is very good"] * 5 + ["i disliked this film"] * 5
    cur = ["bro this is peak"] * 4 + ["absolute cinema"] * 4 + ["this movie is good"] * 2
    v = vocabulary_report(ref, cur, top_k=10, min_count=2)
    new = {t["term"] for t in v["new_terms"]}
    assert {"peak", "absolute cinema", "bro"} <= new
    assert "movie" not in new
    assert 0 < v["oov_rate"] < 1
