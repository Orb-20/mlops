import pytest

from src.evaluation.promotion import GateResult, PromotionDecisionEngine, check_artifact

GATES = {
    "minimum_metrics": {
        "sentiment_macro_f1_min": 0.6,
        "toxicity_f1_min": 0.65,
        "toxicity_recall_min": 0.6,
    },
    "regression": {"max_failed_cases": 5},
    "production_comparison": {
        "max_sentiment_macro_f1_drop": 0.02,
        "max_toxicity_f1_drop": 0.02,
        "max_toxicity_recall_drop": 0.03,
        "max_additional_regression_failures": 1,
    },
}
GOOD = {"sentiment_macro_f1": 0.70, "toxicity_f1": 0.75, "toxicity_recall": 0.80}
REG_OK = {
    "passed": True,
    "failed": 3,
    "max_failed_cases": 5,
    "failed_ids": ["a", "b", "c"],
    "critical_failed_ids": [],
}
ART_OK = GateResult("artifact_validity", True, {})


@pytest.fixture
def engine():
    return PromotionDecisionEngine(GATES)


def test_bootstrap_promotes_without_production(engine):
    d = engine.decide("1", GOOD, REG_OK, ART_OK)
    assert d.decision == "PROMOTE"
    assert "bootstrap_no_production_model" in d.reasons


def test_better_candidate_promoted(engine):
    prod = {"sentiment_macro_f1": 0.68, "toxicity_f1": 0.74, "toxicity_recall": 0.79}
    d = engine.decide("5", GOOD, REG_OK, ART_OK, "4", prod, REG_OK)
    assert d.decision == "PROMOTE"
    assert d.reasons == [
        "artifact_valid",
        "minimum_metrics_passed",
        "regression_tests_passed",
        "no_critical_metric_regression",
    ]


def test_small_drop_within_margin_is_allowed(engine):
    prod = {"sentiment_macro_f1": 0.71, "toxicity_f1": 0.76, "toxicity_recall": 0.82}
    assert engine.decide("5", GOOD, REG_OK, ART_OK, "4", prod, REG_OK).decision == "PROMOTE"


def test_accuracy_alone_cannot_promote_recall_drop(engine):
    """Candidate better on sentiment but loses toxicity recall -> reject."""
    cand = {"sentiment_macro_f1": 0.80, "toxicity_f1": 0.75, "toxicity_recall": 0.70}
    prod = {"sentiment_macro_f1": 0.60, "toxicity_f1": 0.75, "toxicity_recall": 0.80}
    d = engine.decide("5", cand, REG_OK, ART_OK, "4", prod, REG_OK)
    assert d.decision == "REJECT"
    assert d.reasons == ["production_comparison_failed"]


def test_minimum_metrics_fail(engine):
    bad = {**GOOD, "toxicity_f1": 0.5}
    d = engine.decide("2", bad, REG_OK, ART_OK)
    assert d.decision == "REJECT" and "minimum_metrics_failed" in d.reasons


def test_regression_fail(engine):
    reg = {**REG_OK, "passed": False, "failed": 9}
    d = engine.decide("6", GOOD, reg, ART_OK, "5", GOOD, REG_OK)
    assert d.decision == "REJECT" and "regression_gate_failed" not in d.reasons
    assert "regression_failed" in d.reasons


def test_more_regression_failures_than_production(engine):
    reg = {**REG_OK, "failed": 5}
    prod_reg = {**REG_OK, "failed": 2}
    d = engine.decide("6", GOOD, reg, ART_OK, "5", GOOD, prod_reg)
    assert d.decision == "REJECT" and d.reasons == ["production_comparison_failed"]


def test_invalid_artifact_short_circuits(engine):
    d = engine.decide("7", {}, {}, GateResult("artifact_validity", False, {"missing": ["x"]}))
    assert d.decision == "REJECT" and d.reasons == ["artifact_validity_failed"]
    assert len(d.gates) == 1


def test_check_artifact_missing_files(tmp_path):
    g = check_artifact(tmp_path)
    assert not g.passed and "heads.pt" in g.details["missing"]
