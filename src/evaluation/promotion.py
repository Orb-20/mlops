"""PromotionDecisionEngine (spec §37-38).

Gates (all must pass):
  1. minimum metrics         candidate absolute quality floor
  2. regression suite        failures <= tolerance and no critical case failed
  3. production comparison   no critical metric drops by more than the allowed margin
  4. artifact validity       files present, loads, inference returns valid outputs

The decision logic (`decide`) is a pure function of metrics so it is unit-testable; `run_promotion`
gathers the evidence and applies the decision to the MLflow registry. A rejected candidate never
touches the `production` alias, so the old production model stays served.

Usage: python -m src.evaluation.promotion --version <N> | --alias candidate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.config.settings import get_config
from src.data.schemas import SENTIMENT_LABELS, TOXICITY_LABELS
from src.nlp.model import REQUIRED_ARTIFACT_FILES

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    name: str
    passed: bool
    details: dict


@dataclass
class PromotionDecision:
    decision: str  # PROMOTE | REJECT
    candidate_version: str
    production_version: str | None
    reasons: list[str]
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_artifact(model_dir: Path) -> GateResult:
    details: dict = {"model_dir": str(model_dir)}
    missing = [f for f in REQUIRED_ARTIFACT_FILES if not (Path(model_dir) / f).exists()]
    if missing:
        details["missing"] = missing
        return GateResult("artifact_validity", False, details)
    try:
        from src.nlp.inference import Predictor

        out = Predictor(model_dir).run(["this is great", "you are an idiot"])
        ok = (
            len(out.predictions) == 2
            and all(p.sentiment in SENTIMENT_LABELS for p in out.predictions)
            and all(p.toxicity_label in TOXICITY_LABELS for p in out.predictions)
            and np.isfinite(out.embeddings).all()
            and out.embeddings.shape[0] == 2
        )
        details["inference_ok"] = bool(ok)
        return GateResult("artifact_validity", bool(ok), details)
    except Exception as exc:  # noqa: BLE001 - any load/inference failure fails the gate
        details["error"] = f"{type(exc).__name__}: {exc}"
        return GateResult("artifact_validity", False, details)


class PromotionDecisionEngine:
    def __init__(self, gates: dict | None = None):
        self.g = gates or get_config()["gates"]

    def minimum_metrics(self, m: dict) -> GateResult:
        mm = self.g["minimum_metrics"]
        checks = {
            "sentiment_macro_f1": (m["sentiment_macro_f1"], mm["sentiment_macro_f1_min"]),
            "toxicity_f1": (m["toxicity_f1"], mm["toxicity_f1_min"]),
            "toxicity_recall": (m["toxicity_recall"], mm["toxicity_recall_min"]),
        }
        failed = [k for k, (v, lo) in checks.items() if v < lo]
        return GateResult(
            "minimum_metrics",
            not failed,
            {
                "checks": {k: {"value": round(v, 4), "min": lo} for k, (v, lo) in checks.items()},
                "failed": failed,
            },
        )

    def regression(self, reg: dict) -> GateResult:
        return GateResult(
            "regression",
            bool(reg["passed"]),
            {
                "failed": reg["failed"],
                "max_failed_cases": reg["max_failed_cases"],
                "failed_ids": reg["failed_ids"],
                "critical_failed_ids": reg["critical_failed_ids"],
            },
        )

    def production_comparison(
        self, cand: dict, cand_reg: dict, prod: dict | None, prod_reg: dict | None
    ) -> GateResult:
        if prod is None:
            return GateResult(
                "production_comparison",
                True,
                {"note": "no production model yet (bootstrap); comparison skipped"},
            )
        pc = self.g["production_comparison"]
        checks = {
            "sentiment_macro_f1": pc["max_sentiment_macro_f1_drop"],
            "toxicity_f1": pc["max_toxicity_f1_drop"],
            "toxicity_recall": pc["max_toxicity_recall_drop"],
        }
        rows, failed = {}, []
        for k, max_drop in checks.items():
            delta = cand[k] - prod[k]
            rows[k] = {
                "candidate": round(cand[k], 4),
                "production": round(prod[k], 4),
                "delta": round(delta, 4),
                "max_drop": max_drop,
            }
            if delta < -max_drop:
                failed.append(k)
        extra = cand_reg["failed"] - (prod_reg or {"failed": 0})["failed"]
        rows["regression_failures"] = {
            "candidate": cand_reg["failed"],
            "production": (prod_reg or {}).get("failed"),
            "max_additional": pc["max_additional_regression_failures"],
        }
        if prod_reg is not None and extra > pc["max_additional_regression_failures"]:
            failed.append("regression_failures")
        return GateResult("production_comparison", not failed, {"checks": rows, "failed": failed})

    def decide(
        self,
        candidate_version: str,
        cand_metrics: dict,
        cand_reg: dict,
        artifact: GateResult,
        production_version: str | None = None,
        prod_metrics: dict | None = None,
        prod_reg: dict | None = None,
    ) -> PromotionDecision:
        gates = [artifact]
        if artifact.passed:
            gates += [
                self.minimum_metrics(cand_metrics),
                self.regression(cand_reg),
                self.production_comparison(cand_metrics, cand_reg, prod_metrics, prod_reg),
            ]
        failed = [g for g in gates if not g.passed]
        if failed:
            reasons = [f"{g.name}_failed" for g in failed]
            return PromotionDecision(
                "REJECT", candidate_version, production_version, reasons, gates
            )
        reasons = [
            "artifact_valid",
            "minimum_metrics_passed",
            "regression_tests_passed",
            (
                "no_critical_metric_regression"
                if prod_metrics is not None
                else "bootstrap_no_production_model"
            ),
        ]
        return PromotionDecision("PROMOTE", candidate_version, production_version, reasons, gates)


def run_promotion(
    candidate_version: str, extra_test_sets: dict | None = None, apply: bool = True
) -> PromotionDecision:
    """Evaluate candidate vs production on the SAME labelled data and apply the decision."""
    from src.evaluation.evaluate import default_test_sets, evaluate_predictor
    from src.evaluation.regression import run_suite
    from src.nlp.inference import Predictor
    from src.training import registry

    test_sets = default_test_sets()
    cand_dir = registry.download_model(candidate_version)
    artifact = check_artifact(cand_dir)
    engine = PromotionDecisionEngine()
    prod = registry.get_version_by_alias("production")
    prod_version = str(prod.version) if prod else None

    cand_metrics, cand_reg = {}, {
        "passed": False,
        "failed": -1,
        "max_failed_cases": 0,
        "failed_ids": [],
        "critical_failed_ids": [],
    }
    prod_metrics = prod_reg = None
    extra: dict[str, dict] = {}  # informational: e.g. held-out drift test set
    if artifact.passed:
        cp = Predictor(cand_dir, version=candidate_version)
        cand_metrics = evaluate_predictor(cp, test_sets)
        cand_reg = run_suite(cp).to_dict()
        for prefix, sets in (extra_test_sets or {}).items():
            extra.setdefault(prefix, {})["candidate"] = evaluate_predictor(cp, sets)
        del cp
        if prod is not None and str(prod.version) != str(candidate_version):
            pp = Predictor(registry.download_model(prod.version), version=prod.version)
            prod_metrics = evaluate_predictor(pp, test_sets)
            prod_reg = run_suite(pp).to_dict()
            for prefix, sets in (extra_test_sets or {}).items():
                extra.setdefault(prefix, {})["production"] = evaluate_predictor(pp, sets)
            del pp
    decision = engine.decide(
        candidate_version, cand_metrics, cand_reg, artifact, prod_version, prod_metrics, prod_reg
    )
    log.info(
        "Promotion decision for v%s: %s %s", candidate_version, decision.decision, decision.reasons
    )
    if apply:
        if decision.decision == "PROMOTE":
            registry.promote_to_production(candidate_version)
            log.info(
                "Candidate v%s promoted to production (previous: %s)",
                candidate_version,
                prod_version,
            )
        else:
            registry.reject(candidate_version, ", ".join(decision.reasons))
            log.warning(
                "Candidate v%s rejected; production remains %s", candidate_version, prod_version
            )
    decision._evidence = {  # type: ignore[attr-defined]
        "candidate_metrics": cand_metrics,
        "production_metrics": prod_metrics,
        "candidate_regression": {k: v for k, v in cand_reg.items() if k != "cases"},
        "production_regression": (
            {k: v for k, v in prod_reg.items() if k != "cases"} if prod_reg else None
        ),
        "extra_evaluations": extra,
    }
    return decision


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--version")
    g.add_argument("--alias")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not change registry")
    args = ap.parse_args()
    from src.training import registry

    version = args.version
    if args.alias:
        mv = registry.get_version_by_alias(args.alias)
        if mv is None:
            log.error("No model version has alias %r", args.alias)
            return 1
        version = mv.version
    d = run_promotion(version, apply=not args.dry_run)
    print(json.dumps(d.to_dict(), indent=2, default=str))
    return 0 if d.decision == "PROMOTE" else 3


if __name__ == "__main__":
    sys.exit(main())
