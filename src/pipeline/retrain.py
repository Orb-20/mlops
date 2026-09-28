"""Drift-triggered retraining (spec §32). Drift NEVER deploys directly:

drift event → collect labelled data → validate → dedupe → quality checks → version (DVC)
→ train candidate → evaluate → MLflow → regression → compare with production → PROMOTE / REJECT

Label policy (spec §33-34): live comments are unlabelled and are used only for drift signals.
The labelled data that enters training is the controlled drift dataset (data/drift/train.parquet,
intended labels by construction). Model predictions are never used as training labels.

Usage:
    python -m src.pipeline.retrain                        # oldest pending drift event
    python -m src.pipeline.retrain --event-id 7
    python -m src.pipeline.retrain --scenario corrupted_labels   # safety-gate demo (must REJECT)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd

from src.config.settings import get_config, project_path
from src.data.preprocessing import clean_frame
from src.data.validation import check_no_split_leakage, validate_labeled_frame
from src.data.versioning import dataset_version
from src.database import repository as repo
from src.database.connection import init_db, session_scope
from src.database.models import DriftEvent
from src.monitoring.reference import reference_texts
from src.monitoring.simulation import as_task_frames

log = logging.getLogger(__name__)

SCENARIOS = {
    "normal": {"corrupt_labels": 0.0},
    # Fault injection: a broken annotation batch. Used to demonstrate the safety gate.
    "corrupted_labels": {"corrupt_labels": 0.6},
}


class RetrainingError(RuntimeError):
    pass


@dataclass
class RetrainOutcome:
    drift_event_id: int | None
    candidate_version: str | None
    production_before: str | None
    production_after: str | None
    decision: str
    reasons: list[str]
    run_id: str | None
    scenario: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def drift_dir():
    return project_path("data", "drift")


def load_drift_training_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    d = drift_dir()
    train_p, test_p = d / "train.parquet", d / "test.parquet"
    if not train_p.exists() or not test_p.exists():
        raise RetrainingError(
            "No labelled drift data at data/drift/. Run "
            "`python scripts/simulate_drift.py --build-only` first."
        )
    return pd.read_parquet(train_p), pd.read_parquet(test_p), dataset_version(d)


def quality_checks(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate, clean, dedupe, remove leakage against every evaluation set."""
    report: dict = {"rows_in": len(train)}
    train = clean_frame(train, min_chars=get_config()["data"]["text"]["min_chars"])
    report["after_clean_dedupe"] = len(train)
    protected = set(test["text"])
    proc = project_path("data", "processed")
    for task in ("sentiment", "toxicity"):
        protected |= set(pd.read_parquet(proc / f"{task}_test.parquet")["text"])
    from src.evaluation.regression import load_cases

    protected |= set(load_cases()["text"])
    before = len(train)
    train = train[~train["text"].isin(protected)].reset_index(drop=True)
    report["removed_overlap_with_eval_sets"] = before - len(train)
    frames = as_task_frames(train)
    for task, df in frames.items():
        validate_labeled_frame(
            df, f"drift/{task}", 3 if task == "sentiment" else 2, min_rows=50
        ).raise_if_failed()
    leak = check_no_split_leakage({"train": train, "test": test})
    if leak:
        raise RetrainingError(f"drift data leakage: {leak}")
    dist = train["toxicity"].value_counts(normalize=True).to_dict()
    report["toxicity_dist"] = {k: round(v, 3) for k, v in dist.items()}
    if min(dist.values()) < 0.05:
        raise RetrainingError(f"drift training data too imbalanced: {dist}")
    report["rows_out"] = len(train)
    return train, report


def dvc_commit_drift_data() -> str:
    """Record data/drift in dvc.lock. Returns a status string (never pretends success)."""
    exe = shutil.which("dvc")
    if exe is None:
        return "skipped: dvc not on PATH (dataset content hash still logged to MLflow)"
    r = subprocess.run(
        [exe, "commit", "-f", "drift_data"],
        cwd=project_path(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    return "committed" if r.returncode == 0 else f"failed: {r.stderr.strip()[:200]}"


def _pick_event(event_id: int | None, force: bool) -> DriftEvent | None:
    with session_scope() as s:
        if event_id is not None:
            ev = s.get(DriftEvent, event_id)
            if ev is None:
                raise RetrainingError(f"drift event {event_id} not found")
            return ev
        pending = repo.pending_drift_events(s)
        if pending:
            return pending[0]
    if force:
        return None
    raise RetrainingError(
        "No pending drift event. Retraining is only triggered by drift "
        "(use --force to retrain without one)."
    )


def _set_event_status(event_id: int | None, status: str) -> None:
    if event_id is None:
        return
    with session_scope() as s:
        ev = s.get(DriftEvent, event_id)
        if ev is not None:
            ev.retraining_status = status


def run_retraining(
    event_id: int | None = None, scenario: str = "normal", force: bool = False
) -> RetrainOutcome:
    from src.evaluation.promotion import run_promotion
    from src.pipeline.model_service import get_model_service
    from src.training import registry
    from src.training.train import train_model

    if scenario not in SCENARIOS:
        raise RetrainingError(f"unknown scenario {scenario}; choose {sorted(SCENARIOS)}")
    init_db()
    ev = _pick_event(event_id, force)
    ev_id = ev.id if ev else None
    prod = registry.get_version_by_alias("production")
    prod_before = str(prod.version) if prod else None
    log.info(
        "Retraining triggered by drift event %s (scenario=%s, production=v%s)",
        ev_id,
        scenario,
        prod_before,
    )
    _set_event_status(ev_id, "running")

    run_id = version = None
    try:
        train, test, drift_hash = load_drift_training_data()
        train, qreport = quality_checks(train, test)
        log.info("Drift data quality report: %s", qreport)
        dvc_status = dvc_commit_drift_data()
        log.info("DVC versioning of data/drift: %s", dvc_status)

        # reference reset: half historical validation data, half recent (drifted) data
        n = get_config()["drift"]["reference_size"]
        recent = (
            train["text"]
            .sample(n=min(n // 2, len(train)), random_state=get_config()["project"]["seed"])
            .tolist()
        )
        ref = reference_texts(n)[: n - len(recent)] + recent

        result = train_model(
            experiment=get_config()["mlflow"]["retraining_experiment"],
            run_name=f"retrain-drift-{ev_id}-{scenario}",
            extra_train=as_task_frames(train),
            extra_eval={"drift_test": as_task_frames(test)},
            tags={
                "trigger": "drift",
                "drift_event_id": str(ev_id),
                "scenario": scenario,
                "dvc_status": dvc_status[:250],
            },
            extra_dataset_version=f"drift:{drift_hash}",
            reference_texts=ref,
            **SCENARIOS[scenario],
        )
        run_id, version = result.run_id, result.version
        log.info("Candidate model v%s trained (run %s)", version, run_id)
        decision = run_promotion(version, extra_test_sets={"drift_test": as_task_frames(test)})
    except Exception as exc:
        _set_event_status(ev_id, "failed")
        with session_scope() as s:
            repo.save_model_event(
                s,
                model_version=str(version or "none"),
                production_version=prod_before,
                event_type="retraining",
                drift_event_id=ev_id,
                run_id=run_id,
                decision="FAILED",
                reason=f"{type(exc).__name__}: {exc}"[:1000],
                details={"scenario": scenario},
            )
        log.exception("Retraining failed")
        raise

    evidence = getattr(decision, "_evidence", {})
    prod_after = registry.get_version_by_alias("production")
    prod_after_v = str(prod_after.version) if prod_after else None
    with session_scope() as s:
        repo.save_model_event(
            s,
            model_version=str(version),
            production_version=prod_before,
            event_type="retraining",
            drift_event_id=ev_id,
            run_id=run_id,
            candidate_metrics=evidence.get("candidate_metrics"),
            production_metrics=evidence.get("production_metrics"),
            regression={
                "candidate": evidence.get("candidate_regression"),
                "production": evidence.get("production_regression"),
            },
            decision=decision.decision,
            reason=", ".join(decision.reasons),
            details={
                "scenario": scenario,
                "gates": [g.__dict__ for g in decision.gates],
                "drift_test": evidence.get("extra_evaluations", {}).get("drift_test"),
                "quality_report": qreport,
                "dvc": dvc_status,
                "production_after": prod_after_v,
            },
        )
    _set_event_status(ev_id, "promoted" if decision.decision == "PROMOTE" else "rejected")
    get_model_service().invalidate()
    if decision.decision == "PROMOTE":
        log.info(
            "Candidate v%s promoted; production v%s -> v%s", version, prod_before, prod_after_v
        )
    else:
        log.warning(
            "Candidate v%s REJECTED (%s); production remains v%s",
            version,
            ", ".join(decision.reasons),
            prod_after_v,
        )
    return RetrainOutcome(
        ev_id,
        version,
        prod_before,
        prod_after_v,
        decision.decision,
        decision.reasons,
        run_id,
        scenario,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--event-id", type=int)
    ap.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    ap.add_argument("--force", action="store_true", help="retrain even without a drift event")
    args = ap.parse_args()
    try:
        out = run_retraining(args.event_id, args.scenario, args.force)
    except RetrainingError as exc:
        log.error("%s", exc)
        return 1
    print(
        "\n".join(
            [
                "Candidate trained:          v" + str(out.candidate_version),
                "Candidate evaluated:        yes (test sets + drift test set)",
                "Regression tests executed:  yes",
                f"Promotion decision made:    {out.decision}  ({', '.join(out.reasons)})",
                f"Production model:           v{out.production_before} -> v{out.production_after}",
            ]
        )
    )
    print(json.dumps(out.to_dict(), indent=2))
    return 0 if out.decision == "PROMOTE" else 3


if __name__ == "__main__":
    sys.exit(main())
