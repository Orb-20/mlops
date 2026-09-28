# MLOps Lifecycle — what actually happened

This is the recorded history of the local demonstration run (2026-09-29). All numbers come from
MLflow runs and the `model_events` table; none are illustrative.

## Timeline

| Step | Event | Result |
|---|---|---|
| 1 | TF-IDF + LR baseline (run `a44234e9`) | sentiment macro-F1 0.519 · toxicity F1 0.722 |
| 2 | Train multi-task DistilBERT v1 (run `0a5c185b`, 218 s on RTX 3050) | macro-F1 0.675 · tox F1 0.792 · recall 0.808 |
| 3 | Promotion of v1 (bootstrap, no production yet) | **PROMOTE** — all gates; comparison skipped (no production) |
| 4 | Demo analysis (300 synthetic comments) | drift status **warning** (language + prediction shift; embedding MMD² 0.063 < 0.09) — no retraining |
| 5 | `simulate_drift.py`: held-out stable window | **stable** (MMD² 0.000, p = 0.91) |
| 6 | `simulate_drift.py`: slang / Hinglish window | **DRIFT DETECTED** (MMD² 0.117, p = 0.005; language TV 0.25; prediction TV 0.15) → drift event #6, retraining pending |
| 7 | Retrain attempt 1 → candidate v2 | **REJECT** `production_comparison_failed` — toxicity recall 0.771 vs 0.808 (drop 0.037 > 0.03 allowed) |
| 8 | Recipe change: validation-calibrated toxicity threshold (see below) | committed as `64f280e` |
| 9 | Retrain attempt 2 → candidate v3 | **PROMOTE** — recall 0.825, F1 0.798, regression 12 vs 13 failures; production v1 → v3 |
| 10 | Monitor again (same slang window vs v3 reference) | embedding MMD² 0.117 → **0.037**, no drift: model + reference adapted |
| 11 | Safety-gate demo: `--scenario corrupted_labels` → v4 | **REJECT** (minimum metrics, regression incl. 4 critical cases, production comparison); production stays v3 |

## Why v2 was rejected and what changed

v2 was trained on the processed data + 1,199 labelled drift examples. It clearly adapted to the drift
(drift-test sentiment macro-F1 0.389 → 0.577; toxicity F1 0.295 → 0.509) but, at the fixed 0.5
threshold, traded toxicity recall for precision on the standard test set (P 0.777 → 0.813,
R 0.808 → 0.771). The production-comparison gate forbids a recall drop larger than 0.03, so it was
rejected and v1 kept serving. **The gate thresholds were not changed.**

The recipe change: every newly trained model stores an F1-optimal toxicity threshold chosen on the
**validation** split (never the test split) in its `metadata.json`. v1 predates this and keeps 0.5.

**Honest reading of v3:** training is fully seeded, and v3's weights are identical to v2's (same
ROC-AUC 0.973 / PR-AUC 0.882 and sentiment scores). The promotion is entirely due to the calibrated
threshold (0.18): recall 0.771 → 0.825 at precision 0.813 → 0.772. It is not a better-trained encoder.

## Registry state after the run

| version | lifecycle | trigger | notes |
|---|---|---|---|
| v1 | archived | initial training | former production |
| v2 | rejected | drift event #6 | recall regression |
| v3 | **production** | drift event #6 (attempt 2) | calibrated threshold 0.18 |
| v4 | rejected | manual, `corrupted_labels` fault injection | 60% of training labels corrupted |

Note: v2 and v3 carry `git_commit` tags ending in `-dirty` — files were edited while those runs were
in progress. The lineage records that honestly rather than hiding it.

## Bugs the lifecycle surfaced (fixed)

- MLflow 3 returns model versions as `int`; string comparisons made the `candidate` alias survive
  promotion/rejection. Fixed with string normalisation + `tests/integration/test_registry_lifecycle.py`.
- `/health` returned 500 for the same reason (pydantic expected `str`).
