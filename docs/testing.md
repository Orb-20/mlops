# Testing

```bash
python -m pytest -p no:faulthandler        # 111 tests, ~40 s, no GPU / network / API key needed
```

(`-p no:faulthandler` only silences a harmless SEH trace printed on Windows while pyarrow loads.)

| Suite | Path | What it covers |
|---|---|---|
| Unit | `tests/unit/` | settings; text cleaning; schema + data validation; split leakage; Jigsaw label mapping; language detection (en/hi/hinglish/es); metrics + threshold calibration; YouTube URL parsing, pagination, replies, known-id skipping, malformed items, quota / disabled / 404 / invalid key, retry + backoff, network failure; MMD statistic (null vs shift, determinism); TV distance; DriftDecisionEngine (insufficient data, stable, embedding-only drift, auxiliary-only = warning, high severity); vocabulary report; PromotionDecisionEngine (bootstrap, better candidate, margins, recall-drop rejection, regression, artifact short-circuit); Evidently integration |
| Regression | `tests/regression/` | curated suite validity (30–60 cases, required categories, label vocab, rationale), suite runner, critical-case override |
| Integration | `tests/integration/` | pipeline → DB with temp SQLite: persistence, no duplicate comments, incremental monitoring, rolling window, no retraining below minimum samples, report generation, live-without-key; monitor worker scheduling + error capture; MLflow registry alias lifecycle on a real temp registry |
| API | `tests/api/` | FastAPI TestClient: analyze (demo) + report HTML/MD, 422 invalid URL, 400 live without key, input validation, drift / retraining history, monitor start/stop/status, model endpoint, 404 report, degraded health |

Test doubles (`tests/fixtures/fakes.py`, `tests/conftest.py`) replace only the model at the service
boundary (rule-based predictor with the real `.run()` contract) and point DB/MLflow at temp files.
They are never used in the application path.

## Verified end-to-end runs (real model, not mocks)

| Check | Command | Result |
|---|---|---|
| Data pipeline determinism | `dvc repro -f prepare` | same `dataset_version` 1cbb27cc862a |
| Baseline | `dvc repro baseline` | MLflow run with metrics + artifacts |
| Training | `python -m src.training.train` | v1, 218 s on RTX 3050 |
| Demo analysis | `python -m src.pipeline.analyze --demo` | 300 comments, warning |
| API | curl on every endpoint | as expected (see PROGRESS.md) |
| Drift simulation | `python scripts/simulate_drift.py --no-retrain` | stable → STABLE, shifted → DRIFT |
| Retraining pass/fail | `python -m src.pipeline.retrain ...` | v2 REJECT, v3 PROMOTE, v4 REJECT |
| Dashboard | Playwright screenshots of all 3 pages | render with 0 exceptions |

## CI

`.github/workflows/ci.yml` runs pip check, ruff, black --check, curated-data validation, demo-data
byte-reproducibility, unit + regression + API + integration tests on CPU. **Not yet executed on
GitHub** (no remote configured) — first push will be its first run.
`model-validation.yml` trains a small CPU candidate and runs the promotion engine in dry-run mode.
