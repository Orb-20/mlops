# Adaptive NLP MLOps Pipeline

Social-media **sentiment & toxicity monitoring** with **embedding-drift detection**, drift-triggered
**retraining**, and **gated model promotion**. It's an MLOps project first: the core is the
reproducible lifecycle

```
new data → prediction → embedding → drift monitor → drift event → retrain → evaluate
        → regression gate → production comparison → PROMOTE / REJECT → monitor again
```

and the YouTube analysis UI is the demonstration layer. Live status: [`PROGRESS.md`](PROGRESS.md).
What actually happened in the recorded run: [`docs/mlops-lifecycle.md`](docs/mlops-lifecycle.md).

## 1. Overview

- Multi-task DistilBERT (shared encoder → sentiment 3-way + toxicity 2-way), trained on tweet_eval
  (sentiment) and Jigsaw (toxicity); TF-IDF + LR baseline for comparison.
- YouTube Data API v3 ingestion with a deterministic **DEMO DATA** fallback (no key needed).
- Language detection (English / Hindi / Hinglish / Spanish / other) as a monitoring signal.
- Primary drift signal: **MMD² on model embeddings** with permutation test; auxiliary language,
  prediction and vocabulary signals; Evidently report as supporting evidence.
- MLflow tracking + registry (aliases `candidate` / `production` / `rejected`), DVC data versioning.
- Four promotion gates: artifact validity, minimum metrics, 48-case regression suite, production comparison.
- FastAPI service, Streamlit dashboard, HTML/Markdown reports, GitHub Actions, Docker Compose.

Recorded results: baseline macro-F1 0.519 / tox F1 0.722 → DistilBERT v1 0.675 / 0.792.
Controlled drift detected (MMD² 0.117 vs threshold 0.09); retraining produced v2 (**rejected**,
recall regression), v3 (**promoted**), and a corrupted-label v4 (**rejected**, production unchanged).

## 2. Architecture

See [`docs/architecture.md`](docs/architecture.md) (Mermaid diagram of the full lifecycle).

```
YouTube / DEMO → ingestion → cleaning → language → DistilBERT ─┬─ sentiment head
                                                               ├─ toxicity head
                                                               └─ embeddings → DB → drift engine
drift → retraining (DVC data) → MLflow candidate → gates → production alias → FastAPI → Streamlit
```

## 3. Technology stack

Python 3.11 · PyTorch 2.6 · Transformers 5 · scikit-learn · pandas · MLflow 3 · DVC 3 · Evidently 0.7 ·
FastAPI · SQLAlchemy 2 (PostgreSQL / SQLite) · Streamlit + Plotly · pytest · ruff · black ·
GitHub Actions · Docker Compose.

## 4. Setup

```bash
python3.11 -m venv .venv          # Windows with Smart App Control: python -m venv --copies .venv
# GPU (CUDA 12.4):  .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU:              .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[dev]"      # Linux/macOS: .venv/bin/python
cp .env.example .env
python scripts/setup_local.py     # demo data → datasets → train v1 → gated promotion (≈5 min on a 4 GB GPU)
```

## 5. Environment variables

`YOUTUBE_API_KEY` (optional; empty = demo mode), `DATABASE_URL` (default SQLite `local.db`),
`MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`), `ENVIRONMENT`, `API_URL`. Never commit `.env`.
Details: [`docs/deployment.md`](docs/deployment.md).

## 6. Dataset preparation

```bash
python -m src.data.prepare        # or: dvc repro prepare
```
tweet_eval sentiment (native 3 classes) and Jigsaw train.csv (`toxic` = any of the 6 labels),
cleaned, deduplicated, leakage-checked, split with seed 42. [`docs/data-pipeline.md`](docs/data-pipeline.md)

## 7. Training

```bash
python -m src.training.baseline                 # TF-IDF + LR (dvc repro baseline)
python -m src.training.train [--epochs 2 --max-train 8000]
python -m src.evaluation.promotion --alias candidate
```
[`docs/model.md`](docs/model.md) — architecture, masked multi-task loss, metrics table.

## 8. MLflow

Local: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000.
Experiments `sentiment-toxicity-training`, `drift-retraining`; registered model
`sentiment-toxicity-model`. Every run is tagged with `dataset_version`, `git_commit`, library versions.

## 9. DVC

`dvc repro` runs `download → prepare → baseline`, `drift_data`, `validate_curated`
(curated sets are git-tracked and hashed into `dvc.lock`). No remote is configured (₹0 setup);
add one with `dvc remote add` to share data.

## 10. YouTube setup

Create an API key for *YouTube Data API v3* in Google Cloud Console, put it in `.env` as
`YOUTUBE_API_KEY`. Only comment id, parent id, text and publish time are stored — no author identity.
Without a key everything runs on the synthetic demo dataset and is labelled **DEMO DATA**.

## 11. API

`uvicorn src.api.main:app --port 8000` → OpenAPI at `/docs`.

| Method | Path |
|---|---|
| GET | `/health` |
| GET | `/api/v1/model` |
| POST | `/api/v1/analyze/youtube` `{"url": "...", "mode": "auto\|live\|demo", "max_comments": 300}` |
| POST | `/api/v1/monitor/start`, `/api/v1/monitor/stop` · GET `/api/v1/monitor/status` |
| GET | `/api/v1/drift/latest`, `/api/v1/drift/history` |
| GET | `/api/v1/retraining/history` |
| GET | `/api/v1/report/{id}?format=html\|md` · `/api/v1/analyses`, `/api/v1/analyses/{id}` |

CLI equivalents: `python -m src.pipeline.analyze --demo | --url URL`, `python -m src.pipeline.monitor --demo --once`.

## 12. Dashboard

`streamlit run dashboard/app.py` → http://localhost:8501. Pages: **Analyze Video**,
**MLOps Monitoring** (lineage, drift history, monitored videos), **Model Registry** (versions,
retraining events with gates and metric comparisons).

## 13. Drift simulation

```bash
python scripts/simulate_drift.py --no-retrain   # stable window → STABLE, shifted window → DRIFT DETECTED
python scripts/simulate_drift.py                # … and trigger retraining
```
Writes `reports/drift/drift_report_*.{json,md}` and `evidently_shifted_*.html`.
[`docs/drift-detection.md`](docs/drift-detection.md) — statistics and threshold calibration.

## 14. Retraining

```bash
python -m src.pipeline.retrain                                   # oldest pending drift event
python -m src.pipeline.retrain --force --scenario corrupted_labels   # must be REJECTED
```
[`docs/retraining.md`](docs/retraining.md) — data policy, gates, regression suite.

## 15. CI/CD

`.github/workflows/ci.yml` (lint, format, data validation, all tests on CPU) and
`model-validation.yml` (train small candidate → evaluate → regression → promotion decision).
Not yet run on GitHub — see limitations.

## 16. Testing

`python -m pytest -p no:faulthandler` → 111 tests (unit, regression suite, integration, API), no GPU,
network or key required. [`docs/testing.md`](docs/testing.md)

## 17. Docker

`docker compose up --build`, then `docker compose --profile bootstrap run --rm bootstrap`.
Ports: dashboard 8501, API 8000, MLflow 5000, Postgres 5432. **Unverified** — Docker was not
available on the development machine.

## 18. Demo instructions

[`docs/demo-script.md`](docs/demo-script.md): analysis → drift → retraining → safety gate.

## 19. Limitations

[`docs/limitations.md`](docs/limitations.md). Highlights: Docker/Postgres/live YouTube/GitHub Actions
unverified here; drift and demo data are synthetic; English-centric encoder; thresholds calibrated
for this data; live comments are unlabelled and never used as training labels.

## 20. Future work

Queue-based ingestion, multilingual encoder for Hinglish, annotation UI + active learning,
X / Instagram adapters, orchestration (Airflow), GPU inference service, cloud artifact storage.
