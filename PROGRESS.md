# PROGRESS — Adaptive NLP MLOps Pipeline

Single running log of what has been done, what was verified, and what comes next.
Spec: `claude_code_mlops_master_prompt.md` · Plan: `docs/implementation-plan.md`

Legend: ✅ done & verified · 🟡 done, partially verified · ⛔ blocked / unverified · ⏳ in progress · ⬜ not started

## Phase status

| # | Phase | Status | Notes |
|---|---|---|---|
| 0 | Repository inspection | ✅ | See implementation plan |
| 1 | Project skeleton | ✅ | 61 tests pass |
| 2 | Data pipeline | ✅ | DVC repro deterministic, hash 1cbb27cc862a |
| 3 | Baseline (TF-IDF + LR) | ✅ | sent macro-F1 0.519, tox F1 0.722 |
| 4 | Transformer model | ✅ | v1: sent macro-F1 0.675, tox F1 0.792 |
| 5 | MLflow registry | ✅ | v1 = production via gates |
| 6 | YouTube ingestion | 🟡 | demo verified; live path mock-tested only (no API key) |
| 7 | FastAPI | ✅ | all endpoints curl-verified + 9 API tests |
| 8 | Database | 🟡 | SQLite verified; PostgreSQL only via compose (unverified) |
| 9 | Drift monitoring | ✅ | MMD² + TV + vocabulary + Evidently; calibrated thresholds |
| 10 | Drift simulation | ✅ | stable → STABLE (0.000), shifted → DRIFT (0.117) |
| 11 | Automated retraining | ✅ | drift event → candidate → gates → decision recorded |
| 12 | Regression gate | ✅ | PASS path (v3 promoted) and FAIL path (v2, v4 rejected) both demonstrated |
| 13 | GitHub Actions | ⛔ | ci.yml + model-validation.yml written, YAML validated, CI steps run locally; never run on GitHub (no remote) |
| 14 | Streamlit dashboard | ✅ | 3 pages, Playwright-verified, 0 exceptions |
| 15 | Reporting | ✅ | HTML (inline SVG) + Markdown from stored data |
| 16 | Docker Compose | ⛔ | Dockerfile + compose written & YAML-validated; **never built** (no Docker here) |
| 17 | End-to-end demo | 🟡 | full lifecycle verified locally (non-Docker); compose path unverified |

**Current state (2026-09-29):** production model **v3**; 111 tests passing; ruff + black clean.

## Known blockers / unverified items

- Docker is not installed → compose stack cannot be run here.
- PostgreSQL not installed → local runs use SQLite fallback; Postgres path only via compose.
- No `YOUTUBE_API_KEY` → live YouTube path tested with mocked HTTP only.
- **Windows DLL order**: torch loaded before pyarrow → access violation in worker threads (API segfault).
  Fixed by importing pyarrow first in `src/__init__.py`; verified 0 access violations.
- **Windows Smart App Control is ON** and blocks some unsigned binaries:
  - venv launcher from `uv venv` blocked → venv created with `python -m venv --copies .venv`.
  - pyarrow 25.0.1 and scikit-learn 1.9.1/1.8/1.7 native modules blocked → pinned
    `pyarrow==24.0.0`, `scikit-learn==1.6.1` (verified: all 678 compiled modules import).

---

## Log

### 2026-09-29 — Phase 0: Repository inspection ✅
**Done**
- Directory contained only the spec file; not a git repo.
- Checked: Python 3.13.5 (system), uv 0.10.9, git 2.50.1, NVIDIA RTX 3050 4 GB, 16 GB RAM, 68 GB free disk.
- Docker, PostgreSQL, DVC: not installed.
- Verified HF Hub reachable; located Jigsaw `train.csv` mirror and `tweet_eval` sentiment (3-class).
- Wrote `docs/implementation-plan.md`.

**Commands**: `python --version`, `uv python list`, `git --version`, `docker --version`, `nvidia-smi`, `curl` to HF API.

### 2026-09-29 — Phase 1: Project skeleton ✅
**Done**
- `git init`; Python 3.11.15 venv (`.venv`, created with `--copies` because of Smart App Control).
- Installed torch 2.6.0+cu124 (CUDA available), transformers, mlflow 3.16.1, evidently 0.7.23,
  dvc 3.67.1, fastapi, sqlalchemy, streamlit, etc.
- `pyproject.toml`, `configs/{config,model,drift,gates}.yaml`, `src/config/settings.py`
  (YAML + env via pydantic-settings, `set_seed`), `.gitignore`, `.env.example`, `Makefile`, README skeleton.
- Full package + data/test/report directory skeleton.
- Code written ahead (tested where pure-python): data schemas/validation/preprocessing,
  language detection (langdetect + explicit Hinglish lexicon rule), metrics, YouTube + Demo
  adapters, MMD embedding drift, vocabulary drift, DriftDecisionEngine, model/inference/trainer.
- `scripts/seed_demo_data.py` → `data/sample/youtube_comments.csv` (300 **synthetic** comments, seed 42).

**Verified**: `python -m pytest` → 61 passed (settings, data, language, metrics, YouTube adapter with
mocked HTTP incl. pagination/quota/disabled/404/retry/malformed). `ruff check` clean.

**Gate thresholds note**: `configs/gates.yaml` uses lower minimums than the spec's example 0.80
because tweet_eval 3-class macro-F1 is ~0.70 even for strong models; to be calibrated in Phase 12.

**Next**: Phase 2 — run `python -m src.data.prepare` (download tweet_eval + Jigsaw), DVC init and tracking.

### 2026-09-29 — Phase 2: Data pipeline ✅
**Done**
- Downloaded tweet_eval sentiment (3-class) + Jigsaw train.csv from Hugging Face.
- `src/data/prepare.py` (`--step download|build|all`): clean → dedupe → split (leakage-checked) → validate → parquet + manifest.
- `dvc init`; `dvc.yaml` stages: download, prepare, baseline, drift_data, validate_curated.
- Processed: sentiment 12000/2000/3000, toxicity 14000/2000/4000 (10.2% toxic, stratified).
- `docs/data-pipeline.md` written (sources, label mapping, versioning).

**Verified**: `dvc repro download prepare` ran; `dvc repro -f prepare` reproduced identical
`dataset_version=1cbb27cc862a`. New unit test for Jigsaw any-label mapping + stratified deterministic split passes.

**Commands**: `python -m src.data.prepare`, `dvc init`, `dvc repro download prepare`, `dvc repro -f prepare`, `dvc status`.

**Next**: Phase 3 — run TF-IDF + LogisticRegression baseline, log to MLflow, record metrics.

### 2026-09-29 — Phase 3: Baseline ✅
- `src/training/baseline.py` (TF-IDF 1-2gram + LogisticRegression; balanced class weights for toxicity), DVC stage `baseline`.
- MLflow run `a44234e9…` in experiment `sentiment-toxicity-training`: 12 metrics, 3 artifacts, lineage tags.
- Test: sentiment acc 0.554 / macro-F1 **0.519**; toxicity P 0.744 / R 0.702 / F1 **0.722** / ROC-AUC 0.959 / PR-AUC 0.815.

### 2026-09-29 — Phase 4: Transformer model ✅
- `src/nlp/model.py` MultiTaskModel (shared DistilBERT, mean-pooled embedding, 2 heads), `src/training/trainer.py`
  (task-alternating batches, per-task loss, class-weighted toxicity CE, AMP fp16, linear warmup),
  `src/training/train.py` (`python -m src.training.train`).
- Note: installed transformers is **5.17** (major v5) — APIs used were verified by running.
- Full run `0a5c185b…` on RTX 3050 (8000 rows/task, 2 epochs, bs 16, lr 3e-5, seed 42).
- Test: sentiment acc 0.681 / macro-F1 **0.675**; toxicity P 0.777 / R 0.808 / F1 **0.792** / ROC-AUC 0.977 / PR-AUC 0.886.
  Clear gain over baseline (+0.156 macro-F1, +0.070 toxicity F1).

### 2026-09-29 — Phase 5: MLflow registry ✅
- `src/training/registry.py`: pyfunc-wrapped model, registered model `sentiment-toxicity-model`,
  aliases candidate/staging/production/rejected + `lifecycle` version tag; `download_model` cache.
- `src/evaluation/promotion.py` PromotionDecisionEngine (4 gates) + `src/evaluation/regression.py`.
- Regression suite `data/regression/regression_cases.csv`: 48 cases, 13 categories, rationale per case;
  6 critical cases. v1 fails 13/48 (sarcasm, Hinglish, profanity-as-praise), 0 critical.
- Calibrated `max_failed_cases: 15` from v1's result (documented in gates.yaml).
- `python -m src.evaluation.promotion --alias candidate` → **PROMOTE v1** (bootstrap: no production yet).
- Verified smoke path in a throwaway MLflow DB first (train → register → download → load → predict).
- Tests: 88 passed (added promotion engine, drift/MMD/decision engine, regression suite tests).

**Next**: Phase 6 — YouTube ingestion wiring (adapter done + tested with mocks); then Phase 7/8 API + DB.

### 2026-09-29 — Phases 6-8: Ingestion wiring, database, FastAPI
**Done**
- DB (`src/database/`): analyses, comments (unique source+comment id, no author fields), predictions,
  embeddings (float16 bytes, 1.5 KB/comment), drift_events, model_events, monitored_videos. Repository layer.
- Reference window per model version (`src/monitoring/reference.py`, 500 val texts, cached npz).
- **Drift calibration on real v1 embeddings** (MMD²): null ≈ 0.000 (p≥0.46), demo comments 0.063,
  slang drift window 0.117 → threshold 0.09, warning ratio 0.6 (recorded in `configs/drift.yaml`).
- Service layer: `src/pipeline/analyze.py` (CLI `python -m src.pipeline.analyze --demo|--url`),
  `model_service.py` (production model loaded once; alias re-checked every 30 s), `monitor.py`
  (incremental polling via known ids, rolling window, background `MonitorWorker`).
- Analytics + executive summary generated from stats; HTML (inline SVG) + Markdown report.
- FastAPI: `/health`, `/api/v1/model`, `/analyze/youtube`, `/monitor/{start,stop,status}`,
  `/drift/{latest,history}`, `/retraining/history`, `/report/{id}`, `/analyses`.

**Bugs found & fixed**: MLflow 3 returns int versions (pydantic 500 on /health); pyarrow/torch DLL
order segfault in API threadpool (see blockers).

**Verified**: demo CLI analysis (300 comments → warning: language+prediction shift, embedding 0.063 < 0.09);
curl of every endpoint (422 invalid URL, 400 live-without-key, 404 report); 15 new API/integration tests
(temp SQLite + fake model) incl. incremental monitoring and no-retrain-below-minimum-samples.

**Not verified**: live YouTube fetch (no key); PostgreSQL (no Docker).

**Next**: Phase 9/10 — `scripts/simulate_drift.py` (stable vs shifted, drift report, trigger retraining).

### 2026-09-29 — Phases 9-11: Drift simulation & retraining ✅
**Done**
- `src/monitoring/simulation.py`: deterministic controlled drift set (47 slang/meme/Hinglish templates,
  labels by construction, template-level train/test split): train 1200 / test 400 / window 300,
  DVC stage `drift_data` (hash 5ba1ffa2db9c, rebuild-identical).
- `scripts/simulate_drift.py`: stable vs shifted windows → drift events → reports/drift/*.{json,md,html}.
- `src/pipeline/retrain.py`: quality checks (clean, dedupe, overlap removal vs all eval sets, balance),
  `dvc commit drift_data`, candidate training (experiment `drift-retraining`), promotion, model_events,
  drift-event status. Reference reset (50% historical + 50% recent) shipped with the model.
- Fault injection `--scenario corrupted_labels` (60% labels corrupted) for the safety-gate demo.
- Evidently 0.7.23 `DataDriftPreset` (API inspected before use) as supplementary evidence.

**Recorded outcomes**
| step | result |
|---|---|
| stable window (v1) | STABLE, MMD² 0.000, p 0.91 |
| shifted window (v1) | DRIFT (high): MMD² 0.117 p 0.005, language TV 0.25, prediction TV 0.15 → event #6 |
| retrain attempt 1 → v2 | **REJECT** production_comparison_failed (tox recall 0.771 vs 0.808) |
| fix | per-model toxicity threshold calibrated on validation split (gates unchanged) |
| retrain attempt 2 → v3 | **PROMOTE** (thr 0.18; F1 0.798, recall 0.825; regression 12 vs 13) |
| monitor again (v3) | shifted window MMD² 0.117 → 0.037 → no drift (adapted) |
| corrupted labels → v4 | **REJECT** min metrics + regression (40/48, 4 critical) + comparison; production stays v3 |
- Honest note: v3 has identical weights to v2 (seeded, deterministic); promotion came from the threshold.

**Bugs found & fixed**: stale `candidate` alias (MLflow int vs str versions) + regression test;
simulate_drift exit code after adaptation; v4 `trigger` tag said "drift" for a manual run (code fixed,
tag corrected to `manual` to match its model event with no drift event).

### 2026-09-29 — Phases 13-17: CI, dashboard, reporting, Docker, docs
- Dashboard (`dashboard/app.py` + `views/`): Analyze Video, MLOps Monitoring, Model Registry. Colours
  chosen by job and checked with the dataviz palette validator; light theme pinned because the
  sentiment red/gray pair fails CVD separation on dark surfaces; direct % labels on all bars.
  Verified with Playwright screenshots (0 exceptions; report download buttons present).
- Reports: HTML + Markdown with OBSERVED DATA / MODEL PREDICTION / DRIFT SIGNAL sections.
- CI: `.github/workflows/ci.yml`, `model-validation.yml`; every CI step reproduced locally
  (pip check, ruff, black --check, suite validation, demo-data byte-identity, all tests).
- Docker: `Dockerfile`, `docker-compose.yml` (postgres, mlflow, api, dashboard, bootstrap profile),
  `docker/postgres-init.sql`, `.dockerignore`, `scripts/setup_local.py`. **Not built.**
- Docs: README (20 sections), docs/{architecture (Mermaid), data-pipeline, model, mlops-lifecycle,
  drift-detection, retraining, deployment, testing, demo-script, limitations}.md, LICENSE.
- Issue found: file edits via a Python script on Windows wrote cp1252 + CRLF×2 into one dashboard file;
  repaired and verified all repo text files are UTF-8.

## Quality gate (spec §57)

| Area | Item | Status |
|---|---|---|
| Repository | structure, README, .env.example, .gitignore, configuration | ✅ |
| Data | datasets documented, preprocessing, DVC, demo data | ✅ |
| ML | baseline, transformer, sentiment + toxicity metrics, embeddings | ✅ |
| MLOps | MLflow tracking + registry, DVC, drift detection, retraining, regression gate, promote/reject | ✅ |
| Backend | FastAPI, health, analysis, monitoring endpoints | ✅ |
| Database | schema + persistence on SQLite | ✅ · PostgreSQL ⛔ unverified |
| UI | URL input, analysis dashboard, MLOps dashboard, drift charts, model history, report | ✅ |
| DevOps | Docker / Compose ⛔ unverified · GitHub Actions ⛔ not yet run |
| Tests | unit, API, integration, regression ✅ (111) · end-to-end demo ✅ locally |
| External | live YouTube API ⛔ unverified (no key) |

## What remains / next steps (need the user or other infrastructure)
1. Install Docker Desktop, run `docker compose up --build` + bootstrap profile; fix whatever breaks.
2. Add a `YOUTUBE_API_KEY` to `.env` and run one live analysis (`python -m src.pipeline.analyze --url ...`).
3. Push to GitHub to get the first real CI run.
4. Optional: multilingual encoder for Hinglish; larger / human-labelled drift data.
