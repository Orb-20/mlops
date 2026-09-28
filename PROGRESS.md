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
| 3 | Baseline (TF-IDF + LR) | ⏳ | |
| 4 | Transformer model | ⬜ | |
| 5 | MLflow registry | ⬜ | |
| 6 | YouTube ingestion | ⬜ | |
| 7 | FastAPI | ⬜ | |
| 8 | Database | ⬜ | |
| 9 | Drift monitoring | ⬜ | |
| 10 | Drift simulation | ⬜ | |
| 11 | Automated retraining | ⬜ | |
| 12 | Regression gate | ⬜ | |
| 13 | GitHub Actions | ⬜ | |
| 14 | Streamlit dashboard | ⬜ | |
| 15 | Reporting | ⬜ | |
| 16 | Docker Compose | ⬜ | Docker not installed on this machine |
| 17 | End-to-end demo | ⬜ | |

## Known blockers / unverified items

- Docker is not installed → compose stack cannot be run here.
- PostgreSQL not installed → local runs use SQLite fallback; Postgres path only via compose.
- No `YOUTUBE_API_KEY` → live YouTube path tested with mocked HTTP only.
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
