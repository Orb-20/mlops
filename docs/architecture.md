# Architecture

## Lifecycle diagram

```mermaid
flowchart TD
    YT[YouTube Data API v3<br/>or DEMO dataset] --> ING[Ingestion adapter<br/>URL validation · pagination · retry]
    ING --> PRE[Normalisation & validation<br/>clean_text · schema · dedupe]
    PRE --> LANG[Language detection<br/>langdetect + Hinglish rule]
    LANG --> ENC[Shared DistilBERT encoder]
    ENC --> SH[Sentiment head<br/>neg / neu / pos]
    ENC --> TH[Toxicity head<br/>safe / toxic]
    ENC --> EMB[Mean-pooled embeddings]
    SH --> DB[(PostgreSQL / SQLite<br/>comments · predictions · embeddings)]
    TH --> DB
    EMB --> DB
    DB --> MON[Monitoring layer]
    REF[Reference window<br/>per model version] --> MON
    MON --> SIG[Signals: embedding MMD² · language TV ·<br/>prediction TV · vocabulary · Evidently]
    SIG --> DDE{DriftDecisionEngine}
    DDE -- stable / warning --> MON
    DDE -- drift --> EV[Drift event<br/>retraining pending]
    EV --> RT[Retraining pipeline<br/>validate · dedupe · quality · leakage checks]
    DVC[(DVC-versioned data<br/>processed + drift)] --> RT
    RT --> CAND[Candidate model<br/>MLflow run + registry alias 'candidate']
    CAND --> G1[Artifact gate] --> G2[Minimum-metric gate] --> G3[Regression suite gate] --> G4[Production comparison]
    G1 -- fail --> REJ[REJECT<br/>tag 'rejected' · production unchanged]
    G2 -- fail --> REJ
    G3 -- fail --> REJ
    G4 -- fail --> REJ
    G4 -- pass --> PROM[PROMOTE<br/>alias 'production' → candidate]
    PROM --> API[FastAPI<br/>reloads production alias]
    REJ --> API
    API --> UI[Streamlit dashboard<br/>Analyze · MLOps Monitoring · Registry]
    CI[GitHub Actions<br/>CI + model validation] -.-> G3
```

## Components

| Layer | Module | Responsibility |
|---|---|---|
| Config | `src/config/settings.py`, `configs/*.yaml` | All tunables in YAML, secrets/URLs in env vars |
| Data | `src/data/` | schemas, cleaning, validation, split, content-hash versioning |
| Ingestion | `src/ingestion/` | `SocialMediaAdapter` → `YouTubeAdapter` (official API), `DemoAdapter` |
| NLP | `src/nlp/` | language detection, multi-task model, batched inference, embeddings |
| Training | `src/training/` | baseline, trainer, `train.py`, MLflow tracking + registry |
| Evaluation | `src/evaluation/` | test-set evaluation, regression suite, `PromotionDecisionEngine` |
| Monitoring | `src/monitoring/` | reference windows, MMD / TV drift, vocabulary, Evidently, `DriftDecisionEngine`, drift simulation data |
| Pipelines | `src/pipeline/` | `analyze` (on demand), `monitor` (polling worker), `retrain`, `model_service` |
| Storage | `src/database/` | SQLAlchemy models + repository (Postgres in compose, SQLite locally) |
| Reporting | `src/reporting/` | analytics, executive summary, HTML/Markdown report |
| Serving | `src/api/` | FastAPI routes (thin adapters over the pipeline layer) |
| UI | `dashboard/` | Streamlit; talks to the API only |

## Key design decisions

- **One service layer.** CLI, API and dashboard all go through `src/pipeline/*`; the dashboard
  never loads a model itself.
- **MLflow is the source of truth for models.** Serving resolves the `production` alias; files on disk
  (`artifacts/model_cache/v<N>`) are only a cache of registry versions.
- **The reference distribution belongs to the model.** Each model version has its own reference window
  embedded by that model; after a drift-triggered retrain the model ships `reference_texts.json`
  (50% historical, 50% recent) so monitoring measures change relative to what it was trained on.
- **Drift never deploys.** A drift event only sets `retraining_status=pending`; the retraining pipeline
  must pass four gates before the `production` alias moves.
- **No queue infrastructure.** Monitoring is a polling thread inside the API process (configurable
  interval, default 300 s); new comments are detected via stored comment ids.
- **Embeddings in the relational DB** as float16 bytes (1.5 KB each) — enough for windowed drift
  statistics without a vector database.

## Traceability

`prediction → predictions.model_version → registry version (tags run_id, dataset_version, git_commit) →
MLflow run (params, metrics, artifacts, lineage tags) → dvc.lock / data manifest → git commit`.
`dataset_version` for retrained models is `<processed hash>+drift:<drift data hash>`.
