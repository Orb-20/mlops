# Deployment

> **Verification status:** Docker was not available on the development machine, so the Dockerfile
> and `docker-compose.yml` have **not been built or run**. The local (non-Docker) path below is
> verified end to end. Treat the compose stack as unverified until `docker compose up --build` has
> been run once.

## Local, no Docker (verified — Windows 11, Python 3.11)

```bash
python3.11 -m venv .venv            # Windows + Smart App Control: add --copies
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124  # or /cpu
.venv/Scripts/python -m pip install -e ".[dev]"
cp .env.example .env                 # SQLite + file MLflow by default
python scripts/setup_local.py        # demo data, dataset prep, train v1, gated promotion
uvicorn src.api.main:app --port 8000
streamlit run dashboard/app.py       # http://localhost:8501
```

## Docker Compose (unverified)

```bash
cp .env.example .env
docker compose up --build
docker compose --profile bootstrap run --rm bootstrap   # first model (CPU, 4000 rows/task, 1 epoch)
```

| Service | Port | Notes |
|---|---|---|
| postgres | 5432 | DBs `mlops` (app) and `mlflow` (tracking) — `docker/postgres-init.sql` |
| mlflow | 5000 | Postgres backend, artifacts proxied (`--serve-artifacts`) to volume `mlartifacts` |
| api | 8000 | `/docs` for OpenAPI; waits for healthy postgres + mlflow |
| dashboard | 8501 | talks to `http://api:8000` |
| bootstrap | — | profile `bootstrap`; runs `scripts/setup_local.py` once |

All services share one image (CPU PyTorch by default; `--build-arg TORCH_INDEX=...` for CUDA). Health
checks use Python `urllib` (no curl in the slim image). Secrets come only from `.env`
(`YOUTUBE_API_KEY`); nothing is baked into the image (`.dockerignore` excludes `.env`, DBs, data, artifacts).

A CPU bootstrap with the defaults may not pass the minimum-metric gate; the script then exits 3 and
tells you to increase `BOOTSTRAP_MAX_TRAIN` / `BOOTSTRAP_EPOCHS`. Gates are never bypassed.

## Optional free/low-cost hosting

The API + dashboard fit a single small container (CPU inference of DistilBERT is fine for a few hundred
comments per request). Candidates such as Hugging Face Spaces (Docker SDK) could host it; free-tier
limits change, so verify at deployment time. Point `DATABASE_URL` at any managed Postgres and
`MLFLOW_TRACKING_URI` at a reachable tracking server. The local architecture remains the source of truth.

## Environment variables

| Var | Default (local) | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | empty → demo mode | official YouTube Data API v3 key |
| `DATABASE_URL` | `sqlite:///./local.db` | Postgres in compose |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | `http://mlflow:5000` in compose |
| `ENVIRONMENT` | `local` | shown in `/health` |
| `API_URL` | `http://localhost:8000` | dashboard → API |
| `DISABLE_MONITOR_WORKER` | unset | `1` disables the polling thread (tests) |
