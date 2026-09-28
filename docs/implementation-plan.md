# Implementation Plan — Adaptive NLP MLOps Pipeline

Source spec: `claude_code_mlops_master_prompt.md`. Live progress: `PROGRESS.md`.

## Phase 0 findings (2026-09-29)

| Item | Finding | Consequence |
|---|---|---|
| Repository | Only the spec file; not a git repo | Fresh project, built in place; `git init` |
| Python | System 3.13.5; `uv` 0.10.9 available | Project venv on **Python 3.11** via `uv` (spec stack) |
| Docker | **Not installed** | Dockerfile + compose are written but **cannot be verified on this machine** until Docker Desktop is installed |
| PostgreSQL | Not installed | SQLAlchemy layer targets Postgres (compose); local non-Docker runs use a **SQLite fallback** via `DATABASE_URL` |
| GPU | RTX 3050 Laptop, 4 GB VRAM | DistilBERT fine-tuning on subsets, small batches, fp16 where possible; CPU fallback kept |
| RAM / disk | 16 GB / 68 GB free | Fine for subset training |
| DVC | Not installed | Installed into venv |
| Network | HF Hub + PyPI reachable | Public datasets downloadable without Kaggle credentials |
| Secrets | No `YOUTUBE_API_KEY` in env | Demo-data path is the default; live path tested with mocked HTTP only until a key is supplied |

## Dataset decisions

- **Sentiment:** `cardiffnlp/tweet_eval` (config `sentiment`, SemEval-2017 Task 4). Native 3 classes
  (0=negative, 1=neutral, 2=positive) — no invented label mapping. Chosen over Sentiment140 (2 classes only)
  and Amazon reviews (star→class mapping would have to be invented). Social-media domain matches YouTube comments.
- **Toxicity:** Jigsaw Toxic Comment Classification `train.csv` (HF mirror
  `thesofakillers/jigsaw-toxic-comment-classification-challenge`). Binary label
  `toxic_any = max(toxic, severe_toxic, obscene, threat, insult, identity_hate)`; raw columns kept.
- Subsets are sampled with a fixed seed so training fits the hardware; sizes live in `configs/`.

## Architecture decisions

- Multi-task DistilBERT (shared encoder, sentiment head 3-way, toxicity head 2-way), task-alternating
  batches, `L = L_sent + λ·L_tox` (spec §15 preferred option).
- Embedding = mean-pooled last hidden state of the *production* encoder.
- Embedding drift = **MMD² with RBF kernel + permutation test** (custom, numpy). Evidently used for
  tabular/categorical reports (language and prediction distributions) after inspecting the installed API.
- MLflow: local server in compose; file/sqlite store locally. Registry name `sentiment-toxicity-model`,
  aliases `candidate`, `production`, and tag-based `rejected` status.
- Model lookup via MLflow alias, never filenames.

## Phase order

Follow spec §54 phases 0 → 17 in order. Each phase: inspect → implement → run → test → document → update `PROGRESS.md`.
