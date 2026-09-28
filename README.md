# Adaptive NLP MLOps Pipeline

Adaptive social-media sentiment & toxicity monitoring with automated embedding-drift detection,
gated retraining, and safe model promotion.

> Work in progress — see `PROGRESS.md` for the live phase-by-phase status and
> `docs/implementation-plan.md` for design decisions. This README is completed in the final phase.

## Quick start (local, no Docker)

```bash
uv venv --python 3.11 .venv            # or: python3.11 -m venv .venv
# Windows with Smart App Control: python3.11 -m venv --copies .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # (Linux/macOS: .venv/bin/python)
cp .env.example .env
python scripts/seed_demo_data.py
python -m pytest
```
