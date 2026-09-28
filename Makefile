# Convenience targets (Linux/macOS/CI). On Windows run the same commands with .venv\Scripts\python.
PY ?= python

.PHONY: install data baseline train test lint format api dashboard simulate retrain demo up

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/seed_demo_data.py
	$(PY) -m src.data.prepare

baseline:
	$(PY) -m src.training.baseline

train:
	$(PY) -m src.training.train

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

lint:
	ruff check src tests scripts dashboard
	black --check src tests scripts dashboard

format:
	ruff check --fix src tests scripts dashboard
	black src tests scripts dashboard

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

simulate:
	$(PY) scripts/simulate_drift.py

retrain:
	$(PY) -m src.pipeline.retrain

up:
	docker compose up --build
