# One image for api / dashboard / mlflow / bootstrap (different commands in compose).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    HF_HOME=/app/.cache/huggingface

# CPU PyTorch by default (no GPU assumed). Override with --build-arg TORCH_INDEX=... for CUDA.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/__init__.py src/__init__.py
RUN pip install --upgrade pip \
    && pip install torch --index-url ${TORCH_INDEX} \
    && pip install -e .

COPY configs configs
COPY src src
COPY dashboard dashboard
COPY scripts scripts
COPY .streamlit .streamlit
COPY dvc.yaml dvc.lock ./
COPY data/sample data/sample
COPY data/regression data/regression

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/artifacts /app/data /app/reports /app/.cache /mlartifacts \
    && chown -R app:app /app /mlartifacts
USER app

EXPOSE 8000 8501 5000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
