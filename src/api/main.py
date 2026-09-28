"""FastAPI application. Run: uvicorn src.api.main:app --port 8000"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.core import router
from src.database.connection import DatabaseUnavailable, init_db
from src.pipeline.monitor import MonitorWorker

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("api")
worker = MonitorWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except DatabaseUnavailable as exc:
        log.error("%s — API starts in degraded mode", exc)
    if os.getenv("DISABLE_MONITOR_WORKER", "").lower() not in ("1", "true"):
        worker.start()
    yield
    worker.stop()


app = FastAPI(title="Adaptive NLP MLOps API", version="0.1.0", lifespan=lifespan,
              description="Sentiment & toxicity monitoring with embedding-drift detection and "
                          "gated retraining.")
app.include_router(router)
