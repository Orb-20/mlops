"""Pydantic request/response models for the public API. No secrets, no raw vectors."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    mlflow: bool
    production_model: str | None
    youtube_api_configured: bool
    environment: str


class ModelResponse(BaseModel):
    registered_model: str
    version: str
    status: str
    run_id: str | None
    dataset_version: str | None
    git_commit: str | None
    created: datetime | None
    versions: list[dict[str, Any]] = []


class AnalyzeRequest(BaseModel):
    url: str | None = Field(None, description="YouTube video URL (ignored in demo mode)")
    mode: Literal["auto", "live", "demo"] = "auto"
    max_comments: int = Field(300, ge=1, le=5000)


class DriftSummary(BaseModel):
    status: str
    drift_detected: bool
    severity: str
    reasons: list[str]
    scores: dict[str, float]
    thresholds: dict[str, float]
    sample_count: int
    event_id: int | None = None
    vocabulary: dict[str, Any] | None = None


class AnalyzeResponse(BaseModel):
    analysis_id: int
    source: str
    content_id: str
    data_mode: Literal["live", "demo"]
    data_label: str
    comment_count: int
    model_version: str
    sentiment: dict[str, float]
    toxicity: dict[str, float]
    languages: dict[str, float]
    summary: dict[str, Any]
    drift: DriftSummary
    report_url: str


class MonitorRequest(BaseModel):
    url: str | None = None
    mode: Literal["auto", "live", "demo"] = "auto"
    poll_interval_seconds: int | None = Field(None, ge=30)


class MonitorStopRequest(BaseModel):
    content_id: str


class MonitorItem(BaseModel):
    content_id: str
    mode: str
    active: bool
    poll_interval_seconds: int
    last_polled_at: datetime | None
    last_status: str | None


class MonitorStatus(BaseModel):
    worker_running: bool
    monitors: list[MonitorItem]


class DriftEventOut(BaseModel):
    id: int
    content_id: str
    model_version: str
    embedding_score: float | None
    language_score: float | None
    prediction_score: float | None
    vocabulary_oov_rate: float | None
    sample_count: int
    drift_detected: bool
    severity: str
    overall_status: str
    reason: str
    retraining_status: str
    created_at: datetime
    language_distribution: dict[str, float] | None = None
    sentiment_distribution: dict[str, float] | None = None
    toxicity_distribution: dict[str, float] | None = None
    thresholds: dict[str, float] | None = None


class ModelEventOut(BaseModel):
    id: int
    model_version: str
    production_version: str | None
    event_type: str
    drift_event_id: int | None
    run_id: str | None
    candidate_metrics: dict[str, Any] | None
    production_metrics: dict[str, Any] | None
    regression: dict[str, Any] | None
    decision: str
    reason: str
    details: dict[str, Any] | None = None
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
