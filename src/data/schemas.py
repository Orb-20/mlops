"""Canonical record schemas shared by ingestion, pipelines, API and storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SENTIMENT_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
TOXICITY_LABELS: tuple[str, ...] = ("safe", "toxic")
DataMode = Literal["live", "demo"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class CommentRecord(BaseModel):
    """A single social-media comment after normalisation. No author identity is stored."""

    source: str = "youtube"
    source_content_id: str
    source_comment_id: str
    parent_comment_id: str | None = None
    text: str = Field(min_length=1)
    timestamp: datetime | None = None
    ingested_at: datetime = Field(default_factory=utcnow)

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty text")
        return v


class LabeledExample(BaseModel):
    text: str = Field(min_length=1)
    task: Literal["sentiment", "toxicity"]
    label: int

    @field_validator("label")
    @classmethod
    def _label_range(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        task = info.data.get("task")
        n = len(SENTIMENT_LABELS) if task == "sentiment" else len(TOXICITY_LABELS)
        if not 0 <= v < n:
            raise ValueError(f"label {v} out of range for task {task}")
        return v


class Prediction(BaseModel):
    sentiment: str
    sentiment_confidence: float
    sentiment_probs: dict[str, float]
    toxicity_label: str
    toxicity_score: float
