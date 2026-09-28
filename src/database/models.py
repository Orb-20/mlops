"""ORM schema (spec §26). Portable across PostgreSQL (compose) and SQLite (local fallback).

Embeddings are stored as float16 bytes (1.5 KB per 768-d vector) in their own table rather than
turning PostgreSQL into a vector DB; drift statistics only need bulk reads of a window.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.data.schemas import utcnow


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    content_id: Mapped[str] = mapped_column(String(64), index=True)
    mode: Mapped[str] = mapped_column(String(8))  # live | demo
    kind: Mapped[str] = mapped_column(String(16), default="on_demand")  # on_demand | monitor
    model_version: Mapped[str] = mapped_column(String(32))
    comment_count: Mapped[int] = mapped_column(Integer)
    summary: Mapped[dict] = mapped_column(JSON)
    drift_event_id: Mapped[int | None] = mapped_column(ForeignKey("drift_events.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Comment(Base):
    __tablename__ = "comments"
    __table_args__ = (UniqueConstraint("source", "source_comment_id", name="uq_source_comment"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    source_content_id: Mapped[str] = mapped_column(String(64), index=True)
    source_comment_id: Mapped[str] = mapped_column(String(128))
    parent_comment_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mode: Mapped[str] = mapped_column(String(8), default="live")
    text: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    language: Mapped[str] = mapped_column(String(16))
    language_confidence: Mapped[float] = mapped_column(Float)


class PredictionRow(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comment_id: Mapped[int] = mapped_column(ForeignKey("comments.id"), index=True)
    analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("analyses.id"), index=True, nullable=True
    )
    model_version: Mapped[str] = mapped_column(String(32))
    sentiment: Mapped[str] = mapped_column(String(16))
    sentiment_confidence: Mapped[float] = mapped_column(Float)
    toxicity_score: Mapped[float] = mapped_column(Float)
    toxicity_label: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EmbeddingRow(Base):
    __tablename__ = "embeddings"
    comment_id: Mapped[int] = mapped_column(ForeignKey("comments.id"), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float16 bytes
    dim: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DriftEvent(Base):
    __tablename__ = "drift_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    content_id: Mapped[str] = mapped_column(String(64), index=True)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_version: Mapped[str] = mapped_column(String(32))
    embedding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    language_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vocabulary_oov_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer)
    drift_detected: Mapped[bool] = mapped_column(Boolean)
    severity: Mapped[str] = mapped_column(String(16))
    overall_status: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON)
    retraining_status: Mapped[str] = mapped_column(String(24), default="none")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelEvent(Base):
    __tablename__ = "model_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32))
    production_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32))  # retraining | promotion_decision | ...
    drift_event_id: Mapped[int | None] = mapped_column(ForeignKey("drift_events.id"), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    production_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    regression: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision: Mapped[str] = mapped_column(String(16))  # PROMOTE | REJECT | FAILED
    reason: Mapped[str] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MonitoredVideo(Base):
    __tablename__ = "monitored_videos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default="youtube")
    content_id: Mapped[str] = mapped_column(String(64), unique=True)
    mode: Mapped[str] = mapped_column(String(8))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    poll_interval_seconds: Mapped[int] = mapped_column(Integer)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
