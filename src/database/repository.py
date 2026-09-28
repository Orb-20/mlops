"""Data-access functions. Pipelines/API call these; nothing else writes SQL."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.data.schemas import CommentRecord, Prediction
from src.database.models import (
    Analysis,
    Comment,
    DriftEvent,
    EmbeddingRow,
    ModelEvent,
    MonitoredVideo,
    PredictionRow,
)
from src.monitoring.decision import DriftDecision, WindowData
from src.nlp.embeddings import from_bytes, to_bytes
from src.nlp.language import LanguageResult


def known_comment_ids(s: Session, source: str, content_id: str) -> set[str]:
    rows = s.scalars(select(Comment.source_comment_id).where(
        Comment.source == source, Comment.source_content_id == content_id))
    return set(rows)


def upsert_comments(s: Session, comments: list[CommentRecord], langs: list[LanguageResult],
                    mode: str) -> list[Comment]:
    """Insert new comments; return ORM rows (existing rows reused) in input order."""
    if not comments:
        return []
    ids = [c.source_comment_id for c in comments]
    existing = {c.source_comment_id: c for c in s.scalars(select(Comment).where(
        Comment.source == comments[0].source, Comment.source_comment_id.in_(ids)))}
    out = []
    for c, lang in zip(comments, langs, strict=True):
        row = existing.get(c.source_comment_id)
        if row is None:
            row = Comment(source=c.source, source_content_id=c.source_content_id,
                          source_comment_id=c.source_comment_id,
                          parent_comment_id=c.parent_comment_id, mode=mode, text=c.text,
                          timestamp=c.timestamp, ingested_at=c.ingested_at,
                          language=lang.language, language_confidence=lang.confidence)
            s.add(row)
            existing[c.source_comment_id] = row
        out.append(row)
    s.flush()
    return out


def save_analysis(s: Session, *, source: str, content_id: str, mode: str, kind: str,
                  model_version: str, comments: list[CommentRecord],
                  langs: list[LanguageResult], predictions: list[Prediction],
                  embeddings: np.ndarray, summary: dict) -> Analysis:
    analysis = Analysis(source=source, content_id=content_id, mode=mode, kind=kind,
                        model_version=model_version, comment_count=len(comments),
                        summary=summary)
    s.add(analysis)
    s.flush()
    rows = upsert_comments(s, comments, langs, mode)
    have_emb = set(s.scalars(select(EmbeddingRow.comment_id).where(
        EmbeddingRow.model_version == model_version,
        EmbeddingRow.comment_id.in_([r.id for r in rows]))))
    for row, p, e in zip(rows, predictions, embeddings, strict=True):
        s.add(PredictionRow(comment_id=row.id, analysis_id=analysis.id,
                            model_version=model_version, sentiment=p.sentiment,
                            sentiment_confidence=p.sentiment_confidence,
                            toxicity_score=p.toxicity_score, toxicity_label=p.toxicity_label))
        if row.id not in have_emb:
            s.add(EmbeddingRow(comment_id=row.id, model_version=model_version,
                               vector=to_bytes(e), dim=int(e.shape[0])))
            have_emb.add(row.id)
    s.flush()
    return analysis


def load_window(s: Session, source: str, content_id: str, model_version: str,
                limit: int) -> tuple[WindowData, datetime | None, datetime | None]:
    """Most recent `limit` comments of a content item that have embeddings for this model."""
    q = (select(Comment, EmbeddingRow, PredictionRow)
         .join(EmbeddingRow, (EmbeddingRow.comment_id == Comment.id)
               & (EmbeddingRow.model_version == model_version))
         .join(PredictionRow, (PredictionRow.comment_id == Comment.id)
               & (PredictionRow.model_version == model_version))
         .where(Comment.source == source, Comment.source_content_id == content_id)
         .order_by(desc(Comment.ingested_at), desc(Comment.id)))
    seen, rows = set(), []
    for c, e, p in s.execute(q):
        if c.id in seen:
            continue
        seen.add(c.id)
        rows.append((c, e, p))
        if len(rows) >= limit:
            break
    rows.reverse()
    if not rows:
        return WindowData(np.zeros((0, 1), np.float32), [], [], [], []), None, None
    emb = np.stack([from_bytes(e.vector) for _, e, _ in rows])
    w = WindowData(emb, [c.language for c, _, _ in rows], [p.sentiment for _, _, p in rows],
                   [p.toxicity_label for _, _, p in rows], [c.text for c, _, _ in rows])
    stamps = [c.ingested_at for c, _, _ in rows]
    return w, min(stamps), max(stamps)


def save_drift_event(s: Session, decision: DriftDecision, *, source: str, content_id: str,
                     model_version: str, window_start=None, window_end=None) -> DriftEvent:
    ev = DriftEvent(
        source=source, content_id=content_id, model_version=model_version,
        window_start=window_start, window_end=window_end,
        embedding_score=decision.scores.get("embedding"),
        language_score=decision.scores.get("language"),
        prediction_score=decision.scores.get("prediction"),
        vocabulary_oov_rate=(decision.vocabulary or {}).get("oov_rate"),
        sample_count=decision.sample_count, drift_detected=decision.drift_detected,
        severity=decision.severity, overall_status=decision.overall_status,
        reason=", ".join(decision.reasons) or "none", details=_jsonable(decision.to_dict()),
        retraining_status="pending" if decision.retraining_recommended else "none",
    )
    s.add(ev)
    s.flush()
    return ev


def save_model_event(s: Session, **fields) -> ModelEvent:
    ev = ModelEvent(**{k: _jsonable(v) if isinstance(v, dict) else v for k, v in fields.items()})
    s.add(ev)
    s.flush()
    return ev


def latest_drift_event(s: Session, content_id: str | None = None) -> DriftEvent | None:
    q = select(DriftEvent).order_by(desc(DriftEvent.id)).limit(1)
    if content_id:
        q = q.where(DriftEvent.content_id == content_id)
    return s.scalars(q).first()


def drift_history(s: Session, limit: int = 100) -> list[DriftEvent]:
    return list(s.scalars(select(DriftEvent).order_by(desc(DriftEvent.id)).limit(limit)))


def pending_drift_events(s: Session) -> list[DriftEvent]:
    return list(s.scalars(select(DriftEvent).where(DriftEvent.retraining_status == "pending")
                          .order_by(DriftEvent.id)))


def model_events(s: Session, limit: int = 100) -> list[ModelEvent]:
    return list(s.scalars(select(ModelEvent).order_by(desc(ModelEvent.id)).limit(limit)))


def get_analysis(s: Session, analysis_id: int) -> Analysis | None:
    return s.get(Analysis, analysis_id)


def recent_analyses(s: Session, limit: int = 20) -> list[Analysis]:
    return list(s.scalars(select(Analysis).order_by(desc(Analysis.id)).limit(limit)))


def upsert_monitor(s: Session, content_id: str, mode: str, interval: int,
                   active: bool = True) -> MonitoredVideo:
    m = s.scalars(select(MonitoredVideo).where(MonitoredVideo.content_id == content_id)).first()
    if m is None:
        m = MonitoredVideo(content_id=content_id, mode=mode, poll_interval_seconds=interval,
                           active=active)
        s.add(m)
    else:
        m.active, m.mode, m.poll_interval_seconds = active, mode, interval
    s.flush()
    return m


def list_monitors(s: Session, active_only: bool = False) -> list[MonitoredVideo]:
    q = select(MonitoredVideo).order_by(MonitoredVideo.id)
    if active_only:
        q = q.where(MonitoredVideo.active.is_(True))
    return list(s.scalars(q))


def _jsonable(obj):
    """Make numpy / datetime values JSON-serialisable for JSON columns."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj
