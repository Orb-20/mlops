"""All v1 routes. Business logic lives in src.pipeline / src.reporting; routes only adapt."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, PlainTextResponse

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DriftEventOut,
    DriftSummary,
    HealthResponse,
    ModelEventOut,
    ModelResponse,
    MonitorItem,
    MonitorRequest,
    MonitorStatus,
    MonitorStopRequest,
)
from src.config.settings import get_config, get_env
from src.database import repository as repo
from src.database.connection import check_db, session_scope
from src.ingestion.base import (
    CommentsDisabledError,
    ContentNotFoundError,
    IngestionError,
    InvalidURLError,
    MissingAPIKeyError,
    QuotaExceededError,
)
from src.ingestion.youtube import parse_video_id
from src.pipeline.analyze import analyze_url, choose_adapter
from src.pipeline.model_service import ModelUnavailable, get_model_service
from src.reporting.report import ReportNotFound, generate_report
from src.training import registry

router = APIRouter()

_INGESTION_STATUS = {
    InvalidURLError: 422,
    MissingAPIKeyError: 400,
    QuotaExceededError: 429,
    CommentsDisabledError: 409,
    ContentNotFoundError: 404,
}


def _ingestion_http(exc: IngestionError) -> HTTPException:
    for cls, code in _INGESTION_STATUS.items():
        if isinstance(exc, cls):
            return HTTPException(code, str(exc))
    return HTTPException(502, str(exc))


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    db = check_db()
    prod, mlflow_ok = None, True
    try:
        mv = registry.get_version_by_alias("production")
        prod = str(mv.version) if mv else None
        registry.client().search_registered_models(max_results=1)
    except Exception:  # noqa: BLE001
        mlflow_ok = False
    env = get_env()
    return HealthResponse(
        status="ok" if db and mlflow_ok and prod else "degraded",
        database=db,
        mlflow=mlflow_ok,
        production_model=prod,
        youtube_api_configured=env.has_youtube_key,
        environment=env.environment,
    )


@router.get("/api/v1/model", response_model=ModelResponse, tags=["model"])
def model_info() -> ModelResponse:
    try:
        info = get_model_service().info()
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return ModelResponse(
        registered_model=registry.registered_name(),
        version=info.version,
        status=info.lifecycle or "production",
        run_id=info.run_id,
        dataset_version=info.dataset_version,
        git_commit=info.git_commit,
        created=datetime.fromtimestamp(info.created / 1000, UTC) if info.created else None,
        versions=registry.list_versions(),
    )


@router.post("/api/v1/analyze/youtube", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if req.mode != "demo" and req.url and parse_video_id(req.url) is None:
        raise HTTPException(422, f"Not a valid YouTube video URL: {req.url!r}")
    if req.mode == "live" and not req.url:
        raise HTTPException(422, "url is required in live mode")
    try:
        res = await run_in_threadpool(analyze_url, req.url, req.mode, req.max_comments)
    except IngestionError as exc:
        raise _ingestion_http(exc) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    s, d = res.summary, res.drift
    return AnalyzeResponse(
        analysis_id=res.analysis_id,
        source=res.source,
        content_id=res.content_id,
        data_mode=res.mode,
        data_label="LIVE YOUTUBE DATA" if res.mode == "live" else "DEMO DATA",
        comment_count=s["comment_count"],
        model_version=res.model_version,
        sentiment=s["sentiment"],
        toxicity=s["toxicity"],
        languages=s["languages"],
        summary=s,
        drift=DriftSummary(
            status=d["overall_status"],
            drift_detected=d["drift_detected"],
            severity=d["severity"],
            reasons=d["reasons"],
            scores=d["scores"],
            thresholds=d["thresholds"],
            sample_count=d["sample_count"],
            event_id=res.drift_event_id,
            vocabulary=d.get("vocabulary"),
        ),
        report_url=f"/api/v1/report/{res.analysis_id}",
    )


# ------------------------------------------------------------------------------ monitoring
@router.post("/api/v1/monitor/start", response_model=MonitorItem, tags=["monitoring"])
def monitor_start(req: MonitorRequest) -> MonitorItem:
    try:
        adapter = choose_adapter(req.mode)
        content_id = adapter.extract_content_id(req.url or "")
    except IngestionError as exc:
        raise _ingestion_http(exc) from exc
    interval = req.poll_interval_seconds or get_config()["monitoring"]["poll_interval_seconds"]
    mode = "demo" if req.mode == "demo" or adapter.__class__.__name__ == "DemoAdapter" else "live"
    with session_scope() as s:
        m = repo.upsert_monitor(s, content_id, mode, interval, active=True)
        return _monitor_item(m)


@router.post("/api/v1/monitor/stop", response_model=MonitorItem, tags=["monitoring"])
def monitor_stop(req: MonitorStopRequest) -> MonitorItem:
    with session_scope() as s:
        for m in repo.list_monitors(s):
            if m.content_id == req.content_id:
                m.active = False
                s.flush()
                return _monitor_item(m)
    raise HTTPException(404, f"no monitor for {req.content_id}")


@router.get("/api/v1/monitor/status", response_model=MonitorStatus, tags=["monitoring"])
def monitor_status() -> MonitorStatus:
    from src.api.main import worker

    with session_scope() as s:
        items = [_monitor_item(m) for m in repo.list_monitors(s)]
    return MonitorStatus(worker_running=worker.running, monitors=items)


def _monitor_item(m) -> MonitorItem:
    return MonitorItem(
        content_id=m.content_id,
        mode=m.mode,
        active=m.active,
        poll_interval_seconds=m.poll_interval_seconds,
        last_polled_at=m.last_polled_at,
        last_status=m.last_status,
    )


# ------------------------------------------------------------------------------ drift / retraining
def _drift_out(e) -> DriftEventOut:
    out = DriftEventOut.model_validate(e, from_attributes=True)
    sig = (e.details or {}).get("signals", {})
    cur_lang = sig.get("language", {}).get("details", {}).get("current", {})
    cur_pred = sig.get("prediction", {}).get("details", {}).get("current", {})
    out.language_distribution = cur_lang.get("language")
    out.sentiment_distribution = cur_pred.get("sentiment")
    out.toxicity_distribution = cur_pred.get("toxicity")
    out.thresholds = (e.details or {}).get("thresholds") or None
    return out


@router.get("/api/v1/drift/latest", response_model=DriftEventOut | None, tags=["drift"])
def drift_latest(content_id: str | None = None):
    with session_scope() as s:
        e = repo.latest_drift_event(s, content_id)
        return _drift_out(e) if e else None


@router.get("/api/v1/drift/history", response_model=list[DriftEventOut], tags=["drift"])
def drift_history(limit: int = Query(100, ge=1, le=1000)):
    with session_scope() as s:
        return [_drift_out(e) for e in repo.drift_history(s, limit)]


@router.get("/api/v1/retraining/history", response_model=list[ModelEventOut], tags=["retraining"])
def retraining_history(limit: int = Query(100, ge=1, le=1000)):
    with session_scope() as s:
        return [
            ModelEventOut.model_validate(e, from_attributes=True)
            for e in repo.model_events(s, limit)
        ]


@router.get("/api/v1/report/{analysis_id}", tags=["report"])
def report(analysis_id: int, format: str = Query("html", pattern="^(html|md)$")):
    try:
        body = generate_report(analysis_id, format)
    except ReportNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    if format == "html":
        return HTMLResponse(body)
    return PlainTextResponse(body, media_type="text/markdown")


@router.get("/api/v1/analyses", tags=["analysis"])
def analyses(limit: int = Query(20, ge=1, le=200)):
    with session_scope() as s:
        return [
            {
                "id": a.id,
                "content_id": a.content_id,
                "mode": a.mode,
                "kind": a.kind,
                "model_version": a.model_version,
                "comment_count": a.comment_count,
                "created_at": a.created_at,
                "drift_event_id": a.drift_event_id,
            }
            for a in repo.recent_analyses(s, limit)
        ]


@router.get("/api/v1/analyses/{analysis_id}", tags=["analysis"])
def analysis_detail(analysis_id: int):
    """Stored summary + drift decision of one analysis (for re-opening in the dashboard)."""
    from src.database.models import DriftEvent

    with session_scope() as s:
        a = repo.get_analysis(s, analysis_id)
        if a is None:
            raise HTTPException(404, f"analysis {analysis_id} not found")
        ev = s.get(DriftEvent, a.drift_event_id) if a.drift_event_id else None
        drift = (ev.details or {}) if ev else {}
        for sig in drift.get("signals", {}).values():
            sig.get("details", {}).pop("reference", None)
        return {
            "analysis_id": a.id,
            "source": a.source,
            "content_id": a.content_id,
            "data_mode": a.mode,
            "kind": a.kind,
            "model_version": a.model_version,
            "comment_count": a.comment_count,
            "created_at": a.created_at,
            "summary": a.summary,
            "drift": drift,
            "drift_event_id": a.drift_event_id,
            "retraining_status": ev.retraining_status if ev else None,
            "report_url": f"/api/v1/report/{a.id}",
        }
