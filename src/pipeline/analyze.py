"""On-demand analysis pipeline (spec §22). Shared by CLI, FastAPI and Streamlit.

URL → validate → extract id → fetch → normalise/validate → language → sentiment + toxicity
→ embeddings → drift vs reference → persist → analytics → (report on demand)

Usage:
    python -m src.pipeline.analyze --url "https://www.youtube.com/watch?v=VIDEO_ID"
    python -m src.pipeline.analyze --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.config.settings import get_config, get_env
from src.data.schemas import utcnow
from src.data.validation import validate_comment_frame
from src.database import repository as repo
from src.database.connection import init_db, session_scope
from src.ingestion.base import IngestionError, MissingAPIKeyError
from src.ingestion.youtube import DemoAdapter, YouTubeAdapter
from src.monitoring.decision import DriftDecision, DriftDecisionEngine, WindowData
from src.monitoring.reference import get_reference
from src.nlp.language import detect_many
from src.pipeline.model_service import get_model_service
from src.reporting.analytics import compute_summary

log = logging.getLogger(__name__)


@dataclass
class AnalysisOutcome:
    analysis_id: int
    source: str
    content_id: str
    mode: str
    model_version: str
    summary: dict
    drift: dict
    drift_event_id: int | None
    skipped_malformed: int
    new_comments: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def choose_adapter(mode: str):
    """mode: live | demo | auto (live if a key is configured, else demo)."""
    if mode == "demo":
        return DemoAdapter()
    if mode == "live":
        if not get_env().has_youtube_key:
            raise MissingAPIKeyError("YOUTUBE_API_KEY is not set; live mode unavailable. "
                                     "Use demo mode or add the key to .env.")
        return YouTubeAdapter()
    return YouTubeAdapter() if get_env().has_youtube_key else DemoAdapter()


def evaluate_drift(predictor, version: str, current: WindowData) -> DriftDecision:
    ref = get_reference(predictor, version)
    return DriftDecisionEngine().evaluate(ref, current)


def process_comments(comments, content_id: str, mode: str, kind: str, source: str = "youtube",
                     skipped: int = 0,
                     extend_window: Callable[[WindowData, str], WindowData] | None = None
                     ) -> AnalysisOutcome:
    """Run the shared analysis steps on already-fetched comments.
    `extend_window(batch_window, model_version)` lets monitoring prepend stored history."""
    if not comments:
        raise IngestionError("No comments available to analyse for this content.")
    frame = pd.DataFrame([c.model_dump() for c in comments])
    validate_comment_frame(frame).raise_if_failed()

    predictor, info = get_model_service().get()
    texts = [c.text for c in comments]
    log.info("Running language detection on %d comments", len(texts))
    langs = detect_many(texts)
    log.info("Running inference with model v%s", info.version)
    out = predictor.run(texts)
    summary = compute_summary(comments, langs, out.predictions)
    summary["analysed_at"] = utcnow().isoformat()

    current = WindowData(
        out.embeddings, [r.language for r in langs], [p.sentiment for p in out.predictions],
        [p.toxicity_label for p in out.predictions], texts)
    if extend_window is not None:
        current = extend_window(current, info.version)
    log.info("Running drift monitoring (%d samples)", len(current))
    decision = evaluate_drift(predictor, info.version, current)
    if decision.drift_detected:
        log.warning("Drift detected: %s", decision.reasons)

    with session_scope() as s:
        analysis = repo.save_analysis(
            s, source=source, content_id=content_id, mode=mode, kind=kind,
            model_version=info.version, comments=comments, langs=langs,
            predictions=out.predictions, embeddings=out.embeddings, summary=summary)
        ev = repo.save_drift_event(s, decision, source=source, content_id=content_id,
                                   model_version=info.version,
                                   window_start=min(c.ingested_at for c in comments),
                                   window_end=max(c.ingested_at for c in comments))
        analysis.drift_event_id = ev.id
        s.flush()
        analysis_id, ev_id = analysis.id, ev.id

    return AnalysisOutcome(analysis_id, source, content_id, mode, info.version, summary,
                           _compact_drift(decision), ev_id, skipped, len(comments))


def analyze_url(url: str | None, mode: str = "auto", max_comments: int | None = None
                ) -> AnalysisOutcome:
    init_db()
    max_comments = max_comments or get_config()["ingestion"]["default_max_comments"]
    adapter = choose_adapter(mode)
    content_id = adapter.extract_content_id(url or "")
    log.info("Fetching comments for %s (%s mode)", content_id,
             "DEMO" if isinstance(adapter, DemoAdapter) else "LIVE")
    fetched = adapter.fetch_comments(content_id, max_comments=max_comments)
    return process_comments(fetched.comments, content_id, fetched.mode, "on_demand",
                            fetched.source, fetched.skipped_malformed)


def _compact_drift(d: DriftDecision) -> dict:
    """Drift info for API/UI: no raw vectors, vocabulary trimmed."""
    out = d.to_dict()
    for sig in out.get("signals", {}).values():
        sig.get("details", {}).pop("reference", None)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-comments", type=int)
    args = ap.parse_args()
    if not args.url and not args.demo:
        ap.error("--url or --demo required")
    try:
        res = analyze_url(args.url, "demo" if args.demo else "auto", args.max_comments)
    except IngestionError as exc:
        log.error("%s", exc)
        return 2
    d = res.to_dict()
    d["drift"] = {k: d["drift"][k] for k in ("overall_status", "drift_detected", "reasons",
                                             "scores", "thresholds", "sample_count")}
    d["summary"] = {k: v for k, v in d["summary"].items() if k != "trend"}
    print(json.dumps(d, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
