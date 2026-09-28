"""Continuous monitoring (spec §23, Mode B): simple polling worker, no queue infrastructure.

Each poll fetches only comments not already stored (known ids), processes them, then runs
drift on the latest `monitoring.window_size` stored comments of that video.

Usage:
    python -m src.pipeline.monitor --url <URL> --once        # one poll
    python -m src.pipeline.monitor --demo --once --batch 100 # demo stream, 100 new per poll
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from datetime import timedelta

import numpy as np

from src.config.settings import get_config
from src.data.schemas import utcnow
from src.database import repository as repo
from src.database.connection import init_db, session_scope
from src.ingestion.base import IngestionError
from src.monitoring.decision import WindowData
from src.pipeline.analyze import AnalysisOutcome, choose_adapter, process_comments

log = logging.getLogger(__name__)


def poll_once(content_id: str, mode: str, batch: int | None = None,
              source: str = "youtube") -> AnalysisOutcome | None:
    """Process new comments for one content item. Returns None when nothing new arrived."""
    init_db()
    cfg = get_config()
    adapter = choose_adapter(mode)
    with session_scope() as s:
        known = repo.known_comment_ids(s, source, content_id)
    fetched = adapter.fetch_comments(content_id,
                                     max_comments=batch or cfg["ingestion"]["default_max_comments"],
                                     known_ids=known)
    new = [c for c in fetched.comments if c.source_comment_id not in known]
    status = f"{len(new)} new comments"
    if not new:
        log.info("No new comments for %s", content_id)
        _mark_polled(content_id, status)
        return None
    # drift is computed on the rolling window (stored history + new batch)
    outcome = process_comments(
        new, content_id, fetched.mode, "monitor", source, fetched.skipped_malformed,
        extend_window=lambda batch, version: rolling_window(batch, content_id, version, source))
    _mark_polled(content_id, f"{status}; drift={outcome.drift['overall_status']}")
    return outcome


def rolling_window(batch: WindowData, content_id: str, version: str,
                   source: str = "youtube") -> WindowData:
    """Stored window (before this batch is persisted) + new batch, trimmed to window_size."""
    size = get_config()["monitoring"]["window_size"]
    with session_scope() as s:
        stored, _, _ = repo.load_window(s, source, content_id, version, size)
    if not len(stored):
        w = batch
    else:
        w = WindowData(np.vstack([stored.embeddings, batch.embeddings]),
                       stored.languages + batch.languages, stored.sentiments + batch.sentiments,
                       stored.toxicity_labels + batch.toxicity_labels, stored.texts + batch.texts)
    k = max(0, len(w) - size)
    return WindowData(w.embeddings[k:], w.languages[k:], w.sentiments[k:],
                      w.toxicity_labels[k:], w.texts[k:])


def _mark_polled(content_id: str, status: str) -> None:
    with session_scope() as s:
        for m in repo.list_monitors(s):
            if m.content_id == content_id:
                m.last_polled_at = utcnow()
                m.last_status = status[:64]


class MonitorWorker:
    """Background thread polling every active monitor when its interval has elapsed."""

    def __init__(self, tick_seconds: float = 10.0):
        self.tick = tick_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="monitor-worker", daemon=True)
        self._thread.start()
        log.info("Monitor worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due()
            except Exception:  # noqa: BLE001 - worker must survive individual failures
                log.exception("monitor tick failed")
            self._stop.wait(self.tick)

    def run_due(self) -> int:
        now = utcnow()
        with session_scope() as s:
            due = [(m.content_id, m.mode) for m in repo.list_monitors(s, active_only=True)
                   if m.last_polled_at is None
                   or _aware(m.last_polled_at) + timedelta(seconds=m.poll_interval_seconds) <= now]
        for content_id, mode in due:
            try:
                poll_once(content_id, mode)
            except IngestionError as exc:
                log.warning("Monitor poll for %s failed: %s", content_id, exc)
                _mark_polled(content_id, f"error: {exc}")
        return len(due)


def _aware(dt):
    from datetime import UTC

    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()
    mode = "demo" if args.demo else "auto"
    adapter = choose_adapter(mode)
    content_id = adapter.extract_content_id(args.url or "")
    init_db()
    with session_scope() as s:
        repo.upsert_monitor(s, content_id, "demo" if args.demo else "live",
                            get_config()["monitoring"]["poll_interval_seconds"])
    if args.once:
        res = poll_once(content_id, mode, args.batch)
        print("no new comments" if res is None else
              f"analysis {res.analysis_id}: {res.new_comments} new, drift="
              f"{res.drift['overall_status']} scores={res.drift['scores']}")
        return 0
    worker = MonitorWorker()
    worker.start()
    try:
        while worker.running:
            worker._thread.join(1)  # type: ignore[union-attr]
    except KeyboardInterrupt:
        worker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
