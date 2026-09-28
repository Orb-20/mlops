from datetime import timedelta

import pytest

from src.data.schemas import utcnow
from src.database import repository as repo
from src.database.connection import session_scope
from src.pipeline.monitor import MonitorWorker

pytestmark = pytest.mark.integration


def test_worker_polls_only_due_active_monitors(temp_db, fake_model):
    with session_scope() as s:
        repo.upsert_monitor(s, "demo", "demo", interval=300)
        m = repo.upsert_monitor(s, "other", "demo", interval=300)
        m.last_polled_at = utcnow() - timedelta(seconds=10)  # not due yet
        repo.upsert_monitor(s, "stopped", "demo", interval=300, active=False)

    worker = MonitorWorker()
    assert worker.run_due() == 1  # only "demo" (never polled)
    with session_scope() as s:
        mons = {m.content_id: m for m in repo.list_monitors(s)}
        assert mons["demo"].last_polled_at is not None
        assert "new comments" in mons["demo"].last_status
        assert len(repo.known_comment_ids(s, "youtube", "demo")) > 0
    assert worker.run_due() == 0  # just polled -> not due again


def test_worker_records_ingestion_errors(temp_db, fake_model, monkeypatch):
    from src.config import settings

    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    settings.get_env.cache_clear()
    with session_scope() as s:
        repo.upsert_monitor(s, "dQw4w9WgXcQ", "live", interval=300)
    # live mode without a key -> MissingAPIKeyError is recorded and the worker survives
    assert MonitorWorker().run_due() == 1
    with session_scope() as s:
        assert repo.list_monitors(s)[0].last_status.startswith("error")
