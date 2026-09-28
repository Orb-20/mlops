"""API-less integration: pipeline service -> database, with a fake model."""

import pytest

from src.database import repository as repo
from src.database.connection import session_scope
from src.ingestion.base import MissingAPIKeyError
from src.pipeline.analyze import analyze_url, choose_adapter
from src.pipeline.monitor import poll_once
from src.reporting.report import generate_report

pytestmark = pytest.mark.integration


def test_demo_analysis_persists_everything(temp_db, fake_model):
    res = analyze_url(None, "demo", 120)
    assert res.mode == "demo" and res.summary["comment_count"] == 120
    assert abs(sum(res.summary["sentiment"].values()) - 1) < 1e-3
    with session_scope() as s:
        a = repo.get_analysis(s, res.analysis_id)
        assert a.comment_count == 120 and a.model_version == "99"
        window, start, end = repo.load_window(s, "youtube", "demo", "99", 500)
        assert len(window) == 120 and window.embeddings.shape == (120, 8)
        ev = repo.latest_drift_event(s, "demo")
        assert ev.id == res.drift_event_id and ev.sample_count == 120


def test_reanalysis_does_not_duplicate_comments(temp_db, fake_model):
    analyze_url(None, "demo", 50)
    analyze_url(None, "demo", 50)
    with session_scope() as s:
        assert len(repo.known_comment_ids(s, "youtube", "demo")) == 50
        assert len(repo.recent_analyses(s)) == 2


def test_monitoring_is_incremental(temp_db, fake_model):
    first = poll_once("demo", "demo", batch=100)
    second = poll_once("demo", "demo", batch=100)
    assert first.new_comments == 100 and second.new_comments == 100
    # the drift window for the 2nd poll covers stored + new comments
    assert second.drift["sample_count"] == 200
    with session_scope() as s:
        assert len(repo.known_comment_ids(s, "youtube", "demo")) == 200


def test_small_window_never_triggers_retraining(temp_db, fake_model):
    res = analyze_url(None, "demo", 20)
    assert res.drift["overall_status"] == "insufficient_data"
    assert not res.drift["retraining_recommended"]


def test_report_generated_from_stored_data(temp_db, fake_model):
    res = analyze_url(None, "demo", 120)
    html = generate_report(res.analysis_id, "html")
    md = generate_report(res.analysis_id, "md")
    assert "DEMO DATA" in html and "OBSERVED DATA" in html and "MODEL PREDICTION" in html
    assert "DRIFT SIGNAL" in html
    assert f"{res.summary['comment_count']}" in md
    assert "author" not in html.lower()


def test_live_mode_requires_key(temp_db, monkeypatch):
    from src.config import settings

    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    settings.get_env.cache_clear()
    with pytest.raises(MissingAPIKeyError):
        choose_adapter("live")
    assert choose_adapter("auto").__class__.__name__ == "DemoAdapter"
