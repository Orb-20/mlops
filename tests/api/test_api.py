import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(temp_db, fake_model):
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_analyze_demo_and_fetch_report(client):
    r = client.post("/api/v1/analyze/youtube", json={"mode": "demo", "max_comments": 120})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data_mode"] == "demo" and body["data_label"] == "DEMO DATA"
    assert body["comment_count"] == 120 and body["model_version"] == "99"
    assert set(body["sentiment"]) == {"negative", "neutral", "positive"}
    assert set(body["languages"]) >= {"en", "hi", "hinglish", "other"}
    assert body["drift"]["status"] in {"stable", "warning", "drift"}
    assert "embeddings" not in r.text  # no raw vectors in responses

    rep = client.get(body["report_url"])
    assert rep.status_code == 200 and "text/html" in rep.headers["content-type"]
    md = client.get(body["report_url"] + "?format=md")
    assert md.status_code == 200 and md.text.startswith("# Audience Analysis Report")


def test_invalid_url_422(client):
    r = client.post("/api/v1/analyze/youtube", json={"url": "https://vimeo.com/1"})
    assert r.status_code == 422


def test_live_without_key_400(client, monkeypatch):
    from src.config import settings

    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    settings.get_env.cache_clear()
    r = client.post("/api/v1/analyze/youtube",
                    json={"url": "https://youtu.be/dQw4w9WgXcQ", "mode": "live"})
    assert r.status_code == 400 and "YOUTUBE_API_KEY" in r.json()["detail"]


def test_max_comments_validation(client):
    assert client.post("/api/v1/analyze/youtube",
                       json={"mode": "demo", "max_comments": 0}).status_code == 422


def test_drift_endpoints(client):
    assert client.get("/api/v1/drift/latest").json() is None
    client.post("/api/v1/analyze/youtube", json={"mode": "demo", "max_comments": 120})
    latest = client.get("/api/v1/drift/latest").json()
    assert latest["content_id"] == "demo" and latest["sample_count"] == 120
    assert len(client.get("/api/v1/drift/history").json()) == 1
    assert client.get("/api/v1/retraining/history").json() == []


def test_monitor_lifecycle(client):
    r = client.post("/api/v1/monitor/start", json={"mode": "demo", "poll_interval_seconds": 60})
    assert r.status_code == 200 and r.json()["active"]
    st = client.get("/api/v1/monitor/status").json()
    assert st["monitors"][0]["content_id"] == "demo"
    r = client.post("/api/v1/monitor/stop", json={"content_id": "demo"})
    assert r.json()["active"] is False
    assert client.post("/api/v1/monitor/stop", json={"content_id": "nope"}).status_code == 404


def test_model_endpoint(client):
    r = client.get("/api/v1/model")
    assert r.status_code == 200 and r.json()["version"] == "99"


def test_report_404(client):
    assert client.get("/api/v1/report/12345").status_code == 404


def test_health_reports_degraded_without_production_model(client):
    body = client.get("/health").json()
    assert body["database"] is True
    assert body["status"] == "degraded"  # temp MLflow has no production model
    assert "key" not in str(body).lower().replace("youtube_api_configured", "")
