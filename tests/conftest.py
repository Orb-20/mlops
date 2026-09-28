import os

import pytest

# Tests never touch the developer's local.db / mlflow.db.
os.environ.setdefault("DISABLE_MONITOR_WORKER", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite file and reset cached engines/settings."""
    from src.config import settings
    from src.database import connection

    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    settings.get_env.cache_clear()
    connection.get_engine.cache_clear()
    connection._factory.cache_clear()
    connection.init_db()
    yield url
    connection.get_engine().dispose()
    settings.get_env.cache_clear()
    connection.get_engine.cache_clear()
    connection._factory.cache_clear()


@pytest.fixture
def fake_model(monkeypatch):
    """Serve a deterministic rule-based predictor instead of the MLflow production model."""
    from src.monitoring import reference
    from src.monitoring.decision import WindowData
    from src.pipeline import analyze, model_service
    from tests.fixtures.fakes import RulePredictor

    pred = RulePredictor()
    info = model_service.ModelInfo("99", "run-x", "ds-x", "abc123", None, "production")

    class FakeService:
        def get(self):
            return pred, info

        def info(self):
            return info

    monkeypatch.setattr(model_service, "_service", FakeService())
    ref_texts = [f"reference comment number {i} about a good video" for i in range(120)]
    out = pred.run(ref_texts)
    ref = WindowData(out.embeddings, ["en"] * 120, [p.sentiment for p in out.predictions],
                     [p.toxicity_label for p in out.predictions], ref_texts)
    monkeypatch.setattr(analyze, "get_reference", lambda p, v: ref)
    monkeypatch.setattr(reference, "get_reference", lambda p, v, rebuild=False: ref)
    return pred
