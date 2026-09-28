"""Registry lifecycle against a real (temporary) MLflow sqlite store."""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def reg(temp_db, tmp_path):
    from src.training import registry

    c = registry.client()
    name = registry.registered_name()
    c.create_registered_model(name)
    src = tmp_path / "model"
    src.mkdir()
    for _ in range(3):
        mv = c.create_model_version(name, source=src.as_uri())
        c.set_registered_model_alias(name, "candidate", mv.version)
    return registry


def aliases(registry, version):
    return next(v["aliases"] for v in registry.list_versions() if str(v["version"]) == version)


def test_promotion_moves_alias_and_clears_candidate(reg):
    reg.promote_to_production("1")
    reg.client().set_registered_model_alias(reg.registered_name(), "candidate", "2")
    prev = reg.promote_to_production("2")
    assert prev == "1"
    assert aliases(reg, "2") == ["production"]  # candidate alias removed (int/str bug)
    v1 = next(v for v in reg.list_versions() if str(v["version"]) == "1")
    assert v1["tags"]["lifecycle"] == "archived" and v1["aliases"] == []


def test_rejection_never_touches_production(reg):
    reg.promote_to_production("1")
    reg.client().set_registered_model_alias(reg.registered_name(), "candidate", "3")
    reg.reject("3", "regression_failed")
    assert str(reg.get_version_by_alias("production").version) == "1"
    assert aliases(reg, "3") == ["rejected"]
    v3 = next(v for v in reg.list_versions() if str(v["version"]) == "3")
    assert v3["tags"]["rejection_reason"] == "regression_failed"
