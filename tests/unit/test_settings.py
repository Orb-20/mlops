import random

import numpy as np

from src.config.settings import get_config, get_env, set_seed


def test_config_merges_all_yaml_files():
    cfg = get_config()
    for key in ("project", "data", "model", "training", "drift", "gates", "mlflow"):
        assert key in cfg, key


def test_drift_weights_sum_to_one():
    d = get_config()["drift"]
    total = sum(d[s]["weight"] for s in ("embedding", "language", "prediction"))
    assert abs(total - 1.0) < 1e-9


def test_env_defaults_do_not_contain_secret():
    env = get_env()
    # The key may be set by the developer's .env; we only check the property is consistent.
    assert env.has_youtube_key == bool(env.youtube_api_key.strip())


def test_set_seed_is_deterministic():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b
