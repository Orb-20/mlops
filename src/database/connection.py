"""Engine/session management. DATABASE_URL selects PostgreSQL or the SQLite fallback."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_env
from src.database.models import Base

log = logging.getLogger(__name__)


class DatabaseUnavailable(RuntimeError):
    pass


def make_engine(url: str) -> Engine:
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_env().database_url)


@lru_cache
def _factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """Create tables if missing (idempotent). Schema is managed by SQLAlchemy metadata."""
    engine = engine or get_engine()
    try:
        Base.metadata.create_all(engine)
    except OperationalError as exc:
        raise DatabaseUnavailable(f"database unavailable: {exc.orig}") from exc


def check_db() -> bool:
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


@contextmanager
def session_scope(factory: sessionmaker | None = None) -> Iterator[Session]:
    s = (factory or _factory())()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
