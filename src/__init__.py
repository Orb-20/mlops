"""Adaptive NLP MLOps pipeline."""

# Windows DLL load-order workaround: if torch (with numpy/sqlalchemy) loads its C++ runtime
# before pyarrow, pyarrow later hits an access violation inside worker threads (observed as a
# segfault in the FastAPI threadpool while pandas builds pyarrow-backed strings). Loading
# pyarrow first avoids it. Harmless on Linux/macOS. See docs/limitations.md.
try:  # pragma: no cover - platform-specific
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover
    pass
