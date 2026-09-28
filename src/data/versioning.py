"""Dataset/code lineage helpers: content-hash dataset versions and git commit lookup.

DVC tracks the files themselves (dvc.yaml / dvc.lock). The content hash computed here is what
gets logged to MLflow as `dataset_version`, so any model can be traced back to exact data
even when DVC is not available in the runtime container.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.config.settings import PROJECT_ROOT


def hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.as_posix()):
        h.update(p.name.encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


def dataset_version(directory: Path, pattern: str = "*.parquet") -> str:
    files = sorted(directory.glob(pattern))
    if not files:
        return "none"
    return hash_files(files)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = out.stdout.strip()
        if out.returncode != 0 or not commit:
            return "uncommitted"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(directory: Path, extra: dict) -> Path:
    manifest = {"dataset_version": dataset_version(directory), "git_commit": git_commit(), **extra}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
