"""A small on-disk cache for Hugging Face lookups.

Only successful answers are stored, one JSON file per repo, and they expire
after a day. Every failure here is swallowed: a broken or read-only cache must
never stop a name from being explained.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

#: How long a cached answer is trusted, in seconds.
MAX_AGE = 24 * 60 * 60

ENV_VAR = "WHATMODEL_CACHE_DIR"


def default_cache_dir() -> Path:
    """Where the command line keeps its cache. ``WHATMODEL_CACHE_DIR`` overrides it."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "whatmodel" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "whatmodel"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "whatmodel"


def _path(cache_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.lower().encode("utf-8")).hexdigest()[:32]
    return cache_dir / f"{digest}.json"


def read(
    cache_dir: Path, key: str, *, max_age: float = MAX_AGE, now: float | None = None
) -> dict[str, Any] | None:
    """Return the cached answer for ``key``, or ``None`` if missing, stale or unreadable."""
    try:
        stored = json.loads(_path(cache_dir, key).read_text(encoding="utf-8"))
        saved_at = float(stored["saved_at"])
        body = stored["body"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if stored.get("key") != key.lower() or not isinstance(body, dict):
        return None
    age = (time.time() if now is None else now) - saved_at
    if age < 0 or age > max_age:
        return None
    return body


def write(cache_dir: Path, key: str, body: dict[str, Any], *, now: float | None = None) -> None:
    """Store ``body`` for ``key``. Does nothing if the cache cannot be written."""
    record = {"key": key.lower(), "saved_at": time.time() if now is None else now, "body": body}
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = _path(cache_dir, key)
        scratch = target.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps(record), encoding="utf-8")
        scratch.replace(target)
    except (OSError, TypeError, ValueError):
        return


def clear(cache_dir: Path) -> int:
    """Delete every cached answer and return how many were removed."""
    removed = 0
    try:
        files = list(cache_dir.glob("*.json"))
    except OSError:
        return 0
    for file in files:
        try:
            file.unlink()
            removed += 1
        except OSError:
            continue
    return removed


__all__ = ["ENV_VAR", "MAX_AGE", "clear", "default_cache_dir", "read", "write"]
