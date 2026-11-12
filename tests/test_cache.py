"""The on-disk lookup cache."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from whatmodel import cache, explain

BODY = {"id": "org/repo", "cardData": {"license": "mit"}}


def test_round_trip(tmp_path: Path) -> None:
    assert cache.read(tmp_path, "org/repo") is None
    cache.write(tmp_path, "org/repo", BODY)
    assert cache.read(tmp_path, "org/repo") == BODY
    assert cache.read(tmp_path, "Org/Repo") == BODY  # repo names ignore case
    assert cache.read(tmp_path, "org/other") is None


def test_entries_expire(tmp_path: Path) -> None:
    cache.write(tmp_path, "org/repo", BODY, now=1000.0)
    assert cache.read(tmp_path, "org/repo", now=1000.0 + cache.MAX_AGE - 1) == BODY
    assert cache.read(tmp_path, "org/repo", now=1000.0 + cache.MAX_AGE + 1) is None
    assert cache.read(tmp_path, "org/repo", now=999.0) is None  # saved in the future


def test_broken_files_are_ignored(tmp_path: Path) -> None:
    cache.write(tmp_path, "org/repo", BODY)
    (file,) = tmp_path.glob("*.json")
    file.write_text("not json", encoding="utf-8")
    assert cache.read(tmp_path, "org/repo") is None


def test_unwritable_cache_is_ignored(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    cache.write(blocker / "sub", "org/repo", BODY)  # must not raise
    assert cache.read(blocker / "sub", "org/repo") is None


def test_clear(tmp_path: Path) -> None:
    cache.write(tmp_path, "a/b", BODY)
    cache.write(tmp_path, "c/d", BODY)
    assert cache.clear(tmp_path) == 2
    assert cache.read(tmp_path, "a/b") is None
    assert cache.clear(tmp_path / "missing") == 0


def test_default_dir_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cache.ENV_VAR, "/somewhere/else")
    assert cache.default_cache_dir() == Path("/somewhere/else")
    monkeypatch.delenv(cache.ENV_VAR)
    assert cache.default_cache_dir().name.lower() in {"whatmodel", "cache"}


def _client(calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json=BODY)
        return httpx.Response(404, json={})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_second_lookup_is_served_from_the_cache(tmp_path: Path) -> None:
    calls: list[str] = []
    first = explain("org/repo", client=_client(calls), cache_dir=tmp_path)
    assert first.cached == ()
    assert calls.count("huggingface.co") == 1

    second = explain("org/repo", client=_client(calls), cache_dir=tmp_path)
    assert second.cached == ("huggingface",)
    assert calls.count("huggingface.co") == 1  # no second request
    assert second.facts == first.facts
    assert second.lookups["huggingface"] == "ok"


def test_nothing_touches_disk_without_a_cache_dir(tmp_path: Path) -> None:
    calls: list[str] = []
    explain("org/repo", client=_client(calls))
    explain("org/repo", client=_client(calls))
    assert calls.count("huggingface.co") == 2
    assert list(tmp_path.iterdir()) == []


def test_failed_lookups_are_not_cached(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    explain("org/repo", client=client, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []
