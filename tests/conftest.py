"""Shared test setup."""

from __future__ import annotations

import socket

import pytest


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach the network."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a real connection, including to localhost.

    Layer 1 must work offline and Layer 2 tests must mock HTTP. Blocking
    localhost matters too: a developer machine may have a real Ollama running,
    and a test that quietly used it would pass locally and fail in CI.
    """

    def blocked(*args: object, **kwargs: object) -> None:
        raise NetworkBlockedError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
