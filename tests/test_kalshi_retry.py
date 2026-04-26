"""Tests for Kalshi retry/backoff in fetch_kalshi_weather_markets."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from weatheredge.data.kalshi_client import _get_with_retry


class _FakeResp:
    def __init__(self, status: int, payload: dict | None = None):
        self.status = status
        self._payload = payload or {}
        self.request_info = MagicMock()
        self.history = ()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def json(self):
        return self._payload


def _session_yielding(*responses_or_excs):
    """Build a mock aiohttp ClientSession whose .get() yields the given results.

    Each item is either a _FakeResp instance (success path) or an Exception
    instance (raised when entering the context manager).
    """
    calls = iter(responses_or_excs)

    def _get(_url, params=None):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    session = MagicMock(spec=aiohttp.ClientSession)
    session.get = MagicMock(side_effect=_get)
    return session


@pytest.mark.asyncio
async def test_get_with_retry_returns_payload_on_first_success():
    session = _session_yielding(_FakeResp(200, {"markets": ["a", "b"]}))

    out = await _get_with_retry(session, "https://k/markets", {"x": 1})

    assert out == {"markets": ["a", "b"]}
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_get_with_retry_retries_on_5xx_then_succeeds():
    with patch("weatheredge.data.kalshi_client.asyncio.sleep", new=AsyncMock()) as sleep:
        session = _session_yielding(
            _FakeResp(503),
            _FakeResp(503),
            _FakeResp(200, {"markets": []}),
        )

        out = await _get_with_retry(session, "https://k/markets")

        assert out == {"markets": []}
        assert session.get.call_count == 3
        # 2 sleeps between 3 attempts.
        assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_get_with_retry_returns_none_after_exhausted_retries():
    with patch("weatheredge.data.kalshi_client.asyncio.sleep", new=AsyncMock()):
        session = _session_yielding(
            _FakeResp(500), _FakeResp(500), _FakeResp(500),
        )

        out = await _get_with_retry(session, "https://k/markets")

        assert out is None
        assert session.get.call_count == 3


@pytest.mark.asyncio
async def test_get_with_retry_does_not_retry_on_4xx():
    with patch("weatheredge.data.kalshi_client.asyncio.sleep", new=AsyncMock()) as sleep:
        session = _session_yielding(_FakeResp(404))

        out = await _get_with_retry(session, "https://k/markets")

        assert out is None
        assert session.get.call_count == 1
        assert sleep.await_count == 0


@pytest.mark.asyncio
async def test_get_with_retry_retries_on_network_error():
    with patch("weatheredge.data.kalshi_client.asyncio.sleep", new=AsyncMock()):
        session = _session_yielding(
            aiohttp.ClientError("conn reset"),
            _FakeResp(200, {"markets": []}),
        )

        out = await _get_with_retry(session, "https://k/markets")

        assert out == {"markets": []}
        assert session.get.call_count == 2
