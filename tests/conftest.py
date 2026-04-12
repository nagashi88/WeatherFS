"""Pytest configuration for WeatherEdge tests."""

from __future__ import annotations

import os

import pytest

import weatheredge.config as _cfg


@pytest.fixture(autouse=True)
def _enable_all_modules_for_tests():
    """Enable all MODULES flags during tests so existing tests verify behavior.

    PRD v3.1 gates are an integration concern — unit tests should exercise
    the underlying logic regardless of which phase is active.
    """
    original = dict(_cfg.MODULES)
    for key in _cfg.MODULES:
        _cfg.MODULES[key] = True
    yield
    _cfg.MODULES.update(original)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI switch for opt-in live API tests."""
    parser.addoption(
        "--run-live-tests",
        action="store_true",
        default=False,
        help="Run tests marked 'live_api' (also enabled with RUN_LIVE_TESTS=1).",
    )


def _live_tests_enabled(config: pytest.Config) -> bool:
    env_enabled = os.getenv("RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes"}
    return bool(config.getoption("--run-live-tests") or env_enabled)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live API tests unless explicitly enabled."""
    if _live_tests_enabled(config):
        return

    skip_live = pytest.mark.skip(
        reason="live API tests disabled (set RUN_LIVE_TESTS=1 or pass --run-live-tests)"
    )
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip_live)
