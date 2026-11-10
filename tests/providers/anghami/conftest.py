"""Shared fixtures for Anghami provider tests."""

from __future__ import annotations

import asyncio
import json
import pathlib
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from yarl import URL

from music_assistant.helpers.throttle_retry import ThrottlerManager
from music_assistant.providers.anghami.api import AnghamiClient
from music_assistant.providers.anghami.constants import CONF_REFRESH_TOKEN, SUPPORTED_FEATURES
from music_assistant.providers.anghami.provider import AnghamiProvider

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture shaped after the Anghami OpenAPI spec."""
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


class FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(
        self,
        status: int,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        path: str = "/v1/test",
    ) -> None:
        """Initialize the response."""
        self.status = status
        self.headers = headers or {}
        self.url = URL(f"https://sdk.anghami.com{path}")
        self._body = body or {}

    async def json(self) -> dict[str, Any]:
        """Return the JSON body."""
        return self._body

    async def text(self) -> str:
        """Return the body as text."""
        return json.dumps(self._body)


class FakeSession:
    """Minimal stand-in for an aiohttp session that answers with queued responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        """Initialize with the responses to return, in order."""
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    @asynccontextmanager
    async def request(self, method: str, url: str, **kwargs: Any) -> AsyncGenerator[FakeResponse]:
        """Record the request and return the next queued response."""
        self.calls.append((method, url, kwargs))
        yield self.responses.pop(0)

    def get(self, url: str, **kwargs: Any) -> Any:
        """Record a GET request and return the next queued response."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        """Record a POST request and return the next queued response."""
        return self.request("POST", url, **kwargs)


def _create_task(coro: Coroutine[Any, Any, Any], **_kwargs: Any) -> asyncio.Future[Any]:
    """Run a coroutine handed to mass.create_task as a real task."""
    return asyncio.ensure_future(coro)


@pytest.fixture
def mass() -> Mock:
    """Create a mocked Music Assistant with an empty cache."""
    mass = Mock()
    mass.cache.get_with_freshness = AsyncMock(return_value=(None, False, False))
    mass.cache.set = AsyncMock()
    mass.create_task.side_effect = _create_task
    mass.metadata.preferred_language = "en"
    mass.config.get.return_value = {CONF_REFRESH_TOKEN: "refresh-1"}
    mass.config.decrypt_string.side_effect = lambda value: value
    return mass


@pytest.fixture
def provider(mass: Mock) -> AnghamiProvider:
    """Create a real AnghamiProvider on the mocked Music Assistant."""
    manifest = Mock()
    manifest.domain = "anghami"
    config = Mock()
    config.instance_id = "anghami--test"
    config.name = "Anghami Test"
    config.enabled = True
    config.get_value.side_effect = lambda key, default=None: {"log_level": "GLOBAL"}.get(
        key, default
    )
    provider = AnghamiProvider(mass, manifest, config, SUPPORTED_FEATURES)
    provider.client = AnghamiClient(provider)
    provider.client.throttler = ThrottlerManager(rate_limit=1000, retry_attempts=1)
    return provider


@pytest.fixture(autouse=True)
def app_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the Anghami client credentials as app vars."""
    monkeypatch.setenv("MASS_APP_VAR_ANGHAMI_CLIENT_ID", "client-id")
    monkeypatch.setenv("MASS_APP_VAR_ANGHAMI_CLIENT_SECRET", "client-secret")
