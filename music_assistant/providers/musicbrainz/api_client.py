"""MusicBrainz API client."""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from music_assistant_models.errors import RateLimited, ResourceTemporarilyUnavailable

from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.json import json_loads
from music_assistant.helpers.throttle_retry import (
    ThrottlerManager,
    parse_retry_after,
    throttle_with_retries,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from music_assistant.mass import MusicAssistant

MB_BASE_URL = "https://musicbrainz-mirror.music-assistant.io/ws/2"

# the task reading MusicBrainz fresh and the answers it read past the cache so far, None outside
_FRESH_READS: ContextVar[tuple[asyncio.Task[Any] | None, dict[str, Any]] | None] = ContextVar(
    "musicbrainz_fresh_reads", default=None
)


@asynccontextmanager
async def fresh_reads() -> AsyncGenerator[None]:
    """
    Read every MusicBrainz lookup made inside past the cache, each distinct one only once.

    The answers are cached as usual. Nested use shares the answers of the outer one. Only the
    task entering it reads fresh: tasks it starts, such as a queued metadata update, use the
    cache.
    """
    task = asyncio.current_task()
    if (state := _FRESH_READS.get()) is not None and state[0] is task:
        yield
        return
    token = _FRESH_READS.set((task, {}))
    try:
        yield
    finally:
        _FRESH_READS.reset(token)


def _fresh_when_asked(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorate a cached lookup so that inside :func:`fresh_reads` it is read past the cache once."""

    @functools.wraps(func)
    async def wrapper(self: MusicBrainzAPIClient, /, endpoint: str, **kwargs: str) -> Any:
        if (state := _FRESH_READS.get()) is None or state[0] is not asyncio.current_task():
            return await func(self, endpoint, **kwargs)
        fresh = state[1]
        key = "&".join([endpoint, *(f"{k}={v}" for k, v in sorted(kwargs.items()))])
        if key not in fresh:
            async with self.mass.cache.handle_refresh(True):
                fresh[key] = await func(self, endpoint, **kwargs)
        # the caller may change what it gets, as it would a copy read from the cache
        return deepcopy(fresh[key])

    return wrapper


class MusicBrainzAPIClient:
    """Thin HTTP client for the MusicBrainz API."""

    domain = "musicbrainz"
    throttler = ThrottlerManager(rate_limit=30, period=10)

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the API client."""
        self.mass = mass
        self.logger = logging.getLogger(__name__)

    @_fresh_when_asked
    @use_cache(86400 * 30)  # Cache for 30 days
    @throttle_with_retries
    async def get_data(self, /, endpoint: str, **kwargs: str) -> Any:
        """
        Fetch data from the MusicBrainz API.

        Results are cached for 30 days, and read past the cache inside :func:`fresh_reads`.

        :param endpoint: API endpoint path (e.g. ``artist/123?inc=aliases``).
        :param kwargs: Additional query parameters forwarded as URL params.
        """
        return await self._request(endpoint, **kwargs)

    @use_cache(86400 * 7)  # Cache for 7 days
    @throttle_with_retries
    async def get_browse_data(self, /, endpoint: str, **kwargs: str) -> Any:
        """
        Fetch a listing from the MusicBrainz API that grows as the catalog does.

        Results are cached for 7 days, so a discography picks up new releases within a week.

        :param endpoint: API endpoint path (e.g. ``release-group``).
        :param kwargs: Additional query parameters forwarded as URL params.
        """
        return await self._request(endpoint, **kwargs)

    async def _request(self, endpoint: str, **params: str) -> Any:
        """Perform one request against the MusicBrainz API and return its JSON body."""
        url = f"{MB_BASE_URL}/{endpoint}"
        headers = {
            "User-Agent": f"Music Assistant/{self.mass.version} (https://music-assistant.io)"
        }
        params["fmt"] = "json"
        async with self.mass.http_session.get(url, headers=headers, params=params) as response:
            # handle rate limiter
            if response.status == 429:
                backoff_time = parse_retry_after(response.headers.get("Retry-After"))
                raise RateLimited("Rate Limiter", backoff_time=backoff_time)
            # handle temporary server error
            if response.status in (502, 503):
                raise ResourceTemporarilyUnavailable(backoff_time=30)
            # handle 404 not found
            if response.status in (400, 401, 404):
                return None
            response.raise_for_status()
            return await response.json(loads=json_loads)
