"""Tests for reading MusicBrainz past its cache during a refresh."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

from music_assistant.providers.musicbrainz.api_client import MusicBrainzAPIClient, fresh_reads

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

ENDPOINT = "work/3f62d0d8-8ccd-31ab-bfca-768a1ae5708d?inc=aliases"


@pytest.fixture(name="mass")
async def mass_fixture(mass_minimal: MusicAssistant) -> MusicAssistant:
    """Return a minimal instance with its cache database set up."""
    await mass_minimal.cache.setup(await mass_minimal.config.get_core_config("cache"))
    return mass_minimal


async def _settle() -> None:
    """Let the background tasks that store answers in the cache database finish."""
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.wait(pending, timeout=2)


async def test_lookups_are_cached_outside_a_refresh(mass: MusicAssistant) -> None:
    """Outside a refresh a lookup asked before is answered from the cache."""
    client = MusicBrainzAPIClient(mass)
    with patch.object(client, "_request", AsyncMock(return_value={"id": "1"})) as request:
        assert await client.get_data(ENDPOINT) == {"id": "1"}
        await _settle()
        assert await client.get_data(ENDPOINT) == {"id": "1"}

    assert request.await_count == 1


async def test_refresh_reads_each_lookup_fresh_once(mass: MusicAssistant) -> None:
    """A refresh reads a cached lookup again, once, and leaves the fresh answer in the cache."""
    client = MusicBrainzAPIClient(mass)
    with patch.object(client, "_request", AsyncMock(return_value={"title": "old"})):
        await client.get_data(ENDPOINT)
        await _settle()

    with patch.object(client, "_request", AsyncMock(return_value={"title": "new"})) as request:
        async with fresh_reads():
            first = await client.get_data(ENDPOINT)
            # a nested refresh shares the answers of the one it runs in
            async with fresh_reads():
                again = await client.get_data(ENDPOINT)
            # the caller may change what it gets without changing the next answer
            first["title"] = "changed"
            last = await client.get_data(ENDPOINT)
        await _settle()
        after = await client.get_data(ENDPOINT)

    assert again == last == after == {"title": "new"}
    assert request.await_count == 1


async def test_a_refresh_reads_other_query_parameters_apart(mass: MusicAssistant) -> None:
    """Lookups of the same endpoint with other query parameters are read on their own."""
    client = MusicBrainzAPIClient(mass)
    with patch.object(client, "_request", AsyncMock(return_value=None)) as request:
        async with fresh_reads():
            await client.get_data("release", query="a", limit="1")
            await client.get_data("release", limit="1", query="a")
            await client.get_data("release", query="b", limit="1")

    assert request.await_count == 2


async def test_tasks_started_in_a_refresh_read_through_the_cache(mass: MusicAssistant) -> None:
    """A task a refresh starts, such as a queued metadata update, does not read fresh."""
    client = MusicBrainzAPIClient(mass)
    with patch.object(client, "_request", AsyncMock(return_value={"title": "old"})):
        await client.get_data(ENDPOINT)
        await _settle()

    with patch.object(client, "_request", AsyncMock(return_value={"title": "new"})) as request:
        async with fresh_reads():

            async def _queued() -> Any:
                return await client.get_data(ENDPOINT)

            queued = await asyncio.create_task(_queued())

    assert queued == {"title": "old"}
    request.assert_not_awaited()
