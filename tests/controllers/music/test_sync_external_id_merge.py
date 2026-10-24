"""Tests that the library sync stores newly reported ISRCs and merges the duplicates they reveal."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ExternalID, ProviderType
from music_assistant_models.media_items import ProviderMapping, Track, UniqueList

from music_assistant.constants import CONF_LOG_LEVEL, DB_TABLE_TRACKS
from music_assistant.controllers.music import MusicController
from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.test import (
    CONF_KEY_NUM_ALBUMS,
    CONF_KEY_NUM_ARTISTS,
    CONF_KEY_NUM_TRACKS,
)
from music_assistant.providers.test import TestProvider as FakeMusicProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant_models.enums import MediaType

    from music_assistant.mass import MusicAssistant

ISRC = "USRC17607839"
TRACK_ID = "0_0_0"


@pytest.fixture(autouse=True)
def strict_sync_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test when the sync loop swallows a per-item failure."""

    def _raise(
        self: MusicProvider, media_type: MediaType, item_ref: str | None, err: Exception
    ) -> None:
        del self, media_type
        raise AssertionError(f"sync swallowed a failure for {item_ref}: {err!r}")

    monkeypatch.setattr(MusicProvider, "_handle_sync_item_failure", _raise)


@pytest.fixture
async def music(mass_minimal: MusicAssistant) -> AsyncGenerator[MusicController]:
    """Return a music controller with initialized database on the minimal mass instance."""
    mass_minimal.metadata = MagicMock()
    mass_minimal.metadata.invalidate_image_cache = AsyncMock()
    mass_minimal.metadata.set_composer_periods = AsyncMock()
    controller = MusicController(mass_minimal)
    mass_minimal.music = controller
    await controller._setup_database()
    yield controller
    if controller._database:
        await controller._database.close()


@pytest.fixture
def provider(mass_minimal: MusicAssistant, monkeypatch: pytest.MonkeyPatch) -> FakeMusicProvider:
    """Return a fake music provider whose saved track reports an ISRC its album listing lacks."""
    manifest = MagicMock()
    manifest.type = ProviderType.MUSIC
    manifest.domain = "test"
    config = MagicMock()
    config.instance_id = "test--1"
    config.domain = "test"
    values = {
        CONF_KEY_NUM_ARTISTS: 1,
        CONF_KEY_NUM_ALBUMS: 1,
        CONF_KEY_NUM_TRACKS: 1,
        CONF_LOG_LEVEL: "GLOBAL",
    }
    config.get_value.side_effect = lambda key, default=None: values.get(key, default)
    prov = FakeMusicProvider(mass_minimal, manifest, config)

    async def get_library_tracks() -> AsyncGenerator[Track]:
        track = await prov.get_track(TRACK_ID)
        track.external_ids.add((ExternalID.ISRC, ISRC))
        yield track

    monkeypatch.setattr(prov, "get_library_tracks", get_library_tracks)
    return prov


async def _add_other_provider_track(
    music: MusicController, provider: FakeMusicProvider, duration: int
) -> Track:
    """Add the same recording from another provider, under another title and with an ISRC."""
    track = await provider.get_track(TRACK_ID)
    return await music.tracks.add_item_to_library(
        Track(
            item_id="other_track",
            provider="other--1",
            name="Another Title For The Same Recording",
            duration=duration,
            artists=UniqueList(track.artists),
            external_ids={(ExternalID.ISRC, ISRC)},
            provider_mappings={
                ProviderMapping(
                    item_id="other_track",
                    provider_domain="other",
                    provider_instance="other--1",
                    in_library=True,
                )
            },
        )
    )


async def _library_track_ids(music: MusicController) -> list[int]:
    """Return the ids of all library tracks."""
    rows = await music.database.get_rows_from_query(f"SELECT item_id FROM {DB_TABLE_TRACKS}")
    return sorted(row["item_id"] for row in rows)


async def test_sync_merges_duplicate_revealed_by_new_isrc(
    music: MusicController, provider: FakeMusicProvider
) -> None:
    """
    A track imported without an ISRC is merged into its twin once its sync reports the ISRC.

    The album listing carries no ISRC and the titles differ, so the import adds a second
    library track; the saved tracks sync then reports the ISRC the other provider's track holds.
    """
    other = await _add_other_provider_track(music, provider, duration=60)
    await provider.import_album_tracks("0_0")
    assert len(await _library_track_ids(music)) == 2

    await provider._sync_library_tracks()

    assert await _library_track_ids(music) == [int(other.item_id)]
    merged = await music.tracks.get_library_item(other.item_id)
    assert {x.provider_instance for x in merged.provider_mappings} == {"other--1", "test--1"}


async def test_sync_keeps_isrc_twin_with_other_duration_apart(
    music: MusicController, provider: FakeMusicProvider
) -> None:
    """A track sharing the ISRC at a clearly different duration is stored but not merged."""
    await _add_other_provider_track(music, provider, duration=300)
    await provider.import_album_tracks("0_0")

    await provider._sync_library_tracks()

    assert len(await _library_track_ids(music)) == 2
    track = await music.tracks.get_library_item_by_prov_id(TRACK_ID, provider.instance_id)
    assert track is not None
    assert (ExternalID.ISRC, ISRC) in track.external_ids


async def test_sync_leaves_track_with_known_isrc_alone(
    music: MusicController,
    provider: FakeMusicProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A track that already holds the reported ISRC is not rewritten on every sync."""
    await provider.import_album_tracks("0_0")
    await provider._sync_library_tracks()
    updates = AsyncMock(wraps=music.tracks.update_item_in_library)
    monkeypatch.setattr(music.tracks, "update_item_in_library", updates)

    await provider._sync_library_tracks()

    updates.assert_not_awaited()
