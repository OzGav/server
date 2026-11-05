"""Tests for refreshing a library item from its source and MusicBrainz."""

from __future__ import annotations

from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, call, patch

import pytest
from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    MediaItemMetadata,
    MediaItemType,
    ProviderMapping,
    Track,
    UniqueList,
    Work,
)

from music_assistant.controllers.music import MusicController
from music_assistant.mass import MusicAssistant
from music_assistant.models.music_provider import MusicProvider

RELEASE_ID = "5d9c6e8a-1c3b-4f2e-9a7d-2b8c4e6f0a11"
RECORDING_ID = "0c1d2e3f-4a5b-4c6d-9e8f-7a6b5c4d3e22"
OTHER_ID = "9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c33"
ACT_ID = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c44"
NUMBER_ID = "2b3c4d5e-6f7a-4b8c-9d0e-1f2a3b4c5d55"


def _mapping() -> ProviderMapping:
    """Return the item's streaming service mapping."""
    return ProviderMapping(item_id="abc", provider_domain="spotify", provider_instance="spotify_1")


def _album(provider: str) -> Album:
    """Return an album of the library or of its source."""
    return Album(
        item_id="1" if provider == "library" else "abc",
        provider=provider,
        name="Album",
        provider_mappings={_mapping()},
    )


def _track(provider: str, **kwargs: Any) -> Track:
    """Return a track of the library or of its source."""
    return Track(
        item_id="1" if provider == "library" else "abc",
        provider=provider,
        name="Track",
        provider_mappings={_mapping()},
        **kwargs,
    )


async def _refresh(stored: MediaItemType, source: MediaItemType) -> Mock:
    """Refresh a library item whose source sends the given item, return its Music Assistant."""
    mass = Mock()
    mass.metadata = AsyncMock()
    controller = AsyncMock()
    controller.get_provider_item = AsyncMock(return_value=source)
    # a track, so the album's own tracks are not read again
    controller.update_item_in_library = AsyncMock(
        return_value=Mock(media_type=MediaType.TRACK, provider_mappings=set())
    )
    music = MusicController.__new__(MusicController)
    music.mass = mass
    mass.music = music
    music.works = AsyncMock()
    with (
        patch.object(music, "get_controller", return_value=controller),
        patch.object(music, "_visible_provider_for", return_value=Mock(instance_id="spotify_1")),
        patch("music_assistant.controllers.music.controller.invalidate_listings", AsyncMock()),
    ):
        await music.refresh_item(stored)
    return mass


@pytest.mark.parametrize("media_type", [MediaType.ALBUM, MediaType.TRACK])
async def test_refresh_drops_the_musicbrainz_id_its_source_does_not_send(
    music_mass: MusicAssistant, media_type: MediaType
) -> None:
    """A refreshed album or track is linked to MusicBrainz anew, so a corrected link reaches it."""
    ctrl: Any = music_mass.music.get_controller(media_type)
    stored: Album | Track
    source: Album | Track
    if media_type == MediaType.ALBUM:
        linked = {(ExternalID.MB_ALBUM, RELEASE_ID), (ExternalID.MB_RELEASEGROUP, OTHER_ID)}
        stored, source = _album("filesystem_1"), _album("filesystem_1")
    else:
        linked = {(ExternalID.MB_RECORDING, RECORDING_ID)}
        stored, source = _file_track(), _file_track()
    stored.external_ids = {*linked, (ExternalID.ISRC, "GBSTK0700001")}
    source.external_ids = {(ExternalID.ISRC, "GBSTK0700001")}
    library_item = await ctrl.add_item_to_library(stored)
    metadata = AsyncMock()

    with (
        patch.object(music_mass.music, "_visible_provider_for", return_value=Mock()),
        patch.object(ctrl, "get_provider_item", AsyncMock(return_value=source)),
        patch.object(ctrl, "match_providers", AsyncMock()),
        patch.object(music_mass, "metadata", metadata, create=True),
        patch("music_assistant.controllers.music.controller.invalidate_listings", AsyncMock()),
    ):
        await music_mass.music.refresh_item(library_item)

    refreshed = await ctrl.get_library_item(library_item.item_id)
    assert refreshed.external_ids == {(ExternalID.ISRC, "GBSTK0700001")}
    metadata.update_metadata.assert_awaited_once_with(ANY, force_refresh=True, rebuild=True)


async def test_refresh_of_a_track_removes_the_work_it_left_when_empty() -> None:
    """A refreshed track's former work is removed when the refresh left it without tracks."""
    work = ItemMapping(media_type=MediaType.WORK, item_id="7", provider="library", name="Work")

    mass = await _refresh(_track("library", work=work), _track("spotify"))

    mass.music.works.remove_if_empty.assert_awaited_once_with("7")


async def test_refresh_of_a_work_looks_up_its_whole_tree_again() -> None:
    """Refreshing a work reads its work tree from MusicBrainz again, and nothing else."""
    work = Work(item_id="3", provider="library", name="Work", provider_mappings=set())
    mass = Mock()
    mass.metadata.refresh_work_tree = AsyncMock(return_value=work)
    music = MusicController.__new__(MusicController)
    music.mass = mass

    with patch.object(music, "get_controller", return_value=AsyncMock()) as controller:
        assert await music.refresh_item(work) is work

    mass.metadata.refresh_work_tree.assert_awaited_once_with(work)
    controller.return_value.get_provider_item.assert_not_awaited()


def _provider(domain: str, streaming: bool) -> Mock:
    """Return a music provider of the given domain."""
    provider = Mock(spec=MusicProvider)
    provider.domain = domain
    provider.instance_id = domain
    provider.is_streaming_provider = streaming
    return provider


def _album_track(
    item_id: str, domain: str, classical: bool = True, work_id: str | None = None
) -> Track:
    """Return a library track of the album, read from the given provider domain."""
    return Track(
        item_id=item_id,
        provider="library",
        name=f"Track {item_id}",
        provider_mappings={
            ProviderMapping(item_id=f"p{item_id}", provider_domain=domain, provider_instance=domain)
        },
        is_classical=classical,
        work=ItemMapping(media_type=MediaType.WORK, item_id=work_id, provider="library", name="W")
        if work_id
        else None,
    )


async def test_album_refresh_reads_its_file_tracks_again() -> None:
    """
    A refreshed album reads its file tracks anew and leaves its streaming tracks as they are.

    A file track whose work tag changed leaves its old work, which goes when empty.
    """
    files, spotify = _provider("filesystem", False), _provider("spotify", True)
    file_track = Track(item_id="f1", provider="filesystem", name="Track 1", provider_mappings=set())
    files.get_track = AsyncMock(return_value=file_track)
    album_tracks = [
        _album_track("1", "filesystem", work_id="5"),
        _album_track("2", "spotify", work_id="6"),
        _album_track("3", "filesystem", classical=False),
    ]
    mass = Mock()
    mass.metadata = AsyncMock()
    music = MusicController.__new__(MusicController)
    music.mass = mass
    mass.music = music
    music.works = AsyncMock()
    music.albums = Mock(tracks=AsyncMock(return_value=album_tracks))
    music.tracks = Mock(update_item_in_library=AsyncMock())
    album = _album("library")
    controller = AsyncMock()
    controller.get_provider_item = AsyncMock(return_value=_album("spotify"))
    controller.update_item_in_library = AsyncMock(return_value=album)

    with (
        patch.object(music, "get_controller", return_value=controller),
        patch.object(
            music,
            "_visible_provider_for",
            side_effect=lambda mapping: (
                files if mapping.provider_domain == "filesystem" else spotify
            ),
        ),
        patch("music_assistant.controllers.music.controller.invalidate_listings", AsyncMock()),
    ):
        await music.refresh_item(album)

    spotify.get_track.assert_not_called()
    spotify.get_album_tracks.assert_not_called()
    assert music.tracks.update_item_in_library.await_args_list == [
        call("1", file_track, overwrite=True),
        call("3", file_track, overwrite=False),
    ]
    assert sorted(x.args[0] for x in mass.music.works.remove_if_empty.await_args_list) == [
        "5",
        "6",
    ]


def _file_mapping(item_id: str) -> ProviderMapping:
    """Return a mapping of an item read from a file."""
    return ProviderMapping(
        item_id=item_id, provider_domain="filesystem_local", provider_instance="filesystem_1"
    )


def _file_track(**kwargs: Any) -> Track:
    """Return a classical track as its file gives it."""
    return Track(
        item_id="f1",
        provider="filesystem_1",
        name="No. 2 Scène",
        provider_mappings={_file_mapping("f1")},
        artists=UniqueList(
            [
                Artist(
                    item_id="orchestra",
                    provider="filesystem_1",
                    name="Orchestra",
                    provider_mappings={_file_mapping("orchestra")},
                )
            ]
        ),
        metadata=MediaItemMetadata(classical_tag=True),
        **kwargs,
    )
