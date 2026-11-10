"""Tests for the Anghami provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import (
    MediaNotFoundError,
    RetriesExhausted,
    UnplayableMediaError,
)
from music_assistant_models.media_items import Album, Artist, BrowseFolder, Playlist, Track
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.providers.anghami.provider import AnghamiProvider

from .conftest import load_fixture

SONG = load_fixture("song.json")["song"]
ALBUM = load_fixture("album.json")["album"]
ARTIST = load_fixture("artist.json")["artist"]


def _song(song_id: str) -> dict[str, Any]:
    """Return the song fixture under another id."""
    return {**SONG, "id": {"value": song_id}}


async def _batch_get(kind: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Answer batch gets from the fixtures, leaving out the unknown song s404."""
    catalog = {
        "song": {item_id: _song(item_id) for item_id in ids if item_id != "s404"},
        "album": {"a1": ALBUM},
        "artist": {"ar1": ARTIST},
    }[kind]
    return {item_id: catalog[item_id] for item_id in ids if item_id in catalog}


async def test_resolve_contents_keeps_order(provider: AnghamiProvider) -> None:
    """Mixed content resolves to full items in order, skipping video and unknown ids."""
    contents: list[dict[str, Any]] = [
        {"artistId": {"value": "ar1"}},
        {"songId": {"value": "s1"}},
        {"movieId": {"value": "m1"}},
        {"songId": {"value": "s404"}},
        {"playlistId": {"value": "p1"}, "title": "Morning Coffee", "subtitle": "Anghami"},
        {"albumId": {"value": "a1"}},
    ]
    with patch.object(provider.client, "batch_get", side_effect=_batch_get) as batch_get:
        items = await provider._resolve_contents(contents)

    assert [type(item) for item in items] == [Artist, Track, Playlist, Album]
    assert [item.item_id for item in items] == ["ar1", "s1", "p1", "a1"]
    track = items[1]
    assert isinstance(track, Track)
    assert isinstance(track.album, Album)
    assert track.album.name == "Best of Fairuz"
    # one batch per kind, the albums of the songs ride along with the listed albums
    assert [call.args[0] for call in batch_get.call_args_list] == ["song", "album", "artist"]
    assert batch_get.call_args_list[1].args[1] == ["a1", "a1"]


async def test_search_splits_results_by_type(provider: AnghamiProvider) -> None:
    """One search request covers all asked types and results are split per type."""
    response = {
        "results": [
            {"songId": {"value": "s1"}},
            {"songId": {"value": "s2"}},
            {"artistId": {"value": "ar1"}},
        ]
    }
    with (
        patch.object(provider.client, "get", AsyncMock(return_value=response)) as get,
        patch.object(provider.client, "batch_get", side_effect=_batch_get),
    ):
        result = await provider.search("fairuz", [MediaType.TRACK, MediaType.ARTIST], limit=1)

    assert [track.item_id for track in result.tracks] == ["s1"]
    assert [artist.item_id for artist in result.artists] == ["ar1"]
    assert get.call_args.args[1] == {
        "q": "fairuz",
        "types": ["CONTENT_TYPE_SONG", "CONTENT_TYPE_ARTIST"],
        "page_size": 2,
    }


async def test_playlist_tracks_numbers_every_entry(provider: AnghamiProvider) -> None:
    """All playlist pages load on the first page, a song listed twice gets two positions."""
    first = load_fixture("playlist.json")["items"]
    second = [{"content": {"songId": {"value": "s1"}}, "position": 2}]

    async def paginate(*_args: Any, **_kwargs: Any) -> AsyncGenerator[list[dict[str, Any]]]:
        yield first
        yield second

    with (
        patch.object(provider.client, "paginate", paginate),
        patch.object(provider.client, "batch_get", side_effect=_batch_get),
    ):
        tracks = await provider.get_playlist_tracks("p1")
        later_page = await provider.get_playlist_tracks("p1", page=1)

    assert [(track.item_id, track.position) for track in tracks] == [("s1", 1), ("s1", 2)]
    assert later_page == []


async def test_get_track_survives_missing_lyrics(provider: AnghamiProvider) -> None:
    """A song Anghami has no lyrics for leaves the track without lyrics instead of failing it."""
    get = AsyncMock(side_effect=[{"song": SONG}, MediaNotFoundError("no lyrics")])
    with (
        patch.object(provider.client, "get", get),
        patch.object(provider.client, "batch_get", side_effect=_batch_get),
    ):
        track = await provider.get_track("s1")

    assert track.name == "Habibi"
    assert track.metadata.lyrics is None


async def test_get_track_fails_on_unavailable_lyrics(provider: AnghamiProvider) -> None:
    """A lyrics lookup that can not reach Anghami fails the track, so no empty lyrics are cached."""
    get = AsyncMock(side_effect=[{"song": SONG}, RetriesExhausted("down")])
    with (
        patch.object(provider.client, "get", get),
        patch.object(provider.client, "batch_get", side_effect=_batch_get),
        pytest.raises(RetriesExhausted),
    ):
        await provider.get_track("s1")


async def test_get_track_attaches_lyrics(provider: AnghamiProvider) -> None:
    """Synced lyrics are attached as both plain and LRC lyrics."""
    get = AsyncMock(side_effect=[{"song": SONG}, load_fixture("lyrics_synced.json")])
    with (
        patch.object(provider.client, "get", get),
        patch.object(provider.client, "batch_get", side_effect=_batch_get),
    ):
        track = await provider.get_track("s1")

    assert track.metadata.lyrics == "First line\nSecond line"
    assert track.metadata.lrc_lyrics is not None


async def test_stream_details_hls(provider: AnghamiProvider) -> None:
    """An HLS manifest streams as HLS, expires early and takes the best offered format."""
    stream = load_fixture("stream_hls.json")["stream"]
    with patch.object(provider.client, "acquire_stream", AsyncMock(return_value=stream)):
        details = await provider.get_stream_details("s1", MediaType.TRACK)

    assert details.stream_type == StreamType.HLS
    assert details.path == stream["manifestUrl"]
    assert details.expiration == 840
    assert details.audio_format.content_type == ContentType.AAC
    assert details.audio_format.bit_rate == 320


async def test_stream_details_dash(provider: AnghamiProvider) -> None:
    """A DASH manifest is handed to ffmpeg as a plain HTTP source."""
    stream = {
        "manifestUrl": "https://stream.anghami.com/s1/manifest.mpd",
        "expiresInSeconds": 600,
    }
    with patch.object(provider.client, "acquire_stream", AsyncMock(return_value=stream)):
        details = await provider.get_stream_details("s1", MediaType.TRACK)

    assert details.stream_type == StreamType.HTTP
    assert details.audio_format.content_type == ContentType.UNKNOWN


async def test_stream_details_without_expiry(provider: AnghamiProvider) -> None:
    """A stream without an expiry keeps the default expiry instead of expiring at once."""
    stream = {"manifestUrl": "https://stream.anghami.com/s1/master.m3u8"}
    with patch.object(provider.client, "acquire_stream", AsyncMock(return_value=stream)):
        details = await provider.get_stream_details("s1", MediaType.TRACK)

    default = StreamDetails(provider="x", item_id="x", audio_format=details.audio_format)
    assert details.expiration == default.expiration


async def test_stream_details_refuses_drm(provider: AnghamiProvider) -> None:
    """A DRM protected stream is refused with a clear error."""
    stream = {**load_fixture("stream_hls.json")["stream"], "drmScheme": "DRM_SCHEME_WIDEVINE"}
    with (
        patch.object(provider.client, "acquire_stream", AsyncMock(return_value=stream)),
        pytest.raises(UnplayableMediaError, match="protected streams are not supported"),
    ):
        await provider.get_stream_details("s1", MediaType.TRACK)


async def test_featured_folders_skip_video_sections(provider: AnghamiProvider) -> None:
    """Featured sections without music are not offered."""
    response = {
        "sections": [
            {"id": {"value": "music"}, "title": "Trending", "items": [{"songId": {"value": "s1"}}]},
            {"id": {"value": "video"}, "title": "Series", "items": [{"showId": {"value": "x"}}]},
        ]
    }
    with patch.object(provider.client, "get", AsyncMock(return_value=response)):
        folders = await provider.get_recommendation_items("featured")

    assert [folder.name for folder in folders] == ["Trending"]
    assert isinstance(folders[0], BrowseFolder)
    assert folders[0].path == "anghami--test://featured/music"
