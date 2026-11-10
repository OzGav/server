"""Tests for the Anghami parsers."""

from __future__ import annotations

from music_assistant_models.enums import AlbumType, ExternalID, MediaType
from music_assistant_models.media_items import ItemMapping

from music_assistant.providers.anghami.parsers import (
    content_ref,
    parse_album,
    parse_artist,
    parse_lyrics,
    parse_playlist,
    parse_track,
)
from music_assistant.providers.anghami.provider import AnghamiProvider

from .conftest import load_fixture


def test_parse_track(provider: AnghamiProvider) -> None:
    """A song maps to a track with its artists, album, ids and the widest artwork."""
    album = parse_album(provider, load_fixture("album.json")["album"])
    track = parse_track(provider, load_fixture("song.json")["song"], album)

    assert track.item_id == "s1"
    assert track.provider == "anghami--test"
    assert track.name == "Habibi"
    assert track.duration == 245
    assert (track.disc_number, track.track_number) == (1, 3)
    assert track.album is album
    assert track.artists == [
        ItemMapping(
            media_type=MediaType.ARTIST, item_id="ar1", provider="anghami--test", name="Fairuz"
        )
    ]
    assert (ExternalID.ISRC, "LBA011200001") in track.external_ids
    assert track.metadata.explicit is False
    assert track.metadata.images is not None
    assert [image.path for image in track.metadata.images] == ["https://img.anghami.com/s1-640.jpg"]


def test_parse_album(provider: AnghamiProvider) -> None:
    """An album maps its type, release year, barcode and label."""
    album = parse_album(provider, load_fixture("album.json")["album"])

    assert album.name == "Best of Fairuz"
    assert album.album_type == AlbumType.COMPILATION
    assert album.year == 1972
    assert (ExternalID.BARCODE, "0724383628027") in album.external_ids
    assert album.metadata.label == "Voix de l'Orient"
    assert [artist.item_id for artist in album.artists] == ["ar1"]


def test_parse_album_bad_release_date(provider: AnghamiProvider) -> None:
    """An album with an unreadable release date still parses, without a year."""
    album = parse_album(provider, {**load_fixture("album.json")["album"], "releaseDate": "n/a"})

    assert album.year is None
    assert album.metadata.release_date is None


def test_parse_artist(provider: AnghamiProvider) -> None:
    """An artist maps its localized name, biography and artwork."""
    artist = parse_artist(provider, load_fixture("artist.json")["artist"])

    assert artist.name == "Fairuz"
    assert artist.metadata.description == "Lebanese singer."
    assert artist.image is not None


def test_parse_playlist(provider: AnghamiProvider) -> None:
    """A playlist maps its owner and cover and is never editable."""
    playlist = parse_playlist(provider, load_fixture("playlist.json")["playlist"])

    assert playlist.name == "Morning Coffee"
    assert playlist.owner == "Anghami"
    assert playlist.is_editable is False
    assert playlist.image is not None
    assert playlist.image.path == "https://img.anghami.com/p1.jpg"


def test_parse_lyrics_synced() -> None:
    """Synced lyrics give both plain text and LRC lyrics."""
    plain, lrc = parse_lyrics(load_fixture("lyrics_synced.json"))

    assert plain == "First line\nSecond line"
    assert lrc == "[00:01.50]First line\n[01:05.25]Second line"


def test_parse_lyrics_plain_and_missing() -> None:
    """Plain lyrics give no LRC lyrics, and an empty response gives none at all."""
    assert parse_lyrics({"lyrics": {"lines": ["One", "Two"]}}) == ("One\nTwo", None)
    assert parse_lyrics({}) == (None, None)


def test_content_ref() -> None:
    """Music content resolves to its media type and id, video content to nothing."""
    assert content_ref({"albumId": {"value": "a1"}, "title": "x"}) == (MediaType.ALBUM, "a1")
    assert content_ref({"movieId": {"value": "m1"}, "title": "x"}) is None
