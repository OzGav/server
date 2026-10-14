"""Tests for the classical details the Spotify provider reads from its albums and tracks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ArtistRole
from music_assistant_models.media_items import Track, Work

from music_assistant.constants import CLASSICAL_GENRE_NAME
from music_assistant.providers.spotify.parsers import parse_album, parse_track
from music_assistant.providers.spotify.provider import SpotifyProvider

INSTANCE_ID = "spotify--test"
SCHUBERT = "Schubert"
PERFORMERS = ("Boris Berezovsky", "Vasko Vassiliev")
# real Spotify track titles of a classical album and of a pop album
TROUT_QUINTET = (
    "Trout Quintet: Allegro vivace",
    "Trout Quintet: Andante",
    "Trout Quintet: Scherzo",
    "Trout Quintet: Theme and variations",
    "Trout Quintet: Finale",
    "Adagio and Rondo Concertante: Adagio",
    "Adagio and Rondo Concertante: Rondo: Allegro Vivace",
    "Notturno",
)
FIFTY_SECOND_STREET = ("Big Shot", "Honesty", "My Life", "Zanzibar", "Stiletto")
# chapter titles of an audiobook Spotify lists as a music album
AUDIOBOOK = (
    "Kapitel 1: Der Anfang",
    "Kapitel 2: Die Reise",
    "Kapitel 3: Die Rückkehr",
    "Kapitel 4: Op. 1",
)


def _make_provider(mass: MagicMock | None = None) -> SpotifyProvider:
    """Return a Spotify provider with only what parsing and the library re-read need."""
    provider = object.__new__(SpotifyProvider)
    provider.config = MagicMock(instance_id=INSTANCE_ID)
    provider.manifest = MagicMock(domain="spotify")
    provider.logger = MagicMock()
    provider.mass = mass or MagicMock()
    return provider


def _artist_obj(name: str) -> dict[str, Any]:
    """Return a Spotify artist object."""
    artist_id = name.lower().replace(" ", "-")
    return {
        "id": artist_id,
        "name": name,
        "external_urls": {"spotify": f"https://open.spotify.com/artist/{artist_id}"},
    }


def _track_obj(title: str, artists: tuple[str, ...] = (SCHUBERT, *PERFORMERS)) -> dict[str, Any]:
    """Return a Spotify track object as an album's track listing gives it."""
    track_id = f"track-{abs(hash(title))}"
    return {
        "id": track_id,
        "name": title,
        "duration_ms": 180000,
        "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
        "is_local": False,
        "is_playable": True,
        "explicit": False,
        "artists": [_artist_obj(name) for name in artists],
    }


def _album_obj(titles: tuple[str, ...] | None, more_tracks: bool = False) -> dict[str, Any]:
    """Return a Spotify album object, a full one when given the titles of its tracks."""
    album: dict[str, Any] = {
        "id": "album-1",
        "name": "Schubert: Trout Quintet",
        "album_type": "album",
        "external_urls": {"spotify": "https://open.spotify.com/album/album-1"},
        "artists": [_artist_obj(SCHUBERT)],
        "images": [],
    }
    if titles is not None:
        album["tracks"] = {
            "items": [_track_obj(title) for title in titles],
            "next": "https://api.spotify.com/v1/albums/album-1/tracks?offset=50"
            if more_tracks
            else None,
        }
    return album


def _composer_names(track: Track) -> list[str]:
    """Return the names of the artists credited as composer on a track."""
    return [x.artist.name for x in track.credits if x.role == ArtistRole.COMPOSER]


def _is_classical_album(album_obj: dict[str, Any]) -> bool:
    """Return whether a parsed Spotify album is in the Classical genre."""
    album = parse_album(album_obj, _make_provider())
    return CLASSICAL_GENRE_NAME in (album.metadata.genres or ())


@pytest.mark.parametrize(
    ("titles", "expected"),
    [
        (TROUT_QUINTET, True),
        ((*FIFTY_SECOND_STREET, "Interlude: Reprise"), False),
        (AUDIOBOOK, False),
        (None, False),
    ],
    ids=["classical", "pop with one colon", "audiobook", "album without its tracks"],
)
def test_album_is_classical_by_its_work_titles(
    titles: tuple[str, ...] | None, expected: bool
) -> None:
    """
    An album is classical when most of its track titles have the "Work: Piece" form.

    A single colon in the titles of a pop album, the chapters of an audiobook listed as a music
    album, or an album listed without its tracks (as on a track) leave it out of the Classical
    genre.
    """
    assert _is_classical_album(_album_obj(titles)) is expected


def test_work_title_gives_work_movement_and_composer() -> None:
    """The first artist of a "Work: Piece" track is its composer, the rest names the work."""
    track = parse_track(
        _track_obj("La bohème, Act II: Viva Parpignol!", ("Giacomo Puccini", *PERFORMERS)),
        _make_provider(),
    )

    assert [x.name for x in track.artists] == ["Giacomo Puccini", *PERFORMERS]
    assert _composer_names(track) == ["Giacomo Puccini"]
    assert track.movement_name == "Viva Parpignol!"
    assert track.movement_number is None
    assert isinstance(track.work, Work)
    assert track.work.name == "La bohème, Act II"


def test_title_naming_another_composer_credits_no_performer() -> None:
    """A "Composer: Work - 1. Movement" title keeps its performing first artist off the work."""
    track = parse_track(
        _track_obj(
            "Allegri: Miserere - 01. Miserere Mei, Deus", ("The Tallis Scholars", "Peter Phillips")
        ),
        _make_provider(),
        composer_first=True,
    )

    assert _composer_names(track) == []
    assert isinstance(track.work, Work)
    assert track.work.name == "Miserere"
    assert (track.movement_name, track.movement_number) == ("01. Miserere Mei, Deus", 1)


@pytest.mark.parametrize(
    ("title", "artist"),
    [("Suite: Judy Blue Eyes", "Crosby, Stills & Nash"), ("Big Shot", "Billy Joel")],
)
def test_pop_track_gets_no_classical_details(title: str, artist: str) -> None:
    """
    A single track without a "Work: Piece" title gets no work and no composer on its own.

    This holds for a title that only holds a classical form too.
    """
    track = parse_track(_track_obj(title, (artist,)), _make_provider())

    assert track.work is None
    assert track.credits == []


async def test_album_tracks_of_a_classical_album_credit_the_composer() -> None:
    """On a classical album the first artist is the composer, also without a work title."""
    provider = _make_provider()
    provider._get_all_items = MagicMock(  # type: ignore[method-assign]
        return_value=_iterate([_track_obj(title) for title in TROUT_QUINTET])
    )

    tracks = await SpotifyProvider.get_album_tracks.__wrapped__(provider, "album-1")  # type: ignore[attr-defined]

    assert all(_composer_names(x) == [SCHUBERT] for x in tracks)
    [notturno] = [x for x in tracks if x.name == "Notturno"]
    assert notturno.work is None


async def test_album_titles_naming_their_composers_credit_no_performer() -> None:
    """On an album whose titles name the composers, no track credits its first artist."""
    provider = _make_provider()
    titles = (
        "L'homme Armé",
        "Plainchant: Gaudeamus Omnes",
        "Josquin: Missa L'homme Armé Super Voces Musicales - 1a. Kyrie 1",
        "Josquin: Missa L'homme Armé Super Voces Musicales - 1b. Christe",
        "Josquin: Missa L'homme Armé Super Voces Musicales - 1c. Kyrie 2",
    )
    provider._get_all_items = MagicMock(  # type: ignore[method-assign]
        return_value=_iterate(
            [_track_obj(title, ("The Tallis Scholars", "Peter Phillips")) for title in titles]
        )
    )

    tracks = await SpotifyProvider.get_album_tracks.__wrapped__(provider, "album-1")  # type: ignore[attr-defined]

    assert all(_composer_names(x) == [] for x in tracks)
    assert [x.work.name if x.work else None for x in tracks] == [
        None,
        "Plainchant",
        *["Missa L'homme Armé Super Voces Musicales"] * 3,
    ]


@pytest.mark.parametrize(
    ("titles", "artist"),
    [((*FIFTY_SECOND_STREET, "Interlude: Reprise"), "Billy Joel"), (AUDIOBOOK, "Rufus Beck")],
)
async def test_album_tracks_of_a_pop_album_credit_no_composer(
    titles: tuple[str, ...], artist: str
) -> None:
    """
    On a pop album no track gets a work or credits its first artist as composer.

    This holds for a colon title too, and for the chapters of an audiobook listed as a music album.
    """
    provider = _make_provider()
    provider._get_all_items = MagicMock(  # type: ignore[method-assign]
        return_value=_iterate([_track_obj(title, (artist,)) for title in titles])
    )

    tracks = await SpotifyProvider.get_album_tracks.__wrapped__(provider, "album-1")  # type: ignore[attr-defined]

    assert [x.name for x in tracks if x.credits or x.work] == []


def _library_mass(rows: list[dict[str, Any]], library_track: MagicMock | None) -> MagicMock:
    """
    Return a Music Assistant whose library holds the given Spotify albums.

    :param rows: The library id and Spotify id of each album.
    :param library_track: The library track each Spotify track is found as, if any.
    """
    mass = MagicMock()
    mass.music.database.get_rows_from_query = AsyncMock(return_value=rows)
    mass.music.albums.update_item_in_library = AsyncMock(side_effect=lambda _item_id, album: album)
    mass.music.genres.sync_media_item_genres = AsyncMock()
    mass.music.tracks.get_library_item_by_prov_id = AsyncMock(return_value=library_track)
    mass.music.tracks.update_item_in_library = AsyncMock()
    return mass


async def _iterate(items: list[dict[str, Any]]) -> AsyncGenerator[dict[str, Any]]:
    """Yield the given items, as a paged Spotify listing does."""
    for item in items:
        yield item
