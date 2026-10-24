"""Tests for the classical details the Spotify provider reads from its albums and tracks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ArtistRole, MediaType
from music_assistant_models.errors import MediaNotFoundError, ResourceTemporarilyUnavailable
from music_assistant_models.media_items import Album, ProviderMapping, Track, Work

from music_assistant.constants import CLASSICAL_GENRE_NAME
from music_assistant.mass import MusicAssistant
from music_assistant.providers.spotify.constants import CONF_CLASSICAL_REREAD_DONE
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
    assert not album.classical_tag
    return CLASSICAL_GENRE_NAME in (album.metadata.genres or ())


def test_album_with_mostly_work_titles_is_classical() -> None:
    """An album is classical when most of its track titles have the "Work: Piece" form."""
    assert _is_classical_album(_album_obj(TROUT_QUINTET))


def test_one_work_title_does_not_make_an_album_classical() -> None:
    """A single colon in the titles of a pop album leaves it out of the Classical genre."""
    titles = (*FIFTY_SECOND_STREET, "Interlude: Reprise")
    assert not _is_classical_album(_album_obj(titles))


def test_album_of_chapters_is_not_classical() -> None:
    """An audiobook listed as a music album stays out of the Classical genre."""
    assert not _is_classical_album(_album_obj(AUDIOBOOK))


def test_album_without_its_tracks_is_not_tagged() -> None:
    """An album listed without its tracks, as on a track, says nothing about being classical."""
    assert not _is_classical_album(_album_obj(None))


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


def test_numbered_movement_and_catalog_number() -> None:
    """A numbered movement gives its number, the work its name with the catalog number."""
    track = parse_track(
        _track_obj(
            "Elgar: Cello Concerto in E Minor, Op. 85: I. Adagio - Moderato",
            ("Edward Elgar", "Jacqueline du Pré"),
        ),
        _make_provider(),
    )

    assert track.movement_number == 1
    assert track.movement_name == "I. Adagio - Moderato"
    assert isinstance(track.work, Work)
    assert track.work.name == "Cello Concerto in E Minor, Op. 85"


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


def test_track_without_work_title_gets_no_classical_details() -> None:
    """A track without a "Work: Piece" title gets no work and no composer on its own."""
    track = parse_track(_track_obj("Big Shot", ("Billy Joel",)), _make_provider())

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


async def test_album_tracks_of_a_pop_album_credit_no_composer() -> None:
    """On a pop album no track credits its first artist as composer, not even a colon title."""
    provider = _make_provider()
    titles = (*FIFTY_SECOND_STREET, "Interlude: Reprise")
    provider._get_all_items = MagicMock(  # type: ignore[method-assign]
        return_value=_iterate([_track_obj(title, ("Billy Joel",)) for title in titles])
    )

    tracks = await SpotifyProvider.get_album_tracks.__wrapped__(provider, "album-1")  # type: ignore[attr-defined]

    assert [x.name for x in tracks if x.credits or x.work] == []


async def test_album_tracks_of_an_audiobook_credit_no_composer() -> None:
    """The chapters of an audiobook listed as a music album get no work and no composer."""
    provider = _make_provider()
    provider._get_all_items = MagicMock(  # type: ignore[method-assign]
        return_value=_iterate([_track_obj(title, ("Rufus Beck",)) for title in AUDIOBOOK])
    )

    tracks = await SpotifyProvider.get_album_tracks.__wrapped__(provider, "album-1")  # type: ignore[attr-defined]

    assert [x.name for x in tracks if x.credits or x.work] == []


async def test_reread_updates_library_album_and_its_classical_tracks() -> None:
    """The one-off re-read tags a library album and updates its library classical tracks."""
    mass = _library_mass([{"item_id": 7, "provider_item_id": "album-1"}], MagicMock(item_id="3"))
    provider = _make_provider(mass)
    provider._get_data = AsyncMock(return_value=_album_obj(TROUT_QUINTET))  # type: ignore[method-assign]
    provider._update_config_value = MagicMock()  # type: ignore[method-assign]

    await provider._reread_library_albums()

    [(album_args, _)] = mass.music.albums.update_item_in_library.await_args_list
    assert album_args[0] == 7
    mass.music.genres.sync_media_item_genres.assert_awaited_once_with(
        MediaType.ALBUM, 7, {CLASSICAL_GENRE_NAME}
    )
    updated = [args[1] for args, _ in mass.music.tracks.update_item_in_library.await_args_list]
    assert len(updated) == len(TROUT_QUINTET)
    assert all(_composer_names(x) == [SCHUBERT] for x in updated)
    provider._update_config_value.assert_called_once_with(
        CONF_CLASSICAL_REREAD_DONE, True, immediate=True
    )


async def test_reread_reads_every_track_of_a_long_album() -> None:
    """An album with more tracks than its first page is read in full."""
    mass = _library_mass([{"item_id": 7, "provider_item_id": "album-1"}], MagicMock(item_id="3"))
    provider = _make_provider(mass)
    provider._get_data = AsyncMock(  # type: ignore[method-assign]
        return_value=_album_obj(TROUT_QUINTET[:2], more_tracks=True)
    )
    get_all_items = MagicMock(return_value=_iterate([_track_obj(x) for x in TROUT_QUINTET]))
    provider._get_all_items = get_all_items  # type: ignore[method-assign]
    provider._update_config_value = MagicMock()  # type: ignore[method-assign]

    await provider._reread_library_albums()

    get_all_items.assert_called_once_with("albums/album-1/tracks")
    assert mass.music.tracks.update_item_in_library.await_count == len(TROUT_QUINTET)


async def test_reread_leaves_pop_albums_and_tracks_outside_the_library_alone() -> None:
    """A pop album is not updated, nor is a classical track the library does not hold."""
    rows = [
        {"item_id": 7, "provider_item_id": "pop"},
        {"item_id": 8, "provider_item_id": "classical"},
    ]
    mass = _library_mass(rows, None)
    provider = _make_provider(mass)
    provider._get_data = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_album_obj(FIFTY_SECOND_STREET), _album_obj(TROUT_QUINTET)]
    )
    update_config_value = MagicMock()
    provider._update_config_value = update_config_value  # type: ignore[method-assign]

    await provider._reread_library_albums()

    assert [args[0] for args, _ in mass.music.albums.update_item_in_library.await_args_list] == [8]
    mass.music.tracks.update_item_in_library.assert_not_awaited()
    update_config_value.assert_called_once()


async def test_reread_skips_an_album_spotify_no_longer_has() -> None:
    """An album Spotify no longer has is skipped and the re-read still completes."""
    rows = [
        {"item_id": 7, "provider_item_id": "gone"},
        {"item_id": 8, "provider_item_id": "album-1"},
    ]
    mass = _library_mass(rows, None)
    provider = _make_provider(mass)
    provider._get_data = AsyncMock(  # type: ignore[method-assign]
        side_effect=[MediaNotFoundError("gone"), _album_obj(TROUT_QUINTET)]
    )
    update_config_value = MagicMock()
    provider._update_config_value = update_config_value  # type: ignore[method-assign]

    await provider._reread_library_albums()

    assert mass.music.albums.update_item_in_library.await_count == 1
    update_config_value.assert_called_once()


@pytest.mark.parametrize(
    "error", [ResourceTemporarilyUnavailable("busy"), RuntimeError("unexpected")]
)
async def test_failed_reread_runs_again_on_the_next_start(error: Exception) -> None:
    """A re-read that stops on an error never raises and is not marked as done."""
    provider = _make_provider(_library_mass([{"item_id": 7, "provider_item_id": "album-1"}], None))
    provider._get_data = AsyncMock(side_effect=error)  # type: ignore[method-assign]
    update_config_value = MagicMock()
    provider._update_config_value = update_config_value  # type: ignore[method-assign]

    await provider._reread_library_albums()

    update_config_value.assert_not_called()


@pytest.mark.parametrize(("done", "started"), [(False, True), (True, False)])
async def test_reread_starts_only_until_it_is_done(done: bool, started: bool) -> None:
    """The re-read starts once loaded, until a completed run marked it as done."""
    mass = MagicMock()
    mass.config.get_raw_provider_config_value = MagicMock(return_value=done)
    provider = _make_provider(mass)
    provider._reread_library_albums = MagicMock()  # type: ignore[method-assign]

    await provider.loaded_in_mass()

    assert mass.create_task.called == started


# TEMP(classical 5a tag clear): remove before release
async def test_reread_clears_the_tag_of_albums_only_spotify_provides(
    music_mass_module: MusicAssistant,
) -> None:
    """A tagged album only Spotify provides loses its tag for the genre, one with files keeps it."""
    mass = music_mass_module
    spotify_only = await _add_tagged_album(mass, "album-1")
    with_files = await _add_tagged_album(
        mass,
        "album-2",
        ProviderMapping(
            item_id="/music/album-2", provider_domain="filesystem_local", provider_instance="files"
        ),
    )
    provider = _make_provider(mass)  # type: ignore[arg-type]
    provider._get_data = AsyncMock(return_value=_album_obj(TROUT_QUINTET))  # type: ignore[method-assign]
    provider._update_config_value = MagicMock()  # type: ignore[method-assign]

    await provider._reread_library_albums()

    cleared = await mass.music.albums.get_library_item(spotify_only.item_id)
    kept = await mass.music.albums.get_library_item(with_files.item_id)
    genres = await mass.music.genres.get_genres_for_media_item(
        MediaType.ALBUM, spotify_only.item_id
    )
    assert not cleared.classical_tag
    assert cleared.is_classical
    assert "classical" in {x.translation_key for x in genres}
    assert kept.classical_tag
    provider._update_config_value.assert_called_once()


# TEMP(classical 5a tag clear): remove before release
async def _add_tagged_album(
    mass: MusicAssistant, spotify_id: str, *other_mappings: ProviderMapping
) -> Album:
    """
    Store a library album 5a tagged classical, provided by Spotify and the given mappings.

    :param spotify_id: The Spotify id of the album.
    :param other_mappings: The album's mappings to other providers.
    """
    return await mass.music.albums.add_item_to_library(
        Album(
            item_id="0",
            provider="library",
            name=f"Schubert: Trout Quintet {spotify_id}",
            provider_mappings={
                ProviderMapping(
                    item_id=spotify_id, provider_domain="spotify", provider_instance=INSTANCE_ID
                ),
                *other_mappings,
            },
            classical_tag=True,
        )
    )


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
    # TEMP(classical 5a tag clear)
    mass.music.albums.get_library_item = AsyncMock(return_value=MagicMock(classical_tag=False))
    mass.music.tracks.get_library_item_by_prov_id = AsyncMock(return_value=library_track)
    mass.music.tracks.update_item_in_library = AsyncMock()
    return mass


async def _iterate(items: list[dict[str, Any]]) -> AsyncGenerator[dict[str, Any]]:
    """Yield the given items, as a paged Spotify listing does."""
    for item in items:
        yield item
