"""Tests for the classical details the Qobuz provider reads from its albums and tracks."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import ArtistRole, ExternalID
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Album, Track, Work

from music_assistant.constants import CLASSICAL_GENRE_NAME
from music_assistant.providers.qobuz import (
    PARSED_ITEM_CACHE_CHECKSUM,
    SUPPORTED_FEATURES,
    QobuzProvider,
)

INSTANCE_ID = "qobuz--test"
CLASSICAL_GENRE = {"path": [10], "name": "Klassiek", "id": 10, "slug": "classique"}
CHAMBER_MUSIC_GENRE = {"path": [10, 27], "name": "Kamermuziek", "id": 27}
POP_GENRE = {"path": [112, 119], "name": "Pop/Rock", "id": 119}

# real performers strings of the user's Qobuz classical albums
KLEIBER_PERFORMERS = (
    "Wiener Philharmoniker - Carlos Kleiber, Conductor - Ludwig van Beethoven, Composer - "
    "Werner Mayer, Recording Producer - Wolf-Dieter Karwatky, Recording Engineer - "
    "Hans-Rudolf Müller, Recording Engineer - Jobst Eberhardt, Editor - Hans-Peter "
    "Schweigmann, Editor, Balance Engineer"
)
TOSCA_PERFORMERS = (
    "Herbert Von Karajan, Director, MainArtist - Giacomo Puccini, Composer - Orchester der "
    "Wiener Staatsoper, Orchestra, MainArtist - Ferruccio Mazzoli, Bass Vocals, MainArtist - "
    "Floriana Cavalli, Soprano Vocals, MainArtist - Dimiter Uzunov, Tenor Vocals - "
    "Dimiter Usunow, MainArtist"
)
LAMBERT_PERFORMERS = (
    "Lionel Friend, MainArtist, Conductor - Ian Brown, MainArtist - The Nash Ensemble, "
    "MainArtist, ChamberEnsemble - Constant Lambert, Composer"
)
LI_PO_PERFORMERS = (
    "Philip Langridge, MainArtist, TenorVocalist - The Nash Ensemble, MainArtist, "
    "ChamberEnsemble - Shigeyoshi Obata, Translator - Li-Tai-Po, NonLyricAuthor - "
    "Constant Lambert, Composer"
)
SCHUBERT_PERFORMERS = "Andreas Staier, Piano - Alexander Melnikov, Piano - Franz Schubert, Composer"
RAVEL_PERFORMERS = (
    "Maurice Ravel, Composer - D.R, OriginalPublisher - Jacques Rouvier, Artist, Soloist - "
    "Théodore Paraskivesco, Artist, Soloist - Théodore Paraskivesco, Jacques Rouvier, MainArtist"
)

BEETHOVEN = {"id": 38, "name": "Ludwig van Beethoven"}
KLEIBER = {"id": 22805, "name": "Carlos Kleiber"}
WIENER_PHILHARMONIC = {"id": 1463649, "name": "Wiener Philharmonic Orchestra"}
PUCCINI = {"id": 540, "name": "Giacomo Puccini"}
KARAJAN = {"id": 24128, "name": "Herbert von Karajan"}
STAATSOPER = {"id": 113768, "name": "Orchester der Wiener Staatsoper"}
LAMBERT = {"id": 279443, "name": "Constant Lambert"}
FRIEND = {"id": 27886, "name": "Lionel Friend"}
NASH_ENSEMBLE = {"id": 26897, "name": "The Nash Ensemble"}
LANGRIDGE = {"id": 28038, "name": "Philip Langridge"}
STAIER = {"id": 122809, "name": "Andreas Staier"}
SCHUBERT = {"id": 624, "name": "Franz Schubert"}
RIGHTS_RESERVED = {"id": 54664, "name": "DR"}


def _make_provider(mass: MagicMock | None = None) -> QobuzProvider:
    """Return a Qobuz provider with only what parsing and the library re-read need."""
    manifest = Mock()
    manifest.domain = "qobuz"
    config = Mock()
    config.instance_id = INSTANCE_ID
    config.get_value.side_effect = lambda key, default=None: {"log_level": "GLOBAL"}.get(
        key, default
    )
    return QobuzProvider(mass or MagicMock(), manifest, config, SUPPORTED_FEATURES)


def _track_obj(
    track_id: int,
    title: str,
    performers: str,
    *,
    performer: dict[str, Any] | None = None,
    composer: dict[str, Any] | None = None,
    work: str | None = None,
    album: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a Qobuz track object as an album's track listing gives it, unless given an album."""
    track: dict[str, Any] = {
        "id": track_id,
        "title": title,
        "version": None,
        "work": work,
        "duration": 448,
        "isrc": "DEF057401772",
        "streamable": True,
        "displayable": True,
        "maximum_sampling_rate": 96,
        "maximum_bit_depth": 24,
        "media_number": 1,
        "track_number": 1,
        "performers": performers,
    }
    if performer:
        track["performer"] = performer
    if composer:
        track["composer"] = composer
    if album:
        track["album"] = album
    return track


def _album_obj(
    genre: dict[str, Any] | None,
    track_objs: list[dict[str, Any]] | None = None,
    artists: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a Qobuz album object, with its track listing when given the tracks."""
    album: dict[str, Any] = {
        "id": "0002894790548",
        "title": "Beethoven : Symphonies n°5 & n°7",
        "version": None,
        "artist": KLEIBER,
        "artists": artists or [],
        "streamable": True,
        "displayable": True,
        "maximum_sampling_rate": 96,
        "maximum_bit_depth": 24,
        "product_type": "album",
    }
    if genre:
        album["genre"] = genre
    if track_objs is not None:
        album["tracks"] = {"items": track_objs, "total": len(track_objs)}
    return album


def _credits(track: Track) -> dict[str, list[str]]:
    """Return the names of a track's credited artists per role, with the instrument if any."""
    result: dict[str, list[str]] = {}
    for credit in track.credits:
        label = credit.role.value + (f" ({credit.instrument})" if credit.instrument else "")
        result.setdefault(label, []).append(credit.artist.name)
    return result


def _kleiber_album() -> dict[str, Any]:
    """Return the Kleiber album, whose tracks name their work apart from the movement."""
    return _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(
                6826226,
                "I. Allegro con brio",
                KLEIBER_PERFORMERS,
                performer=KLEIBER,
                composer=BEETHOVEN,
                work="Symphony No. 5 in C minor, Op. 67",
            )
        ],
        artists=[
            {**WIENER_PHILHARMONIC, "roles": ["main-artist", "orchestra"]},
            {**KLEIBER, "roles": ["main-artist", "conductor"]},
        ],
    )


@pytest.mark.parametrize(
    ("genre", "expected"),
    [
        (CLASSICAL_GENRE, True),
        (CHAMBER_MUSIC_GENRE, True),
        ({"id": 10, "name": "Classique"}, True),
        (POP_GENRE, False),
        ({"name": "Rock"}, False),
    ],
)
async def test_album_is_classical_by_its_genre(genre: dict[str, Any], expected: bool) -> None:
    """An album filed under classical music or one of its sub-genres is in the Classical genre."""
    album = await _make_provider()._parse_album(_album_obj(genre))

    assert (CLASSICAL_GENRE_NAME in (album.metadata.genres or ())) is expected
    assert genre["name"] in (album.metadata.genres or ())


async def test_work_given_apart_from_the_title_gives_work_movement_and_credits() -> None:
    """
    A track's work comes from Qobuz's work, its movement from the title.

    A name in the performers string without an id or any role is not credited, and a hyphenated
    name stays whole.
    """
    [track] = await _make_provider()._parse_album_tracks(_kleiber_album())

    assert isinstance(track.work, Work)
    assert track.work.name == "Symphony No. 5 in C minor, Op. 67"
    assert track.movement_name == "I. Allegro con brio"
    assert track.movement_number == 1
    # the orchestra the performers string names without an id comes from the album's artists,
    # the production credits are left out
    assert _credits(track) == {
        "composer": ["Ludwig van Beethoven"],
        "conductor": ["Carlos Kleiber"],
        "orchestra": ["Wiener Philharmonic Orchestra"],
    }
    composer = next(x.artist for x in track.credits if x.role == ArtistRole.COMPOSER)
    assert composer.item_id == "38"
    assert [x.name for x in track.artists] == ["Carlos Kleiber"]
    assert "Wolf-Dieter Karwatky, Recording Engineer" in (track.metadata.performers or set())
    assert "Hans-Rudolf Müller, Recording Engineer" in (track.metadata.performers or set())


async def test_director_is_no_conductor_next_to_a_conductor() -> None:
    """A track naming a conductor credits its director with nothing more."""
    performers = "Carlos Kleiber, Conductor - Herbert von Karajan, Director - Puccini, Composer"
    album = _album_obj(
        CLASSICAL_GENRE,
        [_track_obj(1, "Tosca: Act I", performers, performer=KLEIBER)],
        artists=[{**KARAJAN, "roles": ["main-artist"]}],
    )

    [track] = await _make_provider()._parse_album_tracks(album)

    assert _credits(track) == {"conductor": ["Carlos Kleiber"], "composer": ["Puccini"]}


async def test_composer_prefix_is_left_out_and_main_artists_perform() -> None:
    """A "C. Lambert:" prefix naming the composer is no part of the work."""
    title = "C. Lambert: Concerto for Piano and 9 Players: III. Finale. Lugubre"
    ian_brown = {"id": 2, "name": "Ian Brown"}
    album = _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(249499102, title, LAMBERT_PERFORMERS, performer=FRIEND, composer=LAMBERT),
            _track_obj(3, "Piano Sonata: I. Allegro", "", performer=ian_brown, composer=LAMBERT),
        ],
        artists=[
            {**NASH_ENSEMBLE, "roles": ["main-artist"]},
            {**FRIEND, "roles": ["main-artist"]},
        ],
    )

    track, _ = await _make_provider()._parse_album_tracks(album)

    assert isinstance(track.work, Work)
    assert track.work.name == "Concerto for Piano and 9 Players"
    assert track.movement_number == 3
    assert _credits(track) == {
        "composer": ["Constant Lambert"],
        "conductor": ["Lionel Friend"],
        "performer": ["Ian Brown"],
        "ensemble": ["The Nash Ensemble"],
    }


async def test_voice_and_instruments_credit_soloists() -> None:
    """A voice or an instrument credits a soloist with it, a translator or poet nothing."""
    album = _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(
                249499103,
                "C. Lambert: 8 Poems of Li-Po: I. A Summer Day",
                LI_PO_PERFORMERS,
                performer=LANGRIDGE,
                composer=LAMBERT,
            ),
            _track_obj(
                4,
                "I. Allegro molto moderato",
                SCHUBERT_PERFORMERS,
                performer=STAIER,
                composer=SCHUBERT,
                work="Fantasie in F Minor, Op. 103, D. 940",
            ),
        ],
        artists=[{**NASH_ENSEMBLE, "roles": ["main-artist"]}],
    )

    li_po, fantasie = await _make_provider()._parse_album_tracks(album)

    assert _credits(li_po) == {
        "soloist (tenor vocals)": ["Philip Langridge"],
        "ensemble": ["The Nash Ensemble"],
        "composer": ["Constant Lambert"],
    }
    assert _credits(fantasie) == {
        "composer": ["Franz Schubert"],
        "soloist (piano)": ["Andreas Staier", "Alexander Melnikov"],
    }


async def test_joined_main_artist_is_no_performer_of_its_own() -> None:
    """The main artists Qobuz joins into one artist are credited each, not as that one artist."""
    joined = {"id": 3544140, "name": "Théodore Paraskivesco, Jacques Rouvier"}
    paraskivesco = {"id": 5, "name": "Théodore Paraskivesco"}
    album = _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(
                1,
                "Ma mère l'oye, M. 60: No. 1, Pavane de la Belle au bois dormant. Lent",
                RAVEL_PERFORMERS,
                performer=joined,
                composer={"id": 6, "name": "Maurice Ravel"},
            ),
            _track_obj(2, "Pavane", "", performer=paraskivesco),
        ],
    )

    track, _ = await _make_provider()._parse_album_tracks(album)

    assert _credits(track) == {
        "composer": ["Maurice Ravel"],
        "soloist": ["Jacques Rouvier", "Théodore Paraskivesco"],
    }


async def test_joined_main_artist_cut_short_is_no_performer_of_its_own() -> None:
    """A joined main artist Qobuz cuts short is dropped like the full joined name it starts."""
    performers = (
        "Margot Kuijpers, Soprano Vocals - Marcel Schmitz, Tenor Vocals - Soloists of the Comic "
        "Opera Berlin, Choir - Margot Kuijpers, Marcel Schmitz, Soloists of the Comic Opera "
        "Berlin, MainArtist - Kurt Weill, Composer"
    )
    cut_short = {
        "id": 7,
        "name": "Margot Kuijpers, Marcel Schmitz, Soloists of the Comic Opera Berl",
    }
    album = _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(
                1,
                "Die Dreigroschenoper: No. 1, Ouvertüre",
                performers,
                performer=cut_short,
                composer={"id": 8, "name": "Kurt Weill"},
            )
        ],
    )

    [track] = await _make_provider()._parse_album_tracks(album)

    assert _credits(track) == {
        "composer": ["Kurt Weill"],
        "soloist (soprano vocals)": ["Margot Kuijpers"],
        "soloist (tenor vocals)": ["Marcel Schmitz"],
        "choir": ["Soloists of the Comic Opera Berlin"],
    }


async def test_title_naming_its_work_wins_over_the_work_given() -> None:
    """A title that names its work names it the way the album's other titles do."""
    album = _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(
                1,
                "Goldberg Variations, BWV 988: Aria",
                "Andreas Staier, Performer\r - Johann Sebastian Bach, Composer",
                performer=STAIER,
                composer={"id": 21, "name": "Johann Sebastian Bach"},
                work="Goldberg-Variationen BWV 988",
            ),
            _track_obj(
                2,
                "4 Ländler, D. 814",
                SCHUBERT_PERFORMERS,
                performer=STAIER,
                composer=SCHUBERT,
                work="4 Ländler, D. 814",
            ),
        ],
    )

    aria, landler = await _make_provider()._parse_album_tracks(album)

    assert aria.work is not None
    assert aria.work.name == "Goldberg Variations, BWV 988"
    assert aria.movement_name == "Aria"
    assert _credits(aria)["performer"] == ["Andreas Staier"]
    # a track that is a whole work has no movement
    assert landler.work is not None
    assert landler.work.name == "4 Ländler, D. 814"
    assert landler.movement_name is None


async def test_track_of_a_pop_album_gets_no_classical_details() -> None:
    """A track on a pop album keeps its performers alone, even with a composer and a colon."""
    album = _album_obj(POP_GENRE)
    track_obj = _track_obj(
        1,
        "Star Wars: Main Title",
        "Bernie Taupin, Composer - Elton John, Composer - John Keating, MainArtist",
        performer={"id": 149611, "name": "John Keating"},
        composer={"id": 9, "name": "Elton John"},
        work="Star Wars",
        album=album,
    )

    track = await _make_provider()._parse_track(track_obj)

    assert track.credits == []
    assert track.work is None
    assert track.movement_name is None


async def test_single_track_reads_its_embedded_album() -> None:
    """A track fetched on its own is classical by the album it embeds."""
    album = _kleiber_album()
    track_obj = {**album.pop("tracks")["items"][0], "album": album}

    track = await _make_provider()._parse_track(track_obj)

    assert isinstance(track.album, Album)
    assert CLASSICAL_GENRE_NAME in (track.album.metadata.genres or ())
    assert track.work is not None
    assert _credits(track)["conductor"] == ["Carlos Kleiber"]


async def test_artist_named_without_an_id_never_reaches_qobuz() -> None:
    """An artist id that is a name answers without a request."""
    provider = _make_provider()
    provider._get_data = AsyncMock()  # type: ignore[method-assign]
    named = provider._parse_named_artist("Ferruccio Mazzoli")

    with pytest.raises(MediaNotFoundError):
        await provider.get_artist.__wrapped__(provider, named.item_id)  # type: ignore[attr-defined]
    assert await provider.get_artist_albums.__wrapped__(provider, named.item_id) == []  # type: ignore[attr-defined]
    assert await provider.get_artist_toptracks.__wrapped__(provider, named.item_id) == []  # type: ignore[attr-defined]
    assert not await provider.library_add(named)
    assert not await provider.library_remove(named.item_id, named.media_type)
    provider._get_data.assert_not_awaited()


async def test_rights_reserved_placeholder_is_no_composer() -> None:
    """A "D.R." composer, as compilations give it, credits no composer and no artist "DR"."""
    vivaldi = {"id": 30290, "name": "Antonio Vivaldi"}
    performers = (
        "Antonio Vivaldi, Orchestra, Artist, MainArtist - Dr, Composer - D.R, OriginalPublisher"
    )
    album = _album_obj(
        CLASSICAL_GENRE,
        [
            _track_obj(
                1, "The Four Seasons", performers, performer=vivaldi, composer=RIGHTS_RESERVED
            )
        ],
    )

    [track] = await _make_provider()._parse_album_tracks(album)

    assert _credits(track) == {"orchestra": ["Antonio Vivaldi"]}


async def test_composer_respelled_in_the_performers_string_is_credited_once() -> None:
    """A performers string spelling the track's composer another way adds no second composer."""
    chopin = {"id": 26543, "name": "Frédéric Chopin"}
    pianist = {"id": 418, "name": "Maria João Pires"}
    performers = "Frederic Chopin, Composer - Maria João Pires, Piano, Artist, MainArtist"
    album = _album_obj(
        CLASSICAL_GENRE,
        [_track_obj(1, "Nocturnes, Op. 62: No. 1", performers, performer=pianist, composer=chopin)],
    )

    [track] = await _make_provider()._parse_album_tracks(album)

    assert _credits(track) == {
        "composer": ["Frédéric Chopin"],
        "soloist (piano)": ["Maria João Pires"],
    }


async def test_co_composers_in_the_performers_string_are_all_credited() -> None:
    """A performers string naming the track's composer with others credits each of them."""
    lennon = {"id": 1, "name": "John Lennon"}
    performers = "John Lennon, Composer - Paul Mccartney, Composer - The Beatles, MainArtist"
    album = _album_obj(
        CLASSICAL_GENRE,
        [_track_obj(1, "Lucy in the Sky with Diamonds", performers, composer=lennon)],
    )

    [track] = await _make_provider()._parse_album_tracks(album)

    assert _credits(track)["composer"] == ["John Lennon", "Paul Mccartney"]


async def test_album_tracks_are_read_with_their_album() -> None:
    """The album's track listing is read in one request, with the album's classical details."""
    provider = _make_provider()
    provider._get_data = AsyncMock(return_value=_kleiber_album())  # type: ignore[method-assign]

    [track] = await provider.get_album_tracks.__wrapped__(provider, "0002894790548")  # type: ignore[attr-defined]

    assert track.work is not None
    provider._get_data.assert_awaited_once_with(
        "album/get", album_id="0002894790548", limit=500, offset=0
    )


async def test_album_tracks_beyond_the_first_page_are_read_once() -> None:
    """The rest of a long track listing is read from where the album's first page ends."""
    provider = _make_provider()
    first_page = [{"id": number} for number in range(500)]
    rest = [{"id": number} for number in range(500, 600)]
    album = {"id": "0002894790548", "tracks": {"items": first_page, "total": 600}}
    provider._get_data = AsyncMock(  # type: ignore[method-assign]
        side_effect=[album, {"tracks": {"items": rest, "total": 600}}]
    )

    album_obj = await provider._get_album_with_tracks("0002894790548")

    assert album_obj is not None
    assert album_obj["tracks"]["items"] == first_page + rest
    assert [call.kwargs["offset"] for call in provider._get_data.await_args_list] == [0, 500]


class _CacheLookup(Exception):
    """Stops a cached call at its cache lookup."""


@pytest.mark.parametrize(
    ("method", "external_id_type"),
    [
        ("get_track_by_external_id", ExternalID.ISRC),
        ("get_album_by_external_id", ExternalID.BARCODE),
    ],
)
async def test_items_found_by_external_id_are_cached_with_the_parse_checksum(
    method: str, external_id_type: ExternalID
) -> None:
    """A parsing change drops the cached items found by their ISRC or barcode."""
    lookup = AsyncMock(side_effect=_CacheLookup)
    mass = MagicMock()
    mass.cache.get_with_freshness = lookup
    provider = _make_provider(mass)

    with pytest.raises(_CacheLookup):
        await getattr(provider, method)("DEF057401772", external_id_type)

    assert lookup.await_args is not None
    assert lookup.await_args.kwargs["checksum"] == PARSED_ITEM_CACHE_CHECKSUM


def _library_mass(rows: list[dict[str, Any]], library_track: MagicMock | None) -> MagicMock:
    """
    Return a Music Assistant whose library holds the given Qobuz albums.

    :param rows: The library id and Qobuz id of each album.
    :param library_track: The library track each Qobuz track is found as, if any.
    """
    mass = MagicMock()
    mass.music.database.get_rows_from_query = AsyncMock(return_value=rows)
    mass.music.albums.update_item_in_library = AsyncMock(side_effect=lambda _item_id, album: album)
    mass.music.genres.sync_media_item_genres = AsyncMock()
    mass.music.tracks.update_item_in_library = AsyncMock()
    mass.music.tracks.get_library_item_by_prov_id = AsyncMock(return_value=library_track)
    return mass
