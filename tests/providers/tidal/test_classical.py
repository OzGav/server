"""Tests for the classical details the Tidal provider reads from its albums and tracks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import ArtistRole, MediaType
from music_assistant_models.media_items import Work

from music_assistant.constants import CLASSICAL_GENRE_NAME
from music_assistant.providers.tidal.constants import CONF_CLASSICAL_REREAD_DONE
from music_assistant.providers.tidal.jsonapi import JsonApiDocument
from music_assistant.providers.tidal.media import TidalMediaManager
from music_assistant.providers.tidal.provider import TidalProvider

if TYPE_CHECKING:
    from music_assistant_models.media_items import Track

# genres, artists and credits as the official Tidal API gives them for the user's albums
CLASSICAL = ("17", "Classical")
INSTRUMENTAL = ("29", "Instrumental")
POP = ("1", "Pop")
REGGAE = ("12", "Reggae")

BEETHOVEN = ("57386", "Ludwig Van Beethoven")
KLEIBER = ("7246", "Carlos Kleiber")
WIENER_PHILHARMONIKER = ("3652537", "Wiener Philharmoniker")
PUCCINI = ("3687953", "Giacomo Puccini")
KARAJAN = ("64046", "Herbert von Karajan")
STAATSOPER = ("4342108", "Orchester der Wiener Staatsoper")
MAZZOLI = ("4380370", "Ferruccio Mazzoli")
UZUNOV = ("50389245", "Dimiter Uzunov")
USUNOW = ("50393907", "Dimiter Usunow")
PAVAROTTI = ("5557", "Luciano Pavarotti")
PANERAI = ("4878228", "Rolando Panerai")
GHIAUROV = ("4352475", "Nicolai Ghiaurov")
BERLINER_PHILHARMONIKER = ("14327", "Berliner Philharmoniker")
ILLICA = ("6220264", "Luigi Illica")
GIACOSA = ("6220263", "Giuseppe Giacosa")
BACH = ("14753", "Johann Sebastian Bach")
JACOBS = ("7946", "René Jacobs")
RIAS = ("3688549", "RIAS Kammerchor")
AKAMUS = ("5921445", "Akademie für Alte Musik Berlin")
ALLEGRI = ("4169438", "Gregorio Allegri")
TALLIS_SCHOLARS = ("4207564", "The Tallis Scholars")
PHILLIPS = ("3520838", "Peter Phillips")
ROSSINI = ("3530263", "Gioachino Rossini")
DELIBES = ("3591620", "Léo Delibes")
TCHAIKOVSKY = ("3502960", "Pyotr Ilyich Tchaikovsky")
AINSWORTH = ("4998101", "Sarah Ainsworth")
TAYLOR_SWIFT = ("3557299", "Taylor Swift")
ANTONOFF = ("6439819", "Jack Antonoff")
HANES = ("6116431", "John Hanes")
MARLEY = ("5927", "Bob Marley & The Wailers")
TOSH = ("158", "Peter Tosh")
VIVALDI = ("3510498", "Antonio Vivaldi")
PLACEHOLDER = ("4319083", "dr")


class _Track:
    """A Tidal track resource with the genres, artists and credits it links to."""

    def __init__(
        self,
        track_id: str,
        title: str,
        artists: list[tuple[str, str]],
        genres: list[tuple[str, str]],
        credited: list[tuple[str, tuple[str, str], str | None]] | None = None,
        version: str | None = None,
    ) -> None:
        """
        Describe a track.

        :param track_id: The Tidal id of the track.
        :param title: The title of the track.
        :param artists: The id and name of each main artist.
        :param genres: The id and name of each genre.
        :param credited: The role, linked artist and credited name (None for the artist's name).
        :param version: The version of the track.
        """
        self.track_id = track_id
        self.title = title
        self.artists = artists
        self.genres = genres
        self.credits = credited or []
        self.version = version

    def resources(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return the track resource and the resources it links to."""
        included: list[dict[str, Any]] = [_artist(*x) for x in self.artists]
        included += [
            {"id": genre_id, "type": "genres", "attributes": {"genreName": name}}
            for genre_id, name in self.genres
        ]
        credit_ids = []
        for index, (role, artist, name) in enumerate(self.credits):
            credit_id = f"{self.track_id}-{index}"
            credit_ids.append(credit_id)
            included.append(_artist(*artist))
            included.append(
                {
                    "id": credit_id,
                    "type": "credits",
                    "attributes": {"name": name or artist[1], "role": role},
                    "relationships": {"artist": {"data": {"id": artist[0], "type": "artists"}}},
                }
            )
        track = {
            "id": self.track_id,
            "type": "tracks",
            "attributes": {"title": self.title, "version": self.version, "duration": "PT5M"},
            "relationships": {
                "artists": _linkage("artists", [x[0] for x in self.artists]),
                "genres": _linkage("genres", [x[0] for x in self.genres]),
                "credits": _linkage("credits", credit_ids),
            },
        }
        return track, included


KLEIBER_CREDITS: list[tuple[str, tuple[str, str], str | None]] = [
    ("Composer", BEETHOVEN, "Ludwig van Beethoven"),
    ("Conductor", KLEIBER, None),
    ("Orchestra", WIENER_PHILHARMONIKER, None),
    ("Producer", ("9001", "Werner Mayer"), None),
    ("Recording Engineer", ("12538449", "Hans-Rudolf Müller"), None),
    ("Digital Editing Engineer", ("12538113", "Hans-Peter Schweigmann"), None),
]
KLEIBER_TRACKS = [
    _Track(
        "17626400",
        "Beethoven: Symphony No. 5 in C Minor, Op. 67: I. Allegro con brio",
        [WIENER_PHILHARMONIKER, KLEIBER],
        [CLASSICAL],
        KLEIBER_CREDITS,
    ),
    _Track(
        "17626401",
        "Beethoven: Symphony No. 5 in C Minor, Op. 67: II. Andante con moto",
        [WIENER_PHILHARMONIKER, KLEIBER],
        [CLASSICAL],
        KLEIBER_CREDITS,
    ),
]
TOSCA_TRACK = _Track(
    "386769584",
    'Tosca, S.69, IGP 17, Act I: "Ah!... Finalmente!... Nel terror mio stolto" '
    "(Angelotti, Il Sagrestano, Cavaradossi)",
    [STAATSOPER, KARAJAN, MAZZOLI, USUNOW],
    [CLASSICAL],
    [
        ("Composer", PUCCINI, "GIACOMO PUCCINI"),
        ("Bass Vocals", MAZZOLI, None),
        ("Director", KARAJAN, None),
        ("Orchestra", STAATSOPER, None),
        ("Tenor Vocals", UZUNOV, None),
    ],
)
BOHEME_TRACK = _Track(
    "426719254",
    "Puccini: La bohème, Act I: Questo mar rosso",
    [PANERAI, PAVAROTTI, GHIAUROV, BERLINER_PHILHARMONIKER, KARAJAN],
    [CLASSICAL],
    [
        ("Composer", PUCCINI, "GIACOMO PUCCINI"),
        ("Lyricist", ILLICA, None),
        ("Lyricist", GIACOSA, None),
        ("BaritoneVocalist", PANERAI, None),
        ("BassVocalist", GHIAUROV, None),
        ("Conductor", KARAJAN, None),
        ("Orchestra", BERLINER_PHILHARMONIKER, None),
        ("TenorVocalist", PAVAROTTI, None),
    ],
    version="2024 Remaster",
)
PASSION_TRACK = _Track(
    "46998827",
    "Erster Teil, 1. Chorus I & II Kommt, ihr Töchter, helft mir klagen",
    [RIAS, AKAMUS, JACOBS],
    [CLASSICAL],
    [
        ("Composer", BACH, None),
        ("Conductor", JACOBS, None),
        ("Ensemble", AKAMUS, None),
        ("Ensemble", RIAS, None),
    ],
)
GIMELL_CREDITS: list[tuple[str, tuple[str, str], str | None]] = [
    ("Composer", ALLEGRI, None),
    ("Lyricist", ALLEGRI, None),
    ("Songwriter", ALLEGRI, None),
]
GIMELL_TRACKS = [
    _Track(
        f"13331488{number}",
        f"Allegri: Miserere - 0{number}. {title}",
        [TALLIS_SCHOLARS, PHILLIPS],
        [INSTRUMENTAL],
        GIMELL_CREDITS,
    )
    for number, title in ((1, "Miserere Mei, Deus"), (2, "Amplius Lava Me"))
]
COMPILATION_TRACKS = [
    _Track(
        "10581229",
        "Rossini: La scala di seta (The Silken Ladder) Opera: Overture",
        [ROSSINI],
        [CLASSICAL],
    ),
    _Track(
        "10581231",
        'Tchaikovsky: "Chanson Triste" from 12 Pieces, Op 40: No. 2',
        [TCHAIKOVSKY, AINSWORTH],
        [CLASSICAL],
    ),
]
PLACEHOLDER_COMPOSER_TRACK = _Track(
    "21815470",
    "Vivaldi: The Four Seasons, Spring: I. Allegro",
    [VIVALDI],
    [CLASSICAL],
    [("Composer", PLACEHOLDER, None)],
)
LOVER_TRACK = _Track(
    "116125897",
    "Lover",
    [TAYLOR_SWIFT],
    [POP],
    [
        ("Composer", TAYLOR_SWIFT, None),
        ("Lyricist", TAYLOR_SWIFT, None),
        ("Piano", ANTONOFF, None),
        ("Mixing Engineer", HANES, None),
        ("Vocalist", TAYLOR_SWIFT, None),
    ],
)
# a pop track with a "Work: Piece" like title and a songwriter credit
INTERLUDE_TRACK = _Track(
    "116125898",
    "Interlude: Reprise",
    [TAYLOR_SWIFT],
    [POP],
    [("Composer", TAYLOR_SWIFT, None), ("Vocalist", TAYLOR_SWIFT, None)],
)
MARLEY_TRACK = _Track(
    "574875",
    "Get Up, Stand Up",
    [MARLEY],
    [REGGAE, POP],
    [
        ("Composer", TOSH, None),
        ("Composer", MARLEY, "Bob Marley"),
        ("Vocalist", MARLEY, "Bob Marley"),
    ],
)


async def test_kleiber_album_tracks_get_composer_conductor_orchestra_and_work(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """A classical album's tracks get their credits and work, production roles left out."""
    calls = _serve_pages(provider_mock, [KLEIBER_TRACKS])

    first, second = await media_manager.get_album_tracks("17626399")

    assert calls == [
        {
            "include": [
                "items.artists",
                "items.albums.coverArt",
                "items.genres",
                "items.credits",
                "items.credits.artist",
            ],
            "replace_media": "items",
        }
    ]
    assert _credits(first) == {
        ArtistRole.COMPOSER: [("Ludwig Van Beethoven", None)],
        ArtistRole.CONDUCTOR: [("Carlos Kleiber", None)],
        ArtistRole.ORCHESTRA: [("Wiener Philharmoniker", None)],
    }
    assert [x.name for x in first.artists] == ["Wiener Philharmoniker", "Carlos Kleiber"]
    assert CLASSICAL_GENRE_NAME in (first.metadata.genres or set())
    assert isinstance(first.work, Work)
    assert first.work.name == "Symphony No. 5 in C Minor, Op. 67"
    assert (first.movement_name, first.movement_number) == ("I. Allegro con brio", 1)
    assert second.movement_number == 2


async def test_tosca_director_conducts_and_composer_gets_its_proper_name(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """A director conducts, voices make soloists and a main artist without credit performs."""
    provider_mock.api.get_jsonapi.return_value = _track_doc(TOSCA_TRACK)
    provider_mock.api.get.return_value = {}

    track = await media_manager.get_track("386769584")

    provider_mock.api.get_jsonapi.assert_called_with(
        "tracks/386769584",
        include=["artists", "albums", "albums.coverArt", "genres", "credits", "credits.artist"],
    )
    assert _credits(track) == {
        ArtistRole.COMPOSER: [("Giacomo Puccini", None)],
        ArtistRole.SOLOIST: [
            ("Ferruccio Mazzoli", "bass vocals"),
            ("Dimiter Uzunov", "tenor vocals"),
        ],
        ArtistRole.CONDUCTOR: [("Herbert von Karajan", None)],
        ArtistRole.ORCHESTRA: [("Orchester der Wiener Staatsoper", None)],
        ArtistRole.PERFORMER: [("Dimiter Usunow", None)],
    }
    assert isinstance(track.work, Work)
    assert track.work.name == "Tosca, S.69, IGP 17, Act I"


async def test_boheme_gives_voices_lyricists_and_the_act(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """Voices written as one word make soloists, and the title names the act."""
    provider_mock.api.get_jsonapi.return_value = _track_doc(BOHEME_TRACK)
    provider_mock.api.get.return_value = {}

    track = await media_manager.get_track("426719254")

    by_role = _credits(track)
    assert by_role[ArtistRole.SOLOIST] == [
        ("Rolando Panerai", "baritone vocals"),
        ("Nicolai Ghiaurov", "bass vocals"),
        ("Luciano Pavarotti", "tenor vocals"),
    ]
    assert by_role[ArtistRole.LYRICIST] == [("Luigi Illica", None), ("Giuseppe Giacosa", None)]
    assert ArtistRole.PERFORMER not in by_role
    assert (track.name, track.version) == (
        "Puccini: La bohème, Act I: Questo mar rosso",
        "2024 Remaster",
    )
    assert isinstance(track.work, Work)
    assert track.work.name == "La bohème, Act I"
    assert track.movement_name == "Questo mar rosso"


async def test_passion_credits_ensembles_and_names_no_work(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """Ensembles are credited as ensembles; a title naming no work gives none."""
    provider_mock.api.get_jsonapi.return_value = _track_doc(PASSION_TRACK)
    provider_mock.api.get.return_value = {}

    track = await media_manager.get_track("46998827")

    by_role = _credits(track)
    assert by_role[ArtistRole.ENSEMBLE] == [
        ("Akademie für Alte Musik Berlin", None),
        ("RIAS Kammerchor", None),
    ]
    assert by_role[ArtistRole.CONDUCTOR] == [("René Jacobs", None)]
    assert track.work is None


async def test_gimell_album_is_classical_by_its_composers_and_work_titles(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """An album Tidal files elsewhere is classical when its tracks credit composers of works."""
    _serve_pages(provider_mock, [GIMELL_TRACKS[:1], GIMELL_TRACKS[1:]])

    first, second = await media_manager.get_album_tracks("133314886")

    assert first.metadata.genres == {"Instrumental"}
    assert _credits(first) == {
        ArtistRole.COMPOSER: [("Gregorio Allegri", None)],
        ArtistRole.LYRICIST: [("Gregorio Allegri", None)],
        ArtistRole.PERFORMER: [("The Tallis Scholars", None), ("Peter Phillips", None)],
    }
    assert isinstance(first.work, Work)
    assert first.work.name == "Miserere"
    assert (first.movement_name, first.movement_number) == ("01. Miserere Mei, Deus", 1)
    assert second.movement_number == 2

    provider_mock.api.get_jsonapi.return_value = _album_doc("133314886", GIMELL_TRACKS)
    album = await media_manager.get_album("133314886")

    assert album.metadata.genres == {CLASSICAL_GENRE_NAME}


async def test_compilation_without_credits_lists_its_composers_first(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """Tracks without credits credit their first artist as composer, as compilations list."""
    _serve_pages(provider_mock, [COMPILATION_TRACKS])

    rossini, tchaikovsky = await media_manager.get_album_tracks("10581228")

    assert _credits(rossini) == {ArtistRole.COMPOSER: [("Gioachino Rossini", None)]}
    assert isinstance(rossini.work, Work)
    assert rossini.work.name == "La scala di seta (The Silken Ladder) Opera"
    assert rossini.movement_name == "Overture"
    assert _credits(tchaikovsky) == {ArtistRole.COMPOSER: [("Pyotr Ilyich Tchaikovsky", None)]}
    assert isinstance(tchaikovsky.work, Work)
    assert tchaikovsky.work.name == '"Chanson Triste" from 12 Pieces, Op 40'


async def test_placeholder_composer_gives_no_credit_and_keeps_its_role(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """A composer credit naming a placeholder is left out, its performer entry keeps the role."""
    provider_mock.api.get_jsonapi.return_value = _track_doc(PLACEHOLDER_COMPOSER_TRACK)
    provider_mock.api.get.return_value = {}

    track = await media_manager.get_track("21815470")

    assert _credits(track) == {ArtistRole.PERFORMER: [("Antonio Vivaldi", None)]}
    assert track.metadata.performers == {"dr, Composer"}


@pytest.mark.parametrize(
    "track",
    [LOVER_TRACK, MARLEY_TRACK, INTERLUDE_TRACK],
    ids=["taylor-swift", "bob-marley", "colon-title"],
)
async def test_pop_tracks_keep_their_credit_names_only(
    track: _Track, media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """Tracks of pop albums get no credits, work or classical genre, only performer names."""
    _serve_pages(provider_mock, [[track]])
    provider_mock.api.get_jsonapi.return_value = _track_doc(track)
    provider_mock.api.get.return_value = {}

    for parsed in [
        *(await media_manager.get_album_tracks("1")),
        await media_manager.get_track("1"),
    ]:
        assert parsed.credits == []
        assert parsed.work is None
        assert parsed.metadata.genres == {name for _id, name in track.genres}
        assert parsed.metadata.performers == {
            name or artist[1] for _r, artist, name in track.credits
        }

    provider_mock.api.get_jsonapi.return_value = _album_doc("1", [track])
    album = await media_manager.get_album("1")

    assert not album.metadata.genres


async def test_album_is_classical_by_the_first_page_of_its_tracks(
    media_manager: TidalMediaManager, provider_mock: Mock
) -> None:
    """The album's request reads the first page of its tracks, which tells a classical album."""
    provider_mock.api.get_jsonapi.return_value = _album_doc("17626399", KLEIBER_TRACKS)

    album = await media_manager.get_album("17626399")

    provider_mock.api.get_jsonapi.assert_called_with(
        "albums/17626399",
        include=["artists", "coverArt", "genres", "items", "items.genres", "items.credits.artist"],
    )
    assert album.metadata.genres == {CLASSICAL_GENRE_NAME}


async def test_reread_updates_a_classical_library_album_and_its_library_tracks(
    provider: TidalProvider, mass: MagicMock
) -> None:
    """The one-off re-read tags a library album, updates its library tracks and is marked done."""
    _fill_library(mass, [{"item_id": 7, "provider_item_id": "17626399"}])
    provider.api.get_jsonapi = AsyncMock(  # type: ignore[method-assign]
        return_value=_album_doc("17626399", KLEIBER_TRACKS)
    )
    provider.api.paginate_jsonapi = _pages_of([KLEIBER_TRACKS])  # type: ignore[method-assign]
    provider._update_config_value = MagicMock()  # type: ignore[method-assign]
    # the cached album calls, so what the re-read fetches is kept for later
    provider.get_album = AsyncMock(side_effect=provider.media.get_album)  # type: ignore[method-assign]
    provider.get_album_tracks = AsyncMock(  # type: ignore[method-assign]
        side_effect=provider.media.get_album_tracks
    )

    await provider._reread_library_albums()

    provider.get_album.assert_awaited_once_with("17626399")
    provider.get_album_tracks.assert_awaited_once_with("17626399")
    [(album_args, _)] = mass.music.albums.update_item_in_library.await_args_list
    assert album_args[0] == 7
    mass.music.genres.sync_media_item_genres.assert_awaited_once_with(
        MediaType.ALBUM, 7, {CLASSICAL_GENRE_NAME}
    )
    updated = [args[1] for args, _ in mass.music.tracks.update_item_in_library.await_args_list]
    assert [x.item_id for x in updated] == ["17626400", "17626401"]
    assert all(x.work is not None for x in updated)
    provider._update_config_value.assert_called_once_with(
        CONF_CLASSICAL_REREAD_DONE, True, immediate=True
    )


async def test_reread_leaves_pop_albums_alone(provider: TidalProvider, mass: MagicMock) -> None:
    """A pop album is not updated and its tracks are not read."""
    _fill_library(mass, [{"item_id": 7, "provider_item_id": "116125894"}])
    provider.api.get_jsonapi = AsyncMock(  # type: ignore[method-assign]
        return_value=_album_doc("116125894", [LOVER_TRACK])
    )
    provider.api.paginate_jsonapi = MagicMock()  # type: ignore[method-assign]
    provider._update_config_value = MagicMock()  # type: ignore[method-assign]
    provider.get_album = AsyncMock(side_effect=provider.media.get_album)  # type: ignore[method-assign]

    await provider._reread_library_albums()

    mass.music.albums.update_item_in_library.assert_not_awaited()
    provider.api.paginate_jsonapi.assert_not_called()
    provider._update_config_value.assert_called_once()


@pytest.mark.parametrize(("done", "started"), [(False, True), (True, False)])
async def test_reread_starts_only_until_it_is_done(
    provider: TidalProvider, mass: MagicMock, done: bool, started: bool
) -> None:
    """The re-read starts once loaded, until a completed run marked it as done."""
    mass.config.get_raw_provider_config_value = MagicMock(return_value=done)
    provider._reread_library_albums = MagicMock()  # type: ignore[method-assign]

    await provider.loaded_in_mass()

    assert mass.create_task.called == started


@pytest.fixture
def mass() -> MagicMock:
    """Return a stand-in for Music Assistant."""
    return MagicMock()


@pytest.fixture
def provider(mass: MagicMock) -> TidalProvider:
    """Return a Tidal provider with only what parsing and the library re-read need."""
    manifest = Mock()
    manifest.domain = "tidal"
    config = Mock()
    config.instance_id = "tidal_test"
    config.get_value.side_effect = lambda key, default=None: {"log_level": "GLOBAL"}.get(
        key, default
    )
    return TidalProvider(mass, manifest, config)


def _artist(artist_id: str, name: str) -> dict[str, Any]:
    """Return a Tidal artist resource."""
    return {"id": artist_id, "type": "artists", "attributes": {"name": name}}


def _linkage(resource_type: str, ids: list[str]) -> dict[str, Any]:
    """Return a relationship linking to the given resources."""
    return {"data": [{"id": x, "type": resource_type} for x in ids]}


def _included(tracks: list[_Track]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the track resources and every other resource they link to, once each."""
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    track_resources = []
    for track in tracks:
        track_resource, included = track.resources()
        track_resources.append(track_resource)
        for resource in (track_resource, *included):
            resources[(resource["type"], resource["id"])] = resource
    return track_resources, list(resources.values())


def _track_doc(track: _Track) -> JsonApiDocument:
    """Return the document Tidal gives for a full track."""
    track_resource, included = track.resources()
    return JsonApiDocument({"data": track_resource, "included": included})


def _album_doc(album_id: str, tracks: list[_Track]) -> JsonApiDocument:
    """Return the document Tidal gives for an album with the first page of its tracks."""
    track_resources, included = _included(tracks)
    album = {
        "id": album_id,
        "type": "albums",
        "attributes": {"title": "Album", "albumType": "ALBUM"},
        "relationships": {
            "artists": _linkage("artists", []),
            "genres": _linkage("genres", []),
            "items": _linkage("tracks", [x["id"] for x in track_resources]),
        },
    }
    return JsonApiDocument({"data": album, "included": included})


def _pages_of(pages: list[list[_Track]]) -> Any:
    """Return a stand-in for paginate_jsonapi yielding an album's track listing pages."""
    calls: list[dict[str, Any]] = []

    async def paginate(_path: str, **kwargs: Any) -> AsyncGenerator[JsonApiDocument]:
        calls.append(kwargs)
        for page in pages:
            track_resources, included = _included(page)
            data = [
                {"id": x["id"], "type": "tracks", "meta": {"volumeNumber": 1, "trackNumber": n}}
                for n, x in enumerate(track_resources, 1)
            ]
            yield JsonApiDocument({"data": data, "included": included})

    paginate.calls = calls  # type: ignore[attr-defined]
    return paginate


def _serve_pages(provider_mock: Mock, pages: list[list[_Track]]) -> list[dict[str, Any]]:
    """Serve the given track listing pages for any album, returning the requests made."""
    paginate = _pages_of(pages)
    provider_mock.api.paginate_jsonapi = paginate
    calls: list[dict[str, Any]] = paginate.calls
    return calls


def _credits(track: Track) -> dict[ArtistRole, list[tuple[str, str | None]]]:
    """Return the artist name and instrument of each of a track's credits, by role."""
    result: dict[ArtistRole, list[tuple[str, str | None]]] = {}
    for credit in sorted(track.credits, key=lambda x: x.position):
        result.setdefault(credit.role, []).append((credit.artist.name, credit.instrument))
    return result


def _fill_library(mass: MagicMock, rows: list[dict[str, Any]]) -> None:
    """
    Fill the library of a Music Assistant stand-in with the given Tidal albums.

    :param mass: The Music Assistant stand-in.
    :param rows: The library id and Tidal id of each album.
    """
    mass.music.database.get_rows_from_query = AsyncMock(return_value=rows)
    mass.music.albums.update_item_in_library = AsyncMock(side_effect=lambda _item_id, album: album)
    mass.music.genres.sync_media_item_genres = AsyncMock()
    mass.music.tracks.update_item_in_library = AsyncMock()
    mass.music.tracks.get_library_item_by_prov_id = AsyncMock(
        side_effect=lambda item_id, _instance: MagicMock(item_id=item_id)
    )
