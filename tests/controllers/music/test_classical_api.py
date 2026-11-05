"""Tests for the classical browse API of the music controller."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from music_assistant_models.auth import Scope, User, UserRole
from music_assistant_models.config_entries import ProviderAccess
from music_assistant_models.enums import (
    AlbumType,
    ArtistRole,
    ExternalID,
    ImageType,
    MediaType,
    Period,
    ProviderSharing,
    SortDirection,
    SortField,
    WorkType,
)
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import (
    Album,
    Artist,
    ArtistSummary,
    ClassicalComposer,
    ClassicalComposerFilters,
    ClassicalPerformer,
    ClassicalPerformerFilters,
    ClassicalWorkEntry,
    ClassicalWorkFilters,
    Credit,
    ItemMapping,
    MediaItemImage,
    MediaItemMetadata,
    ProviderMapping,
    Recording,
    Track,
    UniqueList,
    Work,
    WorkSummary,
)

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_GENRES,
)
from music_assistant.controllers.music.classical import (
    _Appearance,
    _build_performance,
    _build_recording,
    _group_performances,
    _Performance,
    _performer_row,
    _with_title_work_type,
    _WorkTrack,
)
from music_assistant.helpers.api import parse_arguments
from music_assistant.mass import MusicAssistant
from tests.common import set_music_source_access

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

PROVIDER_INSTANCE = "classical_api_instance"
# a music source private to another user
HIDDEN_INSTANCE = "classical_hidden_instance"
GET_CURRENT_USER = "music_assistant.controllers.music.media.base.get_current_user"
CLASSICAL_GET_CURRENT_USER = "music_assistant.controllers.music.classical.get_current_user"
LISTENER = User(user_id="listener", username="listener", role=UserRole.USER)
WORK_SOURCE = "works_source"
TRACK_DURATION = 300
# the artwork of the seeded artists that have any
ARTIST_IMAGES = {
    "beethoven": [(ImageType.THUMB, "beethoven_thumb"), (ImageType.FANART, "beethoven_fanart")],
    "karajan": [(ImageType.FANART, "karajan_fanart")],
}
WORK_COMPOSERS = {
    "Eroica": "beethoven",
    "Fifth": "beethoven",
    "Emperor": "beethoven",
    "Spring": "vivaldi",
    "Gloria": "vivaldi",
    "Pop": "pop_writer",
}


@dataclass
class _Library:
    """The seeded library, by short names."""

    mass: MusicAssistant
    artists: dict[str, Artist] = field(default_factory=dict)
    albums: dict[str, Album] = field(default_factory=dict)
    works: dict[str, Work] = field(default_factory=dict)
    tracks: dict[str, Track] = field(default_factory=dict)

    def artist_id(self, name: str) -> str:
        """Return the library id of a seeded artist."""
        return self.artists[name].item_id

    def work_id(self, name: str) -> str:
        """Return the library id of a seeded work."""
        return self.works[name].item_id


@pytest.fixture(scope="module")
async def library(music_mass_module: MusicAssistant) -> _Library:
    """Return a module-scoped database-only instance with a seeded classical library."""
    return await _seed_library(music_mass_module)


async def test_commands_are_registered(library: _Library) -> None:
    """Every classical command is served to library readers."""
    for command in (
        "composers",
        "composer_filters",
        "performers",
        "performer_filters",
        "works",
        "work_filters",
        "recordings",
        "other_tracks",
        "link_progress",
    ):
        handler = library.mass.command_handlers[f"music/classical/{command}"]
        assert handler.required_scope == Scope.LIBRARY_READ


async def test_composers(library: _Library) -> None:
    """Composers of classical tracks are listed with their work and recording counts."""
    rows = await library.mass.music.classical.composers()

    assert [(x.artist.name, x.work_count, x.recording_count) for x in rows] == [
        ("Ludwig van Beethoven", 3, 7),
        ("Antonio Vivaldi", 2, 2),
    ]


@pytest.mark.parametrize(
    ("sort_field", "sort_direction", "expected"),
    [
        (None, None, ["Ludwig van Beethoven", "Antonio Vivaldi"]),
        (SortField.SORT_NAME, SortDirection.DESC, ["Antonio Vivaldi", "Ludwig van Beethoven"]),
        (SortField.NAME, None, ["Antonio Vivaldi", "Ludwig van Beethoven"]),
        (SortField.NAME, SortDirection.DESC, ["Ludwig van Beethoven", "Antonio Vivaldi"]),
        (SortField.WORK_COUNT, SortDirection.ASC, ["Antonio Vivaldi", "Ludwig van Beethoven"]),
        (SortField.WORK_COUNT, None, ["Ludwig van Beethoven", "Antonio Vivaldi"]),
    ],
)
async def test_composer_sorts(
    library: _Library,
    sort_field: SortField | None,
    sort_direction: SortDirection | None,
    expected: list[str],
) -> None:
    """Composers sort by sort name, name or work count, the most works first by default."""
    rows = await library.mass.music.classical.composers(
        sort_field=sort_field, sort_direction=sort_direction
    )

    assert [x.artist.name for x in rows] == expected


async def test_performers(library: _Library) -> None:
    """Performers of classical tracks are listed with their roles and counts."""
    rows = await library.mass.music.classical.performers()

    assert [
        (x.artist.name, x.main_role, x.roles, x.work_count, x.recording_count) for x in rows
    ] == [
        ("Berliner Philharmoniker", ArtistRole.ORCHESTRA, [ArtistRole.ORCHESTRA], 3, 4),
        ("claudio abbado", ArtistRole.CONDUCTOR, [ArtistRole.CONDUCTOR], 1, 1),
        (
            "Daniel Barenboim",
            ArtistRole.CONDUCTOR,
            [ArtistRole.CONDUCTOR, ArtistRole.SOLOIST],
            1,
            1,
        ),
        (
            "Gardiner Singers",
            ArtistRole.CHOIR,
            [ArtistRole.CHOIR, ArtistRole.PERFORMER],
            1,
            1,
        ),
        ("Herbert von Karajan", ArtistRole.CONDUCTOR, [ArtistRole.CONDUCTOR], 3, 3),
        ("Leonard Bernstein", ArtistRole.CONDUCTOR, [ArtistRole.CONDUCTOR], 2, 3),
        ("Wiener Philharmoniker", ArtistRole.ORCHESTRA, [ArtistRole.ORCHESTRA], 3, 4),
    ]


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (
            ArtistRole.CONDUCTOR,
            ["claudio abbado", "Daniel Barenboim", "Herbert von Karajan", "Leonard Bernstein"],
        ),
        (ArtistRole.ORCHESTRA, ["Berliner Philharmoniker", "Wiener Philharmoniker"]),
        (ArtistRole.SOLOIST, ["Daniel Barenboim"]),
        (ArtistRole.CHOIR, ["Gardiner Singers"]),
        (ArtistRole.PERFORMER, ["Gardiner Singers"]),
        (ArtistRole.ENSEMBLE, []),
        (ArtistRole.COMPOSER, []),
    ],
)
async def test_performers_role_filter(
    library: _Library, role: ArtistRole, expected: list[str]
) -> None:
    """The role filter keeps the performers holding that role among their roles."""
    rows = await library.mass.music.classical.performers(role=[role])

    assert [x.artist.name for x in rows] == expected


@pytest.mark.parametrize(
    ("sort_field", "sort_direction", "expected"),
    [
        (
            SortField.RECORDING_COUNT,
            SortDirection.ASC,
            [
                "claudio abbado",
                "Daniel Barenboim",
                "Gardiner Singers",
                "Herbert von Karajan",
                "Leonard Bernstein",
                "Berliner Philharmoniker",
                "Wiener Philharmoniker",
            ],
        ),
        (
            SortField.RECORDING_COUNT,
            None,
            [
                "Berliner Philharmoniker",
                "Wiener Philharmoniker",
                "Herbert von Karajan",
                "Leonard Bernstein",
                "claudio abbado",
                "Daniel Barenboim",
                "Gardiner Singers",
            ],
        ),
    ],
)
async def test_performer_sorts(
    library: _Library,
    sort_field: SortField,
    sort_direction: SortDirection | None,
    expected: list[str],
) -> None:
    """Performers sort by name or recording count (the most first by default), ties by name."""
    rows = await library.mass.music.classical.performers(
        sort_field=sort_field, sort_direction=sort_direction
    )

    assert [x.artist.name for x in rows] == expected


async def test_composer_and_performer_fanart(library: _Library) -> None:
    """Composer and performer rows carry the artist's fanart next to the summary thumb."""
    composers = {x.artist.name: x for x in await library.mass.music.classical.composers()}
    performers = {x.artist.name: x for x in await library.mass.music.classical.performers()}

    beethoven = composers["Ludwig van Beethoven"]
    assert beethoven.fanart is not None
    assert (beethoven.fanart.type, beethoven.fanart.path) == (
        ImageType.FANART,
        "beethoven_fanart",
    )
    assert beethoven.artist.metadata.images is not None
    assert [(x.type, x.path) for x in beethoven.artist.metadata.images] == [
        (ImageType.THUMB, "beethoven_thumb")
    ]
    assert composers["Antonio Vivaldi"].fanart is None
    karajan = performers["Herbert von Karajan"]
    assert karajan.fanart is not None
    assert karajan.fanart.path == "karajan_fanart"
    assert karajan.artist.metadata.images is None
    assert performers["Berliner Philharmoniker"].fanart is None


@pytest.mark.parametrize(
    ("artist", "is_composer", "is_performer"),
    [
        ("beethoven", True, False),
        ("karajan", False, True),
        ("barenboim", False, True),
        ("label", False, False),
        ("pop_writer", False, False),
    ],
)
async def test_artist_filter(
    library: _Library, artist: str, is_composer: bool, is_performer: bool
) -> None:
    """The artist filter returns the artist's own listing row, or nothing."""
    classical = library.mass.music.classical
    artist_id = library.artist_id(artist)

    composers = await classical.composers(artist_id=artist_id)
    performers = await classical.performers(artist_id=artist_id)

    assert composers == [x for x in await classical.composers() if x.artist.item_id == artist_id]
    assert performers == [x for x in await classical.performers() if x.artist.item_id == artist_id]
    assert len(composers) == is_composer
    assert len(performers) == is_performer


async def test_artist_filter_combines_with_other_filters(library: _Library) -> None:
    """The artist filter applies on top of the role filter, the search and every sort."""
    classical = library.mass.music.classical
    beethoven = library.artist_id("beethoven")
    barenboim = library.artist_id("barenboim")

    soloist = await classical.performers(role=[ArtistRole.SOLOIST], artist_id=barenboim)
    assert [(x.artist.name, x.roles) for x in soloist] == [
        ("Daniel Barenboim", [ArtistRole.CONDUCTOR, ArtistRole.SOLOIST])
    ]
    assert await classical.performers(role=[ArtistRole.ORCHESTRA], artist_id=barenboim) == []
    assert await classical.composers(search="vivald", artist_id=beethoven) == []
    by_works = await classical.composers(sort_field=SortField.WORK_COUNT, artist_id=beethoven)
    assert [(x.artist.name, x.work_count) for x in by_works] == [("Ludwig van Beethoven", 3)]
    by_recordings = await classical.performers(
        sort_field=SortField.RECORDING_COUNT, sort_direction=SortDirection.ASC, artist_id=barenboim
    )
    assert [(x.artist.name, x.recording_count) for x in by_recordings] == [("Daniel Barenboim", 1)]


async def test_works(library: _Library) -> None:
    """
    Classical works are listed by composer and catalogue number with recording counts.

    A work without a stored type shows the type its title names.
    """
    rows = await library.mass.music.classical.works()

    assert [(x.work.name, x.recording_count) for x in rows] == [
        ("Symphony No. 3 Eroica", 4),
        ("Symphony No. 5", 2),
        ("Piano Concerto No. 5", 1),
        ("Violin Concerto Spring", 1),
        ("Gloria", 1),
    ]
    assert rows[0].work.catalog_numbers == ["Op. 55"]
    assert [x.name for x in rows[0].work.composers] == ["Ludwig van Beethoven"]
    # works without a stored type, alone or listed, show the type their title names
    assert [x.work.work_type for x in rows] == [
        WorkType.SYMPHONY,
        WorkType.SYMPHONY,
        WorkType.CONCERTO,
        WorkType.CONCERTO,
        None,
    ]
    work = await library.mass.music.classical.work(library.work_id("Eroica"))
    assert work.work_type == WorkType.SYMPHONY
    stored = await library.mass.music.works.get_library_item(library.work_id("Eroica"))
    assert stored.work_type is None


@pytest.mark.parametrize(
    ("sort_field", "sort_direction", "expected"),
    [
        (None, None, ["Eroica", "Fifth", "Emperor", "Spring", "Gloria"]),
        (
            SortField.COMPOSER,
            SortDirection.DESC,
            ["Spring", "Gloria", "Eroica", "Fifth", "Emperor"],
        ),
        (SortField.NAME, None, ["Gloria", "Emperor", "Eroica", "Fifth", "Spring"]),
        (SortField.NAME, SortDirection.DESC, ["Spring", "Fifth", "Eroica", "Emperor", "Gloria"]),
        (SortField.YEAR, None, ["Spring", "Eroica", "Fifth", "Emperor", "Gloria"]),
        (SortField.YEAR, SortDirection.DESC, ["Emperor", "Fifth", "Eroica", "Spring", "Gloria"]),
        (
            SortField.RECORDING_COUNT,
            SortDirection.ASC,
            ["Emperor", "Spring", "Gloria", "Fifth", "Eroica"],
        ),
        (SortField.RECORDING_COUNT, None, ["Eroica", "Fifth", "Emperor", "Spring", "Gloria"]),
    ],
)
async def test_work_sorts(
    library: _Library,
    sort_field: SortField | None,
    sort_direction: SortDirection | None,
    expected: list[str],
) -> None:
    """Works sort by composer, title, year composed or recording count."""
    rows = await library.mass.music.classical.works(
        sort_field=sort_field, sort_direction=sort_direction
    )

    assert [x.work.item_id for x in rows] == [library.work_id(x) for x in expected]


async def test_composer_works_sort_by_their_catalogue(music_mass: MusicAssistant) -> None:
    """A composer's works, also one shared with a composer listed first, follow the catalogue."""
    lib = await _new_library(music_mass, ("beethoven", "bach", "karajan"))
    for key, catalog, composers in (
        ("Seventh", "Op. 92", ("beethoven",)),
        ("Eroica", "Op. 55", ("beethoven",)),
        ("Fur Elise", "WoO 59", ("beethoven",)),
        ("Shared", "Op. 61", ("bach", "beethoven")),
        ("Untitled", None, ("beethoven",)),
    ):
        lib.works[key] = await lib.mass.music.works.add_item_to_library(
            Work(
                item_id=uuid4().hex,
                provider=WORK_SOURCE,
                name=key,
                provider_mappings=set(),
                composers=UniqueList(lib.artists[x] for x in composers),
                catalog_numbers=[catalog] if catalog else [],
            )
        )
        await _add_track(lib, key.lower(), key, "karajan")

    rows = await lib.mass.music.classical.works(composer_id=lib.artist_id("beethoven"))

    assert [x.work.item_id for x in rows] == [
        lib.work_id(x) for x in ("Eroica", "Shared", "Seventh", "Fur Elise", "Untitled")
    ]


@pytest.mark.parametrize(
    ("performer", "sort_field", "sort_direction", "expected"),
    [
        ("bpo", SortField.COMPOSER, None, [("Eroica", 2), ("Fifth", 1), ("Emperor", 1)]),
        (
            "vpo",
            SortField.RECORDING_COUNT,
            SortDirection.ASC,
            [("Fifth", 1), ("Spring", 1), ("Eroica", 2)],
        ),
    ],
)
async def test_works_by_performer(
    library: _Library,
    performer: str,
    sort_field: SortField,
    sort_direction: SortDirection | None,
    expected: list[tuple[str, int]],
) -> None:
    """The performer filter keeps their works and counts only their recordings."""
    rows = await library.mass.music.classical.works(
        performer_id=library.artist_id(performer),
        sort_field=sort_field,
        sort_direction=sort_direction,
    )

    assert [(x.work.item_id, x.recording_count) for x in rows] == [
        (library.work_id(name), count) for name, count in expected
    ]


async def test_works_by_performer_name_their_co_performers(music_mass: MusicAssistant) -> None:
    """
    A performer's row names the others on their recordings of the work, each once.

    Orchestras and ensembles come first, then conductors, soloists, choirs and other performers,
    each by credit position, then in the order of the recordings. The performer and the
    recordings they are not on are left out.
    """
    lib = await _co_performer_library(music_mass)

    [row] = await music_mass.music.classical.works(performer_id=lib.artist_id("singer"))

    assert row.recording_count == 2
    assert row.co_performers == [
        "Bpo",
        "Vpo",
        "Consort",
        "Abbado",
        "Karajan",
        "Tenor",
        "Chorus",
        "Pianist",
    ]


async def test_works_by_performer_tell_if_the_user_likes_all_their_tracks(
    music_mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A performer's row is liked when the calling user likes every track of their recordings."""
    lib = await _co_performer_library(music_mass)
    other = User(user_id="other", username="other", role=UserRole.USER)
    favorites = music_mass.music.favorites
    for key in ("karajan_requiem", "abbado_requiem"):
        await favorites.set(MediaType.TRACK, int(lib.tracks[key].item_id), True, [other.user_id])
    await favorites.set(
        MediaType.TRACK, int(lib.tracks["karajan_requiem"].item_id), True, [LISTENER.user_id]
    )
    liked: dict[str, bool | None] = {}
    for user in (LISTENER, other):
        monkeypatch.setattr(GET_CURRENT_USER, lambda user=user: user)
        [row] = await music_mass.music.classical.works(performer_id=lib.artist_id("singer"))
        liked[user.user_id] = row.liked

    assert liked == {LISTENER.user_id: False, other.user_id: True}


async def test_works_without_a_performer_name_no_co_performers(music_mass: MusicAssistant) -> None:
    """Without a performer filter, rows name no co-performers and carry no liked state."""
    await _co_performer_library(music_mass)

    [row] = await music_mass.music.classical.works()

    assert (row.recording_count, row.co_performers, row.liked) == (3, [], None)


@pytest.mark.parametrize(
    ("year_from", "year_to", "expected"),
    [
        (1805, None, ["Fifth", "Emperor"]),
        (None, 1805, ["Eroica", "Spring"]),
        (1800, 1808, ["Eroica", "Fifth"]),
        (1808, 1800, ["Eroica", "Fifth"]),
    ],
)
async def test_works_year_range(
    library: _Library, year_from: int | None, year_to: int | None, expected: list[str]
) -> None:
    """
    The year range filters on the year composed and drops works without one.

    A range given the wrong way round is read the right way round.
    """
    rows = await library.mass.music.classical.works(year_from=year_from, year_to=year_to)

    assert [x.work.item_id for x in rows] == [library.work_id(x) for x in expected]


@pytest.mark.parametrize(
    ("favorite", "composers", "performers"),
    [
        (True, ["Ludwig van Beethoven"], ["Herbert von Karajan"]),
        (False, ["Antonio Vivaldi"], ["Berliner Philharmoniker"]),
    ],
)
async def test_favorite_filter(
    library: _Library,
    monkeypatch: pytest.MonkeyPatch,
    favorite: bool,
    composers: list[str],
    performers: list[str],
) -> None:
    """The favorite filter keeps the calling user's likes or dislikes, not those of others."""
    favorites = library.mass.music.favorites
    states = {"beethoven": True, "karajan": True, "vivaldi": False, "bpo": False}
    for key, state in states.items():
        await favorites.set(
            MediaType.ARTIST, int(library.artist_id(key)), state, [LISTENER.user_id]
        )
    await favorites.set(MediaType.ARTIST, int(library.artist_id("bernstein")), favorite, ["other"])
    monkeypatch.setattr(GET_CURRENT_USER, lambda: LISTENER)
    classical = library.mass.music.classical
    try:
        composer_rows = await classical.composers(favorite=favorite)
        performer_rows = await classical.performers(favorite=favorite)
    finally:
        for key in (*states, "bernstein"):
            await favorites.set(
                MediaType.ARTIST, int(library.artist_id(key)), None, [LISTENER.user_id, "other"]
            )

    assert [x.artist.name for x in composer_rows] == composers
    assert [x.artist.name for x in performer_rows] == performers


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ([Period.CLASSICAL], ["Ludwig van Beethoven"]),
        ([Period.BAROQUE, Period.CLASSICAL], ["Ludwig van Beethoven"]),
        ([Period.BAROQUE], []),
        ([], ["Ludwig van Beethoven", "Antonio Vivaldi"]),
    ],
)
async def test_composers_period_filter(
    library: _Library, period: list[Period], expected: list[str]
) -> None:
    """The period filter keeps the composers of any chosen period, an empty one keeps all."""
    rows = await library.mass.music.classical.composers(period=period)

    assert [x.artist.name for x in rows] == expected


@pytest.mark.parametrize(
    ("roles", "expected"),
    [
        ([ArtistRole.CHOIR, ArtistRole.SOLOIST], ["Daniel Barenboim", "Gardiner Singers"]),
        (
            [ArtistRole.ORCHESTRA, ArtistRole.ENSEMBLE],
            ["Berliner Philharmoniker", "Wiener Philharmoniker"],
        ),
        ([ArtistRole.ENSEMBLE], []),
    ],
)
async def test_performers_roles_filter(
    library: _Library, roles: list[ArtistRole], expected: list[str]
) -> None:
    """A role filter with several roles keeps the performers holding any of them."""
    classical = library.mass.music.classical
    rows = await classical.performers(role=roles)

    assert [x.artist.name for x in rows] == expected
    assert await classical.performers(role=[]) == await classical.performers()


async def test_performers_role_list_from_the_api(library: _Library) -> None:
    """A list of roles sent over the api reaches the role filter as those roles."""
    handler = library.mass.command_handlers["music/classical/performers"]

    arguments = parse_arguments(
        handler.signature, handler.type_hints, {"role": ["conductor", "choir"]}
    )

    assert arguments["role"] == [ArtistRole.CONDUCTOR, ArtistRole.CHOIR]


@pytest.mark.parametrize(
    ("search", "expected"),
    [
        # "bach mass" style, the composer followed by the title
        ("beethoven symphony", ["Eroica", "Fifth"]),
        ("RV 269", ["Spring"]),
        # the title followed by the catalogue number
        ("concerto spring rv", ["Spring"]),
        ("vivaldi rv", []),
    ],
)
async def test_works_search(library: _Library, search: str, expected: list[str]) -> None:
    """The works search looks in the composer, title and catalogue number taken together."""
    rows = await library.mass.music.classical.works(search=search)

    assert [x.work.item_id for x in rows] == [library.work_id(x) for x in expected]


@pytest.mark.parametrize(
    ("work_type", "work_type_unknown", "expected"),
    [
        ([WorkType.CONCERTO], False, ["Emperor", "Spring"]),
        ([WorkType.SYMPHONY, WorkType.OPERA], False, ["Eroica", "Fifth"]),
        ([], True, ["Gloria"]),
        ([WorkType.CONCERTO], True, ["Emperor", "Spring", "Gloria"]),
        ([], False, ["Eroica", "Fifth", "Emperor", "Spring", "Gloria"]),
    ],
)
async def test_works_type_filter(
    library: _Library,
    work_type: list[WorkType],
    work_type_unknown: bool,
    expected: list[str],
) -> None:
    """The type filter matches the type a row shows, also one its title names, or no type."""
    rows = await library.mass.music.classical.works(
        work_type=work_type, work_type_unknown=work_type_unknown
    )

    assert [x.work.item_id for x in rows] == [library.work_id(x) for x in expected]


async def test_filter_values(library: _Library) -> None:
    """The filter value commands return the values found across each whole list."""
    classical = library.mass.music.classical

    assert await classical.composer_filters() == ClassicalComposerFilters(
        periods=[Period.CLASSICAL]
    )
    assert await classical.performer_filters() == ClassicalPerformerFilters(
        roles=[
            ArtistRole.CONDUCTOR,
            ArtistRole.ORCHESTRA,
            ArtistRole.CHOIR,
            ArtistRole.SOLOIST,
            ArtistRole.PERFORMER,
        ]
    )
    assert await classical.work_filters() == ClassicalWorkFilters(
        work_types=[WorkType.SYMPHONY, WorkType.CONCERTO],
        work_type_unknown=True,
        earliest_year=1720,
        latest_year=1809,
    )


async def test_recordings_group_movements(library: _Library) -> None:
    """
    Movements group into recordings and a reissued movement collapses into its original.

    The recordings hold full tracks, with their credits, work and movement.
    """
    recordings = await library.mass.music.classical.recordings(library.work_id("Fifth"))

    karajan, bernstein = recordings
    assert karajan.year == 1962
    assert _track_names(karajan) == ["Karajan I", "Karajan II", "Karajan III", "Karajan IV"]
    # the reissue on the compilation is shown from the original album
    assert {x.album.item_id for x in karajan.tracks if x.album} == {
        library.albums["karajan"].item_id
    }
    assert _album_ids(karajan) == [library.albums[x].item_id for x in ("karajan", "compilation")]
    assert [(x.artist.name, x.role) for x in karajan.credits] == [
        ("Herbert von Karajan", ArtistRole.CONDUCTOR),
        ("Berliner Philharmoniker", ArtistRole.ORCHESTRA),
    ]
    assert karajan.duration == 4 * TRACK_DURATION
    assert karajan.work.item_id == library.work_id("Fifth")
    assert karajan.parent_work is None
    # the tracks are full tracks, with their credits, work and movement
    first = karajan.tracks[0]
    assert (first.movement_number, first.track_number) == (1, 1)
    assert first.work is not None
    assert first.work.item_id == library.work_id("Fifth")
    assert ArtistRole.COMPOSER in {x.role for x in first.credits}
    # without a recording id the same movement, performers and year collapse as well
    assert bernstein.year == 1979
    assert _track_names(bernstein) == [
        "Bernstein I",
        "Bernstein II",
        "Bernstein III",
        "Bernstein IV",
    ]
    assert _album_ids(bernstein) == [library.albums[x].item_id for x in ("bernstein", "mixed")]


async def test_recordings_order(library: _Library) -> None:
    """Recordings come by year, then conductor, then ensemble, undated last."""
    recordings = await library.mass.music.classical.recordings(library.work_id("Eroica"))

    assert [(x.year, _names(x, ArtistRole.CONDUCTOR), _ensembles(x)) for x in recordings] == [
        (1979, ["claudio abbado"], ["Wiener Philharmoniker"]),
        (1979, ["Leonard Bernstein"], ["Berliner Philharmoniker"]),
        (1979, ["Leonard Bernstein"], ["Wiener Philharmoniker"]),
        (None, ["Herbert von Karajan"], ["Berliner Philharmoniker"]),
    ]
    # two performances on one album stay apart
    assert _album_ids(recordings[0]) == _album_ids(recordings[1])
    assert recordings[3].albums == []


async def test_recordings_by_performer(library: _Library) -> None:
    """The performer filter keeps the recordings that performer performs on."""
    classical = library.mass.music.classical
    eroica = library.work_id("Eroica")

    bernstein = await classical.recordings(eroica, performer_id=library.artist_id("bernstein"))
    assert [_ensembles(x) for x in bernstein] == [
        ["Berliner Philharmoniker"],
        ["Wiener Philharmoniker"],
    ]
    bpo = await classical.recordings(eroica, performer_id=library.artist_id("bpo"))
    assert [_names(x, ArtistRole.CONDUCTOR) for x in bpo] == [
        ["Leonard Bernstein"],
        ["Herbert von Karajan"],
    ]


async def test_recordings_of_a_performer_across_works(library: _Library) -> None:
    """Without a work, the recordings of every work the performer performs on come per work."""
    classical = library.mass.music.classical
    bpo = library.artist_id("bpo")

    recordings = await classical.recordings(performer_id=bpo)

    works = [x.work.item_id for x in await classical.works(performer_id=bpo, limit=0)]
    expected: list[str] = []
    for work in works:
        expected.extend(x.key for x in await classical.recordings(work, performer_id=bpo))
    assert [x.key for x in recordings] == expected
    assert len({x.work.item_id for x in recordings}) == len(works) > 1


async def test_recordings_need_a_work_or_a_performer(library: _Library) -> None:
    """Asking for recordings without a work or a performer is refused."""
    with pytest.raises(InvalidDataError):
        await library.mass.music.classical.recordings()


async def test_recording_album_order_tie(library: _Library) -> None:
    """On a tie in movements the earliest album comes first and supplies the tracks."""
    (recording,) = await library.mass.music.classical.recordings(library.work_id("Emperor"))

    assert _album_ids(recording) == [library.albums[x].item_id for x in ("emperor", "emperor_box")]
    assert [x.item_id for x in recording.tracks] == [
        library.tracks[x].item_id for x in ("emperor_1", "emperor_2")
    ]
    assert recording.year == 1983
    assert [(x.artist.name, x.role, x.instrument) for x in recording.credits] == [
        ("Daniel Barenboim", ArtistRole.CONDUCTOR, None),
        ("Berliner Philharmoniker", ArtistRole.ORCHESTRA, None),
        ("Daniel Barenboim", ArtistRole.SOLOIST, "piano"),
    ]


async def test_recording_track_shown_on_its_album(library: _Library) -> None:
    """A track on several albums is shown on the album the recording is taken from."""
    (recording,) = await library.mass.music.classical.recordings(library.work_id("Spring"))

    first = recording.tracks[0]
    assert first.album is not None
    assert first.album.item_id == library.albums["spring"].item_id
    assert first.track_number == 1
    assert _album_ids(recording) == [library.albums[x].item_id for x in ("spring", "spring_extra")]


def _recording_albums(*albums: tuple[str, str, int | None]) -> list[tuple[str, str, int | None]]:
    """Return the (id, name, year) of the albums a recording lists, from albums given in order."""
    mappings = {
        int(item_id): ItemMapping(
            item_id=item_id, provider="library", name=name, media_type=MediaType.ALBUM, year=year
        )
        for item_id, name, year in albums
    }
    work = ItemMapping(item_id="1", provider="library", name="Work", media_type=MediaType.WORK)
    performance = _Performance(
        key="k",
        year=None,
        performer_ids=frozenset(),
        album_ids=[int(x) for x, _, _ in albums],
        tracks=[],
    )
    recording = _build_recording(work, performance, {}, mappings)
    return [(x.item_id, x.name, x.year) for x in recording.albums]


@pytest.mark.parametrize(
    ("albums", "expected"),
    [
        pytest.param(
            [("1", "A", 2000), ("2", "B", 2001)],
            [("1", "A", 2000), ("2", "B", 2001)],
            id="distinct",
        ),
        pytest.param(
            [("1", "A", 2000), ("2", "A", 2000), ("3", "B", 2000)],
            [("1", "A", 2000), ("3", "B", 2000)],
            id="same name and year, first kept",
        ),
        pytest.param(
            [("1", "A", 2000), ("2", "A", 2001)],
            [("1", "A", 2000), ("2", "A", 2001)],
            id="same name, different years",
        ),
        pytest.param(
            [("1", "A", None), ("2", "A", None), ("3", "A", 2000)],
            [("1", "A", None), ("3", "A", 2000)],
            id="same name without a year",
        ),
        pytest.param(
            [("2", "Z", 1999), ("1", "A", 2000)],
            [("2", "Z", 1999), ("1", "A", 2000)],
            id="order kept",
        ),
        pytest.param(
            [("5", "A", None), ("5", "A", None)],
            [("5", "A", None)],
            id="same album id",
        ),
    ],
)
def test_recording_lists_each_album_once(
    albums: list[tuple[str, str, int | None]], expected: list[tuple[str, str, int | None]]
) -> None:
    """A recording lists an album once, however its repeats match, in the order given."""
    assert _recording_albums(*albums) == expected


async def test_recording_key_is_stable(library: _Library) -> None:
    """Every recording of a work has its own key, the same on every request."""
    classical = library.mass.music.classical
    eroica = library.work_id("Eroica")

    keys = [x.key for x in await classical.recordings(eroica)]
    assert len(set(keys)) == len(keys)
    assert keys == [x.key for x in await classical.recordings(eroica)]


async def test_non_classical_tracks_never_count(library: _Library) -> None:
    """The pop track with a work, a composer and a conductor shows up nowhere."""
    classical = library.mass.music.classical

    assert "Pop Writer" not in {x.artist.name for x in await classical.composers()}
    assert {"Pop Star", "Classical Label"}.isdisjoint(
        {x.artist.name for x in await classical.performers()}
    )
    assert library.work_id("Pop") not in {x.work.item_id for x in await classical.works()}
    assert await classical.recordings(library.work_id("Pop")) == []
    vivaldi_tracks = await classical.other_tracks(library.artist_id("vivaldi"), as_composer=True)
    assert "Pop B-Side" not in {x.name for x in vivaldi_tracks}


@pytest.mark.parametrize(
    ("artist", "as_composer", "expected"),
    [
        ("vivaldi", True, ["Aria", "Barcarolle", "Cantilena"]),
        ("vivaldi", False, []),
        ("gardiner", False, ["Aria", "Barcarolle", "Cantilena"]),
        ("gardiner", True, []),
        ("label", False, ["Aria", "Barcarolle", "Cantilena"]),
        ("beethoven", True, []),
    ],
)
async def test_other_tracks(
    library: _Library, artist: str, as_composer: bool, expected: list[str]
) -> None:
    """Other tracks are the classical tracks without a work holding the artist's credit."""
    tracks = await library.mass.music.classical.other_tracks(
        library.artist_id(artist), as_composer=as_composer
    )

    assert [x.name for x in tracks] == expected
    assert all(x.work is None and x.is_classical for x in tracks)


@pytest.mark.parametrize(
    ("sort_field", "sort_direction", "expected"),
    [
        (SortField.NAME, SortDirection.DESC, ["Cantilena", "Barcarolle", "Aria"]),
        (SortField.YEAR, None, ["Cantilena", "Barcarolle", "Aria"]),
        (SortField.YEAR, SortDirection.DESC, ["Barcarolle", "Cantilena", "Aria"]),
        (SortField.TIMESTAMP_ADDED, SortDirection.ASC, ["Barcarolle", "Aria", "Cantilena"]),
        (SortField.TIMESTAMP_ADDED, None, ["Cantilena", "Aria", "Barcarolle"]),
    ],
)
async def test_other_track_sorts(
    library: _Library,
    sort_field: SortField,
    sort_direction: SortDirection | None,
    expected: list[str],
) -> None:
    """Other tracks sort by title, album year or date added, newest first by default."""
    tracks = await library.mass.music.classical.other_tracks(
        library.artist_id("vivaldi"),
        as_composer=True,
        sort_field=sort_field,
        sort_direction=sort_direction,
    )

    assert [x.name for x in tracks] == expected


@pytest.mark.parametrize(
    ("command", "search", "expected"),
    [
        ("composers", "vivald", ["Antonio Vivaldi"]),
        ("performers", "philharmoniker", ["Berliner Philharmoniker", "Wiener Philharmoniker"]),
        ("works", "symphony", ["Symphony No. 3 Eroica", "Symphony No. 5"]),
        ("other_tracks", "barca", ["Barcarolle"]),
    ],
)
async def test_listings_search(
    library: _Library, command: str, search: str, expected: list[str]
) -> None:
    """Composers and performers filter on their name, works and other tracks on their title."""
    rows = await _listing(library, command)(search=search)

    assert [_row_name(x) for x in rows] == expected


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        ("composers", {"sort_field": SortField.WORK_COUNT, "sort_direction": SortDirection.ASC}),
        ("performers", {"role": [ArtistRole.CONDUCTOR], "sort_field": SortField.RECORDING_COUNT}),
        ("works", {"sort_field": SortField.RECORDING_COUNT}),
        ("works", {"search": "beethoven"}),
        ("works", {"work_type": [WorkType.CONCERTO, WorkType.SYMPHONY]}),
        ("performers", {"role": [ArtistRole.CONDUCTOR, ArtistRole.CHOIR]}),
        ("other_tracks", {}),
    ],
)
async def test_listings_page_after_sorting(
    library: _Library, command: str, kwargs: dict[str, Any]
) -> None:
    """A page of a listing is that part of the whole listing, sorted and filtered the same way."""
    listing = _listing(library, command)

    whole = await listing(**kwargs)
    page = await listing(**kwargs, limit=1, offset=1)

    assert page == whole[1:2]
    assert page


@pytest.fixture(name="restricted_mass")
def restricted_mass_fixture(music_mass_class: MusicAssistant) -> MusicAssistant:
    """Return a class-scoped database-only instance with a music source private to another user."""
    set_music_source_access(
        music_mass_class,
        {
            PROVIDER_INSTANCE: None,
            HIDDEN_INSTANCE: ProviderAccess(owner="owner", sharing=ProviderSharing.PRIVATE),
        },
    )
    return music_mass_class


class TestListingVisibility:
    """The classical listings only show the library tracks the calling user sees."""

    async def test_listings_skip_tracks_the_user_does_not_see(
        self, restricted_mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hidden and not in library tracks add no composer, performer, work or recording."""
        mass = restricted_mass
        visible = await _add_mapped_track(mass, "Visible", _mapping())
        composer = next(x.artist for x in visible.credits if x.role == ArtistRole.COMPOSER)
        hidden = await _add_mapped_track(mass, "Hidden", _mapping(HIDDEN_INSTANCE))
        assert hidden.work is not None
        await _add_mapped_track(mass, "Not In Library", _mapping(in_library=False))
        for name, mapping in (
            ("Visible Other", _mapping()),
            ("Hidden Other", _mapping(HIDDEN_INSTANCE)),
            ("Not In Library Other", _mapping(in_library=False)),
        ):
            await _add_mapped_track(mass, name, mapping, composer=composer, work=False)
        monkeypatch.setattr(GET_CURRENT_USER, lambda: LISTENER)
        classical = mass.music.classical

        composers = await classical.composers()
        performers = await classical.performers()
        works = await classical.works()
        hidden_recordings = await classical.recordings(hidden.work.item_id)
        other_tracks = await classical.other_tracks(composer.item_id, as_composer=True)

        assert [x.artist.name for x in composers] == ["Composer Visible"]
        assert [x.artist.name for x in performers] == [
            "Conductor Visible",
            "Conductor Visible Other",
        ]
        assert [x.work.name for x in works] == ["Work Visible"]
        assert hidden_recordings == []
        assert [x.name for x in other_tracks] == ["Visible Other"]


class TestFanartVisibility:
    """The fanart of a listing row prefers the artwork of the music sources the user sees."""

    async def test_fanart_skips_hidden_sources(
        self, restricted_mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fanart of a hidden music source gives way to one of a visible source."""
        mass = restricted_mass
        composer = await mass.music.artists.add_item_to_library(
            Artist(
                item_id="0",
                provider="library",
                name="Composer Fanart",
                provider_mappings={_mapping()},
                metadata=MediaItemMetadata(
                    images=UniqueList(
                        [
                            _image(ImageType.FANART, "hidden_fanart", HIDDEN_INSTANCE),
                            _image(ImageType.FANART, "visible_fanart"),
                        ]
                    )
                ),
            )
        )
        await _add_mapped_track(mass, "Visible", _mapping(), composer=composer)

        for_everyone = (await mass.music.classical.composers())[0].fanart
        monkeypatch.setattr(GET_CURRENT_USER, lambda: LISTENER)
        monkeypatch.setattr(CLASSICAL_GET_CURRENT_USER, lambda: LISTENER)
        for_listener = (await mass.music.classical.composers())[0].fanart

        assert for_everyone is not None
        assert for_everyone.path == "hidden_fanart"
        assert for_listener is not None
        assert for_listener.path == "visible_fanart"


class TestArtistFilter:
    """The artist filter of the composer and performer listings."""

    async def test_artist_both_composer_and_performer(
        self, restricted_mass: MusicAssistant
    ) -> None:
        """An artist who composes and performs gets a row in both listings."""
        mass = restricted_mass
        performed = await _add_mapped_track(mass, "Performed", _mapping())
        artist = _credited(performed, ArtistRole.CONDUCTOR)
        await _add_mapped_track(mass, "Composed", _mapping(), composer=artist)
        classical = mass.music.classical

        composers = await classical.composers(artist_id=artist.item_id)
        performers = await classical.performers(artist_id=artist.item_id)

        assert [(x.artist.name, x.work_count) for x in composers] == [("Conductor Performed", 1)]
        assert [(x.artist.name, x.main_role) for x in performers] == [
            ("Conductor Performed", ArtistRole.CONDUCTOR)
        ]


@pytest.fixture(scope="class", name="main_artist_library")
async def main_artist_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of classical tracks without performing credits."""
    mass = music_mass_class
    set_music_source_access(
        mass,
        {
            PROVIDER_INSTANCE: None,
            HIDDEN_INSTANCE: ProviderAccess(owner="owner", sharing=ProviderSharing.PRIVATE),
        },
    )
    lib = await _new_library(
        mass,
        (
            "tenor",
            "baritone",
            "hidden_tenor",
            "composer",
            "lyricist",
            "songwriter",
            "pianist",
            "label",
            "conductor",
        ),
    )
    a = lib.artists
    for key, work_composer in (
        ("Opera", "composer"),
        ("Song", "songwriter"),
        ("Symphony", "composer"),
        ("Hidden", "composer"),
    ):
        await _add_work(lib, key, work_composer)
    composer = ("composer", ArtistRole.COMPOSER)
    for key, work, movement, main_artists, track_credits, mapping in (
        (
            "tenor_1",
            "Opera",
            1,
            ["tenor"],
            [composer, ("lyricist", ArtistRole.LYRICIST)],
            _mapping(),
        ),
        ("tenor_2", "Opera", 2, ["tenor"], [composer], _mapping()),
        ("baritone_1", "Opera", 1, ["baritone"], [composer], _mapping()),
        (
            "song",
            "Song",
            1,
            ["songwriter", "pianist"],
            [("songwriter", ArtistRole.COMPOSER)],
            _mapping(),
        ),
        (
            "symphony",
            "Symphony",
            1,
            ["label"],
            [composer, ("conductor", ArtistRole.CONDUCTOR)],
            _mapping(),
        ),
        ("hidden", "Hidden", 1, ["hidden_tenor"], [composer], _mapping(HIDDEN_INSTANCE)),
    ):
        lib.tracks[key] = await mass.music.tracks.add_item_to_library(
            Track(
                item_id="0",
                provider="library",
                name=key,
                provider_mappings={mapping},
                artists=UniqueList(a[x] for x in main_artists),
                credits=[Credit(artist=a[artist], role=role) for artist, role in track_credits],
                work=ItemMapping.from_item(lib.works[work]),
                movement_number=movement,
                metadata=MediaItemMetadata(classical_tag=True),
            )
        )
    return lib


class TestMainArtistAsPerformer:
    """A classical track without performing credits counts its main artists as performers."""

    async def test_performers(self, main_artist_library: _Library) -> None:
        """Main artists are listed as performers, except the composer of the track."""
        rows = await main_artist_library.mass.music.classical.performers()

        assert [
            (x.artist.name, x.main_role, x.roles, x.work_count, x.recording_count) for x in rows
        ] == [
            ("Baritone", ArtistRole.PERFORMER, [ArtistRole.PERFORMER], 1, 1),
            ("Conductor", ArtistRole.CONDUCTOR, [ArtistRole.CONDUCTOR], 1, 1),
            ("Hidden Tenor", ArtistRole.PERFORMER, [ArtistRole.PERFORMER], 1, 1),
            ("Pianist", ArtistRole.PERFORMER, [ArtistRole.PERFORMER], 1, 1),
            ("Tenor", ArtistRole.PERFORMER, [ArtistRole.PERFORMER], 1, 1),
        ]

    async def test_performer_filters(self, main_artist_library: _Library) -> None:
        """The works and recordings of a main artist are found by their performer id."""
        lib = main_artist_library
        classical = lib.mass.music.classical
        tenor = lib.artist_id("tenor")

        works = await classical.works(performer_id=tenor)
        (recording,) = await classical.recordings(performer_id=tenor)

        assert [(x.work.name, x.recording_count) for x in works] == [("Opera", 1)]
        assert [x.item_id for x in recording.tracks] == [
            lib.tracks[x].item_id for x in ("tenor_1", "tenor_2")
        ]
        assert [(x.artist.name, x.role) for x in recording.credits] == [
            ("Tenor", ArtistRole.PERFORMER)
        ]

    async def test_recordings(self, main_artist_library: _Library) -> None:
        """Main artists tell recordings apart and are credited as performers on them."""
        lib = main_artist_library
        classical = lib.mass.music.classical

        opera = await classical.recordings(lib.work_id("Opera"))
        (song,) = await classical.recordings(lib.work_id("Song"))
        (symphony,) = await classical.recordings(lib.work_id("Symphony"))

        assert sorted(
            (
                [x.name for x in recording.tracks],
                [(x.artist.name, x.role) for x in recording.credits],
            )
            for recording in opera
        ) == [
            (["baritone_1"], [("Baritone", ArtistRole.PERFORMER)]),
            (["tenor_1", "tenor_2"], [("Tenor", ArtistRole.PERFORMER)]),
        ]
        assert [(x.artist.name, x.role) for x in song.credits] == [
            ("Pianist", ArtistRole.PERFORMER)
        ]
        assert [(x.artist.name, x.role) for x in symphony.credits] == [
            ("Conductor", ArtistRole.CONDUCTOR)
        ]

    async def test_skips_tracks_the_user_does_not_see(
        self, main_artist_library: _Library, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The main artist of a hidden track is no performer for the user."""
        monkeypatch.setattr(GET_CURRENT_USER, lambda: LISTENER)

        rows = await main_artist_library.mass.music.classical.performers()

        assert "Hidden Tenor" not in {x.artist.name for x in rows}
        assert "Tenor" in {x.artist.name for x in rows}


def test_stored_work_type_wins_over_the_title() -> None:
    """A work keeps its stored type when its title names another one."""
    work = WorkSummary(
        item_id="1", provider="library", name="Symphony No. 1", work_type=WorkType.OTHER
    )

    assert _with_title_work_type(work).work_type == WorkType.OTHER


def test_specific_role_leads_over_general_performer_role() -> None:
    """A performer's main role is a specific role even when the general one has more credits."""
    artist = ArtistSummary(item_id="1", provider="library", name="Tenor")
    roles = {1: Counter({ArtistRole.PERFORMER: 16, ArtistRole.SOLOIST: 3})}

    row = _performer_row(artist, None, roles, {1: set()}, {})

    assert row.main_role == ArtistRole.SOLOIST
    assert row.roles == [ArtistRole.SOLOIST, ArtistRole.PERFORMER]


@pytest.mark.parametrize(
    ("shared", "tracks", "expected"),
    [
        # without a conductor or ensemble the tracks of one album are one performance
        (
            {"album_id": 5},
            [
                {"performers": {10}, "year": 1977},
                {"performers": {11}, "year": 1977, "movement": 2},
                {"performers": {10, 11}, "year": 1978, "movement": 3},
            ],
            [([1, 2, 3], 1977, {10, 11})],
        ),
        # and the same performers on two albums are two performances
        (
            {"performers": {10}, "year": 1977},
            [{"album_id": 5}, {"album_id": 6, "movement": 2}],
            [([1], 1977, {10}), ([2], 1977, {10})],
        ),
        # a track on several albums joins the performance of the lowest one
        (
            {},
            [
                {"performers": {10}, "year": 1977, "album_id": 6, "also_on": (5, 1980)},
                {"performers": {11}, "year": 1980, "album_id": 5, "movement": 2},
            ],
            [([1, 2], 1977, {10, 11})],
        ),
        # tracks on a compilation group by their performers, not the album
        (
            {"year": 1977, "album_id": 5, "compilation": True},
            [
                {"performers": {10}},
                {"performers": {11}, "movement": 2},
                {"performers": {10}, "movement": 3},
            ],
            [([1, 3], 1977, {10}), ([2], 1977, {11})],
        ),
        # on one album, tracks with a conductor group by it, those without form their own
        (
            {"year": 1977, "album_id": 5},
            [
                {"performers": {10, 20}, "leaders": {10}},
                {"performers": {11, 20}, "leaders": {11}},
                {"performers": {10, 21}, "leaders": {10}, "movement": 2},
                {"performers": {20}, "movement": 3},
            ],
            [([1, 3], 1977, {10, 20, 21}), ([2], 1977, {11, 20}), ([4], 1977, {20})],
        ),
        # movements with other soloists stay one performance under the same conductor
        (
            {"leaders": {10}, "year": 1977},
            [{"performers": {10, 20}}, {"performers": {10, 21}, "movement": 2}],
            [([1, 2], 1977, {10, 20, 21})],
        ),
        # movements adding a choir under the same conductor and orchestra stay one performance
        (
            {"directors": {10, 11}, "year": 2012},
            [
                {"performers": {10, 11, 30}, "leaders": {10, 11, 30}},
                {"performers": {10, 11, 30, 31}, "leaders": {10, 11, 30, 31}, "movement": 2},
            ],
            [([1, 2], 2012, {10, 11, 30, 31})],
        ),
        # the same conductor with two orchestras in one year gives two performances
        (
            {"year": 1977},
            [
                {"performers": {10, 11}, "leaders": {10, 11}, "directors": {10, 11}},
                {"performers": {10, 12}, "leaders": {10, 12}, "directors": {10, 12}},
            ],
            [([1], 1977, {10, 11}), ([2], 1977, {10, 12})],
        ),
    ],
)
def test_performances(
    shared: dict[str, Any],
    tracks: list[dict[str, Any]],
    expected: list[tuple[list[int], int, set[int]]],
) -> None:
    """
    The tracks of a work group into performances by conductor and ensemble, else by album.

    Without a conductor or ensemble, a compilation's tracks group by their performers. The
    year of their album dates them.

    :param shared: The details all tracks share, as _work_track takes them.
    :param tracks: The details of each track, numbered from 1.
    :param expected: The track ids, year and performer ids of each performance.
    """
    work_tracks = [_work_track(track_id, **shared, **x) for track_id, x in enumerate(tracks, 1)]

    performances = _performances_of(work_tracks)

    assert (
        sorted(
            ([track_id for track_id, _ in x.tracks], x.year, set(x.performer_ids))
            for x in performances
        )
        == expected
    )


def _work_track(
    track_id: int,
    performers: set[int],
    year: int | None,
    leaders: set[int] | None = None,
    movement: int = 1,
    album_id: int | None = None,
    directors: set[int] | None = None,
    compilation: bool = False,
    also_on: tuple[int, int] | None = None,
) -> _WorkTrack:
    """
    Create a work track on the given album, else on its own.

    :param track_id: The library id of the track.
    :param performers: The ids of the track's performers.
    :param year: The year of the track's album.
    :param leaders: The ids of the conductors and ensembles of the track.
    :param movement: The movement number of the track.
    :param album_id: The id of the track's album.
    :param directors: The ids of the conductors and orchestras of the track.
    :param compilation: Whether the track's album is a compilation.
    :param also_on: The id and year of another album the track is on.
    """
    appearances = [
        _Appearance(
            album_id=album_id or track_id,
            year=year,
            disc_number=1,
            track_number=1,
            compilation=compilation,
        )
    ]
    if also_on:
        appearances.append(
            _Appearance(album_id=also_on[0], year=also_on[1], disc_number=1, track_number=1)
        )
    return _WorkTrack(
        track_id=track_id,
        movement=movement,
        mb_recording=None,
        performers=performers,
        leaders=leaders or set(),
        directors=directors or set(),
        appearances=appearances,
    )


def _performances_of(tracks: Sequence[_WorkTrack]) -> list[_Performance]:
    """Return the performances the tracks of a work group into."""
    return [_build_performance(1, key, group) for key, group in _group_performances(tracks).items()]


def _listing(library: _Library, command: str) -> Callable[..., Awaitable[list[Any]]]:
    """Return a classical listing, the other tracks being those Vivaldi composed."""
    listing: Callable[..., Awaitable[list[Any]]] = getattr(library.mass.music.classical, command)
    if command == "other_tracks":
        return partial(listing, library.artist_id("vivaldi"), as_composer=True)
    return listing


def _row_name(row: ClassicalComposer | ClassicalPerformer | ClassicalWorkEntry | Track) -> str:
    """Return the name of a row of a classical listing."""
    if isinstance(row, Track):
        return row.name
    if isinstance(row, ClassicalWorkEntry):
        return row.work.name
    return row.artist.name


def _track_names(recording: Recording) -> list[str]:
    """Return the names of the tracks of a recording."""
    return [x.name for x in recording.tracks]


def _album_ids(recording: Recording) -> list[str]:
    """Return the album ids of a recording."""
    return [x.item_id for x in recording.albums]


def _names(recording: Recording, role: ArtistRole) -> list[str]:
    """Return the names of the artists credited in a role on a recording."""
    return [x.artist.name for x in recording.credits if x.role == role]


def _ensembles(recording: Recording) -> list[str]:
    """Return the names of the orchestras, ensembles and choirs of a recording."""
    return [
        x.artist.name
        for x in recording.credits
        if x.role in (ArtistRole.ORCHESTRA, ArtistRole.ENSEMBLE, ArtistRole.CHOIR)
    ]


def _credited(track: Track, role: ArtistRole) -> ItemMapping | Artist:
    """Return the first artist credited in a role on a track."""
    return next(x.artist for x in track.credits if x.role == role)


def _image(image_type: ImageType, path: str, provider: str = PROVIDER_INSTANCE) -> MediaItemImage:
    """Create an image of a music source."""
    return MediaItemImage(type=image_type, path=path, provider=provider)


def _mapping(instance: str = PROVIDER_INSTANCE, in_library: bool = True) -> ProviderMapping:
    """Create a provider mapping with a unique provider item id."""
    return ProviderMapping(
        item_id=uuid4().hex,
        provider_domain=instance,
        provider_instance=instance,
        in_library=in_library,
    )


async def _add_mapped_track(
    mass: MusicAssistant,
    name: str,
    mapping: ProviderMapping,
    composer: Artist | ItemMapping | None = None,
    work: bool = True,
) -> Track:
    """
    Store a classical track with a single provider mapping and its own conductor.

    :param mass: The MusicAssistant instance.
    :param name: The name the track, its new artists and its work are named after.
    :param mapping: The only provider mapping of the track.
    :param composer: The composer of the track, a new one when not given.
    :param work: Whether to link the track to a new work.
    """
    conductor = await _add_named_artist(mass, f"Conductor {name}")
    composer = composer or await _add_named_artist(mass, f"Composer {name}")
    library_work = (
        await mass.music.works.add_item_to_library(
            Work(
                item_id=uuid4().hex,
                provider=WORK_SOURCE,
                name=f"Work {name}",
                provider_mappings=set(),
                composers=UniqueList([composer]),
            )
        )
        if work
        else None
    )
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=name,
            provider_mappings={mapping},
            artists=UniqueList([conductor]),
            credits=[
                Credit(artist=composer, role=ArtistRole.COMPOSER),
                Credit(artist=conductor, role=ArtistRole.CONDUCTOR),
            ],
            work=ItemMapping.from_item(library_work) if library_work else None,
            movement_number=1,
            metadata=MediaItemMetadata(classical_tag=True),
        )
    )


async def _new_library(mass: MusicAssistant, artists: Sequence[str]) -> _Library:
    """Return an empty library holding new artists named after their keys."""
    lib = _Library(mass=mass)
    for key in artists:
        lib.artists[key] = await _add_named_artist(mass, key.replace("_", " ").title())
    return lib


async def _add_work(
    lib: _Library,
    key: str,
    composer: str | None,
    name: str | None = None,
    parent: str | None = None,
    work_type: WorkType | None = None,
    mbid: str | None = None,
) -> None:
    """
    Store a library work under its key.

    :param lib: The library to store the work in.
    :param key: The key of the work, also its name when no name is given.
    :param composer: The key of the work's composer, None for a work without composers.
    :param name: The name of the work.
    :param parent: The key of the work this work is a part of.
    :param work_type: The stored type of the work.
    :param mbid: The MusicBrainz work id of the work.
    """
    lib.works[key] = await lib.mass.music.works.add_item_to_library(
        Work(
            item_id=uuid4().hex,
            provider=WORK_SOURCE,
            name=name or key,
            provider_mappings=set(),
            composers=UniqueList([lib.artists[composer]] if composer else []),
            parent_work=ItemMapping.from_item(lib.works[parent]) if parent else None,
            work_type=work_type,
            external_ids={(ExternalID.MB_WORK, mbid)} if mbid else set(),
        )
    )


async def _add_track(
    lib: _Library,
    key: str,
    work: str,
    performer: str,
    name: str | None = None,
    movement: int | None = None,
    movement_name: str | None = None,
    role: ArtistRole | None = ArtistRole.CONDUCTOR,
) -> None:
    """
    Store a classical library track of a work under its key, its performer as main artist.

    :param lib: The library to store the track in.
    :param key: The key of the track, also its name when no name is given.
    :param work: The key of the track's work.
    :param performer: The key of the track's main artist.
    :param name: The name of the track.
    :param movement: The movement number of the track.
    :param movement_name: The movement name of the track.
    :param role: The role the performer is credited in, None for no credit.
    """
    artist = lib.artists[performer]
    lib.tracks[key] = await lib.mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=name or key,
            provider_mappings={_mapping()},
            artists=UniqueList([artist]),
            credits=[Credit(artist=artist, role=role)] if role else [],
            work=ItemMapping.from_item(lib.works[work]),
            movement_number=movement,
            movement_name=movement_name,
            metadata=MediaItemMetadata(classical_tag=True),
        )
    )


async def _add_credited_track(
    lib: _Library, key: str, work: str, performers: Sequence[tuple[str, ArtistRole, int]]
) -> None:
    """
    Store a classical library track of a work under its key, with performers at their positions.

    :param lib: The library to store the track in.
    :param key: The key of the track, also its name.
    :param work: The key of the track's work.
    :param performers: The key, role and credit position of each performer, the first of them
        also the track's main artist.
    """
    lib.tracks[key] = await lib.mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=key,
            provider_mappings={_mapping()},
            artists=UniqueList([lib.artists[performers[0][0]]]),
            credits=[
                Credit(artist=lib.artists[artist], role=role, position=position)
                for artist, role, position in performers
            ],
            work=ItemMapping.from_item(lib.works[work]),
            metadata=MediaItemMetadata(classical_tag=True),
        )
    )


async def _co_performer_library(mass: MusicAssistant) -> _Library:
    """Return a library of a requiem with two recordings of a singer and one without them."""
    lib = await _new_library(
        mass,
        (
            "brahms",
            "singer",
            "tenor",
            "abbado",
            "karajan",
            "solti",
            "bpo",
            "vpo",
            "lso",
            "consort",
            "chorus",
            "pianist",
        ),
    )
    await _add_work(lib, "Requiem", "brahms")
    # stored first, though the recording by Abbado comes first on its conductor
    await _add_credited_track(
        lib,
        "karajan_requiem",
        "Requiem",
        [
            ("singer", ArtistRole.SOLOIST, 0),
            ("karajan", ArtistRole.CONDUCTOR, 0),
            ("vpo", ArtistRole.ORCHESTRA, 0),
            ("chorus", ArtistRole.CHOIR, 0),
            ("tenor", ArtistRole.SOLOIST, 1),
            ("pianist", ArtistRole.PERFORMER, 0),
        ],
    )
    await _add_credited_track(
        lib,
        "abbado_requiem",
        "Requiem",
        [
            ("singer", ArtistRole.SOLOIST, 1),
            ("abbado", ArtistRole.CONDUCTOR, 0),
            ("bpo", ArtistRole.ORCHESTRA, 0),
            ("consort", ArtistRole.ENSEMBLE, 1),
            ("tenor", ArtistRole.SOLOIST, 0),
        ],
    )
    await _add_credited_track(
        lib,
        "solti_requiem",
        "Requiem",
        [("solti", ArtistRole.CONDUCTOR, 0), ("lso", ArtistRole.ORCHESTRA, 0)],
    )
    return lib


async def _add_named_artist(mass: MusicAssistant, name: str) -> Artist:
    """Store a library artist with the given name."""
    return await mass.music.artists.add_item_to_library(
        Artist(item_id="0", provider="library", name=name, provider_mappings={_mapping()})
    )


async def _seed_library(mass: MusicAssistant) -> _Library:
    """Seed a small classical library next to a pop album."""
    lib = _Library(mass=mass)
    for key, name, sort_name in (
        ("beethoven", "Ludwig van Beethoven", "Beethoven, Ludwig van"),
        ("vivaldi", "Antonio Vivaldi", "Vivaldi, Antonio"),
        # a lower case name proves the conductor order ignores case
        ("abbado", "claudio abbado", None),
        ("bernstein", "Leonard Bernstein", None),
        ("karajan", "Herbert von Karajan", None),
        ("barenboim", "Daniel Barenboim", None),
        ("bpo", "Berliner Philharmoniker", None),
        ("vpo", "Wiener Philharmoniker", None),
        ("gardiner", "Gardiner Singers", None),
        ("label", "Classical Label", None),
        ("pop_star", "Pop Star", None),
        ("pop_writer", "Pop Writer", None),
    ):
        lib.artists[key] = await mass.music.artists.add_item_to_library(
            Artist(
                item_id="0",
                provider="library",
                name=name,
                sort_name=sort_name,
                provider_mappings={_mapping()},
                metadata=MediaItemMetadata(
                    images=UniqueList(_image(*x) for x in ARTIST_IMAGES.get(key, ())) or None,
                    country="DE" if key == "beethoven" else None,
                ),
                period=Period.CLASSICAL if key == "beethoven" else None,
            )
        )
    a = lib.artists
    for key, name, catalog, year in (
        ("Eroica", "Symphony No. 3 Eroica", "Op. 55", 1804),
        ("Fifth", "Symphony No. 5", "Op. 67", 1808),
        ("Emperor", "Piano Concerto No. 5", "Op. 73", 1809),
        ("Spring", "Violin Concerto Spring", "RV 269", 1720),
        ("Gloria", "Gloria", "RV 589", None),
        ("Pop", "Pop Song", None, None),
    ):
        lib.works[key] = await mass.music.works.add_item_to_library(
            Work(
                item_id=uuid4().hex,
                provider=WORK_SOURCE,
                name=name,
                provider_mappings=set(),
                composers=UniqueList([a[WORK_COMPOSERS[key]]]),
                catalog_numbers=[catalog] if catalog else [],
                composition_year=year,
            )
        )
    genre_rows = await mass.music.database.get_rows_from_query(
        f"SELECT item_id FROM {DB_TABLE_GENRES} "
        "WHERE translation_key = 'classical' AND content_type IS NULL",
        limit=1,
    )
    # the order sets the album ids, the compilation and the extra album come first
    for key, year, classical in (
        ("compilation", 2010, True),
        ("karajan", 1962, True),
        ("bernstein", 1979, True),
        ("mixed", 1979, True),
        ("emperor", 1983, True),
        ("emperor_box", 1995, True),
        ("spring_extra", 1970, True),
        ("spring", 1970, True),
        ("pop", 2000, False),
    ):
        lib.albums[key] = await mass.music.albums.add_item_to_library(
            Album(
                item_id="0",
                provider="library",
                name=f"Album {key}",
                year=year,
                album_type=AlbumType.ALBUM,
                provider_mappings={_mapping()},
                artists=UniqueList([a["label"]]),
            )
        )
        if classical:
            await mass.music.genres.add_media_mapping(
                int(genre_rows[0]["item_id"]), MediaType.ALBUM, lib.albums[key].item_id
            )
    mbid = {key: str(uuid4()) for key in ("k1", "k2", "k3", "k4", "e1", "e2")}
    numerals = ("I", "II", "III", "IV")

    async def add(  # noqa: PLR0913
        key: str,
        work: str | None,
        movement: int | None,
        performers: list[tuple[str, ArtistRole]],
        album: str | None = None,
        track_number: int = 0,
        recording_id: str | None = None,
        composer: str | None = None,
        main_artist: str = "label",
        date_added: datetime | None = None,
        name: str | None = None,
    ) -> None:
        """Store a library track with a composer credit and the given performers."""
        composer_key = composer or WORK_COMPOSERS[str(work)]
        track_credits = [Credit(artist=a[composer_key], role=ArtistRole.COMPOSER)]
        track_credits += [
            Credit(
                artist=a[artist],
                role=role,
                instrument="piano" if role == ArtistRole.SOLOIST else None,
            )
            for artist, role in performers
        ]
        lib.tracks[key] = await mass.music.tracks.add_item_to_library(
            Track(
                item_id="0",
                provider="library",
                name=name or key,
                duration=TRACK_DURATION,
                provider_mappings={_mapping()},
                artists=UniqueList([a[main_artist]]),
                credits=track_credits,
                album=lib.albums[album] if album else None,
                disc_number=1 if album else 0,
                track_number=track_number,
                work=ItemMapping.from_item(lib.works[work]) if work else None,
                movement_number=movement,
                external_ids={(ExternalID.MB_RECORDING, recording_id)} if recording_id else set(),
                # tracks without an album are classical by their own tag
                metadata=MediaItemMetadata(classical_tag=album is None),
                date_added=date_added,
            )
        )

    karajan_bpo = [("karajan", ArtistRole.CONDUCTOR), ("bpo", ArtistRole.ORCHESTRA)]
    bernstein_vpo = [("bernstein", ArtistRole.CONDUCTOR), ("vpo", ArtistRole.ORCHESTRA)]
    # the reissue is stored before the original
    await add("comp_k2", "Fifth", 2, karajan_bpo, "compilation", 1, mbid["k2"], name="Karajan II")
    for number in range(1, 5):
        numeral = numerals[number - 1]
        await add(
            f"k{number}",
            "Fifth",
            number,
            karajan_bpo,
            "karajan",
            number,
            mbid[f"k{number}"],
            name=f"Karajan {numeral}",
        )
        await add(
            f"b{number}",
            "Fifth",
            number,
            bernstein_vpo,
            "bernstein",
            number,
            name=f"Bernstein {numeral}",
        )
    await add("mixed_b4", "Fifth", 4, bernstein_vpo, "mixed", 3, name="Bernstein IV")
    await add("bernstein_eroica", "Eroica", 1, bernstein_vpo, "bernstein", 5)
    await add(
        "abbado_eroica",
        "Eroica",
        1,
        [("abbado", ArtistRole.CONDUCTOR), ("vpo", ArtistRole.ORCHESTRA)],
        "mixed",
        1,
    )
    await add(
        "bernstein_bpo_eroica",
        "Eroica",
        1,
        [("bernstein", ArtistRole.CONDUCTOR), ("bpo", ArtistRole.ORCHESTRA)],
        "mixed",
        2,
    )
    await add("karajan_eroica", "Eroica", 1, karajan_bpo, None)
    barenboim = [
        ("barenboim", ArtistRole.CONDUCTOR),
        ("barenboim", ArtistRole.SOLOIST),
        ("bpo", ArtistRole.ORCHESTRA),
    ]
    for album in ("emperor_box", "emperor"):
        for number in (1, 2):
            key = f"emperor_{number}" if album == "emperor" else f"emperor_box_{number}"
            await add(key, "Emperor", number, barenboim, album, number, mbid[f"e{number}"])
    karajan_vpo = [("karajan", ArtistRole.CONDUCTOR), ("vpo", ArtistRole.ORCHESTRA)]
    await add("spring_1", "Spring", 1, karajan_vpo, "spring", 1)
    await add("spring_2", "Spring", 2, karajan_vpo, "spring", 2)
    await mass.music.database.insert(
        DB_TABLE_ALBUM_TRACKS,
        {
            "track_id": int(lib.tracks["spring_1"].item_id),
            "album_id": int(lib.albums["spring_extra"].item_id),
            "disc_number": 1,
            "track_number": 7,
        },
    )
    await add("gloria", "Gloria", 1, [("gardiner", ArtistRole.CHOIR)], None)
    gardiner = [("gardiner", ArtistRole.PERFORMER)]
    workless: list[tuple[str, str | None, int]] = [
        ("Aria", None, 2021),
        ("Barcarolle", "spring_extra", 2020),
        ("Cantilena", "karajan", 2022),
    ]
    for key, workless_album, year in workless:
        await add(
            key,
            None,
            None,
            gardiner,
            workless_album,
            8,
            composer="vivaldi",
            date_added=datetime(year, 1, 1, tzinfo=UTC),
        )
    await add(
        "pop_song",
        "Pop",
        None,
        [("karajan", ArtistRole.CONDUCTOR)],
        "pop",
        1,
        main_artist="pop_star",
    )
    await add("Pop B-Side", None, None, [], "pop", 2, composer="vivaldi", main_artist="pop_star")
    return lib
