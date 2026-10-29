"""Tests for the classical browse API of the music controller."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock
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
    ClassicalPerformer,
    ClassicalWorkEntry,
    Credit,
    ItemMapping,
    MediaItemImage,
    MediaItemMetadata,
    ProviderMapping,
    Recording,
    Track,
    UniqueList,
    Work,
    WorkMovement,
    WorkSummary,
)

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_GENRES,
    DB_TABLE_TRACKS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.classical import (
    _Appearance,
    _build_performance,
    _group_performances,
    _movement_order,
    _Movements,
    _Performance,
    _performer_row,
    _piece_name,
    _refined_roles,
    _shared_score_number,
    _shown_credits,
    _title_work_type,
    _track_movements,
    _with_title_work_type,
    _WorkTrack,
)
from music_assistant.controllers.player_queues.media_resolver import MediaResolver
from music_assistant.helpers.json import json_dumps
from music_assistant.mass import MusicAssistant
from tests.common import set_music_source_access

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from music_assistant.controllers.player_queues import PlayerQueuesController

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
        "performers",
        "works",
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
        # the performer credits take the one specific role held elsewhere
        ("Gardiner Singers", ArtistRole.CHOIR, [ArtistRole.CHOIR], 1, 1),
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
        (ArtistRole.PERFORMER, []),
        (ArtistRole.ENSEMBLE, []),
        (ArtistRole.COMPOSER, []),
    ],
)
async def test_performers_role_filter(
    library: _Library, role: ArtistRole, expected: list[str]
) -> None:
    """The role filter keeps the performers holding that role among their roles."""
    rows = await library.mass.music.classical.performers(role=role)

    assert [x.artist.name for x in rows] == expected


@pytest.mark.parametrize(
    ("sort_field", "sort_direction", "expected"),
    [
        (
            SortField.NAME,
            SortDirection.DESC,
            [
                "Wiener Philharmoniker",
                "Leonard Bernstein",
                "Herbert von Karajan",
                "Gardiner Singers",
                "Daniel Barenboim",
                "claudio abbado",
                "Berliner Philharmoniker",
            ],
        ),
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
        ("vivaldi", True, False),
        ("karajan", False, True),
        ("barenboim", False, True),
        ("gardiner", False, True),
        ("label", False, False),
        ("pop_star", False, False),
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

    soloist = await classical.performers(role=ArtistRole.SOLOIST, artist_id=barenboim)
    assert [(x.artist.name, x.roles) for x in soloist] == [
        ("Daniel Barenboim", [ArtistRole.CONDUCTOR, ArtistRole.SOLOIST])
    ]
    assert await classical.performers(role=ArtistRole.ORCHESTRA, artist_id=barenboim) == []
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
    assert all(x.part_names == [] for x in rows)
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


async def test_works_by_composer(library: _Library) -> None:
    """The composer filter keeps the works of that composer."""
    rows = await library.mass.music.classical.works(composer_id=library.artist_id("vivaldi"))

    assert [x.work.item_id for x in rows] == [library.work_id(x) for x in ("Spring", "Gloria")]


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
        ("karajan", SortField.COMPOSER, None, [("Eroica", 1), ("Fifth", 1), ("Spring", 1)]),
        ("bpo", SortField.COMPOSER, None, [("Eroica", 2), ("Fifth", 1), ("Emperor", 1)]),
        ("bernstein", SortField.RECORDING_COUNT, None, [("Eroica", 2), ("Fifth", 1)]),
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


@pytest.mark.parametrize(
    ("year_from", "year_to", "expected"),
    [
        (1805, None, ["Fifth", "Emperor"]),
        (None, 1805, ["Eroica", "Spring"]),
        (1800, 1808, ["Eroica", "Fifth"]),
    ],
)
async def test_works_year_range(
    library: _Library, year_from: int | None, year_to: int | None, expected: list[str]
) -> None:
    """The year range filters on the year composed and drops works without one."""
    rows = await library.mass.music.classical.works(year_from=year_from, year_to=year_to)

    assert [x.work.item_id for x in rows] == [library.work_id(x) for x in expected]


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
    ("command", "sort_field"),
    [
        ("composers", SortField.RECORDING_COUNT),
        ("performers", SortField.SORT_NAME),
        ("works", SortField.TIMESTAMP_ADDED),
    ],
)
async def test_sort_fields_a_listing_does_not_offer_are_rejected(
    library: _Library, command: str, sort_field: SortField
) -> None:
    """A sort field the classical listing does not offer is rejected."""
    with pytest.raises(InvalidDataError):
        await getattr(library.mass.music.classical, command)(sort_field=sort_field)


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
        ("composers", {}),
        ("composers", {"sort_field": SortField.WORK_COUNT, "sort_direction": SortDirection.ASC}),
        ("performers", {}),
        ("performers", {"sort_field": SortField.RECORDING_COUNT}),
        ("performers", {"role": ArtistRole.CONDUCTOR, "sort_field": SortField.RECORDING_COUNT}),
        ("works", {}),
        ("works", {"sort_field": SortField.RECORDING_COUNT}),
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
    """The classical listings and probe only count the library tracks the calling user sees."""

    async def test_listings_skip_tracks_the_user_does_not_see(
        self, restricted_mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Hidden and not in library tracks add no composer, performer, work or recording.

        Nor do they make the probe find classical content or give their artists a row.
        """
        mass = restricted_mass
        classical = mass.music.classical
        hidden = await _add_mapped_track(mass, "Hidden", _mapping(HIDDEN_INSTANCE))
        assert hidden.work is not None
        await _add_mapped_track(mass, "Not In Library", _mapping(in_library=False))
        hidden_composer, hidden_conductor = (
            _credited(hidden, role) for role in (ArtistRole.COMPOSER, ArtistRole.CONDUCTOR)
        )
        probe_for_everyone = await mass.music.has_classical_content()
        assert len(await classical.composers(artist_id=hidden_composer.item_id)) == 1
        assert len(await classical.performers(artist_id=hidden_conductor.item_id)) == 1
        with monkeypatch.context() as listener:
            listener.setattr(GET_CURRENT_USER, lambda: LISTENER)
            probe_for_listener = await mass.music.has_classical_content()
        visible = await _add_mapped_track(mass, "Visible", _mapping())
        composer = _credited(visible, ArtistRole.COMPOSER)
        for name, mapping in (
            ("Visible Other", _mapping()),
            ("Hidden Other", _mapping(HIDDEN_INSTANCE)),
            ("Not In Library Other", _mapping(in_library=False)),
        ):
            await _add_mapped_track(mass, name, mapping, composer=composer, work=False)
        monkeypatch.setattr(GET_CURRENT_USER, lambda: LISTENER)

        composers = await classical.composers()
        performers = await classical.performers()
        works = await classical.works()
        hidden_recordings = await classical.recordings(hidden.work.item_id)
        other_tracks = await classical.other_tracks(composer.item_id, as_composer=True)

        assert probe_for_everyone
        assert not probe_for_listener
        assert [x.artist.name for x in composers] == ["Composer Visible"]
        assert [x.artist.name for x in performers] == ["Conductor Visible"]
        assert [x.work.name for x in works] == ["Work Visible"]
        assert hidden_recordings == []
        assert [x.name for x in other_tracks] == ["Visible Other"]
        assert await classical.composers(artist_id=hidden_composer.item_id) == []
        assert await classical.performers(artist_id=hidden_conductor.item_id) == []


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


class TestArtistsOfWorks:
    """The composers and performers listings only hold artists of a work."""

    async def test_artists_without_a_work_are_left_out(
        self, restricted_mass: MusicAssistant
    ) -> None:
        """An artist credited only on tracks without a work gets no row in any filter."""
        mass = restricted_mass
        await _add_mapped_track(mass, "Work", _mapping())
        workless = await _add_mapped_track(mass, "Workless", _mapping(), work=False)
        composer, conductor = (
            _credited(workless, role) for role in (ArtistRole.COMPOSER, ArtistRole.CONDUCTOR)
        )
        classical = mass.music.classical

        composers = await classical.composers()
        performers = await classical.performers()

        assert [(x.artist.name, x.work_count) for x in composers] == [("Composer Work", 1)]
        assert await classical.composers(sort_field=SortField.WORK_COUNT) == composers
        assert await classical.composers(search="workless") == []
        assert await classical.composers(artist_id=composer.item_id) == []
        assert [(x.artist.name, x.recording_count) for x in performers] == [("Conductor Work", 1)]
        assert await classical.performers(role=ArtistRole.CONDUCTOR) == performers
        assert await classical.performers(sort_field=SortField.RECORDING_COUNT) == performers
        assert await classical.performers(search="workless") == []
        assert await classical.performers(artist_id=conductor.item_id) == []


class TestRefinedPerformerRoles:
    """Every classical result shows and filters on the refined performer roles."""

    async def test_roles_from_the_name(self, restricted_mass: MusicAssistant) -> None:
        """A generic or wrong ensemble role shows as the role the name tells, everywhere."""
        lib = _Library(mass=restricted_mass)
        for name in ("Composer", "BBC Symphony Orchestra", "Quatuor SUK"):
            lib.artists[name] = await _add_named_artist(lib.mass, name)
        composer, orchestra, quartet = lib.artists.values()
        await _add_work(lib, "Quartet", "Composer")
        movement = await lib.mass.music.tracks.add_item_to_library(
            Track(
                item_id="0",
                provider="library",
                name="Movement",
                provider_mappings={_mapping()},
                artists=UniqueList([orchestra]),
                credits=[
                    Credit(artist=composer, role=ArtistRole.COMPOSER),
                    Credit(artist=orchestra, role=ArtistRole.PERFORMER),
                    Credit(artist=quartet, role=ArtistRole.ORCHESTRA),
                ],
                work=ItemMapping.from_item(lib.works["Quartet"]),
                movement_number=1,
                metadata=MediaItemMetadata(classical_tag=True),
            )
        )
        classical = lib.mass.music.classical

        rows = await classical.performers()
        (recording,) = await classical.recordings(lib.work_id("Quartet"))
        by_performer = await classical.recordings(performer_id=orchestra.item_id)

        assert [(x.artist.name, x.main_role, x.roles) for x in rows] == [
            ("BBC Symphony Orchestra", ArtistRole.ORCHESTRA, [ArtistRole.ORCHESTRA]),
            ("Quatuor SUK", ArtistRole.ENSEMBLE, [ArtistRole.ENSEMBLE]),
        ]
        orchestras = await classical.performers(role=ArtistRole.ORCHESTRA)
        assert [x.artist.name for x in orchestras] == ["BBC Symphony Orchestra"]
        ensembles = await classical.performers(role=ArtistRole.ENSEMBLE)
        assert [x.artist.name for x in ensembles] == ["Quatuor SUK"]
        assert await classical.performers(role=ArtistRole.PERFORMER) == []
        quartet_credits = [
            ("BBC Symphony Orchestra", ArtistRole.ORCHESTRA),
            ("Quatuor SUK", ArtistRole.ENSEMBLE),
        ]
        assert [(x.artist.name, x.role) for x in recording.credits] == quartet_credits
        assert {
            (x.artist.name, x.role)
            for x in recording.tracks[0].credits
            if x.role != ArtistRole.COMPOSER
        } == set(quartet_credits)
        assert [x.credits for x in by_performer] == [recording.credits]
        stored = await lib.mass.music.tracks.get_library_item(movement.item_id)
        assert ("BBC Symphony Orchestra", ArtistRole.PERFORMER) in {
            (x.artist.name, x.role) for x in stored.credits
        }


async def test_other_tracks_show_refined_roles(library: _Library) -> None:
    """The performer role on other tracks shows the one specific role held elsewhere."""
    tracks = await library.mass.music.classical.other_tracks(library.artist_id("gardiner"))

    assert {
        x.role for track in tracks for x in track.credits if x.artist.name == "Gardiner Singers"
    } == {ArtistRole.CHOIR}


@pytest.mark.parametrize(
    ("name", "stored", "expected"),
    [
        (
            "Mariss Jansons",
            {ArtistRole.PERFORMER: 2, ArtistRole.CONDUCTOR: 5},
            {ArtistRole.CONDUCTOR: 7},
        ),
        ("BBC Symphony Orchestra", {ArtistRole.PERFORMER: 3}, {ArtistRole.ORCHESTRA: 3}),
        (
            "Chœurs de la Maîtrise Notre-Dame de Paris",
            {ArtistRole.ORCHESTRA: 1},
            {ArtistRole.CHOIR: 1},
        ),
        (
            "Moscow Radio Choir",
            {ArtistRole.ORCHESTRA: 1, ArtistRole.PERFORMER: 1},
            {ArtistRole.CHOIR: 2},
        ),
        ("Quatuor SUK", {ArtistRole.ORCHESTRA: 1}, {ArtistRole.ENSEMBLE: 1}),
        ("Wiener Philharmoniker", {ArtistRole.ORCHESTRA: 1}, {ArtistRole.ORCHESTRA: 1}),
        ("RIAS Kammerchor", {ArtistRole.ENSEMBLE: 1}, {ArtistRole.CHOIR: 1}),
        ("Staats- und Domchor Berlin", {ArtistRole.ENSEMBLE: 1}, {ArtistRole.CHOIR: 1}),
        (
            "Soloists of the Royal Opera House Orchestra, Covent Garden",
            {ArtistRole.PERFORMER: 1},
            {ArtistRole.PERFORMER: 1},
        ),
        ("The Tallis Scholars", {ArtistRole.PERFORMER: 1}, {ArtistRole.PERFORMER: 1}),
        # a name with choir and orchestra words tells neither
        (
            "Chorus and Orchestra of the Opera",
            {ArtistRole.PERFORMER: 1},
            {ArtistRole.PERFORMER: 1},
        ),
        ("Choir and Orchestra", {ArtistRole.ORCHESTRA: 1}, {ArtistRole.ORCHESTRA: 1}),
        # several specific roles held elsewhere leave the name to tell the role
        (
            "Daniel Barenboim",
            {ArtistRole.PERFORMER: 1, ArtistRole.CONDUCTOR: 1, ArtistRole.SOLOIST: 1},
            {ArtistRole.PERFORMER: 1, ArtistRole.CONDUCTOR: 1, ArtistRole.SOLOIST: 1},
        ),
        (
            "Academy Sinfonia",
            {ArtistRole.PERFORMER: 1, ArtistRole.CONDUCTOR: 1, ArtistRole.SOLOIST: 1},
            {ArtistRole.ORCHESTRA: 1, ArtistRole.CONDUCTOR: 1, ArtistRole.SOLOIST: 1},
        ),
        # a word ending in orchester counts as orchestra, no other word ending does
        ("Gewandhausorchester Leipzig", {ArtistRole.PERFORMER: 1}, {ArtistRole.ORCHESTRA: 1}),
        ("Rundfunkchor Berlin", {ArtistRole.PERFORMER: 1}, {ArtistRole.CHOIR: 1}),
        # words only count whole
        ("Triolet Lane", {ArtistRole.PERFORMER: 1}, {ArtistRole.PERFORMER: 1}),
    ],
)
def test_refined_roles(
    name: str, stored: dict[ArtistRole, int], expected: dict[ArtistRole, int]
) -> None:
    """The generic performer role and a wrong orchestra role are refined, never guessed."""
    assert _refined_roles(name, Counter(stored)) == Counter(expected)


@pytest.mark.parametrize(
    ("roles", "refinements", "expected"),
    [
        # the main artist of one track counts as performer, refined to their soloist role
        (
            [(ArtistRole.PERFORMER, None), (ArtistRole.SOLOIST, "tenor")],
            {ArtistRole.PERFORMER: ArtistRole.SOLOIST},
            [(ArtistRole.SOLOIST, "tenor")],
        ),
        (
            [
                (ArtistRole.SOLOIST, None),
                (ArtistRole.PERFORMER, None),
                (ArtistRole.SOLOIST, "tenor"),
            ],
            {},
            [(ArtistRole.SOLOIST, "tenor")],
        ),
        # a chorus master who conducts elsewhere is not shown as conductor for that task
        (
            [(ArtistRole.PERFORMER, "chorus master"), (ArtistRole.PERFORMER, None)],
            {ArtistRole.PERFORMER: ArtistRole.CONDUCTOR},
            [(ArtistRole.PERFORMER, "chorus master"), (ArtistRole.CONDUCTOR, None)],
        ),
    ],
)
def test_shown_credits(
    roles: list[tuple[ArtistRole, str | None]],
    refinements: dict[ArtistRole, ArtistRole],
    expected: list[tuple[ArtistRole, str | None]],
) -> None:
    """
    A performer credited generally and as soloist with an instrument shows once, as the soloist.

    A credit naming a task, such as chorus master, keeps its role.
    """
    artist = ItemMapping(media_type=MediaType.ARTIST, item_id="7", provider="library", name="A")
    track_credits = [Credit(artist=artist, role=role, instrument=x) for role, x in roles]

    shown = _shown_credits(track_credits, {7: refinements})

    assert [(x.role, x.instrument) for x in shown] == expected


@pytest.fixture(scope="class", name="parts_library")
async def parts_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of two operas whose arias are parts of them."""
    lib = await _new_library(music_mass_class, ("delibes", "bizet", "sutherland", "callas"))
    await _add_work(lib, "Lakme", "delibes")
    await _add_work(lib, "Flower Duet", "delibes", parent="Lakme")
    await _add_work(lib, "Bell Song", "delibes", parent="Lakme")
    # an opera without composers is filtered by those of its arias
    await _add_work(lib, "Carmen", None)
    await _add_work(lib, "Habanera", "bizet", parent="Carmen")
    for key, work, soloist in (
        ("Overture", "Lakme", "callas"),
        ("Duet Sutherland", "Flower Duet", "sutherland"),
        ("Duet Callas", "Flower Duet", "callas"),
        ("Bell Sutherland", "Bell Song", "sutherland"),
        ("Habanera Callas", "Habanera", "callas"),
    ):
        await _add_track(lib, key, work, soloist, role=ArtistRole.SOLOIST)
    return lib


class TestPartsOfWorks:
    """The parts of a work, such as arias, are listed through the work they belong to."""

    async def test_parts_count_in_their_work_row(self, parts_library: _Library) -> None:
        """Only the operas are rows, each counting the recordings of its arias."""
        rows = await parts_library.mass.music.classical.works()

        assert {(x.work.name, x.recording_count) for x in rows} == {("Lakme", 4), ("Carmen", 1)}

    @pytest.mark.parametrize(("composer", "expected"), [("delibes", "Lakme"), ("bizet", "Carmen")])
    async def test_composer_filter_falls_back_to_the_parts(
        self, parts_library: _Library, composer: str, expected: str
    ) -> None:
        """A work matches its own composers, or those of its parts when it has none."""
        rows = await parts_library.mass.music.classical.works(
            composer_id=parts_library.artist_id(composer)
        )

        assert [x.work.name for x in rows] == [expected]

    async def test_performer_filter_matches_through_the_parts(
        self, parts_library: _Library
    ) -> None:
        """A work matches a performer of one of its parts and counts only their recordings."""
        rows = await parts_library.mass.music.classical.works(
            performer_id=parts_library.artist_id("sutherland")
        )

        assert [(x.work.name, x.recording_count) for x in rows] == [("Lakme", 2)]

    @pytest.mark.parametrize(("search", "expected"), [("habanera", "Carmen"), ("lakme", "Lakme")])
    async def test_search_finds_the_work_of_a_part(
        self, parts_library: _Library, search: str, expected: str
    ) -> None:
        """A search matching a part finds the work it belongs to."""
        rows = await parts_library.mass.music.classical.works(search=search)

        assert [x.work.name for x in rows] == [expected]

    async def test_recordings_include_those_of_the_parts(self, parts_library: _Library) -> None:
        """The recordings of a work are its own, then its parts', each with its own work."""
        recordings = await parts_library.mass.music.classical.recordings(
            parts_library.work_id("Lakme")
        )

        assert [x.work.name for x in recordings] == [
            "Lakme",
            "Bell Song",
            "Flower Duet",
            "Flower Duet",
        ]

    @pytest.mark.parametrize(
        ("composer", "performer", "expected"),
        [
            (None, None, {"Lakme": ["Bell Song", "Flower Duet"], "Carmen": ["Habanera"]}),
            ("delibes", None, {"Lakme": ["Bell Song", "Flower Duet"]}),
            # all parts are named, also those the performer is not on
            (None, "callas", {"Lakme": ["Bell Song", "Flower Duet"], "Carmen": ["Habanera"]}),
        ],
    )
    async def test_rows_carry_their_part_names(
        self,
        parts_library: _Library,
        composer: str | None,
        performer: str | None,
        expected: dict[str, list[str]],
    ) -> None:
        """A work row names all its parts in the library, ordered by name."""
        rows = await parts_library.mass.music.classical.works(
            composer_id=parts_library.artist_id(composer) if composer else None,
            performer_id=parts_library.artist_id(performer) if performer else None,
        )

        assert {x.work.name: x.part_names for x in rows} == expected

    async def test_recordings_carry_the_parent_work(self, parts_library: _Library) -> None:
        """The recordings of a part carry the work it is a part of, the work's own carry none."""
        recordings = await parts_library.mass.music.classical.recordings(
            parts_library.work_id("Lakme")
        )

        assert [
            (x.work.name, x.parent_work.name if x.parent_work else None) for x in recordings
        ] == [
            ("Lakme", None),
            ("Bell Song", "Lakme"),
            ("Flower Duet", "Lakme"),
            ("Flower Duet", "Lakme"),
        ]
        [habanera] = await parts_library.mass.music.classical.recordings(
            parts_library.work_id("Habanera")
        )
        assert habanera.parent_work is not None
        assert habanera.parent_work.item_id == parts_library.work_id("Carmen")

    async def test_works_sharing_a_name_carry_their_version(self, parts_library: _Library) -> None:
        """Works sharing a name carry their versions, a work of its own name carries none."""
        mass = parts_library.mass
        versions = {"Carmen": "1875 version", "Habanera": "with dialogue"}
        for key, version in versions.items():
            await _set_version(parts_library, key, version)
        namesake = await mass.music.works.add_item_to_library(
            Work(
                item_id=uuid4().hex,
                provider=WORK_SOURCE,
                name="Carmen",
                version="1883 version",
                provider_mappings=set(),
            )
        )
        try:
            rows = await mass.music.classical.works(composer_id=parts_library.artist_id("bizet"))
            carmen = await mass.music.classical.work(parts_library.work_id("Carmen"))
            [habanera] = await mass.music.classical.recordings(parts_library.work_id("Habanera"))
            listed = await mass.music.works.get_library_items_by_query(
                limit=0,
                extra_query_parts=["works.item_id IN :item_ids"],
                extra_query_params={"item_ids": [int(parts_library.work_id(x)) for x in versions]},
                summary=True,
            )
        finally:
            await mass.music.works.remove_item_from_library(namesake.item_id)
            for key in versions:
                await _set_version(parts_library, key, "")

        assert [(x.work.name, x.work.version) for x in rows] == [("Carmen", "1875 version")]
        assert carmen.version == "1875 version"
        assert [x.version for x in carmen.parts] == [""]
        assert habanera.work.version == ""
        assert habanera.parent_work is not None
        assert habanera.parent_work.version == "1875 version"
        assert [x.work.version for x in habanera.tracks if x.work] == [""]
        # the library listings follow the same rule
        assert sorted((x.name, x.version) for x in listed) == [
            ("Carmen", "1875 version"),
            ("Habanera", ""),
        ]

    async def test_parts_are_the_movements_of_a_work_without_any(
        self, parts_library: _Library
    ) -> None:
        """A work without movements lists its parts, and their recordings cover them."""
        lakme = parts_library.work_id("Lakme")
        work = await parts_library.mass.music.classical.work(lakme)
        recordings = await parts_library.mass.music.classical.recordings(lakme)

        assert [(x.number, x.name) for x in work.movements] == [
            (1, "Bell Song"),
            (2, "Flower Duet"),
        ]
        assert not work.movements_from_musicbrainz
        assert [(x.work.name, x.movement_numbers, x.complete) for x in recordings] == [
            ("Lakme", [], None),
            ("Bell Song", [1], None),
            ("Flower Duet", [2], None),
            ("Flower Duet", [2], None),
        ]
        assert [[x.movement_number for x in r.tracks] for r in recordings] == [
            [None],
            [1],
            [2],
            [2],
        ]

    async def test_composer_counts_a_work_with_parts_once(self, parts_library: _Library) -> None:
        """A composer's work count holds the opera once, its recordings those of the arias."""
        [row] = await parts_library.mass.music.classical.composers(
            artist_id=parts_library.artist_id("delibes")
        )

        assert (row.work_count, row.recording_count) == (1, 4)


@pytest.fixture(scope="class", name="play_library")
async def play_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of works with and without recordings of their own."""
    lib = await _new_library(music_mass_class, ("verdi", "abbado", "karajan", "solti"))
    await _add_work(lib, "Requiem", "verdi")
    await _add_work(lib, "Aida", "verdi")
    await _add_work(lib, "Ritorna vincitor", "verdi", parent="Aida")
    await _add_work(lib, "Celeste Aida", "verdi", parent="Aida")
    await _add_work(lib, "Otello", "verdi")
    await _add_work(lib, "Ave Maria", "verdi", parent="Otello")
    await _add_work(lib, "Nabucco", "verdi")
    for key, work, conductor, movement in (
        ("Karajan Introit", "Requiem", "karajan", 1),
        ("Abbado Dies irae", "Requiem", "abbado", 2),
        ("Abbado Introit", "Requiem", "abbado", 1),
        ("Solti Celeste", "Celeste Aida", "solti", None),
        ("Abbado Celeste", "Celeste Aida", "abbado", None),
        ("Karajan Ritorna", "Ritorna vincitor", "karajan", None),
        ("Solti Otello", "Otello", "solti", None),
        ("Solti Ave Maria", "Ave Maria", "solti", None),
    ):
        await _add_track(lib, key, work, conductor, movement=movement)
    return lib


class TestWorkTracks:
    """Playing a work plays one recording of it, or one of each of its parts."""

    @pytest.mark.parametrize(
        ("work", "expected"),
        [
            # the first recording of the work, in movement order
            ("Requiem", ["Abbado Introit", "Abbado Dies irae"]),
            # the first recording of each part, in the work's order
            ("Aida", ["Abbado Celeste", "Karajan Ritorna"]),
            # a recording of the work itself wins over those of its parts
            ("Otello", ["Solti Otello"]),
            ("Nabucco", []),
        ],
    )
    async def test_work_tracks(
        self, play_library: _Library, work: str, expected: list[str]
    ) -> None:
        """A work plays its first own recording, else the first recording of each part."""
        tracks = await play_library.mass.music.classical.work_tracks(play_library.work_id(work))

        assert [x.name for x in tracks] == expected

    async def test_a_work_address_resolves_to_its_tracks(self, play_library: _Library) -> None:
        """A work's address resolves to the work, which plays as the container of its tracks."""
        mass = play_library.mass
        work_id = play_library.work_id("Aida")
        resolver = MediaResolver(
            cast("PlayerQueuesController", SimpleNamespace(mass=mass, logger=Mock()))
        )

        work = await mass.music.get_item_by_uri(f"library://work/{work_id}")
        resolved = await resolver._resolve_media_items(work)

        assert isinstance(work, Work)
        assert work.uri == f"library://work/{work_id}"
        assert [x.item.name for x in resolved] == ["Abbado Celeste", "Karajan Ritorna"]
        assert [x.origin.container.uri for x in resolved if x.origin and x.origin.container] == [
            work.uri
        ] * 2


@pytest.fixture(scope="class", name="one_act_library")
async def one_act_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of a one-act opera, its tracks naming their numbers."""
    lib = await _new_library(music_mass_class, ("puccini", "freni"))
    await _add_work(lib, "Schicchi", "puccini", name="Gianni Schicchi", work_type=WorkType.OPERA)
    for number, (key, name) in enumerate(
        (("babbino", "O mio babbino caro"), ("firenze", "Firenze è come un albero")), 1
    ):
        await _add_track(
            lib,
            key,
            "Schicchi",
            "freni",
            name=f"Gianni Schicchi: {name}",
            movement=number,
            movement_name=name,
            role=None,
        )
    return lib


class TestOneActOpera:
    """An opera the library holds no acts of lists its numbers as its movements."""

    async def test_opera_without_acts_lists_its_numbers(self, one_act_library: _Library) -> None:
        """The numbers its tracks name are the movements of a one-act opera."""
        work = await one_act_library.mass.music.classical.work(one_act_library.work_id("Schicchi"))

        assert {x.name for x in work.movements} == {
            "O mio babbino caro",
            "Firenze è come un albero",
        }

    async def test_opera_lists_the_numbers_musicbrainz_gives_it(
        self, one_act_library: _Library
    ) -> None:
        """An opera MusicBrainz lists numbers right under lists all of them."""
        movements = [
            WorkMovement(number=x, name=name, mbid=str(uuid4()))
            for x, name in enumerate(
                ("O mio babbino caro", "Firenze è come un albero", "Ditemi voi, signori"), 1
            )
        ]
        schicchi = one_act_library.work_id("Schicchi")
        await _set_musicbrainz_movements(one_act_library, "Schicchi", movements)
        try:
            work = await one_act_library.mass.music.classical.work(schicchi)
            [recording] = await one_act_library.mass.music.classical.recordings(schicchi)
        finally:
            await _set_musicbrainz_movements(one_act_library, "Schicchi", None)

        assert work.movements == movements
        assert work.movements_from_musicbrainz
        assert (recording.movement_numbers, recording.complete) == ([1, 2], False)


@pytest.fixture(scope="class", name="opera_library")
async def opera_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera holding one of the four acts MusicBrainz lists."""
    lib = await _new_library(music_mass_class, ("puccini", "karajan", "levine"))
    act_mbids = [str(uuid4()) for _ in range(4)]
    aria_mbids = [str(uuid4()) for _ in range(2)]
    await _add_work(lib, "Manon", "puccini", name="Manon Lescaut", work_type=WorkType.OPERA)
    await _add_work(
        lib,
        "Act II",
        "puccini",
        name="Manon Lescaut: Atto II",
        parent="Manon",
        work_type=WorkType.OPERA,
        mbid=act_mbids[1],
    )
    for key, name, work, conductor, work_mbids in (
        ("aria", "Manon Lescaut: In quelle trine morbide", "Act II", "karajan", aria_mbids[:1]),
        # a recording of the whole opera, each track holding two acts
        ("first", "Manon Lescaut: Atti I e II", "Manon", "levine", act_mbids[:2]),
        ("second", "Manon Lescaut: Atti III e IV", "Manon", "levine", act_mbids[2:]),
    ):
        await _add_track(lib, key, work, conductor, name=name)
        await lib.mass.music.tracks.set_external_ids(
            lib.tracks[key].item_id, {(ExternalID.MB_WORK, x) for x in work_mbids}
        )
    await _set_musicbrainz_movements(
        lib,
        "Manon",
        [
            WorkMovement(number=x, name=f"Atto {numeral}", mbid=mbid)
            for x, (numeral, mbid) in enumerate(
                zip(("I", "II", "III", "IV"), act_mbids, strict=True), 1
            )
        ],
    )
    await _set_musicbrainz_movements(
        lib,
        "Act II",
        [
            WorkMovement(number=x, name=name, mbid=mbid)
            for x, (name, mbid) in enumerate(
                zip(("In quelle trine morbide", "Tu, tu, amore? Tu?!"), aria_mbids, strict=True), 1
            )
        ],
    )
    return lib


class TestOperaActs:
    """An opera lists the acts MusicBrainz gives it, the library's acts mapped onto them."""

    async def test_opera_lists_all_its_acts(self, opera_library: _Library) -> None:
        """All acts are listed, the act the library holds linking to its work."""
        work = await opera_library.mass.music.classical.work(opera_library.work_id("Manon"))

        assert [(x.number, x.name, x.work_id) for x in work.movements] == [
            (1, "Atto I", None),
            (2, "Atto II", opera_library.work_id("Act II")),
            (2001, "In quelle trine morbide", None),
            (2002, "Tu, tu, amore? Tu?!", None),
            (3, "Atto III", None),
            (4, "Atto IV", None),
        ]
        assert work.movements_from_musicbrainz

    async def test_recordings_cover_their_acts(self, opera_library: _Library) -> None:
        """A recording of an aria covers its act and that aria, one of all acts is complete."""
        recordings = await opera_library.mass.music.classical.recordings(
            opera_library.work_id("Manon")
        )

        assert {
            x.work.name: (
                x.movement_numbers,
                x.complete,
                [y.name for y in x.part_movements or []],
            )
            for x in recordings
        } == {
            "Manon Lescaut": ([1, 2, 3, 4], True, []),
            "Manon Lescaut: Atto II": ([2], False, ["In quelle trine morbide"]),
        }


@pytest.fixture(scope="class", name="tagged_act_library")
async def tagged_act_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera whose one held act has no MusicBrainz numbers."""
    lib = await _new_library(music_mass_class, ("puccini", "karajan", "levine"))
    act_mbids = [str(uuid4()) for _ in range(2)]
    await _add_work(lib, "Opera", "puccini", name="La rondine", work_type=WorkType.OPERA)
    await _add_work(
        lib, "Act II", "puccini", name="La rondine: Atto II", parent="Opera", mbid=act_mbids[1]
    )
    for key, conductor, movement, movement_name in (
        ("full_1", "karajan", 1, "Nel cielo"),
        ("full_2", "karajan", 2, "Chi il bel sogno"),
        ("full_3", "karajan", 3, "Bevo al tuo fresco sorriso"),
        ("part_2", "levine", 2, "Chi il bel sogno"),
    ):
        await _add_track(
            lib, key, "Act II", conductor, movement=movement, movement_name=movement_name
        )
    await _set_musicbrainz_movements(
        lib,
        "Opera",
        [
            WorkMovement(number=x, name=f"Atto {numeral}", mbid=mbid)
            for x, (numeral, mbid) in enumerate(zip(("I", "II"), act_mbids, strict=True), 1)
        ],
    )
    return lib


class TestTaggedActNumbers:
    """An act without MusicBrainz numbers lists the numbers its tracks name on its opera."""

    async def test_opera_lists_the_numbers_its_act_tracks_name(
        self, tagged_act_library: _Library
    ) -> None:
        """The act is followed by the numbers its own page lists from its tracks."""
        work = await tagged_act_library.mass.music.classical.work(
            tagged_act_library.work_id("Opera")
        )

        assert [(x.number, x.name, x.parent) for x in work.movements] == [
            (1, "Atto I", None),
            (2, "Atto II", None),
            (2001, "Nel cielo", 2),
            (2002, "Chi il bel sogno", 2),
            (2003, "Bevo al tuo fresco sorriso", 2),
        ]

    async def test_partial_recording_covers_the_numbers_it_holds(
        self, tagged_act_library: _Library
    ) -> None:
        """A recording of some of the act's numbers names them, its tracks numbered as listed."""
        recordings = await tagged_act_library.mass.music.classical.recordings(
            tagged_act_library.work_id("Opera")
        )

        assert {
            x.credits[0].artist.name: (
                x.movement_numbers,
                [y.number for y in x.part_movements or []],
                [t.movement_number for t in x.tracks],
            )
            for x in recordings
        } == {
            "Karajan": ([2], [], [2001, 2002, 2003]),
            "Levine": ([2], [2], [2002]),
        }


@pytest.fixture(scope="class", name="staged_library")
async def staged_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera with acts and arias and a Passion in parts."""
    lib = await _new_library(music_mass_class, ("puccini", "bach", "karajan", "richter"))
    mbids = {"vissi": str(uuid4()), "ed_io": str(uuid4())}
    for key, name, composer, parent, work_type in (
        ("Tosca", "Tosca", "puccini", None, WorkType.OPERA),
        ("Act I", "Tosca, Act I", "puccini", "Tosca", None),
        ("Recondita", "Tosca: Act I: Recondita armonia", "puccini", "Act I", None),
        ("Act II", "Tosca, Act II", "puccini", "Tosca", None),
        ("Vissi", "Tosca: Act II: Vissi d'arte", "puccini", "Act II", None),
        ("Ed io", "Tosca: Act II: Ed io venivo a lei", "puccini", "Act II", None),
        ("Act III", "Tosca, Act III", "puccini", "Tosca", None),
        ("Passion", "St Matthew Passion", "bach", None, None),
        ("Part II", "St Matthew Passion, Part II", "bach", "Passion", None),
        ("Part I", "St Matthew Passion, Part I", "bach", "Passion", None),
    ):
        mbid = mbids.get(key.lower().replace(" ", "_"))
        await _add_work(
            lib, key, composer, name=name, parent=parent, work_type=work_type, mbid=mbid
        )
    for key, name, work, conductor, movement in (
        ("opera_recondita", "Tosca: Recondita armonia", "Tosca", "karajan", None),
        ("opera_vissi", "Tosca: Vissi d'arte", "Tosca", "karajan", None),
        ("act_ii", "Tosca, Act II: Scene", "Act II", "karajan", None),
        ("recondita", "Tosca: Act I: Recondita armonia", "Recondita", "richter", None),
        ("vissi", "Tosca: Act II: Vissi d'arte", "Vissi", "richter", None),
        ("ed_io", "Tosca: Act II: Ed io venivo a lei", "Ed io", "richter", None),
        ("part_i_1", "St Matthew Passion, Part I: No. 1 Kommt", "Part I", "richter", 1),
        ("part_i_2", "St Matthew Passion, Part I: No. 2 Da Jesus", "Part I", "richter", 2),
        ("part_ii", "St Matthew Passion, Part II: No. 63 Wahrlich", "Part II", "richter", 63),
    ):
        await _add_track(lib, key, work, conductor, name=name, movement=movement)
    return lib


class TestPartsAsMovements:
    """A work whose parts have parts lists its parts with tracks as movements."""

    async def test_parts_of_parts_count_in_their_top_work_row(
        self, staged_library: _Library
    ) -> None:
        """Only the top works are rows, each counting the recordings of its parts at any depth."""
        rows = await staged_library.mass.music.classical.works()
        [composer] = await staged_library.mass.music.classical.composers(
            artist_id=staged_library.artist_id("puccini")
        )

        assert {(x.work.name, x.recording_count) for x in rows} == {
            ("Tosca", 5),
            ("St Matthew Passion", 2),
        }
        assert (composer.work_count, composer.recording_count) == (1, 5)

    @pytest.mark.parametrize(
        ("composer", "performer", "search"),
        [("puccini", None, None), (None, "richter", "recondita"), (None, None, "vissi")],
    )
    async def test_filters_find_the_top_work_of_a_part_of_a_part(
        self,
        staged_library: _Library,
        composer: str | None,
        performer: str | None,
        search: str | None,
    ) -> None:
        """A composer, performer or search matching an aria of an act finds the opera."""
        rows = await staged_library.mass.music.classical.works(
            composer_id=staged_library.artist_id(composer) if composer else None,
            performer_id=staged_library.artist_id(performer) if performer else None,
            search=search,
        )

        assert [x.work.name for x in rows] == ["Tosca"]

    async def test_opera_lists_its_parts_with_tracks_in_work_order(
        self, staged_library: _Library
    ) -> None:
        """Acts come by number, arias after their act by the order MusicBrainz gives them."""
        # MusicBrainz lists the arias of the second act against the order of their names
        await _set_musicbrainz_movements(
            staged_library,
            "Act II",
            [
                WorkMovement(
                    number=1, name="Vissi d'arte", mbid=staged_library.works["Vissi"].mbid
                ),
                WorkMovement(number=2, name="Ed io", mbid=staged_library.works["Ed io"].mbid),
            ],
        )
        try:
            work = await staged_library.mass.music.classical.work(staged_library.work_id("Tosca"))
        finally:
            await _set_musicbrainz_movements(staged_library, "Act II", None)

        assert [(x.number, x.name, x.mbid) for x in work.movements] == [
            (1, "Recondita armonia", None),
            (2, "Act II", None),
            (3, "Vissi d'arte", staged_library.works["Vissi"].mbid),
            (4, "Ed io venivo a lei", staged_library.works["Ed io"].mbid),
        ]
        # an aria sits under its act when that is listed, at the top when its act has no tracks
        assert [x.parent for x in work.movements] == [None, None, 2, 2]
        assert not work.movements_from_musicbrainz

    async def test_acts_show_as_act_with_a_roman_number(self, staged_library: _Library) -> None:
        """An act named after a service's title, as "Tosca, SC 69 / Act 2", shows as "Act II"."""
        works = staged_library.mass.music.works
        act_id = int(staged_library.work_id("Act II"))
        await works.rename_library_work(act_id, "Tosca, SC 69 / Act 2", [])
        try:
            work = await staged_library.mass.music.classical.work(staged_library.work_id("Tosca"))
        finally:
            # the library is shared with the other tests of the module
            await works.rename_library_work(act_id, "Tosca, Act II", [])

        assert "Act II" in [x.name for x in work.movements]
        assert not any("SC 69" in (x.name or "") for x in work.movements)

    async def test_opera_recordings_cover_the_parts_their_tracks_hold(
        self, staged_library: _Library
    ) -> None:
        """A recording of an aria covers that aria, one of the whole opera each aria it holds."""
        recordings = await staged_library.mass.music.classical.recordings(
            staged_library.work_id("Tosca")
        )

        # without MusicBrainz movements the arias of an act come in the order of their names
        assert {x.work.name: (x.movement_numbers, x.complete) for x in recordings} == {
            "Tosca": ([1, 4], None),
            "Tosca: Act I: Recondita armonia": ([1], None),
            "Tosca, Act II": ([2], None),
            "Tosca: Act II: Ed io venivo a lei": ([3], None),
            "Tosca: Act II: Vissi d'arte": ([4], None),
        }
        opera = next(x for x in recordings if x.work.name == "Tosca")
        assert {(x.name, x.movement_number, x.movement_name) for x in opera.tracks} == {
            ("Tosca: Recondita armonia", 1, "Recondita armonia"),
            ("Tosca: Vissi d'arte", 4, "Vissi d'arte"),
        }

    async def test_passion_lists_its_parts(self, staged_library: _Library) -> None:
        """The parts of a Passion are its movements, each track showing under its part."""
        passion = staged_library.work_id("Passion")
        work = await staged_library.mass.music.classical.work(passion)
        recordings = await staged_library.mass.music.classical.recordings(passion)

        assert [(x.number, x.name) for x in work.movements] == [(1, "Part I"), (2, "Part II")]
        assert {
            x.work.name: (x.movement_numbers, [t.movement_number for t in x.tracks])
            for x in recordings
        } == {
            "St Matthew Passion, Part I": ([1], [1, 1]),
            "St Matthew Passion, Part II": ([2], [2]),
        }

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("Passion", ["Part I", "Part II"]),
            ("Tosca", ["Act I", "Act II", "Act III"]),
            ("Act II", ["Ed io venivo a lei", "Vissi d'arte"]),
        ],
    )
    async def test_parts_are_named_as_parts_of_their_work(
        self, staged_library: _Library, key: str, expected: list[str]
    ) -> None:
        """A work's parts leave out the start of their names they share with it."""
        work = await staged_library.mass.music.classical.work(staged_library.work_id(key))

        assert sorted(x.name for x in work.parts) == expected

    async def test_parts_listed_as_musicbrainz_movements_count_as_them(
        self, staged_library: _Library
    ) -> None:
        """A part that is one of the movements MusicBrainz lists covers that movement."""
        movements = [
            WorkMovement(number=x, name=f"Teil {numeral}", mbid=str(uuid4()))
            for x, numeral in ((1, "I"), (2, "II"))
        ]
        await _set_musicbrainz_movements(staged_library, "Passion", movements)
        for key, movement in (("Part I", movements[0]), ("Part II", movements[1])):
            await _set_work_musicbrainz_id(staged_library, key, movement.mbid)
        try:
            recordings = await staged_library.mass.music.classical.recordings(
                staged_library.work_id("Passion")
            )
        finally:
            await _set_musicbrainz_movements(staged_library, "Passion", None)
            for key in ("Part I", "Part II"):
                await _set_work_musicbrainz_id(staged_library, key, None)

        assert {x.work.name: (x.movement_numbers, x.complete) for x in recordings} == {
            "St Matthew Passion, Part I": ([1], False),
            "St Matthew Passion, Part II": ([2], False),
        }

    async def test_act_of_an_opera_lists_its_own_movements(self, staged_library: _Library) -> None:
        """An act typed as an opera lists the movements MusicBrainz gives it."""
        movements = [
            WorkMovement(number=1, name="Vissi d'arte", mbid=staged_library.works["Vissi"].mbid),
            WorkMovement(number=2, name="Ed io", mbid=staged_library.works["Ed io"].mbid),
        ]
        await _set_musicbrainz_movements(staged_library, "Act II", movements)
        database = staged_library.mass.music.database
        work_id = int(staged_library.work_id("Act II"))
        await database.execute_write(
            f"UPDATE {DB_TABLE_WORKS} SET work_type = :work_type WHERE item_id = :work_id",
            {"work_type": WorkType.OPERA.value, "work_id": work_id},
        )
        try:
            work = await staged_library.mass.music.classical.work(str(work_id))
            recordings = await staged_library.mass.music.classical.recordings(str(work_id))
        finally:
            await _set_musicbrainz_movements(staged_library, "Act II", None)
            await database.execute_write(
                f"UPDATE {DB_TABLE_WORKS} SET work_type = NULL WHERE item_id = :work_id",
                {"work_id": work_id},
            )

        # the arias the library holds as works link to them
        assert work.movements == [
            replace(x, work_id=staged_library.work_id(key))
            for x, key in zip(movements, ("Vissi", "Ed io"), strict=True)
        ]
        assert work.movements_from_musicbrainz
        assert {x.work.name: (x.movement_numbers, x.complete) for x in recordings} == {
            "Tosca, Act II": ([], False),
            "Tosca: Act II: Vissi d'arte": ([1], False),
            "Tosca: Act II: Ed io venivo a lei": ([2], False),
        }


@pytest.fixture(scope="class", name="aria_library")
async def aria_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera whose act lists arias MusicBrainz names."""
    lib = await _new_library(music_mass_class, ("puccini", "kaufmann", "karajan", "richter"))
    for key, name, parent, work_type in (
        ("Bohème", "La bohème", None, WorkType.OPERA),
        ("Act I", "La bohème, Act I", "Bohème", None),
        ("Gelida", 'Atto I, no. 6. "Che gelida manina!" (Rodolfo)', "Act I", None),
        # MusicBrainz names some arias after the opera they are in
        ("Soave", 'La bohème: Atto I, no. 9. "O soave fanciulla" (Rodolfo, Mimì)', "Act I", None),
        ("Act II", "La bohème, Act II", "Bohème", None),
    ):
        await _add_work(lib, key, "puccini", name=name, parent=parent, work_type=work_type)
    for key, name, work, conductor in (
        ("kaufmann_gelida", "La bohème, Act I: Che gelida manina", "Act I", "kaufmann"),
        ("kaufmann_soave", "La bohème, Act I: O soave fanciulla", "Act I", "kaufmann"),
        ("karajan_mar", "La bohème, Act I: Questo Mar Rosso", "Act I", "karajan"),
        ("karajan_gelida", "La bohème, Act I: Che gelida manina", "Act I", "karajan"),
        ("karajan_soave", "La bohème, Act I: O soave fanciulla", "Act I", "karajan"),
        ("karajan_act_ii", "La bohème, Act II: Viva Parpignol!", "Act II", "karajan"),
        ("richter_gelida", "Che gelida manina!", "Gelida", "richter"),
        ("richter_soave", "O soave fanciulla", "Soave", "richter"),
    ):
        await _add_track(lib, key, work, conductor, name=name)
    return lib


class TestAriasOfAnAct:
    """Arias with works of their own sit under their act, the tracks of the act count for it."""

    async def test_arias_sit_under_their_act(self, aria_library: _Library) -> None:
        """Each aria comes after its act and names it as its parent."""
        work = await aria_library.mass.music.classical.work(aria_library.work_id("Bohème"))

        assert [(x.number, x.name, x.parent) for x in work.movements] == [
            (1, "Act I", None),
            (2, 'Atto I, no. 6. "Che gelida manina!" (Rodolfo)', 1),
            (3, 'Atto I, no. 9. "O soave fanciulla" (Rodolfo, Mimì)', 1),
            (4, "Act II", None),
        ]
        assert [x.work_id for x in work.movements[:3]] == [
            aria_library.work_id(key) for key in ("Act I", "Gelida", "Soave")
        ]

    async def test_act_tracks_count_for_the_act(self, aria_library: _Library) -> None:
        """Tracks of an act count for the act, tracks of an aria for the aria."""
        recordings = await aria_library.mass.music.classical.recordings(
            aria_library.work_id("Bohème")
        )
        by_conductor = {
            (x.work.name, x.credits[0].artist.name): (
                x.movement_numbers,
                [t.movement_number for t in x.tracks],
            )
            for x in recordings
        }

        assert by_conductor == {
            ("La bohème, Act I", "Kaufmann"): ([1], [1, 1]),
            ("La bohème, Act I", "Karajan"): ([1], [1, 1, 1]),
            ("La bohème, Act II", "Karajan"): ([4], [4]),
            ('Atto I, no. 6. "Che gelida manina!" (Rodolfo)', "Richter"): ([2], [2]),
            ('La bohème: Atto I, no. 9. "O soave fanciulla" (Rodolfo, Mimì)', "Richter"): (
                [3],
                [3],
            ),
        }

    async def test_act_names_its_arias_as_its_opera_does(self, aria_library: _Library) -> None:
        """An act page leaves the names of the act and of its opera off its arias."""
        work = await aria_library.mass.music.classical.work(aria_library.work_id("Act I"))

        assert [(x.number, x.name) for x in work.movements] == [
            (1, 'Atto I, no. 6. "Che gelida manina!" (Rodolfo)'),
            (2, 'Atto I, no. 9. "O soave fanciulla" (Rodolfo, Mimì)'),
        ]


@pytest.fixture(scope="class", name="act_library")
async def act_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of a complete act whose aria tracks are its movements."""
    lib = await _new_library(
        music_mass_class,
        ("puccini", "karajan", "kaufmann", "pavarotti", "armiliato", "solti", "levine"),
    )
    await _add_work(lib, "Bohème", "puccini", name="La bohème", work_type=WorkType.OPERA)
    await _add_work(lib, "Act I", "puccini", name="La bohème, Act I", parent="Bohème")
    await _add_work(lib, "Act II", "puccini", name="La bohème, Act II", parent="Bohème")
    for key, name, work, movement_number, movement_name, conductor in (
        ("mar", "La bohème, Act I: Questo Mar Rosso", "Act I", None, None, "karajan"),
        (
            "gelida",
            "La bohème, Act I: Che gelida manina",
            "Act I",
            6,
            'no. 6. "Che gelida manina!"',
            "karajan",
        ),
        (
            "soave",
            "La bohème, Act I: O soave fanciulla",
            "Act I",
            9,
            'no. 9. "O soave fanciulla"',
            "karajan",
        ),
        ("parpignol", "La bohème, Act II: Viva Parpignol!", "Act II", None, None, "karajan"),
        ("kaufmann_gelida", "La bohème, Act I: Che gelida manina", "Act I", None, None, "kaufmann"),
        ("kaufmann_soave", "La bohème, Act I: O soave fanciulla", "Act I", None, None, "kaufmann"),
        ("pavarotti_gelida", "Che gelida manina", "Act I", None, None, "pavarotti"),
        (
            "armiliato_gelida",
            "La bohème, Act I: Che gelida manina",
            "Act I",
            None,
            None,
            "armiliato",
        ),
        ("armiliato_mimi", "La bohème, Act I: Entrata di Mimì", "Act I", None, None, "armiliato"),
        ("solti_duet", "La bohème, Act I: Duet", "Act I", None, None, "solti"),
        ("levine_gelida", "La bohème: Che gelida manina", "Bohème", None, None, "levine"),
    ):
        await _add_track(
            lib,
            key,
            work,
            conductor,
            name=name,
            movement=movement_number,
            movement_name=movement_name,
        )
    gelida_mbid, soave_mbid = str(uuid4()), str(uuid4())
    # a track performing two arias of the act holds the MusicBrainz works of both
    await lib.mass.music.tracks.set_external_ids(
        lib.tracks["solti_duet"].item_id,
        {(ExternalID.MB_WORK, gelida_mbid), (ExternalID.MB_WORK, soave_mbid)},
    )
    # a track of the opera itself holding an aria of an act
    await lib.mass.music.tracks.set_external_ids(
        lib.tracks["levine_gelida"].item_id, {(ExternalID.MB_WORK, gelida_mbid)}
    )
    await _set_musicbrainz_movements(
        lib,
        "Act I",
        [
            WorkMovement(number=1, name='no. 1. "Questo Mar Rosso"', mbid=str(uuid4())),
            WorkMovement(number=6, name='Atto I, no. 6. "Che gelida manina!"', mbid=gelida_mbid),
            WorkMovement(number=9, name='Atto I, no. 9. "O soave fanciulla"', mbid=soave_mbid),
        ],
    )
    return lib


class TestAriasAsMovementsOfAnAct:
    """A complete act whose arias are its movements counts as the whole act."""

    async def test_complete_act_covers_the_act(self, act_library: _Library) -> None:
        """Every track of the act, its arias too, counts for the act on the opera page."""
        recordings = await act_library.mass.music.classical.recordings(
            act_library.work_id("Bohème")
        )

        assert {
            x.work.name: (x.movement_numbers, [t.movement_number for t in x.tracks])
            for x in recordings
            if x.credits[0].artist.name == "Karajan"
        } == {
            "La bohème, Act I": ([1], [1001, 1006, 1009]),
            "La bohème, Act II": ([2], [2]),
        }

    async def test_opera_lists_the_numbers_of_its_acts(self, act_library: _Library) -> None:
        """Each act is followed by its own numbers, named as the act's page names them."""
        work = await act_library.mass.music.classical.work(act_library.work_id("Bohème"))

        assert [(x.number, x.name, x.parent) for x in work.movements] == [
            (1, "Act I", None),
            (1001, 'no. 1. "Questo Mar Rosso"', 1),
            (1006, 'Atto I, no. 6. "Che gelida manina!"', 1),
            (1009, 'Atto I, no. 9. "O soave fanciulla"', 1),
            (2, "Act II", None),
        ]

    async def test_opera_tracks_carry_the_number_they_hold_in_their_act(
        self, act_library: _Library
    ) -> None:
        """A track of an act carries its number in the act, keeping its name and its place."""
        recordings = await act_library.mass.music.classical.recordings(
            act_library.work_id("Bohème")
        )

        assert {
            x.credits[0].artist.name: (
                x.movement_numbers,
                [(t.name, t.movement_number, t.movement_name) for t in x.tracks],
            )
            for x in recordings
            if x.work.name == "La bohème, Act I"
        } == {
            "Armiliato": (
                [1],
                [
                    ("La bohème, Act I: Che gelida manina", 1006, "Act I"),
                    # a track matching none of the act's numbers keeps the act's
                    ("La bohème, Act I: Entrata di Mimì", 1, "Act I"),
                ],
            ),
            "Karajan": (
                [1],
                [
                    ("La bohème, Act I: Questo Mar Rosso", 1001, "Act I"),
                    ("La bohème, Act I: Che gelida manina", 1006, '"Che gelida manina!"'),
                    ("La bohème, Act I: O soave fanciulla", 1009, '"O soave fanciulla"'),
                ],
            ),
            "Kaufmann": (
                [1],
                [
                    ("La bohème, Act I: Che gelida manina", 1006, "Act I"),
                    ("La bohème, Act I: O soave fanciulla", 1009, "Act I"),
                ],
            ),
            "Pavarotti": ([1], [("Che gelida manina", 1006, "Act I")]),
            # a track holding two numbers carries the first
            "Solti": ([1], [("La bohème, Act I: Duet", 1006, "Act I")]),
        }

    async def test_opera_track_carries_no_number_of_an_act(self, act_library: _Library) -> None:
        """A track of the opera itself never carries a number of an act, even one it holds."""
        recordings = await act_library.mass.music.classical.recordings(
            act_library.work_id("Bohème")
        )

        levine = next(x for x in recordings if x.credits[0].artist.name == "Levine")
        assert [x.movement_number for x in levine.tracks] == [None]

    async def test_act_recording_plays_in_the_order_of_the_movements_it_shows(
        self, act_library: _Library
    ) -> None:
        """A track showing a movement plays in its order, named without the number it opens with."""
        recordings = await act_library.mass.music.classical.recordings(act_library.work_id("Act I"))

        karajan = next(x for x in recordings if x.credits[0].artist.name == "Karajan")
        assert [(x.movement_number, x.movement_name) for x in karajan.tracks] == [
            (1, '"Questo Mar Rosso"'),
            (6, '"Che gelida manina!"'),
            (9, '"O soave fanciulla"'),
        ]

    async def test_excerpt_names_the_movements_of_the_act_it_covers(
        self, act_library: _Library
    ) -> None:
        """An excerpt of an act names the movements it covers, a recording of the act none."""
        recordings = await act_library.mass.music.classical.recordings(
            act_library.work_id("Bohème")
        )

        assert {
            x.credits[0].artist.name: (x.movement_numbers, [m.number for m in x.part_movements])
            for x in recordings
            if x.work.name == "La bohème, Act I"
        } == {
            "Karajan": ([1], []),
            "Kaufmann": ([1], [6, 9]),
            "Pavarotti": ([1], [6]),
            # a track holding none of the listed movements makes it a recording of the act
            "Armiliato": ([1], []),
            "Solti": ([1], [6, 9]),
        }


@pytest.fixture(scope="class", name="scene_library")
async def scene_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of a Passion whose second part holds a scene of its own."""
    lib = await _new_library(music_mass_class, ("bach", "jacobs", "richter"))
    mbids = {key: str(uuid4()) for key in ("Part I", "Part II", "Scene", "XXX", "XXXVII", "XXXIX")}
    for key, name, parent in (
        ("Passion", "St Matthew Passion", None),
        ("Part II", "St Matthew Passion: Part II", "Passion"),
        ("Scene", "St Matthew Passion: Part II, XXXVIa - XXXVIc", "Part II"),
    ):
        await _add_work(lib, key, "bach", name=name, parent=parent, mbid=mbids.get(key))
    for key, movements in (
        ("Passion", [(1, "Part I"), (2, "Part II")]),
        ("Part II", [(1, "XXX"), (2, "Scene"), (3, "XXXVII"), (4, "XXXIX")]),
    ):
        await _set_musicbrainz_movements(
            lib,
            key,
            [WorkMovement(number=x, name=name, mbid=mbids[name]) for x, name in movements],
        )
    await _set_musicbrainz_movements(
        lib,
        "Scene",
        [
            WorkMovement(number=x, name=name, mbid=str(uuid4()))
            for x, name in ((1, "XXXVIa"), (2, "XXXVIb"), (3, "XXXVIc"))
        ],
    )
    # the numbers of the scene's tracks are those of its own pieces, never those of the act
    for key, work, movement, conductor in (
        ("jacobs_xxx", "Part II", 1, "jacobs"),
        ("jacobs_xxxvi", "Scene", 3, "jacobs"),
        ("jacobs_xxxvii", "Part II", 3, "jacobs"),
        ("richter_xxxvi", "Scene", 1, "richter"),
    ):
        await _add_track(lib, key, work, conductor, movement=movement)
    return lib


class TestScenesOfAnAct:
    """A scene of an act counts as the one movement of the act it is."""

    async def test_work_lists_the_movements_musicbrainz_gives_it(
        self, scene_library: _Library
    ) -> None:
        """A work whose parts have parts lists the movements MusicBrainz gives it."""
        work = await scene_library.mass.music.classical.work(scene_library.work_id("Passion"))

        assert [(x.number, x.name, x.work_id) for x in work.movements if x.parent is None] == [
            (1, "Part I", None),
            (2, "Part II", scene_library.work_id("Part II")),
        ]
        assert work.movements_from_musicbrainz

    async def test_scene_track_joins_the_recording_of_its_act(
        self, scene_library: _Library
    ) -> None:
        """A scene's track of the act's performance joins it, holding the scene's movement."""
        recordings = await scene_library.mass.music.classical.recordings(
            scene_library.work_id("Passion")
        )

        jacobs = next(x for x in recordings if x.credits[0].artist.name == "Jacobs")
        assert jacobs.work.name == "St Matthew Passion: Part II"
        assert sorted(_track_names(jacobs)) == ["jacobs_xxx", "jacobs_xxxvi", "jacobs_xxxvii"]
        assert [x.movement_number for x in jacobs.tracks] == [2001, 2002003, 2003]
        assert (jacobs.movement_numbers, [x.name for x in jacobs.part_movements]) == (
            [2],
            ["XXX", "Scene", "XXXVII"],
        )

    async def test_lone_scene_recording_counts_as_one_movement_of_its_act(
        self, scene_library: _Library
    ) -> None:
        """A recording of only the scene covers the act's movement the scene is, never all."""
        recordings = await scene_library.mass.music.classical.recordings(
            scene_library.work_id("Passion")
        )

        richter = next(x for x in recordings if x.credits[0].artist.name == "Richter")
        assert richter.work.item_id == scene_library.work_id("Scene")
        assert richter.parent_work
        assert richter.parent_work.item_id == scene_library.work_id("Part II")
        assert (richter.movement_numbers, richter.complete) == ([2], False)
        assert [x.name for x in richter.part_movements] == ["Scene"]

    async def test_work_lists_the_movements_of_its_part_under_it(
        self, scene_library: _Library
    ) -> None:
        """A part's own movements follow it, the pieces of its scene one level deeper."""
        work = await scene_library.mass.music.classical.work(scene_library.work_id("Passion"))

        assert [(x.number, x.name, x.parent) for x in work.movements] == [
            (1, "Part I", None),
            (2, "Part II", None),
            (2001, "XXX", 2),
            (2002, "Scene", 2),
            (2002001, "XXXVIa", 2002),
            (2002002, "XXXVIb", 2002),
            (2002003, "XXXVIc", 2002),
            (2003, "XXXVII", 2),
            (2004, "XXXIX", 2),
        ]

    async def test_scene_tracks_carry_their_piece_in_the_part(
        self, scene_library: _Library
    ) -> None:
        """A track of a scene carries the number of its piece within the part."""
        recordings = await scene_library.mass.music.classical.recordings(
            scene_library.work_id("Passion")
        )

        assert {
            (x.credits[0].artist.name, t.name): t.movement_number
            for x in recordings
            for t in x.tracks
        } == {
            ("Jacobs", "jacobs_xxx"): 2001,
            ("Jacobs", "jacobs_xxxvi"): 2002003,
            ("Jacobs", "jacobs_xxxvii"): 2003,
            ("Richter", "richter_xxxvi"): 2002001,
        }

    async def test_performer_recordings_carry_the_numbers_of_the_work(
        self, scene_library: _Library
    ) -> None:
        """A performer's recordings number their tracks by the movements of the listed work."""
        recordings = await scene_library.mass.music.classical.recordings(
            performer_id=scene_library.artist_id("jacobs")
        )

        [jacobs] = recordings
        assert (jacobs.movement_numbers, [x.movement_number for x in jacobs.tracks]) == (
            [2],
            [2, 2, 2],
        )

    async def test_joined_scene_counts_once_in_the_work_row(self, scene_library: _Library) -> None:
        """The work row counts a scene that joined its act's recording as part of that one."""
        rows = await scene_library.mass.music.classical.works()

        assert [(x.work.name, x.recording_count) for x in rows] == [("St Matthew Passion", 2)]

    async def test_act_lists_the_pieces_of_a_scene_under_it(self, scene_library: _Library) -> None:
        """The act's own movements list the pieces of its scene under it, past its own numbers."""
        work = await scene_library.mass.music.classical.work(scene_library.work_id("Part II"))

        assert [(x.number, x.name, x.parent) for x in work.movements] == [
            (1, "XXX", None),
            (2, "Scene", None),
            (2001, "XXXVIa", 2),
            (2002, "XXXVIb", 2),
            (2003, "XXXVIc", 2),
            (3, "XXXVII", None),
            (4, "XXXIX", None),
        ]

    async def test_stored_pieces_list_under_their_group_named_by_its_number(
        self, scene_library: _Library
    ) -> None:
        """Stored pieces list under their group, a group joining their names named by its number."""
        part_ii = await scene_library.mass.music.works.get_library_item(
            scene_library.work_id("Part II")
        )
        assert part_ii.metadata.movements
        listed = [
            replace(x, name="XXXVIIa. Choral - XXXVIIb. Coro") if x.number == 3 else x
            for x in part_ii.metadata.movements
        ]
        pieces = [
            WorkMovement(number=3001, name="XXXVIIa. Choral", mbid=str(uuid4()), parent=3),
            WorkMovement(number=3002, name="XXXVIIb. Coro", mbid=str(uuid4()), parent=3),
        ]
        await _set_musicbrainz_movements(scene_library, "Part II", [*listed, *pieces])
        try:
            work = await scene_library.mass.music.classical.work(scene_library.work_id("Part II"))
            recordings = await scene_library.mass.music.classical.recordings(
                scene_library.work_id("Passion")
            )
        finally:
            await _set_musicbrainz_movements(scene_library, "Part II", part_ii.metadata.movements)

        assert [(x.number, x.name) for x in work.movements][5:] == [
            (3, "XXXVII"),
            (3001, "XXXVIIa. Choral"),
            (3002, "XXXVIIb. Coro"),
            (4, "XXXIX"),
        ]
        # what a recording holds is still counted in the act's own movements alone
        jacobs = next(x for x in recordings if x.credits[0].artist.name == "Jacobs")
        assert [x.number for x in jacobs.part_movements] == [1, 2, 3]

    async def test_track_holding_one_piece_shows_it(self, scene_library: _Library) -> None:
        """On the act, a track holding one piece of the scene shows that piece, in its place."""
        recordings = await scene_library.mass.music.classical.recordings(
            scene_library.work_id("Part II")
        )

        assert {
            x.credits[0].artist.name: [(t.name, t.movement_number) for t in x.tracks]
            for x in recordings
            if x.work.item_id == scene_library.work_id("Scene")
        } == {"Jacobs": [("jacobs_xxxvi", 2003)], "Richter": [("richter_xxxvi", 2001)]}

    async def test_track_holding_the_whole_scene_shows_the_scene(
        self, scene_library: _Library
    ) -> None:
        """A track holding several pieces of the scene shows the scene alone."""
        scene = await scene_library.mass.music.classical.work(scene_library.work_id("Scene"))
        await _set_musicbrainz_work(scene_library, ["richter_xxxvi"], scene.movements[0].mbid)
        await _set_musicbrainz_work(scene_library, ["richter_xxxvi"], scene.movements[1].mbid)
        try:
            recordings = await scene_library.mass.music.classical.recordings(
                scene_library.work_id("Part II")
            )
        finally:
            await _set_musicbrainz_work(scene_library, ["richter_xxxvi"], None)

        richter = next(x for x in recordings if x.credits[0].artist.name == "Richter")
        assert [x.movement_number for x in richter.tracks] == [2]


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [
        ([2, 1001, 1], [1, 1001, 2]),
        ([1002, 1001003, 1001, 1], [1, 1001, 1001003, 1002]),
        ([2001, 1999, 1001001, 2, 1], [1, 1001001, 1999, 2, 2001]),
    ],
)
def test_movement_order(numbers: list[int], expected: list[int]) -> None:
    """A piece comes right after its group, at any depth."""
    assert sorted(numbers, key=_movement_order) == expected


@pytest.mark.parametrize(
    ("name", "pieces", "expected"),
    [
        ("IVa. Evangelist - IVb. Chor", ["IVa. Evangelist", "IVb. Chor"], "IV"),
        ("XXVII. Arie - XXVIIb. Cori", ["XXVIIa. Arie", "XXVIIb. Cori"], "XXVII"),
        ("no. 13: Danses des cygnes", ["I. Tempo di valse", "II. Moderato"], None),
        ("I. Allegro - II. Andante", ["I. Allegro", "II. Andante"], None),
        ("IVa. Evangelist - IVb. Chor", ["IVa. Evangelist"], None),
    ],
)
def test_shared_score_number(name: str, pieces: list[str], expected: str | None) -> None:
    """A group is named by its pieces' score number only when its name joins their names."""
    movements = [WorkMovement(number=x, name=piece) for x, piece in enumerate(pieces, 1)]

    assert _shared_score_number(name, movements) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "Excerpt from Die Walküre, WWV 86B: Akt III, Scene I, Walkürenritt",
            "Walkürenritt (excerpt)",
        ),
        (
            'Excerpt from Die Meistersinger von Nürnberg, WWV 96: Akt III, Scene V. "Morgenlich '
            'leuchtend" (Preislied)',
            '"Morgenlich leuchtend" (Preislied) (excerpt)',
        ),
        (
            "Excerpt from Die Meistersinger von Nürnberg: Akt III, Szene V. Tanz der Lehrbuben",  # codespell:ignore
            "Tanz der Lehrbuben (excerpt)",
        ),
        (
            'Matthäus-Passion, BWV 244: Teil II, XXXVIa. "Und der Hohepriester antwortete"',
            'XXXVIa. "Und der Hohepriester antwortete"',
        ),
        ("I. Tempo di valse", "I. Tempo di valse"),
    ],
)
def test_piece_name(name: str, expected: str) -> None:
    """A piece is named as a part of its act, an excerpt by its own name, marked as one."""
    names = ["Matthäus-Passion, BWV 244: Teil II", "St Matthew Passion, BWV 244"]

    assert _piece_name(name, names) == expected


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


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Piano Concerto No. 21 in C Major, K. 467", WorkType.CONCERTO),
        (
            'The Four Seasons, Violin Concerto No. 2 in G Minor, RV 315 "L\'estate"',
            WorkType.CONCERTO,
        ),
        (
            'Grand Sonata No. 3 in F Minor, Op. 14 "Concert sans orchestre"',
            WorkType.SONATA,
        ),
        ("Sonata «Concerto»", WorkType.SONATA),
        ("Missa Papae Marcelli", WorkType.MASS),
        ("Symphonie fantastique in C Minor, Op. 14", WorkType.SYMPHONY),
        ("The Sorcerer's Apprentice (symphonic poem)", WorkType.SYMPHONIC_POEM),
        ("String Quartet in D Minor, Op. 103", WorkType.QUARTET),
        ("3 Études, Op. 65", WorkType.ETUDE),
        ("Carmen (Opera) Suite No. 1", WorkType.SUITE),
        ("La scala di seta Opera: Overture", WorkType.OVERTURE),
        ("Madama Butterfly", None),
        ("Adagio for Strings, Op. 11", None),
        ('Prelude "Symphony"', None),
        ("Swan Lake (Ballet)", WorkType.BALLET),
        ('Dance of the Hours (Ballet) "La Gioconda"', None),
        ("Requiem in D minor, K. 626", None),
        ("Overture (Opera)", WorkType.OVERTURE),
        ("Aida (Opera)", WorkType.OPERA),
        ("Cavalleria Rusticana (Rustic Chivalry) Opera", WorkType.OPERA),
        ('Die Fledermaus "The Bat" Operetta', WorkType.OPERETTA),
        ("The Phantom of the Opera", None),
        ("La gazza ladra (The Thieving Magpie) Opera.", WorkType.OPERA),
        ("The Phantom of the Opera Suite", WorkType.SUITE),
        ("Opera Overture", WorkType.OVERTURE),
        ("Opera Arias", None),
    ],
)
def test_title_work_type(title: str, expected: WorkType | None) -> None:
    """A work title names its type as a whole word or bracket label, outside a quoted nickname."""
    assert _title_work_type(title) == expected


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


async def test_work_movements_come_from_its_tracks(library: _Library) -> None:
    """Without MusicBrainz movements, the tracks give them their most common name."""
    names = {"k1": "I. Allegro con brio", "b1": "I. Allegro con brio", "k2": "Andante"}
    names |= {"comp_k2": "II. Andante con moto", "b2": "II. Andante con moto"}
    for key, name in names.items():
        await _set_movement_name(library, key, name)
    try:
        work = await library.mass.music.classical.work(library.work_id("Fifth"))
        recordings = await library.mass.music.classical.recordings(library.work_id("Fifth"))
    finally:
        for key in names:
            await _set_movement_name(library, key, None)

    assert [(x.number, x.name, x.mbid) for x in work.movements] == [
        (1, "I. Allegro con brio", None),
        (2, "II. Andante con moto", None),
        (3, "", None),
        (4, "", None),
    ]
    assert not work.movements_from_musicbrainz
    assert [(x.movement_numbers, x.complete) for x in recordings] == [
        ([1, 2, 3, 4], None),
        ([1, 2, 3, 4], None),
    ]


async def test_work_movements_come_from_musicbrainz(library: _Library) -> None:
    """The movements MusicBrainz lists win over the tracks and tell complete recordings."""
    movements = [WorkMovement(number=x, name=f"Movement {x}", mbid=str(uuid4())) for x in (4, 1)]
    movements += [WorkMovement(number=x, name=f"Movement {x}") for x in (3, 2)]
    await _set_musicbrainz_movements(library, "Fifth", movements)
    try:
        work = await library.mass.music.classical.work(library.work_id("Fifth"))
        recordings = await library.mass.music.classical.recordings(library.work_id("Fifth"))
    finally:
        await _set_musicbrainz_movements(library, "Fifth", None)

    assert [x.number for x in work.movements] == [1, 2, 3, 4]
    assert work.movements[0] == movements[1]
    assert work.movements_from_musicbrainz
    assert work.metadata.movements is None
    assert [(x.movement_numbers, x.complete) for x in recordings] == [
        ([1, 2, 3, 4], True),
        ([1, 2, 3, 4], True),
    ]


async def test_musicbrainz_movement_numbers_still_match_beside_a_listed_part(
    library: _Library,
) -> None:
    """Tracks match the movements MusicBrainz lists by number, also when a part is one of them."""
    part_mbid = str(uuid4())
    movements = [
        WorkMovement(number=x, name=f"Movement {x}", mbid=part_mbid if x == 2 else str(uuid4()))
        for x in (1, 2, 3, 4)
    ]
    await _set_musicbrainz_movements(library, "Fifth", movements)
    works = library.mass.music.works
    part = await works.add_item_to_library(
        Work(
            item_id=uuid4().hex,
            provider=WORK_SOURCE,
            name="Fifth, Andante",
            provider_mappings=set(),
            parent_work=ItemMapping.from_item(library.works["Fifth"]),
            external_ids={(ExternalID.MB_WORK, part_mbid)},
        )
    )
    try:
        recordings = await library.mass.music.classical.recordings(library.work_id("Fifth"))
    finally:
        await works.remove_item_from_library(part.item_id)
        await _set_musicbrainz_movements(library, "Fifth", None)

    assert [(x.movement_numbers, x.complete) for x in recordings] == [
        ([1, 2, 3, 4], True),
        ([1, 2, 3, 4], True),
    ]


async def test_recording_covers_the_movement_of_its_musicbrainz_work(library: _Library) -> None:
    """A track's MusicBrainz work names the movement it covers, missing movements show."""
    movements = [WorkMovement(number=x, name=f"Movement {x}", mbid=str(uuid4())) for x in (1, 2, 3)]
    await _set_musicbrainz_movements(library, "Emperor", movements)
    keys = ("emperor_2", "emperor_box_2")
    await _set_musicbrainz_work(library, keys, movements[2].mbid)
    try:
        (recording,) = await library.mass.music.classical.recordings(library.work_id("Emperor"))
    finally:
        await _set_musicbrainz_movements(library, "Emperor", None)
        await _set_musicbrainz_work(library, keys, None)

    assert recording.movement_numbers == [1, 3]
    assert recording.complete is False


async def test_track_holding_two_movements_shows_the_one_it_is_named_after(
    library: _Library,
) -> None:
    """A track whose MusicBrainz works are two movements shows as the one it is named after."""
    names = ("Allegro", "Adagio un poco mosso", "Rondo: Allegro")
    movements = [
        WorkMovement(number=x, name=name, mbid=str(uuid4())) for x, name in enumerate(names, 1)
    ]
    await _set_musicbrainz_movements(library, "Emperor", movements)
    for movement in movements[1:]:
        await _set_musicbrainz_work(library, ["emperor_2"], movement.mbid)
    await _set_movement_name(library, "emperor_2", names[2])
    try:
        (recording,) = await library.mass.music.classical.recordings(library.work_id("Emperor"))
    finally:
        await _set_movement_name(library, "emperor_2", None)
        await _set_musicbrainz_movements(library, "Emperor", None)
        await _set_musicbrainz_work(library, ["emperor_2"], None)

    track_id = library.tracks["emperor_2"].item_id
    assert recording.movement_numbers == [1, 2, 3]
    assert [x.movement_number for x in recording.tracks if x.item_id == track_id] == [3]


async def test_track_matched_by_musicbrainz_work_carries_its_movement(
    library: _Library,
) -> None:
    """A track without a movement number shows under the movement its MusicBrainz work is."""
    movements = [
        WorkMovement(number=x, name=f"{numeral}. Movement", mbid=str(uuid4()))
        for x, numeral in enumerate(("I", "II", "III", "IV"), 1)
    ]
    await _set_musicbrainz_movements(library, "Fifth", movements)
    await _set_movement_name(library, "b1", "Mvt. I Allegro con brio")
    await _set_movement_number(library, "b1", None)
    await _set_musicbrainz_work(library, ["b1"], movements[0].mbid)
    try:
        recordings = await library.mass.music.classical.recordings(library.work_id("Fifth"))
    finally:
        await _set_musicbrainz_movements(library, "Fifth", None)
        await _set_movement_name(library, "b1", None)
        await _set_movement_number(library, "b1", 1)
        await _set_musicbrainz_work(library, ["b1"], None)

    bernstein = next(x for x in recordings if "Bernstein I" in _track_names(x))
    track = next(x for x in bernstein.tracks if x.name == "Bernstein I")
    assert (track.movement_number, track.movement_name) == (1, "Mvt. I Allegro con brio")
    assert (bernstein.movement_numbers, bernstein.complete) == ([1, 2, 3, 4], True)
    # the tracks of each movement are as many as the recordings covering it
    assert Counter(x.movement_number for r in recordings for x in r.tracks)[1] == len(
        [x for x in recordings if 1 in x.movement_numbers]
    )


GOLDBERG = [
    WorkMovement(number=1, name="Aria"),
    WorkMovement(number=2, name="Variatio 1. a 1 Clav."),
    WorkMovement(number=5, name="Variatio 4. a 1 Clav."),
    WorkMovement(number=32, name="Aria da capo"),
]


@pytest.mark.parametrize(
    ("movement_name", "name", "movements", "expected"),
    [
        ("Aria", "Aria", GOLDBERG, [1]),
        (None, "Goldberg Variations, BWV 988: Aria", GOLDBERG, [1]),
        ("Variation No. 1", "Variation No. 1", GOLDBERG, [2]),
        ("Var. 1", "Var. 1", GOLDBERG, [2]),
        ("Variation IV", "Variation IV", GOLDBERG, [5]),
        ("ARIA DA CAPO!", "Aria da capo", GOLDBERG, [32]),
        ("Variation No. 2", "Variation No. 2", GOLDBERG, []),
        # a name several movements go by matches none of them
        (
            "Allegro",
            "Allegro",
            [WorkMovement(number=1, name="Allegro"), WorkMovement(number=3, name="Allegro")],
            [],
        ),
        (
            "Allegro",
            "Allegro",
            [
                WorkMovement(number=1, name="I. Allegro con brio"),
                WorkMovement(number=4, name="IV. Allegro"),
            ],
            [],
        ),
    ],
)
def test_track_matches_a_movement_by_name(
    movement_name: str | None, name: str, movements: list[WorkMovement], expected: list[int]
) -> None:
    """A track without a movement id or number matches the one movement its name names."""
    track = Track(
        item_id="1",
        provider="library",
        name=name,
        provider_mappings=set(),
        movement_name=movement_name,
    )

    assert [x.number for x in _track_movements(track, _Movements(movements=movements), None)] == (
        expected
    )


@pytest.mark.parametrize(
    ("movement_name", "name", "expected"),
    [
        (None, "La bohème, Act I: Che gelida manina", [2]),
        ('"O soave fanciulla"', "O soave fanciulla", [3]),
        (None, "Act I: Mi chiamano Mimì", [4]),
        (None, "Act I: Questo Mar Rosso", []),
        # a name two arias go by matches none of them
        (None, "Act I: Sì, mi chiamano Mimì", []),
    ],
)
def test_track_matches_an_aria_by_its_quoted_name(
    movement_name: str | None, name: str, expected: list[int]
) -> None:
    """A track of an act matches the one aria of the act whose quoted name it is."""
    movements = [
        WorkMovement(number=2, name='No. 6. "Che gelida manina!" (Rodolfo)'),
        WorkMovement(number=3, name="No. 9. «O soave fanciulla»"),
        WorkMovement(number=4, name="No. 7. 'Mi chiamano Mimì'"),
        WorkMovement(number=5, name="“Sì, mi chiamano Mimì” (Mimì)"),
        WorkMovement(number=6, name="“Sì, mi chiamano Mimì” (reprise)"),
    ]
    track = Track(
        item_id="1",
        provider="library",
        name=name,
        provider_mappings=set(),
        movement_name=movement_name,
    )

    assert [x.number for x in _track_movements(track, _Movements(movements=movements), None)] == (
        expected
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


async def _set_movement_name(library: _Library, key: str, name: str | None) -> None:
    """Store the movement name of a seeded track."""
    await library.mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_TRACKS} SET movement_name = :name WHERE item_id = :track_id",
        {"name": name, "track_id": int(library.tracks[key].item_id)},
    )


async def _set_movement_number(library: _Library, key: str, number: int | None) -> None:
    """Store the movement number of a seeded track."""
    await library.mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_TRACKS} SET movement_number = :number WHERE item_id = :track_id",
        {"number": number, "track_id": int(library.tracks[key].item_id)},
    )


async def _set_musicbrainz_work(library: _Library, keys: Sequence[str], mbid: str | None) -> None:
    """Give seeded tracks a MusicBrainz work id, removing theirs when None."""
    database = library.mass.music.database
    for key in keys:
        lookup = {
            "media_type": MediaType.TRACK.value,
            "external_id_type": ExternalID.MB_WORK.value,
            "item_id": int(library.tracks[key].item_id),
        }
        if mbid:
            await database.insert(DB_TABLE_EXTERNAL_ID_LOOKUP, {**lookup, "external_id": mbid})
        else:
            await database.delete(DB_TABLE_EXTERNAL_ID_LOOKUP, lookup)


async def _set_version(library: _Library, key: str, version: str) -> None:
    """Store the version of a seeded work."""
    await library.mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_WORKS} SET version = :version WHERE item_id = :work_id",
        {"version": version, "work_id": int(library.work_id(key))},
    )


async def _set_musicbrainz_movements(
    library: _Library, key: str, movements: list[WorkMovement] | None
) -> None:
    """Store the MusicBrainz movements of a seeded work, removing them when None."""
    params: dict[str, int | str] = {"work_id": int(library.work_id(key))}
    if movements is None:
        query = "json_remove(metadata, '$.movements')"
    else:
        query = "json_set(metadata, '$.movements', json(:movements))"
        params["movements"] = json_dumps([x.to_dict() for x in movements])
    await library.mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_WORKS} SET metadata = {query} WHERE item_id = :work_id", params
    )


async def _set_work_musicbrainz_id(library: _Library, key: str, mbid: str | None) -> None:
    """Give a seeded work a MusicBrainz work id, removing its ids when None."""
    lookup = {
        "media_type": MediaType.WORK.value,
        "external_id_type": ExternalID.MB_WORK.value,
        "item_id": int(library.work_id(key)),
    }
    if mbid:
        await library.mass.music.database.insert(
            DB_TABLE_EXTERNAL_ID_LOOKUP, {**lookup, "external_id": mbid}
        )
    else:
        await library.mass.music.database.delete(DB_TABLE_EXTERNAL_ID_LOOKUP, lookup)


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
