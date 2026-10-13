"""Tests for the classical browse API of the music controller."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Any
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
    WorkSummary,
)

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_GENRES,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.classical import (
    _Appearance,
    _build_performance,
    _group_performances,
    _Performance,
    _performer_row,
    _refined_roles,
    _shown_credits,
    _title_work_type,
    _with_title_work_type,
    _WorkTrack,
)
from music_assistant.mass import MusicAssistant
from tests.common import set_music_source_access


def _artist_mapping(name: str) -> ItemMapping:
    """Create a library artist mapping."""
    return ItemMapping(media_type=MediaType.ARTIST, item_id=name, provider="library", name=name)


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
    assert [x.artist.period for x in rows] == [Period.CLASSICAL, None]


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
    assert [x.period for x in rows] == [Period.CLASSICAL] * 3 + [None] * 2
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
                classical_tag=True,
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

    async def test_composer_counts_a_work_with_parts_once(self, parts_library: _Library) -> None:
        """A composer's work count holds the opera once, its recordings those of the arias."""
        [row] = await parts_library.mass.music.classical.composers(
            artist_id=parts_library.artist_id("delibes")
        )

        assert (row.work_count, row.recording_count) == (1, 4)


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
                classical_tag=True,
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
        # two recordings by the same conductor on one album are told apart by their years
        (
            {"performers": {10}, "leaders": {10}, "year": 2002, "album_id": 5},
            [{"recording_year": 1955}, {"recording_year": 1981}],
            [([1], 1955, {10}), ([2], 1981, {10})],
        ),
        # a performance takes the year it was recorded, its album's year only without one
        (
            {"year": 1990},
            [
                {"performers": {10}, "leaders": {10}, "recording_year": 1962},
                {"performers": {10}, "leaders": {10}, "recording_year": 1962, "movement": 2},
                {"performers": {11}, "leaders": {11}},
            ],
            [([1, 2], 1962, {10}), ([3], 1990, {11})],
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
    recording year tells performances apart and dates them, else the year of their album.

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


async def test_recordings_read_the_recording_year_of_their_tracks(library: _Library) -> None:
    """A recording is dated by the recording year stored on its tracks."""
    karajan_tracks = [x for x in library.tracks.values() if x.name.startswith("Karajan")]
    database = library.mass.music.database
    for track in karajan_tracks:
        await database.execute_write(
            "UPDATE tracks SET metadata = json_set(metadata, '$.recording_year', 1961) "
            "WHERE item_id = :item_id",
            {"item_id": int(track.item_id)},
        )
    try:
        recordings = await library.mass.music.classical.recordings(library.work_id("Fifth"))
    finally:
        for track in karajan_tracks:
            await database.execute_write(
                "UPDATE tracks SET metadata = json_remove(metadata, '$.recording_year') "
                "WHERE item_id = :item_id",
                {"item_id": int(track.item_id)},
            )

    assert [(x.year, _track_names(x)[0]) for x in recordings] == [
        (1961, "Karajan I"),
        (1979, "Bernstein I"),
    ]


def _work_track(
    track_id: int,
    performers: set[int],
    year: int | None,
    leaders: set[int] | None = None,
    movement: int = 1,
    recording_year: int | None = None,
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
    :param recording_year: The year the track was recorded.
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
        recording_year=recording_year,
        performers=performers,
        leaders=leaders or set(),
        directors=directors or set(),
        appearances=appearances,
    )


def _performances_of(tracks: Sequence[_WorkTrack]) -> list[_Performance]:
    """Return the performances the tracks of a work group into."""
    return [_build_performance(1, key, group) for key, group in _group_performances(tracks).items()]


async def _set_version(library: _Library, key: str, version: str) -> None:
    """Store the version of a seeded work."""
    await library.mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_WORKS} SET version = :version WHERE item_id = :work_id",
        {"version": version, "work_id": int(library.work_id(key))},
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
            classical_tag=True,
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
            classical_tag=True,
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
                classical_tag=album is None,
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
