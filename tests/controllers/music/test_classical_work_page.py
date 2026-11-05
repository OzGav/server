"""Tests for what a work's page shows of each recording of the work."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from music_assistant_models.enums import (
    ExternalID,
    MediaType,
    RecordingCoverageKind,
    RecordingHeadingKind,
    WorkType,
)
from music_assistant_models.media_items import (
    ItemMapping,
    Recording,
    RecordingCoverage,
    Track,
    WorkMovement,
)

from music_assistant.constants import DB_TABLE_WORKS
from music_assistant.controllers.music.classical import _add_part_headings, _WorkPage
from music_assistant.helpers.json import json_loads
from tests.controllers.music.test_classical_api import (
    _add_track,
    _add_work,
    _Library,
    _new_library,
    _set_musicbrainz_movements,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from music_assistant.mass import MusicAssistant


PARTS = [
    ItemMapping(item_id=x, provider="library", name=x, media_type=MediaType.WORK)
    for x in ("act1", "act2")
]


def test_part_headings_list_the_work_first_then_each_part() -> None:
    """The work's own recordings come first, then those of each part in the given order."""
    recordings = [
        _recording("act2", "act2"),
        _recording("act1", "act1"),
        _recording("own", "opera"),
    ]

    _add_part_headings(recordings, _work("opera"), PARTS)

    assert _headings(recordings) == [
        ("opera", ["own"]),
        ("act1", ["act1"]),
        ("act2", ["act2"]),
    ]
    assert all(x.heading and x.heading.kind == RecordingHeadingKind.WORK for x in recordings)
    heading = recordings[1].heading
    assert heading
    assert heading.work == PARTS[0]


def test_part_headings_file_a_part_of_a_listed_part_under_it() -> None:
    """A recording of an aria of a listed act shows under that act."""
    aria = _recording("aria", "aria", parent="act1")

    _add_part_headings([aria], _work("opera"), PARTS)

    assert _headings([aria]) == [("act1", ["aria"])]


def test_part_headings_put_a_work_the_page_does_not_list_last() -> None:
    """A recording of a work that is no listed part shows under its own heading, after them."""
    recordings = [_recording("b", "other"), _recording("a", "act1")]

    _add_part_headings(recordings, _work("opera"), PARTS)

    assert _headings(recordings) == [("act1", ["a"]), ("other", ["b"])]


def test_part_headings_keep_the_order_of_the_recordings() -> None:
    """Under a heading the recordings keep the order they are listed in, whatever their year."""
    recordings = [
        _recording("late", "act1", year=2000),
        _recording("undated", "act1"),
        _recording("early", "act1", year=1960),
    ]

    _add_part_headings(recordings, _work("opera"), PARTS)

    assert _headings(recordings) == [("act1", ["late", "undated", "early"])]


@pytest.fixture(scope="class", name="parts_page_library")
async def parts_page_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera with two acts, a scene and an aria of a scene."""
    lib = await _new_library(
        music_mass_class, ("verdi", "levine", "abbado", "karajan", "solti", "chailly")
    )
    await _add_work(lib, "Opera", "verdi", work_type=WorkType.OPERA)
    await _add_work(lib, "Act 1", "verdi", parent="Opera")
    await _add_work(lib, "Act 2", "verdi", parent="Opera")
    await _add_work(lib, "Scene", "verdi", parent="Act 1")
    await _add_work(lib, "Aria", "verdi", parent="Scene")
    for work, conductor in (
        ("Opera", "levine"),
        ("Act 1", "abbado"),
        ("Act 2", "karajan"),
        ("Scene", "solti"),
        ("Aria", "chailly"),
    ):
        await _add_track(lib, conductor, work, conductor)
    return lib


class TestPartsAsMovementHeadings:
    """A work listing its parts as movements groups its recordings by its top-level parts."""

    async def test_opera_lists_its_scene_and_aria_under_their_act(
        self, parts_page_library: _Library
    ) -> None:
        """The opera's page marks the scene and the aria of the first act as entries under it."""
        lib = parts_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Opera"))

        assert [(x.name, x.under_part) for x in work.movements] == [
            ("Act 1", False),
            ("Scene", True),
            ("Aria", True),
            ("Act 2", False),
        ]

    async def test_recordings_show_under_their_act(self, parts_page_library: _Library) -> None:
        """The opera's own recording is complete, the scene and the aria follow their act."""
        lib = parts_page_library
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Opera"))

        assert _named_headings(lib, recordings) == [
            ("complete", ["levine"]),
            ("movement-1", ["abbado", "solti", "chailly"]),
            ("movement-4", ["karajan"]),
        ]
        by_name = {_name(lib, x): x for x in recordings}
        assert [by_name[x].part for x in ("levine", "abbado", "solti", "chailly", "karajan")] == [
            None,
            1,
            1,
            None,
            4,
        ]
        # the aria is no movement the page lists, so it counts as one of the two acts
        assert by_name["chailly"].coverage == RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENTS, held=1, total=2
        )
        assert by_name["abbado"].coverage == RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENT, movement=1
        )

    async def test_act_page_keeps_its_scene_and_aria_as_movements(
        self, parts_page_library: _Library
    ) -> None:
        """An act's own page marks nothing as under a part and groups by its own movements."""
        lib = parts_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Act 1"))
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Act 1"))

        assert [(x.name, x.under_part) for x in work.movements] == [
            ("Scene", False),
            ("Aria", False),
        ]
        assert _named_headings(lib, recordings) == [
            ("complete", ["abbado"]),
            ("movement-1", ["solti", "chailly"]),
        ]
        # the aria is held whole, and is of the part it sits under
        aria = next(x for x in recordings if _name(lib, x) == "chailly")
        assert aria.coverage == RecordingCoverage(kind=RecordingCoverageKind.MOVEMENT, movement=2)
        assert aria.part == 1

    async def test_recordings_of_a_performer_have_no_page_fields(
        self, parts_page_library: _Library
    ) -> None:
        """Recordings asked for by performer carry none of what a work's page shows."""
        lib = parts_page_library
        recordings = await lib.mass.music.classical.recordings(performer_id=lib.artist_id("solti"))

        assert recordings
        assert all(
            (x.heading, x.part, x.coverage, x.track_movements, x.track_names)
            == (None, None, None, {}, {})
            for x in recordings
        )


# an act (1) with two arias (2, 3), and a second act (4)
OPERA = [
    WorkMovement(number=1, name="Act 1"),
    WorkMovement(number=2, name="Aria 1", parent=1),
    WorkMovement(number=3, name="Aria 2", parent=1),
    WorkMovement(number=4, name="Act 2"),
]

# two acts that are works of their own (1, 4) with two numbers under the first (2, 3), a piece of
# the first number (6), and a prologue that is no work (5)
ACTS = [
    WorkMovement(number=1, name="Act 1", work_id="act1"),
    WorkMovement(number=2, name="No. 1", parent=1, mbid="mbid-2", work_id="no1"),
    WorkMovement(number=6, name="No. 1a", parent=2, mbid="mbid-6"),
    WorkMovement(number=3, name="No. 2", parent=1, mbid="mbid-3"),
    WorkMovement(number=4, name="Act 2", work_id="act2"),
    WorkMovement(number=5, name="Prologue"),
]


def test_lineage_lists_each_movement_and_those_above_it() -> None:
    """Each movement number is followed by those of the movements it sits under."""
    page = _WorkPage([*OPERA, WorkMovement(number=5, name="Recitative", parent=2)], is_part=False)

    assert page.lineage[1] == [1]
    assert page.lineage[3] == [3, 1]
    assert page.lineage[5] == [5, 2, 1]


@pytest.mark.parametrize(("is_part", "expected"), [(False, {2, 3, 6}), (True, set())])
def test_entries_under_parts_are_marked(is_part: bool, expected: set[int]) -> None:
    """The entries under the parts are marked, except on the page of a part of another work."""
    movements = _WorkPage(ACTS, is_part=is_part).listed_movements()

    assert {x.number for x in movements if x.under_part} == expected
    assert [x.number for x in movements] == [x.number for x in ACTS]


def test_track_movements_bring_an_acts_arias_along() -> None:
    """A track counts towards its own movement and every movement above it."""
    aria, other = _track("a", movement=2), _track("b", movement=4)
    recording = _recording("r", "w", tracks=[aria, other])

    _WorkPage(OPERA, is_part=False).describe(recording)

    assert recording.track_movements == {aria.uri: [2, 1], other.uri: [4]}


def test_track_movements_fall_back_to_the_musicbrainz_work_of_an_entry() -> None:
    """
    A track of an entry right under a part counts by its MusicBrainz work when none is numbered.

    The fallback is not used for an entry a track is numbered as, nor for a piece of an entry.
    """
    linked = _track("linked", mbids=("mbid-2", "mbid-6"))
    numbered = _track("numbered", movement=3)
    also = _track("also", mbids=("mbid-3",))
    recording = _recording("r", "act1", tracks=[linked, numbered, also])

    _WorkPage(ACTS, is_part=False).describe(recording)

    assert recording.track_movements == {linked.uri: [2], numbered.uri: [3, 1]}


def test_track_movements_do_not_fall_back_on_the_page_of_a_part() -> None:
    """On the page of a part of another work there are no entries under parts to fall back for."""
    linked = _track("linked", mbids=("mbid-2",))
    recording = _recording("r", "act1", tracks=[linked])

    _WorkPage(ACTS, is_part=True).describe(recording)

    assert recording.track_movements == {}


@pytest.mark.parametrize(
    ("name", "short_name"),
    [
        ('XXXIX. Aria (Alto) "Erbarme dich, mein Gott"', "XXXIX. Erbarme dich, mein Gott"),
        (
            'IVa. "Da versammelten sich die Hohenpriester" (Evangelista)',
            "IVa. Da versammelten sich die Hohenpriester",
        ),
        ('VI. Arie (Alt) "Buß und Reu"', "VI. Buß und Reu"),
        ('36a. Chor "Weissage uns"', "36a. Weissage uns"),
        ("no. 6. “Che gelida manina!” (Rodolfo)", "no. 6. Che gelida manina!"),
        ('Scene V. "Sankt Krispin, lobet ihn!"', "Scene V. Sankt Krispin, lobet ihn!"),
        ('Scene I "Hojotoho! Hojotoho!"', "Scene I. Hojotoho! Hojotoho!"),
        (
            'Scene II "Wo ist Brünnhild\', wo die Verbrecherin?"',  # codespell:ignore
            "Scene II. Wo ist Brünnhild', wo die Verbrecherin?",  # codespell:ignore
        ),
        ("“E lucevan le stelle” (Cavaradossi)", "E lucevan le stelle"),
        # a word in front is no score number
        ('Il "Lacrimosa"', "Lacrimosa"),
        (
            "LX. Aria (Alto) “Sehet, Jesus hat die Hand”, Coro “Wohin?”",
            "LX. Sehet, Jesus hat die Hand \u2013 Wohin?",
        ),
        (
            "LXVII. Recitativo (Soprano, Alto, Tenore, Basso) "
            "“Nun ist der Herr zur Ruh gebracht” "  # codespell:ignore
            "\u2013 Coro “Mein Jesu, gute Nacht”",
            "LXVII. Nun ist der Herr zur Ruh gebracht "  # codespell:ignore
            "\u2013 Mein Jesu, gute Nacht",
        ),
        ("no. 3. “Si può?” \u2013 “Chi è là?” (Rodolfo)", "no. 3. Si può? \u2013 Chi è là?"),
        ("« Quel trouble » … Cavatine « Salut ! »", "Quel trouble \u2013 Salut !"),
        # a name without quotes stays whole
        ("no. 6. Che gelida manina", "no. 6. Che gelida manina"),
        ("IV", "IV"),
        ("Scene II", "Scene II"),
    ],
)
def test_movements_have_short_names(name: str, short_name: str) -> None:
    """A movement is named by its quoted words, led by its score number or scene, else whole."""
    movements = _WorkPage([WorkMovement(number=1, name=name)], is_part=False).listed_movements()

    assert (movements[0].short_name, movements[0].excerpt) == (short_name, False)


@pytest.mark.parametrize(
    ("name", "short_name"),
    [
        ('"Morgenlich leuchtend" (Preislied) (excerpt)', "Morgenlich leuchtend"),
        ('"Wacht auf! es nahet gen den Tag" (Volk) (excerpt)', "Wacht auf! es nahet gen den Tag"),
        # a name without quotes stays whole, without its excerpt ending
        ("Tanz der Lehrbuben (excerpt)", "Tanz der Lehrbuben"),
        ("Walkürenritt (Excerpt)", "Walkürenritt"),
    ],
)
def test_excerpts_are_marked(name: str, short_name: str) -> None:
    """A movement whose name ends in " (excerpt)", in any case, is an excerpt."""
    movements = _WorkPage([WorkMovement(number=1, name=name)], is_part=False).listed_movements()

    assert (movements[0].short_name, movements[0].excerpt) == (short_name, True)


@pytest.mark.parametrize(
    ("name", "movement_name", "mbids", "expected"),
    [
        # its movement name
        ("Act 1", "Che gelida manina!", (), "Che gelida manina!"),
        # its own name without the work name in front, when its movement name is a work's
        ("Act 1: No. 2. Duet", "ACT 1", (), "No. 2. Duet"),
        ("opera \u2013 Finale", None, (), "Finale"),
        ("Duet", None, ("mbid-3",), "Duet"),
        # the entries of its part its MusicBrainz work ids are, in page order
        ("Act 1", None, ("mbid-3",), "No. 2"),
        ("Act 1 -", None, ("mbid-6", "mbid-2"), "No. 1, No. 1a"),
        # else its own name
        ("Act 1: act 2", None, (), "Act 1: act 2"),
        ("Act 1", None, ("other",), "Act 1"),
        ("Act 1", None, ("MBID-3",), "Act 1"),
    ],
)
def test_track_names(
    name: str, movement_name: str | None, mbids: tuple[str, ...], expected: str
) -> None:
    """
    A track shows its movement name, else its own name without a work name in front.

    Else it shows the names of the entries under its part its MusicBrainz work ids are, else its
    own name. It never shows just the name of its work, of that work's parent or of an act.
    """
    track = _track("t", mbids=mbids, name=name, movement_name=movement_name)
    recording = _recording("r", "act1", parent="opera", tracks=[track])

    _WorkPage(ACTS, is_part=False).describe(recording)

    assert recording.track_names == {track.uri: expected}


@pytest.mark.parametrize("work", ["act2", "x"])
def test_track_names_take_only_the_entries_of_the_recordings_part(work: str) -> None:
    """A track is not named after an entry of another part, nor of any part without one."""
    track = _track("t", mbids=("mbid-3",), name="Act 2")
    recording = _recording("r", work, tracks=[track])

    _WorkPage(ACTS, is_part=False).describe(recording)

    assert recording.track_names == {track.uri: "Act 2"}


@pytest.mark.parametrize(("is_part", "expected"), [(False, "No. 1"), (True, "No. 2")])
def test_track_names_on_the_page_of_a_part(is_part: bool, expected: str) -> None:
    """
    On the page of a part, such as an act's own, every movement that is a work is a work name.

    The entries under the recording's part still name its tracks.
    """
    track = _track("t", mbids=("mbid-3",), name="Act 1", movement_name="No. 1")
    recording = _recording("r", "act1", tracks=[track])

    _WorkPage(ACTS, is_part=is_part).describe(recording)

    assert recording.track_names == {track.uri: expected}


def test_movement_headings_follow_the_movements() -> None:
    """Recordings show under their top-level movement in movement order, complete ones first."""
    recordings = [
        _recording("stray", "x"),
        _recording("act2", "w", numbers=[4]),
        _recording("aria", "w", numbers=[3]),
        _recording("complete", "w"),
    ]

    _WorkPage(OPERA, is_part=False).add_headings(recordings, "w")

    assert _headings(recordings) == [
        ("complete", ["complete"]),
        ("movement-1", ["aria"]),
        ("movement-4", ["act2"]),
        ("other", ["stray"]),
    ]


def test_movement_headings_of_recordings_reaching_into_several_movements() -> None:
    """
    A recording of the work itself reaching into every top-level movement is complete.

    One reaching into some but not all of them fits nowhere, one holding numbers of a single
    top-level movement stays under it, as does a recording matched to a movement through its
    work, whatever numbers it holds.
    """
    recordings = [
        _recording("full", "w", numbers=[1, 4, 5]),
        _recording("two acts", "w", numbers=[2, 4]),
        _recording("act part", "w", numbers=[2, 3]),
        _recording("aria", "aria", parent="act1", numbers=[2, 4]),
        _recording("bare", "w"),
    ]

    _WorkPage(ACTS, is_part=False).add_headings(recordings, "w")

    assert _headings(recordings) == [
        ("complete", ["full", "bare"]),
        ("movement-1", ["act part", "aria"]),
        ("other", ["two acts"]),
    ]


def test_movement_headings_list_a_whole_movement_first_then_by_year() -> None:
    """Under a movement the recording of all of it comes first, then by year, undated last."""
    recordings = [
        _recording("later", "w", numbers=[2], year=1990),
        _recording("undated", "w", numbers=[2]),
        _recording("whole", "w", numbers=[1, 2, 3], year=2000),
        _recording("earlier", "w", numbers=[2], year=1960),
    ]

    _WorkPage(OPERA, is_part=False).add_headings(recordings, "w")

    assert _headings(recordings) == [("movement-1", ["whole", "earlier", "later", "undated"])]


def test_movement_headings_rank_by_the_first_movement_held() -> None:
    """A recording of a later number of a movement comes after one of an earlier number."""
    recordings = [
        _recording("second", "w", numbers=[3], year=1950),
        _recording("first", "w", numbers=[2], year=2000),
    ]

    _WorkPage(OPERA, is_part=False).add_headings(recordings, "w")

    assert _headings(recordings) == [("movement-1", ["first", "second"])]


def test_movement_headings_find_a_part_by_its_work() -> None:
    """A recording of a part, or of a part of a part, shows under that part's movement."""
    recordings = [
        _recording("act2", "act2"),
        _recording("number", "no1", parent="act1"),
        _recording("prologue", "w", numbers=[5]),
        # an entry under a part is no heading of its own
        _recording("loose", "no1"),
    ]

    _WorkPage(ACTS, is_part=False).add_headings(recordings, "w")

    assert _headings(recordings) == [
        ("movement-1", ["number"]),
        ("movement-4", ["act2"]),
        ("movement-5", ["prologue"]),
        ("other", ["loose"]),
    ]


@pytest.mark.parametrize(
    ("numbers", "work", "expected"),
    [
        ([1], "w", 1),
        ([1, 4], "act2", 4),
        ([], "no1", 1),
        ([5], "w", None),
    ],
)
def test_part_of_a_recording(numbers: list[int], work: str, expected: int | None) -> None:
    """A recording is of the one part it covers, else the part that is its work or parent work."""
    recording = _recording("r", work, parent="act1" if work == "no1" else None, numbers=numbers)

    _WorkPage(ACTS, is_part=False).describe(recording)

    assert recording.part == expected


@pytest.mark.parametrize(
    ("movements", "numbers", "work", "pieces", "expected"),
    [
        # a number under a covered movement adds nothing to it
        (
            OPERA,
            [1, 2],
            "w",
            0,
            RecordingCoverage(kind=RecordingCoverageKind.MOVEMENTS, held=1, total=2),
        ),
        (
            ACTS,
            [1, 4],
            "w",
            0,
            RecordingCoverage(kind=RecordingCoverageKind.MOVEMENTS, held=2, total=3),
        ),
        (ACTS, [4], "act2", 0, RecordingCoverage(kind=RecordingCoverageKind.MOVEMENT, movement=4)),
        (
            ACTS,
            [1],
            "act1",
            1,
            RecordingCoverage(
                kind=RecordingCoverageKind.MOVEMENT_PIECES, movement=1, held=1, total=2
            ),
        ),
        # a movement without entries under it counts none
        (
            ACTS,
            [5],
            "act1",
            2,
            RecordingCoverage(
                kind=RecordingCoverageKind.MOVEMENT_PIECES, movement=5, held=2, total=0
            ),
        ),
        (OPERA, [], "w", 0, None),
    ],
)
def test_coverage_of_a_recording(
    movements: list[WorkMovement],
    numbers: list[int],
    work: str,
    pieces: int,
    expected: RecordingCoverage | None,
) -> None:
    """A recording holds a whole movement, some pieces of one, or some of the movements."""
    recording = _recording(
        "r",
        work,
        numbers=numbers,
        part_movements=[WorkMovement(number=x, name=f"No. {x}") for x in range(1, pieces + 1)],
    )

    _WorkPage(movements, is_part=False).describe(recording)

    assert recording.coverage == expected


@pytest.mark.parametrize(("work", "total"), [("act1", 1), ("act2", 0)])
def test_coverage_counts_the_pieces_listed_under_the_part_of_the_recording(
    work: str, total: int
) -> None:
    """The pieces of a number count only when listed under the part the recording is of."""
    recording = _recording(
        "r", work, numbers=[2], part_movements=[WorkMovement(number=1, name="No. 1a")]
    )

    _WorkPage(ACTS, is_part=True).describe(recording)

    assert recording.coverage == RecordingCoverage(
        kind=RecordingCoverageKind.MOVEMENT_PIECES, movement=2, held=1, total=total
    )


def test_complete_recording_has_no_coverage() -> None:
    """A complete recording holds the whole work."""
    recording = _recording("r", "w", numbers=[4])
    recording.complete = True

    _WorkPage(OPERA, is_part=False).describe(recording)

    assert recording.coverage is None


@pytest.fixture(scope="class", name="musicbrainz_page_library")
async def musicbrainz_page_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera with one act and a symphony, from MusicBrainz."""
    lib = await _new_library(
        music_mass_class, ("verdi", "beethoven", "levine", "karajan", "abbado", "solti")
    )
    act_mbids = [str(uuid4()) for _ in range(2)]
    aria_mbids = [str(uuid4()) for _ in range(2)]
    await _add_work(lib, "Opera", "verdi", work_type=WorkType.OPERA)
    await _add_work(
        lib, "Act 1", "verdi", parent="Opera", work_type=WorkType.OPERA, mbid=act_mbids[0]
    )
    await _add_work(lib, "Symphony", "beethoven", work_type=WorkType.SYMPHONY)
    for key, work, conductor, mbids in (
        ("whole 1", "Opera", "levine", act_mbids[:1]),
        ("whole 2", "Opera", "levine", act_mbids[1:]),
        # one track holding both arias, numbered as only one of them
        ("both arias", "Act 1", "karajan", aria_mbids),
        ("second aria", "Act 1", "abbado", aria_mbids[1:]),
    ):
        await _add_track(lib, key, work, conductor)
        await lib.mass.music.tracks.set_external_ids(
            lib.tracks[key].item_id, {(ExternalID.MB_WORK, x) for x in mbids}
        )
    for number in (1, 2):
        await _add_track(lib, f"movement {number}", "Symphony", "solti", movement=number)
    await _set_musicbrainz_movements(
        lib,
        "Opera",
        [WorkMovement(number=x, name=f"Act {x}", mbid=y) for x, y in enumerate(act_mbids, 1)],
    )
    await _set_musicbrainz_movements(
        lib,
        "Act 1",
        [WorkMovement(number=x, name=f"Aria {x}", mbid=y) for x, y in enumerate(aria_mbids, 1)],
    )
    await _set_musicbrainz_movements(
        lib, "Symphony", [WorkMovement(number=x, name=f"Movement {x}") for x in range(1, 5)]
    )
    return lib


class TestMusicBrainzMovementsOnTheWorkPage:
    """A work's page with the movements MusicBrainz lists, an act among them."""

    async def test_opera_marks_the_arias_under_its_act(
        self, musicbrainz_page_library: _Library
    ) -> None:
        """The arias of the act the library holds are listed under it."""
        lib = musicbrainz_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Opera"))

        assert [(x.name, x.parent is not None, x.under_part) for x in work.movements] == [
            ("Act 1", False, False),
            ("Aria 1", True, True),
            ("Aria 2", True, True),
            ("Act 2", False, False),
        ]

    async def test_opera_recordings(self, musicbrainz_page_library: _Library) -> None:
        """
        Every recording shows under the act it holds.

        A track holding both arias counts towards the one it is not numbered as by its
        MusicBrainz work.
        """
        lib = musicbrainz_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Opera"))
        arias = [x.number for x in work.movements if x.under_part]
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Opera"))
        by_name = {_name(lib, x): x for x in recordings}

        assert _named_headings(lib, recordings) == [
            ("complete", ["whole 1"]),
            ("movement-1", ["second aria", "both arias"]),
        ]
        assert {x: y.part for x, y in by_name.items()} == {
            "whole 1": 1,
            "second aria": 1,
            "both arias": 1,
        }
        assert by_name["whole 1"].coverage is None
        assert by_name["both arias"].coverage == RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENT, movement=1
        )
        assert by_name["second aria"].coverage == RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENT_PIECES, movement=1, held=1, total=2
        )
        assert by_name["both arias"].track_movements == {
            lib.tracks["both arias"].uri: [arias[0], 1, arias[1]],
        }
        assert by_name["second aria"].track_movements == {
            lib.tracks["second aria"].uri: [arias[1], 1],
        }

    async def test_symphony_recordings(self, musicbrainz_page_library: _Library) -> None:
        """A work of plain movements groups by work and counts the movements a recording holds."""
        lib = musicbrainz_page_library
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Symphony"))

        assert _named_headings(lib, recordings) == [("Symphony", ["movement 1"])]
        assert recordings[0].part is None
        assert recordings[0].coverage == RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENTS, held=2, total=4
        )
        assert recordings[0].track_movements == {
            lib.tracks["movement 1"].uri: [1],
            lib.tracks["movement 2"].uri: [2],
        }

    async def test_act_page(self, musicbrainz_page_library: _Library) -> None:
        """An act's own page lists its arias as its movements, its recordings under the act."""
        lib = musicbrainz_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Act 1"))
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Act 1"))
        by_name = {_name(lib, x): x for x in recordings}

        assert [(x.name, x.under_part) for x in work.movements] == [
            ("Aria 1", False),
            ("Aria 2", False),
        ]
        assert _named_headings(lib, recordings) == [("Act 1", ["second aria", "both arias"])]
        assert by_name["both arias"].coverage is None
        assert by_name["second aria"].coverage == RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENTS, held=1, total=2
        )

    async def test_under_part_is_never_stored(self, musicbrainz_page_library: _Library) -> None:
        """Saving the work the page shows keeps its stored movements unmarked."""
        lib = musicbrainz_page_library
        work_id = lib.work_id("Opera")
        shown = await lib.mass.music.classical.work(work_id)
        assert any(x.under_part for x in shown.movements)

        await lib.mass.music.works.update_item_in_library(work_id, shown)

        stored = await lib.mass.music.works.get_library_item(work_id)
        assert stored.metadata.movements
        assert not any(x.under_part for x in stored.metadata.movements)
        rows = await lib.mass.music.database.get_rows_from_query(
            f"SELECT metadata FROM {DB_TABLE_WORKS}", limit=0
        )
        assert not [
            x
            for row in rows
            for x in json_loads(row["metadata"]).get("movements") or []
            if x.get("under_part")
        ]


@pytest.fixture(scope="class", name="names_page_library")
async def names_page_library_fixture(music_mass_class: MusicAssistant) -> _Library:
    """Return a class-scoped library of an opera and its act, their movements from MusicBrainz."""
    lib = await _new_library(music_mass_class, ("puccini", "karajan"))
    act_mbid = str(uuid4())
    aria_mbids = [str(uuid4()) for _ in range(3)]
    await _add_work(lib, "Opera", "puccini", work_type=WorkType.OPERA)
    await _add_work(
        lib, "Act 1", "puccini", parent="Opera", work_type=WorkType.OPERA, mbid=act_mbid
    )
    await _add_track(lib, "aria", "Act 1", "karajan", name="Act 1")
    await lib.mass.music.tracks.set_external_ids(
        lib.tracks["aria"].item_id, {(ExternalID.MB_WORK, aria_mbids[0])}
    )
    await _add_track(lib, "quartet", "Act 1", "karajan", name="Act 1: Quartet")
    await _set_musicbrainz_movements(
        lib, "Opera", [WorkMovement(number=1, name="Act 1", mbid=act_mbid)]
    )
    await _set_musicbrainz_movements(
        lib,
        "Act 1",
        [
            WorkMovement(
                number=1, name="no. 6. “Che gelida manina!” (Rodolfo)", mbid=aria_mbids[0]
            ),
            WorkMovement(
                number=2, name='"Morgenlich leuchtend" (Preislied) (excerpt)', mbid=aria_mbids[1]
            ),
            WorkMovement(number=3, name="Tanz der Lehrbuben (excerpt)", mbid=aria_mbids[2]),
        ],
    )
    return lib


class TestNamesOnTheWorkPage:
    """A work's page names its movements short and the tracks of its recordings."""

    async def test_opera_movements(self, names_page_library: _Library) -> None:
        """Each movement has its short name and is marked when an excerpt."""
        lib = names_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Opera"))

        assert [(x.name, x.short_name, x.excerpt) for x in work.movements] == [
            ("Act 1", "Act 1", False),
            (
                "no. 6. “Che gelida manina!” (Rodolfo)",
                "no. 6. Che gelida manina!",
                False,
            ),
            ('"Morgenlich leuchtend" (Preislied) (excerpt)', "Morgenlich leuchtend", True),
            ("Tanz der Lehrbuben (excerpt)", "Tanz der Lehrbuben", True),
        ]

    async def test_opera_recordings(self, names_page_library: _Library) -> None:
        """A track named after its act shows the aria of the act its MusicBrainz work id is."""
        lib = names_page_library
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Opera"))

        assert [x.track_names for x in recordings] == [
            {
                lib.tracks["aria"].uri: "no. 6. “Che gelida manina!” (Rodolfo)",
                lib.tracks["quartet"].uri: "Quartet",
            }
        ]

    async def test_act_page(self, names_page_library: _Library) -> None:
        """
        An act's own page names its arias short and the tracks of its recordings.

        A track named after the act shows its movement on this page, the aria.
        """
        lib = names_page_library
        work = await lib.mass.music.classical.work(lib.work_id("Act 1"))
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Act 1"))

        assert [x.short_name for x in work.movements] == [
            "no. 6. Che gelida manina!",
            "Morgenlich leuchtend",
            "Tanz der Lehrbuben",
        ]
        assert [x.track_names for x in recordings] == [
            {
                lib.tracks["aria"].uri: "“Che gelida manina!” (Rodolfo)",
                lib.tracks["quartet"].uri: "Quartet",
            }
        ]

    async def test_short_names_are_never_stored(self, names_page_library: _Library) -> None:
        """Saving the work the page shows stores its movements without short names or marks."""
        lib = names_page_library
        work_id = lib.work_id("Opera")
        shown = await lib.mass.music.classical.work(work_id)
        assert any(x.excerpt for x in shown.movements)

        await lib.mass.music.works.update_item_in_library(work_id, shown)

        stored = await lib.mass.music.works.get_library_item(work_id)
        assert stored.metadata.movements
        assert all((x.short_name, x.excerpt) == (None, False) for x in stored.metadata.movements)
        rows = await lib.mass.music.database.get_rows_from_query(
            f"SELECT metadata FROM {DB_TABLE_WORKS}", limit=0
        )
        assert not [
            x
            for row in rows
            for x in json_loads(row["metadata"]).get("movements") or []
            if x.get("short_name") or x.get("excerpt")
        ]


def _work(item_id: str) -> ItemMapping:
    """Return a work mapping named after its id."""
    return ItemMapping(item_id=item_id, provider="library", name=item_id, media_type=MediaType.WORK)


def _recording(
    key: str,
    work: str,
    parent: str | None = None,
    year: int | None = None,
    numbers: Sequence[int] = (),
    part_movements: Sequence[WorkMovement] = (),
    tracks: Sequence[Track] = (),
) -> Recording:
    """
    Return a recording of a work.

    :param key: The key of the recording.
    :param work: The id of its work.
    :param parent: The id of the work its work is a part of.
    :param year: The year it was recorded.
    :param numbers: The movement numbers it covers.
    :param part_movements: The movements of its part it covers.
    :param tracks: Its tracks.
    """
    return Recording(
        key=key,
        work=_work(work),
        parent_work=_work(parent) if parent else None,
        year=year,
        movement_numbers=list(numbers),
        part_movements=list(part_movements),
        tracks=list(tracks),
    )


def _track(
    item_id: str,
    movement: int | None = None,
    mbids: Sequence[str] = (),
    name: str | None = None,
    movement_name: str | None = None,
) -> Track:
    """Return a library track with a movement and MusicBrainz work ids, named after its id."""
    return Track(
        item_id=item_id,
        provider="library",
        name=name or item_id,
        provider_mappings=set(),
        movement_number=movement,
        movement_name=movement_name,
        external_ids={(ExternalID.MB_WORK, x) for x in mbids},
    )


def _headings(recordings: Sequence[Recording]) -> list[tuple[str, list[str]]]:
    """Return the headings in page order, each with the keys of its recordings in order."""
    by_order: dict[int, list[Recording]] = defaultdict(list)
    for recording in recordings:
        if recording.heading:
            by_order[recording.heading.order].append(recording)
    return [
        (
            _heading_key(listed[0]),
            [x.key for x in sorted(listed, key=lambda x: x.heading.rank if x.heading else 0)],
        )
        for _, listed in sorted(by_order.items())
    ]


def _heading_key(recording: Recording) -> str:
    """Return the key of the heading a recording shows under, as the page keys it."""
    assert recording.heading
    if recording.heading.kind == RecordingHeadingKind.WORK:
        assert recording.heading.work
        return recording.heading.work.item_id
    if recording.heading.kind == RecordingHeadingKind.MOVEMENT:
        return f"movement-{recording.heading.movement}"
    return recording.heading.kind.value


def _name(lib: _Library, recording: Recording) -> str:
    """Return the key of the seeded track a recording starts with."""
    track_ids = {x.item_id: key for key, x in lib.tracks.items()}
    return track_ids[recording.tracks[0].item_id]


def _named_headings(lib: _Library, recordings: Sequence[Recording]) -> list[tuple[str, list[str]]]:
    """Return the headings in page order, a work by its key, each with its recordings by name."""
    keys = {x.item_id: key for key, x in lib.works.items()}
    names = {x.key: _name(lib, x) for x in recordings}
    return [
        (keys.get(key, key), [names[x] for x in listed]) for key, listed in _headings(recordings)
    ]
