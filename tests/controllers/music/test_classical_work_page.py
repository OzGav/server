"""Tests for what a work's page shows of each recording of the work."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pytest
from music_assistant_models.enums import (
    MediaType,
    RecordingHeadingKind,
    WorkType,
)
from music_assistant_models.media_items import (
    ItemMapping,
    Recording,
)

from music_assistant.controllers.music.classical import _add_part_headings
from tests.controllers.music.test_classical_api import (
    _add_track,
    _add_work,
    _Library,
    _new_library,
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


class TestPartHeadings:
    """The recordings of a work's page show under the work, each part, or their own work."""

    async def test_recordings_show_under_their_work_or_part(
        self, parts_page_library: _Library
    ) -> None:
        """A scene's recording shows under its act, an aria of the scene under its own work."""
        lib = parts_page_library
        recordings = await lib.mass.music.classical.recordings(lib.work_id("Opera"))

        assert _named_headings(lib, recordings) == [
            ("Opera", ["levine"]),
            ("Act 1", ["abbado", "solti"]),
            ("Act 2", ["karajan"]),
            ("Aria", ["chailly"]),
        ]

    async def test_recordings_of_a_performer_have_no_heading(
        self, parts_page_library: _Library
    ) -> None:
        """Recordings asked for by performer carry no heading."""
        lib = parts_page_library
        recordings = await lib.mass.music.classical.recordings(performer_id=lib.artist_id("abbado"))

        assert recordings
        assert all(x.heading is None for x in recordings)


def _work(item_id: str) -> ItemMapping:
    """Return a work mapping named after its id."""
    return ItemMapping(item_id=item_id, provider="library", name=item_id, media_type=MediaType.WORK)


def _recording(
    key: str,
    work: str,
    parent: str | None = None,
    year: int | None = None,
) -> Recording:
    """
    Return a recording of a work.

    :param key: The key of the recording.
    :param work: The id of its work.
    :param parent: The id of the work its work is a part of.
    :param year: The year it was recorded.
    """
    return Recording(
        key=key,
        work=_work(work),
        parent_work=_work(parent) if parent else None,
        year=year,
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
