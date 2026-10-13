"""Tests for the performing relations and recording years read off MusicBrainz recordings."""

from __future__ import annotations

import json
import pathlib
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ArtistRole
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.musicbrainz.models import (
    MusicBrainzRecording,
    MusicBrainzRelation,
    MusicBrainzRelease,
)
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "performers"


def _fixture(name: str) -> dict[str, Any]:
    """Return a real MusicBrainz lookup, trimmed to a few tracks."""
    return cast("dict[str, Any]", json.loads((FIXTURES / f"{name}.json").read_text()))


def _recordings(release: MusicBrainzRelease) -> dict[str, MusicBrainzRecording]:
    """Return the recordings of a release by their title."""
    return {
        track.recording.title: track.recording
        for medium in release.media
        for track in medium.tracks
        if track.recording
    }


def _relation(
    rel_type: str, artist_type: str | None = "Person", attributes: list[str] | None = None
) -> MusicBrainzRelation:
    """Build an artist relation of a recording."""
    return MusicBrainzRelation.from_dict(
        {
            "type": rel_type,
            "artist": {"id": "a", "name": "A", "sort_name": "A", "type": artist_type},
            "attributes": attributes or [],
        }
    )


def _provider(responses: dict[str, Any]) -> tuple[MusicbrainzProvider, AsyncMock]:
    """Return a MusicbrainzProvider answering the endpoints starting with the given keys."""
    with patch.object(MusicbrainzProvider, "__init__", lambda *_a, **_kw: None):
        provider = MusicbrainzProvider.__new__(MusicbrainzProvider)
    provider._logged_hints = set()

    async def _answer(endpoint: str, **_kwargs: Any) -> Any:
        return next((value for key, value in responses.items() if endpoint.startswith(key)), None)

    get_data = AsyncMock(side_effect=_answer)
    provider._api_client = MagicMock()
    provider._api_client.get_data = get_data
    return provider, get_data


# --------------------------------------------------------------------------- #
#  roles and instruments                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rel_type", "artist_type", "attributes", "role", "instrument"),
    [
        ("conductor", "Person", [], ArtistRole.CONDUCTOR, None),
        ("performing orchestra", "Orchestra", [], ArtistRole.ORCHESTRA, None),
        ("vocal", "Choir", ["choir vocals"], ArtistRole.CHOIR, None),
        ("vocal", "Person", ["solo", "tenor vocals"], ArtistRole.SOLOIST, "tenor vocals"),
        ("vocal", "Person", [], ArtistRole.SOLOIST, None),
        ("instrument", "Person", ["guest", "piano"], ArtistRole.SOLOIST, "piano"),
        ("instrument", None, ["violin", "viola"], ArtistRole.SOLOIST, "violin, viola"),
        ("instrument", "Group", ["cello", "additional"], ArtistRole.ENSEMBLE, "cello"),
        ("performer", "Orchestra", [], ArtistRole.ORCHESTRA, None),
        ("performer", "Choir", [], ArtistRole.CHOIR, None),
        ("performer", "Group", [], ArtistRole.ENSEMBLE, None),
        ("performer", "Person", [], ArtistRole.PERFORMER, None),
        ("chorus master", "Person", [], ArtistRole.PERFORMER, "chorus master"),
        ("concertmaster", "Person", [], ArtistRole.PERFORMER, "concertmaster"),
    ],
)
def test_performing_relation_maps_to_a_role_and_instrument(
    rel_type: str,
    artist_type: str | None,
    attributes: list[str],
    role: ArtistRole,
    instrument: str | None,
) -> None:
    """Each performing relation type gives the role and instrument a credit names it by."""
    relation = _relation(rel_type, artist_type, attributes)

    assert (relation.performing_role, relation.performing_instrument) == (role, instrument)


@pytest.mark.parametrize(
    "rel_type", ["composer", "lyricist", "arranger", "producer", "engineer", "mix", "recording"]
)
def test_other_relations_credit_no_performing_role(rel_type: str) -> None:
    """Writing, arranging and studio credits are no performing role."""
    assert _relation(rel_type).performing_role is None


def test_release_lookup_reads_the_performers_of_its_recordings() -> None:
    """A real release lists each recording's performers, without its studio credits."""
    release = MusicBrainzRelease.from_raw(_fixture("best_classical_album"))
    recording = _recordings(release)["Carmina Burana: Fortuna Imperatrix Mundi: O Fortuna"]

    assert [
        (x.artist.name if x.artist else None, x.performing_role, x.performing_instrument)
        for x in recording.performing_relations
    ] == [
        ("Arthur Oldham", ArtistRole.PERFORMER, "chorus master"),
        ("André Previn", ArtistRole.CONDUCTOR, None),
        ("London Symphony Orchestra", ArtistRole.ORCHESTRA, None),
        ("London Symphony Chorus", ArtistRole.CHOIR, None),
    ]


def test_release_lookup_lists_a_repeated_relation_once() -> None:
    """A relation MusicBrainz repeats for each session is listed once, other attributes apart."""
    release = MusicBrainzRelease.from_raw(_fixture("best_classical_album"))
    recording = _recordings(release)[
        "Concerto no. 4 in E-flat major for Horn, K. 495: III. Rondo. Allegro vivace"
    ]

    assert [(x.artist.name if x.artist else None) for x in recording.performing_relations] == [
        "Herbert von Karajan",
        "Dennis Brain",
        "Dennis Brain",
        "Philharmonia Orchestra",
    ]


def test_release_lookup_keeps_the_name_a_relation_credits() -> None:
    """The name MusicBrainz credits an artist as on a relation is kept beside its own name."""
    release = MusicBrainzRelease.from_raw(_fixture("best_classical_album"))
    recording = _recordings(release)["Schindler\u2019s List Theme"]

    orchestra = next(
        x for x in recording.performing_relations if x.performing_role == ArtistRole.ORCHESTRA
    )
    assert orchestra.artist is not None
    assert (orchestra.artist.name, orchestra.target_credit) == (
        "The New World Philharmonic",
        "New World Philharmonic",
    )


# --------------------------------------------------------------------------- #
#  recording years                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("title", "year"),
    [
        # one session date for every performer, the place and the work
        ("Carmina Burana: Fortuna Imperatrix Mundi: O Fortuna", 1974),
        # a range of dates
        ("Concerto no. 1 in E, RV 269 “La primavera”: I. Allegro", 1986),
        # full dates and a bare year of the same session, repeated for its performers
        ("Concerto no. 4 in E-flat major for Horn, K. 495: III. Rondo. Allegro vivace", 1953),
        # only a year
        ("Schindler\u2019s List Theme", 1994),
        # a conductor dated while the orchestra is not
        ("Flower Duet from Lakmé", 1970),
        # no dates at all
        ("O mio babbino caro (Gianni Schicchi)", None),
        ("Piano Concerto No. 1 in B-flat minor, Op. 23", None),
    ],
)
def test_recording_year_is_the_earliest_session(title: str, year: int | None) -> None:
    """A recording is dated by its earliest session of its performers, work or place."""
    release = MusicBrainzRelease.from_raw(_fixture("best_classical_album"))

    assert _recordings(release)[title].recording_year == year


def test_recording_year_reads_a_place_and_a_work_performance() -> None:
    """The place a recording was made at and the work it performs date it too."""
    recording = MusicBrainzRecording.from_dict(
        {
            "id": "rec",
            "title": "Movement",
            "relations": [
                {"type": "recorded at", "begin": "1962-05"},
                {"type": "performance", "begin": "1963"},
                {
                    "type": "producer",
                    "begin": "1950",
                    "artist": {"id": "a", "name": "A", "sort_name": "A"},
                },
            ],
        }
    )

    assert recording.recording_year == 1962


# --------------------------------------------------------------------------- #
#  lookups                                                                     #
# --------------------------------------------------------------------------- #


async def test_release_details_request_the_relations_of_its_recordings() -> None:
    """One release lookup carries the performers, works and places of every recording."""
    provider, get_data = _provider({"release/": _fixture("best_classical_album")})

    release = await provider.get_release_details("rel-1", with_performers=True)

    get_data.assert_awaited_once_with(
        "release/rel-1?inc=artist-credits+aliases+labels+release-groups+recordings+isrcs"
        "+url-rels+genres+artist-rels+recording-level-rels+work-rels+work-level-rels+place-rels"
    )
    assert sum(len(x.performing_relations) for x in _recordings(release).values()) == 25


async def test_recording_relations_request_its_performers_works_and_places() -> None:
    """A recording on its own is looked up with its performers, works and places."""
    provider, get_data = _provider({"recording/": _fixture("recording_o_mio_babbino_caro")})

    recording = await provider.get_recording_relations("rec-1")

    get_data.assert_awaited_once_with("recording/rec-1?inc=artist-rels+work-rels+place-rels")
    assert [(x.artist.name if x.artist else None) for x in recording.performing_relations] == [
        "Amanda Roocroft"
    ]


async def test_unknown_recording_raises() -> None:
    """A recording MusicBrainz does not know raises InvalidDataError."""
    provider, _ = _provider({})

    with pytest.raises(InvalidDataError):
        await provider.get_recording_relations("rec-1")
