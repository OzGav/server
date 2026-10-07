"""Tests for the helpers that turn classical tags into credits, works and periods."""

from __future__ import annotations

from typing import Any

import pytest
from music_assistant_models.enums import ArtistRole, Period

from music_assistant.helpers.tags import AudioTags
from music_assistant.providers.filesystem_local.helpers import (
    TrackWork,
    parse_track_work,
    performer_credit_role,
    period_from_genres,
)

WORK_ID = "a1b2c3d4-0000-4000-8000-000000000001"
MOVEMENT_WORK_ID = "a1b2c3d4-0000-4000-8000-000000000002"


def _tags(**tags: Any) -> AudioTags:
    """Create the tags of a track file holding the given tags."""
    return AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=1411,
        duration=300.0,
        tags=tags,
        has_cover_image=False,
        filename="track.flac",
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (None, (ArtistRole.PERFORMER, None)),
        ("", (ArtistRole.PERFORMER, None)),
        ("Performer", (ArtistRole.PERFORMER, None)),
        ("orchestra", (ArtistRole.ORCHESTRA, None)),
        ("Symphony Orchestra", (ArtistRole.ORCHESTRA, None)),
        ("philharmonic", (ArtistRole.ORCHESTRA, None)),
        ("sinfonia", (ArtistRole.ORCHESTRA, None)),
        ("Symphoniker", (ArtistRole.ORCHESTRA, None)),
        ("Philharmoniker", (ArtistRole.ORCHESTRA, None)),
        ("choir vocals", (ArtistRole.CHOIR, None)),
        ("chorus", (ArtistRole.CHOIR, None)),
        ("chorale", (ArtistRole.CHOIR, None)),
        ("Chor", (ArtistRole.CHOIR, None)),
        ("schola", (ArtistRole.CHOIR, None)),
        ("singers", (ArtistRole.CHOIR, None)),
        ("ensemble", (ArtistRole.ENSEMBLE, None)),
        ("string quartet", (ArtistRole.ENSEMBLE, None)),
        ("quintet", (ArtistRole.ENSEMBLE, None)),
        ("piano trio", (ArtistRole.ENSEMBLE, None)),
        ("consort", (ArtistRole.ENSEMBLE, None)),
        ("brass band", (ArtistRole.ENSEMBLE, None)),
        ("conductor", (ArtistRole.CONDUCTOR, None)),
        # a person leading a choir is not a choir
        ("chorus master", (ArtistRole.PERFORMER, "chorus master")),
        ("Choirmaster", (ArtistRole.PERFORMER, "Choirmaster")),
        ("choir master", (ArtistRole.PERFORMER, "choir master")),
        ("chorus director", (ArtistRole.PERFORMER, "chorus director")),
        ("piano", (ArtistRole.SOLOIST, "piano")),
        ("soprano vocals", (ArtistRole.SOLOIST, "soprano vocals")),
        ("vocals", (ArtistRole.SOLOIST, "vocals")),
        # whole words only, so these instruments are not mistaken for a choir or a band
        ("harpsichord", (ArtistRole.SOLOIST, "harpsichord")),
        ("bandoneon", (ArtistRole.SOLOIST, "bandoneon")),
    ],
)
def test_performer_credit_role(role: str | None, expected: tuple[ArtistRole, str | None]) -> None:
    """A performer's role or instrument text maps to a credit role and instrument."""
    assert performer_credit_role(role) == expected


@pytest.mark.parametrize(
    ("genres", "expected"),
    [
        (["Medieval"], Period.MEDIEVAL),
        (["Renaissance"], Period.RENAISSANCE),
        (["Classical", "Baroque"], Period.BAROQUE),
        (["Classical Period"], Period.CLASSICAL),
        (["classical era"], Period.CLASSICAL),
        (["Romantic"], Period.ROMANTIC),
        (["Modern"], Period.MODERN),
        (["20th Century"], Period.MODERN),
        (["Contemporary"], Period.CONTEMPORARY),
        (["21st Century"], Period.CONTEMPORARY),
        (["Classical"], None),
        (["Rock", "Pop"], None),
        ([], None),
        (["Romantic", "Classical Period", "Romantic"], Period.ROMANTIC),
    ],
)
def test_period_from_genres(genres: list[str], expected: Period | None) -> None:
    """Period names among the genres set the period, the plain genre Classical never does."""
    assert period_from_genres(genres) == expected


def test_no_work_without_a_work_tag() -> None:
    """A track without a work tag has no work, even with a MusicBrainz work id."""
    assert parse_track_work(_tags(musicbrainzworkid=[WORK_ID])) is None


def test_single_work_with_its_id() -> None:
    """A single work tag is the work, with its MusicBrainz id."""
    work = parse_track_work(_tags(work=["Symphony No. 5"], musicbrainzworkid=[WORK_ID]))
    assert work == TrackWork("Symphony No. 5", WORK_ID)


def test_movement_tags_pick_the_most_general_work() -> None:
    """With movement tags the track belongs to the first (most general) work."""
    work = parse_track_work(
        _tags(
            work=["Symphony No. 5", "Symphony No. 5: I. Allegro con brio"],
            musicbrainzworkid=[WORK_ID, MOVEMENT_WORK_ID],
            movementname="Allegro con brio",
        )
    )
    assert work == TrackWork("Symphony No. 5", WORK_ID)


def test_movement_tags_leave_the_id_off_when_the_counts_differ() -> None:
    """Work ids only pair up with the work values when there are as many of each."""
    work = parse_track_work(
        _tags(
            work=["Symphony No. 5"],
            musicbrainzworkid=[WORK_ID, MOVEMENT_WORK_ID],
            movement="1",
        )
    )
    assert work == TrackWork("Symphony No. 5")


def test_without_movement_tags_the_most_specific_work_is_picked() -> None:
    """Without movement tags the track belongs to the last (most specific) work."""
    work = parse_track_work(
        _tags(
            work=["Pictures at an Exhibition", "Pictures at an Exhibition (orch. Ravel)"],
            musicbrainzworkid=[WORK_ID, MOVEMENT_WORK_ID],
        )
    )
    assert work == TrackWork("Pictures at an Exhibition (orch. Ravel)", MOVEMENT_WORK_ID)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "Orchestersuite Nr. 3 D-Dur, BWV 1068: II. Air",
            TrackWork("Orchestersuite Nr. 3 D-Dur, BWV 1068", None, "II. Air", 2),
        ),
        (
            "Water Music, Suite no. 1 in F major, HWV 348: V. Air. Presto",
            TrackWork("Water Music, Suite no. 1 in F major, HWV 348", None, "V. Air. Presto", 5),
        ),
        (
            "Symphony No. 9: 4. Presto",
            TrackWork("Symphony No. 9", None, "4. Presto", 4),
        ),
        (
            "Five Mystical Songs: No. 4. The Call",
            TrackWork("Five Mystical Songs", None, "No. 4. The Call", 4),
        ),
    ],
)
def test_numbered_movement_title_links_the_parent_work(title: str, expected: TrackWork) -> None:
    """A work titled as a numbered movement of a parent work links the parent instead."""
    assert parse_track_work(_tags(work=[title], musicbrainzworkid=[WORK_ID])) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Te Deum, H. 146: Prélude",  # codespell:ignore
        "Piano Sonata No. 32 in C minor, op. 111: Maestoso",
        'Kantate, BWV 147 "Herz und Mund und Tat und Leben": Teil II, X. Choral '
        '"Jesus bleibet meine Freude"',
        'Lohengrin, WWV 75: Akt III, Szene I. "Treulich geführt ziehet dahin" (Chor)',  # codespell:ignore
        "Le nozze di Figaro, K. 492: Act II, No. 14 Terzetto",
        # Roman numerals beyond XX are not taken for a movement number
        "Suite: XXI. Finale",
    ],
)
def test_unnumbered_movement_title_stays_as_tagged(title: str) -> None:
    """A work title that does not name a numbered movement is the work, with its id."""
    assert parse_track_work(_tags(work=[title], musicbrainzworkid=[WORK_ID])) == TrackWork(
        title, WORK_ID
    )


def test_movement_tags_still_split_a_movement_title() -> None:
    """With movement tags the general work is split as well when it names a movement."""
    work = parse_track_work(
        _tags(
            work=["Symphony No. 5: I. Allegro con brio"],
            musicbrainzworkid=[MOVEMENT_WORK_ID],
            movementname="Allegro con brio",
        )
    )
    assert work == TrackWork("Symphony No. 5", None, "I. Allegro con brio", 1)
