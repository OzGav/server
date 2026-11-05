"""Tests for turning the credit roles music services name into credits."""

from __future__ import annotations

import pytest
from music_assistant_models.enums import ArtistRole, MediaType
from music_assistant_models.media_items import ItemMapping

from music_assistant.helpers.credit_roles import (
    NO_ROLE,
    is_known_role,
    is_performing_role,
    role_credits,
    track_role_credits,
)


@pytest.mark.parametrize(
    ("roles", "has_conductor", "is_main_artist", "expected"),
    [
        (["Conductor"], False, False, [(ArtistRole.CONDUCTOR, None)]),
        # a director conducts unless the track names a conductor
        (["Director"], False, False, [(ArtistRole.CONDUCTOR, None)]),
        (["Director"], True, False, []),
        (["Tenor Vocals"], False, False, [(ArtistRole.SOLOIST, "tenor vocals")]),
        (["BaritoneVocalist"], False, False, [(ArtistRole.SOLOIST, "baritone vocals")]),
        (["Piano"], False, False, [(ArtistRole.SOLOIST, "piano")]),
        # a soloist with an instrument needs no plain soloist credit
        (["Soloist", "Piano"], False, False, [(ArtistRole.SOLOIST, "piano")]),
        # a main artist holding a performing role is no plain performer as well
        (["ChamberEnsemble", "MainArtist"], False, False, [(ArtistRole.ENSEMBLE, None)]),
        (["MainArtist"], False, False, [(ArtistRole.PERFORMER, None)]),
        ([NO_ROLE], False, False, [(ArtistRole.PERFORMER, None)]),
        ([], False, True, [(ArtistRole.PERFORMER, None)]),
        (["Lyricist", "Librettist"], False, False, [(ArtistRole.LYRICIST, None)]),
        # roles that are no credit, known or not, give nothing
        (["Recording Engineer", "Translator", "NonLyricAuthor"], False, False, []),
    ],
)
def test_role_credits(
    roles: list[str],
    has_conductor: bool,
    is_main_artist: bool,
    expected: list[tuple[ArtistRole, str | None]],
) -> None:
    """The roles a service names for an artist give its credit roles and instruments."""
    assert role_credits(roles, has_conductor, is_main_artist) == expected


def test_track_role_credits() -> None:
    """
    Each artist of a track is credited with the roles it holds, in order within each role.

    A main artist that composed the track is no performer, one without a role is. A director
    next to a conductor, a role that is no credit and an artist given no credit are left out.
    """
    roles_by_artist = {
        "abbado": ["Conductor"],
        "director": ["Director"],
        "verdi": ["Composer"],
        "lso": [],
        "first_bass": ["Bass Vocals"],
        "second_bass": ["Bass Vocals"],
        "producer": ["Producer"],
        "unknown": ["Piano"],
    }

    def _artist(key: str) -> ItemMapping | None:
        if key == "unknown":
            return None
        return ItemMapping(media_type=MediaType.ARTIST, item_id=key, provider="test", name=key)

    result = track_role_credits(roles_by_artist, {"verdi", "lso"}, _artist)

    assert [(x.artist.item_id, x.role, x.instrument, x.position) for x in result] == [
        ("abbado", ArtistRole.CONDUCTOR, None, 0),
        ("verdi", ArtistRole.COMPOSER, None, 0),
        ("lso", ArtistRole.PERFORMER, None, 0),
        ("first_bass", ArtistRole.SOLOIST, "bass vocals", 0),
        ("second_bass", ArtistRole.SOLOIST, "bass vocals", 1),
    ]


@pytest.mark.parametrize(
    ("role", "performing", "known"),
    [
        ("Tenor Vocals", True, True),
        ("Piano", True, True),
        ("Orchestra", True, True),
        ("Composer", False, True),
        ("Translator", False, True),
        ("Producer", False, False),
    ],
)
def test_performing_and_known_roles(role: str, performing: bool, known: bool) -> None:
    """A voice, an instrument or a performing role performs; other credit roles are only known."""
    assert is_performing_role(role) is performing
    assert is_known_role(role) is known
