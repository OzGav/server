"""Helpers to turn the credit roles music services name, such as "Tenor Vocals", into credits."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole
from music_assistant_models.media_items import Credit

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Mapping

    from music_assistant_models.media_items import Artist, ItemMapping

# the role of an artist a service names without any role
NO_ROLE = ""

# the credit role of each role a service names, keyed by the role in lowercase without spaces
# or hyphens; None for a role that is no credit, such as a producer or a publisher. The main
# artist role marks an artist performing on the recording without naming how.
_ROLE_WORDS: dict[str, ArtistRole | None] = {
    "conductor": ArtistRole.CONDUCTOR,
    "director": ArtistRole.CONDUCTOR,
    "orchestra": ArtistRole.ORCHESTRA,
    "ensemble": ArtistRole.ENSEMBLE,
    "chamberensemble": ArtistRole.ENSEMBLE,
    "choir": ArtistRole.CHOIR,
    "chorus": ArtistRole.CHOIR,
    "soloist": ArtistRole.SOLOIST,
    "narrator": ArtistRole.PERFORMER,
    "composer": ArtistRole.COMPOSER,
    "lyricist": ArtistRole.LYRICIST,
    "librettist": ArtistRole.LYRICIST,
    "arranger": ArtistRole.ARRANGER,
    "mainartist": ArtistRole.MAIN_ARTIST,
    "artist": ArtistRole.MAIN_ARTIST,
    "performer": ArtistRole.MAIN_ARTIST,
    "featuredartist": ArtistRole.MAIN_ARTIST,
    "author": None,
    "writer": None,
    "nonlyricauthor": None,
    "translator": None,
    "label": None,
    "aandradministrator": None,
}
_PERFORMING_ROLES = frozenset(
    (
        ArtistRole.CONDUCTOR,
        ArtistRole.ORCHESTRA,
        ArtistRole.ENSEMBLE,
        ArtistRole.CHOIR,
        ArtistRole.SOLOIST,
        ArtistRole.PERFORMER,
    )
)
# a voice, such as "Tenor Vocals" or "TenorVocalist"
_VOICE = re.compile(r"^(?:(.+?)\s?)?(?:vocals|vocalist|singer)$", re.IGNORECASE)
_INSTRUMENTS = frozenset(
    (
        "accordion",
        "bandoneon",
        "bass clarinet",
        "bassoon",
        "basso continuo",
        "celesta",
        "cello",
        "cembalo",
        "clarinet",
        "clavichord",
        "continuo",
        "contrabassoon",
        "cor anglais",
        "cornet",
        "cornett",
        "double bass",
        "english horn",
        "flute",
        "fortepiano",
        "french horn",
        "guitar",
        "harmonium",
        "harp",
        "harpsichord",
        "horn",
        "keyboards",
        "lute",
        "mandolin",
        "oboe",
        "oboe d'amore",
        "organ",
        "percussion",
        "piano",
        "piccolo",
        "recorder",
        "saxophone",
        "synthesizer",
        "theorbo",
        "timpani",
        "trombone",
        "trumpet",
        "tuba",
        "viol",
        "viola",
        "viola da gamba",
        "viola d'amore",
        "violin",
        "violoncello",
    )
)


def role_credits(
    roles: list[str], has_conductor: bool, is_main_artist: bool = False
) -> list[tuple[ArtistRole, str | None]]:
    """
    Return the credit roles and instruments the roles a service names for one artist give.

    A voice or an instrument makes the artist a soloist with that voice or instrument. A
    director is credited as conductor unless the track names a conductor. An artist named
    without any role or as main artist is credited as performer, unless it holds a performing
    role. Roles that are no credit, such as a producer, give nothing.

    :param roles: The roles of the artist, such as ["Conductor"] or ["Tenor Vocals"].
    :param has_conductor: Whether the track names a conductor.
    :param is_main_artist: Whether the artist is a main artist of the track.
    """
    result: dict[tuple[ArtistRole, str | None], None] = {}
    performs = False
    for role in roles:
        word = role_word(role)
        if word == "director" and has_conductor:
            continue
        if voice := _VOICE.match(role):
            # a voice is written as MusicBrainz and Picard write it, as in "tenor vocals"
            result[(ArtistRole.SOLOIST, f"{voice.group(1) or ''} vocals".lower().lstrip())] = None
        elif role.lower() in _INSTRUMENTS:
            result[(ArtistRole.SOLOIST, role.lower())] = None
        elif role == NO_ROLE or _ROLE_WORDS.get(word) == ArtistRole.MAIN_ARTIST:
            performs = True
        elif credit_role := _ROLE_WORDS.get(word):
            result[(credit_role, None)] = None
    performing = [x for x in result if x[0] in _PERFORMING_ROLES]
    if any(instrument for _role, instrument in performing):
        # a soloist credited with an instrument or voice needs no plain soloist credit
        result.pop((ArtistRole.SOLOIST, None), None)
    if (performs or is_main_artist) and not performing:
        result[(ArtistRole.PERFORMER, None)] = None
    return list(result)


def track_role_credits(
    roles_by_artist: Mapping[str, list[str]],
    main_artist_keys: Container[str],
    credited_artist: Callable[[str], Artist | ItemMapping | None],
) -> list[Credit]:
    """
    Return the credits the roles a service names for each artist of a track give.

    A main artist is credited as performer as well, unless it holds a performing role or
    composed the track. A director is credited as conductor when the track names no conductor.

    :param roles_by_artist: The roles of each artist, keyed by the service's key of the artist.
    :param main_artist_keys: The keys of the main artists of the track.
    :param credited_artist: Returns the artist for a key, or None for an artist given no credit.
    """
    has_conductor = any(
        role_word(x) == "conductor" for roles in roles_by_artist.values() for x in roles
    )
    result: list[Credit] = []
    for key, roles in roles_by_artist.items():
        is_main_artist = key in main_artist_keys and all(role_word(x) != "composer" for x in roles)
        if not (role_instruments := role_credits(roles, has_conductor, is_main_artist)) or not (
            artist := credited_artist(key)
        ):
            continue
        for role, instrument in role_instruments:
            position = sum(x.role == role for x in result)
            result.append(
                Credit(artist=artist, role=role, instrument=instrument, position=position)
            )
    return result


def is_performing_role(role: str) -> bool:
    """Return whether a role names a performing credit, such as a conductor or a voice."""
    if _VOICE.match(role) or role.lower() in _INSTRUMENTS:
        return True
    return _ROLE_WORDS.get(role_word(role)) in _PERFORMING_ROLES


def is_known_role(role: str) -> bool:
    """Return whether a role is one of the credit roles, voices or instruments known here."""
    return (
        role_word(role) in _ROLE_WORDS or bool(_VOICE.match(role)) or role.lower() in _INSTRUMENTS
    )


def role_word(role: str) -> str:
    """Return a role in lowercase without spaces or hyphens, as in "mainartist"."""
    return re.sub(r"[\s-]", "", role).lower()
