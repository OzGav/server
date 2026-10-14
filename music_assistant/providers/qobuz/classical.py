"""Helpers to read the classical details Qobuz gives for its albums and tracks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ArtistRole
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Credit

from music_assistant.helpers.compare import compare_strings
from music_assistant.helpers.util import is_placeholder_artist
from music_assistant.helpers.work_titles import TrackWork, parse_work_title, work_with_movement

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from music_assistant_models.media_items import Artist, ItemMapping

# the Qobuz genre of classical music, which the path of every classical sub-genre starts with
CLASSICAL_GENRE_ID = 10

# the credit role of each role a performers string names, keyed by the role in lowercase without
# spaces or hyphens; None for a role that is no credit, such as a producer or a publisher. The
# main artist role marks an artist performing on the recording without naming how.
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
# the role of an artist the performers string names without any role
_NO_ROLE = ""
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
# production and rights roles, such as "Recording Engineer" or "OriginalPublisher"
_NO_CREDIT_ROLE = re.compile(
    r"(engineer|producer|publisher|editor|mixer|mixing|mastering|assistant|administrator|"
    r"programmer|programming|manager|coordinator)$"
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


def is_classical_album(album_obj: dict[str, Any]) -> bool:
    """
    Return whether Qobuz files an album under classical music or one of its sub-genres.

    :param album_obj: A Qobuz album object.
    """
    genre = album_obj.get("genre") or {}
    return CLASSICAL_GENRE_ID in (genre.get("path") or [genre.get("id")])


def parse_performers(performers: str) -> list[tuple[str, list[str]]]:
    """
    Return the names and roles a Qobuz performers string credits.

    Each credit names an artist followed by its roles, as in "Carlos Kleiber, Conductor". Qobuz
    joins several main artists into one name ("Ernest Bour, Orchestra X, MainArtist"), which is
    returned as that one name.

    :param performers: The performers string of a track, credits separated by " - ".
    """
    result: list[tuple[str, list[str]]] = []
    for credit in performers.split(" - "):
        parts = [x.strip() for x in credit.split(", ") if x.strip()]
        if not parts:
            continue
        name_end = len(parts)
        while name_end > 1 and _is_role(parts[name_end - 1]):
            name_end -= 1
        result.append((", ".join(parts[:name_end]), parts[name_end:]))
    return result


def track_credits(
    performers: str,
    composer: Artist | None,
    main_artists: Sequence[Artist | ItemMapping],
    album_artists: Mapping[str, Artist],
    album_roles: Mapping[str, list[str]],
    named_artist: Callable[[str], Artist],
) -> list[Credit]:
    """
    Return the credits of a Qobuz track.

    An artist the performers string names as main artist or without any role is credited as
    performer, unless it holds a performing role. So is a main artist of the track, unless it
    composed the track. A director is credited as conductor when the track names no conductor.

    :param performers: The performers string of the track.
    :param composer: The composer of the track.
    :param main_artists: The main artists of the track.
    :param album_artists: The Qobuz artists the album names, keyed by artist_key of their name.
    :param album_roles: The roles the album credits its artists with, keyed by artist_key.
    :param named_artist: Returns the artist for a name Qobuz gives no artist id for.
    """
    artists: dict[str, Artist | ItemMapping] = dict(album_artists)
    roles_by_artist: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    if composer:
        artists[artist_key(composer.name)] = composer
        roles_by_artist[artist_key(composer.name)] = ["Composer"]
    for name, roles in parse_performers(performers):
        if is_placeholder_artist(name):
            continue
        names.setdefault(artist_key(name), name)
        # an artist named without any role is one of the performers
        roles_by_artist.setdefault(artist_key(name), []).extend(roles or [_NO_ROLE])
    if composer:
        _drop_respelled_composer(roles_by_artist, artist_key(composer.name))
    for key, roles in album_roles.items():
        # an album artist credited with a performing role on the album performs on every track
        if performing_roles := [x for x in roles if _performing_role(x)]:
            roles_by_artist.setdefault(key, []).extend(performing_roles)
    main_artist_keys = {artist_key(x.name) for x in main_artists}
    for main_artist in main_artists:
        artists[artist_key(main_artist.name)] = main_artist
        roles_by_artist.setdefault(artist_key(main_artist.name), [])
    # Qobuz joins the main artists into one artist ("Ernest Bour, Orchestra X"), which needs no
    # credit of its own when the performers string names each of them
    joined = {
        artist_key(name)
        for name in (*(x.name for x in main_artists), *names.values())
        if ", " in name and all(artist_key(x) in roles_by_artist for x in name.split(", "))
    }
    # Qobuz cuts long main artist names short, so a joined main artist that starts a longer
    # joined name of the performers string is that name
    joined.update(
        key
        for key in (artist_key(x.name) for x in main_artists if ", " in x.name)
        if any(", " in name and x != key and x.startswith(key) for x, name in names.items())
    )
    # a director leads the performance, unless the track names a conductor as well
    has_conductor = any(
        _word(x) == "conductor" for roles in roles_by_artist.values() for x in roles
    )
    result: list[Credit] = []
    for key, roles in roles_by_artist.items():
        # an artist named without an id or any role may well be another credit's artist under
        # another name, as "Wiener Philharmoniker" next to the album's "Wiener Philharmonic
        # Orchestra"
        if key in joined or (
            key not in artists and (key not in names or all(x == _NO_ROLE for x in roles))
        ):
            continue
        # a composer is a performer as well only when the performers string says so
        is_main_artist = key in main_artist_keys and all(_word(x) != "composer" for x in roles)
        for role, instrument in _artist_credits(roles, has_conductor, is_main_artist):
            if not (credited := artists.get(key)):
                credited = artists[key] = named_artist(names[key])
            position = sum(x.role == role for x in result)
            result.append(
                Credit(artist=credited, role=role, instrument=instrument, position=position)
            )
    return result


def track_work(title: str, work: str | None, composer: str | None) -> TrackWork | None:
    """
    Return the work and movement of a Qobuz track, from its title or else its work.

    :param title: The title of the track, such as "I. Allegro con brio".
    :param work: The work Qobuz gives for the track, such as "Symphony No. 5 in C minor, Op. 67".
    :param composer: The name of the track's composer.
    """
    # a title naming its work names it the way the album's other titles do, which may differ
    # from the work Qobuz gives ("Goldberg-Variationen BWV 988")
    if title_work := parse_work_title(title, composer):
        return title_work
    if not work or not (work := work.strip()):
        return None
    # a track that is a whole work has the work as its title
    return work_with_movement(work, None if compare_strings(title, work) else title)


def artist_key(name: str) -> str:
    """Return the key two names of one artist share, ignoring case and diacritics."""
    return create_safe_string(name, True, True)


def _artist_credits(
    roles: list[str], has_conductor: bool, is_main_artist: bool = False
) -> list[tuple[ArtistRole, str | None]]:
    """Return the credit roles and instruments one artist's roles give."""
    result: dict[tuple[ArtistRole, str | None], None] = {}
    performs = False
    for role in roles:
        word = _word(role)
        if word == "director" and has_conductor:
            continue
        if voice := _VOICE.match(role):
            # a voice is written as MusicBrainz and Picard write it, as in "tenor vocals"
            result[(ArtistRole.SOLOIST, f"{voice.group(1) or ''} vocals".lower().lstrip())] = None
        elif role.lower() in _INSTRUMENTS:
            result[(ArtistRole.SOLOIST, role.lower())] = None
        elif role == _NO_ROLE or _ROLE_WORDS.get(word) == ArtistRole.MAIN_ARTIST:
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


def _drop_respelled_composer(roles_by_artist: dict[str, list[str]], composer_key: str) -> None:
    """Drop the composer roles of other names when the track's composer is not among them."""
    composer_keys = [
        key
        for key, roles in roles_by_artist.items()
        if key != composer_key and any(_word(x) == "composer" for x in roles)
    ]
    # a performers string naming co-composers names the track's composer too, so one that
    # leaves it out spells that composer another way ("Frederic Chopin" for "Frédéric Chopin")
    if any(_word(x) == "composer" for x in roles_by_artist[composer_key][1:]):
        return
    for key in composer_keys:
        if not (roles := [x for x in roles_by_artist[key] if _word(x) != "composer"]):
            del roles_by_artist[key]
        else:
            roles_by_artist[key] = roles


def _performing_role(role: str) -> bool:
    """Return whether a role, from a performers string or an album, names a performing credit."""
    if _VOICE.match(role) or role.lower() in _INSTRUMENTS:
        return True
    return _ROLE_WORDS.get(_word(role)) in _PERFORMING_ROLES


def _is_role(part: str) -> bool:
    """Return whether a part of a performers string credit is a role rather than a name."""
    word = _word(part)
    return (
        word in _ROLE_WORDS
        or bool(_NO_CREDIT_ROLE.search(word) or _VOICE.match(part))
        or part.lower() in _INSTRUMENTS
        # Qobuz writes its other roles as one word, as in "SoundEditor"
        or " " not in part
    )


def _word(role: str) -> str:
    """Return a role in lowercase without spaces or hyphens, as in "mainartist"."""
    return re.sub(r"[\s-]", "", role).lower()
