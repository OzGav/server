"""Helpers to read the classical details Qobuz gives for its albums and tracks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from music_assistant_models.helpers import create_safe_string

from music_assistant.helpers.compare import compare_strings
from music_assistant.helpers.credit_roles import (
    NO_ROLE,
    is_known_role,
    is_performing_role,
    role_word,
    track_role_credits,
)
from music_assistant.helpers.util import is_placeholder_artist
from music_assistant.helpers.work_titles import TrackWork, parse_work_title, work_with_movement

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from music_assistant_models.media_items import Artist, Credit, ItemMapping

# the Qobuz genre of classical music, which the path of every classical sub-genre starts with
CLASSICAL_GENRE_ID = 10

# production and rights roles, such as "Recording Engineer" or "OriginalPublisher"
_NO_CREDIT_ROLE = re.compile(
    r"(engineer|producer|publisher|editor|mixer|mixing|mastering|assistant|administrator|"
    r"programmer|programming|manager|coordinator)$"
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
        roles_by_artist.setdefault(artist_key(name), []).extend(roles or [NO_ROLE])
    if composer:
        _drop_respelled_composer(roles_by_artist, artist_key(composer.name))
    for key, roles in album_roles.items():
        # an album artist credited with a performing role on the album performs on every track
        if performing_roles := [x for x in roles if is_performing_role(x)]:
            roles_by_artist.setdefault(key, []).extend(performing_roles)
    for main_artist in main_artists:
        artists[artist_key(main_artist.name)] = main_artist
        roles_by_artist.setdefault(artist_key(main_artist.name), [])
    joined = _joined_names(main_artists, names, roles_by_artist)

    def credited_artist(key: str) -> Artist | ItemMapping | None:
        if key in joined:
            return None
        if key not in artists:
            # an artist named without an id or any role may well be another credit's artist
            # under another name, as "Wiener Philharmoniker" next to the album's "Wiener
            # Philharmonic Orchestra"
            if key not in names or all(x == NO_ROLE for x in roles_by_artist[key]):
                return None
            artists[key] = named_artist(names[key])
        return artists[key]

    return track_role_credits(
        roles_by_artist, {artist_key(x.name) for x in main_artists}, credited_artist
    )


def track_work(title: str, work: str | None, composer: str | None) -> TrackWork | None:
    """
    Return the work and movement of a classical Qobuz track, from its title or else its work.

    :param title: The title of the track, such as "I. Allegro con brio".
    :param work: The work Qobuz gives for the track, such as "Symphony No. 5 in C minor, Op. 67".
    :param composer: The name of the track's composer.
    """
    # a title naming its work names it the way the album's other titles do, which may differ
    # from the work Qobuz gives ("Goldberg-Variationen BWV 988")
    if title_work := parse_work_title(title, composer, classical=True):
        return title_work
    if not work or not (work := work.strip()):
        return None
    # a track that is a whole work has the work as its title
    return work_with_movement(work, None if compare_strings(title, work) else title)


def artist_key(name: str) -> str:
    """Return the key two names of one artist share, ignoring case and diacritics."""
    return create_safe_string(name, True, True)


def _joined_names(
    main_artists: Sequence[Artist | ItemMapping],
    names: Mapping[str, str],
    roles_by_artist: Mapping[str, list[str]],
) -> set[str]:
    """Return the keys of the joined names of several artists, which get no credit of their own."""
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
    return joined


def _drop_respelled_composer(roles_by_artist: dict[str, list[str]], composer_key: str) -> None:
    """Drop the composer roles of other names when the track's composer is not among them."""
    composer_keys = [
        key
        for key, roles in roles_by_artist.items()
        if key != composer_key and any(role_word(x) == "composer" for x in roles)
    ]
    # a performers string naming co-composers names the track's composer too, so one that
    # leaves it out spells that composer another way ("Frederic Chopin" for "Frédéric Chopin")
    if any(role_word(x) == "composer" for x in roles_by_artist[composer_key][1:]):
        return
    for key in composer_keys:
        if not (roles := [x for x in roles_by_artist[key] if role_word(x) != "composer"]):
            del roles_by_artist[key]
        else:
            roles_by_artist[key] = roles


def _is_role(part: str) -> bool:
    """Return whether a part of a performers string credit is a role rather than a name."""
    return (
        is_known_role(part)
        or bool(_NO_CREDIT_ROLE.search(role_word(part)))
        # Qobuz writes its other roles as one word, as in "SoundEditor"
        or " " not in part
    )
