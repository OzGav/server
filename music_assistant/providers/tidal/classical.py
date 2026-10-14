"""Helpers to read the classical details Tidal gives for its albums and tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.media_items import Credit

from music_assistant.helpers.credit_roles import role_credits, role_word
from music_assistant.helpers.util import is_placeholder_artist
from music_assistant.helpers.work_titles import composers_listed_first, parse_work_title

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from music_assistant_models.media_items import Artist, ItemMapping

    from .jsonapi import JsonApiDocument

# the name of Tidal's genre of classical music
CLASSICAL_GENRE = "Classical"


def is_classical_track(doc: JsonApiDocument, resource: dict[str, Any]) -> bool:
    """
    Return whether Tidal files a track under classical music.

    :param doc: The JSON:API document holding the track.
    :param resource: The track resource.
    """
    return any(
        (genre.get("attributes") or {}).get("genreName") == CLASSICAL_GENRE
        for genre in doc.related(resource, "genres")
    )


def is_classical_album(tracks: Sequence[tuple[JsonApiDocument, dict[str, Any]]]) -> bool:
    """
    Return whether an album is classical, judged by its tracks.

    An album is classical when more than half of its tracks are filed under classical music, or
    when more than half have a composer credit and a "Work: Piece" title.

    :param tracks: The document and resource of each track of the album.
    """
    by_genre = sum(is_classical_track(doc, resource) for doc, resource in tracks)
    by_title = sum(
        _has_composer_credit(doc, resource) and parse_work_title(_title(resource)) is not None
        for doc, resource in tracks
    )
    return max(by_genre, by_title) * 2 > len(tracks)


def lists_composers_first(tracks: Sequence[tuple[JsonApiDocument, dict[str, Any]]]) -> bool:
    """
    Return whether a classical album lists each track's composer as its first artist.

    :param tracks: The document and resource of each track of the album.
    """
    return composers_listed_first(
        [
            (
                _title(resource),
                (artists[0].get("attributes") or {}).get("name")
                if (artists := doc.related(resource, "artists"))
                else None,
            )
            for doc, resource in tracks
        ]
    )


def track_credits(
    doc: JsonApiDocument,
    resource: dict[str, Any],
    main_artists: Sequence[Artist | ItemMapping],
    parse_artist: Callable[[dict[str, Any]], Artist],
) -> list[Credit]:
    """
    Return the credits of a Tidal track, each linked to the Tidal artist it credits.

    A main artist of the track is credited as performer as well, unless it holds a performing
    role or composed the track. A director is credited as conductor when the track names no
    conductor. Production roles and placeholder names such as "D.R." give no credit.

    :param doc: The JSON:API document holding the track and its credits.
    :param resource: The track resource.
    :param main_artists: The main artists of the track.
    :param parse_artist: Returns the artist for a Tidal artist resource.
    """
    artists: dict[str, Artist | ItemMapping] = {x.item_id: x for x in main_artists}
    roles_by_artist: dict[str, list[str]] = {}
    for credit in doc.related(resource, "credits"):
        attributes = credit.get("attributes") or {}
        artist_ids = doc.linkage_ids(credit, "artist")
        if not (role := attributes.get("role")) or not artist_ids:
            continue
        artist_id = next(iter(artist_ids))
        if artist_id not in artists:
            # the linked artist has the proper name, the credit's own may be in capitals
            artist_resource = doc.related_one(credit, "artist") or {
                "id": artist_id,
                "attributes": {"name": attributes.get("name")},
            }
            if not (artist_resource.get("attributes") or {}).get("name"):
                continue
            artists[artist_id] = parse_artist(artist_resource)
        if is_placeholder_artist(artists[artist_id].name):
            continue
        roles_by_artist.setdefault(artist_id, []).append(role)
    for main_artist in main_artists:
        roles_by_artist.setdefault(main_artist.item_id, [])
    has_conductor = any(
        role_word(x) == "conductor" for roles in roles_by_artist.values() for x in roles
    )
    main_artist_ids = {x.item_id for x in main_artists}
    result: list[Credit] = []
    for artist_id, roles in roles_by_artist.items():
        is_main_artist = artist_id in main_artist_ids and all(
            role_word(x) != "composer" for x in roles
        )
        for role, instrument in role_credits(roles, has_conductor, is_main_artist):
            position = sum(x.role == role for x in result)
            result.append(
                Credit(
                    artist=artists[artist_id], role=role, instrument=instrument, position=position
                )
            )
    return result


def _has_composer_credit(doc: JsonApiDocument, resource: dict[str, Any]) -> bool:
    """Return whether a track credits a composer."""
    return any(
        role_word((credit.get("attributes") or {}).get("role") or "") == "composer"
        for credit in doc.related(resource, "credits")
    )


def _title(resource: dict[str, Any]) -> str:
    """Return the title of a track resource."""
    return str((resource.get("attributes") or {}).get("title") or "")
