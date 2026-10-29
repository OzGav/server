"""
Helper utilities for the Metadata Controller.

Pure functions used by the controller and its mixins that do not need access to
the controller instance.
"""

from __future__ import annotations

import pathlib
import unicodedata
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistEntityType
from music_assistant_models.helpers import create_safe_string, create_sort_name
from music_assistant_models.media_items.metadata import LifeSpan

from music_assistant.providers.musicbrainz.models import MusicBrainzArtist, MusicBrainzWork

from .constants import _IMAGEPROXY_CONTENT_TYPES

if TYPE_CHECKING:
    from music_assistant_models.media_items import Artist


def fill_artist_from_musicbrainz(
    artist: Artist, mb_artist: MusicBrainzArtist, replace: bool = False
) -> None:
    """
    Fill an artist's empty entity type and life span in from its MusicBrainz artist.

    :param artist: The artist to fill in.
    :param mb_artist: Its MusicBrainz artist.
    :param replace: Replace the life span by MusicBrainz's when it has one.
    """
    metadata = artist.metadata
    if not metadata.artist_entity_type and mb_artist.type:
        entity_type = ArtistEntityType(mb_artist.type.lower())
        if entity_type != ArtistEntityType.UNKNOWN:
            metadata.artist_entity_type = entity_type
    if (replace or not metadata.life_span) and (life_span := mb_artist.life_span):
        metadata.life_span = LifeSpan(
            begin=life_span.begin, end=life_span.end, ended=life_span.ended
        )


def musicbrainz_work_name(mb_work: MusicBrainzWork, language: str = "en") -> str:
    """
    Return the name MusicBrainz gives a work, in the given language when it has one.

    The primary alias in the given language comes first, then the primary English alias, then
    the work's own title, which is in the work's original language. A title not in Latin script
    gives way to a Latin alias, one in the given language or English first.

    :param mb_work: The MusicBrainz work with its aliases.
    :param language: The language to name the work in first, such as "de".
    """
    primaries = {x.locale: x.name for x in mb_work.aliases if x.primary and x.locale}
    if named := primaries.get(language) or primaries.get("en"):
        return named
    if is_latin(mb_work.title):
        return mb_work.title
    aliases = [
        x for x in mb_work.aliases if is_latin(x.name) and (x.type or "").lower() != "search hint"
    ]
    alias = min(aliases, key=lambda x: (x.locale != language, x.locale != "en"), default=None)
    return alias.name if alias else mb_work.title


def musicbrainz_name(mb_artist: MusicBrainzArtist, language: str = "en") -> tuple[str, str] | None:
    """
    Return the name and sort name MusicBrainz gives an artist in Latin script.

    The primary alias in the given language comes first, then the primary English alias, then
    a Latin main name, then an English artist name alias. A Latin main name is kept over an
    alias that only writes it without its accents. None when MusicBrainz has no such name.

    :param mb_artist: The MusicBrainz artist with its aliases.
    :param language: The language to name the artist in first, such as "de".
    """
    aliases = [x for x in mb_artist.aliases or [] if is_latin(x.name)]
    primaries = [
        x
        for locale in dict.fromkeys((language, "en"))
        for x in aliases
        if x.primary and x.locale == locale
    ]
    alias = next(iter(primaries), None)
    if (
        alias
        and is_latin(mb_artist.name)
        and create_safe_string(alias.name, True, True)
        == create_safe_string(mb_artist.name, True, True)
    ):
        alias = None
    if not alias and not is_latin(mb_artist.name):
        alias = next(
            (x for x in aliases if x.locale == "en" and (x.type or "").lower() == "artist name"),
            None,
        )
        if not alias:
            return None
    if alias:
        name = alias.name
        sort_name = alias.sort_name or (
            mb_artist.sort_name if is_latin(mb_artist.sort_name) else ""
        )
    else:
        name, sort_name = mb_artist.name, mb_artist.sort_name
    name = _plain_hyphens(name)
    return name, (_plain_hyphens(sort_name) if sort_name else create_sort_name(name))


def is_latin(text: str) -> bool:
    """Return whether every letter of a text is a Latin letter."""
    return all(unicodedata.name(x, "").startswith("LATIN") for x in text if x.isalpha())


def _detect_image_format(path: str) -> str:
    """Detect image format from file path extension, defaulting to jpg."""
    # strip any query suffix (e.g. a cache-busting ?cs=) before extension detection
    match pathlib.PurePath(path.split("?", 1)[0]).suffix.lower():
        case ".svg":
            return "svg"
        case ".png":
            return "png"
        case _:
            return "jpg"


def _normalize_imageproxy_format(value: str | None) -> str | None:
    """Return a validated, lowercase imageproxy format, or None when invalid."""
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in _IMAGEPROXY_CONTENT_TYPES:
        return normalized
    return None


def _plain_hyphens(text: str) -> str:
    """Return a text with MusicBrainz's typographic hyphens as plain ones."""
    return text.replace("\u2010", "-").replace("\u2011", "-")
