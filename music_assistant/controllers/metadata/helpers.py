"""
Helper utilities for the Metadata Controller.

Pure functions used by the controller and its mixins that do not need access to
the controller instance.
"""

from __future__ import annotations

import pathlib
import unicodedata
from typing import TYPE_CHECKING

from .constants import _IMAGEPROXY_CONTENT_TYPES

if TYPE_CHECKING:
    from music_assistant_models.media_items import Artist

    from music_assistant.providers.musicbrainz.models import MusicBrainzArtist, MusicBrainzWork


def fill_artist_from_musicbrainz(artist: Artist, mb_artist: MusicBrainzArtist) -> None:
    """
    Fill an artist's empty entity type and life span in from its MusicBrainz artist.

    :param artist: The artist to fill in.
    :param mb_artist: Its MusicBrainz artist.
    """
    metadata = artist.metadata
    if not metadata.artist_entity_type and (entity_type := mb_artist.entity_type):
        metadata.artist_entity_type = entity_type
    if not metadata.life_span and (life_span := mb_artist.metadata_life_span):
        metadata.life_span = life_span


def musicbrainz_work_name(mb_work: MusicBrainzWork, language: str) -> str:
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
