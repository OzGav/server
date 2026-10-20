"""
Helper utilities for the Metadata Controller.

Pure functions used by the controller and its mixins that do not need access to
the controller instance.
"""

from __future__ import annotations

import pathlib
import unicodedata
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistEntityType, Period
from music_assistant_models.helpers import create_sort_name
from music_assistant_models.media_items.metadata import LifeSpan

from music_assistant.helpers.datetime import utc

from .constants import (
    _IMAGEPROXY_CONTENT_TYPES,
    COMPOSER_PERIOD_START_YEARS,
    COMPOSER_WORKING_LIFE_END_OFFSET,
    COMPOSER_WORKING_LIFE_START_AGE,
)

if TYPE_CHECKING:
    from music_assistant_models.media_items import Artist

    from music_assistant.providers.musicbrainz.models import MusicBrainzArtist


def composer_period(birth_year: int, death_year: int | None) -> Period:
    """
    Return the classical period of a composer, judged by the middle of their working life.

    :param birth_year: The year the composer was born.
    :param death_year: The year the composer died, None for a living composer.
    """
    start = birth_year + COMPOSER_WORKING_LIFE_START_AGE
    # a living composer is still working
    end = death_year - COMPOSER_WORKING_LIFE_END_OFFSET if death_year else utc().year
    midpoint = (start + end) // 2
    return next(
        (period for year, period in reversed(COMPOSER_PERIOD_START_YEARS) if midpoint >= year),
        Period.MEDIEVAL,
    )


def fill_artist_from_musicbrainz(artist: Artist, mb_artist: MusicBrainzArtist) -> None:
    """Fill an artist's empty entity type, life span and country in from its MusicBrainz artist."""
    metadata = artist.metadata
    if not metadata.artist_entity_type and mb_artist.type:
        entity_type = ArtistEntityType(mb_artist.type.lower())
        if entity_type != ArtistEntityType.UNKNOWN:
            metadata.artist_entity_type = entity_type
    if not metadata.life_span and (life_span := mb_artist.life_span):
        metadata.life_span = LifeSpan(
            begin=life_span.begin, end=life_span.end, ended=life_span.ended
        )
    if not metadata.country:
        metadata.country = mb_artist.country_code


def musicbrainz_name(mb_artist: MusicBrainzArtist, language: str = "en") -> tuple[str, str] | None:
    """
    Return the name and sort name MusicBrainz gives an artist in Latin script.

    A main name in another script gives way to the primary alias in the given language, else to
    the primary English alias, else to an English artist name alias. A Latin main name gives
    way to such a primary alias that extends it, such as "Johann Strauss II". None when
    MusicBrainz has no such name.

    :param mb_artist: The MusicBrainz artist with its aliases.
    :param language: The language to name the artist in first, such as "de".
    """
    aliases = [x for x in mb_artist.aliases or [] if _is_latin(x.name)]
    primaries = [
        x
        for locale in dict.fromkeys((language, "en"))
        for x in aliases
        if x.primary and x.locale == locale
    ]
    if _is_latin(mb_artist.name):
        name, sort_name = mb_artist.name, mb_artist.sort_name
        # a namesake father and son share the main name, only their alias tells them apart
        if alias := next((x for x in primaries if x.name.startswith(f"{mb_artist.name} ")), None):
            name, sort_name = alias.name, alias.sort_name or mb_artist.sort_name
    else:
        alias = next(iter(primaries), None) or next(
            (x for x in aliases if x.locale == "en" and (x.type or "").lower() == "artist name"),
            None,
        )
        if not alias:
            return None
        name = alias.name
        sort_name = alias.sort_name or (
            mb_artist.sort_name if _is_latin(mb_artist.sort_name) else ""
        )
    name = _plain_hyphens(name)
    return name, (_plain_hyphens(sort_name) if sort_name else create_sort_name(name))


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


def _is_latin(text: str) -> bool:
    """Return whether every letter of a text is a Latin letter."""
    return all(unicodedata.name(x, "").startswith("LATIN") for x in text if x.isalpha())


def _plain_hyphens(text: str) -> str:
    """Return a text with MusicBrainz's typographic hyphens as plain ones."""
    return text.replace("\u2010", "-").replace("\u2011", "-")
