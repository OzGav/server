"""
Helper utilities for the Metadata Controller.

Pure functions used by the controller and its mixins that do not need access to
the controller instance.
"""

from __future__ import annotations

import pathlib

from music_assistant_models.enums import Period

from music_assistant.helpers.datetime import utc

from .constants import (
    _IMAGEPROXY_CONTENT_TYPES,
    COMPOSER_PERIOD_START_YEARS,
    COMPOSER_WORKING_LIFE_END_OFFSET,
    COMPOSER_WORKING_LIFE_START_AGE,
)


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
