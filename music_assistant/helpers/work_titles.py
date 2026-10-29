"""Helpers to read the work, movement and catalog numbers a classical title names."""

from __future__ import annotations

import re
from dataclasses import dataclass

# a movement title such as "I. Allegro", "3. Presto" or "No. 4. The Call" starts with its
# number and a full stop, one such as "Mvt. I-Allegro" or "Mvt. 3 Andantino" with "Mvt."
_NUMBERED_MOVEMENT = re.compile(r"^(?:(?:No\.\s)?([IVX]+|\d+)\.\s|Mvt\.\s([IVX]+|\d+)[-\s])")
ROMAN_NUMERALS = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)

# catalog prefixes such as "BWV" or "op.", the single letter ones (Köchel, Deutsch, ...)
# only after a comma so "Symphony no. 5 in D minor" or "Livre I" never count
_CATALOG_PREFIXES = (
    "BWV|BuxWV|CD|FP|GWV|Hob|HWV|Kk|KV|MWV|[Oo]pus|[Oo]p|RCT|RV|SV|Sz|TrV|TWV|WAB|WoO|WWV|ZT"
)
_SINGLE_LETTER_CATALOG_PREFIXES = "DHKLPSTZ"
# a catalog number such as "op. 8 no. 1", "K. 297c/299", "Hob. XVI:52" or "TWV 51:D7"
_CATALOG_NUMBER = re.compile(
    rf"(?:(?<=, )[{_SINGLE_LETTER_CATALOG_PREFIXES}]|\b(?:{_CATALOG_PREFIXES}))\.?\s?"
    r"(?:[IVXL]+:|\d+:[A-Z]?)?\d+[A-Za-z]*(?:/\d+[A-Za-z]*)?"
    r"(?:\s(?:[Nn]o\.|[Nn]r\.|[Nn]°)\s?\d+)?"
)


@dataclass(frozen=True)
class TrackWork:
    """The work a track belongs to, with the movement its work title names."""

    name: str
    mbid: str | None = None
    movement_name: str | None = None
    movement_number: int | None = None


def split_movement_title(title: str) -> TrackWork | None:
    """
    Return the parent work of a numbered movement title, or None for any other title.

    :param title: A title such as "Symphony No. 5 in C minor, Op. 67: I. Allegro con brio".
    """
    # MusicBrainz titles movement works "Parent: I. Movement" (see its classical style guide)
    parent, _, movement = title.rpartition(": ")
    if parent and (number := _movement_number(movement)):
        return TrackWork(parent, movement_name=movement, movement_number=number)
    return None


def parse_catalog_numbers(title: str) -> list[str]:
    """
    Return the catalog numbers a work title names, written as the title writes them.

    :param title: The title of a work, such as "Symphony no. 5 in C minor, op. 67".
    """
    return list(dict.fromkeys(match.group() for match in _CATALOG_NUMBER.finditer(title)))


def _movement_number(movement: str) -> int | None:
    """Return the number a movement title starts with, or None when it starts with none."""
    if not (match := _NUMBERED_MOVEMENT.match(movement)):
        return None
    numeral = match.group(1) or match.group(2)
    return int(numeral) if numeral.isdecimal() else _roman_number(numeral)


def _roman_number(numeral: str) -> int | None:
    """Return the value of a Roman numeral from I to XX, or None for any other text."""
    return ROMAN_NUMERALS.index(numeral) + 1 if numeral in ROMAN_NUMERALS else None
