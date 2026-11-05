"""Helpers to read the work, movement and catalog numbers a classical title names."""

from __future__ import annotations

import re
from dataclasses import dataclass

# a movement title such as "I. Allegro", "3. Presto" or "No. 4. The Call" starts with its
# number and a full stop, one such as "Mvt. I-Allegro" or "Mvt. 3 Andantino" with "Mvt."
_NUMBERED_MOVEMENT = re.compile(r"^(?:(?:No\.\s)?([IVX]+|\d+)\.\s|Mvt\.\s([IVX]+|\d+)[-\s])")
_ROMAN_NUMERALS = (
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
    "BWV|BuxWV|CD|FP|GWV|Hob|HWV|Kk|KV|MWV|[Oo]pus|[Oo]p|RCT|RV|SV|Sz|TH|TrV|TWV|WAB|WD|WoO|WWV|ZT"
)
_SINGLE_LETTER_CATALOG_PREFIXES = "DHKLPSTZ"
# a catalog number such as "op. 8 no. 1", "K. 297c/299", "Hob. XVI:52" or "TWV 51:D7"
_CATALOG_NUMBER = re.compile(
    rf"(?:(?<=, )[{_SINGLE_LETTER_CATALOG_PREFIXES}]|\b(?:{_CATALOG_PREFIXES}))\.?\s?"
    r"(?:[IVXL]+:|\d+:[A-Z]?)?\d+[A-Za-z]*(?:/\d+[A-Za-z]*)?"
    r"(?:,?\s(?:[Nn]o\.|[Nn]r\.|[Nn]°)\s?\d+)?"
)
# the words naming a classical form, in English, German, French and Italian, where a word
# German joins to an instrument also counts at the end of a word, as in "Klaviersonate"
_FORM_WORD = re.compile(
    r"\b(?:symphony|symphonies|symphonie|symphonien|sinfonia|sinfonie|sinfonien|sinfonietta"
    r"|concerto|concertos|concerti|concertino|mass|missa|messe|requiem|suite|suites"
    r"|prelude|preludes|prélude|préludes|präludium|präludien|preludio|fugue|fugues|fuga|fuge"
    r"|fugen|overture|overtures|ouverture|ouvertüre|cantata|cantatas|kantate|kantaten"
    r"|oratorio|oratorium|motet|motets|motette|madrigal|madrigals|madrigale|madrigali"
    r"|partita|partitas|etude|etudes|étude|études|etüde|etüden|nocturne|nocturnes|notturno"
    r"|variation|variations|variationen|variazioni|fantasia|fantasie"
    r"|\w*(?:sonata|sonatas|sonate|sonaten|sonatina|quartet|quartets|quartett|quartette"
    r"|quartetto|quatuor|quintet|quintets|quintett|quintetto|konzert|konzerte))\b",
    re.IGNORECASE,
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


def names_classical_work(title: str) -> bool:
    """
    Return whether a title names a classical form or a catalog number.

    :param title: The title of a work, such as "Symphony No. 5" or "Suite, BWV 1068".
    """
    return _FORM_WORD.search(title) is not None or _CATALOG_NUMBER.search(title) is not None


def numeral_value(numeral: str) -> int | None:
    """
    Return the value of a number written in digits or as a Roman numeral from I to XX.

    :param numeral: A number such as "3" or "IV", with a Roman numeral in capitals.
    """
    if numeral.isdecimal():
        return int(numeral)
    return _ROMAN_NUMERALS.index(numeral) + 1 if numeral in _ROMAN_NUMERALS else None


def _movement_number(movement: str) -> int | None:
    """Return the number a movement title starts with, or None when it starts with none."""
    if not (match := _NUMBERED_MOVEMENT.match(movement)):
        return None
    return numeral_value(match.group(1) or match.group(2))
