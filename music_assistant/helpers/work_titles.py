"""Helpers to read the work, movement and catalog numbers a classical title names."""

from __future__ import annotations

import re
from dataclasses import dataclass

from music_assistant_models.helpers import create_safe_string

# a movement title such as "I. Allegro", "3. Presto" or "No. 4. The Call" starts with its
# number and a full stop, one such as "Mvt. I-Allegro" or "Mvt. 3 Andantino" with "Mvt."
_NUMBERED_MOVEMENT = re.compile(r"^(?:(?:No\.\s)?([IVX]+|\d+)\.\s|Mvt\.\s([IVX]+|\d+)[-\s])")
# the number opening a movement, as in "XXX. Aria", "no. 44. Chorus", "no. 9: Finale",
# "38a. Evangelista" or "N°13b Aria" but not "No. 1 in A major", with any scene before it, as
# in "Scena 2. no. 15. ..." or "Картина III, no. 14a. ..."
_NUMBER_WORD = r"(?:[Nn][or]\.|N[°º])\s?"
_OPENING_MOVEMENT_NUMBER = re.compile(
    r"^(?:(?:[Ss]cen[ae]|Картина) (?:\d+|[IVXL]+)[.,]\s)?"
    rf"(?:(?:{_NUMBER_WORD})?(?:\d{{1,3}}[A-Za-z]?|[IVXL]+[a-e]?)[.,:]\s"
    rf"|{_NUMBER_WORD}\d{{1,3}}[A-Za-z]?\s(?!in\s))"
)
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

# the words naming an act or part of a work, the numbered ones with the kind of part they name
_NUMBERED_PART_WORDS = {"Act": "act", "Part": "part", "Pt.": "part"}
# the German, Italian and French words for an act or part, only read where one closes a title
_CLOSING_PART_WORDS = {
    **_NUMBERED_PART_WORDS,
    "Teil": "part",
    "Atto": "act",
    "Acte": "act",
    "Akt": "act",
    "Aufzug": "act",
}
_UNNUMBERED_PART_WORDS = ("Anhang", "Appendix")
# an act or part closing a work title, only followed by a scene of the act and catalog
# numbers, such as "La bohème, Act II", "Messiah, HWV 56, Pt. 2", "Swan Lake, Act II, Op. 20,
# TH 12", "The Nutcracker, Act II, Scene 3, Op. 71", "La bohème, SC 67 / Act 1" or
# MusicBrainz's "Turandot: Atto III"
_ACT_OR_PART = re.compile(
    rf"^(?P<work>.+)(?:, | / |: )(?:(?P<word>{'|'.join(map(re.escape, _CLOSING_PART_WORDS))}) "
    rf"(?P<number>[IVX]+|\d+)(?:, (?:Scene|[Ss]cena|[Ss]cène) (?P<scene>[IVX]+|\d+))?"
    rf"|(?P<label>{'|'.join(_UNNUMBERED_PART_WORDS)}))"
    rf"(?P<catalog>(?:, {_CATALOG_NUMBER.pattern})*)$"
)
# the subtitle right after the number of an act or part, as in "Luisa Miller: Atto II.
# “L'intrigo”" or "Don Carlo: Atto I (II)"
_PART_SUBTITLE = re.compile(
    rf"(?P<part>\b(?:{'|'.join(map(re.escape, _CLOSING_PART_WORDS))}) (?:[IVX]+|\d+))"
    r"(?:\.\s.+|\s\(.+\))$"
)
# a name starting with a word and a number, such as "Variation No. 1", "Var. 1" or "Act II"
_WORD_AND_NUMBER = re.compile(r"^([^\W\d_]{3,})\.?\s+(?:(?i:no|nr)\.?\s*|n°\s*)?(\d+|[IVX]+)\b")


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


def part_key(work: str) -> tuple[str, int | str] | None:
    """
    Return what tells an act or part apart from the other parts of its work, or None.

    The same act or part written differently, such as "Part 1", "Part I" and "Pt. 1", has
    one key. An unnumbered part, such as an "Anhang", is told apart by its label.

    :param work: The name of a work, such as "Messiah, HWV 56, Pt. 2".
    """
    if not (match := _ACT_OR_PART.match(work)):
        return None
    if match["label"]:
        return "label", create_safe_string(match["label"], True, True)
    value = _numeral_value(match["number"])
    return (_CLOSING_PART_WORDS[match["word"]], value) if value else None


def part_label(work: str) -> str | None:
    """
    Return how the act or part a work name ends with is shown, or None for any other work.

    The act or part is named in English with a Roman number, keeping its scene and leaving out
    its subtitle, so "Tosca, SC 69 / Act 3" gives "Act III" and MusicBrainz's "Luisa Miller:
    Atto II. “L'intrigo”" gives "Act II".

    :param work: The name of a work, such as "Swan Lake, Act II, Scene 1, Op. 20".
    """
    match = _ACT_OR_PART.match(work) or _ACT_OR_PART.match(without_part_subtitle(work))
    if not match or not match["word"]:
        return None
    kind, value = _CLOSING_PART_WORDS[match["word"]], _numeral_value(match["number"])
    if not value or value > len(ROMAN_NUMERALS):
        return None
    label = f"{kind.capitalize()} {ROMAN_NUMERALS[value - 1]}"
    return f"{label}, Scene {match['scene']}" if match["scene"] else label


def without_part_subtitle(work: str) -> str:
    """
    Return the name of an act or part without the subtitle after its number.

    MusicBrainz's "Luisa Miller: Atto II. “L'intrigo”" gives "Luisa Miller: Atto II" and
    "Don Carlo: Atto I (II)" gives "Don Carlo: Atto I". Any other name comes back as it is.

    :param work: The name of a work, such as "La bohème: Atto I. In soffitta".
    """
    return _PART_SUBTITLE.sub(r"\g<part>", work)


def word_and_number(name: str) -> tuple[str, int] | None:
    """
    Return the word and number a name starts with, or None when it starts otherwise.

    The word comes lowercase and without accents, so "Variation No. 1" gives "variation"
    and 1, and "Variatio IV." gives "variatio" and 4.

    :param name: The name of a movement or part, such as "Var. 1".
    """
    if not (match := _WORD_AND_NUMBER.match(name)):
        return None
    numeral = match.group(2)
    number = int(numeral) if numeral.isdecimal() else _roman_number(numeral)
    return (create_safe_string(match.group(1), True, True), number) if number else None


def without_movement_number(name: str) -> str:
    """
    Return a movement name without the number it opens with, such as "Christe" for "02. Christe".

    A scene before the number goes with it, so "Scene II. No. 2 Aria" gives "Aria".

    :param name: The name of a movement, such as a track's "No. 3, O mio babbino caro".
    """
    return _OPENING_MOVEMENT_NUMBER.sub("", name, count=1) or name


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


def _numeral_value(numeral: str) -> int | None:
    """Return the value of a number written in digits or as a Roman numeral up to XX."""
    return int(numeral) if numeral.isdecimal() else _roman_number(numeral)


def _roman_number(numeral: str) -> int | None:
    """Return the value of a Roman numeral from I to XX, or None for any other text."""
    return ROMAN_NUMERALS.index(numeral) + 1 if numeral in ROMAN_NUMERALS else None
