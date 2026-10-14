"""Helpers to read the work, movement and catalog numbers a classical title names."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole, MediaType
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Credit, ItemMapping, Work
from music_assistant_models.unique_list import UniqueList

from music_assistant.helpers.compare import compare_strings, is_named_in

if TYPE_CHECKING:
    from collections.abc import Sequence

    from music_assistant_models.media_items import Artist, Track

# credit wording in a joined name, such as "Beethoven & Conducted By William Bowles"
_CREDIT_WORDING = re.compile(
    r"\b(?:conducted|performed|arranged|orchestrated|directed)\s+by\b", re.IGNORECASE
)

# a movement title such as "I. Allegro", "3. Presto" or "No. 4. The Call" starts with its
# number and a full stop
_NUMBERED_MOVEMENT = re.compile(r"^(?:No\.\s)?([IVX]+|\d+)\.\s")
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
    "BWV|BuxWV|CD|FP|GWV|Hob|HWV|Kk|KV|MWV|[Oo]pus|[Oo]p|RCT|RV|SV|Sz|TH|TrV|TWV|WAB|WoO|WWV|ZT"
)
_SINGLE_LETTER_CATALOG_PREFIXES = "DHKLPSTZ"
# a catalog number such as "op. 8 no. 1", "K. 297c/299", "Hob. XVI:52" or "TWV 51:D7"
_CATALOG_NUMBER = re.compile(
    rf"(?:(?<=, )[{_SINGLE_LETTER_CATALOG_PREFIXES}]|\b(?:{_CATALOG_PREFIXES}))\.?\s?"
    r"(?:[IVXL]+:|\d+:[A-Z]?)?\d+[A-Za-z]*(?:/\d+[A-Za-z]*)?"
    r"(?:,?\s(?:[Nn]o\.|[Nn]r\.|[Nn]°)\s?\d+)?"
)
# the number of a piece within its opus closing a catalog number, as in "Op. 48, No. 2"
_PIECE_NUMBER = re.compile(r",?\s(?:[Nn]o\.|[Nn]r\.|[Nn]°)\s?(\d+)$")

# the colon between the work and the movement of a "Work: Piece" title, followed by a space
# or straight by the quoted text of an aria ('La Bohème, Act I:"Che gelida manina"'), so
# catalog numbers such as "Hob. I:94" are never split
_WORK_SEPARATOR = re.compile(r':(?:\s+|(?=["“]))')
# the words naming an act or part of a work, the numbered ones with the kind of part they name
_NUMBERED_PART_WORDS = {"Act": "act", "Part": "part", "Pt.": "part"}
_UNNUMBERED_PART_WORDS = ("Anhang", "Appendix")
# an act or part closing a work title, only followed by catalog numbers, such as
# "La bohème, Act II", "Messiah, HWV 56, Pt. 2", "Swan Lake, Act II, Op. 20, TH 12" or
# "La bohème, SC 67 / Act 1"
_ACT_OR_PART = re.compile(
    rf"^(?P<work>.+)(?:, | / )(?:(?P<word>{'|'.join(map(re.escape, _NUMBERED_PART_WORDS))}) "
    rf"(?P<number>[IVX]+|\d+)|(?P<label>{'|'.join(_UNNUMBERED_PART_WORDS)}))"
    rf"(?P<catalog>(?:, {_CATALOG_NUMBER.pattern})*)$"
)
# a name starting with a word and a number, such as "Variation No. 1", "Var. 1" or "Act II"
_WORD_AND_NUMBER = re.compile(r"^([^\W\d_]{3,})\.?\s+(?:(?i:no|nr)\.?\s*|n°\s*)?(\d+|[IVX]+)\b")
# a work and its numbered movement joined by a dash, as in "Miserere - 01. Miserere Mei, Deus"
_DASHED_MOVEMENT = re.compile(r"^(.+?) - ((?:No\.\s)?(?:\d+[a-z]?|[IVX]+)\.\s.+)$")


@dataclass(frozen=True)
class TrackWork:
    """The work a track belongs to, with the movement its work title names."""

    name: str
    mbid: str | None = None
    movement_name: str | None = None
    movement_number: int | None = None
    parent_name: str | None = None  # the work this work is a part of, such as an opera
    other_composer: str | None = None  # a composer the title names who is not the given one


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


def parse_work_title(title: str, composer: str | None = None) -> TrackWork | None:
    """
    Return the work a "Work: Piece" title names, or None for any other title.

    The text after the last colon is the movement, numbered when it starts with a numeral. A
    leading segment naming the composer is left out of the work, and a work that is an act or
    part, such as "La bohème, Act II", comes with the work it is part of. In a "Composer: Work -
    1. Movement" title the leading segment is the composer, returned as the other composer
    when it is not the given one.

    :param title: A track title such as "Symphony No. 5 in C Minor, Op. 67: I. Allegro con brio".
    :param composer: The name of the track's composer, matched against a leading segment.
    """
    segments = _split_work_title(title)
    if len(segments) < 2:
        return None
    if len(segments) == 2 and (dashed := _DASHED_MOVEMENT.match(segments[1])):
        # a "Composer: Work - 1. Movement" title
        other = None if composer and _names_composer(segments[0], composer) else segments[0]
        return TrackWork(
            dashed.group(1),
            movement_name=dashed.group(2),
            movement_number=_movement_number(dashed.group(2)),
            other_composer=other,
        )
    if composer and _names_composer(segments[0], composer):
        segments = segments[1:]
    if len(segments) == 1:
        # a "Composer: Work" title names the work alone
        work, movement = segments[0], None
    else:
        work, movement = ": ".join(segments[:-1]), segments[-1]
    return work_with_movement(work, movement)


def composers_listed_first(tracks: Sequence[tuple[str, str | None]]) -> bool:
    """
    Return whether the tracks of a classical album list each track's composer as first artist.

    Some labels name the composer in the title instead ("Allegri: Miserere - 01. Miserere Mei,
    Deus") and list only the performers.

    :param tracks: The title and first artist name of each track of the album.
    """
    work_titles = [
        work_title
        for title, first_artist in tracks
        if (work_title := parse_work_title(title, first_artist))
    ]
    return sum(1 for x in work_titles if x.other_composer) * 2 <= len(work_titles)


def work_with_movement(work: str, movement: str | None = None) -> TrackWork:
    """
    Return a work and movement named apart, with the work it is part of for an act or part.

    :param work: The name of the work, such as "La bohème, Act II".
    :param movement: The title of the movement, such as "I. Allegro con brio".
    """
    return TrackWork(
        work,
        movement_name=movement,
        movement_number=_movement_number(movement) if movement else None,
        parent_name=parent_work_name(work),
    )


def parent_work_name(work: str) -> str | None:
    """
    Return the name of the work an act or part is part of, or None for any other work.

    The catalog numbers after the act or part belong to that work.

    :param work: The name of a work, such as "Swan Lake, Act II, Op. 20, TH 12".
    """
    if not (match := _ACT_OR_PART.match(work)):
        return None
    return match["work"] + match["catalog"]


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
    number = match["number"]
    value = int(number) if number.isdecimal() else _roman_number(number)
    return (_NUMBERED_PART_WORDS[match["word"]], value) if value else None


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


def parse_catalog_numbers(title: str) -> list[str]:
    """
    Return the catalog numbers a work title names, written as the title writes them.

    :param title: The title of a work, such as "Symphony no. 5 in C minor, op. 67".
    """
    return list(dict.fromkeys(match.group() for match in _CATALOG_NUMBER.finditer(title)))


def numbered_piece(work: str) -> tuple[str, int, str | None] | None:
    """
    Return the opus, number and name of a work titled as a numbered piece of an opus, or None.

    "Piano Sonata No. 11 in A major, K 331, No. 3 Rondo Alla Turca" gives "K 331", 3 and
    "Rondo Alla Turca". The name is None when the title names none after the number.

    :param work: The name of a work.
    """
    numbered = [
        (catalog, piece)
        for catalog in _CATALOG_NUMBER.finditer(work)
        if (piece := _PIECE_NUMBER.search(catalog.group()))
    ]
    if len(numbered) != 1:
        return None
    [(catalog, piece)] = numbered
    name = work[catalog.end() :].strip(" ,.-")
    # text that is another catalog number names no piece
    return (
        catalog.group()[: piece.start()],
        int(piece.group(1)),
        name if name and not parse_catalog_numbers(name) else None,
    )


def set_track_work(track: Track, track_work: TrackWork, composer: Artist | None) -> None:
    """
    Set the work and movement a provider track names, for the library to add with the track.

    :param track: The provider track.
    :param track_work: The work and movement the track names.
    :param composer: The composer of the work, if known.
    """
    track.movement_name = track_work.movement_name
    track.movement_number = track_work.movement_number
    parent_name = track_work.parent_name
    # the library adds the work, and the work it is part of, once the track is added to it
    track.work = Work(
        item_id=track_work.name,
        provider=track.provider,
        name=track_work.name,
        provider_mappings=set(),
        composers=UniqueList([composer] if composer else []),
        catalog_numbers=parse_catalog_numbers(track_work.name),
        parent_work=ItemMapping(
            media_type=MediaType.WORK,
            item_id=parent_name,
            provider=track.provider,
            name=parent_name,
        )
        if parent_name
        else None,
    )


def set_listed_composer_work(
    track: Track, title: str, first_artist: Artist | None, composer_first: bool | None
) -> None:
    """
    Set the work and movement a classical track's title names, with its first artist as composer.

    The first artist is credited as composer when the title names a work, or when the album
    lists each track's composer first, unless the titles name the composers instead.

    :param track: The provider track.
    :param title: The title of the track, without its version.
    :param first_artist: The first artist the track lists.
    :param composer_first: True on a classical album that lists each track's composer as its
        first artist, False on one whose titles name the composers instead, None when unknown.
    """
    composer = first_artist
    track_work = parse_work_title(title, composer.name if composer else None)
    if composer_first is False or (track_work and track_work.other_composer):
        # the titles name the composers, so the first artist is a performer
        composer = None
    if composer and (track_work or composer_first):
        track.credits.append(Credit(artist=composer, role=ArtistRole.COMPOSER))
    if track_work:
        set_track_work(track, track_work, composer)


def joined_composer_name(name: str, title: str) -> str | None:
    """
    Return the composer a joined artist name starts with, or None when it names no composer.

    A name such as "Beethoven & Conducted By William Bowles & Sarah Ainsworth" starts with the
    composer when the rest of it credits performers by their part, or when the title starts
    with the composer, as in "Chopin & Sarah Ainsworth" for "Chopin: Nocturne No. 2".

    :param name: The artist name, such as "Frédéric Chopin & Sarah Ainsworth".
    :param title: The title of the track the artist is credited on.
    """
    first, separator, rest = name.partition(" & ")
    if not separator or not (first := first.strip()):
        return None
    if _CREDIT_WORDING.search(rest):
        return first
    segments = _split_work_title(title)
    if len(segments) > 1 and _names_composer(segments[0], first):
        return first
    return None


def _movement_number(movement: str) -> int | None:
    """Return the number a movement title starts with, or None when it starts with none."""
    if not (match := _NUMBERED_MOVEMENT.match(movement)):
        return None
    numeral = match.group(1)
    return int(numeral) if numeral.isdecimal() else _roman_number(numeral)


def _roman_number(numeral: str) -> int | None:
    """Return the value of a Roman numeral from I to XX, or None for any other text."""
    return _ROMAN_NUMERALS.index(numeral) + 1 if numeral in _ROMAN_NUMERALS else None


def _split_work_title(title: str) -> list[str]:
    """Return the non-empty segments a title's work separators outside brackets divide it in."""
    segments: list[str] = []
    start = 0
    for match in _WORK_SEPARATOR.finditer(title):
        before = title[: match.start()]
        # a colon inside brackets belongs to an annotation, as in "(Excerpt: Opening)"
        if before.count("(") > before.count(")") or before.count("[") > before.count("]"):
            continue
        segments.append(title[start : match.start()].strip())
        start = match.end()
    segments.append(title[start:].strip())
    return [segment for segment in segments if segment]


def _names_composer(segment: str, composer: str) -> bool:
    """Return whether a title segment is a composer's name or surname, as in "Beethoven"."""
    return (
        compare_strings(segment, composer)
        or is_named_in(segment, composer)
        or compare_strings(segment.rsplit(maxsplit=1)[-1], composer.rsplit(maxsplit=1)[-1])
    )
