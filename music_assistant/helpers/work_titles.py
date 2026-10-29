"""Helpers to read the work, movement and catalog numbers a classical title names."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Credit, Work

from music_assistant.helpers.compare import compare_strings, is_named_in

if TYPE_CHECKING:
    from collections.abc import Sequence

    from music_assistant_models.media_items import Artist, Track

# credit wording in a joined name, such as "Beethoven & Conducted By William Bowles"
_CREDIT_WORDING = re.compile(
    r"\b(?:conducted|performed|arranged|orchestrated|directed)\s+by\b", re.IGNORECASE
)

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
    "BWV|BuxWV|CD|FP|GWV|Hob|HWV|Kk|KV|MWV|[Oo]pus|[Oo]p|RCT|RV|SV|Sz|TH|TrV|TWV|WAB|WD|WoO|WWV|ZT"
)
_SINGLE_LETTER_CATALOG_PREFIXES = "DHKLPSTZ"
# a catalog number such as "op. 8 no. 1", "K. 297c/299", "Hob. XVI:52" or "TWV 51:D7"
_CATALOG_NUMBER = re.compile(
    rf"(?:(?<=, )[{_SINGLE_LETTER_CATALOG_PREFIXES}]|\b(?:{_CATALOG_PREFIXES}))\.?\s?"
    r"(?:[IVXL]+:|\d+:[A-Z]?)?\d+[A-Za-z]*(?:/\d+[A-Za-z]*)?"
    r"(?:,?\s(?:[Nn]o\.|[Nn]r\.|[Nn]°)\s?\d+)?"
)

# a movement marked as such, as in "Mvt. III-Rondo" or "Mvt. 3 The Young Prince"
_MARKED_MOVEMENT = re.compile(r"Mvt\.\s(?:[IVX]+|\d+)[-\s]")
# what ends the work of a "Work: Piece" title: a colon followed by a space or straight by the
# quoted text of an aria ('La Bohème, Act I:"Che gelida manina"'), so catalog numbers such as
# "Hob. I:94" are never split, or a comma followed by a marked movement, as in "Scheherazade,
# Op. 35, Mvt. 3 The Young Prince"
_WORK_SEPARATOR = re.compile(rf':(?:\s+|(?=["“]))|,\s+(?={_MARKED_MOVEMENT.pattern})')
# a catalog number followed by the part of its work, as in MusicBrainz's "Messiah, HWV 56:
# Part II, no. 44. Chorus"
_CATALOGUED_PART = re.compile(
    rf"(?:{_CATALOG_NUMBER.pattern})(?P<separator>{_WORK_SEPARATOR.pattern})(?=\S)"
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
# an act or part opening the text after a work, as in "Act 1, No. 2 Waltz"
_LEADING_PART = re.compile(
    rf"^(?P<part>(?:{'|'.join(map(re.escape, _NUMBERED_PART_WORDS))}) (?:[IVX]+|\d+))(?:, |: )"
)
# a name starting with a word and a number, such as "Variation No. 1", "Var. 1" or "Act II"
_WORD_AND_NUMBER = re.compile(r"^([^\W\d_]{3,})\.?\s+(?:(?i:no|nr)\.?\s*|n°\s*)?(\d+|[IVX]+)\b")
# a work and its numbered movement joined by a dash, as in "Miserere - 01. Miserere Mei, Deus"
_DASHED_MOVEMENT = re.compile(r"^(.+?) - ((?:No\.\s)?(?:\d+[a-z]?|[IVX]+)\.\s.+)$")
# a numbered piece after a comma or dash, as in "Erster Teil, 9a. Evangelista" or "Miserere -
# 01. Miserere Mei", whose own text may hold a colon
_NUMBERED_PIECE_WITHIN = re.compile(r"(?:, | - )\d+[a-z]?\.\s")
# an act or part of a work after a comma or slash, as in "La bohème, Act II, Scene 1"
_WORK_PART = re.compile(
    rf"(?:, | / )(?:{'|'.join(map(re.escape, _NUMBERED_PART_WORDS))}) (?:[IVX]+|\d+)\b"
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
# a chapter or episode of an audiobook or radio play, such as "Kapitel 1" or "Folge 5"
_CHAPTER = re.compile(
    r"(?:chapter|kapitel|teil|folge|part|episode)\s+(?:(?:no|nr)\.?\s*)?"
    r"(?:\d+(?:\.\d+)*|[ivxlc]+)\.?",
    re.IGNORECASE,
)
# a chapter or episode numbered before its word, such as "1. Kapitel" or "Nr. 3. Folge"
_NUMBERED_CHAPTER = re.compile(
    r"(?:(?:no|nr)\.?\s*)?(?:\d+|[ivxlc]+)\.\s*(?:chapter|kapitel|folge|episode)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrackWork:
    """The work a track belongs to, with the movement its work title names."""

    name: str
    mbid: str | None = None
    movement_name: str | None = None
    movement_number: int | None = None
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


def parse_work_title(
    title: str, composer: str | None = None, classical: bool = False
) -> TrackWork | None:
    """
    Return the work a "Work: Piece" title names, or None for any other title.

    The text after the last colon is the movement, numbered when it starts with a numeral, but
    a work ends at a catalog number followed by a colon, as in "Swan Lake, Op. 20: Act 1, No. 2
    Waltz: Tempo di valse", and an act or part opening the rest is part of the work, as in
    "Lakmé (Opera): Act 1, Flower Duet". A leading segment naming the composer is left out of
    the work, and a work that is an act or part, such as "La bohème, Act II", comes with the
    work it is part of. In a "Composer: Work - 1. Movement" title the leading segment is the
    composer, returned as the other composer when it is not the given one.

    A title of a track not known to be classical names a work only when it also has a catalog
    number, an act or part, a numbered movement or a leading segment naming the composer. A
    classical form alone, as in "Suite: Judy Blue Eyes", is not enough. A title starting or
    ending with a chapter, such as "Kapitel 1: Der Anfang" or "Der Hobbit: 1. Kapitel", never
    names a work.

    :param title: A track title such as "Symphony No. 5 in C Minor, Op. 67: I. Allegro con brio".
    :param composer: The name of the track's composer, matched against a leading segment.
    :param classical: Whether the track is known to be classical, such as by its genre or album.
    """
    segments = _split_work_title(title)
    if len(segments) < 2 or not _names_work(title, segments, composer, classical):
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
        if catalogued := _CATALOGUED_PART.search(work):
            work, movement = _catalogued_work_and_part(work, catalogued, movement)
        elif leading := _LEADING_PART.match(movement):
            # "Lakmé (Opera): Act 1, Flower Duet" is the act "Lakmé (Opera), Act 1"
            work, movement = f"{work}, {leading['part']}", movement[leading.end() :]
    return work_with_movement(work, movement)


def is_work_title(title: str) -> bool:
    """
    Return whether a title has the "Work: Piece" form most titles of a classical album have.

    A classical form, as in "Trout Quintet: Andante", counts here, as most titles of an album
    having it is a sign the album is classical, where one title having it is not.

    :param title: A track title such as "Trout Quintet: Andante".
    """
    segments = _split_work_title(title)
    return len(segments) > 1 and _names_work(
        title, segments, None, classical=_FORM_WORD.search(title) is not None
    )


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
        if (work_title := parse_work_title(title, first_artist, classical=True))
    ]
    return sum(1 for x in work_titles if x.other_composer) * 2 <= len(work_titles)


def work_with_movement(work: str, movement: str | None = None) -> TrackWork:
    """
    Return a work and movement named apart.

    :param work: The name of the work, such as "La bohème, Act II".
    :param movement: The title of the movement, such as "I. Allegro con brio".
    """
    return TrackWork(
        work,
        movement_name=movement,
        movement_number=_movement_number(movement) if movement else None,
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


def names_classical_work(title: str) -> bool:
    """
    Return whether a title names a classical form or a catalog number.

    :param title: The title of a work, such as "Symphony No. 5" or "Suite, BWV 1068".
    """
    return _FORM_WORD.search(title) is not None or _CATALOG_NUMBER.search(title) is not None


def set_track_work(track: Track, track_work: TrackWork) -> None:
    """
    Set the work and movement a provider track names.

    The library only adds a work named this way when the track carries its MusicBrainz work id.

    :param track: The provider track.
    :param track_work: The work and movement the track names.
    """
    track.movement_name = track_work.movement_name
    track.movement_number = track_work.movement_number
    track.work = Work(
        item_id=track_work.name,
        provider=track.provider,
        name=track_work.name,
        provider_mappings=set(),
    )


def set_listed_composer_work(
    track: Track,
    title: str,
    first_artist: Artist | None,
    composer_first: bool | None,
    classical: bool = False,
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
    :param classical: Whether the track is known to be classical, as on a classical album.
    """
    composer = first_artist
    track_work = parse_work_title(
        title, composer.name if composer else None, classical or composer_first is not None
    )
    if composer_first is False or (track_work and track_work.other_composer):
        # the titles name the composers, so the first artist is a performer
        composer = None
    if composer and (track_work or composer_first):
        track.credits.append(Credit(artist=composer, role=ArtistRole.COMPOSER))
    if track_work:
        set_track_work(track, track_work)


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


def _catalogued_work_and_part(
    work: str, catalogued: re.Match[str], movement: str
) -> tuple[str, str]:
    """Return the work ending at its catalog number and the movement, with an act it opens with."""
    part = f"{work[catalogued.end() :]}: {movement}"
    work = work[: catalogued.start("separator")]
    if leading := _LEADING_PART.match(part):
        # "Swan Lake, Op. 20: Act 1, No. 2 Waltz" is the act "Swan Lake, Op. 20, Act 1"
        work, part = f"{work}, {leading['part']}", part[leading.end() :]
    return work, part


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


def _split_work_title(title: str) -> list[str]:
    """Return the non-empty segments a title's work separators outside brackets divide it in."""
    segments: list[str] = []
    start = 0
    for match in _WORK_SEPARATOR.finditer(title):
        before = title[: match.start()]
        # a colon inside brackets belongs to an annotation, as in "(Excerpt: Opening)"
        if before.count("(") > before.count(")") or before.count("[") > before.count("]"):
            continue
        segment = title[start : match.start()]
        # a colon after the number of a piece, or in a marked movement, belongs to the text of
        # that piece, as in "Mvt. III-Rondo: Allegro"
        if _NUMBERED_PIECE_WITHIN.search(segment) or _MARKED_MOVEMENT.match(segment.strip()):
            break
        segments.append(segment.strip())
        start = match.end()
    segments.append(title[start:].strip())
    return [segment for segment in segments if segment]


def _names_work(title: str, segments: list[str], composer: str | None, classical: bool) -> bool:
    """Return whether a title, split in its segments, names a work rather than a chapter or song."""
    names_composer = bool(composer and _names_composer(segments[0], composer))
    movement = segments[-1]
    if _CHAPTER.fullmatch(segments[1] if names_composer else segments[0]) or (
        _NUMBERED_CHAPTER.match(movement)
    ):
        return False
    return (
        classical
        or names_composer
        or _CATALOG_NUMBER.search(title) is not None
        or _WORK_PART.search(title) is not None
        or _movement_number(movement) is not None
        or _NUMBERED_PIECE_WITHIN.search(movement) is not None
    )


def _names_composer(segment: str, composer: str) -> bool:
    """Return whether a title segment is a composer's name or surname, as in "Beethoven"."""
    return (
        compare_strings(segment, composer)
        or is_named_in(segment, composer)
        or compare_strings(segment.rsplit(maxsplit=1)[-1], composer.rsplit(maxsplit=1)[-1])
    )
