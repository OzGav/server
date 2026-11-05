"""Tests for reading the work and movement a classical track title names."""

from __future__ import annotations

import pytest
from music_assistant_models.enums import WorkType

from music_assistant.helpers.work_titles import (
    joined_composer_name,
    parent_work_name,
    parse_catalog_numbers,
    part_key,
    part_label,
    title_work_type,
    without_movement_number,
    word_and_number,
)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Symphony No. 5 in C minor, Op. 67", ["Op. 67"]),
        ("Nocturne Op. 9 No. 2", ["Op. 9 No. 2"]),
        ("Nocturne In F Sharp Major, Op 15, No. 2", ["Op 15, No. 2"]),
        ('Concerto in E major, op. 8 no. 1, RV 269 "La primavera"', ["op. 8 no. 1", "RV 269"]),
        ("Concerto in C major for Flute, Harp & Orchestra, K. 297c/299", ["K. 297c/299"]),
        ("Concerto for Violin & Oboe in C minor, BWV 1060R", ["BWV 1060R"]),
        ("Excerpt from Die Walküre, WWV 86B: Akt III, Scene I, Walkürenritt", ["WWV 86B"]),
        ("Romeo and Juliet: Suite no. 2, op. 64ter", ["op. 64ter"]),
        ("Symphonie pour orgue n° 5 en fa mineur, op. 42 n° 1", ["op. 42 n° 1"]),
        ('Bagatelle for Piano in A minor, WoO 59 "Für Elise": Poco moto', ["WoO 59"]),
        ('Kantate, BWV 147 "Herz und Mund und Tat und Leben": Teil II', ["BWV 147"]),
        ('Symfonie č. 9 e moll, op. 95 "Z Nového světa"', ["op. 95"]),
        ('Symphony no. 1 in D major "Titan"', []),
        ("Five Mystical Songs", []),
        ("Messe solennelle en la majeur, op. 12 ", ["op. 12"]),
        ("Winterreise, op. 89, D. 911", ["op. 89", "D. 911"]),
        ("Piano Sonata in E-flat major, Hob. XVI:52", ["Hob. XVI:52"]),
        ("Overture in D major, TWV 55:D18", ["TWV 55:D18"]),
        ("Années de pèlerinage, S. 160", ["S. 160"]),
        ("Dido and Aeneas, Z. 626", ["Z. 626"]),
        ("Swan Lake, Op. 20, TH 12", ["Op. 20", "TH 12"]),
        ("Carmen, WD 31", ["WD 31"]),
    ],
)
def test_parse_catalog_numbers(title: str, expected: list[str]) -> None:
    """
    Catalog numbers are read off a work title as written, with the number within an opus.

    Single letter catalogs count only after a comma.
    """
    assert parse_catalog_numbers(title) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Matthäus-Passion, BWV 244, Part 1", "St. Matthew Passion, BWV 244, Part I"),
        ("Messiah, HWV 56, Pt. 2", "Messiah, HWV 56, Part II"),
        ("La bohème, SC 67 / Act 1", "La bohème, SC 67, Act I"),
        ("The Nutcracker, Act II, Scene 3, Op. 71, TH 14", "The Nutcracker, Op. 71, Act II"),
        ("Matthäus-Passion, BWV 244, Anhang", "St. Matthew Passion, BWV 244, Anhang"),
        ("Matthäus-Passion, BWV 244: Teil I", "St. Matthew Passion, BWV 244, Part I"),
        ("Turandot: Atto III, scena 1", "Turandot, Act III"),
        ("Die Meistersinger von Nürnberg: Zweiter Aufzug / Aufzug 2", "Meistersinger, Act II"),
    ],
)
def test_same_part_written_differently(first: str, second: str) -> None:
    """The same act or part written differently has one key."""
    assert part_key(first) is not None
    assert part_key(first) == part_key(second)


def test_act_is_part_of_the_work_before_it() -> None:
    """An act after a comma, a slash or a colon is part of the work named before it."""
    assert parent_work_name("La bohème, SC 67 / Act 1") == "La bohème, SC 67"
    assert parent_work_name("La bohème, SC 67, Act I") == "La bohème, SC 67"
    assert parent_work_name("Turandot: Atto III") == "Turandot"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Matthäus-Passion, BWV 244, Part 1", "Matthäus-Passion, BWV 244, Part 2"),
        ("Opera, Act 1", "Opera, Part 1"),
        ("Passion, Anhang", "Passion, Appendix"),
        ("Passion, Anhang", "Passion, Part 1"),
    ],
)
def test_different_parts_have_different_keys(first: str, second: str) -> None:
    """Different acts or parts never share a key."""
    assert part_key(first) != part_key(second)


@pytest.mark.parametrize(
    "work",
    [
        "Symphony No. 5, Op. 67",
        "Symphony No. 3 Part 2",
        "La bohème, Scene 1",
        "Rigoletto: Atto III. “La donna è mobile” (Duca)",
        "Matthäus-Passion, BWV 244: Teil II, XXXVIa. Rezitativ",
        "La bohème: Atto I. In soffitta",
    ],
)
def test_whole_work_has_no_part_key(work: str) -> None:
    """A work that is not an act or part, or a piece within one, has no part key."""
    assert part_key(work) is None


@pytest.mark.parametrize(
    ("name", "title", "expected"),
    [
        (
            "Ludwig van Beethoven & Conducted By William Bowles & Sarah Ainsworth",
            "Für Elise",
            "Ludwig van Beethoven",
        ),
        ("Chopin & Sarah Ainsworth", "Chopin: Nocturne No. 2: Andante", "Chopin"),
        ("Frédéric Chopin & Sarah Ainsworth", "Chopin: Nocturne No. 2", "Frédéric Chopin"),
        ("Simon & Garfunkel", "The Sound of Silence", None),
        ("Simon & Garfunkel", "Bridge over Troubled Water: Live", None),
        ("Chopin & Sarah Ainsworth", "Chopin", None),
        ("Ludwig van Beethoven", "Beethoven: Für Elise", None),
    ],
)
def test_joined_composer_name(name: str, title: str, expected: str | None) -> None:
    """A joined name starts with its composer only when its credits or the title say so."""
    assert joined_composer_name(name, title) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Variation No. 1", ("variation", 1)),
        ("Variatio 1. a 1 Clav.", ("variatio", 1)),
        ("Var. 1", ("var", 1)),
        ("Variation nr. 12", ("variation", 12)),
        ("Variation IV", ("variation", 4)),
        ("Étude 3", ("etude", 3)),
        ("No. 1 Kommt", None),
        ("I. Allegro con brio", None),
        ("Aria", None),
        ("Andante vivace", None),
    ],
)
def test_word_and_number(name: str, expected: tuple[str, int] | None) -> None:
    """A name starting with a word and a number gives the word, lowercase, and the number."""
    assert word_and_number(name) == expected


@pytest.mark.parametrize(
    ("work", "expected"),
    [
        ("Tosca, SC 69 / Act 3", "Act III"),
        ("Rigoletto, Act III", "Act III"),
        ("Turandot: Atto III", "Act III"),
        ("Swan Lake, Act II, Scene 1, Op. 20", "Act II, Scene 1"),
        ("Messiah, HWV 56, Pt. 2", "Part II"),
        ("La bohème: Atto I. In soffitta", "Act I"),
        ("Don Carlo: Atto I (II)", "Act I"),
        # a scene of an act with its own name is no act
        ('Die Meistersinger, WWV 96: Akt III, Scene V. "Sankt Krispin"', None),
        ("La Damnation de Faust: Quatrième Partie", None),
        ("Symphony No. 5", None),
    ],
)
def test_act_or_part_is_shown_in_english_with_a_roman_number(
    work: str, expected: str | None
) -> None:
    """An act or part closing a work name, or before its subtitle, shows with a Roman number."""
    assert part_label(work) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("02. Christe", "Christe"),
        ("No. 3, O mio babbino caro", "O mio babbino caro"),
        ("IV. In the Hall of the Mountain King", "In the Hall of the Mountain King"),
        (
            "No. 5 Habanera « L'amour est un oiseau rebelle »",
            "Habanera « L'amour est un oiseau rebelle »",
        ),
        ('N°13b Aria "Ah, la paterna mano"', 'Aria "Ah, la paterna mano"'),
        ("Scena 2. no. 15. “Tu che a Dio spiegasti l'ali”", "“Tu che a Dio spiegasti l'ali”"),
        ("Картина III, no. 14a. Па-де-де", "Па-де-де"),
        ("O soave fanciulla", "O soave fanciulla"),
        ("No. 1 in A major", "No. 1 in A major"),
        ("Notturno. Andante", "Notturno. Andante"),
        ("I Dovregubbens hall", "I Dovregubbens hall"),
        ("Scene I. Recitativo", "Scene I. Recitativo"),
    ],
)
def test_movement_name_loses_the_number_it_opens_with(name: str, expected: str) -> None:
    """A movement name without its opening number reads as the movement's own name."""
    assert without_movement_number(name) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Piano Concerto No. 21 in C Major, K. 467", WorkType.CONCERTO),
        (
            'The Four Seasons, Violin Concerto No. 2 in G Minor, RV 315 "L\'estate"',
            WorkType.CONCERTO,
        ),
        (
            'Grand Sonata No. 3 in F Minor, Op. 14 "Concert sans orchestre"',
            WorkType.SONATA,
        ),
        ("Missa Papae Marcelli", WorkType.MASS),
        ("Symphonie fantastique in C Minor, Op. 14", WorkType.SYMPHONY),
        ("The Sorcerer's Apprentice (symphonic poem)", WorkType.SYMPHONIC_POEM),
        ("String Quartet in D Minor, Op. 103", WorkType.QUARTET),
        ("3 Études, Op. 65", WorkType.ETUDE),
        ("Carmen (Opera) Suite No. 1", WorkType.SUITE),
        ("La scala di seta Opera: Overture", WorkType.OVERTURE),
        ("Madama Butterfly", None),
        ("Adagio for Strings, Op. 11", None),
        ('Prelude "Symphony"', None),
        ("Swan Lake (Ballet)", WorkType.BALLET),
        ('Dance of the Hours (Ballet) "La Gioconda"', None),
        ("Requiem in D minor, K. 626", None),
        ("Aida (Opera)", WorkType.OPERA),
        ('Die Fledermaus "The Bat" Operetta', WorkType.OPERETTA),
        ("The Phantom of the Opera", None),
        ("La gazza ladra (The Thieving Magpie) Opera.", WorkType.OPERA),
        ("Opera Overture", WorkType.OVERTURE),
        ("Opera Arias", None),
    ],
)
def test_title_work_type(title: str, expected: WorkType | None) -> None:
    """A work title names its type as a whole word or bracket label, outside a quoted nickname."""
    assert title_work_type(title) == expected
