"""Tests for reading the work and movement a classical track title names."""

from __future__ import annotations

import pytest

from music_assistant.helpers.work_titles import (
    parent_work_name,
    part_key,
    part_label,
    without_movement_number,
    word_and_number,
)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Matthäus-Passion, BWV 244, Part 1", "St. Matthew Passion, BWV 244, Part I"),
        ("Messiah, HWV 56, Pt. 2", "Messiah, HWV 56, Part II"),
        ("La bohème, Act II", "La Bohème, Act 2"),
        ("La bohème, SC 67 / Act 1", "La bohème, SC 67, Act I"),
        ("Matthäus-Passion, BWV 244, Anhang", "St. Matthew Passion, BWV 244, Anhang"),
        ("Matthäus-Passion, BWV 244: Teil I", "St. Matthew Passion, BWV 244, Part I"),
        ("Turandot: Atto III, scena 1", "Turandot, Act III"),
        ("Carmen: Acte I", "Carmen, Act I"),
        ("Die Walküre: Akt 1", "Die Walküre, Act I"),
        ("Die Meistersinger von Nürnberg: Zweiter Aufzug / Aufzug 2", "Meistersinger, Act II"),
    ],
)
def test_same_part_written_differently(first: str, second: str) -> None:
    """The same act or part written differently has one key."""
    assert part_key(first) is not None
    assert part_key(first) == part_key(second)


def test_slashed_act_is_part_of_the_same_work() -> None:
    """An act after a slash is part of the work the same act after a comma is part of."""
    assert parent_work_name("La bohème, SC 67 / Act 1") == "La bohème, SC 67"
    assert parent_work_name("La bohème, SC 67, Act I") == "La bohème, SC 67"


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


def test_act_after_a_colon_is_part_of_the_work_before_it() -> None:
    """An act a MusicBrainz title names after a colon is part of the work before the colon."""
    assert parent_work_name("Turandot: Atto III") == "Turandot"


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
        ("Rigoletto / Act 2", "Act II"),
        ("Rigoletto, Act III", "Act III"),
        ("Turandot: Atto III", "Act III"),
        ("Swan Lake, Act II, Scene 1, Op. 20", "Act II, Scene 1"),
        ("Messiah, HWV 56, Pt. 2", "Part II"),
        ("La bohème: Atto I. In soffitta", "Act I"),
        ("Luisa Miller: Atto II. “L'intrigo”", "Act II"),
        ("Don Carlo: Atto I (II)", "Act I"),
        # a scene of an act with its own name is no act
        ('Die Meistersinger, WWV 96: Akt III, Scene V. "Sankt Krispin"', None),
        ("La Damnation de Faust: Quatrième Partie", None),
        ("Symphony No. 5. Allegro", None),
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
        ("no. 20A: Danse russe", "Danse russe"),
        ("Scena 2. no. 15. “Tu che a Dio spiegasti l'ali”", "“Tu che a Dio spiegasti l'ali”"),
        ('Scene II. No. 2 Aria "Non ho colpa"', 'Aria "Non ho colpa"'),
        ("Картина III, no. 14a. Па-де-де", "Па-де-де"),
        ("O soave fanciulla", "O soave fanciulla"),
        ("No. 1 in A major", "No. 1 in A major"),
        ("Nocturne in E-flat major", "Nocturne in E-flat major"),
        ("Notturno. Andante", "Notturno. Andante"),
        ("I Dovregubbens hall", "I Dovregubbens hall"),
        ("Scene I. Recitativo", "Scene I. Recitativo"),
    ],
)
def test_movement_name_loses_the_number_it_opens_with(name: str, expected: str) -> None:
    """A movement name without its opening number reads as the movement's own name."""
    assert without_movement_number(name) == expected
