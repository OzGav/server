"""Tests for reading the work and movement a classical track title names."""

from __future__ import annotations

import pytest
from music_assistant_models.enums import WorkType

from music_assistant.helpers.work_titles import parse_catalog_numbers, title_work_type


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
