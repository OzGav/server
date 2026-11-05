"""Tests for reading the work and movement a classical track title names."""

from __future__ import annotations

import pytest

from music_assistant.helpers.work_titles import parse_catalog_numbers


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
