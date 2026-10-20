"""Tests for reading the work and movement a classical track title names."""

from __future__ import annotations

import pytest

from music_assistant.helpers.work_titles import (
    TrackWork,
    joined_composer_name,
    name_different_keys,
    names_catalogued_part,
    numbered_piece,
    parent_work_name,
    parse_catalog_numbers,
    parse_work_title,
    part_key,
    word_and_number,
)

# real Spotify titles, with the first artist Spotify lists for them
BEETHOVEN = "Ludwig van Beethoven"
PUCCINI = "Giacomo Puccini"


@pytest.mark.parametrize(
    ("title", "composer", "expected"),
    [
        (
            "Symphony No. 5 in C Minor, Op. 67: I. Allegro con brio",
            BEETHOVEN,
            TrackWork("Symphony No. 5 in C Minor, Op. 67", None, "I. Allegro con brio", 1),
        ),
        (
            "14 Romances, Op. 34: XIV. Vocalise",
            "Sergei Rachmaninoff",
            TrackWork("14 Romances, Op. 34", None, "XIV. Vocalise", 14),
        ),
        (
            'Symphony No. 9 in D Minor, Op. 125 "Choral": IV. (a) Presto - Allegro assai',
            BEETHOVEN,
            TrackWork(
                'Symphony No. 9 in D Minor, Op. 125 "Choral"',
                None,
                "IV. (a) Presto - Allegro assai",
                4,
            ),
        ),
        (
            "Symphony No. 7 in C Major, Op. 105: Vivacissimo - Adagio",
            "Jean Sibelius",
            TrackWork("Symphony No. 7 in C Major, Op. 105", None, "Vivacissimo - Adagio"),
        ),
        (
            "Goldberg Variations, BWV 988: Aria",
            "Johann Sebastian Bach",
            TrackWork("Goldberg Variations, BWV 988", None, "Aria"),
        ),
        (
            "21 Hungarian Dances, WoO 1: No. 5 in F-Sharp Minor",
            "Johannes Brahms",
            TrackWork("21 Hungarian Dances, WoO 1", None, "No. 5 in F-Sharp Minor"),
        ),
        (
            'Symphony No. 94 in G Major, Hob. I:94 "Surprise": II. Andante',
            "Joseph Haydn",
            TrackWork('Symphony No. 94 in G Major, Hob. I:94 "Surprise"', None, "II. Andante", 2),
        ),
    ],
)
def test_work_and_movement(title: str, composer: str, expected: TrackWork) -> None:
    """The text before the last colon is the work, the text after it the movement."""
    assert parse_work_title(title, composer) == expected


@pytest.mark.parametrize(
    ("title", "composer", "expected"),
    [
        (
            "La bohème, Act II: Viva Parpignol!",
            PUCCINI,
            TrackWork("La bohème, Act II", None, "Viva Parpignol!", None, "La bohème"),
        ),
        (
            "The Nutcracker, Op. 71, Act II: No. 13, Waltz of the Flowers",
            "Pyotr Ilyich Tchaikovsky",
            TrackWork(
                "The Nutcracker, Op. 71, Act II",
                None,
                "No. 13, Waltz of the Flowers",
                None,
                "The Nutcracker, Op. 71",
            ),
        ),
        (
            'Messiah, HWV 56, Pt. 2: "Hallelujah" (Chorus)',
            "Georg Friedrich Händel",
            TrackWork(
                "Messiah, HWV 56, Pt. 2", None, '"Hallelujah" (Chorus)', None, "Messiah, HWV 56"
            ),
        ),
        (
            'La Fanciulla del West, Act I: "Quello che tacete"',
            PUCCINI,
            TrackWork(
                "La Fanciulla del West, Act I",
                None,
                '"Quello che tacete"',
                None,
                "La Fanciulla del West",
            ),
        ),
        (
            "The Valkyrie, WWV 86b, Act III: Ride of the Valkyries",
            "Richard Wagner",
            TrackWork(
                "The Valkyrie, WWV 86b, Act III",
                None,
                "Ride of the Valkyries",
                None,
                "The Valkyrie, WWV 86b",
            ),
        ),
        (
            "Swan Lake, Act II, Op. 20, TH 12: No. 10, Scene",
            "Pyotr Ilyich Tchaikovsky",
            TrackWork(
                "Swan Lake, Act II, Op. 20, TH 12",
                None,
                "No. 10, Scene",
                None,
                "Swan Lake, Op. 20, TH 12",
            ),
        ),
        (
            "La bohème, SC 67 / Act 1: Che gelida manina",
            PUCCINI,
            TrackWork(
                "La bohème, SC 67 / Act 1",
                None,
                "Che gelida manina",
                None,
                "La bohème, SC 67",
            ),
        ),
        (
            "Messiah, HWV 56 / Part 2: Hallelujah",
            "Georg Friedrich Händel",
            TrackWork("Messiah, HWV 56 / Part 2", None, "Hallelujah", None, "Messiah, HWV 56"),
        ),
        (
            "St. Matthew Passion, BWV 244, Anhang: Aria",
            "Johann Sebastian Bach",
            TrackWork(
                "St. Matthew Passion, BWV 244, Anhang",
                None,
                "Aria",
                None,
                "St. Matthew Passion, BWV 244",
            ),
        ),
        (
            "Symphony No. 3 Part 2: I. Lento",
            None,
            TrackWork("Symphony No. 3 Part 2", None, "I. Lento", 1),
        ),
        (
            "La bohème, Act II, Scene 1: Viva Parpignol!",
            PUCCINI,
            TrackWork("La bohème, Act II, Scene 1", None, "Viva Parpignol!", None, "La bohème"),
        ),
        (
            "The Nutcracker, Act II, Scene 3, Op. 71, TH 14: No. 15, Pas de deux",
            "Pyotr Ilyich Tchaikovsky",
            TrackWork(
                "The Nutcracker, Act II, Scene 3, Op. 71, TH 14",
                None,
                "No. 15, Pas de deux",
                None,
                "The Nutcracker, Op. 71, TH 14",
            ),
        ),
    ],
)
def test_act_or_part_comes_with_its_work(
    title: str, composer: str | None, expected: TrackWork
) -> None:
    """A work that is an act or part, after a comma or slash, comes with the work it is part of."""
    assert parse_work_title(title, composer) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "Tchaikovsky: Swan Lake Ballet, Op. 20: Act 1, No. 2 Waltz: Tempo di valse",
            TrackWork(
                "Swan Lake Ballet, Op. 20, Act 1",
                None,
                "No. 2 Waltz: Tempo di valse",
                None,
                "Swan Lake Ballet, Op. 20",
            ),
        ),
        (
            "Swan Lake, Op. 20: Act 2: No. 10 Scene",
            TrackWork("Swan Lake, Op. 20, Act 2", None, "No. 10 Scene", None, "Swan Lake, Op. 20"),
        ),
        (
            "Symphony No. 5 in C minor, Op. 67: I. Allegro con brio: Coda",
            TrackWork("Symphony No. 5 in C minor, Op. 67", None, "I. Allegro con brio: Coda", 1),
        ),
    ],
)
def test_work_ends_at_its_catalog_number(title: str, expected: TrackWork) -> None:
    """A work ends at a catalog number followed by a colon, taking in an act opening the rest."""
    assert parse_work_title(title, "Pyotr Ilyich Tchaikovsky") == expected


def test_aria_quoted_straight_after_the_colon() -> None:
    """A quoted aria straight after the colon is the movement, as with a space in between."""
    assert parse_work_title('La Bohème, Act I:"Che gelida manina"', PUCCINI) == TrackWork(
        "La Bohème, Act I", None, '"Che gelida manina"', None, "La Bohème"
    )


@pytest.mark.parametrize(
    ("title", "composer", "expected"),
    [
        (
            "Beethoven: Trio for Oboe, Clarinet and Bassoon C Major, Op. 87: I. Allegro",
            BEETHOVEN,
            TrackWork("Trio for Oboe, Clarinet and Bassoon C Major, Op. 87", None, "I. Allegro", 1),
        ),
        (
            "Elgar: Cello Concerto in E Minor, Op. 85: I. Adagio - Moderato",
            "Edward Elgar",
            TrackWork("Cello Concerto in E Minor, Op. 85", None, "I. Adagio - Moderato", 1),
        ),
        (
            "J.S. Bach: Keyboard Concerto No. 1 in D Major, BWV 972: II. Larghetto",
            "Johann Sebastian Bach",
            TrackWork("Keyboard Concerto No. 1 in D Major, BWV 972", None, "II. Larghetto", 2),
        ),
        (
            "Beethoven: Für Elise",
            BEETHOVEN,
            TrackWork("Für Elise"),
        ),
        (
            "Chopin: Nocturne in E-flat major, Op. 55, No. 2",
            "Frédéric Chopin",
            TrackWork("Nocturne in E-flat major, Op. 55, No. 2"),
        ),
        (
            "Strauss: The Blue Danube Waltz, Op. 314",
            "Johann Strauss II",
            TrackWork("The Blue Danube Waltz, Op. 314"),
        ),
        (
            'Vaughan-Williams: Fantasia on "Greensleeves"',
            "Ralph Vaughan Williams",
            TrackWork('Fantasia on "Greensleeves"'),
        ),
        (
            'Beethoven: Piano Concerto No. 5 in E-flat major, Op. 73 "Emperor": Mvt. III-Rondo',
            BEETHOVEN,
            TrackWork(
                'Piano Concerto No. 5 in E-flat major, Op. 73 "Emperor"', None, "Mvt. III-Rondo"
            ),
        ),
        (
            'Mozart: Symphony No. 41 in C major, K. 551 "Jupiter" Allegro vivace (excerpt)',
            "Wolfgang Amadeus Mozart",
            TrackWork('Symphony No. 41 in C major, K. 551 "Jupiter" Allegro vivace (excerpt)'),
        ),
        (
            "Haydn: Concerto in D major, No. 11, Rondo all Ungarese (Allegro assai)",
            "Joseph Haydn",
            TrackWork("Concerto in D major, No. 11, Rondo all Ungarese (Allegro assai)"),
        ),
    ],
)
def test_composer_prefix_is_left_out(title: str, composer: str, expected: TrackWork) -> None:
    """A leading segment naming the composer is no part of the work."""
    assert parse_work_title(title, composer) == expected


@pytest.mark.parametrize(
    ("title", "composer", "expected"),
    [
        (
            "Richard Strauss: Also sprach Zarathustra, Op. 30",
            "Johann Strauss II",
            TrackWork("Richard Strauss", None, "Also sprach Zarathustra, Op. 30"),
        ),
        (
            "Chopin Nocturnes: No. 1 in B-flat minor",
            "Frédéric Chopin",
            TrackWork("Chopin Nocturnes", None, "No. 1 in B-flat minor"),
        ),
    ],
)
def test_prefix_with_words_outside_the_composer_name_stays(
    title: str, composer: str, expected: TrackWork
) -> None:
    """A leading segment with a word that is not in the composer's name stays."""
    assert parse_work_title(title, composer) == expected


def test_prefix_naming_another_artist_stays() -> None:
    """A leading segment that does not name the composer stays part of the work."""
    assert parse_work_title(
        "Beethoven: Trio for Oboe, Clarinet and Bassoon C Major, Op. 87: I. Allegro",
        "Les Vents Français",
    ) == TrackWork(
        "Beethoven: Trio for Oboe, Clarinet and Bassoon C Major, Op. 87", None, "I. Allegro", 1
    )


@pytest.mark.parametrize(
    "title",
    [
        "Boléro, M. 81",
        "Tapiola, Op. 112",
        'Ave Maria, D. 839 "Ellens Gesang III"',
        "The Lark Ascending (Excerpt: Opening)",
        "Rhapsody in Blue",
        "Silver Springs",
        "Movin' Out (Anthony's Song)",
        "Hob. I:94",
        "Intro:",
    ],
)
def test_title_without_a_work(title: str) -> None:
    """A title without a colon between two parts, outside brackets, names no work."""
    assert parse_work_title(title, "Some Artist") is None


@pytest.mark.parametrize(
    ("title", "composer", "expected"),
    [
        (
            "Allegri: Miserere - 01. Miserere Mei, Deus",
            "The Tallis Scholars",
            TrackWork("Miserere", None, "01. Miserere Mei, Deus", 1, other_composer="Allegri"),
        ),
        (
            "Palestrina: Missa Papae Marcelli - 2a. Gloria in Excelsis Deo",
            "The Tallis Scholars",
            TrackWork(
                "Missa Papae Marcelli",
                None,
                "2a. Gloria in Excelsis Deo",
                other_composer="Palestrina",
            ),
        ),
        (
            "Allegri: Miserere - 01. Miserere Mei, Deus",
            "Gregorio Allegri",
            TrackWork("Miserere", None, "01. Miserere Mei, Deus", 1),
        ),
    ],
)
def test_composer_work_and_dashed_movement(title: str, composer: str, expected: TrackWork) -> None:
    """A "Composer: Work - 1. Movement" title names its composer, the given one or another."""
    assert parse_work_title(title, composer) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "Goldberg Variations, BWV 988: Aria",
            TrackWork("Goldberg Variations, BWV 988", None, "Aria"),
        ),
        (
            "Matthäus-Passion, BWV 244: Teil II",
            TrackWork("Matthäus-Passion, BWV 244", None, "Teil II"),
        ),
        (
            "Matthäus-Passion, BWV 244: 1. Teil",
            TrackWork("Matthäus-Passion, BWV 244", None, "1. Teil", 1),
        ),
        ("Trout Quintet: Andante", TrackWork("Trout Quintet", None, "Andante")),
        ("Klavierkonzert Nr. 2: Adagio", TrackWork("Klavierkonzert Nr. 2", None, "Adagio")),
        (
            "La bohème, Act II, Scene 1: Viva Parpignol!",
            TrackWork("La bohème, Act II, Scene 1", None, "Viva Parpignol!", None, "La bohème"),
        ),
        ("Peer Gynt: II. Ases Tod", TrackWork("Peer Gynt", None, "II. Ases Tod", 2)),
        (
            "5. Sinfonie c-Moll, op. 67: I. Allegro con brio",
            TrackWork("5. Sinfonie c-Moll, op. 67", None, "I. Allegro con brio", 1),
        ),
    ],
)
def test_classical_sign_names_a_work(title: str, expected: TrackWork) -> None:
    """A title with a catalog number, form, act or numbered movement names a work on its own."""
    assert parse_work_title(title) == expected


@pytest.mark.parametrize(
    ("title", "composer"),
    [
        ("Interlude: Reprise", "Billy Joel"),
        ("Mr. Brightside: Live", "The Killers"),
        ("Gianni Schicchi: O mio babbino caro", PUCCINI),
        ("Swan Lake: Act II", "Pyotr Ilyich Tchaikovsky"),
    ],
)
def test_colon_alone_names_no_work(title: str, composer: str) -> None:
    """A title with only a colon names no work, unless the track is known to be classical."""
    assert parse_work_title(title, composer) is None


@pytest.mark.parametrize(
    ("title", "composer", "expected"),
    [
        (
            "Gianni Schicchi: O mio babbino caro",
            PUCCINI,
            TrackWork("Gianni Schicchi", None, "O mio babbino caro"),
        ),
        (
            "Swan Lake: Act II",
            "Pyotr Ilyich Tchaikovsky",
            TrackWork("Swan Lake", None, "Act II"),
        ),
        (
            "St Matthew Passion: Part I",
            "Johann Sebastian Bach",
            TrackWork("St Matthew Passion", None, "Part I"),
        ),
    ],
)
def test_classical_track_needs_no_other_sign(
    title: str, composer: str, expected: TrackWork
) -> None:
    """The title of a track known to be classical names a work with only a colon."""
    assert parse_work_title(title, composer, classical=True) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Kapitel 1: Der Anfang",
        "Chapter 12: The Boy Who Lived",
        "Folge 5: Die drei ???",
        "Teil 3: Die Rückkehr",
        "Part 2: Symphony of Lies",
        "Episode No. 4: A New Hope",
        "Kapitel 12.3: Op. 1",
        "Der Hobbit: 1. Kapitel",
        "Harry Potter: 12. Chapter",
        "Die drei ???: 3. Folge",
        "Der Hobbit: Nr. IV. Kapitel - Eine Sonate",
    ],
)
@pytest.mark.parametrize("classical", [False, True])
def test_chapter_names_no_work(title: str, classical: bool) -> None:
    """A title starting or ending with a chapter or episode names no work, whatever it holds."""
    assert parse_work_title(title, "Rufus Beck", classical) is None


def test_colon_within_a_numbered_piece_names_no_work() -> None:
    """A colon within the text of a numbered piece leaves the piece whole."""
    assert (
        parse_work_title(
            "Erster Teil, 9a. Evangelista, Jesus: Er sprach: Gehet hin in die Stadt",
            "Johann Sebastian Bach",
            classical=True,
        )
        is None
    )
    assert parse_work_title(
        "Matthäus-Passion, BWV 244: Erster Teil, 9a. Evangelista, Jesus: Er sprach"
    ) == TrackWork(
        "Matthäus-Passion, BWV 244", None, "Erster Teil, 9a. Evangelista, Jesus: Er sprach"
    )


def test_tchaikovsky_catalog_number() -> None:
    """A Tchaikovsky thematic catalog number is read like any other catalog number."""
    assert parse_catalog_numbers("Swan Lake, Op. 20, TH 12") == ["Op. 20", "TH 12"]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Nocturne Op. 9 No. 2", ["Op. 9 No. 2"]),
        ("Nocturne In F Sharp Major, Op 15, No. 2", ["Op 15, No. 2"]),
        ("Symphony No. 5 in C minor, Op. 67", ["Op. 67"]),
    ],
)
def test_catalog_number_keeps_its_number_within_the_opus(title: str, expected: list[str]) -> None:
    """The number within an opus belongs to the catalog number, with or without a comma."""
    assert parse_catalog_numbers(title) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Matthäus-Passion, BWV 244, Part 1", "St. Matthew Passion, BWV 244, Part I"),
        ("Messiah, HWV 56, Pt. 2", "Messiah, HWV 56, Part II"),
        ("La bohème, Act II", "La Bohème, Act 2"),
        ("La bohème, SC 67 / Act 1", "La bohème, SC 67, Act I"),
        ("Messiah, HWV 56 / Part 2", "Messiah, HWV 56, Pt. 2"),
        ("Swan Lake, Act II, Op. 20, TH 12", "Swan Lake, Act 2"),
        ("The Nutcracker, Act II, Scene 3, Op. 71, TH 14", "The Nutcracker, Op. 71, Act II"),
        ("Matthäus-Passion, BWV 244, Anhang", "St. Matthew Passion, BWV 244, Anhang"),
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
    ("work", "expected"),
    [
        ('Messiah, HWV 56: Part II, no. 44. Chorus "Hallelujah"', True),
        ("Sonata in B-flat major, D. 960: No. 3 Scherzo", True),
        ("Symphony No. 94, Hob. I:94: II. Andante", True),
        ("The Nutcracker, Act II, Scene 3, Op. 71, TH 14", True),
        ("Messiah, HWV 56", False),
        ("Symphony No. 94 in G major, Hob. I:94", False),
        ("Symphony No. 5: I. Allegro", False),
    ],
)
def test_catalogued_part(work: str, expected: bool) -> None:
    """A title naming an act or going on after its catalog number names a part of the work."""
    assert names_catalogued_part(work) is expected


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
    "work", ["Symphony No. 5, Op. 67", "Symphony No. 3 Part 2", "La bohème, Scene 1"]
)
def test_whole_work_has_no_part_key(work: str) -> None:
    """A work that is not an act or part has no part key."""
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
    ("work", "expected"),
    [
        (
            "Piano Sonata No. 11 in A major, K 331, No. 3 Rondo Alla Turca",
            ("K 331", 3, "Rondo Alla Turca"),
        ),
        ("Serenade for Strings in C major, Op. 48, No. 2 Waltz", ("Op. 48", 2, "Waltz")),
        ("Fantasie in C major, Op. 17, No. 1", ("Op. 17", 1, None)),
        ("Serenade, Op. 48, No. 2, TH 48", ("Op. 48", 2, None)),
        ("Piano Sonata No. 11 in A Major, K. 331", None),
        ("Symphony No. 5 in C minor, Op. 67", None),
    ],
)
def test_numbered_piece(work: str, expected: tuple[str, int, str | None] | None) -> None:
    """A work titled as a numbered piece of an opus gives its opus, number and name."""
    assert numbered_piece(work) == expected


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
    ("title", "other_title", "expected"),
    [
        ("Violin Concerto in A Minor, BWV 1041", "Violin Concerto in E major, BWV 1041", True),
        ("Violin Concerto in A Minor, BWV 1041", "Violin Concerto in A minor, BWV 1041", False),
        ("Violin Concerto in A minor", "Violinkonzert a-Moll", False),
        ("Symphony in E-flat major", "Symphony in E flat major", False),
        ("Symphony in E-flat major", "Symphony in E♭ major", False),
        ("Symphony in E-flat major", "Sinfonie Es-Dur", False),
        ("Symphony in E-flat major", "Symphony in E major", True),
        ("Sonata in F sharp minor", "Sonata in F# minor", False),
        ("Sonata in F sharp minor", "Sonate fis-Moll", False),
        ("Sonata in B-flat major", "Sonate B-Dur", False),
        ("Sonata in B minor", "Sonate h-Moll", False),
        ("Sonata in B minor", "Sonate b-Moll", True),
        ("Mass in C, Op. 86", "Mass in C minor, Op. 86", False),
        ("Toccata and Fugue, BWV 565", "Toccata and Fugue in D Minor, BWV 565", False),
    ],
)
def test_name_different_keys(title: str, other_title: str, expected: bool) -> None:
    """Titles differ in key only when both name one, however each spells it."""
    assert name_different_keys(title, other_title) is expected
