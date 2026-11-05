"""Tests for the pure helper functions of the Metadata Controller."""

import pytest

from music_assistant.controllers.metadata.helpers import (
    _detect_image_format,
    _normalize_imageproxy_format,
    musicbrainz_work_name,
)
from music_assistant.providers.musicbrainz.models import MusicBrainzAlias, MusicBrainzWork


class TestDetectImageFormat:
    """`_detect_image_format` maps a path/extension to a served image format."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("cover.svg", "svg"),
            ("cover.png", "png"),
            ("cover.jpg", "jpg"),
            ("cover.jpeg", "jpg"),
            ("cover.webp", "jpg"),
            ("noextension", "jpg"),
            ("http://host/path/IMAGE.PNG", "png"),
            ("http://host/path/art.png?cs=abc123", "png"),
        ],
    )
    def test_detect(self, path: str, expected: str) -> None:
        """The extension (after stripping any query suffix) drives the format."""
        assert _detect_image_format(path) == expected


class TestNormalizeImageproxyFormat:
    """`_normalize_imageproxy_format` validates the client-supplied `fmt` value."""

    @pytest.mark.parametrize("value", ["jpg", "JPG", " png ", "jpeg", "svg"])
    def test_valid(self, value: str) -> None:
        """Known formats normalize to their lowercase, trimmed form."""
        assert _normalize_imageproxy_format(value) == value.strip().lower()

    @pytest.mark.parametrize("value", [None, "", "gif", "bmp", "exe"])
    def test_invalid(self, value: str | None) -> None:
        """Unknown or empty formats return None."""
        assert _normalize_imageproxy_format(value) is None


@pytest.mark.parametrize(
    ("language", "aliases", "expected"),
    [
        ("de", [("24 Präludien", "de"), ("24 Preludes", "en")], "24 Präludien"),
        ("nl", [("24 Präludien", "de"), ("24 Preludes", "en")], "24 Preludes"),
        ("en", [], "24 Préludes, op. 28"),
    ],
)
def test_work_name_prefers_the_language_then_english_then_the_title(
    language: str, aliases: list[tuple[str, str]], expected: str
) -> None:
    """A work is named by its primary alias in the language, else in English, else its title."""
    work = MusicBrainzWork(
        id="1",
        title="24 Préludes, op. 28",
        aliases=[MusicBrainzAlias(name, name, locale, None, True) for name, locale in aliases],
    )

    assert musicbrainz_work_name(work, language) == expected


@pytest.mark.parametrize(
    ("aliases", "expected"),
    [
        (
            [
                ("Щелкунчик", "ru", "Work name"),
                ("Nussknacker", None, "Search hint"),
                ("Casse-Noisette", "fr", "Work name"),
                ("The Nutcracker", "en", "Work name"),
            ],
            "The Nutcracker",
        ),
        ([("Casse-Noisette", "fr", "Work name")], "Casse-Noisette"),
        ([("Nussknacker", None, "Search hint"), ("Щелкунчик", "ru", None)], "Щелкунчик, op. 71"),
    ],
)
def test_work_name_of_a_non_latin_title_falls_back_to_a_latin_alias(
    aliases: list[tuple[str, str | None, str | None]], expected: str
) -> None:
    """A title not in Latin script gives way to a Latin alias, English first, else stays."""
    work = MusicBrainzWork(
        id="1",
        title="Щелкунчик, op. 71",
        aliases=[MusicBrainzAlias(name, name, locale, kind) for name, locale, kind in aliases],
    )

    assert musicbrainz_work_name(work, "nl") == expected
