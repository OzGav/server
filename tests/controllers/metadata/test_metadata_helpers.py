"""Tests for the pure helper functions of the Metadata Controller."""

import pytest
from music_assistant_models.enums import Period

from music_assistant.controllers.metadata.helpers import (
    _detect_image_format,
    _normalize_imageproxy_format,
    composer_period,
)


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
    ("birth_year", "death_year", "expected"),
    [
        (1098, 1179, Period.MEDIEVAL),  # Hildegard von Bingen
        (1525, 1594, Period.RENAISSANCE),  # Palestrina
        (1685, 1759, Period.BAROQUE),  # Händel
        (1685, 1750, Period.BAROQUE),  # Bach
        (1756, 1791, Period.CLASSICAL),  # Mozart
        (1732, 1809, Period.CLASSICAL),  # Haydn
        (1770, 1827, Period.CLASSICAL),  # Beethoven
        (1797, 1828, Period.ROMANTIC),  # Schubert
        (1833, 1897, Period.ROMANTIC),  # Brahms
        (1860, 1911, Period.ROMANTIC),  # Mahler
        (1874, 1951, Period.MODERN),  # Schoenberg
        (1935, None, Period.CONTEMPORARY),  # Pärt, alive
    ],
)
def test_composer_period(birth_year: int, death_year: int | None, expected: Period) -> None:
    """A composer's period follows from the middle of their working life."""
    assert composer_period(birth_year, death_year) == expected
