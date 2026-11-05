"""Helpers to read the numbers a classical title names."""

from __future__ import annotations

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


def numeral_value(numeral: str) -> int | None:
    """
    Return the value of a number written in digits or as a Roman numeral from I to XX.

    :param numeral: A number such as "3" or "IV", with a Roman numeral in capitals.
    """
    if numeral.isdecimal():
        return int(numeral)
    return _ROMAN_NUMERALS.index(numeral) + 1 if numeral in _ROMAN_NUMERALS else None
