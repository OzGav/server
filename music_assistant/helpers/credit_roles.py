"""
The credit roles of performing artists, and helpers for the roles music services name.

The helpers read a role such as "Tenor Vocals".
"""

from __future__ import annotations

import re
from typing import Final

from music_assistant_models.enums import ArtistRole

# the roles of the artists performing a recording, in the order that settles a performer's
# main role when two roles have as many credits
PERFORMING_ROLES: Final[tuple[ArtistRole, ...]] = (
    ArtistRole.CONDUCTOR,
    ArtistRole.ORCHESTRA,
    ArtistRole.ENSEMBLE,
    ArtistRole.CHOIR,
    ArtistRole.SOLOIST,
    ArtistRole.PERFORMER,
)
# the performing roles as quoted SQL values, for an IN clause
SQL_PERFORMING_ROLES: Final[str] = ", ".join(f"'{x.value}'" for x in PERFORMING_ROLES)


def role_word(role: str) -> str:
    """Return a role in lowercase without spaces or hyphens, as in "mainartist"."""
    return re.sub(r"[\s-]", "", role).lower()
