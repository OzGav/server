"""The credit roles of the artists performing a recording."""

from __future__ import annotations

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
