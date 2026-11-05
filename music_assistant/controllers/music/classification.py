"""
Classical classification for the Music controller.

Every library track, album and artist has a stored is_classical flag, derived by the
server from these rules:

- a track is classical when it is tagged classical, is mapped to the curated classical genre,
  belongs to a work of a classical type (a symphony, a mass, ...) or is on a classical album
- an album is classical when it is mapped to the curated classical genre or when more
  than half of its tracks are classical by their own tag, genre or work type
- an artist is classical when it has any credit on a classical track

The flags are kept current as the library changes, and a classical composer without a period
gets the period of its stored life dates.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

from music_assistant_models.enums import ArtistEntityType, ArtistRole, MediaType, Period, WorkType

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_ALBUMS,
    DB_TABLE_ARTISTS,
    DB_TABLE_GENRE_MEDIA_ITEM_EXCLUSION,
    DB_TABLE_GENRE_MEDIA_ITEM_MAPPING,
    DB_TABLE_GENRES,
    DB_TABLE_TRACK_CREDITS,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.helpers import CREDITED_TRACK_ARTISTS
from music_assistant.helpers.datetime import release_year, utc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from music_assistant_models.media_items import Artist

    from music_assistant import MusicAssistant

# the translation key of the curated classical genre, which stays put when its name is edited
CLASSICAL_GENRE_KEY: Final[str] = "classical"


def _has_classical_genre(item: str, media_type: MediaType) -> str:
    """Return the condition that an item has its own mapping to the curated classical genre."""
    gm = DB_TABLE_GENRE_MEDIA_ITEM_MAPPING
    return f"""EXISTS (
        SELECT 1 FROM {gm} JOIN {DB_TABLE_GENRES} ON {DB_TABLE_GENRES}.item_id = {gm}.genre_id
        WHERE {gm}.media_id = {item}.item_id
        AND {gm}.media_type = '{media_type.value}'
        AND {gm}.is_derived = 0
        AND {DB_TABLE_GENRES}.translation_key = '{CLASSICAL_GENRE_KEY}'
        AND {DB_TABLE_GENRES}.content_type IS NULL
        AND {DB_TABLE_GENRES}.is_excluded = 0
        AND NOT EXISTS (
            SELECT 1 FROM {DB_TABLE_GENRE_MEDIA_ITEM_EXCLUSION} e
            WHERE e.genre_id = {gm}.genre_id
            AND e.media_id = {gm}.media_id
            AND e.media_type = {gm}.media_type))"""


def _track_is_classical_by_own_signal(track: str) -> str:
    """Return the condition that a track is classical by its own tag, genre or work type."""
    return f"""({track}.classical_tag = 1 OR {_has_classical_genre(track, MediaType.TRACK)}
        OR EXISTS (SELECT 1 FROM {DB_TABLE_WORKS} w WHERE w.item_id = {track}.work_id
        AND w.work_type IS NOT NULL AND w.work_type != '{WorkType.OTHER.value}'))"""


# album tracks only count by their own signals, as their album inheritance depends on this flag
ALBUM_IS_CLASSICAL: Final[str] = f"""({_has_classical_genre(DB_TABLE_ALBUMS, MediaType.ALBUM)}
    OR COALESCE((
    SELECT 2 * SUM({_track_is_classical_by_own_signal("t")}) > COUNT(*)
    FROM {DB_TABLE_ALBUM_TRACKS} at JOIN {DB_TABLE_TRACKS} t ON t.item_id = at.track_id
    WHERE at.album_id = {DB_TABLE_ALBUMS}.item_id), 0))"""

TRACK_IS_CLASSICAL: Final[str] = f"""({_track_is_classical_by_own_signal(DB_TABLE_TRACKS)} OR
    EXISTS (SELECT 1 FROM {DB_TABLE_ALBUM_TRACKS} at
    JOIN {DB_TABLE_ALBUMS} a ON a.item_id = at.album_id
    WHERE at.track_id = {DB_TABLE_TRACKS}.item_id AND a.is_classical = 1))"""

ARTIST_IS_CLASSICAL: Final[str] = f"""EXISTS (
    SELECT 1 FROM {CREDITED_TRACK_ARTISTS} ta JOIN {DB_TABLE_TRACKS} t ON t.item_id = ta.track_id
    WHERE ta.artist_id = {DB_TABLE_ARTISTS}.item_id AND t.is_classical = 1)"""

# only a classical artist credited as a composer, on a track or a work, has a period
IS_CLASSICAL_COMPOSER: Final[str] = (
    f"({DB_TABLE_ARTISTS}.is_classical = 1 AND ("
    f"EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_CREDITS} ta WHERE ta.artist_id = "
    f"{DB_TABLE_ARTISTS}.item_id AND ta.role = '{ArtistRole.COMPOSER.value}') "
    f"OR EXISTS (SELECT 1 FROM {DB_TABLE_WORK_ARTISTS} wa WHERE wa.artist_id = "
    f"{DB_TABLE_ARTISTS}.item_id AND wa.role = '{ArtistRole.COMPOSER.value}')))"
)

# a composer's working life, from which their period is estimated, is taken to start at this
# age and to end this many years before their death
COMPOSER_WORKING_LIFE_START_AGE = 25
COMPOSER_WORKING_LIFE_END_OFFSET = 5

# the first year of each classical period after the medieval one, oldest first
COMPOSER_PERIOD_START_YEARS = (
    (1400, Period.RENAISSANCE),
    (1600, Period.BAROQUE),
    (1750, Period.CLASSICAL),
    (1820, Period.ROMANTIC),
    (1900, Period.MODERN),
    (1975, Period.CONTEMPORARY),
)


def composer_period(birth_year: int, death_year: int | None) -> Period:
    """
    Return the classical period of a composer, judged by the middle of their working life.

    :param birth_year: The year the composer was born.
    :param death_year: The year the composer died, None for a living composer.
    """
    start = birth_year + COMPOSER_WORKING_LIFE_START_AGE
    # a living composer is still working
    end = death_year - COMPOSER_WORKING_LIFE_END_OFFSET if death_year else utc().year
    midpoint = (start + end) // 2
    return next(
        (period for year, period in reversed(COMPOSER_PERIOD_START_YEARS) if midpoint >= year),
        Period.MEDIEVAL,
    )


def life_span_period(artist: Artist) -> Period | None:
    """
    Return the classical period of a composer's stored life span, None when it has none.

    :param artist: The artist, a person with a known year of birth, and of death once died.
    """
    life_span = artist.metadata.life_span
    if (
        artist.metadata.artist_entity_type != ArtistEntityType.PERSON
        or not life_span
        or not (birth_year := release_year(life_span.begin))
    ):
        return None
    death_year = release_year(life_span.end)
    if life_span.ended and not death_year:
        return None  # died in an unknown year, so their working life is unknown too
    return composer_period(birth_year, death_year)


class ClassicalClassifier:
    """Keeps the classical flags of library tracks, albums and artists current."""

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the classifier."""
        self.mass = mass

    async def update(
        self,
        track_ids: Iterable[int] = (),
        album_ids: Iterable[int] = (),
        artist_ids: Iterable[int] = (),
    ) -> None:
        """
        Update the classical flags of the given library items and of the items depending on them.

        :param track_ids: Library ids of the changed tracks.
        :param album_ids: Library ids of the changed albums.
        :param artist_ids: Library ids of artists to recompute, such as those that lost a credit.
        """
        track_ids = set(track_ids)
        album_ids = set(album_ids)
        artist_ids = set(artist_ids)
        album_ids |= await self._get_ids(DB_TABLE_ALBUM_TRACKS, "album_id", "track_id", track_ids)
        track_ids |= await self._get_ids(DB_TABLE_ALBUM_TRACKS, "track_id", "album_id", album_ids)
        artist_ids |= await self._get_ids(
            CREDITED_TRACK_ARTISTS, "artist_id", "track_id", track_ids
        )
        # albums first and artists last, each flag builds on the one before
        await self._write_flags(DB_TABLE_ALBUMS, ALBUM_IS_CLASSICAL, album_ids)
        await self._write_flags(DB_TABLE_TRACKS, TRACK_IS_CLASSICAL, track_ids)
        await self._write_flags(DB_TABLE_ARTISTS, ARTIST_IS_CLASSICAL, artist_ids)
        await self.set_composer_periods(artist_ids)

    async def update_item(self, media_type: MediaType, item_id: int) -> None:
        """
        Update the classical flags after a change to a single library item.

        :param media_type: Media type of the changed item.
        :param item_id: Library id of the changed item.
        """
        if media_type == MediaType.TRACK:
            await self.update(track_ids=[item_id])
        elif media_type == MediaType.ALBUM:
            await self.update(album_ids=[item_id])
        elif media_type == MediaType.ARTIST:
            await self.update(artist_ids=[item_id])
        elif media_type == MediaType.WORK:
            await self.update(
                track_ids=await self._get_ids(DB_TABLE_TRACKS, "item_id", "work_id", {item_id})
            )
        elif media_type == MediaType.GENRE:
            await self.update_all()

    async def update_all(self) -> None:
        """Update the classical flags of the whole library."""
        await self._write_flags(DB_TABLE_ALBUMS, ALBUM_IS_CLASSICAL)
        await self._write_flags(DB_TABLE_TRACKS, TRACK_IS_CLASSICAL)
        await self._write_flags(DB_TABLE_ARTISTS, ARTIST_IS_CLASSICAL)
        await self.set_composer_periods()

    async def set_composer_periods(self, artist_ids: Iterable[int] | None = None) -> None:
        """
        Give the classical composers without a period the period of their stored life dates.

        :param artist_ids: Library ids of the artists to consider, None for the whole library.
        """
        scope = ""
        if artist_ids is not None:
            if not (ids := ", ".join(str(int(x)) for x in artist_ids)):
                return
            scope = f"item_id IN ({ids}) AND "
        # only people with a year of birth, and of death once died, can be given a period
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id FROM {DB_TABLE_ARTISTS} WHERE {scope}period IS NULL "
            "AND json_extract(metadata, '$.artist_entity_type') = :person "
            "AND json_extract(metadata, '$.life_span.begin') != '' "
            "AND (COALESCE(json_extract(metadata, '$.life_span.ended'), 0) = 0 "
            "OR json_extract(metadata, '$.life_span.end') != '') "
            f"AND {IS_CLASSICAL_COMPOSER}",
            {"person": ArtistEntityType.PERSON.value},
            limit=0,
        )
        for row in rows:
            artist = await self.mass.music.artists.get_library_item(row["item_id"])
            if period := life_span_period(artist):
                artist.period = period
                await self.mass.music.artists.update_item_in_library(artist.item_id, artist)

    @asynccontextmanager
    async def track_removal(self, track_id: int) -> AsyncGenerator[None]:
        """
        Keep the classical flags current while a track is removed from the library.

        :param track_id: Library id of the track about to be removed.
        """
        album_ids = await self._get_ids(DB_TABLE_ALBUM_TRACKS, "album_id", "track_id", {track_id})
        artist_ids = await self._get_ids(
            CREDITED_TRACK_ARTISTS, "artist_id", "track_id", {track_id}
        )
        yield
        await self.update(album_ids=album_ids, artist_ids=artist_ids)

    async def _get_ids(
        self, table: str, column: str, match_column: str, match_ids: set[int]
    ) -> set[int]:
        """Return the ids linked to any of the given ids through a relation."""
        if not match_ids:
            return set()
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT {column} FROM {table} WHERE {match_column} IN ({_sql_ids(match_ids)})",
            limit=0,
        )
        return {int(row[column]) for row in rows}

    async def _write_flags(
        self, table: str, is_classical: str, item_ids: set[int] | None = None
    ) -> None:
        """Store the computed classical flag of the given items, or of all items."""
        if item_ids is not None and not item_ids:
            return
        scope = f"item_id IN ({_sql_ids(item_ids)}) AND " if item_ids is not None else ""
        # only rows whose flag changes are written, so the other rows keep their modified time
        await self.mass.music.database.execute_write(
            f"UPDATE {table} SET is_classical = {is_classical} "
            f"WHERE {scope}is_classical != {is_classical}"
        )


def _sql_ids(item_ids: Iterable[int]) -> str:
    """Return library ids as a comma separated list."""
    return ", ".join(str(int(item_id)) for item_id in item_ids)
