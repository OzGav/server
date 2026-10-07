"""
Classical classification for the Music controller.

Every library track, album and artist carries a stored is_classical flag, derived by the
server from these rules:

- a track is classical when it is tagged classical, is mapped to the curated classical genre
  or is on a classical album
- an album is classical when it is tagged classical, is mapped to the curated classical genre
  or when more than half of its tracks are classical by their own tag or genre
- an artist is classical when it holds any credit on a classical track

The flags are kept current as the library changes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

from music_assistant_models.enums import MediaType

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_ALBUMS,
    DB_TABLE_ARTISTS,
    DB_TABLE_GENRE_MEDIA_ITEM_EXCLUSION,
    DB_TABLE_GENRE_MEDIA_ITEM_MAPPING,
    DB_TABLE_GENRES,
    DB_TABLE_TRACK_ARTISTS,
    DB_TABLE_TRACKS,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from music_assistant import MusicAssistant

# the translation key of the curated classical genre, which stays put when its name is edited
CLASSICAL_GENRE_KEY: Final[str] = "classical"


def _has_classical_genre(item: str, media_type: MediaType) -> str:
    """Return the condition that an item is mapped to the curated classical genre."""
    gm = DB_TABLE_GENRE_MEDIA_ITEM_MAPPING
    return f"""EXISTS (
        SELECT 1 FROM {gm} JOIN {DB_TABLE_GENRES} ON {DB_TABLE_GENRES}.item_id = {gm}.genre_id
        WHERE {gm}.media_id = {item}.item_id
        AND {gm}.media_type = '{media_type.value}'
        AND {DB_TABLE_GENRES}.translation_key = '{CLASSICAL_GENRE_KEY}'
        AND {DB_TABLE_GENRES}.content_type IS NULL
        AND {DB_TABLE_GENRES}.is_excluded = 0
        AND NOT EXISTS (
            SELECT 1 FROM {DB_TABLE_GENRE_MEDIA_ITEM_EXCLUSION} e
            WHERE e.genre_id = {gm}.genre_id
            AND e.media_id = {gm}.media_id
            AND e.media_type = {gm}.media_type))"""


def _track_is_classical_by_own_signal(track: str) -> str:
    """Return the condition that a track is classical by its own tag or genre."""
    return f"""({track}.classical_tag = 1 OR {_has_classical_genre(track, MediaType.TRACK)})"""


# album tracks only count by their own signals, as their album inheritance depends on this flag
ALBUM_IS_CLASSICAL: Final[str] = f"""({DB_TABLE_ALBUMS}.classical_tag = 1
    OR {_has_classical_genre(DB_TABLE_ALBUMS, MediaType.ALBUM)} OR COALESCE((
    SELECT 2 * SUM({_track_is_classical_by_own_signal("t")}) > COUNT(*)
    FROM {DB_TABLE_ALBUM_TRACKS} at JOIN {DB_TABLE_TRACKS} t ON t.item_id = at.track_id
    WHERE at.album_id = {DB_TABLE_ALBUMS}.item_id), 0))"""

TRACK_IS_CLASSICAL: Final[str] = f"""({_track_is_classical_by_own_signal(DB_TABLE_TRACKS)} OR
    EXISTS (SELECT 1 FROM {DB_TABLE_ALBUM_TRACKS} at
    JOIN {DB_TABLE_ALBUMS} a ON a.item_id = at.album_id
    WHERE at.track_id = {DB_TABLE_TRACKS}.item_id AND a.is_classical = 1))"""

ARTIST_IS_CLASSICAL: Final[str] = f"""EXISTS (
    SELECT 1 FROM {DB_TABLE_TRACK_ARTISTS} ta JOIN {DB_TABLE_TRACKS} t ON t.item_id = ta.track_id
    WHERE ta.artist_id = {DB_TABLE_ARTISTS}.item_id AND t.is_classical = 1)"""


class ClassicalClassifier:
    """Keeps the classical flags of library tracks, albums and artists current."""

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the classifier."""
        self.mass = mass

    async def has_classical_content(self) -> bool:
        """Return True when the library holds any classical track."""
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT EXISTS (SELECT 1 FROM {DB_TABLE_TRACKS} WHERE is_classical = 1) AS has_content",
            limit=0,
        )
        return bool(rows[0]["has_content"])

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
            DB_TABLE_TRACK_ARTISTS, "artist_id", "track_id", track_ids
        )
        # albums first and artists last, each flag builds on the one before
        await self._write_flags(DB_TABLE_ALBUMS, ALBUM_IS_CLASSICAL, album_ids)
        await self._write_flags(DB_TABLE_TRACKS, TRACK_IS_CLASSICAL, track_ids)
        await self._write_flags(DB_TABLE_ARTISTS, ARTIST_IS_CLASSICAL, artist_ids)

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
        elif media_type == MediaType.GENRE:
            await self.update_all()

    async def update_all(self) -> None:
        """Update the classical flags of the whole library."""
        await self._write_flags(DB_TABLE_ALBUMS, ALBUM_IS_CLASSICAL)
        await self._write_flags(DB_TABLE_TRACKS, TRACK_IS_CLASSICAL)
        await self._write_flags(DB_TABLE_ARTISTS, ARTIST_IS_CLASSICAL)

    @asynccontextmanager
    async def track_removal(self, track_id: int) -> AsyncGenerator[None]:
        """
        Keep the classical flags current while a track is removed from the library.

        :param track_id: Library id of the track about to be removed.
        """
        album_ids = await self._get_ids(DB_TABLE_ALBUM_TRACKS, "album_id", "track_id", {track_id})
        artist_ids = await self._get_ids(
            DB_TABLE_TRACK_ARTISTS, "artist_id", "track_id", {track_id}
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
