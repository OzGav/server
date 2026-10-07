"""Manage MediaItems of type Work."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ArtistRole, ExternalID, MediaType, WorkType
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import (
    Artist,
    ItemMapping,
    ItemMappingSummary,
    UniqueList,
    Work,
    WorkSummary,
)

from music_assistant.constants import (
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARRANGEMENTS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
)
from music_assistant.helpers.database import UNSET
from music_assistant.helpers.json import json_loads, serialize_to_json

from .base import SUMMARY_ARTIST_FIELDS, MediaControllerBase

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# the first composer of a work, for sorting works by composer
FIRST_COMPOSER_SORT_NAME = f"""(SELECT artists.search_sort_name FROM {DB_TABLE_WORK_ARTISTS}
    JOIN artists ON artists.item_id = {DB_TABLE_WORK_ARTISTS}.artist_id
    WHERE {DB_TABLE_WORK_ARTISTS}.work_id = {DB_TABLE_WORKS}.item_id
    AND {DB_TABLE_WORK_ARTISTS}.role = '{ArtistRole.COMPOSER.value}'
    ORDER BY {DB_TABLE_WORK_ARTISTS}.position, {DB_TABLE_WORK_ARTISTS}.rowid LIMIT 1)"""

# works sort by title, by composer in catalogue order or by year composed, works without a
# value come last
WORK_SORT_KEYS = {
    "name": f"{DB_TABLE_WORKS}.name_sort ASC",
    "name_desc": f"{DB_TABLE_WORKS}.name_sort DESC",
    "composer": f"{FIRST_COMPOSER_SORT_NAME} ASC NULLS LAST, "
    f"{DB_TABLE_WORKS}.catalog_sort ASC NULLS LAST, {DB_TABLE_WORKS}.name_sort ASC",
    "composer_desc": f"{FIRST_COMPOSER_SORT_NAME} DESC NULLS LAST, "
    f"{DB_TABLE_WORKS}.catalog_sort ASC NULLS LAST, {DB_TABLE_WORKS}.name_sort ASC",
    "composition_year": f"{DB_TABLE_WORKS}.composition_year ASC NULLS LAST, "
    f"{DB_TABLE_WORKS}.name_sort ASC",
    "composition_year_desc": f"{DB_TABLE_WORKS}.composition_year DESC NULLS LAST, "
    f"{DB_TABLE_WORKS}.name_sort ASC",
}

# The json_object arguments of a slim work mapping, selected from the works table aliased as w.
SUMMARY_WORK_FIELDS = "'item_id', w.item_id, 'name', w.name, 'sort_name', w.sort_name"


class WorksController(MediaControllerBase[Work]):
    """Controller managing MediaItems of type Work."""

    db_table = DB_TABLE_WORKS
    media_type = MediaType.WORK
    item_cls = Work
    summary_item_cls = WorkSummary

    @property
    def base_query(self) -> tuple[str, dict[str, Any]]:
        """Return the base SELECT query for works and its bound query params."""
        artist_fields = f"""
                'item_id', artists.item_id,
                'provider', 'library',
                'name', artists.name,
                'sort_name', artists.sort_name,
                'media_type', '{MediaType.ARTIST.value}',
                'external_ids', json({self._external_ids_query(MediaType.ARTIST, "artists")})"""
        work_fields = f"""
                'item_id', w.item_id,
                'provider', 'library',
                'name', w.name,
                'sort_name', w.sort_name,
                'media_type', '{MediaType.WORK.value}'"""
        query = f"""
        SELECT
            {DB_TABLE_WORKS}.*,
            {self._external_ids_query()} AS external_ids,
            {self._favorite_query()} AS favorite,
            {self._provider_mappings_query()} AS provider_mappings,
            {self._composers_query(artist_fields)} AS composers,
            {self._parent_work_query(work_fields)} AS parent_work,
            {self._arrangement_of_query(work_fields)} AS arrangement_of
            FROM {DB_TABLE_WORKS}"""
        return query, {}

    @property
    def summary_query(self) -> tuple[str, dict[str, Any]]:
        """Return the slim SELECT query used for work summary listings."""
        query = f"""
        SELECT
            {self._summary_base_columns()},
            {DB_TABLE_WORKS}.version,
            {DB_TABLE_WORKS}.catalog_numbers,
            {DB_TABLE_WORKS}.work_type,
            {DB_TABLE_WORKS}.composition_year,
            {self._provider_mappings_query()} AS provider_mappings,
            {self._composers_query(SUMMARY_ARTIST_FIELDS)} AS composers,
            {self._parent_work_query(SUMMARY_WORK_FIELDS)} AS parent_work,
            {self._arrangement_of_query(SUMMARY_WORK_FIELDS)} AS arrangement_of
            FROM {DB_TABLE_WORKS}"""
        return query, {}

    async def remove_item_from_library(self, item_id: str | int, recursive: bool = True) -> None:
        """
        Delete a work from the library.

        Its tracks stay in the library without a work; works that name it as their parent
        or as the original of an arrangement lose that link.

        :param item_id: Database ID of the work to remove.
        :param recursive: Unused for works, kept for base-class compatibility.
        """
        db_id = int(item_id)  # ensure integer
        await super().remove_item_from_library(db_id)
        # the movement fields stay on the tracks as they describe the recording
        await self.mass.music.database.execute_write(
            f"UPDATE {DB_TABLE_TRACKS} SET work_id = NULL WHERE work_id = :db_id",
            {"db_id": db_id},
        )
        await self.mass.music.database.execute_write(
            f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = NULL WHERE parent_work_id = :db_id",
            {"db_id": db_id},
        )
        await self.mass.music.database.delete(DB_TABLE_WORK_ARTISTS, {"work_id": db_id})
        await self.mass.music.database.execute_write(
            f"DELETE FROM {DB_TABLE_WORK_ARRANGEMENTS} "
            "WHERE work_id = :db_id OR source_work_id = :db_id",
            {"db_id": db_id},
        )

    async def library_count(self, favorite_only: bool = False) -> int:
        """
        Return the total number of works in the library.

        Never restricted by the current user's provider filter.

        :param favorite_only: Only count the works the current user likes.
        """
        # works are library-only items without provider mappings, so the user's
        # provider filter does not apply here
        if favorite_only:
            query_params: dict[str, Any] = {}
            clause = self._favorite_filter_clause(query_params, True)
            sql_query = f"SELECT item_id FROM {self.db_table} WHERE {clause}"
            return await self.mass.music.database.get_count_from_query(sql_query, query_params)
        return await self.mass.music.database.get_count(self.db_table)

    async def library_items(
        self,
        favorite: bool | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        order_by: str = "name",
        provider: str | list[str] | None = None,
        genre: int | list[int] | None = None,
        played_only: bool = False,
        *,
        summary: bool = True,
        **kwargs: Any,
    ) -> list[Work]:
        """
        Get the works in the library.

        :param favorite: Only include the current user's likes (True) or dislikes (False).
        :param search: Filter by search query.
        :param limit: Maximum number of items to return.
        :param offset: Number of items to skip.
        :param order_by: Order by 'name' (title), 'composer' (then catalogue number) or
            'composition_year', each also with a '_desc' suffix. Other keys fall back to 'name'.
        :param provider: Ignored, works are library-only items.
        :param genre: Filter by genre id(s).
        :param played_only: Works are not played themselves, so this returns no items.
        :param summary: When True (default), return slim summary items containing only the
            fields needed for a list view. Set to False to get fully hydrated items.
        """
        if played_only:
            return []
        if order_by not in WORK_SORT_KEYS:
            order_by = "name"
        return await self.get_library_items_by_query(
            favorite=favorite,
            search=search,
            limit=limit,
            offset=offset,
            order_by=order_by,
            genre_ids=genre,
            summary=summary,
        )

    async def match_providers(self, db_item: Work) -> None:
        """No provider matching for works, they are library-only items."""
        return

    async def get_library_work_id(self, work: Work | ItemMapping) -> int | None:
        """
        Return the library id of the given work, or None when the library does not hold it.

        :param work: A library work (mapping), or a work identified by its MusicBrainz work id.
        """
        if work.provider == "library":
            return int(work.item_id)
        if (mbid := work.get_external_id(ExternalID.MB_WORK)) and (
            library_work := await self.get_library_item_by_external_id(mbid, ExternalID.MB_WORK)
        ):
            return int(library_work.item_id)
        return None

    async def _add_library_item(self, item: Work, overwrite_existing: bool = False) -> int:
        """Add a new work record to the database."""
        db_id = await self.mass.music.database.insert(
            self.db_table,
            {
                "name": item.name,
                "sort_name": item.sort_name,
                "version": item.version,
                "catalog_numbers": serialize_to_json(item.catalog_numbers),
                "catalog_sort": _catalog_sort_key(item.catalog_numbers),
                "name_sort": _natural_sort_key(create_safe_string(item.name, True, True)),
                "work_type": item.work_type,
                "composition_year": item.composition_year,
                "language": item.language,
                "musical_key": item.musical_key,
                "parent_work_id": await self.get_library_work_id(item.parent_work)
                if item.parent_work
                else None,
                "metadata": serialize_to_json(item.metadata),
                "search_name": create_safe_string(item.name, True, True),
                "search_sort_name": create_safe_string(item.sort_name or "", True, True),
                "timestamp_added": int(item.date_added.timestamp()) if item.date_added else UNSET,
            },
        )
        await self.set_external_ids(db_id, item.external_ids)
        await self._set_composers(db_id, item.composers)
        await self._set_arrangements(db_id, item.arrangement_of)
        self.logger.debug("added %s to database (id: %s)", item.name, db_id)
        return db_id

    async def _update_library_item(
        self, item_id: str | int, update: Work, overwrite: bool = False
    ) -> None:
        """Update an existing work record in the database."""
        db_id = int(item_id)  # ensure integer
        cur_item = await self.get_library_item(db_id)
        metadata = update.metadata if overwrite else cur_item.metadata.update(update.metadata)
        cur_item.external_ids.update(update.external_ids)
        name = update.name if overwrite else cur_item.name
        sort_name = update.sort_name if overwrite else cur_item.sort_name or update.sort_name
        cur_parent_id = int(cur_item.parent_work.item_id) if cur_item.parent_work else None
        new_parent_id = (
            await self.get_library_work_id(update.parent_work) if update.parent_work else None
        )
        # like the composers, a source without a value keeps the stored one
        if overwrite:
            catalog_numbers = update.catalog_numbers or cur_item.catalog_numbers
            parent_work_id = new_parent_id or cur_parent_id
        else:
            catalog_numbers = list(dict.fromkeys(cur_item.catalog_numbers + update.catalog_numbers))
            parent_work_id = cur_parent_id or new_parent_id
        await self.mass.music.database.update(
            self.db_table,
            {"item_id": db_id},
            {
                "name": name,
                "sort_name": sort_name,
                "version": update.version if overwrite else cur_item.version or update.version,
                "catalog_numbers": serialize_to_json(catalog_numbers),
                "catalog_sort": _catalog_sort_key(catalog_numbers),
                "name_sort": _natural_sort_key(create_safe_string(name, True, True)),
                "work_type": _merge_value(cur_item.work_type, update.work_type, overwrite),
                "composition_year": _merge_value(
                    cur_item.composition_year, update.composition_year, overwrite
                ),
                "language": _merge_value(cur_item.language, update.language, overwrite),
                "musical_key": _merge_value(cur_item.musical_key, update.musical_key, overwrite),
                "parent_work_id": parent_work_id,
                "metadata": serialize_to_json(metadata),
                "search_name": create_safe_string(name, True, True),
                "search_sort_name": create_safe_string(sort_name or "", True, True),
                "timestamp_added": int(update.date_added.timestamp())
                if update.date_added
                else UNSET,
            },
        )
        await self.set_external_ids(
            db_id, update.external_ids if overwrite else cur_item.external_ids
        )
        composers = update.composers if overwrite else cur_item.composers + update.composers
        await self._set_composers(db_id, composers, overwrite=overwrite)
        await self._set_arrangements(db_id, update.arrangement_of, overwrite=overwrite)
        self.logger.debug("updated %s in database: (id %s)", update.name, db_id)

    async def _set_composers(
        self, db_id: int, composers: Iterable[Artist | ItemMapping], overwrite: bool = False
    ) -> None:
        """Store the composers of a work in the given order, keeping the stored ones if none."""
        artist_ids = [await self._get_library_artist_id(x, overwrite) for x in composers]
        if not artist_ids:
            return
        if overwrite:
            await self.mass.music.database.delete(
                DB_TABLE_WORK_ARTISTS, {"work_id": db_id, "role": ArtistRole.COMPOSER.value}
            )
        # different (provider) artists may resolve to the same library artist
        for position, artist_id in enumerate(dict.fromkeys(artist_ids)):
            await self.mass.music.database.insert_or_replace(
                DB_TABLE_WORK_ARTISTS,
                {
                    "work_id": db_id,
                    "artist_id": artist_id,
                    "role": ArtistRole.COMPOSER.value,
                    "position": position,
                },
            )

    async def _set_arrangements(
        self, db_id: int, source_works: Iterable[Work | ItemMapping], overwrite: bool = False
    ) -> None:
        """Store the library works a work arranges, keeping the stored ones if none resolve."""
        source_ids = [
            source_id
            for source_work in source_works
            if (source_id := await self.get_library_work_id(source_work))
        ]
        if not source_ids:
            return
        if overwrite:
            await self.mass.music.database.delete(DB_TABLE_WORK_ARRANGEMENTS, {"work_id": db_id})
        for source_id in dict.fromkeys(source_ids):
            await self.mass.music.database.execute_write(
                f"INSERT OR IGNORE INTO {DB_TABLE_WORK_ARRANGEMENTS}(work_id, source_work_id) "
                "VALUES (:work_id, :source_work_id)",
                {"work_id": db_id, "source_work_id": source_id},
            )

    def _composers_query(self, artist_fields: str) -> str:
        """Return a subquery selecting the composers as a JSON array, in credited order."""
        # relies on the same json_group_array ordering as _main_artists_query
        return f"""(SELECT JSON_GROUP_ARRAY(json(artist)) FROM (
            SELECT json_object({artist_fields}) AS artist
            FROM {DB_TABLE_WORK_ARTISTS}
            JOIN artists ON artists.item_id = {DB_TABLE_WORK_ARTISTS}.artist_id
            WHERE {DB_TABLE_WORK_ARTISTS}.work_id = {DB_TABLE_WORKS}.item_id
            AND {DB_TABLE_WORK_ARTISTS}.role = '{ArtistRole.COMPOSER.value}'
            ORDER BY {DB_TABLE_WORK_ARTISTS}.position, {DB_TABLE_WORK_ARTISTS}.rowid))"""

    def _parent_work_query(self, work_fields: str) -> str:
        """Return a subquery selecting the parent work mapping as a JSON object."""
        return f"""(SELECT json_object({work_fields})
            FROM {DB_TABLE_WORKS} w WHERE w.item_id = {DB_TABLE_WORKS}.parent_work_id)"""

    def _arrangement_of_query(self, work_fields: str) -> str:
        """Return a subquery selecting the arranged source works as a JSON array."""
        return f"""(SELECT JSON_GROUP_ARRAY(json(work)) FROM (
            SELECT json_object({work_fields}) AS work
            FROM {DB_TABLE_WORK_ARRANGEMENTS} wa
            JOIN {DB_TABLE_WORKS} w ON w.item_id = wa.source_work_id
            WHERE wa.work_id = {DB_TABLE_WORKS}.item_id
            ORDER BY wa.rowid))"""

    def _sort_key(self, order_by: str) -> str | None:
        """Return the ORDER BY expression for a works listing sort, None when unknown."""
        return WORK_SORT_KEYS.get(order_by)

    def _summary_base_columns(self) -> str:
        """Return the SELECT columns shared by every work summary query."""
        return f"""
            {self.db_table}.item_id,
            {self.db_table}.name,
            {self.db_table}.sort_name,
            {self._favorite_query()} AS favorite,
            {self.db_table}.search_name AS search_name,
            {self.db_table}.search_sort_name AS search_sort_name,
            {self.db_table}.timestamp_added AS timestamp_added,
            {self.db_table}.timestamp_modified AS timestamp_modified,
            json_extract({self.db_table}.metadata, '$.images') AS images,
            json_extract({self.db_table}.metadata, '$.collections') AS collections"""

    def _parse_summary_row(
        self, db_row: Mapping[str, Any], hidden_sources: set[str]
    ) -> WorkSummary:
        """Parse a raw summary db row into a WorkSummary object."""
        item = cast("WorkSummary", super()._parse_summary_row(db_row, hidden_sources))
        item.version = db_row["version"] or ""
        item.catalog_numbers = json_loads(db_row["catalog_numbers"])
        item.work_type = WorkType(work_type) if (work_type := db_row["work_type"]) else None
        item.composition_year = db_row["composition_year"]
        item.composers = UniqueList(
            _summary_mappings(json_loads(db_row["composers"]), MediaType.ARTIST)
        )
        if raw_parent := db_row["parent_work"]:
            item.parent_work = _summary_mappings([json_loads(raw_parent)], MediaType.WORK)[0]
        item.arrangement_of = UniqueList(
            _summary_mappings(json_loads(db_row["arrangement_of"]), MediaType.WORK)
        )
        return item


def _catalog_sort_key(catalog_numbers: list[str]) -> str | None:
    """Return the natural sort key of the first catalogue number, None when there is none."""
    return _natural_sort_key(catalog_numbers[0].casefold()) if catalog_numbers else None


def _natural_sort_key(value: str) -> str:
    """Return the given text with its numbers zero padded, so text order follows number order."""
    return re.sub(r"\d+", lambda match: match.group().zfill(8), value)


def _merge_value[T](current: T | None, update: T | None, overwrite: bool) -> T | None:
    """Return the value to store, the update winning on overwrite unless it is empty."""
    return (update or current) if overwrite else (current or update)


def _summary_mappings(
    raw_mappings: list[dict[str, Any]], media_type: MediaType
) -> list[ItemMappingSummary]:
    """Build the slim library mappings of a summary row."""
    return [
        ItemMappingSummary(
            media_type=media_type,
            item_id=str(raw_mapping["item_id"]),
            provider="library",
            name=raw_mapping["name"],
            sort_name=raw_mapping["sort_name"],
        )
        for raw_mapping in raw_mappings
    ]
