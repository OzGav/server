"""Manage MediaItems of type Work."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final, cast

from music_assistant_models.enums import (
    ArtistRole,
    EventType,
    ExternalID,
    ListingType,
    MediaType,
    SortDirection,
    SortField,
    WorkType,
)
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
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
from music_assistant.controllers.music.helpers import search_name_match_clause
from music_assistant.controllers.music.sorting import get_listing_default_direction
from music_assistant.helpers.database import UNSET
from music_assistant.helpers.json import json_loads, serialize_to_json
from music_assistant.helpers.work_titles import parse_catalog_numbers

from .base import SUMMARY_ARTIST_FIELDS, SUPPRESS_MEDIA_ITEM_UPDATES, MediaControllerBase

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# the first composer of a work, for sorting works by composer
FIRST_COMPOSER_SORT_NAME = f"""(SELECT artists.search_sort_name FROM {DB_TABLE_WORK_ARTISTS}
    JOIN artists ON artists.item_id = {DB_TABLE_WORK_ARTISTS}.artist_id
    WHERE {DB_TABLE_WORK_ARTISTS}.work_id = {DB_TABLE_WORKS}.item_id
    AND {DB_TABLE_WORK_ARTISTS}.role = '{ArtistRole.COMPOSER.value}'
    ORDER BY {DB_TABLE_WORK_ARTISTS}.position, {DB_TABLE_WORK_ARTISTS}.rowid LIMIT 1)"""

# The json_object arguments of a slim work mapping, selected from the works table aliased as w.
SUMMARY_WORK_FIELDS = "'item_id', w.item_id, 'name', w.name, 'sort_name', w.sort_name"


# the sort fields a works listing renders itself
WORK_SORT_FIELDS: Final[frozenset[SortField]] = frozenset(
    {SortField.NAME, SortField.COMPOSER, SortField.YEAR}
)


def work_sort_sql(field: SortField, direction: SortDirection | None) -> str:
    """
    Return the ORDER BY clause for a works sort on the title, composer or year composed.

    Works without a composer or year come last. Sorting by composer orders each composer's
    works by catalog number, then title.

    :param field: One of WORK_SORT_FIELDS.
    :param direction: The sort direction, the works listing's default when None.
    """
    default = get_listing_default_direction(ListingType.LIBRARY_WORKS, field)
    order = (direction or default).value.upper()
    name_sort = f"{DB_TABLE_WORKS}.name_sort"
    if field == SortField.COMPOSER:
        return (
            f"{FIRST_COMPOSER_SORT_NAME} {order} NULLS LAST, "
            f"{DB_TABLE_WORKS}.catalog_sort ASC NULLS LAST, {name_sort} ASC"
        )
    if field == SortField.YEAR:
        return f"{DB_TABLE_WORKS}.composition_year {order} NULLS LAST, {name_sort} ASC"
    return f"{name_sort} {order}"


def work_search_sql(search_term: str, query_params: dict[str, Any]) -> str:
    """
    Return the condition matching the works whose title or a part's title holds a search term.

    A part itself never matches, it is found through the work it is a part of.

    :param search_term: The normalized search term.
    :param query_params: Query params dict; the term's bound param is added to it.
    """
    # the unaliased works table of the subquery is the one the match refers to
    match = search_name_match_clause(DB_TABLE_WORKS, search_term, "search", query_params)
    return (
        f"{DB_TABLE_WORKS}.item_id IN (SELECT COALESCE(parent_work_id, item_id) "
        f"FROM {DB_TABLE_WORKS} WHERE {match})"
    )


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
        # the related works are slim works that carry their composers
        related_work_fields = f"""{work_fields},
                'provider_mappings', json('[]'),
                'composers', json({self._composers_query(artist_fields, "w.item_id")})"""
        query = f"""
        SELECT
            {DB_TABLE_WORKS}.*,
            {self._external_ids_query()} AS external_ids,
            {self._favorite_query()} AS favorite,
            {self._provider_mappings_query()} AS provider_mappings,
            {self._composers_query(artist_fields)} AS composers,
            {self._parent_work_query(work_fields)} AS parent_work,
            {self._arrangement_of_query(related_work_fields)} AS arrangement_of,
            {self._arrangements_query(related_work_fields)} AS arrangements,
            {self._parts_query(related_work_fields)} AS parts
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
        track_rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id FROM {DB_TABLE_TRACKS} WHERE work_id = :db_id",
            {"db_id": db_id},
            limit=0,
        )
        await super().remove_item_from_library(db_id)
        # the movement fields stay on the tracks as they describe the recording
        await self.mass.music.database.execute_write(
            f"UPDATE {DB_TABLE_TRACKS} SET work_id = NULL WHERE work_id = :db_id",
            {"db_id": db_id},
        )
        # the work's type may have been what made its tracks classical
        await self.mass.music.classification.update(
            track_ids=[int(row["item_id"]) for row in track_rows]
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

        Counts the works a works listing shows, never restricted by the current user's
        provider filter.

        :param favorite_only: Only count the works the current user likes.
        """
        # works are library-only items without provider mappings, so the user's
        # provider filter does not apply here
        query_params: dict[str, Any] = {}
        clauses = self.listing_filter(query_params)
        if favorite_only:
            clauses.append(self._favorite_filter_clause(query_params, True))
        sql_query = f"SELECT item_id FROM {self.db_table} WHERE {' AND '.join(clauses)}"
        return await self.mass.music.database.get_count_from_query(sql_query, query_params)

    async def library_items(  # noqa: PLR0913
        self,
        favorite: bool | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        order_by: str | None = None,
        provider: str | list[str] | None = None,
        genre: int | list[int] | None = None,
        played_only: bool = False,
        *,
        sort_field: SortField | None = None,
        sort_direction: SortDirection | None = None,
        summary: bool = True,
        **kwargs: Any,
    ) -> list[Work]:
        """
        Get the works in the library.

        The parts of a work are listed through the work they are a part of, which a search
        matching a part also finds.

        :param favorite: Only include the current user's likes (True) or dislikes (False).
        :param search: Filter by search query.
        :param limit: Maximum number of items to return.
        :param offset: Number of items to skip.
        :param order_by: DEPRECATED - use sort_field and sort_direction instead.
        :param provider: Ignored, works are library-only items.
        :param genre: Filter by genre id(s).
        :param played_only: Works are not played themselves, so this returns no items.
        :param sort_field: Sort on the composer (then catalogue number, the default), the
            title or the year composed.
        :param sort_direction: Sort direction, the field's default when omitted.
        :param summary: When True (default), return slim summary items containing only the
            fields needed for a list view. Set to False to get fully hydrated items.
        """
        field, direction = self.resolve_sort(sort_field, sort_direction, order_by)
        if played_only:
            return []
        query_params: dict[str, Any] = {}
        return await self.get_library_items_by_query(
            favorite=favorite,
            search=search,
            limit=limit,
            offset=offset,
            sort_field=field,
            sort_direction=direction,
            extra_query_parts=self.listing_filter(query_params),
            extra_query_params=query_params,
            genre_ids=genre,
            summary=summary,
        )

    async def match_providers(self, db_item: Work) -> None:
        """No provider matching for works, they are library-only items."""
        return

    async def merge_library_works_by_mbid(self, work_id: int, mbid: str) -> int:
        """
        Merge the library works that are the given MusicBrainz work into one.

        The oldest work with tracks of its own survives, or the oldest when none has tracks.

        :param work_id: Library id of a work that is the MusicBrainz work.
        :param mbid: The MusicBrainz work id.
        :return: The library id the given work is found under after the merge.
        """
        same_mbid = await self.get_library_items_by_external_id(
            mbid, ExternalID.MB_WORK, limit=None
        )
        all_ids = sorted({work_id, *(int(x.item_id) for x in same_mbid)})
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT work_id FROM {DB_TABLE_TRACKS} WHERE work_id IN :ids",
            {"ids": all_ids},
            limit=0,
        )
        with_tracks = {int(row["work_id"]) for row in rows}
        # the survivor keeps its name and own fields
        target_id = next((x for x in all_ids if x in with_tracks), all_ids[0])
        source_ids = [x for x in all_ids if x != target_id]
        for source_id in source_ids:
            try:
                await self.merge_library_items(target_id, source_id)
            except MediaNotFoundError:
                continue  # already merged by a concurrent run
            except InvalidDataError as err:
                self.logger.info("Not merging works with MusicBrainz ID %s: %s", mbid, err)
                if source_id == work_id:
                    return work_id
        return target_id

    async def create_library_item(self, item: Work) -> Work:
        """
        Add a work to the library as a new work, never matched to a stored one.

        :param item: The work to add.
        """
        async with self.mass.music.database.deferred_commit(), self._db_add_lock:
            db_id = await self._add_library_item(item)
        library_item = await self.get_library_item(db_id)
        if not SUPPRESS_MEDIA_ITEM_UPDATES.get():
            self.mass.signal_event(EventType.MEDIA_ITEM_ADDED, library_item.uri, library_item)
        return library_item

    async def add_arrangement(self, work_id: int, source_work_id: int) -> None:
        """
        Link a library work as an arrangement of another library work.

        :param work_id: Library id of the arrangement.
        :param source_work_id: Library id of the work it arranges.
        """
        if work_id == source_work_id:
            return
        await self.mass.music.database.execute_write(
            f"INSERT OR IGNORE INTO {DB_TABLE_WORK_ARRANGEMENTS}(work_id, source_work_id) "
            "VALUES (:work_id, :source_work_id)",
            {"work_id": work_id, "source_work_id": source_work_id},
        )

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

    async def get_or_add_library_work_id(self, work: Work | ItemMapping) -> int | None:
        """
        Return the library id of a track's work, adding a work a provider describes if needed.

        A work given as a full Work is added together with the work it is a part of, matched
        to the library works by name and composer so the tracks of one work share it.

        :param work: The work a track names.
        """
        if (work_id := await self.get_library_work_id(work)) or not isinstance(work, Work):
            return work_id
        # matched on the library artists, as a provider may name a composer differently
        work.composers = UniqueList(
            [
                ItemMapping(
                    media_type=MediaType.ARTIST,
                    item_id=str(await self._get_library_artist_id(x)),
                    provider="library",
                    name=x.name,
                )
                for x in work.composers
            ]
        )
        if work.parent_work and not await self.get_library_work_id(work.parent_work):
            parent = await self.add_item_to_library(
                Work(
                    item_id=work.parent_work.item_id,
                    provider=work.parent_work.provider,
                    name=work.parent_work.name,
                    provider_mappings=set(),
                    composers=UniqueList(work.composers),
                    catalog_numbers=parse_catalog_numbers(work.parent_work.name),
                )
            )
            work.parent_work = ItemMapping.from_item(parent)
        return int((await self.add_item_to_library(work)).item_id)

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
        work_type = _merge_value(cur_item.work_type, update.work_type, overwrite)
        # like the composers, a source without a value keeps the stored one
        if overwrite:
            catalog_numbers = update.catalog_numbers or cur_item.catalog_numbers
            parent_work_id = new_parent_id or cur_parent_id
        else:
            catalog_numbers = _unique_catalog_numbers(
                cur_item.catalog_numbers + update.catalog_numbers
            )
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
                "work_type": work_type,
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
        if work_type != cur_item.work_type:
            await self.mass.music.classification.update_item(self.media_type, db_id)
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
            await self.add_arrangement(db_id, source_id)

    async def _validate_library_item_merge(self, target: Work, source: Work) -> None:
        """Validate that two works are not different MusicBrainz works."""
        await super()._validate_library_item_merge(target, source)
        if target.mbid and source.mbid and target.mbid != source.mbid:
            msg = (
                f"Cannot merge work '{source.name}' into '{target.name}': "
                "they are different MusicBrainz works."
            )
            raise InvalidDataError(msg)

    async def _merge_play_statistics(
        self, target_row: Mapping[str, Any], source_row: Mapping[str, Any]
    ) -> None:
        """Keep nothing, works are not played themselves."""
        return

    async def _merge_library_item_references(self, target_id: int, source_id: int) -> None:
        """Move the tracks, parts and arrangement links of a merged work onto the target."""
        database = self.mass.music.database
        ids = {"target_id": target_id, "source_id": source_id}
        # the movement fields stay on the tracks as they describe the recording
        await database.execute_write(
            f"UPDATE {DB_TABLE_TRACKS} SET work_id = :target_id WHERE work_id = :source_id", ids
        )
        await database.execute_write(
            f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = :target_id "
            "WHERE parent_work_id = :source_id",
            ids,
        )
        # the two works may have been part of one another
        await database.execute_write(
            f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = NULL "
            "WHERE item_id = :target_id AND parent_work_id = :target_id",
            ids,
        )
        for column, other_column in (("work_id", "source_work_id"), ("source_work_id", "work_id")):
            await database.execute_write(
                f"INSERT OR IGNORE INTO {DB_TABLE_WORK_ARRANGEMENTS}({column}, {other_column}) "
                f"SELECT :target_id, {other_column} FROM {DB_TABLE_WORK_ARRANGEMENTS} "
                f"WHERE {column} = :source_id",
                ids,
            )
        await database.execute_write(
            f"DELETE FROM {DB_TABLE_WORK_ARRANGEMENTS} WHERE work_id = :source_id "
            "OR source_work_id = :source_id OR work_id = source_work_id",
            ids,
        )
        await self.mass.music.tracks.credit_work_composers(target_id)

    def _composers_query(
        self, artist_fields: str, work_id: str = f"{DB_TABLE_WORKS}.item_id"
    ) -> str:
        """Return a subquery selecting a work's composers as a JSON array, in credited order."""
        # relies on the same json_group_array ordering as _main_artists_query
        return f"""(SELECT JSON_GROUP_ARRAY(json(artist)) FROM (
            SELECT json_object({artist_fields}) AS artist
            FROM {DB_TABLE_WORK_ARTISTS}
            JOIN artists ON artists.item_id = {DB_TABLE_WORK_ARTISTS}.artist_id
            WHERE {DB_TABLE_WORK_ARTISTS}.work_id = {work_id}
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

    def _arrangements_query(self, work_fields: str) -> str:
        """Return a subquery selecting the works arranged from this one as a JSON array."""
        return f"""(SELECT JSON_GROUP_ARRAY(json(work)) FROM (
            SELECT json_object({work_fields}) AS work
            FROM {DB_TABLE_WORK_ARRANGEMENTS} wa
            JOIN {DB_TABLE_WORKS} w ON w.item_id = wa.work_id
            WHERE wa.source_work_id = {DB_TABLE_WORKS}.item_id
            ORDER BY wa.rowid))"""

    def _parts_query(self, work_fields: str) -> str:
        """Return a subquery selecting the works that are a part of this one as a JSON array."""
        return f"""(SELECT JSON_GROUP_ARRAY(json(work)) FROM (
            SELECT json_object({work_fields}) AS work
            FROM {DB_TABLE_WORKS} w WHERE w.parent_work_id = {DB_TABLE_WORKS}.item_id
            ORDER BY w.name_sort, w.item_id))"""

    def _listing_filter_clause(self, query_params: dict[str, Any]) -> str | None:
        """Return the condition that lists the parts of a work through that work."""
        visible = self.mass.music.tracks.visible_library_clause(query_params)
        # a work with parts is listed while it or one of its parts has a track the user sees
        return f"""({DB_TABLE_WORKS}.parent_work_id IS NULL
            AND (NOT EXISTS (SELECT 1 FROM {DB_TABLE_WORKS} p
                WHERE p.parent_work_id = {DB_TABLE_WORKS}.item_id)
            OR EXISTS (SELECT 1 FROM {DB_TABLE_TRACKS}
                WHERE {DB_TABLE_TRACKS}.work_id = {DB_TABLE_WORKS}.item_id AND {visible})
            OR EXISTS (SELECT 1 FROM {DB_TABLE_WORKS} p
                JOIN {DB_TABLE_TRACKS} ON {DB_TABLE_TRACKS}.work_id = p.item_id
                WHERE p.parent_work_id = {DB_TABLE_WORKS}.item_id AND {visible})))"""

    def _search_filter_clause(self, search: str, query_params: dict[str, Any]) -> str:
        """Return the search condition, which a work also meets through a matching part."""
        return work_search_sql(search, query_params)

    def _get_sort_sql(self, field: SortField, direction: SortDirection | None) -> str:
        """Return the ORDER BY clause for a works sort field."""
        if field in WORK_SORT_FIELDS:
            return work_sort_sql(field, direction)
        return super()._get_sort_sql(field, direction)

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


def _unique_catalog_numbers(catalog_numbers: list[str]) -> list[str]:
    """Return the catalog numbers without repeats, keeping the first spelling of each."""
    # spellings differing only in case, dots or spacing ("Op. 67", "op.67") are one number
    unique: dict[str, str] = {}
    for number in catalog_numbers:
        unique.setdefault(re.sub(r"[.\s]", "", number.casefold()), number)
    return list(unique.values())


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
