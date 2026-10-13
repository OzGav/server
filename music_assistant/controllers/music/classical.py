"""
Classical browse API of the Music controller.

Serves the composer, performer, work, recording and other track listings of the Classical
view. Every listing only considers classical library tracks.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

from music_assistant_models.auth import Scope
from music_assistant_models.enums import (
    AlbumType,
    ArtistRole,
    ExternalID,
    ImageType,
    ListingType,
    MediaType,
    SortDirection,
    SortField,
    WorkType,
)
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import (
    ArtistSummary,
    ClassicalComposer,
    ClassicalPerformer,
    ClassicalWorkEntry,
    Credit,
    ItemMapping,
    MediaItemImage,
    MediaItemType,
    Recording,
    Track,
    Work,
    WorkSummary,
)

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_ALBUMS,
    DB_TABLE_ARTISTS,
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACK_ARTISTS,
    DB_TABLE_TRACK_CREDITS,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.constants import BASE_SORT_FIELD_SQL
from music_assistant.controllers.music.helpers import preferred_thumb, search_name_match_clause
from music_assistant.controllers.music.listing import resolve_listing_sort
from music_assistant.controllers.music.media.works import work_search_sql, work_sort_sql
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.json import json_loads
from music_assistant.helpers.provider_access import hidden_music_sources

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from music_assistant import MusicAssistant
    from music_assistant.controllers.music.media.base import MediaControllerBase

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
ENSEMBLE_ROLES: Final[tuple[ArtistRole, ...]] = (
    ArtistRole.ORCHESTRA,
    ArtistRole.ENSEMBLE,
    ArtistRole.CHOIR,
)
# the roles that tell one performance of a work from another
LEADING_ROLES: Final[tuple[ArtistRole, ...]] = (ArtistRole.CONDUCTOR, *ENSEMBLE_ROLES)
# the leading roles that stay the same across a performance, where choirs come and go
DIRECTING_ROLES: Final[tuple[ArtistRole, ...]] = (ArtistRole.CONDUCTOR, ArtistRole.ORCHESTRA)

_SQL_PERFORMING_ROLES = ", ".join(f"'{role.value}'" for role in PERFORMING_ROLES)

# the work types a title names, the first match wins; opera is left out as titles use it
# loosely, such as "La scala di seta Opera: Overture"
_TITLE_WORK_TYPES: Final[tuple[tuple[WorkType, re.Pattern[str]], ...]] = tuple(
    (work_type, re.compile(pattern))
    for work_type, pattern in (
        (WorkType.QUARTET, r"\bstring quartets?\b"),
        (WorkType.SYMPHONIC_POEM, r"\bsymphonic poems?\b"),
        (WorkType.SYMPHONY, r"\bsympho(ny|nie|nies)\b|\bsinfonie\b"),
        (WorkType.CONCERTO, r"\bconcert(o|os|i)\b"),
        (WorkType.SONATA, r"\bsonat(a|as|e)\b"),
        (WorkType.MASS, r"\b(mass|missa|messe)\b"),
        (WorkType.OVERTURE, r"\bovertures?\b|\bouvertüre\b"),
        (WorkType.SUITE, r"\bsuites?\b"),
        (WorkType.CANTATA, r"\bcantatas?\b|\bkantate\b"),
        (WorkType.ORATORIO, r"\boratorios?\b"),
        (WorkType.MOTET, r"\bmotets?\b"),
        (WorkType.MADRIGAL, r"\bmadrigals?\b"),
        (WorkType.PARTITA, r"\bpartitas?\b"),
        (WorkType.ETUDE, r"\b[ée]tudes?\b"),
    )
)
# a nickname in double quotes or guillemets, such as "Concert sans orchestre"
_QUOTED_NICKNAME = re.compile(r'"[^"]*"|«[^»]*»')

# a type label in brackets, such as "(Opera)", unless a quoted name follows as in an excerpt,
# or as the last word after a bracket or quote, such as "Cavalleria Rusticana (Rustic Chivalry)
# Opera", so a name such as "The Phantom of the Opera" never counts
_TYPE_LABEL: Final[re.Pattern[str]] = re.compile(
    r'\((opera|ballet|operetta)\)(?!\s*["«])|[)"”»]\s*(opera|ballet|operetta)\W*$',
    re.IGNORECASE,
)

# the performing credits on classical tracks aliased as t, which callers narrow to the visible
# ones, where a track without any counts its main artists that did not compose it as performers
PERFORMER_CREDITS: Final[str] = f"""FROM (
        SELECT p.track_id, p.artist_id, p.role
        FROM {DB_TABLE_TRACK_CREDITS} p WHERE p.role IN ({_SQL_PERFORMING_ROLES})
        UNION ALL
        SELECT m.track_id, m.artist_id, '{ArtistRole.PERFORMER.value}' AS role
        FROM {DB_TABLE_TRACK_ARTISTS} m
        WHERE NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_CREDITS} p
                WHERE p.track_id = m.track_id AND p.role IN ({_SQL_PERFORMING_ROLES}))
            AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_CREDITS} c
                WHERE c.track_id = m.track_id AND c.artist_id = m.artist_id
                AND c.role = '{ArtistRole.COMPOSER.value}')
    ) ta
    JOIN {DB_TABLE_TRACKS} t ON t.item_id = ta.track_id
    WHERE t.is_classical = 1"""

# the work a work aliased as w is listed as, the work it is a part of or else itself
LISTED_WORK: Final[str] = "COALESCE(w.parent_work_id, w.item_id)"

# the composers of each listed work, its own or else those of its parts
LISTED_WORK_COMPOSERS: Final[str] = f"""
    SELECT {LISTED_WORK} AS work_id, wa.artist_id
    FROM {DB_TABLE_WORKS} w
    JOIN {DB_TABLE_WORK_ARTISTS} wa ON wa.work_id = w.item_id
        AND wa.role = '{ArtistRole.COMPOSER.value}'
    WHERE w.parent_work_id IS NULL OR NOT EXISTS (SELECT 1 FROM {DB_TABLE_WORK_ARTISTS} pa
        WHERE pa.work_id = w.parent_work_id AND pa.role = '{ArtistRole.COMPOSER.value}')"""

# the year of the album a track is listed with
TRACK_ALBUM_YEAR: Final[str] = f"""(SELECT {DB_TABLE_ALBUMS}.year FROM {DB_TABLE_ALBUM_TRACKS}
    JOIN {DB_TABLE_ALBUMS} ON {DB_TABLE_ALBUMS}.item_id = {DB_TABLE_ALBUM_TRACKS}.album_id
    WHERE {DB_TABLE_ALBUM_TRACKS}.track_id = {DB_TABLE_TRACKS}.item_id
    ORDER BY {DB_TABLE_ALBUM_TRACKS}.album_id LIMIT 1)"""

# the work tracks with the albums they are on, one row per album
WORK_TRACKS_QUERY: Final[str] = f"""
    SELECT t.item_id AS track_id, t.work_id, t.movement_number, t.movement_name,
        t.search_name,
        (SELECT MIN(e.external_id) FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} e
            WHERE e.media_type = '{MediaType.TRACK.value}' AND e.item_id = t.item_id
            AND e.external_id_type = '{ExternalID.MB_RECORDING.value}') AS mb_recording,
        at.album_id, at.disc_number, at.track_number, a.year, a.album_type
    FROM {DB_TABLE_TRACKS} t
    LEFT JOIN {DB_TABLE_ALBUM_TRACKS} at ON at.track_id = t.item_id
    LEFT JOIN {DB_TABLE_ALBUMS} a ON a.item_id = at.album_id
    WHERE """


class ClassicalController:
    """Serves the classical browse listings of the library."""

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the controller and register its api commands."""
        self.mass = mass
        for command, handler in (
            ("composers", self.composers),
            ("performers", self.performers),
            ("works", self.works),
            ("work", self.work),
            ("recordings", self.recordings),
            ("other_tracks", self.other_tracks),
            ("link_progress", self.link_progress),
        ):
            self.mass.register_api_command(
                f"music/classical/{command}", handler, required_scope=Scope.LIBRARY_READ
            )

    async def composers(
        self,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        sort_field: SortField | None = None,
        sort_direction: SortDirection | None = None,
        artist_id: str | None = None,
    ) -> list[ClassicalComposer]:
        """
        Get the composers of the classical library, with their work and recording counts.

        :param search: Filter on the composer's name.
        :param limit: Maximum number of composers to return.
        :param offset: Number of composers to skip.
        :param sort_field: Sort on the sort name (default), the name or the work count.
        :param sort_direction: Sort direction, the field's default when omitted.
        :param artist_id: Only include this artist (library artist id), returning no rows
            when they are not a classical composer.
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_COMPOSERS, sort_field, sort_direction
        )
        params: dict[str, Any] = {}
        # only composers of a work are listed, so each row counts at least one work
        conditions = [
            f"{DB_TABLE_ARTISTS}.item_id IN "
            f"(SELECT artist_id FROM ({self._composer_works_query(params)}))"
        ]
        if artist_id:
            params["artist_id"] = int(artist_id)
            conditions.append(f"{DB_TABLE_ARTISTS}.item_id = :artist_id")
        if field == SortField.WORK_COUNT:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, SortField.SORT_NAME, SortDirection.ASC)
            ids = await self._ids(DB_TABLE_ARTISTS, conditions, params, search, sort_key)
            works = await self._composer_works(None)
            ids = _page(
                sorted(ids, key=lambda x: len(works[x]), reverse=direction == SortDirection.DESC),
                limit,
                offset,
            )
        else:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, field, direction)
            ids = await self._ids(
                DB_TABLE_ARTISTS, conditions, params, search, sort_key, limit, offset
            )
            works = await self._composer_works(ids)
        performances = await self._listed_work_performances(
            {x for item_id in ids for x in works[item_id]}
        )
        fanarts = await self._artist_fanarts(ids)
        return [
            ClassicalComposer(
                artist=cast("ArtistSummary", artist),
                fanart=fanarts.get(int(artist.item_id)),
                work_count=len(works[int(artist.item_id)]),
                recording_count=sum(len(performances[x]) for x in works[int(artist.item_id)]),
            )
            for artist in await self._items(self.mass.music.artists, ids)
        ]

    async def performers(
        self,
        role: ArtistRole | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        sort_field: SortField | None = None,
        sort_direction: SortDirection | None = None,
        artist_id: str | None = None,
    ) -> list[ClassicalPerformer]:
        """
        Get the performers of the classical library, with their roles and counts.

        :param role: Only include performers holding this performing role.
        :param search: Filter on the performer's name.
        :param limit: Maximum number of performers to return.
        :param offset: Number of performers to skip.
        :param sort_field: Sort on the name (default) or the recording count.
        :param sort_direction: Sort direction, the field's default when omitted.
        :param artist_id: Only include this artist (library artist id), returning no rows
            when they are not a classical performer.
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_PERFORMERS, sort_field, sort_direction
        )
        params: dict[str, Any] = {}
        role_filter = ""
        if role:
            role_filter = " AND ta.role = :role"
            params["role"] = role.value
        conditions = [
            f"{DB_TABLE_ARTISTS}.item_id IN (SELECT ta.artist_id {PERFORMER_CREDITS} "
            f"AND t.item_id IN ({self._classical_tracks(params)}){role_filter})"
        ]
        if artist_id:
            params["artist_id"] = int(artist_id)
            conditions.append(f"{DB_TABLE_ARTISTS}.item_id = :artist_id")
        if field == SortField.RECORDING_COUNT:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, SortField.NAME, SortDirection.ASC)
            ids = await self._ids(DB_TABLE_ARTISTS, conditions, params, search, sort_key)
            roles, works = await self._performer_credits(None)
            performances = await self._listed_work_performances(None)
            counts = {x: _performer_recording_count(x, works[x], performances) for x in ids}
            ids = _page(
                sorted(ids, key=lambda x: counts[x], reverse=direction == SortDirection.DESC),
                limit,
                offset,
            )
        else:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, field, direction)
            ids = await self._ids(
                DB_TABLE_ARTISTS, conditions, params, search, sort_key, limit, offset
            )
            roles, works = await self._performer_credits(ids)
            performances = await self._listed_work_performances(
                {x for item_id in ids for x in works[item_id]}
            )
        fanarts = await self._artist_fanarts(ids)
        return [
            _performer_row(
                cast("ArtistSummary", artist),
                fanarts.get(int(artist.item_id)),
                roles,
                works,
                performances,
            )
            for artist in await self._items(self.mass.music.artists, ids)
        ]

    async def works(
        self,
        composer_id: str | None = None,
        performer_id: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        sort_field: SortField | None = None,
        sort_direction: SortDirection | None = None,
    ) -> list[ClassicalWorkEntry]:
        """
        Get the works of the classical library, with their recording counts.

        The parts of a work, such as the arias of an opera, are listed through that work:
        its recordings, filters and search cover those of its parts, and each row carries
        the names of all its parts in the library.

        :param composer_id: Only include works by this composer (library artist id).
        :param performer_id: Only include works this performer (library artist id) performs
            on, counting only their recordings.
        :param year_from: Only include works composed in or after this year.
        :param year_to: Only include works composed in or before this year.
        :param search: Filter on the work's title.
        :param limit: Maximum number of works to return.
        :param offset: Number of works to skip.
        :param sort_field: Sort on the composer (default, then catalogue number), the title,
            the year composed or the recording count. A composer's works sort on the composer by
            catalogue number alone, those shared with another composer among them.
        :param sort_direction: Sort direction, the field's default when omitted.
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_WORKS, sort_field, sort_direction
        )
        performer = int(performer_id) if performer_id else None
        filters: dict[str, Any] = {
            "composer_id": composer_id,
            "performer": performer,
            "year_from": year_from,
            "year_to": year_to,
            "search": search,
        }
        if field == SortField.RECORDING_COUNT:
            sort_key = work_sort_sql(SortField.COMPOSER, SortDirection.ASC, bool(composer_id))
            ids = await self._work_ids(sort_key, **filters)
            performances = await self._listed_work_performances(None)
            ids = _page(
                sorted(
                    ids,
                    key=lambda x: _recording_count(performances[x], performer),
                    reverse=direction == SortDirection.DESC,
                ),
                limit,
                offset,
            )
        else:
            sort_key = work_sort_sql(field, direction, bool(composer_id))
            ids = await self._work_ids(sort_key, limit=limit, offset=offset, **filters)
            performances = await self._listed_work_performances(ids)
        part_names = await self._part_names(ids)
        return [
            ClassicalWorkEntry(
                work=_with_title_work_type(cast("WorkSummary", work)),
                recording_count=_recording_count(performances[int(work.item_id)], performer),
                part_names=part_names[int(work.item_id)],
            )
            for work in await self._items(self.mass.music.works, ids)
        ]

    async def work(self, item_id: str) -> Work:
        """
        Get a library work as the classical listings show it.

        A work without a type of its own shows the type its title names, as in the works listing.

        :param item_id: The library id of the work.
        """
        return _with_title_work_type(await self.mass.music.works.get_library_item(item_id))

    async def recordings(
        self, work_id: str | None = None, performer_id: str | None = None
    ) -> list[Recording]:
        """
        Get the recordings of a work, or every recording a performer performs on.

        Recordings of a work come by year (undated last), then conductor, then ensemble, followed by
        those of each of its parts, each recording carrying its own work and the work that one is a
        part of, if any. A performer's recordings across their works come per work, in the order of
        the works listing, and within a work in that same order.

        :param work_id: Library id of the work. Required unless performer_id is given.
        :param performer_id: Only include the recordings this performer (library artist id)
            performs on.
        """
        if work_id:
            listed_ids = [int(work_id)]
        elif performer_id:
            sort_key = work_sort_sql(*resolve_listing_sort(ListingType.CLASSICAL_WORKS))
            listed_ids = await self._work_ids(sort_key, performer=int(performer_id))
        else:
            msg = "Either work_id or performer_id is required"
            raise InvalidDataError(msg)
        parts = await self._work_parts(listed_ids)
        work_ids = [x for listed_id in listed_ids for x in parts.get(listed_id, [listed_id])]
        work_items = await self._items(self.mass.music.works, work_ids)
        works = {int(work.item_id): ItemMapping.from_item(work) for work in work_items}
        parent_works = {
            int(work.item_id): ItemMapping.from_item(work.parent_work)
            for work in work_items
            if work.parent_work
        }
        performances = await self._performances(
            work_ids, {x: y for ids in parts.values() for x, y in ids.items()}
        )
        if performer_id:
            for key, value in performances.items():
                performances[key] = [x for x in value if int(performer_id) in x.performer_ids]
        tracks = {
            int(track.item_id): track
            for track in await self._items(
                self.mass.music.tracks,
                [
                    track_id
                    for value in performances.values()
                    for x in value
                    for track_id, _ in x.tracks
                ],
                summary=False,
            )
        }
        albums = {
            int(album.item_id): ItemMapping.from_item(album)
            for album in await self._items(
                self.mass.music.albums,
                list(
                    dict.fromkeys(
                        album_id
                        for value in performances.values()
                        for x in value
                        for album_id in x.album_ids
                    )
                ),
            )
        }
        return [
            recording
            for work in work_ids
            if work in works
            for recording in sorted(
                (
                    _build_recording(works[work], parent_works.get(work), x, tracks, albums)
                    for x in performances[work]
                ),
                key=_recording_sort_key,
            )
        ]

    async def other_tracks(
        self,
        artist_id: str,
        as_composer: bool = False,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        sort_field: SortField | None = None,
        sort_direction: SortDirection | None = None,
    ) -> list[Track]:
        """
        Get the classical tracks of an artist that are not linked to a work.

        :param artist_id: Library id of the artist.
        :param as_composer: True for the tracks the artist composed, False for the tracks
            holding any other credit of the artist.
        :param search: Filter on the track's title.
        :param limit: Maximum number of tracks to return.
        :param offset: Number of tracks to skip.
        :param sort_field: Sort on the title (default), the album year or the date added.
        :param sort_direction: Sort direction, the field's default when omitted.
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_OTHER_TRACKS, sort_field, sort_direction
        )
        params: dict[str, Any] = {"artist_id": int(artist_id)}
        role_match = "=" if as_composer else "!="
        credited = (
            f"SELECT track_id FROM {DB_TABLE_TRACK_CREDITS} "
            f"WHERE artist_id = :artist_id AND role {role_match} '{ArtistRole.COMPOSER.value}'"
        )
        if not as_composer:
            credited += (
                f" UNION SELECT track_id FROM {DB_TABLE_TRACK_ARTISTS} WHERE artist_id = :artist_id"
            )
        conditions = [
            f"{DB_TABLE_TRACKS}.is_classical = 1",
            self.mass.music.tracks.visible_library_clause(params),
            f"{DB_TABLE_TRACKS}.work_id IS NULL",
            f"{DB_TABLE_TRACKS}.item_id IN ({credited})",
        ]
        if field == SortField.YEAR:
            # tracks without an album year come last
            order = (direction or SortDirection.ASC).value.upper()
            name_sort = _sort_sql(DB_TABLE_TRACKS, SortField.NAME, SortDirection.ASC)
            sort_key = f"{TRACK_ALBUM_YEAR} {order} NULLS LAST, {name_sort}"
        else:
            sort_key = _sort_sql(DB_TABLE_TRACKS, field, direction)
        ids = await self._ids(DB_TABLE_TRACKS, conditions, params, search, sort_key, limit, offset)
        return await self._items(self.mass.music.tracks, ids, summary=False)

    async def link_progress(self) -> dict[str, int]:
        """
        Get how many classical items MusicBrainz linking still has to check.

        While any count is above zero, tracks under Other tracks may still move into their works.
        ``albums`` counts the albums still to identify, ``works`` the works still to look up.
        All zero means linking is done, or is not running.
        """
        return await self.mass.metadata.get_classical_link_progress()

    def _classical_tracks(self, params: dict[str, Any]) -> str:
        """Return a subquery selecting the classical library tracks the calling user sees."""
        visible = self.mass.music.tracks.visible_library_clause(params)
        return (
            f"SELECT {DB_TABLE_TRACKS}.item_id FROM {DB_TABLE_TRACKS} "
            f"WHERE {DB_TABLE_TRACKS}.is_classical = 1 AND {visible}"
        )

    def _classical_work_ids(self, params: dict[str, Any]) -> str:
        """Return a subquery selecting the works of the classical tracks the calling user sees."""
        return (
            f"SELECT work_id FROM {DB_TABLE_TRACKS} WHERE work_id IS NOT NULL "
            f"AND item_id IN ({self._classical_tracks(params)})"
        )

    async def _work_ids(
        self,
        sort_key: str,
        composer_id: str | None = None,
        performer: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        search: str | None = None,
        limit: int = 0,
        offset: int = 0,
    ) -> list[int]:
        """Return the ids of the listed works matching the filters, in order."""
        params: dict[str, Any] = {}
        if performer:
            params["performer_id"] = performer
            work_ids = (
                f"SELECT t.work_id {PERFORMER_CREDITS} "
                f"AND t.item_id IN ({self._classical_tracks(params)}) "
                "AND ta.artist_id = :performer_id"
            )
        else:
            work_ids = self._classical_work_ids(params)
        conditions = [
            f"{DB_TABLE_WORKS}.item_id IN (SELECT {LISTED_WORK} FROM {DB_TABLE_WORKS} w "
            f"WHERE w.item_id IN ({work_ids}))"
        ]
        if composer_id:
            params["composer_id"] = int(composer_id)
            conditions.append(
                f"{DB_TABLE_WORKS}.item_id IN (SELECT work_id FROM ({LISTED_WORK_COMPOSERS}) "
                "WHERE artist_id = :composer_id)"
            )
        if search and (search_term := create_safe_string(search, True, True)):
            conditions.append(work_search_sql(search_term, params))
        # works without a composition year drop out once a bound is set
        if year_from is not None:
            params["year_from"] = year_from
            conditions.append(f"{DB_TABLE_WORKS}.composition_year >= :year_from")
        if year_to is not None:
            params["year_to"] = year_to
            conditions.append(f"{DB_TABLE_WORKS}.composition_year <= :year_to")
        return await self._ids(DB_TABLE_WORKS, conditions, params, None, sort_key, limit, offset)

    async def _ids(
        self,
        table: str,
        conditions: list[str],
        params: dict[str, Any],
        search: str | None,
        sort_key: str,
        limit: int = 0,
        offset: int = 0,
    ) -> list[int]:
        """Return the ids of the matching library items in the given order, all without a limit."""
        conditions = list(conditions)
        if search and (search_term := create_safe_string(search, True, True)):
            conditions.append(search_name_match_clause(table, search_term, "search", params))
        query = (
            f"SELECT {table}.item_id FROM {table} WHERE {' AND '.join(conditions)} "
            f"ORDER BY {sort_key}, {table}.item_id"
        )
        rows = await self.mass.music.database.get_rows_from_query(
            query, params, limit=limit, offset=offset
        )
        return [int(row["item_id"]) for row in rows]

    async def _items[ItemT: MediaItemType](
        self, controller: MediaControllerBase[ItemT], ids: Sequence[int], summary: bool = True
    ) -> list[ItemT]:
        """Return the library items with the given ids, in the given order."""
        if not ids:
            return []
        items = await controller.get_library_items_by_query(
            limit=0,
            extra_query_parts=[f"{controller.db_table}.item_id IN :item_ids"],
            extra_query_params={"item_ids": list(ids)},
            summary=summary,
        )
        by_id = {int(item.item_id): item for item in items}
        return [by_id[item_id] for item_id in ids if item_id in by_id]

    async def _artist_fanarts(self, artist_ids: Sequence[int]) -> dict[int, MediaItemImage]:
        """Return the fanart to show of the given library artists, by artist id."""
        if not artist_ids:
            return {}
        user = get_current_user()
        hidden_sources = hidden_music_sources(self.mass, user) if user else set()
        fanarts: dict[int, MediaItemImage] = {}
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id, json_extract(metadata, '$.images') AS images "
            f"FROM {DB_TABLE_ARTISTS} WHERE item_id IN :artist_ids",
            {"artist_ids": list(artist_ids)},
            limit=0,
        ):
            images = json_loads(row["images"]) if row["images"] else None
            if image := preferred_thumb(images, hidden_sources, ImageType.FANART):
                fanarts[int(row["item_id"])] = MediaItemImage(
                    type=ImageType.FANART,
                    path=image["path"],
                    provider=image["provider"],
                    remotely_accessible=image.get("remotely_accessible", False),
                )
        return fanarts

    async def _composer_works(self, artist_ids: Sequence[int] | None) -> dict[int, set[int]]:
        """Return the listed classical works of the given composers, or of every composer."""
        params: dict[str, Any] = {}
        query = self._composer_works_query(params)
        if artist_ids is not None:
            query += " AND artist_id IN :artist_ids"
            params["artist_ids"] = list(artist_ids)
        works: dict[int, set[int]] = defaultdict(set)
        if artist_ids is None or artist_ids:
            for row in await self.mass.music.database.get_rows_from_query(query, params, limit=0):
                works[int(row["artist_id"])].add(int(row["work_id"]))
        return works

    def _composer_works_query(self, params: dict[str, Any]) -> str:
        """Return a query selecting the composers with their listed works the calling user sees."""
        return (
            f"SELECT artist_id, work_id FROM ({LISTED_WORK_COMPOSERS}) "
            f"WHERE work_id IN (SELECT {LISTED_WORK} FROM {DB_TABLE_WORKS} w "
            f"WHERE w.item_id IN ({self._classical_work_ids(params)}))"
        )

    async def _performer_credits(
        self, artist_ids: Sequence[int] | None
    ) -> tuple[dict[int, Counter[ArtistRole]], dict[int, set[int]]]:
        """
        Return the credit count per performing role and the listed works of performers.

        :param artist_ids: The performers to return, or None for every performer.
        """
        roles: dict[int, Counter[ArtistRole]] = defaultdict(Counter)
        works: dict[int, set[int]] = defaultdict(set)
        if artist_ids is not None and not artist_ids:
            return roles, works
        params: dict[str, Any] = {}
        scope = f" AND t.item_id IN ({self._classical_tracks(params)})"
        if artist_ids is not None:
            scope += " AND ta.artist_id IN :artist_ids"
            params["artist_ids"] = list(artist_ids)
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT ta.artist_id, ta.role, COUNT(DISTINCT ta.track_id) AS credit_count "
            f"{PERFORMER_CREDITS}{scope} GROUP BY ta.artist_id, ta.role",
            params,
            limit=0,
        ):
            roles[int(row["artist_id"])][ArtistRole(row["role"])] = int(row["credit_count"])
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT c.artist_id, {LISTED_WORK} AS work_id "
            f"FROM (SELECT ta.artist_id, t.work_id {PERFORMER_CREDITS}{scope}) c "
            f"JOIN {DB_TABLE_WORKS} w ON w.item_id = c.work_id",
            params,
            limit=0,
        ):
            works[int(row["artist_id"])].add(int(row["work_id"]))
        return roles, works

    async def _work_parts(self, listed_ids: Iterable[int] | None) -> dict[int, dict[int, int]]:
        """
        Return the given listed works, or all of them, each with itself and then its parts.

        Each work maps to the work whose recordings its own join: itself, or for a part of a part,
        such as a scene of an act, the part of the listed work it sits under.
        """
        params: dict[str, Any] = {}
        query = f"SELECT w.item_id, {LISTED_WORK} AS listed_id FROM {DB_TABLE_WORKS} w"
        if listed_ids is not None:
            if not (unique_ids := list(set(listed_ids))):
                return {}
            params["listed_ids"] = unique_ids
            query += " WHERE w.item_id IN :listed_ids OR w.parent_work_id IN :listed_ids"
        query += " ORDER BY w.parent_work_id IS NOT NULL, w.name_sort, w.item_id"
        parts: dict[int, dict[int, int]] = defaultdict(dict)
        for row in await self.mass.music.database.get_rows_from_query(query, params, limit=0):
            item_id = int(row["item_id"])
            parts[int(row["listed_id"])][item_id] = item_id
        return parts

    async def _part_names(self, work_ids: Sequence[int]) -> dict[int, list[str]]:
        """Return the names of the parts of the given works in the library, by work id."""
        names: dict[int, list[str]] = defaultdict(list)
        if not work_ids:
            return names
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT parent_work_id, name FROM {DB_TABLE_WORKS} "
            "WHERE parent_work_id IN :work_ids ORDER BY name_sort, item_id",
            {"work_ids": list(work_ids)},
            limit=0,
        ):
            names[int(row["parent_work_id"])].append(row["name"])
        return names

    async def _listed_work_performances(
        self, listed_ids: Iterable[int] | None
    ) -> dict[int, list[_Performance]]:
        """Return the performances of the given listed works or of all, including their parts'."""
        parts = await self._work_parts(listed_ids)
        performances = await self._performances(
            None if listed_ids is None else [x for ids in parts.values() for x in ids],
            {x: y for ids in parts.values() for x, y in ids.items()},
        )
        listed: dict[int, list[_Performance]] = defaultdict(list)
        for listed_id, work_ids in parts.items():
            for work_id in work_ids:
                listed[listed_id].extend(performances[work_id])
        return listed

    async def _performances(
        self, work_ids: Iterable[int] | None, joined: Mapping[int, int]
    ) -> dict[int, list[_Performance]]:
        """
        Return the performances of the given classical works, or of every classical work.

        :param work_ids: Library ids of the works, None for every work.
        :param joined: The work each work's performances join when they are one of its own,
            such as a scene joining the performance of its act, by work id.
        """
        performances: dict[int, list[_Performance]] = defaultdict(list)
        params: dict[str, Any] = {}
        scope = "t.work_id IS NOT NULL"
        if work_ids is not None:
            if not (unique_work_ids := list(set(work_ids))):
                return performances
            params["work_ids"] = unique_work_ids
            scope = "t.work_id IN :work_ids"
        scope += f" AND t.item_id IN ({self._classical_tracks(params)})"
        tracks: dict[int, _WorkTrack] = {}
        work_tracks: dict[int, list[_WorkTrack]] = defaultdict(list)
        for row in await self.mass.music.database.get_rows_from_query(
            WORK_TRACKS_QUERY + scope, params, limit=0
        ):
            if not (track := tracks.get(row["track_id"])):
                track = tracks[row["track_id"]] = _WorkTrack(
                    track_id=int(row["track_id"]),
                    movement=row["movement_number"]
                    or create_safe_string(row["movement_name"] or "", True, True)
                    or row["search_name"],
                    mb_recording=(row["mb_recording"] or "").casefold() or None,
                )
                work_tracks[int(row["work_id"])].append(track)
            if row["album_id"] is not None:
                track.appearances.append(
                    _Appearance(
                        album_id=int(row["album_id"]),
                        year=row["year"],
                        disc_number=row["disc_number"] or 0,
                        track_number=row["track_number"] or 0,
                        compilation=row["album_type"] == AlbumType.COMPILATION.value,
                    )
                )
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT ta.track_id, ta.artist_id, ta.role {PERFORMER_CREDITS} AND {scope}",
            params,
            limit=0,
        ):
            if track := tracks.get(row["track_id"]):
                track.performers.add(int(row["artist_id"]))
                if row["role"] in LEADING_ROLES:
                    track.leaders.add(int(row["artist_id"]))
                if row["role"] in DIRECTING_ROLES:
                    track.directors.add(int(row["artist_id"]))
        groups = {work_id: _group_performances(x) for work_id, x in work_tracks.items()}
        # the tracks of a scene that are one performance with tracks of its act join those
        for work_id, joined_id in joined.items():
            if joined_id != work_id and work_id in groups and joined_id in groups:
                for key in groups[work_id].keys() & groups[joined_id].keys():
                    groups[joined_id][key].extend(groups[work_id].pop(key))
        for work_id, work_groups in groups.items():
            performances[work_id] = [
                _build_performance(work_id, key, group) for key, group in work_groups.items()
            ]
        return performances


@dataclass(kw_only=True)
class _Appearance:
    """The place of a track on an album."""

    album_id: int
    year: int | None
    disc_number: int
    track_number: int
    compilation: bool = False


@dataclass(kw_only=True)
class _WorkTrack:
    """A classical track of a work, with what tells its performance apart."""

    track_id: int
    movement: int | str
    mb_recording: str | None
    performers: set[int] = field(default_factory=set)
    leaders: set[int] = field(default_factory=set)
    directors: set[int] = field(default_factory=set)
    appearances: list[_Appearance] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        """Return the earliest year of the albums the track is on."""
        return min((x.year for x in self.appearances if x.year), default=None)

    @property
    def performance_key(self) -> str:
        """Return what identifies the performance of this track."""
        # a choir credited on only some movements does not make them another performance
        if self.directors:
            return f"{sorted(self.directors)}:{self.year}"
        if self.leaders:
            return f"{sorted(self.leaders)}:{self.year}"
        # without a conductor or ensemble, the album identifies the performance, unless it is a
        # compilation of performances
        if albums := [x.album_id for x in self.appearances if not x.compilation]:
            return f"album:{min(albums)}"
        return f"{sorted(self.performers)}:{self.year}"


@dataclass(kw_only=True)
class _Performance:
    """One performance of a work, the grouping a Recording is built from."""

    key: str
    year: int | None
    performer_ids: frozenset[int]
    album_ids: list[int]
    # one track per movement, in movement order, with the album it is taken from
    tracks: list[tuple[int, _Appearance | None]]


def _group_performances(tracks: Iterable[_WorkTrack]) -> dict[str, list[list[_WorkTrack]]]:
    """
    Group the classical tracks of a work into its performances, by performance key.

    Each performance holds the tracks of each of its movements.
    """
    # the same movement found on several albums is one recording of that movement
    movements: dict[tuple[Any, ...], list[_WorkTrack]] = defaultdict(list)
    for track in tracks:
        if track.mb_recording:
            movements[(track.mb_recording,)].append(track)
        else:
            movements[(track.performance_key, track.movement)].append(track)
    groups: dict[str, list[list[_WorkTrack]]] = defaultdict(list)
    for movement in movements.values():
        # a reissued movement belongs to the performance of its earliest release
        first = min(movement, key=lambda x: (x.year is None, x.year or 0, x.track_id))
        groups[first.performance_key].append(movement)
    return groups


def _build_performance(
    work_id: int, performance_key: str, movements: list[list[_WorkTrack]]
) -> _Performance:
    """Build a performance from the tracks of each of its movements."""
    album_movements: Counter[int] = Counter()
    album_years: dict[int, int | None] = {}
    for movement in movements:
        album_movements.update({x.album_id for track in movement for x in track.appearances})
        album_years.update({x.album_id: x.year for track in movement for x in track.appearances})
    # the album holding most movements comes first, the earliest one on a tie
    album_ids = sorted(
        album_movements,
        key=lambda x: (-album_movements[x], album_years[x] is None, album_years[x] or 0, x),
    )
    album_rank = {album_id: rank for rank, album_id in enumerate(album_ids)}
    chosen: list[tuple[_WorkTrack, _Appearance | None]] = []
    for movement in movements:
        if placed := [(track, x) for track in movement for x in track.appearances]:
            chosen.append(min(placed, key=lambda x: (album_rank[x[1].album_id], x[0].track_id)))
        else:
            chosen.append((min(movement, key=lambda x: x.track_id), None))
    # in album order, which the recording puts in the order of the movements it shows
    chosen.sort(
        key=lambda x: (
            x[1].disc_number if x[1] else 0,
            x[1].track_number if x[1] else 0,
            x[0].track_id,
        )
    )
    digest = hashlib.sha1(performance_key.encode(), usedforsecurity=False)
    return _Performance(
        key=f"{work_id}-{digest.hexdigest()[:16]}",
        year=min((t.year for movement in movements for t in movement if t.year), default=None),
        performer_ids=frozenset(
            x for movement in movements for t in movement for x in t.performers
        ),
        album_ids=album_ids,
        tracks=[(track.track_id, appearance) for track, appearance in chosen],
    )


def _build_recording(
    work: ItemMapping,
    parent_work: ItemMapping | None,
    performance: _Performance,
    tracks: dict[int, Track],
    albums: dict[int, ItemMapping],
) -> Recording:
    """
    Build a Recording from a performance and its library tracks and albums.

    :param work: The work of the performance.
    :param parent_work: The work that work is a part of, if any.
    :param performance: The performance to build the recording of.
    :param tracks: The library tracks of the performance, by track id.
    :param albums: The albums of the performance, by album id.
    """
    movement_tracks: list[Track] = []
    for track_id, appearance in performance.tracks:
        if not (track := tracks.get(track_id)):
            continue
        # a track on several albums is shown on the album the recording is taken from
        if appearance and (not track.album or int(track.album.item_id) != appearance.album_id):
            track.album = albums.get(appearance.album_id, track.album)
            track.disc_number = appearance.disc_number
            track.track_number = appearance.track_number
        movement_tracks.append(track)
    performing_credits: dict[tuple[str, ArtistRole, str | None], Credit] = {}
    for track in movement_tracks:
        for credit in _track_performing_credits(track):
            performing_credits.setdefault(
                (credit.artist.item_id, credit.role, credit.instrument), credit
            )
    return Recording(
        key=performance.key,
        work=work,
        parent_work=parent_work,
        tracks=movement_tracks,
        credits=sorted(performing_credits.values(), key=lambda x: PERFORMING_ROLES.index(x.role)),
        year=performance.year,
        albums=[albums[x] for x in performance.album_ids if x in albums],
        duration=sum(x.duration or 0 for x in movement_tracks),
    )


def _track_performing_credits(track: Track) -> list[Credit]:
    """Return the performing credits of a track, as counted by PERFORMER_CREDITS."""
    if performing := [x for x in track.credits if x.role in PERFORMING_ROLES]:
        return performing
    composer_ids = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
    return [
        Credit(artist=x, role=ArtistRole.PERFORMER, position=position)
        for position, x in enumerate(track.artists)
        if x.item_id not in composer_ids
    ]


def _recording_sort_key(recording: Recording) -> tuple[Any, ...]:
    """Return the key that puts recordings by year, then conductor, then ensemble."""

    def first_name(roles: tuple[ArtistRole, ...]) -> str:
        return next((x.artist.name.casefold() for x in recording.credits if x.role in roles), "")

    return (
        recording.year is None,
        recording.year or 0,
        first_name((ArtistRole.CONDUCTOR,)),
        first_name(ENSEMBLE_ROLES),
        recording.key,
    )


def _performer_row(
    artist: ArtistSummary,
    fanart: MediaItemImage | None,
    roles: dict[int, Counter[ArtistRole]],
    works: dict[int, set[int]],
    performances: dict[int, list[_Performance]],
) -> ClassicalPerformer:
    """Build the performers list row of an artist."""
    artist_id = int(artist.item_id)
    role_counts = roles[artist_id]
    return ClassicalPerformer(
        artist=artist,
        fanart=fanart,
        # the general performer role only leads when the artist holds no more specific one
        main_role=max(
            role_counts,
            key=lambda x: (
                x != ArtistRole.PERFORMER,
                role_counts[x],
                -PERFORMING_ROLES.index(x),
            ),
        ),
        roles=[x for x in PERFORMING_ROLES if x in role_counts],
        work_count=len(works[artist_id]),
        recording_count=_performer_recording_count(artist_id, works[artist_id], performances),
    )


def _recording_count(performances: list[_Performance], performer_id: int | None) -> int:
    """Return the number of performances, of those by the given performer when set."""
    if performer_id is None:
        return len(performances)
    return sum(1 for x in performances if performer_id in x.performer_ids)


def _performer_recording_count(
    performer_id: int, work_ids: Iterable[int], performances: dict[int, list[_Performance]]
) -> int:
    """Return the number of recordings a performer performs on across the given works."""
    return sum(_recording_count(performances[x], performer_id) for x in work_ids)


def _with_title_work_type[WorkT: Work](work: WorkT) -> WorkT:
    """Return a work with the type its title names when it has no type of its own."""
    work.work_type = work.work_type or _title_work_type(work.name)
    return work


def _title_work_type(title: str) -> WorkType | None:
    """Return the work type a work title names as a whole word, None when it names none."""
    text = _QUOTED_NICKNAME.sub(" ", title.lower())
    return next(
        (work_type for work_type, pattern in _TITLE_WORK_TYPES if pattern.search(text)), None
    ) or (WorkType((m[1] or m[2]).lower()) if (m := _TYPE_LABEL.search(title)) else None)


def _sort_sql(table: str, field: SortField, direction: SortDirection | None) -> str:
    """Return the ORDER BY clause of a base sort field on the columns of a table."""
    sql = BASE_SORT_FIELD_SQL[field].format(prefix=f"{table}.")
    return f"{sql} {(direction or SortDirection.ASC).value.upper()}"


def _page[T](items: list[T], limit: int, offset: int) -> list[T]:
    """Return one page of the given items, all from the offset without a limit."""
    return items[offset : offset + limit] if limit else items[offset:]
