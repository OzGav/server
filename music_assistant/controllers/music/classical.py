"""
Classical browse API of the Music controller.

Serves the composer, performer, work, recording and other track listings of the Classical
view. Every listing only considers classical library tracks.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final, cast

from music_assistant_models.auth import Scope
from music_assistant_models.enums import (
    AlbumType,
    ArtistRole,
    ExternalID,
    ImageType,
    ListingType,
    MediaType,
    Period,
    RecordingHeadingKind,
    SortDirection,
    SortField,
    WorkType,
)
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import (
    ArtistSummary,
    ClassicalComposer,
    ClassicalComposerFilters,
    ClassicalPerformer,
    ClassicalPerformerFilters,
    ClassicalWorkEntry,
    ClassicalWorkFilters,
    Credit,
    ItemMapping,
    MediaItemImage,
    MediaItemType,
    Recording,
    RecordingHeading,
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
from music_assistant.controllers.music.media.works import (
    WORK_ROOTS,
    work_sort_sql,
)
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.credit_roles import PERFORMING_ROLES, SQL_PERFORMING_ROLES
from music_assistant.helpers.json import json_loads
from music_assistant.helpers.provider_access import hidden_music_sources
from music_assistant.helpers.work_titles import title_work_type

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from music_assistant import MusicAssistant
    from music_assistant.controllers.music.media.base import MediaControllerBase

ENSEMBLE_ROLES: Final[tuple[ArtistRole, ...]] = (
    ArtistRole.ORCHESTRA,
    ArtistRole.ENSEMBLE,
    ArtistRole.CHOIR,
)
# the roles that tell one performance of a work from another
LEADING_ROLES: Final[tuple[ArtistRole, ...]] = (ArtistRole.CONDUCTOR, *ENSEMBLE_ROLES)
# the leading roles that stay the same across a performance, where choirs come and go
DIRECTING_ROLES: Final[tuple[ArtistRole, ...]] = (ArtistRole.CONDUCTOR, ArtistRole.ORCHESTRA)
# the role order a performer's works rows list their co-performers in
CO_PERFORMER_RANK: Final[dict[ArtistRole, int]] = {
    ArtistRole.ORCHESTRA: 0,
    ArtistRole.ENSEMBLE: 0,
    ArtistRole.CONDUCTOR: 1,
    ArtistRole.SOLOIST: 2,
    ArtistRole.CHOIR: 3,
    ArtistRole.PERFORMER: 4,
}


# the performing credits on classical tracks aliased as t, which callers narrow to the visible
# ones, where a track without any counts its main artists that did not compose it as performers
PERFORMER_CREDITS: Final[str] = f"""FROM (
        SELECT p.track_id, p.artist_id, p.role
        FROM {DB_TABLE_TRACK_CREDITS} p WHERE p.role IN ({SQL_PERFORMING_ROLES})
        UNION ALL
        SELECT m.track_id, m.artist_id, '{ArtistRole.PERFORMER.value}' AS role
        FROM {DB_TABLE_TRACK_ARTISTS} m
        WHERE NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_CREDITS} p
                WHERE p.track_id = m.track_id AND p.role IN ({SQL_PERFORMING_ROLES}))
            AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_CREDITS} c
                WHERE c.track_id = m.track_id AND c.artist_id = m.artist_id
                AND c.role = '{ArtistRole.COMPOSER.value}')
    ) ta
    JOIN {DB_TABLE_TRACKS} t ON t.item_id = ta.track_id
    WHERE t.is_classical = 1"""

# the composers of each listed work, its own or else those of its parts
LISTED_WORK_COMPOSERS: Final[str] = f"""
    SELECT r.root_id AS work_id, wa.artist_id
    FROM ({WORK_ROOTS}) r
    JOIN {DB_TABLE_WORK_ARTISTS} wa ON wa.work_id = r.item_id
        AND wa.role = '{ArtistRole.COMPOSER.value}'
    WHERE r.item_id = r.root_id OR NOT EXISTS (SELECT 1 FROM {DB_TABLE_WORK_ARTISTS} pa
        WHERE pa.work_id = r.root_id AND pa.role = '{ArtistRole.COMPOSER.value}')"""

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
            ("composer_filters", self.composer_filters),
            ("performers", self.performers),
            ("performer_filters", self.performer_filters),
            ("works", self.works),
            ("work_filters", self.work_filters),
            ("work", self.work),
            ("recordings", self.recordings),
            ("other_tracks", self.other_tracks),
            ("link_progress", self.link_progress),
            ("has_content", self.has_content),
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
        favorite: bool | None = None,
        period: list[Period] | None = None,
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
        :param favorite: Only include the calling user's likes (True) or dislikes (False).
        :param period: Only include composers of any of these periods.
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_COMPOSERS, sort_field, sort_direction
        )
        params: dict[str, Any] = {}
        conditions = [self._listed_composers(params)]
        if artist_id:
            params["artist_id"] = int(artist_id)
            conditions.append(f"{DB_TABLE_ARTISTS}.item_id = :artist_id")
        if favorite is not None:
            conditions.append(self.mass.music.artists._favorite_filter_clause(params, favorite))
        if period:
            params["periods"] = [x.value for x in period]
            conditions.append(f"{DB_TABLE_ARTISTS}.period IN :periods")
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

    async def composer_filters(self) -> ClassicalComposerFilters:
        """Get the values the composers list can be filtered on, across all its composers."""
        params: dict[str, Any] = {}
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT {DB_TABLE_ARTISTS}.period "
            f"FROM {DB_TABLE_ARTISTS} WHERE {self._listed_composers(params)}",
            params,
            limit=0,
        )
        periods = {Period(row["period"]) for row in rows if row["period"] is not None}
        return ClassicalComposerFilters(periods=[x for x in Period if x in periods])

    async def performers(
        self,
        role: list[ArtistRole] | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        sort_field: SortField | None = None,
        sort_direction: SortDirection | None = None,
        artist_id: str | None = None,
        favorite: bool | None = None,
    ) -> list[ClassicalPerformer]:
        """
        Get the performers of the classical library, with their roles and counts.

        :param role: Only include performers holding any of these performing roles.
        :param search: Filter on the performer's name.
        :param limit: Maximum number of performers to return.
        :param offset: Number of performers to skip.
        :param sort_field: Sort on the name (default) or the recording count.
        :param sort_direction: Sort direction, the field's default when omitted.
        :param artist_id: Only include this artist (library artist id), returning no rows
            when they are not a classical performer.
        :param favorite: Only include the calling user's likes (True) or dislikes (False).
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_PERFORMERS, sort_field, sort_direction
        )
        params: dict[str, Any] = {}
        conditions = [self._listed_performers(params, role or [])]
        if artist_id:
            params["artist_id"] = int(artist_id)
            conditions.append(f"{DB_TABLE_ARTISTS}.item_id = :artist_id")
        if favorite is not None:
            conditions.append(self.mass.music.artists._favorite_filter_clause(params, favorite))
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

    async def performer_filters(self) -> ClassicalPerformerFilters:
        """Get the values the performers list can be filtered on, across all its performers."""
        params: dict[str, Any] = {}
        ids = await self._ids(
            DB_TABLE_ARTISTS,
            [self._listed_performers(params)],
            params,
            None,
            f"{DB_TABLE_ARTISTS}.item_id",
        )
        roles, _ = await self._performer_credits(ids)
        present = {x for artist_id in ids for x in roles[artist_id]}
        return ClassicalPerformerFilters(roles=[x for x in ArtistRole if x in present])

    async def works(  # noqa: PLR0913
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
        work_type: list[WorkType] | None = None,
        work_type_unknown: bool = False,
    ) -> list[ClassicalWorkEntry]:
        """
        Get the works of the classical library, with their recording counts.

        The parts of a work, such as the arias of an opera, are listed through that work:
        its recordings, filters and search cover those of its parts, and each row carries
        the names of all its parts in the library.

        :param composer_id: Only include works by this composer (library artist id).
        :param performer_id: Only include works this performer (library artist id) performs
            on, counting only their recordings. Each row then also names the others performing
            on those recordings and tells whether the calling user likes all of their tracks.
        :param year_from: Only include works composed in or after this year.
        :param year_to: Only include works composed in or before this year. A range given
            the wrong way round is read the right way round.
        :param search: Filter on the work's composer, title, catalogue number and part names
            together, so "bach mass" finds the masses of Bach.
        :param limit: Maximum number of works to return.
        :param offset: Number of works to skip.
        :param sort_field: Sort on the composer (default, then catalogue number), the title,
            the year composed or the recording count. A composer's works sort on the composer by
            catalogue number alone, those shared with another composer among them.
        :param sort_direction: Sort direction, the field's default when omitted.
        :param work_type: Only include works of any of these types, as shown.
        :param work_type_unknown: Also include the works without a type; on its own, only
            those.
        """
        field, direction = resolve_listing_sort(
            ListingType.CLASSICAL_WORKS, sort_field, sort_direction
        )
        if year_from is not None and year_to is not None and year_from > year_to:
            year_from, year_to = year_to, year_from
        performer = int(performer_id) if performer_id else None
        filters: dict[str, Any] = {
            "composer_id": composer_id,
            "performer": performer,
            "year_from": year_from,
            "year_to": year_to,
        }
        search_term = create_safe_string(search, True, True) if search else ""
        shown_filters = bool(search_term or work_type or work_type_unknown)
        if field == SortField.RECORDING_COUNT:
            sort_key = work_sort_sql(SortField.COMPOSER, SortDirection.ASC, bool(composer_id))
        else:
            sort_key = work_sort_sql(field, direction, bool(composer_id))
        # the search and the filters on what the rows show match in Python, so they page
        # after reading every listed work
        if shown_filters or field == SortField.RECORDING_COUNT:
            ids = await self._work_ids(sort_key, **filters)
            if shown_filters:
                ids = await self._matching_works(ids, search_term, work_type, work_type_unknown)
        else:
            ids = await self._work_ids(sort_key, limit=limit, offset=offset, **filters)
        if field == SortField.RECORDING_COUNT:
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
            if shown_filters:
                ids = _page(ids, limit, offset)
            performances = await self._listed_work_performances(ids)
        part_names = await self._part_names(ids)
        entries = [
            ClassicalWorkEntry(
                work=_with_title_work_type(cast("WorkSummary", work)),
                recording_count=_recording_count(performances[int(work.item_id)], performer),
                part_names=part_names[int(work.item_id)],
            )
            for work in await self._items(self.mass.music.works, ids)
        ]
        if performer:
            await self._add_co_performers_and_liked(entries, performances, performer)
        return entries

    async def work_filters(self) -> ClassicalWorkFilters:
        """Get the values the works list can be filtered on, across all its works."""
        ids = await self._work_ids(f"{DB_TABLE_WORKS}.item_id")
        shown = (await self._shown_works(ids)).values()
        work_types = {x.work_type for x in shown}
        years = [x.year for x in shown if x.year is not None]
        return ClassicalWorkFilters(
            work_types=[x for x in WorkType if x in work_types],
            work_type_unknown=None in work_types,
            earliest_year=min(years, default=None),
            latest_year=max(years, default=None),
        )

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

        Requested for a work, each recording also carries the heading it shows under on the
        work's page.

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
        performances = await self._performances(work_ids)
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
        found = [
            recording
            for work in work_ids
            if work in works
            for recording in sorted(
                (
                    _build_recording(works[work], parent_works.get(work), x, tracks, albums)
                    for x in performances[work]
                ),
                key=lambda x: _recording_sort_key(x.year, x.credits, x.key),
            )
        ]
        if work_id and (page := works.get(int(work_id))):
            listed_parts = [
                works[x]
                for x in work_ids
                if x in parent_works and parent_works[x].item_id == work_id
            ]
            _add_part_headings(found, page, listed_parts)
        return found

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

    async def has_content(self) -> bool:
        """Return True when the library holds any classical track the calling user sees."""
        params: dict[str, Any] = {}
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT EXISTS ({self._classical_tracks(params)}) AS has_content", params, limit=0
        )
        return bool(rows[0]["has_content"])

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

    def _listed_composers(self, params: dict[str, Any]) -> str:
        """
        Return the condition matching the composers the composers list shows the calling user.

        :param params: Query params dict; the condition's bound params are added to it.
        """
        # only composers of a work are listed, so each row counts at least one work
        return (
            f"{DB_TABLE_ARTISTS}.item_id IN "
            f"(SELECT artist_id FROM ({self._composer_works_query(params)}))"
        )

    def _listed_performers(self, params: dict[str, Any], roles: Sequence[ArtistRole] = ()) -> str:
        """
        Return the condition matching the performers the performers list shows the calling user.

        :param params: Query params dict; the condition's bound params are added to it.
        :param roles: Only match the performers holding any of these performing roles.
        """
        role_filter = ""
        if roles:
            role_filter = " AND ta.role IN :roles"
            params["roles"] = [x.value for x in roles]
        return (
            f"{DB_TABLE_ARTISTS}.item_id IN (SELECT ta.artist_id {PERFORMER_CREDITS} "
            f"AND t.item_id IN ({self._classical_tracks(params)}){role_filter})"
        )

    async def _work_ids(
        self,
        sort_key: str,
        composer_id: str | None = None,
        performer: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
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
            f"{DB_TABLE_WORKS}.item_id IN (SELECT r.root_id FROM ({WORK_ROOTS}) r "
            f"WHERE r.item_id IN ({work_ids}))"
        ]
        if composer_id:
            params["composer_id"] = int(composer_id)
            conditions.append(
                f"{DB_TABLE_WORKS}.item_id IN (SELECT work_id FROM ({LISTED_WORK_COMPOSERS}) "
                "WHERE artist_id = :composer_id)"
            )
        # works without a composition year drop out once a bound is set
        if year_from is not None:
            params["year_from"] = year_from
            conditions.append(f"{DB_TABLE_WORKS}.composition_year >= :year_from")
        if year_to is not None:
            params["year_to"] = year_to
            conditions.append(f"{DB_TABLE_WORKS}.composition_year <= :year_to")
        return await self._ids(DB_TABLE_WORKS, conditions, params, None, sort_key, limit, offset)

    async def _matching_works(
        self,
        work_ids: list[int],
        search_term: str,
        work_types: Sequence[WorkType] | None,
        work_type_unknown: bool,
    ) -> list[int]:
        """
        Return the given works whose row matches the search and the filters, in the given order.

        :param work_ids: The listed works to match.
        :param search_term: The normalized search term, or empty to match any work.
        :param work_types: Only match works of any of these types, as shown.
        :param work_type_unknown: Also match the works without a type; on its own, only those.
        """
        shown = await self._shown_works(work_ids)
        part_names = await self._part_names(work_ids) if search_term else {}
        return [
            x
            for x in work_ids
            if (not search_term or search_term in shown[x].search_text(part_names[x]))
            and (
                not (work_types or work_type_unknown)
                or (
                    shown[x].work_type in (work_types or ())
                    if shown[x].work_type
                    else work_type_unknown
                )
            )
        ]

    async def _shown_works(self, work_ids: Sequence[int]) -> dict[int, _ShownWork]:
        """Return what the works list shows of the given works that it searches and filters on."""
        if not work_ids:
            return {}
        rows = await self.mass.music.database.get_rows_from_query(
            "SELECT w.item_id, w.name, w.work_type, w.catalog_numbers, w.composition_year, "
            "c.name AS composer "
            f"FROM {DB_TABLE_WORKS} w LEFT JOIN {DB_TABLE_ARTISTS} c ON c.item_id = "
            f"(SELECT wa.artist_id FROM {DB_TABLE_WORK_ARTISTS} wa WHERE wa.work_id = w.item_id "
            f"AND wa.role = '{ArtistRole.COMPOSER.value}' ORDER BY wa.position, wa.rowid LIMIT 1) "
            "WHERE w.item_id IN :work_ids",
            {"work_ids": list(work_ids)},
            limit=0,
        )
        # the first composer as the composers of a work summary order them, and the type the
        # title names for a work without its own, as the rows show them
        return {
            int(row["item_id"]): _ShownWork(
                names=[
                    row["composer"] or "",
                    row["name"],
                    next(iter(json_loads(row["catalog_numbers"])), ""),
                ],
                work_type=WorkType(row["work_type"])
                if row["work_type"] is not None
                else title_work_type(row["name"]),
                year=row["composition_year"],
            )
            for row in rows
        }

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
            f"WHERE work_id IN (SELECT r.root_id FROM ({WORK_ROOTS}) r "
            f"WHERE r.item_id IN ({self._classical_work_ids(params)}))"
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
            f"SELECT DISTINCT c.artist_id, r.root_id AS work_id "
            f"FROM (SELECT ta.artist_id, t.work_id {PERFORMER_CREDITS}{scope}) c "
            f"JOIN ({WORK_ROOTS}) r ON r.item_id = c.work_id",
            params,
            limit=0,
        ):
            works[int(row["artist_id"])].add(int(row["work_id"]))
        return roles, works

    async def _work_parts(self, listed_ids: Iterable[int] | None) -> dict[int, list[int]]:
        """Return the given listed works, or all of them, each with itself and then its parts."""
        params: dict[str, Any] = {}
        roots = "parent_work_id IS NULL"
        if listed_ids is not None:
            if not (unique_ids := list(set(listed_ids))):
                return {}
            params["listed_ids"] = unique_ids
            roots = "item_id IN :listed_ids"
        # the parts of parts, such as the arias of an act, count as parts of the listed work
        query = (
            f"WITH RECURSIVE tree(item_id, listed_id) AS (SELECT item_id, item_id "
            f"FROM {DB_TABLE_WORKS} WHERE {roots} "
            f"UNION SELECT w.item_id, tree.listed_id "
            f"FROM {DB_TABLE_WORKS} w JOIN tree ON w.parent_work_id = tree.item_id) "
            "SELECT tree.item_id, tree.listed_id "
            f"FROM tree JOIN {DB_TABLE_WORKS} w ON w.item_id = tree.item_id "
            "ORDER BY tree.item_id != tree.listed_id, w.name_sort, w.item_id"
        )
        parts: dict[int, list[int]] = defaultdict(list)
        for row in await self.mass.music.database.get_rows_from_query(query, params, limit=0):
            parts[int(row["listed_id"])].append(int(row["item_id"]))
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
            None if listed_ids is None else [x for ids in parts.values() for x in ids]
        )
        listed: dict[int, list[_Performance]] = defaultdict(list)
        for listed_id, work_ids in parts.items():
            for work_id in work_ids:
                listed[listed_id].extend(performances[work_id])
        return listed

    async def _add_co_performers_and_liked(
        self,
        entries: Sequence[ClassicalWorkEntry],
        performances: dict[int, list[_Performance]],
        performer_id: int,
    ) -> None:
        """
        Fill in who else is on the performer's recordings of each row and if the user likes them.

        :param entries: The rows of the works list filtered on the performer.
        :param performances: The performances of the works of the rows and their parts, by
            work id of the row.
        :param performer_id: Library id of the performer.
        """
        own = {
            work_id: [x for x in performances[work_id] if performer_id in x.performer_ids]
            for work_id in (int(entry.work.item_id) for entry in entries)
        }
        tracks = await self.mass.music.tracks.get_library_tracks_with_credits(
            [track_id for value in own.values() for x in value for track_id, _ in x.tracks]
        )
        for entry in entries:
            recordings = _performer_recordings(own[int(entry.work.item_id)], tracks)
            entry.co_performers = _co_performer_names(
                (x for shown, _ in recordings for x in shown), performer_id
            )
            row_tracks = [x for _, value in recordings for x in value]
            entry.liked = all(x.favorite is True for x in row_tracks) if row_tracks else None

    async def _performances(self, work_ids: Iterable[int] | None) -> dict[int, list[_Performance]]:
        """Return the performances of the given classical works, or of every classical work."""
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
        for work_id, work_groups in groups.items():
            performances[work_id] = [
                _build_performance(work_id, key, group) for key, group in work_groups.items()
            ]
        return performances


@dataclass(kw_only=True)
class _ShownWork:
    """What a works list row shows of a work, as far as the list searches and filters on it."""

    names: list[str]  # the first composer, the title and the first catalogue number
    work_type: WorkType | None = None
    year: int | None = None

    def search_text(self, part_names: Sequence[str]) -> str:
        """Return the text the works search looks in, with the names of the work's parts."""
        return create_safe_string(" ".join([*self.names, *part_names]), True, True)


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
    work_id: int
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
    # in album order
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
        work_id=work_id,
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
    performing_credits = _recording_credits(movement_tracks)
    return Recording(
        key=performance.key,
        work=work,
        parent_work=parent_work,
        tracks=movement_tracks,
        credits=performing_credits,
        year=performance.year,
        albums=_distinct_albums(albums[x] for x in performance.album_ids if x in albums),
        duration=sum(x.duration or 0 for x in movement_tracks),
    )


def _distinct_albums(albums: Iterable[ItemMapping]) -> list[ItemMapping]:
    """Return the albums without repeats, the first of the same id or the same name and year."""
    seen: set[str] = set()
    distinct: list[ItemMapping] = []
    for album in albums:
        keys = (album.item_id, f"{album.name}|{'' if album.year is None else album.year}")
        if seen.intersection(keys):
            continue
        seen.update(keys)
        distinct.append(album)
    return distinct


def _add_part_headings(
    recordings: Sequence[Recording], work: ItemMapping, parts: Sequence[ItemMapping]
) -> None:
    """
    Give each recording of a work the heading of the work it shows under on the work's page.

    The headings are the work, then each of its parts, then any other work a recording is of, in
    the order of the recordings. A recording of a part of a listed part, such as an aria of an
    act, shows under that part.

    :param recordings: The recordings of the work and its parts, in the order the page lists them.
    :param work: The work of the page.
    :param parts: The parts of the work, in the order the page lists them.
    """
    works = {x.item_id: x for x in (work, *parts)}
    order = {x: index for index, x in enumerate(works)}
    ranks: Counter[str] = Counter()
    for recording in recordings:
        key = recording.work.item_id
        if key not in works and recording.parent_work and recording.parent_work.item_id in works:
            key = recording.parent_work.item_id
        recording.heading = RecordingHeading(
            kind=RecordingHeadingKind.WORK,
            work=works.setdefault(key, recording.work),
            order=order.setdefault(key, len(order)),
            rank=ranks[key],
        )
        ranks[key] += 1


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


def _recording_credits(tracks: Iterable[Track]) -> list[Credit]:
    """Return the performing credits of a recording of the given tracks, in role order."""
    performing_credits: dict[tuple[str, ArtistRole, str | None], Credit] = {}
    for track in tracks:
        for credit in _track_performing_credits(track):
            performing_credits.setdefault(
                (credit.artist.item_id, credit.role, credit.instrument), credit
            )
    return sorted(performing_credits.values(), key=lambda x: PERFORMING_ROLES.index(x.role))


def _performer_recordings(
    performances: Iterable[_Performance],
    tracks: dict[int, Track],
) -> list[tuple[list[Credit], list[Track]]]:
    """
    Return the credits and tracks of performances, in the order the recordings command lists them.

    :param performances: The performances of a listed work and of its parts, per work in the
        order the recordings command lists the works.
    :param tracks: The library tracks of the performances, by track id.
    """
    recordings: list[tuple[list[Credit], list[Track]]] = []
    for _, work_performances in groupby(performances, key=lambda x: x.work_id):
        built: list[tuple[_Performance, list[Credit], list[Track]]] = []
        for performance in work_performances:
            performance_tracks = [tracks[x] for x, _ in performance.tracks if x in tracks]
            built.append((performance, _recording_credits(performance_tracks), performance_tracks))
        built.sort(key=lambda x: _recording_sort_key(x[0].year, x[1], x[0].key))
        recordings.extend((shown, performance_tracks) for _, shown, performance_tracks in built)
    return recordings


def _co_performer_names(recording_credits: Iterable[Credit], performer_id: int) -> list[str]:
    """
    Return the names of the others performing on a performer's recordings, each once.

    They come by role as CO_PERFORMER_RANK orders them, then by credit position.

    :param recording_credits: The credits of the recordings, in the order of the recordings.
    :param performer_id: Library id of the performer.
    """
    named = sorted(
        (
            x
            for x in recording_credits
            if x.role in CO_PERFORMER_RANK and x.artist.item_id != str(performer_id)
        ),
        key=lambda x: (CO_PERFORMER_RANK[x.role], x.position),
    )
    return list({x.artist.item_id: x.artist.name for x in named}.values())


def _recording_sort_key(
    year: int | None, recording_credits: Sequence[Credit], key: str
) -> tuple[Any, ...]:
    """Return the key that puts recordings by year, then conductor, then ensemble."""

    def first_name(roles: tuple[ArtistRole, ...]) -> str:
        return next((x.artist.name.casefold() for x in recording_credits if x.role in roles), "")

    return (
        year is None,
        year or 0,
        first_name((ArtistRole.CONDUCTOR,)),
        first_name(ENSEMBLE_ROLES),
        key,
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
    work.work_type = work.work_type or title_work_type(work.name)
    return work


def _sort_sql(table: str, field: SortField, direction: SortDirection | None) -> str:
    """Return the ORDER BY clause of a base sort field on the columns of a table."""
    sql = BASE_SORT_FIELD_SQL[field].format(prefix=f"{table}.")
    return f"{sql} {(direction or SortDirection.ASC).value.upper()}"


def _page[T](items: list[T], limit: int, offset: int) -> list[T]:
    """Return one page of the given items, all from the offset without a limit."""
    return items[offset : offset + limit] if limit else items[offset:]
