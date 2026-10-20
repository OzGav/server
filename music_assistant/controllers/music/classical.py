"""
Classical browse API of the Music controller.

Serves the composer, performer, work, recording and other track listings of the Classical
view. Every listing only considers classical library tracks.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, cast

from music_assistant_models.auth import Scope
from music_assistant_models.enums import (
    ArtistRole,
    ExternalID,
    ImageType,
    ListingType,
    MediaType,
    Period,
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
    WorkMovement,
    WorkSummary,
)

from music_assistant.constants import (
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_ALBUMS,
    DB_TABLE_ARTISTS,
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACK_ARTISTS,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
    OPERA_WORK_TYPES,
)
from music_assistant.controllers.music.constants import BASE_SORT_FIELD_SQL
from music_assistant.controllers.music.helpers import preferred_thumb, search_name_match_clause
from music_assistant.controllers.music.listing import resolve_listing_sort
from music_assistant.controllers.music.media.works import (
    WORK_ROOTS,
    work_search_sql,
    work_sort_sql,
)
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.json import json_loads
from music_assistant.helpers.provider_access import hidden_music_sources
from music_assistant.helpers.work_titles import part_key, word_and_number

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
# a quoted name within a movement name, such as the aria of 'Atto I, no. 6. "Che gelida
# manina!" (Rodolfo)', where a single quote within a word is an apostrophe
_QUOTED_NAME = re.compile(r"\"([^\"]+)\"|“([^”]+)”|«([^»]+)»|(?<!\w)'([^']+)'(?!\w)")

# a type label in brackets, such as "(Opera)", unless a quoted name follows as in an excerpt,
# or as the last word after a bracket or quote, such as "Cavalleria Rusticana (Rustic Chivalry)
# Opera", so a name such as "The Phantom of the Opera" never counts
_TYPE_LABEL: Final[re.Pattern[str]] = re.compile(
    r'\((opera|ballet|operetta)\)(?!\s*["«])|[)"”»]\s*(opera|ballet|operetta)\W*$',
    re.IGNORECASE,
)

# the words that tell the kind of ensemble a name is, matched on the name without accents,
# where a word ending in orchester counts too, as in Gewandhausorchester
_ENSEMBLE_NAME_WORDS: Final[tuple[tuple[ArtistRole, re.Pattern[str]], ...]] = tuple(
    (role, re.compile(pattern))
    for role, pattern in (
        (
            ArtistRole.ORCHESTRA,
            r"\b(\w*orchester|orchestr(a|as|e|es)|orquestas?|philharmoni(a|c|e|ker)|symphon(y|iker)"
            r"|sinfoni(a|etta))\b",
        ),
        (
            ArtistRole.CHOIR,
            r"\b(choirs?|chor|choeurs?|coro|singers|chorus|kammerchor|domchor|knabenchor"
            r"|rundfunkchor|kinderchor|opernchor|konzertchor)\b",
        ),
        (
            ArtistRole.ENSEMBLE,
            r"\b(quartet|quartett|quartetto|quatuor|trio|quintet|octet|octuor|ensemble"
            r"|consort)s?\b",
        ),
    )
)
# a name of soloists, such as "Soloists of the Royal Opera House Orchestra", tells no ensemble
_SOLOISTS_NAME_WORD: Final[re.Pattern[str]] = re.compile(r"\bsoloists?\b")

# the performing credits on classical tracks aliased as t, which callers narrow to the visible
# ones, where a track without any counts its main artists that did not compose it as performers
PERFORMER_CREDITS: Final[str] = f"""FROM (
        SELECT m.track_id, m.artist_id,
            CASE m.role WHEN '{ArtistRole.MAIN_ARTIST.value}' THEN '{ArtistRole.PERFORMER.value}'
                ELSE m.role END AS role
        FROM {DB_TABLE_TRACK_ARTISTS} m
        WHERE m.role IN ({_SQL_PERFORMING_ROLES}) OR (m.role = '{ArtistRole.MAIN_ARTIST.value}'
            AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_ARTISTS} p
                WHERE p.track_id = m.track_id AND p.role IN ({_SQL_PERFORMING_ROLES}))
            AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_ARTISTS} c
                WHERE c.track_id = m.track_id AND c.artist_id = m.artist_id
                AND c.role = '{ArtistRole.COMPOSER.value}'))
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
        t.search_name, CASE WHEN json_valid(t.metadata)
            THEN json_extract(t.metadata, '$.recording_year') END AS recording_year,
        (SELECT MIN(e.external_id) FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} e
            WHERE e.media_type = '{MediaType.TRACK.value}' AND e.item_id = t.item_id
            AND e.external_id_type = '{ExternalID.MB_RECORDING.value}') AS mb_recording,
        at.album_id, at.disc_number, at.track_number, a.year
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
        countries = await self._artist_countries(ids)
        return [
            ClassicalComposer(
                artist=cast("ArtistSummary", artist),
                fanart=fanarts.get(int(artist.item_id)),
                work_count=len(works[int(artist.item_id)]),
                recording_count=sum(len(performances[x]) for x in works[int(artist.item_id)]),
                country=countries.get(int(artist.item_id)),
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

        :param role: Only include performers holding this performing role, as shown.
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
        # only performers of a work are listed, so each row counts at least one recording
        conditions = [
            f"{DB_TABLE_ARTISTS}.item_id IN (SELECT ta.artist_id {PERFORMER_CREDITS} "
            f"AND t.work_id IS NOT NULL AND t.item_id IN ({self._classical_tracks(params)}))"
        ]
        if artist_id:
            params["artist_id"] = int(artist_id)
            conditions.append(f"{DB_TABLE_ARTISTS}.item_id = :artist_id")
        if field == SortField.RECORDING_COUNT:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, SortField.NAME, SortDirection.ASC)
        else:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, field, direction)
        # the role filter matches the refined roles, so it pages after reading every credit
        if role or field == SortField.RECORDING_COUNT:
            ids = await self._ids(DB_TABLE_ARTISTS, conditions, params, search, sort_key)
            roles, works = await self._performer_credits(ids)
            ids = [x for x in ids if not role or role in roles[x]]
        else:
            ids = await self._ids(
                DB_TABLE_ARTISTS, conditions, params, search, sort_key, limit, offset
            )
            roles, works = await self._performer_credits(ids)
        if field == SortField.RECORDING_COUNT:
            performances = await self._listed_work_performances(None)
            counts = {x: _performer_recording_count(x, works[x], performances) for x in ids}
            ids = _page(
                sorted(ids, key=lambda x: counts[x], reverse=direction == SortDirection.DESC),
                limit,
                offset,
            )
        else:
            if role:
                ids = _page(ids, limit, offset)
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
            the year composed or the recording count.
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
            sort_key = work_sort_sql(SortField.COMPOSER, SortDirection.ASC)
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
            sort_key = work_sort_sql(field, direction)
            ids = await self._work_ids(sort_key, limit=limit, offset=offset, **filters)
            performances = await self._listed_work_performances(ids)
        part_names = await self._part_names(ids)
        works = await self._items(self.mass.music.works, ids)
        periods = await self._composer_periods(works)
        return [
            ClassicalWorkEntry(
                work=_with_title_work_type(cast("WorkSummary", work)),
                recording_count=_recording_count(performances[int(work.item_id)], performer),
                part_names=part_names[int(work.item_id)],
                period=periods.get(int(work.item_id)),
            )
            for work in works
        ]

    async def work(self, item_id: str) -> Work:
        """
        Get a library work as the classical listings show it, with its movements.

        A work without a type of its own shows the type its title names, as in the works listing.
        The movements are those MusicBrainz lists, else those its tracks name. A staged work,
        such as an opera, a work whose parts have parts of their own and a work without
        movements list its parts with tracks instead, at every depth and in work order.

        :param item_id: The library id of the work.
        """
        work = _with_title_work_type(await self.mass.music.works.get_library_item(item_id))
        movements = (await self._movements([int(work.item_id)])).get(int(work.item_id))
        if movements:
            work.movements = movements.movements
            work.movements_from_musicbrainz = movements.from_musicbrainz
        return work

    async def recordings(
        self, work_id: str | None = None, performer_id: str | None = None
    ) -> list[Recording]:
        """
        Get the recordings of a work, or every recording a performer performs on.

        Recordings of a work come by year (undated last), then conductor, then ensemble,
        followed by those of each of its parts, each recording carrying its own work and the
        work that one is a part of, if any. A performer's recordings across their works come per work, in the order of the works
        listing, and within a work in that same order. Each recording names the movements of
        the requested work it covers, and whether that is all of them when MusicBrainz lists
        them. Its tracks carry the number of the movement they are matched to.

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
        listed_works = {
            x: listed_id for listed_id in listed_ids for x in parts.get(listed_id, [listed_id])
        }
        work_ids = list(listed_works)
        work_items = await self._items(self.mass.music.works, work_ids)
        works = {int(work.item_id): ItemMapping.from_item(work) for work in work_items}
        parent_works = {
            int(work.item_id): ItemMapping.from_item(work.parent_work)
            for work in work_items
            if work.parent_work
        }
        performances = await self._performances(work_ids)
        movements = await self._movements(listed_ids)
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
        refinements = await self._show_refined_roles(tracks.values())
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
                    _build_recording(
                        works[work],
                        parent_works.get(work),
                        x,
                        tracks,
                        albums,
                        refinements,
                        movements.get(listed_works[work], _Movements()),
                    )
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
        conditions = [
            f"{DB_TABLE_TRACKS}.is_classical = 1",
            self.mass.music.tracks.visible_library_clause(params),
            f"{DB_TABLE_TRACKS}.work_id IS NULL",
            f"{DB_TABLE_TRACKS}.item_id IN (SELECT track_id FROM {DB_TABLE_TRACK_ARTISTS} "
            f"WHERE artist_id = :artist_id AND role {role_match} '{ArtistRole.COMPOSER.value}')",
        ]
        if field == SortField.YEAR:
            # tracks without an album year come last
            order = (direction or SortDirection.ASC).value.upper()
            name_sort = _sort_sql(DB_TABLE_TRACKS, SortField.NAME, SortDirection.ASC)
            sort_key = f"{TRACK_ALBUM_YEAR} {order} NULLS LAST, {name_sort}"
        else:
            sort_key = _sort_sql(DB_TABLE_TRACKS, field, direction)
        ids = await self._ids(DB_TABLE_TRACKS, conditions, params, search, sort_key, limit, offset)
        tracks = await self._items(self.mass.music.tracks, ids, summary=False)
        await self._show_refined_roles(tracks)
        return tracks

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
        """Return the ids of the listed works matching the filters, in order, all without a limit."""
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

    async def _artist_countries(self, artist_ids: Sequence[int]) -> dict[int, str]:
        """Return the country code of the given library artists that have one, by artist id."""
        if not artist_ids:
            return {}
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id, json_extract(metadata, '$.country') AS country "
            f"FROM {DB_TABLE_ARTISTS} WHERE item_id IN :artist_ids "
            "AND json_extract(metadata, '$.country') IS NOT NULL",
            {"artist_ids": list(artist_ids)},
            limit=0,
        )
        return {int(row["item_id"]): row["country"] for row in rows}

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
        Return the credit count per shown performing role and the listed works of performers.

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
        names, stored = await self._stored_roles(artist_ids)
        for artist_id, role_counts in stored.items():
            roles[artist_id] = _refined_roles(names[artist_id], role_counts)
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT c.artist_id, r.root_id AS work_id "
            f"FROM (SELECT ta.artist_id, t.work_id {PERFORMER_CREDITS}{scope}) c "
            f"JOIN ({WORK_ROOTS}) r ON r.item_id = c.work_id",
            params,
            limit=0,
        ):
            works[int(row["artist_id"])].add(int(row["work_id"]))
        return roles, works

    async def _stored_roles(
        self, artist_ids: Sequence[int] | None
    ) -> tuple[dict[int, str], dict[int, Counter[ArtistRole]]]:
        """
        Return the names and the credit count per stored performing role of performers.

        :param artist_ids: The performers to return, or None for every performer.
        """
        names: dict[int, str] = {}
        roles: dict[int, Counter[ArtistRole]] = defaultdict(Counter)
        if artist_ids is not None and not artist_ids:
            return names, roles
        params: dict[str, Any] = {}
        scope = f" AND t.item_id IN ({self._classical_tracks(params)})"
        if artist_ids is not None:
            scope += " AND ta.artist_id IN :artist_ids"
            params["artist_ids"] = list(artist_ids)
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT ta.artist_id, ta.role, COUNT(DISTINCT ta.track_id) AS credit_count, "
            f"(SELECT name FROM {DB_TABLE_ARTISTS} WHERE item_id = ta.artist_id) AS name "
            f"{PERFORMER_CREDITS}{scope} GROUP BY ta.artist_id, ta.role",
            params,
            limit=0,
        ):
            names[int(row["artist_id"])] = row["name"] or ""
            roles[int(row["artist_id"])][ArtistRole(row["role"])] = int(row["credit_count"])
        return names, roles

    async def _show_refined_roles(
        self, tracks: Iterable[Track]
    ) -> dict[int, dict[ArtistRole, ArtistRole]]:
        """
        Show the refined performing roles in the credits of tracks, as the performers listing does.

        Returns the shown role of each refined stored role, by performer.

        :param tracks: The library tracks to update.
        """
        tracks = list(tracks)
        names, stored = await self._stored_roles(
            list(
                {
                    int(x.artist.item_id)
                    for track in tracks
                    for x in track.credits
                    if x.role in PERFORMING_ROLES or x.role == ArtistRole.MAIN_ARTIST
                }
            )
        )
        refinements = {x: _role_refinements(names[x], stored[x]) for x in stored}
        for track in tracks:
            track.credits = _shown_credits(track.credits, refinements)
        return refinements

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
            f"UNION SELECT w.item_id, tree.listed_id FROM {DB_TABLE_WORKS} w "
            "JOIN tree ON w.parent_work_id = tree.item_id) "
            f"SELECT tree.item_id, tree.listed_id FROM tree JOIN {DB_TABLE_WORKS} w "
            "ON w.item_id = tree.item_id "
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

    async def _composer_periods(self, works: Sequence[Work]) -> dict[int, Period]:
        """Return the period of the first composer of the given works, for those with one."""
        composer_ids = {int(x.item_id): int(x.composers[0].item_id) for x in works if x.composers}
        if not composer_ids:
            return {}
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id, period FROM {DB_TABLE_ARTISTS} "
            "WHERE item_id IN :artist_ids AND period IS NOT NULL",
            {"artist_ids": list(set(composer_ids.values()))},
            limit=0,
        )
        periods = {int(row["item_id"]): Period(row["period"]) for row in rows}
        return {
            work_id: periods[artist_id]
            for work_id, artist_id in composer_ids.items()
            if artist_id in periods
        }

    async def _movements(self, work_ids: Sequence[int]) -> dict[int, _Movements]:
        """
        Return the movements of the given works by work id, leaving out works without any.

        :param work_ids: Library ids of the works.
        """
        movements: dict[int, _Movements] = {}
        if not work_ids:
            return movements
        database = self.mass.music.database
        params: dict[str, Any] = {"work_ids": list(set(work_ids))}
        # a staged work, such as an opera, and a work whose parts have parts, such as acts,
        # list their parts instead
        rows = await database.get_rows_from_query(
            "SELECT w.item_id, CASE WHEN json_valid(w.metadata) "
            "THEN json_extract(w.metadata, '$.movements') END AS movements "
            f"FROM {DB_TABLE_WORKS} w WHERE w.item_id IN :work_ids "
            "AND (w.work_type IS NULL OR w.work_type NOT IN :staged_types) "
            f"AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_WORKS} p JOIN {DB_TABLE_WORKS} c "
            "ON c.parent_work_id = p.item_id WHERE p.parent_work_id = w.item_id)",
            {**params, "staged_types": [x.value for x in OPERA_WORK_TYPES]},
            limit=0,
        )
        unlisted: list[int] = []
        for row in rows:
            if stored := _stored_movements(row["movements"]):
                movements[int(row["item_id"])] = _Movements(movements=stored, from_musicbrainz=True)
            else:
                unlisted.append(int(row["item_id"]))
        if unlisted:
            movements |= await self._track_movements({"work_ids": unlisted})
        # a work without movements of its own lists its parts instead
        return movements | await self._part_movements(
            [x for x in set(work_ids) if x not in movements]
        )

    async def _track_movements(self, params: dict[str, Any]) -> dict[int, _Movements]:
        """Return the movements the classical tracks of the works in params name, by work id."""
        # the most common name each movement number of a work goes by on its tracks
        names: dict[int, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT work_id, movement_number, movement_name FROM {DB_TABLE_TRACKS} "
            "WHERE work_id IN :work_ids AND movement_number IS NOT NULL "
            f"AND item_id IN ({self._classical_tracks(params)}) ORDER BY item_id",
            params,
            limit=0,
        ):
            counts = names[int(row["work_id"])][int(row["movement_number"])]
            if row["movement_name"]:
                counts[row["movement_name"]] += 1
        return {
            work_id: _Movements(
                movements=[
                    WorkMovement(
                        number=number,
                        name=counts.most_common(1)[0][0] if counts else "",
                    )
                    for number, counts in sorted(numbers.items())
                ]
            )
            for work_id, numbers in names.items()
        }

    async def _part_movements(self, work_ids: Sequence[int]) -> dict[int, _Movements]:
        """
        Return the parts of the given works that have classical tracks, as their movements.

        The parts at every depth come in work order, each after the listed part it belongs
        to, which is its parent.

        :param work_ids: Library ids of the works.
        """
        if not work_ids:
            return {}
        params: dict[str, Any] = {"work_ids": list(work_ids)}
        rows = await self.mass.music.database.get_rows_from_query(
            f"WITH RECURSIVE tree(item_id, root_id) AS (SELECT item_id, item_id "
            f"FROM {DB_TABLE_WORKS} WHERE item_id IN :work_ids "
            f"UNION SELECT w.item_id, tree.root_id FROM {DB_TABLE_WORKS} w "
            "JOIN tree ON w.parent_work_id = tree.item_id) "
            "SELECT tree.root_id, w.item_id, w.parent_work_id, w.name, w.name_sort, "
            "CASE WHEN json_valid(w.metadata) "
            "THEN json_extract(w.metadata, '$.movements') END AS movements, "
            f"(SELECT MIN(e.external_id) FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} e "
            f"WHERE e.media_type = '{MediaType.WORK.value}' AND e.item_id = w.item_id "
            f"AND e.external_id_type = '{ExternalID.MB_WORK.value}') AS mbid, "
            f"w.item_id IN ({self._classical_work_ids(params)}) AS recorded "
            f"FROM tree JOIN {DB_TABLE_WORKS} w ON w.item_id = tree.item_id",
            params,
            limit=0,
        )
        trees: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
        for row in rows:
            trees[int(row["root_id"])][int(row["item_id"])] = row
        ancestors = await self._ancestor_names(list(trees))
        movements: dict[int, _Movements] = {}
        for root_id, tree in trees.items():
            if parts := _ordered_parts(root_id, tree, ancestors[root_id]):
                movements[root_id] = _Movements(movements=list(parts.values()), parts=parts)
        return movements

    async def _ancestor_names(self, work_ids: Sequence[int]) -> dict[int, list[str]]:
        """Return the names of the works the given works are a part of, at any depth."""
        names: dict[int, list[str]] = defaultdict(list)
        if not work_ids:
            return names
        for row in await self.mass.music.database.get_rows_from_query(
            f"WITH RECURSIVE up(work_id, item_id) AS (SELECT item_id, parent_work_id "
            f"FROM {DB_TABLE_WORKS} WHERE item_id IN :work_ids AND parent_work_id IS NOT NULL "
            f"UNION SELECT up.work_id, w.parent_work_id FROM {DB_TABLE_WORKS} w "
            "JOIN up ON w.item_id = up.item_id WHERE w.parent_work_id IS NOT NULL) "
            f"SELECT up.work_id, w.name FROM up JOIN {DB_TABLE_WORKS} w ON w.item_id = up.item_id",
            {"work_ids": list(work_ids)},
            limit=0,
        ):
            names[int(row["work_id"])].append(row["name"])
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
                    movement_number=row["movement_number"],
                    movement=row["movement_number"]
                    or create_safe_string(row["movement_name"] or "", True, True)
                    or row["search_name"],
                    mb_recording=(row["mb_recording"] or "").casefold() or None,
                    recording_year=row["recording_year"],
                )
                work_tracks[int(row["work_id"])].append(track)
            if row["album_id"] is not None:
                track.appearances.append(
                    _Appearance(
                        album_id=int(row["album_id"]),
                        year=row["year"],
                        disc_number=row["disc_number"] or 0,
                        track_number=row["track_number"] or 0,
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
        for work_id, work_track_list in work_tracks.items():
            performances[work_id] = _group_performances(work_id, work_track_list)
        return performances


@dataclass(kw_only=True)
class _Appearance:
    """The place of a track on an album."""

    album_id: int
    year: int | None
    disc_number: int
    track_number: int


@dataclass(kw_only=True)
class _WorkTrack:
    """A classical track of a work, with what tells its performance apart."""

    track_id: int
    movement_number: int | None
    movement: int | str
    mb_recording: str | None
    recording_year: int | None = None
    performers: set[int] = field(default_factory=set)
    leaders: set[int] = field(default_factory=set)
    appearances: list[_Appearance] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        """Return the year the track was recorded, else the earliest year of its albums."""
        return self.recording_year or min(
            (x.year for x in self.appearances if x.year), default=None
        )

    @property
    def signature(self) -> frozenset[int]:
        """Return the performers that identify the performance of this track."""
        # without a conductor or ensemble, the soloists and other performers identify it
        return frozenset(self.leaders or self.performers)


@dataclass(kw_only=True)
class _Performance:
    """One performance of a work, the grouping a Recording is built from."""

    key: str
    year: int | None
    performer_ids: frozenset[int]
    album_ids: list[int]
    # one track per movement, in movement order, with the album it is taken from
    tracks: list[tuple[int, _Appearance | None]]


@dataclass(kw_only=True)
class _Movements:
    """The movements of a work, as its listing and the recordings of it show them."""

    movements: list[WorkMovement] = field(default_factory=list)
    from_musicbrainz: bool = False
    # the movement each part is, by part id, when the work lists its parts as its movements
    parts: dict[int, WorkMovement] = field(default_factory=dict)


def _group_performances(work_id: int, tracks: Iterable[_WorkTrack]) -> list[_Performance]:
    """Group the classical tracks of a work into its performances."""
    # the same movement found on several albums is one recording of that movement
    movements: dict[tuple[Any, ...], list[_WorkTrack]] = defaultdict(list)
    for track in tracks:
        if track.mb_recording:
            movements[(track.mb_recording,)].append(track)
        else:
            movements[(track.signature, track.movement, track.year)].append(track)
    groups: dict[tuple[frozenset[int], int | None], list[list[_WorkTrack]]] = defaultdict(list)
    for movement in movements.values():
        # a reissued movement belongs to the performance of its earliest release
        first = min(movement, key=lambda x: (x.year is None, x.year or 0, x.track_id))
        groups[(first.signature, first.year)].append(movement)
    return [
        _build_performance(work_id, signature, year, group)
        for (signature, year), group in groups.items()
    ]


def _build_performance(
    work_id: int, signature: frozenset[int], year: int | None, movements: list[list[_WorkTrack]]
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
    chosen.sort(
        key=lambda x: (
            x[0].movement_number is None,
            x[0].movement_number or 0,
            x[1].disc_number if x[1] else 0,
            x[1].track_number if x[1] else 0,
            x[0].track_id,
        )
    )
    digest = hashlib.sha1(f"{sorted(signature)}:{year}".encode(), usedforsecurity=False)
    return _Performance(
        key=f"{work_id}-{digest.hexdigest()[:16]}",
        year=year,
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
    refinements: dict[int, dict[ArtistRole, ArtistRole]],
    movements: _Movements,
) -> Recording:
    """
    Build a Recording from a performance and its library tracks and albums.

    :param work: The work of the performance.
    :param parent_work: The work that work is a part of, if any.
    :param performance: The performance to build the recording of.
    :param tracks: The library tracks of the performance, by track id.
    :param albums: The albums of the performance, by album id.
    :param refinements: The shown role of each refined stored performing role, by performer.
    :param movements: The movements of the requested work, which may be the parent of work.
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
    performing_credits = _shown_credits(
        (x for track in movement_tracks for x in _track_performing_credits(track)), refinements
    )
    covered: set[int] = set()
    own_part = movements.parts.get(int(work.item_id))
    for track in movement_tracks:
        if matched := _track_movements(track, movements, own_part):
            covered.update(x.number for x in matched)
            # the movement fields are only changed on the returned track, never stored
            track.movement_number = matched[0].number
            track.movement_name = track.movement_name or matched[0].name or None
    return Recording(
        key=performance.key,
        work=work,
        parent_work=parent_work,
        tracks=movement_tracks,
        credits=sorted(performing_credits, key=lambda x: PERFORMING_ROLES.index(x.role)),
        year=performance.year,
        albums=[albums[x] for x in performance.album_ids if x in albums],
        duration=sum(x.duration or 0 for x in movement_tracks),
        movement_numbers=sorted(covered),
        complete=(
            {x.number for x in movements.movements} <= covered
            if movements.from_musicbrainz
            else None
        ),
    )


def _track_movements(
    track: Track, movements: _Movements, own_part: WorkMovement | None
) -> list[WorkMovement]:
    """
    Return the movements a track of a recording holds, none when unknown or ambiguous.

    :param track: The track.
    :param movements: The movements of the requested work.
    :param own_part: The part of the requested work the track's work is, if any.
    """
    name = track.movement_name or track.name.rpartition(": ")[2]
    if own_part:
        # a track of an act may be one of the arias listed with it
        arias = [
            x
            for x in movements.movements
            if x.number != own_part.number
            and _QUOTED_NAME.search(x.name)
            and _same_movement_name(name, x.name)
        ]
        return arias if len(arias) == 1 else [own_part]
    mb_works = {
        mbid.casefold() for id_type, mbid in track.external_ids if id_type == ExternalID.MB_WORK
    }
    # a track's MusicBrainz work names its movements, as one track may hold two
    if by_work := [x for x in movements.movements if x.mbid and x.mbid.casefold() in mb_works]:
        return by_work
    # the numbers of listed parts are their positions, not the numbers on their tracks
    if track.movement_number and not movements.parts:
        if by_number := [x for x in movements.movements if x.number == track.movement_number]:
            return by_number
    by_name = [x for x in movements.movements if _same_movement_name(name, x.name)]
    return by_name if len(by_name) == 1 else []


def _same_movement_name(name: str, other: str) -> bool:
    """
    Return whether two names name the same movement, as "Var. 1" and "Variatio 1." do.

    A quoted name within either, such as the aria in 'No. 6. "Che gelida manina!"', also
    names the movement.
    """
    keys = {create_safe_string(x, True, True) for x in (name, *_quoted_names(name))}
    other_keys = {create_safe_string(x, True, True) for x in (other, *_quoted_names(other))}
    if (keys & other_keys) - {""}:
        return True
    if not (numbered := word_and_number(name)) or not (other_numbered := word_and_number(other)):
        return False
    (word, number), (other_word, other_number) = sorted(
        (numbered, other_numbered), key=lambda x: len(x[0])
    )
    return number == other_number and other_word.startswith(word)


def _quoted_names(name: str) -> list[str]:
    """Return the quoted names within a name, such as the aria of a numbered opera scene."""
    return [next(x for x in match.groups() if x) for match in _QUOTED_NAME.finditer(name)]


def _stored_movements(raw: str | None) -> list[WorkMovement]:
    """Return the movements stored in a work's metadata, in movement order."""
    if not raw:
        return []
    # a later lookup that renamed a movement adds it after the earlier name
    stored = {x.number: x for x in map(WorkMovement.from_dict, json_loads(raw))}
    return sorted(stored.values(), key=lambda x: x.number)


def _ordered_parts(
    root_id: int, tree: Mapping[int, Mapping[str, Any]], ancestors: Sequence[str]
) -> dict[int, WorkMovement]:
    """
    Return the movement each part of a work with tracks is, by part id, in work order.

    The parent of a movement is the nearest part it belongs to that is listed too.

    :param root_id: Library id of the work.
    :param tree: The work and all its parts, by work id.
    :param ancestors: Names of the works the work is a part of.
    """
    children: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item_id, row in tree.items():
        if item_id != root_id and row["parent_work_id"] is not None:
            children[int(row["parent_work_id"])].append(row)
    ordered: dict[int, WorkMovement] = {}

    def add_parts(parent_id: int, ancestors: list[str], parent: int | None) -> None:
        # the movements MusicBrainz lists for a work give the order of its parts
        positions = {
            x.mbid.casefold(): x.number
            for x in _stored_movements(tree[parent_id]["movements"])
            if x.mbid
        }
        names = [tree[parent_id]["name"], *ancestors]
        for row in sorted(children[parent_id], key=lambda x: _part_order(x, positions, names)):
            part_id = int(row["item_id"])
            movement = None
            if row["recorded"]:
                movement = ordered[part_id] = WorkMovement(
                    number=len(ordered) + 1,
                    name=_part_name(row["name"], names),
                    mbid=row["mbid"],
                    parent=parent,
                    work_id=str(part_id),
                )
            add_parts(part_id, names, movement.number if movement else parent)

    add_parts(root_id, list(ancestors), None)
    return ordered


def _part_order(
    row: Mapping[str, Any], positions: dict[str, int], names: Sequence[str]
) -> tuple[Any, ...]:
    """Return the sort key of a part among the other parts of its work."""
    position = positions.get((row["mbid"] or "").casefold())
    if position is None and (key := part_key(row["name"])) and isinstance(key[1], int):
        position = key[1]
    if position is None and (numbered := word_and_number(_part_name(row["name"], names))):
        position = numbered[1]
    return position is None, position or 0, row["name_sort"], int(row["item_id"])


def _part_name(name: str, names: Sequence[str]) -> str:
    """Return the name of a part without the longest of the given work names it starts with."""
    shortest = name
    for prefix in names:
        if not (key := create_safe_string(prefix, True, True)):
            continue
        for end in range(1, len(name)):
            head = create_safe_string(name[:end], True, True)
            if not key.startswith(head):
                break
            # the work name ends where a word does, as "Tosca" never starts "Toscanini"
            if head == key and not name[end].isalnum():
                rest = name[end:].lstrip(" ,.:;/-\u2013\u2014")
                if rest and len(rest) < len(shortest):
                    shortest = rest
                break
    return shortest


def _track_performing_credits(track: Track) -> list[Credit]:
    """Return the performing credits of a track, as counted by PERFORMER_CREDITS."""
    if performing := [x for x in track.credits if x.role in PERFORMING_ROLES]:
        return performing
    composer_ids = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
    return [
        Credit(artist=x.artist, role=ArtistRole.PERFORMER, position=x.position)
        for x in track.credits
        if x.role == ArtistRole.MAIN_ARTIST and x.artist.item_id not in composer_ids
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


def _refined_roles(name: str, role_counts: Counter[ArtistRole]) -> Counter[ArtistRole]:
    """
    Return the credit count per shown performing role of a performer.

    :param name: The name of the performer.
    :param role_counts: The credit count per stored performing role.
    """
    refinements = _role_refinements(name, role_counts)
    refined: Counter[ArtistRole] = Counter()
    for role, count in role_counts.items():
        refined[refinements.get(role, role)] += count
    return refined


def _role_refinements(name: str, roles: Iterable[ArtistRole]) -> dict[ArtistRole, ArtistRole]:
    """
    Return the shown role of each stored performing role of a performer that is refined.

    The performer role takes the one specific role the performer holds elsewhere, or else the
    ensemble role their name tells. The orchestra role gives way to a choir or ensemble role
    their name tells, the ensemble role to a choir role.

    :param name: The name of the performer.
    :param roles: The stored performing roles of the performer on the classical tracks.
    """
    roles = set(roles)
    named = _name_ensemble_role(name)
    refinements: dict[ArtistRole, ArtistRole] = {}
    if named in (ArtistRole.CHOIR, ArtistRole.ENSEMBLE) and ArtistRole.ORCHESTRA in roles:
        refinements[ArtistRole.ORCHESTRA] = named
    if named == ArtistRole.CHOIR and ArtistRole.ENSEMBLE in roles:
        refinements[ArtistRole.ENSEMBLE] = named
    if ArtistRole.PERFORMER in roles:
        specific = {refinements.get(x, x) for x in roles if x != ArtistRole.PERFORMER}
        if target := (specific.pop() if len(specific) == 1 else named):
            refinements[ArtistRole.PERFORMER] = target
    return refinements


def _shown_credits(
    track_credits: Iterable[Credit], refinements: dict[int, dict[ArtistRole, ArtistRole]]
) -> list[Credit]:
    """
    Return credits with their refined roles, each credit once.

    :param track_credits: The credits to show.
    :param refinements: The shown role of each refined stored performing role, by performer.
    """
    shown: dict[tuple[str, ArtistRole, str | None], Credit] = {}
    for x in track_credits:
        role = x.role
        # a chorus master who conducts elsewhere would otherwise show as conductor
        if not x.instrument:
            role = refinements.get(int(x.artist.item_id), {}).get(x.role, x.role)
        credit = x if role == x.role else replace(x, role=role)
        shown.setdefault((credit.artist.item_id, credit.role, credit.instrument), credit)
    return list(shown.values())


def _name_ensemble_role(name: str) -> ArtistRole | None:
    """Return the ensemble role the words of a name tell, None when they tell none or several."""
    words = create_safe_string(re.sub(r"\W", " ", name))
    if _SOLOISTS_NAME_WORD.search(words):
        return None
    named = {role for role, pattern in _ENSEMBLE_NAME_WORDS if pattern.search(words)}
    return named.pop() if len(named) == 1 else None


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
