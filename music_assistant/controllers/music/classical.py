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
    RecordingCoverageKind,
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
    RecordingCoverage,
    RecordingHeading,
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
    DB_TABLE_TRACK_CREDITS,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.constants import BASE_SORT_FIELD_SQL
from music_assistant.controllers.music.helpers import (
    PIECE_NUMBERS,
    group_piece,
    preferred_thumb,
    search_name_match_clause,
)
from music_assistant.controllers.music.listing import resolve_listing_sort
from music_assistant.controllers.music.media.works import (
    SHOWN_WORK_VERSION,
    WORK_ROOTS,
    work_sort_sql,
)
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.credit_roles import PERFORMING_ROLES, SQL_PERFORMING_ROLES
from music_assistant.helpers.json import json_loads
from music_assistant.helpers.provider_access import hidden_music_sources
from music_assistant.helpers.work_titles import (
    part_key,
    part_label,
    title_work_type,
    without_movement_number,
    word_and_number,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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

# a quoted name within a movement name, such as the aria of 'Atto I, no. 6. "Che gelida
# manina!" (Rodolfo)', where a single quote within a word is an apostrophe
_QUOTED_NAME = re.compile(r"\"([^\"]+)\"|“([^”]+)”|«([^»]+)»|(?<!\w)'([^']+)'(?!\w)")

# the score number leading a movement name, such as "XXXIX.", "IVa.", "36a." or "no. 6:",
# where the "." or ":" keeps a word such as "Il" from passing for one
_LEADING_SCORE_NUMBER: Final[re.Pattern[str]] = re.compile(
    r"((?:[IVXLCDM]+|[0-9]+)[a-z]?|(?:no\.|No\.|Nr\.|N°)\s*[0-9]+[a-z]?)[.:]"
)
# a scene leading a movement name, such as "Scene V." before its quoted words
_LEADING_SCENE: Final[re.Pattern[str]] = re.compile(
    r"((?:Scene|Scène|Szene|Scena|Bild)\s+(?:[IVXLCDM]+|[0-9]+))[.:\s]"  # codespell:ignore
)
# the end of a movement name that marks it as an excerpt
_EXCERPT_END: Final[re.Pattern[str]] = re.compile(r"\s\(excerpt\)$", re.IGNORECASE)
# a quoted part of a movement name, such as “Che gelida manina!” or « Salut ! »
_QUOTED_WORDS: Final[re.Pattern[str]] = re.compile(r"“([^”]+)”|«\s*([^»]+?)\s*»|\"([^\"]+)\"")
# the separators between a work name and the rest of a track name, such as ": "
_LEADING_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"^[\s:,.\u2013\u2014-]+")

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

# the country an artist is from, as the composers list shows it
ARTIST_COUNTRY: Final[str] = f"json_extract({DB_TABLE_ARTISTS}.metadata, '$.country')"

# the year of the album a track is listed with
TRACK_ALBUM_YEAR: Final[str] = f"""(SELECT {DB_TABLE_ALBUMS}.year FROM {DB_TABLE_ALBUM_TRACKS}
    JOIN {DB_TABLE_ALBUMS} ON {DB_TABLE_ALBUMS}.item_id = {DB_TABLE_ALBUM_TRACKS}.album_id
    WHERE {DB_TABLE_ALBUM_TRACKS}.track_id = {DB_TABLE_TRACKS}.item_id
    ORDER BY {DB_TABLE_ALBUM_TRACKS}.album_id LIMIT 1)"""

# a score number opening a name, with the letter of a piece of a group if any, such as "XXXVIa. "
_SCORE_NUMBER: Final[re.Pattern[str]] = re.compile(r"([IVXLCDM]+)[a-z]?\.\s")

# the start of the MusicBrainz title of an excerpt of a scene, such as of the Walkürenritt
_EXCERPT: Final = "Excerpt from "
# a section and its number in a title, such as "Akt III, " or "Scene V. ", which the own name of
# an excerpt follows
_SECTION_LABEL: Final[re.Pattern[str]] = re.compile(r"\b[^\W\d_]+\s(?:[IVXLC]+|\d+)[.,]\s")

# the work tracks with the albums they are on, one row per album
WORK_TRACKS_QUERY: Final[str] = f"""
    SELECT t.item_id AS track_id, t.work_id, t.movement_number, t.movement_name,
        t.search_name, CASE WHEN json_valid(t.metadata)
            THEN json_extract(t.metadata, '$.recording_year') END AS recording_year,
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
        country: list[str] | None = None,
        country_unknown: bool = False,
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
        :param country: Only include composers from any of these countries (ISO 3166-1 codes).
        :param country_unknown: Also include the composers without a country; on its own,
            only those.
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
        if country or country_unknown:
            matches = [f"{ARTIST_COUNTRY} IS NULL"] if country_unknown else []
            if country:
                params["countries"] = list(country)
                matches.append(f"{ARTIST_COUNTRY} IN :countries")
            conditions.append(f"({' OR '.join(matches)})")
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
        fanarts, countries = await self._artist_fanarts_and_countries(ids)
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

    async def composer_filters(self) -> ClassicalComposerFilters:
        """Get the values the composers list can be filtered on, across all its composers."""
        params: dict[str, Any] = {}
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT {DB_TABLE_ARTISTS}.period, {ARTIST_COUNTRY} AS country "
            f"FROM {DB_TABLE_ARTISTS} WHERE {self._listed_composers(params)}",
            params,
            limit=0,
        )
        periods = {Period(row["period"]) for row in rows if row["period"] is not None}
        countries = {row["country"] for row in rows}
        return ClassicalComposerFilters(
            periods=[x for x in Period if x in periods],
            countries=sorted(x for x in countries if x is not None),
            country_unknown=None in countries,
        )

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

        :param role: Only include performers holding any of these performing roles, as
            shown.
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
        conditions = [self._listed_performers(params)]
        if artist_id:
            params["artist_id"] = int(artist_id)
            conditions.append(f"{DB_TABLE_ARTISTS}.item_id = :artist_id")
        if favorite is not None:
            conditions.append(self.mass.music.artists._favorite_filter_clause(params, favorite))
        if field == SortField.RECORDING_COUNT:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, SortField.NAME, SortDirection.ASC)
        else:
            sort_key = _sort_sql(DB_TABLE_ARTISTS, field, direction)
        # the role filter matches the refined roles, so it pages after reading every credit
        if role or field == SortField.RECORDING_COUNT:
            ids = await self._ids(DB_TABLE_ARTISTS, conditions, params, search, sort_key)
            roles, works = await self._performer_credits(ids)
            ids = [x for x in ids if not role or any(y in roles[x] for y in role)]
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
        fanarts, _ = await self._artist_fanarts_and_countries(ids)
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
        period: list[Period] | None = None,
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
        :param period: Only include works whose composer is of any of these periods, as shown.
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
        shown_filters = bool(search_term or period or work_type or work_type_unknown)
        if field == SortField.RECORDING_COUNT:
            sort_key = work_sort_sql(SortField.COMPOSER, SortDirection.ASC, bool(composer_id))
        else:
            sort_key = work_sort_sql(field, direction, bool(composer_id))
        # the search and the filters on what the rows show match in Python, so they page
        # after reading every listed work
        if shown_filters or field == SortField.RECORDING_COUNT:
            ids = await self._work_ids(sort_key, **filters)
            if shown_filters:
                ids = await self._matching_works(
                    ids, search_term, period, work_type, work_type_unknown
                )
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
        shown = await self._shown_works(ids)
        entries = [
            ClassicalWorkEntry(
                work=_with_title_work_type(cast("WorkSummary", work)),
                recording_count=_recording_count(performances[int(work.item_id)], performer),
                part_names=part_names[int(work.item_id)],
                period=shown[int(work.item_id)].period,
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
        periods = {x.period for x in shown}
        work_types = {x.work_type for x in shown}
        years = [x.year for x in shown if x.year is not None]
        return ClassicalWorkFilters(
            periods=[x for x in Period if x in periods],
            work_types=[x for x in WorkType if x in work_types],
            work_type_unknown=None in work_types,
            earliest_year=min(years, default=None),
            latest_year=max(years, default=None),
        )

    async def work(self, item_id: str) -> Work:
        """
        Get a library work as the classical listings show it, with its movements.

        The movements are those MusicBrainz lists, else those its tracks name, else its parts
        with tracks, named as parts of it, such as "Part II". The pieces of a group or of an act
        follow it, each naming it as its parent. The entries the page lists under a part, such as
        the arias of an act, are marked as such, unless the work is itself a part of another
        work. Each movement also carries a short name, such as "XXXIX. Erbarme dich, mein Gott"
        for 'XXXIX. Aria (Alto) "Erbarme dich, mein Gott"', and whether it is an excerpt. A work
        without a type of its own shows the type its title names.

        :param item_id: The library id of the work.
        """
        work = _with_title_work_type(await self.mass.music.works.get_library_item(item_id))
        related = [*work.parts, *work.arrangement_of, *work.arrangements]
        shown_works = (work, *related, *([work.parent_work] if work.parent_work else []))
        versions = await self._shown_work_versions(shown_works)
        for shown in shown_works:
            _show_work_version(shown, versions)
        names = [work.name, *(await self._ancestor_names([int(work.item_id)]))[int(work.item_id)]]
        for part in work.parts:
            part.name = part_label(part.name) or _part_name(part.name, names)
        work_movements = await self._movements([int(work.item_id)])
        await self._add_part_lists(work_movements, int(work.item_id))
        if movements := work_movements.get(int(work.item_id)):
            work.movements = _WorkPage(
                _with_pieces(movements), is_part=work.parent_work is not None
            ).listed_movements()
            work.movements_from_musicbrainz = movements.from_musicbrainz
        # the movements are sent once, as the work's own movements
        work.metadata.movements = None
        return work

    async def recordings(
        self, work_id: str | None = None, performer_id: str | None = None
    ) -> list[Recording]:
        """
        Get the recordings of a work, or every recording a performer performs on.

        Recordings of a work come by year (undated last), then conductor, then ensemble, followed by
        those of each of its parts. A performer's recordings come per work, in the order of the
        works listing. Each recording carries its work and the work that one is a part of, the
        movements it covers and whether that is all of them, and its tracks with the number of
        the movement each holds. Tracks of one performance join one recording: those of a scene
        join its act, and those of a part join the recording of the requested work itself.

        Requested for a work, each recording also carries what the work's page shows of it: the
        heading it shows under, the part of the work it is of, what it holds, the movements each
        of its tracks counts towards and the name each track shows.

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
        joined = {x: y for ids in parts.values() for x, y in ids.items()}
        if work_id:
            # a part's tracks that are one performance with the work's own join its recording,
            # after taking in those of their own parts
            joined = {x: y for x, y in joined.items() if x != y} | {
                x: int(work_id) for x, y in joined.items() if x == y
            }
        performances = await self._performances(work_ids, joined)
        movements = await self._movements(listed_ids)
        if work_id:
            await self._add_part_lists(movements, int(work_id))
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
        shown_works = (*parent_works.values(), *(x.work for x in tracks.values() if x.work))
        versions = await self._shown_work_versions(shown_works)
        for shown in shown_works:
            _show_work_version(shown, versions)
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
        found = [
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
                key=lambda x: _recording_sort_key(x.year, x.credits, x.key),
            )
        ]
        if work_id and (page := works.get(int(work_id))):
            listed = movements.get(int(work_id))
            work_page = _WorkPage(
                _with_pieces(listed) if listed else [], is_part=int(work_id) in parent_works
            )
            # a work whose movements are works of their own, such as acts, groups by those
            if any(x.work_id for x in work_page.movements):
                work_page.add_headings(found, work_id)
            else:
                listed_parts = [
                    works[x]
                    for x in work_ids
                    if x in parent_works and parent_works[x].item_id == work_id
                ]
                _add_part_headings(found, page, listed_parts)
            for recording in found:
                work_page.describe(recording)
        return found

    async def work_tracks(self, item_id: str) -> list[Track]:
        """
        Get the tracks that playing a work plays, in the order they play.

        These are the tracks of the most complete recording of the work itself: a complete one
        first, then the longest. A work without one plays the performance of the most of its parts,
        the longest on a tie, and the most complete recording of each part that one leaves out, in
        the work's order. A part without a recording counts through its own parts, such as the
        scenes of an act.

        :param item_id: The library id of the work.
        """
        recordings = await self.recordings(item_id)
        if own := [x for x in recordings if x.work.item_id == item_id]:
            return max(own, key=_completeness).tracks
        recorded = {x.work.item_id for x in recordings}
        # a part of a part, such as a scene of an act, counts only when the part it sits under
        # has no recording
        parts = [x for x in recordings if x.parent_work and x.parent_work.item_id not in recorded]
        # the recordings of one performance share their key past the work id, across the parts,
        # as _build_performance makes it
        performances: dict[str, dict[str, Recording]] = defaultdict(dict)
        for recording in parts:
            performances[recording.key.partition("-")[2]][recording.work.item_id] = recording
        played = max(
            performances.values(),
            key=lambda x: (len(x), sum(r.duration for r in x.values())),
            default={},
        )
        most_complete = {x.work.item_id: x for x in sorted(parts, key=_completeness)}
        picked = [
            played.get(x, most_complete[x]) for x in dict.fromkeys(r.work.item_id for r in parts)
        ]
        # in the order of the movements the parts are, else of their names
        picked.sort(key=lambda x: min(x.movement_numbers, default=float("inf")))
        return [track for recording in picked for track in recording.tracks]

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
        tracks = await self._items(self.mass.music.tracks, ids, summary=False)
        await self._show_refined_roles(tracks)
        return tracks

    async def link_progress(self) -> dict[str, int]:
        """
        Get how many classical items MusicBrainz linking still has to check.

        While any count is above zero, tracks under Other tracks may still move into their works.
        ``albums`` counts the albums still to identify, ``recordings`` the tracks whose
        recording is still to read, ``works`` and ``composers`` the works and composers still
        to look up. All zero means linking is done, or is not running.
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

    def _listed_performers(self, params: dict[str, Any]) -> str:
        """
        Return the condition matching the performers the performers list shows the calling user.

        :param params: Query params dict; the condition's bound params are added to it.
        """
        # only performers of a work are listed, so each row counts at least one recording
        return (
            f"{DB_TABLE_ARTISTS}.item_id IN (SELECT ta.artist_id {PERFORMER_CREDITS} "
            f"AND t.work_id IS NOT NULL AND t.item_id IN ({self._classical_tracks(params)}))"
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
        periods: Sequence[Period] | None,
        work_types: Sequence[WorkType] | None,
        work_type_unknown: bool,
    ) -> list[int]:
        """
        Return the given works whose row matches the search and the filters, in the given order.

        :param work_ids: The listed works to match.
        :param search_term: The normalized search term, or empty to match any work.
        :param periods: Only match works whose composer is of any of these periods.
        :param work_types: Only match works of any of these types, as shown.
        :param work_type_unknown: Also match the works without a type; on its own, only those.
        """
        shown = await self._shown_works(work_ids)
        part_names = await self._part_names(work_ids) if search_term else {}
        return [
            x
            for x in work_ids
            if (not search_term or search_term in shown[x].search_text(part_names[x]))
            and (not periods or shown[x].period in periods)
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
            "c.name AS composer, c.period "
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
                period=Period(row["period"]) if row["period"] is not None else None,
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

    async def _artist_fanarts_and_countries(
        self, artist_ids: Sequence[int]
    ) -> tuple[dict[int, MediaItemImage], dict[int, str]]:
        """
        Return the fanart to show and the country code of the given library artists.

        Both maps are by artist id and leave out the artists without one.

        :param artist_ids: Library ids of the artists.
        """
        if not artist_ids:
            return {}, {}
        user = get_current_user()
        hidden_sources = hidden_music_sources(self.mass, user) if user else set()
        fanarts: dict[int, MediaItemImage] = {}
        countries: dict[int, str] = {}
        for row in await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id, json_extract(metadata, '$.images') AS images, "
            f"{ARTIST_COUNTRY} AS country "
            f"FROM {DB_TABLE_ARTISTS} WHERE item_id IN :artist_ids",
            {"artist_ids": list(artist_ids)},
            limit=0,
        ):
            if row["country"] is not None:
                countries[int(row["item_id"])] = row["country"]
            images = json_loads(row["images"]) if row["images"] else None
            if image := preferred_thumb(images, hidden_sources, ImageType.FANART):
                fanarts[int(row["item_id"])] = MediaItemImage(
                    type=ImageType.FANART,
                    path=image["path"],
                    provider=image["provider"],
                    remotely_accessible=image.get("remotely_accessible", False),
                )
        return fanarts, countries

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
                    *(
                        int(x.artist.item_id)
                        for track in tracks
                        for x in track.credits
                        if x.role in PERFORMING_ROLES
                    ),
                    *(int(x.item_id) for track in tracks for x in track.artists),
                }
            )
        )
        refinements = {x: _role_refinements(names[x], stored[x]) for x in stored}
        for track in tracks:
            track.credits = _shown_credits(track.credits, refinements)
        return refinements

    async def _work_parts(self, listed_ids: Iterable[int] | None) -> dict[int, dict[int, int]]:
        """
        Return the given listed works, or all of them, each with itself and then its parts.

        Each work maps to the work whose recordings its own join: itself, or for a part of a part,
        such as a scene of an act, the part of the listed work it sits under.
        """
        params: dict[str, Any] = {}
        roots = "parent_work_id IS NULL"
        if listed_ids is not None:
            if not (unique_ids := list(set(listed_ids))):
                return {}
            params["listed_ids"] = unique_ids
            roots = "item_id IN :listed_ids"
        # the parts of parts, such as the arias of an act, count as parts of the listed work
        query = (
            f"WITH RECURSIVE tree(item_id, listed_id, part_id) AS (SELECT item_id, item_id, NULL "
            f"FROM {DB_TABLE_WORKS} WHERE {roots} "
            f"UNION SELECT w.item_id, tree.listed_id, COALESCE(tree.part_id, w.item_id) "
            f"FROM {DB_TABLE_WORKS} w JOIN tree ON w.parent_work_id = tree.item_id) "
            "SELECT tree.item_id, tree.listed_id, tree.part_id "
            f"FROM tree JOIN {DB_TABLE_WORKS} w ON w.item_id = tree.item_id "
            "ORDER BY tree.item_id != tree.listed_id, w.name_sort, w.item_id"
        )
        parts: dict[int, dict[int, int]] = defaultdict(dict)
        for row in await self.mass.music.database.get_rows_from_query(query, params, limit=0):
            item_id = int(row["item_id"])
            parts[int(row["listed_id"])][item_id] = int(row["part_id"] or item_id)
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
        rows = await database.get_rows_from_query(
            "SELECT w.item_id, w.name, w.parent_work_id, CASE WHEN json_valid(w.metadata) "
            "THEN json_extract(w.metadata, '$.movements') END AS movements, "
            f"EXISTS (SELECT 1 FROM {DB_TABLE_WORKS} p JOIN {DB_TABLE_WORKS} c "
            "ON c.parent_work_id = p.item_id WHERE p.parent_work_id = w.item_id) AS nested "
            f"FROM {DB_TABLE_WORKS} w WHERE w.item_id IN :work_ids",
            params,
            limit=0,
        )
        unlisted: list[int] = []
        # the parts listing MusicBrainz movements, such as acts, whose groups show their pieces
        listed_parts: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            if stored := _stored_movements(row["movements"]):
                movements[int(row["item_id"])] = _Movements(movements=stored, from_musicbrainz=True)
                if row["parent_work_id"] is not None:
                    listed_parts[int(row["item_id"])] = row
            # a work whose parts have parts, such as acts, lists its parts instead
            elif not row["nested"]:
                unlisted.append(int(row["item_id"]))
        await self._add_listed_parts(movements)
        await self._add_pieces(movements, listed_parts)
        if unlisted:
            movements |= await self._track_movements({"work_ids": unlisted})
        # a work without movements of its own lists its parts instead
        return movements | await self._part_movements(
            [x for x in set(work_ids) if x not in movements]
        )

    async def _add_listed_parts(self, movements: dict[int, _Movements]) -> None:
        """
        Link the parts of works with MusicBrainz movements to the movement each one is.

        A part of a part, such as a scene of an act, is linked to the movement of the act's own
        MusicBrainz movements it is, through its own id or that of its work below the act.
        """
        if not movements:
            return
        # every part with its listed work and the part of that it sits under, a part of a part
        # also with its work right below that part, whose MusicBrainz id is its movement there
        for row in await self.mass.music.database.get_rows_from_query(
            "WITH RECURSIVE tree(item_id, listed_id, part_id, entry_id) AS ("
            f"SELECT item_id, parent_work_id, item_id, NULL FROM {DB_TABLE_WORKS} "
            "WHERE parent_work_id IN :work_ids "
            "UNION SELECT w.item_id, tree.listed_id, tree.part_id, "
            f"COALESCE(tree.entry_id, w.item_id) FROM {DB_TABLE_WORKS} w "
            "JOIN tree ON w.parent_work_id = tree.item_id) "
            "SELECT tree.item_id, tree.listed_id, tree.part_id, tree.entry_id, "
            "e.external_id AS mbid, CASE WHEN json_valid(w.metadata) "
            "THEN json_extract(w.metadata, '$.movements') END AS movements "
            f"FROM tree JOIN {DB_TABLE_WORKS} w ON w.item_id = tree.item_id "
            f"JOIN {DB_TABLE_EXTERNAL_ID_LOOKUP} e ON e.media_type = :work "
            "AND e.item_id = COALESCE(tree.entry_id, tree.item_id) "
            "AND e.external_id_type = :mb_work "
            "ORDER BY tree.entry_id IS NOT NULL",
            {
                "work_ids": list(movements),
                "work": MediaType.WORK.value,
                "mb_work": ExternalID.MB_WORK.value,
            },
            limit=0,
        ):
            listed = movements[int(row["listed_id"])]
            item_id, part_id = int(row["item_id"]), int(row["part_id"])
            mbid = row["mbid"].casefold()
            if row["entry_id"] is not None:
                if entry := next(
                    (
                        x
                        for x in listed.part_movements.get(part_id, [])
                        if x.mbid and x.mbid.casefold() == mbid
                    ),
                    None,
                ):
                    listed.parts[item_id] = listed.parts[part_id]
                    listed.part_movements[item_id] = listed.part_movements[part_id]
                    listed.part_entries[item_id] = entry
                continue
            for index, movement in enumerate(listed.movements):
                if not movement.mbid or movement.mbid.casefold() != mbid:
                    continue
                listed.movements[index] = replace(movement, work_id=str(item_id))
                listed.parts[item_id] = listed.movements[index]
                if stored := _stored_movements(row["movements"]):
                    listed.part_movements[item_id] = stored
                break

    async def _add_pieces(
        self, movements: dict[int, _Movements], parts: Mapping[int, Mapping[str, Any]]
    ) -> None:
        """
        Add the pieces of the groups the given parts list, such as the scenes of a number of an act.

        A group's pieces are the movements MusicBrainz lists for its library work, else those
        stored under it in the part's own movements. They are named as parts of the part.

        :param movements: The movements of the listed works, by work id.
        :param parts: The works rows of the listed works that are parts of another work, by id.
        """
        if not parts:
            return
        ancestors = await self._ancestor_names(list(parts))
        for work_id, row in parts.items():
            listed = movements[work_id]
            names = [row["name"], *ancestors[work_id]]
            stored = _stored_pieces(row["movements"])
            for movement in listed.movements:
                if movement.work_id and (own := listed.part_movements.get(int(movement.work_id))):
                    pieces = [group_piece(movement.number, x) for x in own]
                else:
                    pieces = [x for x in stored if x.parent == movement.number]
                if pieces:
                    listed.pieces[movement.number] = [
                        replace(x, name=_piece_name(x.name, names)) for x in pieces
                    ]

    async def _add_part_lists(self, movements: dict[int, _Movements], work_id: int) -> None:
        """
        Add the movements of each empty part a work lists, as the part's own page lists them.

        :param movements: The movements of the listed works, by work id.
        :param work_id: Library id of the work whose page is shown.
        """
        if not (listed := movements.get(work_id)):
            return
        # a part with pieces or parts listed under it already shows what it holds
        filled = set(listed.pieces) | {x.parent for x in listed.movements}
        numbers = {
            int(x.work_id): x.number
            for x in listed.movements
            if x.work_id and x.parent is None and x.number not in filled
        }
        for part_id, part in (await self._movements(list(numbers))).items():
            # numbers its tracks give without names tell nothing about the part
            if part.from_musicbrainz or all(x.name for x in part.movements):
                listed.part_lists[numbers[part_id]] = part

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
                movements[root_id] = _Movements(
                    movements=list(parts.values()),
                    parts=parts,
                    part_movements={
                        x: stored
                        for x in parts
                        if (stored := _stored_movements(tree[x]["movements"]))
                    },
                )
        return movements

    async def _shown_work_versions(self, works: Iterable[Work | ItemMapping]) -> dict[int, str]:
        """Return the versions the given works show, by work id, for those sharing a name only."""
        if not (work_ids := list({int(x.item_id) for x in works})):
            return {}
        return {
            int(row["item_id"]): row["version"]
            for row in await self.mass.music.database.get_rows_from_query(
                f"SELECT item_id, {SHOWN_WORK_VERSION} AS version FROM {DB_TABLE_WORKS} "
                f"WHERE item_id IN :work_ids AND {SHOWN_WORK_VERSION} != ''",
                {"work_ids": work_ids},
                limit=0,
            )
        }

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
            None if listed_ids is None else [x for ids in parts.values() for x in ids],
            {x: y for ids in parts.values() for x, y in ids.items()},
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
        refinements = await self._show_refined_roles(tracks.values())
        for entry in entries:
            recordings = _performer_recordings(own[int(entry.work.item_id)], tracks, refinements)
            entry.co_performers = _co_performer_names(
                (x for shown, _ in recordings for x in shown), performer_id
            )
            row_tracks = [x for _, value in recordings for x in value]
            entry.liked = all(x.favorite is True for x in row_tracks) if row_tracks else None

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
class _ShownWork:
    """What a works list row shows of a work, as far as the list searches and filters on it."""

    names: list[str]  # the first composer, the title and the first catalogue number
    work_type: WorkType | None = None
    period: Period | None = None
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
    recording_year: int | None = None
    performers: set[int] = field(default_factory=set)
    leaders: set[int] = field(default_factory=set)
    directors: set[int] = field(default_factory=set)
    appearances: list[_Appearance] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        """Return the year the track was recorded, else the earliest year of its albums."""
        return self.recording_year or min(
            (x.year for x in self.appearances if x.year), default=None
        )

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


@dataclass(kw_only=True)
class _Movements:
    """The movements of a work, as its listing and the recordings of it show them."""

    movements: list[WorkMovement] = field(default_factory=list)
    from_musicbrainz: bool = False
    # the movement each part of the work is, by part id, a part of a part counting as the part
    # it sits under
    parts: dict[int, WorkMovement] = field(default_factory=dict)
    # the movements MusicBrainz lists for each of those parts, by part id, a part of a part
    # sharing those of the part it sits under
    part_movements: dict[int, list[WorkMovement]] = field(default_factory=dict)
    # the movement of its part's MusicBrainz movements each part of a part is, such as a scene
    # of an act, by part id
    part_entries: dict[int, WorkMovement] = field(default_factory=dict)
    # the pieces of each group the movements of a part hold, such as the scenes of a number of an
    # act, numbered as group_piece numbers them, by the group's number
    pieces: dict[int, list[WorkMovement]] = field(default_factory=dict)
    # the own movements of each part listed with nothing under it, such as the numbers of an act,
    # as the part's own page lists them, by the part's number
    part_lists: dict[int, _Movements] = field(default_factory=dict)


class _WorkPage:
    """The movements of a work's page, and how the page shows each recording of the work."""

    def __init__(self, movements: list[WorkMovement], is_part: bool) -> None:
        """
        Read the movements of a work as its page lists them.

        :param movements: The movements of the work, each group followed by its pieces.
        :param is_part: Whether the work is a part of another work, whose page keeps the entries
            listed under its parts as its own movements.
        """
        self.movements = movements
        by_number = {x.number: x for x in movements}
        # each movement number followed by those of the movements it sits under
        self.lineage: dict[int, list[int]] = {}
        for movement in by_number.values():
            chain = [movement.number]
            parent = movement.parent
            while parent is not None:
                chain.append(parent)
                parent = by_number[parent].parent if parent in by_number else None
            self.lineage[movement.number] = chain
        # the top-level movements that are works of their own, such as acts, by work id
        self.parts = {x.work_id: x.number for x in movements if x.parent is None and x.work_id}
        part_numbers = set(self.parts.values())
        self.under_parts = [
            x for x in movements if x.parent is not None and self._top(x.number) in part_numbers
        ]
        self.entries = set() if is_part else {x.number for x in self.under_parts}
        # the movements the page lists and groups by, without the entries under its parts
        self.own = [x for x in movements if x.number not in self.entries]

    def listed_movements(self) -> list[WorkMovement]:
        """Return the movements, each with its short name and marked when listed under a part."""
        return [
            replace(
                x,
                under_part=x.number in self.entries,
                short_name=_short_movement_name(x.name),
                excerpt=bool(_EXCERPT_END.search(x.name)),
            )
            for x in self.movements
        ]

    def describe(self, recording: Recording) -> None:
        """Give a recording the part it is of, what it holds and its tracks' movements and names."""
        recording.part = self._part_of(recording)
        recording.coverage = self._coverage(recording, recording.part)
        recording.track_movements = self._track_movements(recording)
        recording.track_names = self._track_names(recording)

    def add_headings(self, recordings: Sequence[Recording], work_id: str) -> None:
        """
        Give each recording the heading of the top-level movement it shows under, such as its act.

        Those of the work itself that hold no movement or reach into every top-level movement come
        first. Those fitting nowhere come last, as do those of the work itself reaching into some
        but not all of its top-level movements. Under a movement, a recording of all of it comes
        first, then by the first movement it holds, then by year.

        :param recordings: The recordings, in the order the page lists them.
        :param work_id: Library id of the work of the page.
        """
        by_work = {x.work_id: x.number for x in self.own if x.work_id}
        order = {x.number: index for index, x in enumerate(self.own, 1)}
        top_level = {x.number for x in self.own if x.parent is None}
        complete: list[Recording] = []
        other: list[Recording] = []
        ranked: dict[int, list[tuple[int, Recording]]] = defaultdict(list)
        for recording in recordings:
            number = by_work.get(recording.work.item_id)
            if number is None and recording.parent_work:
                number = by_work.get(recording.parent_work.item_id)
            own_work = recording.work.item_id == work_id
            tops = {self._top(x) for x in recording.movement_numbers}
            # a recording of the work reaching into several top-level movements is complete when
            # it reaches into all of them
            if number is None and own_work and len(tops) > 1:
                (complete if top_level <= tops else other).append(recording)
                continue
            if number is None and recording.movement_numbers:
                number = recording.movement_numbers[0]
            if number is None:
                (complete if own_work else other).append(recording)
                continue
            top = self._top(number)
            held = recording.movement_numbers
            rank = 0 if top in held else next((x for x in held if x != top), number)
            ranked[top].append((rank, recording))
        for kind, position, listed in (
            (RecordingHeadingKind.COMPLETE, 0, complete),
            (RecordingHeadingKind.OTHER, len(order) + 1, other),
        ):
            for rank, recording in enumerate(listed):
                recording.heading = RecordingHeading(kind=kind, order=position, rank=rank)
        for top, under in ranked.items():
            # a movement the page does not list shows no heading
            if top not in order:
                continue
            under.sort(key=lambda x: (x[0], x[1].year is None, x[1].year or 0))
            for rank, (_, recording) in enumerate(under):
                recording.heading = RecordingHeading(
                    kind=RecordingHeadingKind.MOVEMENT, movement=top, order=order[top], rank=rank
                )

    def _top(self, number: int) -> int:
        """Return the number of the top-level movement a movement sits under, else its own."""
        return self.lineage.get(number, [number])[-1]

    def _part_of(self, recording: Recording) -> int | None:
        """Return the number of the part a recording is of, such as its act."""
        part_numbers = set(self.parts.values())
        if len(covered := [x for x in recording.movement_numbers if x in part_numbers]) == 1:
            return covered[0]
        part = self.parts.get(recording.work.item_id)
        if part is None and recording.parent_work:
            part = self.parts.get(recording.parent_work.item_id)
        return part

    def _coverage(self, recording: Recording, part: int | None) -> RecordingCoverage | None:
        """
        Return what a recording holds, None for all of the work or nothing known.

        :param recording: The recording.
        :param part: The number of the part the recording is of, such as its act.
        """
        if recording.complete:
            return None
        own = {x.number: x for x in self.own}
        covered = set(recording.movement_numbers)
        # a movement under a covered one, such as an aria of a covered act, adds nothing to it
        numbers = [
            x
            for x in dict.fromkeys(recording.movement_numbers)
            if x not in own or own[x].parent is None or own[x].parent not in covered
        ]
        if not numbers:
            return None
        single = own.get(numbers[0]) if len(numbers) == 1 else None
        if single and recording.part_movements:
            # a number's pieces are not numbers of the part
            pieces = [
                x
                for x in self.under_parts
                if x.parent == single.number and part is not None and self._top(x.number) == part
            ]
            return RecordingCoverage(
                kind=RecordingCoverageKind.MOVEMENT_PIECES,
                movement=single.number,
                held=len(recording.part_movements),
                total=len(pieces),
            )
        if single and single.work_id:
            return RecordingCoverage(kind=RecordingCoverageKind.MOVEMENT, movement=single.number)
        return RecordingCoverage(
            kind=RecordingCoverageKind.MOVEMENTS,
            held=len(numbers),
            total=sum(1 for x in self.own if x.parent is None),
        )

    def _track_movements(self, recording: Recording) -> dict[str, list[int]]:
        """Return the movement numbers each track of a recording counts towards, by track uri."""
        numbered = {
            cast("str", x.uri): self.lineage.get(x.movement_number, [x.movement_number])
            for x in recording.tracks
            if x.movement_number is not None
        }
        found = {uri: list(numbers) for uri, numbers in numbered.items()}
        for movement in self.movements:
            # an entry listed right under a part, unlike a piece of one, also counts the tracks of
            # its MusicBrainz work when none is numbered as it, as a track holding several entries
            # is numbered as only one of them
            if (
                not movement.mbid
                or movement.number not in self.entries
                or movement.parent in self.entries
                or any(movement.number in x for x in numbered.values())
            ):
                continue
            for track in recording.tracks:
                if (ExternalID.MB_WORK, movement.mbid) in track.external_ids:
                    found.setdefault(cast("str", track.uri), []).append(movement.number)
        return found

    def _track_names(self, recording: Recording) -> dict[str, str]:
        """Return the name each track of a recording shows, by track uri."""
        # the work's, its parent's and the page's act names, which a track never goes by
        work_names = [
            recording.work.name,
            *([recording.parent_work.name] if recording.parent_work else []),
            *(x.name for x in self.own if x.work_id),
        ]
        # the entries under the part it is of, also on that part's own page
        entries = [
            x
            for x in self.movements
            if x.parent is not None and self._top(x.number) == recording.part
        ]
        return {cast("str", x.uri): _track_name(x, work_names, entries) for x in recording.tracks}


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
    performing_credits = _recording_credits(movement_tracks, refinements)
    covered: set[int] = set()
    own_part = movements.parts.get(int(work.item_id))
    # the own movements of the part a recording of a part is of, such as the numbers of an act
    part_list = movements.part_lists.get(own_part.number) if own_part else None
    part_movements = _part_coverage(
        movement_tracks,
        movements.part_movements.get(int(work.item_id))
        # a part without MusicBrainz movements counts those its own page lists
        or (part_list.movements if part_list and not part_list.parts else []),
        movements.part_entries,
    )
    # the movement each track shows in the part's own movements, as the part's own page shows it
    in_part: dict[int, WorkMovement] = {}
    for track in movement_tracks:
        if part_list and track.work:
            inner_part = part_list.parts.get(int(track.work.item_id))
            if inner := _shown_track_movement(track, part_list, inner_part)[1]:
                in_part[int(track.item_id)] = inner
        # a track of a part joining the work's own recording holds the movement that part is
        track_part = own_part or (
            movements.parts.get(int(track.work.item_id)) if track.work else None
        )
        matched, shown = _shown_track_movement(track, movements, track_part)
        if shown:
            covered.update(x.number for x in matched)
            # the movement fields are only changed on the returned track, never stored
            track.movement_number = shown.number
            track.movement_name = without_movement_number(track.movement_name or shown.name) or None
    # in the order of the movements shown, a track showing none after them
    movement_tracks.sort(
        key=lambda x: (x.movement_number is None, _movement_order(x.movement_number or 0))
    )
    if own_part and in_part:
        act_numbers = {int(x.item_id): x.movement_number for x in movement_tracks}
        for track in movement_tracks:
            if inner := in_part.get(int(track.item_id)):
                track.movement_number = _in_group(own_part.number, inner.number)
        # within a part the tracks follow its own movements, those showing none after them
        movement_tracks.sort(
            key=lambda x: (
                act_numbers[int(x.item_id)] is None,
                _movement_order(act_numbers[int(x.item_id)] or 0),
                int(x.item_id) not in in_part,
                _movement_order(x.movement_number or 0),
            )
        )
    return Recording(
        key=performance.key,
        work=work,
        parent_work=parent_work,
        tracks=movement_tracks,
        credits=performing_credits,
        year=performance.year,
        albums=_distinct_albums(albums[x] for x in performance.album_ids if x in albums),
        duration=sum(x.duration or 0 for x in movement_tracks),
        movement_numbers=sorted(covered),
        complete=(
            {x.number for x in movements.movements} <= covered
            if movements.from_musicbrainz
            else None
        ),
        part_movements=part_movements,
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


def _part_coverage(
    tracks: Sequence[Track],
    part_movements: list[WorkMovement],
    part_entries: Mapping[int, WorkMovement],
) -> list[WorkMovement]:
    """
    Return the movements of a part the tracks of a recording of it cover, in movement order.

    None are returned when the tracks cover all of them, or when a track holds none of them, as
    the recording then counts as the whole part.

    :param tracks: The tracks of the recording, before their movement fields are changed.
    :param part_movements: The movements MusicBrainz lists for the part.
    :param part_entries: The movement of the part each of its own parts is, by work id.
    """
    listed = _Movements(movements=part_movements)
    covered: dict[int, WorkMovement] = {}
    for track in tracks:
        # a track of a part of the part, such as a scene of an act, holds that one movement
        entry = part_entries.get(int(track.work.item_id)) if track.work else None
        if not (matched := [entry] if entry else _held_movements(track, listed, None)):
            return []
        covered |= {x.number: x for x in matched}
    if len(covered) == len(part_movements):
        return []
    return sorted(covered.values(), key=lambda x: x.number)


def _held_movements(
    track: Track, movements: _Movements, own_part: WorkMovement | None
) -> list[WorkMovement]:
    """
    Return the movements a track of a recording holds, none when unknown or ambiguous.

    :param track: The track.
    :param movements: The movements of the requested work.
    :param own_part: The part of the requested work the track's work is, if any.
    """
    if own_part:
        return [own_part]
    mb_works = {
        mbid.casefold() for id_type, mbid in track.external_ids if id_type == ExternalID.MB_WORK
    }
    # a track's MusicBrainz work names its movements, as one track may hold two
    if by_work := [x for x in movements.movements if x.mbid and x.mbid.casefold() in mb_works]:
        return by_work
    # the numbers of listed parts are their positions, not the numbers on their tracks
    if track.movement_number and (movements.from_musicbrainz or not movements.parts):
        if by_number := [x for x in movements.movements if x.number == track.movement_number]:
            return by_number
    name = _movement_name(track)
    by_name = [x for x in movements.movements if _same_movement_name(name, x.name)]
    return by_name if len(by_name) == 1 else []


def _shown_track_movement(
    track: Track, movements: _Movements, own_part: WorkMovement | None
) -> tuple[list[WorkMovement], WorkMovement | None]:
    """
    Return the movements a track of a recording holds and the one it shows, if any.

    :param track: The track, before its movement fields are changed.
    :param movements: The movements of the requested work.
    :param own_part: The part of the requested work the track's work is, if any.
    """
    if not (matched := _held_movements(track, movements, own_part)):
        return [], None
    # a track holding several movements shows the one it is named after
    shown = next(
        (x for x in matched if _same_movement_name(_movement_name(track), x.name)), matched[0]
    )
    return matched, _held_piece(track, shown, movements) or shown


def _held_piece(track: Track, group: WorkMovement, movements: _Movements) -> WorkMovement | None:
    """
    Return the one piece of a group a track holds, None when it holds the whole group or unknown.

    :param track: The track, before its movement fields are changed.
    :param group: The movement of the requested work the track holds.
    :param movements: The movements of the requested work.
    """
    if not (pieces := movements.pieces.get(group.number)):
        return None
    mb_works = {
        mbid.casefold() for id_type, mbid in track.external_ids if id_type == ExternalID.MB_WORK
    }
    if held := [x for x in pieces if x.mbid and x.mbid.casefold() in mb_works]:
        return held[0] if len(held) == 1 else None
    # a track of the group's own work is numbered as the piece it holds
    if track.work and track.work.item_id == group.work_id and track.movement_number:
        number = group.number * PIECE_NUMBERS + track.movement_number
        return next((x for x in pieces if x.number == number), None)
    return None


def _movement_order(number: int) -> tuple[int, ...]:
    """Return the sort key of a movement number, which puts a piece right after its group."""
    pieces: list[int] = []
    while number >= PIECE_NUMBERS:
        number, piece = divmod(number, PIECE_NUMBERS)
        pieces.insert(0, piece)
    return (number, *pieces)


def _in_group(group: int, number: int) -> int:
    """Return the number of an entry of a group's own list in the list holding the group."""
    return int(group * PIECE_NUMBERS ** len(_movement_order(number))) + number


def _movement_name(track: Track) -> str:
    """Return the name of the movement a track performs, from its title when it names none."""
    return track.movement_name or track.name.rpartition(": ")[2]


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


def _short_movement_name(name: str) -> str:
    """
    Return a movement name shortened to its quoted words, led by its score number or scene.

    'XXXIX. Aria (Alto) "Erbarme dich, mein Gott"' becomes "XXXIX. Erbarme dich, mein Gott",
    several quoted parts are joined by a spaced en dash and a name without quotes stays whole.
    The name's " (excerpt)" ending is always left out.
    """
    words = [
        next(x for x in match.groups() if x is not None) for match in _QUOTED_WORDS.finditer(name)
    ]
    if not words:
        return _EXCERPT_END.sub("", name)
    joined = " \u2013 ".join(words)
    number = _LEADING_SCORE_NUMBER.match(name) or _LEADING_SCENE.match(name)
    return f"{number.group(1)}. {joined}" if number else joined


def _track_name(track: Track, work_names: Sequence[str], movements: Sequence[WorkMovement]) -> str:
    """
    Return the name a track shows in its recording's track list, never just a work name.

    That is its movement name, else its own name without a work name in front, else the names
    of the movements its MusicBrainz work ids are, else its own name.

    :param track: The track.
    :param work_names: The names of the works it is of, such as its work and act.
    :param movements: The movements it may hold, such as the arias of its act.
    """
    folded = {x.lower() for x in work_names}
    if track.movement_name and track.movement_name.lower() not in folded:
        return track.movement_name
    prefix = next((x for x in work_names if track.name.lower().startswith(x.lower())), "")
    title = _LEADING_SEPARATORS.sub("", track.name[len(prefix) :]).strip()
    if title and title.lower() not in folded:
        return title
    held = [
        x.name for x in movements if x.mbid and (ExternalID.MB_WORK, x.mbid) in track.external_ids
    ]
    return ", ".join(held) or track.name


def _stored_movements(raw: str | None) -> list[WorkMovement]:
    """Return the movements stored in a work's metadata, in movement order, without pieces."""
    return [x for x in _stored_entries(raw) if x.parent is None]


def _stored_pieces(raw: str | None) -> list[WorkMovement]:
    """Return the pieces stored under the groups a work's movements hold, in movement order."""
    return [x for x in _stored_entries(raw) if x.parent is not None]


def _stored_entries(raw: str | None) -> list[WorkMovement]:
    """Return the movements and pieces stored in a work's metadata, in number order."""
    if not raw:
        return []
    # a later lookup that renamed a movement adds it after the earlier name
    stored = {x.number: x for x in map(WorkMovement.from_dict, json_loads(raw))}
    return sorted(stored.values(), key=lambda x: x.number)


def _with_pieces(movements: _Movements) -> list[WorkMovement]:
    """
    Return the movements of a work, each group followed by its pieces and each part by its own.

    A group whose name only joins the names of its pieces, such as "IVa. ... - IVb. ...", is named
    by the score number they share, such as "IV".
    """
    listed: list[WorkMovement] = []
    for movement in movements.movements:
        pieces = movements.pieces.get(movement.number, [])
        number = _shared_score_number(movement.name, pieces)
        listed.extend((replace(movement, name=number) if number else movement, *pieces))
        if part_list := movements.part_lists.get(movement.number):
            group = movement.number
            listed.extend(
                replace(
                    x,
                    number=_in_group(group, x.number),
                    parent=group if x.parent is None else _in_group(group, x.parent),
                )
                for x in _with_pieces(part_list)
            )
    return listed


def _shared_score_number(name: str, pieces: Sequence[WorkMovement]) -> str | None:
    """Return the score number a group's pieces share when its name only joins their names."""
    # each part of the name and each piece opens with the same score number, such as "IVb. "
    parts = name.split(" - ")
    if not pieces or len(parts) != len(pieces):
        return None
    numbers = {m.group(1) if (m := _SCORE_NUMBER.match(x)) else None for x in parts}
    numbers |= {m.group(1) if (m := _SCORE_NUMBER.match(x.name)) else None for x in pieces}
    return numbers.pop() if len(numbers) == 1 and None not in numbers else None


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
                    name=part_label(row["name"]) or _part_name(row["name"], names),
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


def _piece_name(name: str, names: Sequence[str]) -> str:
    """
    Return the name of a piece as a part of the given works, an excerpt by its own name.

    An excerpt, such as "Excerpt from Die Walküre, WWV 86B: Akt III, Scene I, Walkürenritt",
    is named "Walkürenritt (excerpt)".

    :param name: The piece's MusicBrainz name.
    :param names: The names of the works it is a part of, the nearest first.
    """
    if not name.startswith(_EXCERPT):
        return _part_name(name, names)
    rest = name.removeprefix(_EXCERPT)
    sections = list(_SECTION_LABEL.finditer(rest))
    own = rest[sections[-1].end() :] if sections else _part_name(rest, names)
    return f"{own} (excerpt)"


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


def _completeness(recording: Recording) -> tuple[bool, int]:
    """Return how complete a recording is, as a sort key: complete first, then the longest."""
    return bool(recording.complete), recording.duration


def _recording_credits(
    tracks: Iterable[Track], refinements: dict[int, dict[ArtistRole, ArtistRole]]
) -> list[Credit]:
    """
    Return the performing credits a recording of the given tracks shows, in role order.

    :param tracks: The library tracks of the recording.
    :param refinements: The shown role of each refined stored performing role, by performer.
    """
    performing_credits = _shown_credits(
        (x for track in tracks for x in _track_performing_credits(track)), refinements
    )
    return sorted(performing_credits, key=lambda x: PERFORMING_ROLES.index(x.role))


def _performer_recordings(
    performances: Iterable[_Performance],
    tracks: dict[int, Track],
    refinements: dict[int, dict[ArtistRole, ArtistRole]],
) -> list[tuple[list[Credit], list[Track]]]:
    """
    Return the credits and tracks of performances, in the order the recordings command lists them.

    :param performances: The performances of a listed work and of its parts, per work in the
        order the recordings command lists the works.
    :param tracks: The library tracks of the performances, by track id.
    :param refinements: The shown role of each refined stored performing role, by performer.
    """
    recordings: list[tuple[list[Credit], list[Track]]] = []
    for _, work_performances in groupby(performances, key=lambda x: x.work_id):
        built: list[tuple[_Performance, list[Credit], list[Track]]] = []
        for performance in work_performances:
            performance_tracks = [tracks[x] for x, _ in performance.tracks if x in tracks]
            built.append(
                (
                    performance,
                    _recording_credits(performance_tracks, refinements),
                    performance_tracks,
                )
            )
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

    A performing credit without an instrument gives way to a more specific one of the same
    performer, which is the same role with an instrument or, for the general performer role,
    any other performing role.

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

    def outranked(artist_id: str, role: ArtistRole, instrument: str | None) -> bool:
        return not instrument and any(
            other_id == artist_id
            and (
                (other_role == role and other_instrument)
                or (role == ArtistRole.PERFORMER and other_role != role)
            )
            for other_id, other_role, other_instrument in shown
            if other_role in PERFORMING_ROLES
        )

    return [x for key, x in shown.items() if key[1] not in PERFORMING_ROLES or not outranked(*key)]


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
    work.work_type = work.work_type or title_work_type(work.name)
    return work


def _show_work_version(work: Work | ItemMapping, versions: Mapping[int, str]) -> None:
    """Give a work the version it shows, none unless another library work shares its name."""
    work.version = versions.get(int(work.item_id), "")


def _sort_sql(table: str, field: SortField, direction: SortDirection | None) -> str:
    """Return the ORDER BY clause of a base sort field on the columns of a table."""
    sql = BASE_SORT_FIELD_SQL[field].format(prefix=f"{table}.")
    return f"{sql} {(direction or SortDirection.ASC).value.upper()}"


def _page[T](items: list[T], limit: int, offset: int) -> list[T]:
    """Return one page of the given items, all from the offset without a limit."""
    return items[offset : offset + limit] if limit else items[offset:]
