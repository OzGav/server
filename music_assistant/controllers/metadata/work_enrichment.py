"""
MusicBrainz work enrichment for the Metadata Controller.

Provides the WorkEnrichmentMixin, mixed into the MetaDataController, which fills the empty
fields of library works in from MusicBrainz: their MusicBrainz id, type, year, language,
catalog numbers, key, movements and composers. A work is found through its own MusicBrainz id or
through the ids on its tracks, which mostly name one of its movements. The composers
MusicBrainz names for a work without any are taken from the library's own artists, or added
to the library when MusicBrainz links them to a loaded music provider, and are credited on
its tracks without a composer too. A composer holding another MusicBrainz id than the
MusicBrainz composer of its name, such as a pop singer named after a composer, gives way to
that composer on the work and its tracks.

Library works that turn out to be the same MusicBrainz work are merged into the oldest one
with tracks of its own. A work gets linked to the library works MusicBrainz names as its
originals or arrangements, and a part of a larger work, such as an aria, to the nearest work
above it the library holds, such as an act, else to the nearest one MusicBrainz gives a type,
such as an opera, which is added to the library without tracks when the library lacks it.
The parts of a work found to sit below a newly identified work, such as the arias of an act,
move under it. A part, such as an act, never takes the top of its MusicBrainz hierarchy, and
its parent without an id takes the level above it when their titles match. Otherwise the part
moves under the parent MusicBrainz names, and its former parent leaves the library once empty.
"""

from __future__ import annotations

import logging
from time import time
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from music_assistant_models.enums import ArtistRole, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Artist, ItemMapping, UniqueList, Work
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.constants import (
    DB_TABLE_ARTISTS,
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACK_ARTISTS,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.helpers import provider_mappings_from_urls
from music_assistant.helpers.compare import compare_strings, is_named_by
from music_assistant.helpers.logging import log_once
from music_assistant.helpers.work_titles import parse_catalog_numbers
from music_assistant.providers.musicbrainz.constants import GROUP_ARTIST_TYPES, MUSICBRAINZ_WEB_URL
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider, relation_urls

from .constants import MAX_MUSICBRAINZ_WORK_DEPTH, MUSICBRAINZ_LINK_DOMAINS
from .helpers import fill_artist_from_musicbrainz

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from music_assistant_models.enums import Period
    from music_assistant_models.media_items import Album, MediaItemType

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz.models import (
        MusicBrainzArtist,
        MusicBrainzRelation,
        MusicBrainzWork,
    )


class WorkEnrichmentMixin:
    """
    MusicBrainz work enrichment for the MetaDataController.

    Expects to be mixed with a class providing ``mass``, ``logger``, ``_logged_hints``,
    ``schedule_update_metadata``, ``_musicbrainz_provider``, ``_composer_period`` and
    ``_name_after_musicbrainz``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _logged_hints: set[Hashable]

        def schedule_update_metadata(  # noqa: D102
            self, item: MediaItemType, force_refresh: bool = False
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

        async def _composer_period(self, artist: Artist) -> Period | None: ...

        async def _name_after_musicbrainz(
            self, artist: Artist, details: MusicBrainzArtist, composer_only: bool = False
        ) -> bool: ...

    async def _link_work_to_musicbrainz(self, work: Work) -> None:
        """Fill the empty fields of a library work in from MusicBrainz and mark it looked up."""
        if not (musicbrainz := self._musicbrainz_provider()):
            return
        works = self.mass.music.works
        hierarchy = await self._musicbrainz_work_hierarchy(musicbrainz, work)
        newly_identified = bool(hierarchy) and not work.mbid
        work.metadata.last_musicbrainz_lookup = int(time())
        if not hierarchy:
            await works.update_item_in_library(work.item_id, work)
            return
        # the caller reads the outcome off the work it passed in, which a merge may remove
        _fill_work_from_musicbrainz(work, hierarchy)
        if survivor := await self._merge_duplicate_works(work, hierarchy[0].id):
            work = survivor
            _fill_work_from_musicbrainz(work, hierarchy)
        new_composers: list[Artist] = []
        if not work.composers:
            new_composers = await self._get_library_composers(_composer_relations(hierarchy))
            work.composers = UniqueList(new_composers)
        old_parent = (
            await works.get_library_item(work.parent_work.item_id) if work.parent_work else None
        )
        moved_from: Work | None = None
        if (not old_parent or _gives_way_to_musicbrainz(old_parent, hierarchy)) and (
            mb_parent := await self._parent_level(hierarchy)
        ):
            parent = await self._get_or_add_parent_work(mb_parent, work)
            if old_parent and old_parent.item_id != parent.item_id:
                moved_from = old_parent
            work.parent_work = ItemMapping.from_item(parent)
        await works.update_item_in_library(work.item_id, work)
        if moved_from and work.parent_work:
            # a library update keeps the parent a work has, so the move is written here
            await self.mass.music.database.execute_write(
                f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = :parent_id WHERE item_id = :work_id",
                {"parent_id": int(work.parent_work.item_id), "work_id": int(work.item_id)},
            )
        await self._link_arrangements(int(work.item_id), hierarchy[0])
        if work.parent_work:
            await self._identify_parent_by_title(work.parent_work, hierarchy)
        if moved_from:
            await self._remove_if_empty(moved_from.item_id)
        if newly_identified:
            await self._adopt_parts(int(work.item_id), hierarchy)
        mb_composers = _composer_relations(hierarchy)
        if new_composers:
            await self.mass.music.tracks.credit_work_composers(int(work.item_id))
        elif not await self._replace_namesake_composers(work, mb_composers):
            await self._identify_sole_composer(musicbrainz, work, mb_composers)
        await self._fill_composer_periods(work.item_id)

    async def _musicbrainz_work_hierarchy(
        self, musicbrainz: MusicbrainzProvider, work: Work
    ) -> list[MusicBrainzWork]:
        """Return the MusicBrainz work a library work is and the works above it, or nothing."""
        if work.mbid:
            mb_work = await musicbrainz.get_work_details(work.mbid)
            return await self._with_parent_works(musicbrainz, mb_work) if mb_work else []
        work_ids = await self._track_work_ids(int(work.item_id))
        for work_id in work_ids:
            if not (mb_work := await musicbrainz.get_work_details(work_id)):
                continue
            hierarchy = await self._with_parent_works(musicbrainz, mb_work)
            # a track's work is usually a movement of the work the track is linked to, the
            # lowest one above it listing every track's work as its part, unless a title shows
            # the library work is that work itself or one further up;
            # a work without parts, which MusicBrainz may title in another language, is
            # recognised by its catalogue numbers, or by several tracks all performing it
            single = len(hierarchy) == 1 and (
                _shares_catalog_numbers(hierarchy[0], work)
                or (len(work_ids) == 1 and await self._tracks_all_linked(int(work.item_id)))
            )
            level = next(
                (index for index, x in enumerate(hierarchy) if compare_strings(x.title, work.name)),
                0 if single else _level_holding_all(hierarchy, work_ids),
            )
            if level < len(hierarchy) and await self._fits_below_parent(work, hierarchy, level):
                return hierarchy[level:]
        return []

    async def _fits_below_parent(
        self, work: Work, hierarchy: Sequence[MusicBrainzWork], level: int
    ) -> bool:
        """
        Return whether a library work may be the given level of a MusicBrainz hierarchy.

        A work with a parent, such as an act, is below the top of the hierarchy and below the
        level its parent holds.

        :param work: The library work.
        :param hierarchy: A MusicBrainz work followed by the works above it.
        :param level: The index of the level the work would be.
        """
        if not work.parent_work:
            return True
        if level >= len(hierarchy) - 1:
            return False
        parent = await self.mass.music.works.get_library_item(work.parent_work.item_id)
        parent_level = next(
            (index for index, x in enumerate(hierarchy) if parent.mbid and x.id == parent.mbid),
            None,
        )
        return parent_level is None or level < parent_level

    async def _parent_level(self, hierarchy: Sequence[MusicBrainzWork]) -> MusicBrainzWork | None:
        """Return the level above a hierarchy's lowest one that is to be the work's parent."""
        typed = _nearest_typed_parent(hierarchy)
        # a level the library holds, such as an act, up to the typed one, such as the opera
        above = hierarchy[1 : hierarchy.index(typed) + 1] if typed else hierarchy[1:]
        works = self.mass.music.works
        for mb_work in above:
            if await works.get_library_item_by_external_id(mb_work.id, ExternalID.MB_WORK):
                return mb_work
        return typed

    async def _identify_parent_by_title(
        self, parent: ItemMapping, hierarchy: Sequence[MusicBrainzWork]
    ) -> None:
        """Give a work's parent without an id the level above the work, if their titles match."""
        if len(hierarchy) < 2:
            return
        works = self.mass.music.works
        library_parent = await works.get_library_item(parent.item_id)
        if library_parent.mbid or not compare_strings(library_parent.name, hierarchy[1].title):
            return
        library_parent.mbid = hierarchy[1].id
        await works.update_item_in_library(library_parent.item_id, library_parent)
        await self._link_work_to_musicbrainz(library_parent)

    async def _adopt_parts(self, work_id: int, hierarchy: Sequence[MusicBrainzWork]) -> None:
        """
        Move the library works MusicBrainz has as parts of a work under it.

        Considered are the parts of the work's parent and of the library work holding the
        level above it, such as the arias next to an act in its opera.

        :param work_id: Library id of the work.
        :param hierarchy: The work's MusicBrainz work followed by the works above it.
        """
        works = self.mass.music.works
        work = await works.get_library_item(work_id)
        parent_ids = {int(work.parent_work.item_id)} if work.parent_work else set()
        if len(hierarchy) > 1 and (
            holder := await works.get_library_item_by_external_id(
                hierarchy[1].id, ExternalID.MB_WORK
            )
        ):
            parent_ids.add(int(holder.item_id))
        if not parent_ids or not (musicbrainz := self._musicbrainz_provider()):
            return
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id FROM {DB_TABLE_WORKS} "
            "WHERE parent_work_id IN :parent_ids AND item_id != :work_id",
            {"parent_ids": sorted(parent_ids), "work_id": work_id},
            limit=0,
        )
        for row in rows:
            sibling = await works.get_library_item(row["item_id"])
            if not sibling.mbid or not (
                mb_sibling := await musicbrainz.get_work_details(sibling.mbid)
            ):
                continue
            mb_parent = mb_sibling.parent_relation
            if mb_parent and mb_parent.work and mb_parent.work.id == hierarchy[0].id:
                await self.mass.music.database.execute_write(
                    f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = :work_id "
                    "WHERE item_id = :sibling_id",
                    {"work_id": work_id, "sibling_id": int(sibling.item_id)},
                )

    async def _remove_if_empty(self, work_id: str) -> None:
        """Remove a library work without a MusicBrainz id that has no tracks, parts or arrangements."""
        works = self.mass.music.works
        work = await works.get_library_item(work_id)
        if (
            work.mbid
            or work.parts
            or work.arrangements
            or work.arrangement_of
            or await self.mass.music.database.get_row(DB_TABLE_TRACKS, {"work_id": int(work_id)})
        ):
            return
        await works.remove_item_from_library(work_id)

    async def _with_parent_works(
        self, musicbrainz: MusicbrainzProvider, mb_work: MusicBrainzWork
    ) -> list[MusicBrainzWork]:
        """Return a MusicBrainz work followed by the works it is a part of, top-level last."""
        hierarchy = [mb_work]
        while (
            len(hierarchy) <= MAX_MUSICBRAINZ_WORK_DEPTH
            and (parent := hierarchy[-1].parent_relation)
            and parent.work
        ):
            if not (parent_work := await musicbrainz.get_work_details(parent.work.id)):
                break
            hierarchy.append(parent_work)
        return hierarchy

    async def _merge_duplicate_works(self, work: Work, mbid: str) -> Work | None:
        """Merge the library works that are a given MusicBrainz work, return the survivor if any."""
        works = self.mass.music.works
        holders = await works.get_library_items_by_external_id(mbid, ExternalID.MB_WORK, limit=None)
        if all(x.item_id == work.item_id for x in holders):
            return None
        survivor = await works.get_library_item(
            await works.merge_library_works_by_mbid(int(work.item_id), mbid)
        )
        survivor.metadata.last_musicbrainz_lookup = work.metadata.last_musicbrainz_lookup
        return survivor

    async def _get_or_add_parent_work(self, mb_work: MusicBrainzWork, child: Work) -> Work:
        """Return the library work of a child's parent MusicBrainz work, adding it if needed."""
        works = self.mass.music.works
        if holder := await works.get_library_item_by_external_id(mb_work.id, ExternalID.MB_WORK):
            return holder
        parent = Work(
            item_id=mb_work.id,
            provider="musicbrainz",
            name=mb_work.title,
            provider_mappings=set(),
            # filled in right here, so the link run need not look it up again
            metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time())),
            composers=UniqueList(
                await self._get_library_composers(mb_work.composer_relations) or child.composers
            ),
        )
        _fill_work_from_musicbrainz(parent, [mb_work])
        # the library may hold the parent without its id, added from the title of a track
        match = await works.get_library_item_by_match(parent)
        if match and match.item_id != child.item_id:
            added = await works.update_item_in_library(match.item_id, parent)
        else:
            added = await works.create_library_item(parent)
        await self._link_arrangements(int(added.item_id), mb_work)
        return added

    async def _link_arrangements(self, work_id: int, mb_work: MusicBrainzWork) -> None:
        """Link a work to the library works MusicBrainz names as its originals or arrangements."""
        works = self.mass.music.works
        for relation in mb_work.arrangement_of_relations:
            if original_id := await self._get_library_work_id(relation):
                await works.add_arrangement(work_id, original_id)
        for relation in mb_work.arrangement_relations:
            if arrangement_id := await self._get_library_work_id(relation):
                await works.add_arrangement(arrangement_id, work_id)

    async def _get_library_work_id(self, relation: MusicBrainzRelation) -> int | None:
        """Return the id of the library work a MusicBrainz work relation points at, if any."""
        if not relation.work:
            return None
        work = await self.mass.music.works.get_library_item_by_external_id(
            relation.work.id, ExternalID.MB_WORK
        )
        return int(work.item_id) if work else None

    async def _track_work_ids(self, work_id: int) -> list[str]:
        """Return the MusicBrainz work ids the tracks of a library work carry."""
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT {DB_TABLE_EXTERNAL_ID_LOOKUP}.external_id "
            f"FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} JOIN {DB_TABLE_TRACKS} "
            f"ON {DB_TABLE_TRACKS}.item_id = {DB_TABLE_EXTERNAL_ID_LOOKUP}.item_id "
            f"WHERE {DB_TABLE_EXTERNAL_ID_LOOKUP}.media_type = :media_type "
            f"AND {DB_TABLE_EXTERNAL_ID_LOOKUP}.external_id_type = :id_type "
            f"AND {DB_TABLE_TRACKS}.work_id = :work_id "
            f"ORDER BY {DB_TABLE_EXTERNAL_ID_LOOKUP}.external_id",
            {
                "media_type": MediaType.TRACK.value,
                "id_type": ExternalID.MB_WORK.value,
                "work_id": work_id,
            },
            limit=0,
        )
        return [row["external_id"] for row in rows]

    async def _tracks_all_linked(self, work_id: int) -> bool:
        """Return whether a library work has several tracks, each with a MusicBrainz work id."""
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT COUNT(DISTINCT {DB_TABLE_TRACKS}.item_id) AS tracks, "
            f"COUNT(DISTINCT {DB_TABLE_EXTERNAL_ID_LOOKUP}.item_id) AS linked "
            f"FROM {DB_TABLE_TRACKS} LEFT JOIN {DB_TABLE_EXTERNAL_ID_LOOKUP} "
            f"ON {DB_TABLE_EXTERNAL_ID_LOOKUP}.item_id = {DB_TABLE_TRACKS}.item_id "
            f"AND {DB_TABLE_EXTERNAL_ID_LOOKUP}.media_type = :media_type "
            f"AND {DB_TABLE_EXTERNAL_ID_LOOKUP}.external_id_type = :id_type "
            f"WHERE {DB_TABLE_TRACKS}.work_id = :work_id",
            {
                "media_type": MediaType.TRACK.value,
                "id_type": ExternalID.MB_WORK.value,
                "work_id": work_id,
            },
        )
        tracks, linked = int(rows[0]["tracks"]), int(rows[0]["linked"])
        return tracks >= 2 and linked == tracks

    async def _get_library_composers(
        self, mb_composers: Sequence[MusicBrainzRelation]
    ) -> list[Artist]:
        """Return the library artists of MusicBrainz composers, leaving out those it cannot add."""
        return [
            artist
            for relation in mb_composers
            if relation.artist and (artist := await self._get_library_composer(relation.artist))
        ]

    async def _get_library_composer(self, mb_artist: MusicBrainzArtist) -> Artist | None:
        """
        Return the library artist of a MusicBrainz composer, adding it when the library lacks it.

        A composer is only added when MusicBrainz links it to a loaded music provider.

        :param mb_artist: The composer as MusicBrainz names it.
        """
        if artist := await self._match_library_artist(mb_artist):
            return artist
        if artist := await self._add_musicbrainz_artist(mb_artist):
            await self._merge_composer_by_aliases(mb_artist.id)
        return artist

    async def _match_library_artist(self, mb_artist: MusicBrainzArtist) -> Artist | None:
        """Return the library artist of a MusicBrainz artist, None when the library lacks it."""
        # the library's own match, by the MusicBrainz id first and else by the exact name
        return await self.mass.music.artists.get_library_item_by_match(
            Artist(
                item_id=mb_artist.id,
                provider="musicbrainz",
                name=mb_artist.name,
                sort_name=mb_artist.sort_name,
                provider_mappings=set(),
                external_ids={(ExternalID.MB_ARTIST, mb_artist.id)},
            )
        )

    async def _add_musicbrainz_artist(
        self, mb_artist: MusicBrainzArtist, album: Album | None = None, fresh: bool = False
    ) -> Artist | None:
        """
        Add a MusicBrainz artist to the library on the music providers it links to, if any.

        :param mb_artist: The MusicBrainz artist.
        :param album: The album the artist performs on, named when the artist is not added.
        :param fresh: Read the artist's links from MusicBrainz again instead of the cache.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return None
        try:
            async with self.mass.cache.handle_refresh(fresh):
                details = await musicbrainz.get_artist_details(mb_artist.id)
        except InvalidDataError:
            return None
        mappings = await provider_mappings_from_urls(
            self.mass, relation_urls(details.relations), MediaType.ARTIST, set()
        )
        if not mappings:
            providers = [
                instances[0].manifest.name
                for domain in MUSICBRAINZ_LINK_DOMAINS
                if (
                    instances := self.mass.music.get_provider_instances(
                        domain, return_unavailable=True
                    )
                )
            ]
            if providers:
                log_once(
                    self.logger,
                    self._logged_hints,
                    ("artist", details.id),
                    logging.INFO,
                    'Artist "%s" has no link to %s on MusicBrainz, so it is not added%s; '
                    "add one at %s/artist/%s%s",
                    details.name,
                    ", ".join(providers),
                    f' to album "{album.name}"' if album else "",
                    MUSICBRAINZ_WEB_URL,
                    details.id,
                    ", then use Update metadata on the album" if album else "",
                )
            return None
        return await self.mass.music.artists.add_item_to_library(
            Artist(
                item_id=mappings[0].item_id,
                provider=mappings[0].provider_instance,
                name=details.name,
                sort_name=details.sort_name,
                provider_mappings=set(mappings),
                external_ids={(ExternalID.MB_ARTIST, details.id)},
            )
        )

    async def _replace_namesake_composers(
        self, work: Work, mb_composers: Sequence[MusicBrainzRelation]
    ) -> bool:
        """
        Credit MusicBrainz's composers in place of their namesakes holding another MusicBrainz id.

        The composer takes the namesake's composer credit on the work and all of the namesake's
        credits on its tracks, and a namesake left without a composer credit loses its period.
        Return whether any composer was replaced.

        :param work: The library work.
        :param mb_composers: The composers MusicBrainz names for the work.
        """
        artists = self.mass.music.artists
        replaced = False
        for composer in work.composers:
            namesake = await artists.get_library_item(composer.item_id)
            if not namesake.mbid:
                continue
            for relation in mb_composers:
                mb_artist = relation.artist
                if (
                    not mb_artist
                    or mb_artist.id == namesake.mbid
                    or not compare_strings(namesake.name, mb_artist.name)
                ):
                    continue
                self.logger.debug(
                    'Work "%s" is by %s/artist/%s, but library artist "%s" holds %s/artist/%s',
                    work.name,
                    MUSICBRAINZ_WEB_URL,
                    mb_artist.id,
                    namesake.name,
                    MUSICBRAINZ_WEB_URL,
                    namesake.mbid,
                )
                artist = await self._get_library_composer(mb_artist)
                if artist and artist.item_id != namesake.item_id:
                    await self._replace_work_composer(
                        int(work.item_id), int(namesake.item_id), int(artist.item_id)
                    )
                    replaced = True
                break
        return replaced

    async def _replace_work_composer(self, work_id: int, old_id: int, new_id: int) -> None:
        """Credit one artist in place of another as composer of a work and on its tracks."""
        database = self.mass.music.database
        params = {
            "work_id": work_id,
            "old_id": old_id,
            "new_id": new_id,
            "composer": ArtistRole.COMPOSER.value,
        }
        tracks = f"SELECT item_id FROM {DB_TABLE_TRACKS} WHERE work_id = :work_id"
        # on the tracks the namesake is the composer under any role, such as the main artist
        for table, owner in (
            (DB_TABLE_WORK_ARTISTS, "work_id = :work_id AND role = :composer"),
            (DB_TABLE_TRACK_ARTISTS, f"track_id IN ({tracks})"),
        ):
            where = f"WHERE {owner} AND artist_id = :old_id"
            # a credit the new artist already holds stays, so the old one is deleted instead
            await database.execute_write(
                f"UPDATE OR IGNORE {table} SET artist_id = :new_id {where}", params
            )
            await database.execute_write(f"DELETE FROM {table} {where}", params)
        classification = self.mass.music.classification
        await classification.update_item(MediaType.WORK, work_id)
        await classification.update_item(MediaType.ARTIST, old_id)
        await database.execute_write(
            f"UPDATE {DB_TABLE_ARTISTS} SET period = NULL WHERE item_id = :old_id "
            f"AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_WORK_ARTISTS} "
            "WHERE artist_id = :old_id AND role = :composer) "
            f"AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_ARTISTS} "
            "WHERE artist_id = :old_id AND role = :composer)",
            params,
        )

    async def _identify_sole_composer(
        self,
        musicbrainz: MusicbrainzProvider,
        work: Work,
        mb_composers: Sequence[MusicBrainzRelation],
    ) -> None:
        """Give a work's only composer the only MusicBrainz composer's id and life dates."""
        if len(work.composers) != 1 or len(mb_composers) != 1 or not mb_composers[0].artist:
            return
        mb_artist = mb_composers[0].artist
        artists = self.mass.music.artists
        artist = await artists.get_library_item(work.composers[0].item_id)
        if artist.mbid or not is_named_by(artist.name, mb_artist):
            return
        # a name such as "Bach" may belong to another confirmed composer of the family
        if any(
            is_named_by(artist.name, x)
            for x in await artists.get_library_composers(with_mbid=True)
            if x.mbid != mb_artist.id
        ):
            return
        try:
            details = await musicbrainz.get_artist_details(mb_artist.id)
        except InvalidDataError:
            details = None
        artist.mbid = mb_artist.id
        if details:
            fill_artist_from_musicbrainz(artist, details)
        await artists.update_item_in_library(artist.item_id, artist)
        # the oldest of the artists sharing the id survives
        await artists.merge_library_artists_by_mbid(int(artist.item_id), mb_artist.id)
        await self._merge_composer_by_aliases(mb_artist.id, details)

    async def _merge_composer_by_aliases(
        self, mbid: str, details: MusicBrainzArtist | None = None
    ) -> None:
        """
        Merge the library composers carrying a name of a confirmed composer into it.

        A library composer without a MusicBrainz id is merged only when its whole name is one
        of the composer's MusicBrainz names (never a search hint), no other confirmed composer
        in the library carries it, and no other MusicBrainz person carries it either. Artists
        already holding the composer's id are always merged. The composer then takes the name
        MusicBrainz gives it, in Latin script.

        :param mbid: The MusicBrainz artist id of the composer.
        :param details: The composer's MusicBrainz artist with its aliases, fetched when omitted.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return
        if not details:
            try:
                details = await musicbrainz.get_artist_details(mbid)
            except InvalidDataError:
                return
        alias_names = _alias_names(details)
        artists = self.mass.music.artists
        holders = await artists.get_library_items_by_external_id(
            mbid, ExternalID.MB_ARTIST, limit=None
        )
        approved: list[Artist] = []
        others: list[Artist] | None = None
        for candidate in await artists.get_library_composers(with_mbid=False):
            if not await self._carries_only_composer_name(
                musicbrainz, candidate.name, alias_names, mbid
            ):
                continue
            if others is None:
                others = [
                    x for x in await artists.get_library_composers(with_mbid=True) if x.mbid != mbid
                ]
            # any other confirmed composer sharing a word of the name, such as a family name,
            # leaves it apart
            if any(is_named_by(candidate.name, x) for x in others):
                self.logger.debug('Not joining "%s" to %s', candidate.name, details.name)
                continue
            self.logger.debug('Joining "%s" to %s', candidate.name, details.name)
            candidate.mbid = mbid
            await artists.update_item_in_library(candidate.item_id, candidate)
            approved.append(candidate)
        if approved or len(holders) > 1:
            self.logger.debug("Joining the artists holding the id of %s", details.name)
            # the oldest of the artists sharing the id survives
            await artists.merge_library_artists_by_mbid(int((approved or holders)[0].item_id), mbid)
        for holder in await artists.get_library_items_by_external_id(
            mbid, ExternalID.MB_ARTIST, limit=None
        ):
            await self._name_after_musicbrainz(holder, details)
            # tracks saved before the composer was confirmed may still credit it joined with others
            await self.mass.music.tracks.split_joined_composer_credits(holder, alias_names)

    async def _join_composer_by_name(
        self, musicbrainz: MusicbrainzProvider, artist: Artist
    ) -> None:
        """
        Join a composer its own links did not identify to the confirmed composer of its name.

        Only the one MusicBrainz person carrying the exact name qualifies, when a library
        composer holds that person's id, under the rules of the alias join. Without a join,
        where to add the composer's streaming link on MusicBrainz is logged.

        :param musicbrainz: The MusicBrainz provider.
        :param artist: The library composer without a MusicBrainz id.
        """
        # a composer is a person, so a band of the same name is never them
        persons = [
            x
            for x in await musicbrainz.get_artists_by_name(artist.name)
            if x.type not in GROUP_ARTIST_TYPES
        ]
        person = persons[0] if len(persons) == 1 else None
        artists = self.mass.music.artists
        if person and await artists.get_library_items_by_external_id(
            person.id, ExternalID.MB_ARTIST, limit=None
        ):
            await self._merge_composer_by_aliases(person.id)
            try:
                if (await artists.get_library_item(artist.item_id)).mbid:
                    return
            except MediaNotFoundError:
                # merged into the composer holding the id
                return
            person = None
        if not (urls := MusicbrainzProvider.reverse_lookup_urls(artist)):
            return
        link = (
            f"{MUSICBRAINZ_WEB_URL}/artist/{person.id}"
            if person
            else f"{MUSICBRAINZ_WEB_URL}/search?{urlencode({'type': 'artist', 'query': artist.name})}"
        )
        log_once(
            self.logger,
            self._logged_hints,
            ("artist", artist.uri),
            logging.INFO,
            'Composer "%s" is not on MusicBrainz through %s, add %s to their MusicBrainz page: %s',
            artist.name,
            ", ".join(urls),
            urls[0],
            link,
        )

    async def _carries_only_composer_name(
        self, musicbrainz: MusicbrainzProvider, name: str, alias_names: set[str], mbid: str
    ) -> bool:
        """Return whether a name is one of a composer's names and no other MusicBrainz person's."""
        if not any(compare_strings(name, alias) for alias in alias_names):
            return False
        # a composer is a person, so a band of the same name is never them
        namesakes = [
            x
            for x in await musicbrainz.get_artists_by_name(name)
            if x.type not in GROUP_ARTIST_TYPES
        ]
        return bool(namesakes) and all(x.id == mbid for x in namesakes)

    async def _fill_composer_periods(self, work_id: str) -> None:
        """Give the composers of a work without a period one, now or after a metadata refresh."""
        artists = self.mass.music.artists
        work = await self.mass.music.works.get_library_item(work_id)
        for composer in work.composers:
            artist = await artists.get_library_item(composer.item_id)
            if artist.period:
                continue
            if artist.metadata.life_span:
                if period := await self._composer_period(artist):
                    artist.period = period
                    await artists.update_item_in_library(artist.item_id, artist)
            elif artist.mbid:
                # the refresh fills in the life dates the composer's period is derived from
                self.schedule_update_metadata(artist)


def _fill_work_from_musicbrainz(work: Work, hierarchy: Sequence[MusicBrainzWork]) -> None:
    """Fill the empty fields of a work in from its MusicBrainz work and the works above it."""
    mb_work = hierarchy[0]
    if not work.mbid:
        work.mbid = mb_work.id
    if work.work_type is None:
        # a part, such as an aria, is typed by the work it belongs to, such as an opera
        work.work_type = next((x.work_type for x in reversed(hierarchy) if x.work_type), None)
    work.composition_year = work.composition_year or mb_work.composition_year
    work.language = work.language or mb_work.language
    work.musical_key = work.musical_key or mb_work.musical_key
    if not work.catalog_numbers:
        work.catalog_numbers = mb_work.catalog_numbers
    if not work.metadata.movements:
        work.metadata.movements = mb_work.movements or None


def _gives_way_to_musicbrainz(parent: Work, hierarchy: Sequence[MusicBrainzWork]) -> bool:
    """Return whether a work's parent is to make way for the parent MusicBrainz names."""
    # a parent titled as the level above the work is identified as that level instead
    return not parent.mbid and not (
        len(hierarchy) > 1 and compare_strings(parent.name, hierarchy[1].title)
    )


def _nearest_typed_parent(hierarchy: Sequence[MusicBrainzWork]) -> MusicBrainzWork | None:
    """Return the nearest work above a hierarchy's lowest level that MusicBrainz gives a type."""
    return next((x for x in hierarchy[1:] if x.type), None)


def _level_holding_all(hierarchy: Sequence[MusicBrainzWork], work_ids: Sequence[str]) -> int:
    """
    Return the lowest level above a hierarchy's lowest one that is or holds all given works.

    The hierarchy's length is returned when no level does.
    """
    for index, mb_work in enumerate(hierarchy[1:], 1):
        parts = {
            x.work.id
            for x in mb_work.relations
            if x.type == "parts" and x.direction == "forward" and x.work
        }
        if set(work_ids) <= {mb_work.id, *parts}:
            return index
    return len(hierarchy)


def _composer_relations(hierarchy: Sequence[MusicBrainzWork]) -> list[MusicBrainzRelation]:
    """Return the composer relations of the lowest work in a hierarchy that has any."""
    return next((x.composer_relations for x in hierarchy if x.composer_relations), [])


def _shares_catalog_numbers(mb_work: MusicBrainzWork, work: Work) -> bool:
    """Return whether a MusicBrainz work carries the catalogue numbers of a library work."""
    if not work.catalog_numbers:
        return False
    mb_numbers = [*mb_work.catalog_numbers, *parse_catalog_numbers(mb_work.title)]
    return {create_safe_string(x, True, True) for x in work.catalog_numbers} <= {
        create_safe_string(x, True, True) for x in mb_numbers
    }


def _alias_names(mb_artist: MusicBrainzArtist) -> set[str]:
    """Return a MusicBrainz artist's main name and its aliases that are not search hints."""
    aliases = mb_artist.aliases or []
    return {mb_artist.name, *(x.name for x in aliases if (x.type or "").lower() != "search hint")}
