"""
MusicBrainz work enrichment for the Metadata Controller.

Provides the WorkEnrichmentMixin, mixed into the MetaDataController, which fills the empty
fields of library works in from MusicBrainz: their MusicBrainz id, type, year, language,
catalog numbers, key and composers. A work is found through its own MusicBrainz id or
through the ids on its tracks, which mostly name one of its movements. The composers
MusicBrainz names for a work without any are taken from the library's own artists and are
credited on its tracks without a composer too.

Library works that turn out to be the same MusicBrainz work are merged into the oldest one
with tracks of its own. A work gets linked to the library works MusicBrainz names as its
originals or arrangements, and a part of a larger work, such as an aria, to the nearest work
above it that MusicBrainz gives a type, such as an opera, which is added to the library
without tracks when the library lacks it.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Artist, ItemMapping, UniqueList, Work
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.constants import (
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACKS,
)
from music_assistant.helpers.compare import compare_strings

from .constants import MAX_MUSICBRAINZ_WORK_DEPTH

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence

    from music_assistant_models.enums import Period
    from music_assistant_models.media_items import MediaItemType

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz.models import (
        MusicBrainzArtist,
        MusicBrainzRelation,
        MusicBrainzWork,
    )
    from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider


class WorkEnrichmentMixin:
    """
    MusicBrainz work enrichment for the MetaDataController.

    Expects to be mixed with a class providing ``mass``, ``logger``,
    ``schedule_update_metadata``, ``_musicbrainz_provider`` and ``_composer_period``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger

        def schedule_update_metadata(  # noqa: D102
            self, item: MediaItemType, force_refresh: bool = False
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

        async def _composer_period(self, artist: Artist) -> Period | None: ...

    async def _link_work_to_musicbrainz(self, work: Work) -> None:
        """Fill the empty fields of a library work in from MusicBrainz and mark it looked up."""
        if not (musicbrainz := self._musicbrainz_provider()):
            return
        works = self.mass.music.works
        hierarchy = await self._musicbrainz_work_hierarchy(musicbrainz, work)
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
        if not work.parent_work and (mb_parent := _nearest_typed_parent(hierarchy)):
            parent = await self._get_or_add_parent_work(mb_parent, work)
            work.parent_work = ItemMapping.from_item(parent)
        await works.update_item_in_library(work.item_id, work)
        await self._link_arrangements(int(work.item_id), hierarchy[0])
        identified = False
        if new_composers:
            await self.mass.music.tracks.credit_work_composers(int(work.item_id))
        else:
            identified = await self._identify_sole_composer(work, _composer_relations(hierarchy))
        await self._fill_composer_periods(work.item_id, force_refresh=identified)

    async def _musicbrainz_work_hierarchy(
        self, musicbrainz: MusicbrainzProvider, work: Work
    ) -> list[MusicBrainzWork]:
        """Return the MusicBrainz work a library work is and the works above it, or nothing."""
        if work.mbid:
            mb_work = await musicbrainz.get_work_details(work.mbid)
            return await self._with_parent_works(musicbrainz, mb_work) if mb_work else []
        for work_id in await self._track_work_ids(int(work.item_id)):
            if not (mb_work := await musicbrainz.get_work_details(work_id)):
                continue
            hierarchy = await self._with_parent_works(musicbrainz, mb_work)
            # a track's work is usually a movement of the work the track is linked to,
            # unless a title shows the library work is that work itself or one further up
            level = next(
                (index for index, x in enumerate(hierarchy) if compare_strings(x.title, work.name)),
                1,
            )
            if level < len(hierarchy):
                return hierarchy[level:]
        return []

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

    async def _get_library_composers(
        self, mb_composers: Sequence[MusicBrainzRelation]
    ) -> list[Artist]:
        """Return the library artists of MusicBrainz composers, leaving out those it lacks."""
        return [
            artist
            for relation in mb_composers
            if relation.artist and (artist := await self._get_library_composer(relation.artist))
        ]

    async def _get_library_composer(self, mb_artist: MusicBrainzArtist) -> Artist | None:
        """Return the library artist of a MusicBrainz composer, None when the library has none."""
        artists = self.mass.music.artists
        # the library's own match, by the MusicBrainz id first and else by the exact name
        artist_id = await artists._get_library_item_by_match(
            Artist(
                item_id=mb_artist.id,
                provider="musicbrainz",
                name=mb_artist.name,
                sort_name=mb_artist.sort_name,
                provider_mappings=set(),
                external_ids={(ExternalID.MB_ARTIST, mb_artist.id)},
            )
        )
        return await artists.get_library_item(artist_id) if artist_id else None

    async def _identify_sole_composer(
        self, work: Work, mb_composers: Sequence[MusicBrainzRelation]
    ) -> bool:
        """Give a work's only composer the only MusicBrainz composer's id, return whether it did."""
        if len(work.composers) != 1 or len(mb_composers) != 1 or not mb_composers[0].artist:
            return False
        mb_artist = mb_composers[0].artist
        artists = self.mass.music.artists
        artist = await artists.get_library_item(work.composers[0].item_id)
        if artist.mbid or not _is_named_by(artist.name, mb_artist):
            return False
        artist.mbid = mb_artist.id
        await artists.update_item_in_library(artist.item_id, artist)
        # the oldest of the artists sharing the id survives
        await artists.merge_library_artists_by_mbid(int(artist.item_id), mb_artist.id)
        return True

    async def _fill_composer_periods(self, work_id: str, force_refresh: bool) -> None:
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
                self.schedule_update_metadata(artist, force_refresh=force_refresh)


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


def _nearest_typed_parent(hierarchy: Sequence[MusicBrainzWork]) -> MusicBrainzWork | None:
    """Return the nearest work above a hierarchy's lowest level that MusicBrainz gives a type."""
    return next((x for x in hierarchy[1:] if x.type), None)


def _composer_relations(hierarchy: Sequence[MusicBrainzWork]) -> list[MusicBrainzRelation]:
    """Return the composer relations of the lowest work in a hierarchy that has any."""
    return next((x.composer_relations for x in hierarchy if x.composer_relations), [])


def _is_named_by(name: str, mb_artist: MusicBrainzArtist) -> bool:
    """Return whether every word of a name is in a MusicBrainz artist's name or sort name."""
    words = set(create_safe_string(name).split())
    return bool(words) and any(
        words <= set(create_safe_string(mb_name).split())
        for mb_name in (mb_artist.name, mb_artist.sort_name)
    )
