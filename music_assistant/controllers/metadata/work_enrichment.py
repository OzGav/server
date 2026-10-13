"""
MusicBrainz work enrichment for the Metadata Controller.

Provides the WorkEnrichmentMixin, mixed into the MetaDataController, which fills the empty
fields of library works in from MusicBrainz: their type, year, language, catalog numbers, key
and composers. Every library work is a MusicBrainz work, found through its own
MusicBrainz id. The composers MusicBrainz names for a work without any are taken from the
library's own artists, and are credited on its tracks without a composer too.

Library works that turn out to be the same MusicBrainz work are merged into the one holding
the most tracks. A work gets linked to the library works MusicBrainz names as its
originals or arrangements.

A work looked up for the first time takes its MusicBrainz name, in the user's language.
A work refreshed by hand is looked up again as on its first lookup, taking MusicBrainz's
current name, version, type, year, language, key and catalog numbers; a work MusicBrainz
merged into another takes the other's id.
"""

from __future__ import annotations

import logging
from time import time
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Artist, UniqueList, Work

from music_assistant.constants import (
    DB_TABLE_WORK_ARRANGEMENTS,
    DB_TABLE_WORKS,
)
from music_assistant.providers.musicbrainz.api_client import fresh_reads
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider

from .constants import MAX_MUSICBRAINZ_WORK_DEPTH
from .helpers import (
    fill_artist_from_musicbrainz,
    musicbrainz_work_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from music_assistant_models.media_items import Album

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz.models import (
        MusicBrainzArtist,
        MusicBrainzRelation,
        MusicBrainzWork,
    )


class WorkEnrichmentMixin:
    """
    MusicBrainz work enrichment for the MetaDataController.

    Expects to be mixed with a class providing ``mass``, ``logger``, ``preferred_language``,
    ``set_composer_periods``, ``_musicbrainz_provider`` and ``_follow_merged_id``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger

        @property
        def preferred_language(self) -> str: ...  # noqa: D102

        async def set_composer_periods(  # noqa: D102
            self, artist_ids: Iterable[int] | None = None
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

        async def _follow_merged_id(self, item: Album | Artist | Work, mbid: str) -> None: ...

    async def refresh_work_tree(self, work: Work) -> Work:
        """
        Look a library work and all the works of its tree up on MusicBrainz again, read fresh.

        The tree is the work's top work in the library and every library work under it. Each
        takes MusicBrainz's current data as on its first lookup.

        :param work: The library work, at any level of its tree.
        :return: The work as it is in the library afterwards, or the work given when it left.
        """
        works = self.mass.music.works
        top, seen = work, {work.item_id}
        while top.parent_work and top.parent_work.item_id not in seen:
            seen.add(top.parent_work.item_id)
            top = await works.get_library_item(top.parent_work.item_id)
        result: Work | None = None
        async with fresh_reads():
            pending, done = [int(top.item_id)], set[int]()
            while pending:
                if (work_id := pending.pop(0)) in done:
                    continue
                done.add(work_id)
                try:
                    reloaded = await self._link_work_to_musicbrainz(
                        await works.get_library_item(work_id), reload=True
                    )
                except MediaNotFoundError:
                    # merged into another work or emptied earlier in the reload
                    continue
                if str(work_id) == work.item_id:
                    result = reloaded
                pending.extend(int(x.item_id) for x in reloaded.parts)
        try:
            return await works.get_library_item((result or work).item_id)
        except MediaNotFoundError:
            return work

    async def _link_work_to_musicbrainz(self, work: Work, reload: bool = False) -> Work:
        """
        Fill the empty fields of a library work in from MusicBrainz and mark it looked up.

        :param work: The library work.
        :param reload: Look it up again as on its first lookup, replacing what MusicBrainz
            gives now.
        :return: The library work, or the work it merged into.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return work
        works = self.mass.music.works
        first_lookup = reload or not work.metadata.last_musicbrainz_lookup
        hierarchy = await self._musicbrainz_work_hierarchy(musicbrainz, work)
        if not hierarchy:
            work.metadata.last_musicbrainz_lookup = int(time())
            await works.update_item_in_library(work.item_id, work)
            return work
        given = work
        # MusicBrainz answers a merged work's old id with the work it was merged into
        await self._follow_merged_id(work, hierarchy[0].id)
        # the caller reads the outcome off the work it passed in, which a merge may remove
        _fill_work_from_musicbrainz(work, hierarchy, reload)
        if survivor := await self._merge_duplicate_works(work, hierarchy[0].id):
            work = survivor
            _fill_work_from_musicbrainz(work, hierarchy, reload)
        new_composers: list[Artist] = []
        if not work.composers:
            new_composers = await self._get_library_composers(_composer_relations(hierarchy))
            work.composers = UniqueList(new_composers)
        await self._store_work(work, reload)
        if first_lookup:
            await self._take_musicbrainz_name(work, hierarchy[0], reload)
        await self._link_arrangements(int(work.item_id), hierarchy[0], replace=reload)
        if new_composers:
            await self.mass.music.tracks.credit_work_composers(int(work.item_id))
        else:
            await self._identify_sole_composer(musicbrainz, work, _composer_relations(hierarchy))
        # read again, as the composers may have been merged above
        work = await works.get_library_item(work.item_id)
        await self.set_composer_periods(int(x.item_id) for x in work.composers)
        # marked once every step is done, so a lookup cut short is done again on the next run
        looked_up = int(time())
        given.metadata.last_musicbrainz_lookup = work.metadata.last_musicbrainz_lookup = looked_up
        await works.mark_musicbrainz_lookup(int(work.item_id), looked_up)
        return work

    async def _store_work(self, work: Work, reload: bool) -> None:
        """
        Store a work filled in from MusicBrainz in the library.

        :param work: The library work.
        :param reload: The work is looked up again, so MusicBrainz's type, year, language and
            key replace the stored ones.
        """
        works = self.mass.music.works
        stored_type = (await works.get_library_item(work.item_id)).work_type if reload else None
        await works.update_item_in_library(work.item_id, work)
        # a library update keeps the values a work has, so MusicBrainz's current ones are written
        values: dict[str, Any] = {"version": work.version}
        if reload:
            values |= {
                "work_type": work.work_type,
                "composition_year": work.composition_year,
                "language": work.language,
                "musical_key": work.musical_key,
            }
        await self.mass.music.database.update(
            DB_TABLE_WORKS, {"item_id": int(work.item_id)}, values
        )
        if reload and work.work_type != stored_type:
            await self.mass.music.classification.update_item(MediaType.WORK, int(work.item_id))

    async def _musicbrainz_work_hierarchy(
        self, musicbrainz: MusicbrainzProvider, work: Work
    ) -> list[MusicBrainzWork]:
        """Return the MusicBrainz work a library work is and the works above it, or nothing."""
        if not work.mbid or not (mb_work := await musicbrainz.get_work_details(work.mbid)):
            return []
        return await self._with_parent_works(musicbrainz, mb_work)

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
        return await works.get_library_item(
            await works.merge_library_works_by_mbid(int(work.item_id), mbid)
        )

    async def _link_arrangements(
        self, work_id: int, mb_work: MusicBrainzWork, replace: bool = False
    ) -> None:
        """
        Link a work to the library works MusicBrainz names as its originals or arrangements.

        :param work_id: Library id of the work.
        :param mb_work: Its MusicBrainz work, with its work relations.
        :param replace: Drop the links of the work MusicBrainz no longer names.
        """
        works = self.mass.music.works
        if replace:
            await self.mass.music.database.execute_write(
                f"DELETE FROM {DB_TABLE_WORK_ARRANGEMENTS} "
                "WHERE work_id = :work_id OR source_work_id = :work_id",
                {"work_id": work_id},
            )
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

    async def _take_musicbrainz_name(
        self, work: Work, mb_work: MusicBrainzWork, reload: bool = False
    ) -> None:
        """
        Give a work the name MusicBrainz gives it, with MusicBrainz's catalog numbers.

        :param work: The library work.
        :param mb_work: Its MusicBrainz work.
        :param reload: Also replace the catalog numbers of a work that keeps its name.
        """
        named = musicbrainz_work_name(mb_work, self.preferred_language)
        if named == work.name and not reload:
            return
        await self.mass.music.works.rename_library_work(
            int(work.item_id),
            named,
            mb_work.catalog_numbers,
        )

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
        if artist.mbid or not _is_named_by(artist.name, mb_artist):
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


def _fill_work_from_musicbrainz(
    work: Work, hierarchy: Sequence[MusicBrainzWork], replace: bool = False
) -> None:
    """
    Fill the empty fields of a work in from its MusicBrainz work and the works above it.

    The work's version is the note MusicBrainz tells same-named works apart by, if any.

    :param work: The library work.
    :param hierarchy: Its MusicBrainz work followed by the works above it.
    :param replace: Replace the type, year, language and key by MusicBrainz's where it has them.
    """
    mb_work = hierarchy[0]
    if not work.mbid:
        work.mbid = mb_work.id
    work.version = mb_work.disambiguation or ""
    if work.work_type is None or replace:
        # a part, such as an aria, is typed by the work it belongs to, such as an opera
        work.work_type = (
            next((x.work_type for x in reversed(hierarchy) if x.work_type), None) or work.work_type
        )
    if replace:
        work.composition_year = mb_work.composition_year or work.composition_year
        work.language = mb_work.language or work.language
        work.musical_key = mb_work.musical_key or work.musical_key
    else:
        work.composition_year = work.composition_year or mb_work.composition_year
        work.language = work.language or mb_work.language
        work.musical_key = work.musical_key or mb_work.musical_key
    if not work.catalog_numbers:
        work.catalog_numbers = mb_work.catalog_numbers


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
