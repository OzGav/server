"""
MusicBrainz work enrichment for the Metadata Controller.

Provides the WorkEnrichmentMixin, mixed into the MetaDataController, which fills the empty
fields of library works in from MusicBrainz: their type, year, language, catalog numbers, key
and composers. Every library work is a MusicBrainz work, found through its own
MusicBrainz id. The composers MusicBrainz names for a work without any are taken from the
library's own artists, and are credited on its tracks without a composer too.

Library works that turn out to be the same MusicBrainz work are merged into the one holding
the most tracks. A work gets linked to the library works MusicBrainz names as its
originals or arrangements, and a part of a larger work, such as an aria or a movement, to
the work right above it, such as the act of an opera; a typed work below no typed level,
such as a concerto of a set, only when the library holds that work. A level the library
lacks is added without tracks, named in the user's language, together with the levels above
it up to the topmost typed one, or up to the top when none is typed.
A work looked up for the first time, and a level added, takes in the library works
MusicBrainz lists as its parts, such as the pieces of a set, and a part whose parent is
further up its MusicBrainz hierarchy moves under the level right above it.

A work looked up for the first time takes its MusicBrainz name, in the user's language, and
lists the catalog numbers its name gives first.
A classical track without a work joins the library work of its MusicBrainz work, which is added
from MusicBrainz when the library lacks it. A track performing several parts of one work, such
as the arias of an act, joins that work.
A work refreshed by hand, or by the 90 day re-read, is looked up again as on its first lookup,
taking MusicBrainz's current name, version, type, year, language, key, catalog numbers,
parent, parts and arrangements; a work MusicBrainz merged into another takes the other's id.
"""

from __future__ import annotations

import logging
from time import time
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Artist, ItemMapping, UniqueList, Work
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.constants import (
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARRANGEMENTS,
    DB_TABLE_WORKS,
)
from music_assistant.helpers.compare import is_named_by
from music_assistant.helpers.work_titles import (
    parse_catalog_numbers,
)
from music_assistant.providers.musicbrainz.api_client import fresh_reads
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider

from .constants import MAX_MUSICBRAINZ_WORK_DEPTH
from .helpers import (
    fill_artist_from_musicbrainz,
    musicbrainz_work_name,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

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

    async def remove_work_if_empty(self, work_id: str) -> None:
        """
        Remove a library work that has no tracks, parts or arrangements, and so its parents.

        :param work_id: Library id of the work, which may have left the library already.
        """
        try:
            await self._remove_if_empty(work_id)
        except MediaNotFoundError:
            return

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
        moved_from = await self._set_parent_work(musicbrainz, work, hierarchy, reload)
        await self._store_work(work, reload)
        if first_lookup:
            await self._take_musicbrainz_name(work, hierarchy[0], reload)
        if moved_from and work.parent_work:
            # a library update keeps the parent a work has, so the move is written here
            await self.mass.music.database.execute_write(
                f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = :parent_id WHERE item_id = :work_id",
                {"parent_id": int(work.parent_work.item_id), "work_id": int(work.item_id)},
            )
        await self._link_arrangements(int(work.item_id), hierarchy[0], replace=reload)
        if moved_from:
            await self._remove_if_empty(moved_from.item_id)
        if first_lookup:
            await self._adopt_parts(musicbrainz, int(work.item_id), hierarchy)
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

    async def _set_parent_work(
        self,
        musicbrainz: MusicbrainzProvider,
        work: Work,
        hierarchy: Sequence[MusicBrainzWork],
        reload: bool,
    ) -> Work | None:
        """
        Give a work the library work of the MusicBrainz level above it as its parent.

        A parent from a track title, or a level further up the work's own hierarchy, gives way
        to MusicBrainz's. On a reload MusicBrainz's level always wins, and a work MusicBrainz no
        longer has as a part of another loses its parent.

        :param musicbrainz: The MusicBrainz provider.
        :param work: The library work, given its new parent.
        :param hierarchy: The work's MusicBrainz work followed by the works above it.
        :param reload: The work is looked up again.
        :return: The parent the work moved away from, if any.
        """
        old_parent = (
            await self.mass.music.works.get_library_item(work.parent_work.item_id)
            if work.parent_work
            else None
        )
        levels = await self._parent_levels(hierarchy)
        if levels and (
            not old_parent or reload or _gives_way_to_musicbrainz(old_parent, hierarchy)
        ):
            parent = await self._get_or_add_parent_work(musicbrainz, levels, work)
            work.parent_work = ItemMapping.from_item(parent)
            return old_parent if old_parent and old_parent.item_id != parent.item_id else None
        if reload and old_parent:
            work.parent_work = None
            await self.mass.music.database.execute_write(
                f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = NULL WHERE item_id = :work_id",
                {"work_id": int(work.item_id)},
            )
            return old_parent
        return None

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

    async def _add_track_work(self, track_id: str, work_ids: Collection[str]) -> None:
        """
        Give a classical library track without a work the work its MusicBrainz work ids name.

        A work the library lacks is added from MusicBrainz, named in the user's language, and
        filled in right away. A track that stays on the work its one id names has no movement
        number or name. A track performing several parts of one work, such as the arias of an
        act, becomes a recording of that work without a movement number. A track performing
        parts of different works is left without one.

        :param track_id: Library id of the track.
        :param work_ids: The MusicBrainz ids of the works the track performs, from a source
            that identifies its recording reliably.
        """
        if not work_ids or not (musicbrainz := self._musicbrainz_provider()):
            return
        tracks, works = self.mass.music.tracks, self.mass.music.works
        track = await tracks.get_library_item(track_id)
        if track.work or not track.is_classical:
            return
        if not (work_id := await self._work_of_parts(musicbrainz, work_ids)):
            return
        if len(work_ids) > 1:
            track.movement_number = None
            await self.mass.music.database.execute_write(
                f"UPDATE {DB_TABLE_TRACKS} SET movement_number = NULL WHERE item_id = :track_id",
                {"track_id": int(track.item_id)},
            )
        if work := await works.get_library_item_by_external_id(work_id, ExternalID.MB_WORK):
            track.work = ItemMapping.from_item(work)
            await tracks.update_item_in_library(track.item_id, track)
        else:
            try:
                mb_work = await musicbrainz.get_work_details(work_id)
            except InvalidDataError:
                return
            if not mb_work:
                return
            track.work = Work(
                item_id=mb_work.id,
                provider="musicbrainz",
                name=musicbrainz_work_name(mb_work, self.preferred_language),
                provider_mappings=set(),
                external_ids={(ExternalID.MB_WORK, mb_work.id)},
                composers=UniqueList(track.composers),
            )
            await tracks.update_item_in_library(track.item_id, track)
            if (stored := await tracks.get_library_item(track.item_id)).work:
                await self._link_work_to_musicbrainz(
                    await works.get_library_item(stored.work.item_id)
                )
        if len(work_ids) == 1:
            # a track left on the work its id names performs that work, not a movement its
            # title names
            await self.mass.music.database.execute_write(
                f"UPDATE {DB_TABLE_TRACKS} SET movement_number = NULL, movement_name = NULL "
                "WHERE item_id = :track_id AND work_id IN (SELECT item_id "
                f"FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} WHERE media_type = :media_type "
                "AND external_id_type = :id_type AND external_id = :mbid)",
                {
                    "track_id": int(track.item_id),
                    "media_type": MediaType.WORK.value,
                    "id_type": ExternalID.MB_WORK.value,
                    "mbid": work_id,
                },
            )

    async def _work_of_parts(
        self, musicbrainz: MusicbrainzProvider, work_ids: Collection[str]
    ) -> str | None:
        """
        Return the MusicBrainz id of the one work the given MusicBrainz works are, or are parts of.

        That is the work itself for a single id. Several ids name the work right above each of
        them, or one of them when the others are its parts; a work and one of its parts, as tags
        name a movement together with its work, name that part. Ids under different works name
        none.

        :param musicbrainz: The MusicBrainz provider.
        :param work_ids: The MusicBrainz ids of the works a track performs.
        """
        if len(work_ids) == 1:
            return next(iter(work_ids))
        above: dict[str, str | None] = {}
        for work_id in work_ids:
            try:
                mb_work = await musicbrainz.get_work_details(work_id)
            except InvalidDataError:
                return None
            relation = await self._parent_to_follow(mb_work) if mb_work else None
            above[work_id] = relation.work.id if relation and relation.work else None
        candidates = sorted({*above, *(x for x in above.values() if x)})
        shared = next((x for x in candidates if all(y == x or above[y] == x for y in above)), None)
        if shared in above and len(above) == 2:
            return next(x for x in above if x != shared)
        return shared

    async def _musicbrainz_work_hierarchy(
        self, musicbrainz: MusicbrainzProvider, work: Work
    ) -> list[MusicBrainzWork]:
        """Return the MusicBrainz work a library work is and the works above it, or nothing."""
        if not work.mbid or not (mb_work := await musicbrainz.get_work_details(work.mbid)):
            return []
        return await self._with_parent_works(musicbrainz, mb_work)

    async def _parent_levels(self, hierarchy: Sequence[MusicBrainzWork]) -> list[MusicBrainzWork]:
        """
        Return the levels above a work that are to be library works, the level right above first.

        The levels reach up to the topmost level with a type, or to the top when none has one.
        The level right above the topmost typed level, such as the set of a concerto, is only
        included when the library holds it.

        :param hierarchy: The work's MusicBrainz work followed by the works above it.
        """
        typed = [index for index, mb_work in enumerate(hierarchy) if mb_work.type]
        end = typed[-1] + 1 if typed else len(hierarchy)
        if end < len(hierarchy) and await self.mass.music.works.get_library_item_by_external_id(
            hierarchy[end].id, ExternalID.MB_WORK
        ):
            end += 1
        return list(hierarchy[1:end])

    async def _adopt_parts(
        self,
        musicbrainz: MusicbrainzProvider,
        work_id: int,
        hierarchy: Sequence[MusicBrainzWork],
        child_id: int | None = None,
    ) -> None:
        """
        Move the library works MusicBrainz has as parts of a work under it.

        Considered are the parts of the work's parent and of the library work holding the
        level above it, such as the arias next to an act in its opera, and the library works
        of the parts MusicBrainz lists for it anywhere, such as the pieces of a set. The tracks
        of a part moving under it become its movements, and a former parent left empty leaves
        the library.

        :param musicbrainz: The MusicBrainz provider.
        :param work_id: Library id of the work.
        :param hierarchy: The work's MusicBrainz work followed by the works above it.
        :param child_id: Library id of a work that is left where it is, such as the work being
            linked below the work.
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
        part_ids = _part_ids(hierarchy[0])
        if not (parent_ids or part_ids):
            return
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT item_id FROM {DB_TABLE_WORKS} w WHERE item_id NOT IN :skip_ids "
            "AND (parent_work_id IN :parent_ids "
            f"OR EXISTS (SELECT 1 FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} e "
            "WHERE e.media_type = :media_type AND e.item_id = w.item_id "
            "AND e.external_id_type = :id_type AND e.external_id IN :part_ids))",
            {
                "parent_ids": sorted(parent_ids),
                "skip_ids": [x for x in (work_id, child_id) if x is not None],
                "media_type": MediaType.WORK.value,
                "id_type": ExternalID.MB_WORK.value,
                "part_ids": sorted(part_ids),
            },
            limit=0,
        )
        for row in rows:
            sibling = await works.get_library_item(row["item_id"])
            if not sibling.mbid or not (
                mb_sibling := await musicbrainz.get_work_details(sibling.mbid)
            ):
                continue
            if any(x.work and x.work.id == hierarchy[0].id for x in mb_sibling.parent_relations):
                await self.mass.music.database.execute_write(
                    f"UPDATE {DB_TABLE_WORKS} SET parent_work_id = :work_id "
                    "WHERE item_id = :sibling_id",
                    {"work_id": work_id, "sibling_id": int(sibling.item_id)},
                )
                if sibling.parent_work and int(sibling.parent_work.item_id) != work_id:
                    await self._remove_if_empty(sibling.parent_work.item_id)

    async def _remove_if_empty(self, work_id: str) -> None:
        """Remove a library work that has no tracks, parts or arrangements, and so its parents."""
        works = self.mass.music.works
        work = await works.get_library_item(work_id)
        if (
            work.parts
            or work.arrangements
            or work.arrangement_of
            or await self.mass.music.database.get_row(DB_TABLE_TRACKS, {"work_id": int(work_id)})
        ):
            return
        await works.remove_item_from_library(work_id)
        if work.parent_work:
            await self._remove_if_empty(work.parent_work.item_id)

    async def _with_parent_works(
        self, musicbrainz: MusicbrainzProvider, mb_work: MusicBrainzWork
    ) -> list[MusicBrainzWork]:
        """Return a MusicBrainz work followed by the works it is a part of, top-level last."""
        hierarchy = [mb_work]
        while (
            len(hierarchy) <= MAX_MUSICBRAINZ_WORK_DEPTH
            and (parent := await self._parent_to_follow(hierarchy[-1]))
            and parent.work
        ):
            if not (parent_work := await musicbrainz.get_work_details(parent.work.id)):
                break
            hierarchy.append(parent_work)
        return hierarchy

    async def _parent_to_follow(self, mb_work: MusicBrainzWork) -> MusicBrainzRelation | None:
        """
        Return the relation to the work a MusicBrainz work is a part of, if any.

        Of a part of several works, such as of a work and of an edition of it, the one the
        library holds comes first, then the one MusicBrainz tells apart without a comment.
        """
        relations = mb_work.parent_relations
        if len(relations) < 2:
            return mb_work.parent_relation
        works = self.mass.music.works
        for relation in relations:
            if relation.work and await works.get_library_item_by_external_id(
                relation.work.id, ExternalID.MB_WORK
            ):
                return relation
        return next((x for x in relations if x.work and not x.work.disambiguation), relations[0])

    async def _merge_duplicate_works(self, work: Work, mbid: str) -> Work | None:
        """Merge the library works that are a given MusicBrainz work, return the survivor if any."""
        works = self.mass.music.works
        holders = await works.get_library_items_by_external_id(mbid, ExternalID.MB_WORK, limit=None)
        if all(x.item_id == work.item_id for x in holders):
            return None
        return await works.get_library_item(
            await works.merge_library_works_by_mbid(int(work.item_id), mbid)
        )

    async def _get_or_add_parent_work(
        self, musicbrainz: MusicbrainzProvider, levels: Sequence[MusicBrainzWork], child: Work
    ) -> Work:
        """
        Return the library work of a child's parent MusicBrainz work, adding it if needed.

        A parent the library lacks is added, named in the user's language, under the levels
        above it, which are added as well up to the first one the library holds. An added level
        takes in the library works MusicBrainz lists as its parts, other than the child.

        :param musicbrainz: The MusicBrainz provider.
        :param levels: The parent MusicBrainz work followed by the levels above it that are to
            be library works.
        :param child: The library work it is the parent of.
        """
        mb_work = levels[0]
        works = self.mass.music.works
        if holder := await works.get_library_item_by_external_id(mb_work.id, ExternalID.MB_WORK):
            return holder
        grandparent = (
            await self._get_or_add_parent_work(musicbrainz, levels[1:], child)
            if len(levels) > 1
            else None
        )
        parent = Work(
            item_id=mb_work.id,
            provider="musicbrainz",
            name=musicbrainz_work_name(mb_work, self.preferred_language),
            provider_mappings=set(),
            # filled in right here, so the link run need not look it up again
            metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time())),
            composers=UniqueList(
                await self._get_library_composers(mb_work.composer_relations) or child.composers
            ),
            parent_work=ItemMapping.from_item(grandparent) if grandparent else None,
        )
        _fill_work_from_musicbrainz(parent, levels)
        added = await works.create_library_item(parent)
        await self._link_arrangements(int(added.item_id), mb_work)
        await self._adopt_parts(musicbrainz, int(added.item_id), levels, int(child.item_id))
        return added

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
        Give a work the name MusicBrainz gives it and the catalog numbers that go with it.

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
            _catalog_numbers(mb_work, named) or parse_catalog_numbers(named),
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
        if artist.mbid or not is_named_by(artist.name, mb_artist):
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
        work.catalog_numbers = _catalog_numbers(mb_work, work.name)


def _gives_way_to_musicbrainz(parent: Work, hierarchy: Sequence[MusicBrainzWork]) -> bool:
    """Return whether a work's parent is to make way for the parent MusicBrainz names."""
    # a level further up the work's own hierarchy gives way to the levels below it
    return parent.mbid in {x.id for x in hierarchy[2:]}


def _catalog_numbers(mb_work: MusicBrainzWork, name: str) -> list[str]:
    """Return the catalog numbers of a MusicBrainz work, those the given name of it gives first."""
    named = {create_safe_string(x, True, True) for x in parse_catalog_numbers(name)}
    return sorted(
        mb_work.catalog_numbers, key=lambda x: create_safe_string(x, True, True) not in named
    )


def _part_ids(mb_work: MusicBrainzWork) -> set[str]:
    """Return the MusicBrainz ids of the works MusicBrainz lists as parts of a work."""
    return {
        x.work.id
        for x in mb_work.relations
        if x.type == "parts" and x.direction == "forward" and x.work
    }


def _composer_relations(hierarchy: Sequence[MusicBrainzWork]) -> list[MusicBrainzRelation]:
    """Return the composer relations of the lowest work in a hierarchy that has any."""
    return next((x.composer_relations for x in hierarchy if x.composer_relations), [])
