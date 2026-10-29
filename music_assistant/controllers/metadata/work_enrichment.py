"""
MusicBrainz work enrichment for the Metadata Controller.

Provides the WorkEnrichmentMixin, mixed into the MetaDataController, which fills the empty
fields of library works in from MusicBrainz: their type, year, language, catalog numbers, key,
movements and composers. Every library work is a MusicBrainz work, found through its own
MusicBrainz id. The composers MusicBrainz names for a work without any are taken from the
library's own artists, and are credited on its tracks without a composer too.

Library works that turn out to be the same MusicBrainz work are merged into the one holding
the most tracks. A work gets linked to the library works MusicBrainz names as its
originals or arrangements, and a part of a larger work, such as an aria or a movement, to the
work right above it, such as the act of an opera; a typed work below no typed level, such as a
concerto of a set, only when the library holds that work. A level the library lacks is added
without tracks, named in the user's language, together with the levels above it up to the
topmost typed one, or up to the top when none is typed.
A work looked up for the first time, and a level added, takes in the library works MusicBrainz
lists as its parts, such as the pieces of a set, and a part whose parent is further up its
MusicBrainz hierarchy moves under the level right above it. A work looked up for the first
time takes its MusicBrainz name, in the user's language, and lists the catalog numbers its name
gives first, and takes the movements MusicBrainz lists for it. A movement whose MusicBrainz title
is not in Latin script is named by a Latin alias of it, when MusicBrainz has one.
The tracks of a movement, such as an aria, move onto the work right above it as its movement,
when the library holds that work, such as the act. The movement's own work then leaves the
library once empty. When the movement is a group of pieces, such as a scene of an act, the work
above it keeps the group's pieces under the group in its movements.
A work refreshed by hand is looked up again as on its first lookup, taking MusicBrainz's
current name, version, type, year, language, key, catalog numbers, parent, parts, movements and
arrangements; a work MusicBrainz merged into another takes the other's id.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import replace
from time import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import Artist, ItemMapping, UniqueList, Work
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.constants import (
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACKS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.classical import group_piece
from music_assistant.helpers.compare import is_named_by
from music_assistant.helpers.work_titles import (
    parse_catalog_numbers,
    part_label,
    without_movement_number,
    without_part_subtitle,
)
from music_assistant.providers.musicbrainz.api_client import fresh_reads
from music_assistant.providers.musicbrainz.models import part_name
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider

from .constants import MAX_MUSICBRAINZ_WORK_DEPTH
from .helpers import (
    fill_artist_from_musicbrainz,
    is_latin,
    musicbrainz_work_name,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from music_assistant_models.media_items import Album, WorkMovement

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
    ``_musicbrainz_provider`` and ``_follow_merged_id``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger

        @property
        def preferred_language(self) -> str: ...  # noqa: D102

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
        if first_lookup or not work.metadata.movements:
            # the pieces stored under its groups, such as the scenes of a number, stay
            pieces = [x for x in work.metadata.movements or [] if x.parent is not None]
            work.metadata.movements = [*await self._shown_movements(hierarchy[0]), *pieces] or None
        new_composers: list[Artist] = []
        if not work.composers:
            new_composers = await self._get_library_composers(_composer_relations(hierarchy))
            work.composers = UniqueList(new_composers)
        moved_from = await self._set_parent_work(musicbrainz, work, hierarchy, reload)
        if first_lookup:
            # a library update adds to the movements a work has, so these are cleared to be replaced
            await works.set_movements(int(work.item_id), None)
        # a library update keeps the parent a work has, so a move is stored with the work
        await works.update_from_musicbrainz(work, replace=reload, parent=moved_from is not None)
        if first_lookup:
            await self._take_musicbrainz_name(work, hierarchy[0], reload)
        await self._link_arrangements(int(work.item_id), hierarchy[0], replace=reload)
        if moved_from:
            await self.mass.music.works.remove_if_empty(moved_from.item_id)
        if first_lookup:
            await self._adopt_parts(musicbrainz, int(work.item_id), hierarchy)
        if new_composers:
            await self.mass.music.tracks.credit_work_composers(int(work.item_id))
        else:
            await self._identify_sole_composer(musicbrainz, work, _composer_relations(hierarchy))
        # read again, as the composers may have been merged above
        work = await works.get_library_item(work.item_id)
        await self.mass.music.artists.set_composer_periods(int(x.item_id) for x in work.composers)
        if reload:
            await self._number_movement_tracks(work)
        await self._move_movement_tracks(work, hierarchy)
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
            return old_parent
        return None

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
                await works.set_parent_work(int(sibling.item_id), work_id)
                await self._move_movement_tracks(sibling, [mb_sibling, *hierarchy])
                if sibling.parent_work and int(sibling.parent_work.item_id) != work_id:
                    await self.mass.music.works.remove_if_empty(sibling.parent_work.item_id)

    async def _move_movement_tracks(self, work: Work, hierarchy: Sequence[MusicBrainzWork]) -> None:
        """
        Move the tracks of a movement work onto the work right above it, if the library holds it.

        Only tracks performing the movement alone move, whatever works above it they also name.
        They are numbered by the movement's place in the list of parts of the work above it and
        named as the track names it, else as that list does, both without a number of their own.
        A moved track carries the movement's MusicBrainz work id, as the movement it performs.
        A movement that is a group of pieces, such as a scene, leaves its pieces under it in the
        movements of the work above. The movement work leaves the library once empty.

        :param work: The library work.
        :param hierarchy: The work's MusicBrainz work followed by the works above it.
        """
        if len(hierarchy) < 2:
            return
        mb_work, mb_parent = hierarchy[0], hierarchy[1]
        works = self.mass.music.works
        parent = await works.get_library_item_by_external_id(mb_parent.id, ExternalID.MB_WORK)
        if not parent or parent.item_id == work.item_id:
            return
        database = self.mass.music.database
        rows = await database.get_rows_from_query(
            f"SELECT item_id, movement_name FROM {DB_TABLE_TRACKS} t WHERE work_id = :work_id "
            f"AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} e "
            "WHERE e.media_type = :media_type AND e.item_id = t.item_id "
            "AND e.external_id_type = :id_type AND e.external_id NOT IN :chain_ids)",
            {
                "work_id": int(work.item_id),
                "media_type": MediaType.TRACK.value,
                "id_type": ExternalID.MB_WORK.value,
                # the ids of the works above, which tags may name together with the movement
                "chain_ids": [x.id for x in hierarchy],
            },
            limit=0,
        )
        if not rows:
            return
        track_ids = [int(row["item_id"]) for row in rows]
        if movement := mb_parent.part_movement(mb_work.id):
            movement = await self._shown_movement(mb_parent, movement)
        number, mb_name = (movement.number, movement.name) if movement else (None, mb_work.title)
        for row in rows:
            # the track's own name of the movement, in its language and telling sections apart
            # that MusicBrainz joins, such as a Christe within the Kyrie
            name = row["movement_name"]
            # a track linked by its file's tags lacks the movement's id, without which a move of
            # the work above would later take it on as performing that whole work
            await self.mass.music.tracks.set_work(
                int(row["item_id"]),
                int(parent.item_id),
                number,
                without_movement_number(name or mb_name),
                mb_work_id=mb_work.id,
            )
        self.logger.debug(
            'Moved %s tracks of work "%s" onto "%s"', len(rows), work.name, parent.name
        )
        if number is not None:
            await self._store_group_pieces(int(parent.item_id), number, mb_work)
        await self.mass.music.classification.update(track_ids=track_ids)
        await self.mass.music.tracks.credit_work_composers(int(parent.item_id))
        await self.mass.music.works.remove_if_empty(work.item_id)

    async def _number_movement_tracks(
        self, work: Work, track_ids: Collection[int] | None = None
    ) -> None:
        """
        Give the tracks of a work performing one movement it lists that movement's number.

        :param work: The library work, with its movements.
        :param track_ids: Library ids of the tracks of the work to number, all of them if None.
        """
        numbers = {
            x.mbid.casefold(): x.number
            for x in work.metadata.movements or []
            if x.parent is None and x.mbid
        }
        if not numbers:
            return
        database = self.mass.music.database
        rows = await database.get_rows_from_query(
            f"SELECT t.item_id, t.movement_number, e.external_id FROM {DB_TABLE_TRACKS} t "
            f"JOIN {DB_TABLE_EXTERNAL_ID_LOOKUP} e ON e.media_type = :media_type "
            "AND e.item_id = t.item_id AND e.external_id_type = :id_type "
            "WHERE t.work_id = :work_id",
            {
                "work_id": int(work.item_id),
                "media_type": MediaType.TRACK.value,
                "id_type": ExternalID.MB_WORK.value,
            },
            limit=0,
        )
        performed: dict[int, tuple[int | None, set[int]]] = {}
        for row in rows:
            stored, found = performed.setdefault(row["item_id"], (row["movement_number"], set()))
            if (number := numbers.get(row["external_id"].casefold())) is not None:
                found.add(number)
        for track_id, (stored, found) in performed.items():
            # a track performing several movements has no number of one
            if (track_ids is None or track_id in track_ids) and len(found) == 1:
                number = found.pop()
                if number != stored:
                    await self.mass.music.tracks.set_movement_number(track_id, number)

    async def _store_group_pieces(
        self, work_id: int, number: int, mb_group: MusicBrainzWork
    ) -> None:
        """
        Store the pieces of a group in the movements of the work listing it, under the group.

        The pieces replace those stored for the group before. A group without pieces, such as a
        single aria, stores nothing.

        :param work_id: Library id of the work listing the group, such as the act.
        :param number: The number of the group in the work's movements.
        :param mb_group: The group's MusicBrainz work, with its parts.
        """
        if not (pieces := mb_group.movements):
            return
        work = await self.mass.music.works.get_library_item(work_id)
        movements = [
            *(x for x in work.metadata.movements or [] if x.parent != number),
            *(group_piece(number, x) for x in pieces),
        ]
        # written as a whole, as a library update would keep the pieces it replaces
        await self.mass.music.works.set_movements(work_id, movements)

    async def _shown_movements(self, mb_work: MusicBrainzWork) -> list[WorkMovement]:
        """
        Return the movements of a MusicBrainz work named as its list of movements shows them.

        Movements that would be named alike, such as two acts of different versions both
        shown as "Act I", keep MusicBrainz's names.

        :param mb_work: The MusicBrainz work.
        """
        movements = mb_work.movements
        shown = [await self._shown_movement(mb_work, x) for x in movements]
        counts = Counter(x.name for x in shown)
        return [
            movement if new.name != movement.name and counts[new.name] > 1 else new
            for movement, new in zip(movements, shown, strict=True)
        ]

    async def _shown_movement(
        self, mb_work: MusicBrainzWork, movement: WorkMovement
    ) -> WorkMovement:
        """
        Return a movement of a MusicBrainz work named as the work's list of movements shows it.

        A movement is named in Latin script, if MusicBrainz has such a name. An act or part with
        parts of its own is then named as "Act I" or "Part II", and a single piece named after its
        act, such as an act's intermezzo, by its own name.

        :param mb_work: The MusicBrainz work the movement is a part of.
        :param movement: The movement, as the work lists it.
        """
        name, work_name = movement.name, mb_work.title
        part: MusicBrainzWork | None = None
        musicbrainz = self._musicbrainz_provider()
        if not is_latin(name) and movement.mbid and musicbrainz:
            part = await musicbrainz.get_work_details(movement.mbid)
            work_name = musicbrainz_work_name(mb_work, self.preferred_language)
            if part and is_latin(latin := musicbrainz_work_name(part, self.preferred_language)):
                name = part_name(latin, work_name)
        # part_label reads the end of a work's name, which an act's subtitle must not end
        if not (label := part_label(f"{without_part_subtitle(work_name)}: {name}")):
            return replace(movement, name=name)
        if not part and movement.mbid and musicbrainz:
            part = await musicbrainz.get_work_details(movement.mbid)
        if not part:
            return replace(movement, name=name)
        if part.movements:
            return replace(movement, name=label)
        # a single piece named after its act, such as "Atto III. Intermezzo"
        own_name = name[len(without_part_subtitle(name)) :]
        return replace(movement, name=own_name[2:] if own_name.startswith(". ") else name)

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
        parent.metadata.movements = await self._shown_movements(mb_work) or None
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
            await works.remove_arrangement_links(work_id)
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

    The work's version is the note MusicBrainz tells same-named works apart by, if any. A work
    MusicBrainz gives no catalog numbers takes those its name gives of the work itself.

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
        work.catalog_numbers = _catalog_numbers(mb_work, work.name) or _named_catalog_numbers(
            work.name, hierarchy[1:]
        )


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


def _named_catalog_numbers(name: str, above: Sequence[MusicBrainzWork]) -> list[str]:
    """
    Return the catalog numbers a work's name gives of the work itself.

    :param name: The name of the work.
    :param above: The MusicBrainz works above it, whose numbers its parts, such as acts, name
        without carrying them.
    """
    named_above = {
        create_safe_string(x, True, True)
        for level in above
        for x in parse_catalog_numbers(level.title)
    }
    return [
        x
        for x in parse_catalog_numbers(name)
        if create_safe_string(x, True, True) not in named_above
        # the start of a range, as in "HWV 348-350", numbers no single work
        and not re.search(rf"{re.escape(x)}\s*[-\u2010\u2013]\s*\d", name)
    ]


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
