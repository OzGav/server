"""
MusicBrainz work enrichment for the Metadata Controller.

Provides the WorkEnrichmentMixin, mixed into the MetaDataController, which fills the empty
fields of library works in from MusicBrainz: their type, year, language, catalog numbers, key,
movements and composers. Every library work is a MusicBrainz work, found through its own
MusicBrainz id. The composers MusicBrainz names for a work without any are taken from the
library's own artists, or added to the library when MusicBrainz links them to a loaded music
provider, and are credited on its tracks without a composer too. A composer holding another
MusicBrainz id than the MusicBrainz composer of its name, such as a pop singer named after a
composer, gives way to that composer on the work and its tracks.

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
A classical track without a work joins the library work of its MusicBrainz work, which is added
from MusicBrainz when the library lacks it. A track performing several parts of one work, such
as the arias of an act, joins that work.
A work refreshed by hand, or by the 90 day re-read, is looked up again as on its first lookup,
taking MusicBrainz's current name, version, type, year, language, key, catalog numbers, parent,
parts, movements and arrangements; a work MusicBrainz merged into another takes the other's id.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from time import time
from typing import TYPE_CHECKING, Any
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
    DB_TABLE_TRACK_CREDITS,
    DB_TABLE_TRACKS,
    DB_TABLE_WORK_ARRANGEMENTS,
    DB_TABLE_WORK_ARTISTS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.classical import group_piece
from music_assistant.controllers.music.helpers import provider_mappings_from_urls
from music_assistant.helpers.compare import compare_strings, is_named_by
from music_assistant.helpers.json import json_dumps
from music_assistant.helpers.logging import log_once
from music_assistant.helpers.work_titles import (
    parent_work_name,
    parse_catalog_numbers,
    part_label,
    split_movement_title,
    without_movement_number,
    without_part_subtitle,
)
from music_assistant.providers.musicbrainz.api_client import fresh_reads
from music_assistant.providers.musicbrainz.constants import GROUP_ARTIST_TYPES, MUSICBRAINZ_WEB_URL
from music_assistant.providers.musicbrainz.models import part_name
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider, relation_urls

from .constants import MAX_MUSICBRAINZ_WORK_DEPTH, MUSICBRAINZ_LINK_DOMAINS
from .helpers import (
    fill_artist_from_musicbrainz,
    is_latin,
    musicbrainz_name,
    musicbrainz_work_name,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Hashable, Iterable, Sequence

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

    Expects to be mixed with a class providing ``mass``, ``logger``, ``_logged_hints``,
    ``preferred_language``, ``set_composer_periods``, ``_musicbrainz_provider``,
    ``_follow_merged_id`` and ``_name_after_musicbrainz``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _logged_hints: set[Hashable]

        @property
        def preferred_language(self) -> str: ...  # noqa: D102

        async def set_composer_periods(  # noqa: D102
            self, artist_ids: Iterable[int] | None = None
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

        async def _follow_merged_id(self, item: Album | Artist | Work, mbid: str) -> None: ...

        async def _name_after_musicbrainz(
            self, artist: Artist, details: MusicBrainzArtist
        ) -> bool: ...

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
        self._log_part_without_parent(hierarchy[0])
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
            await self.mass.music.database.execute_write(
                f"UPDATE {DB_TABLE_WORKS} SET metadata = json_remove(metadata, '$.movements') "
                "WHERE item_id = :work_id AND json_valid(metadata)",
                {"work_id": int(work.item_id)},
            )
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
        mb_composers = _composer_relations(hierarchy)
        if new_composers:
            await self.mass.music.tracks.credit_work_composers(int(work.item_id))
        elif not await self._replace_namesake_composers(work, mb_composers):
            await self._identify_sole_composer(musicbrainz, work, mb_composers)
        # read again, as the composers may have been replaced or merged above
        work = await works.get_library_item(work.item_id)
        await self.set_composer_periods(int(x.item_id) for x in work.composers)
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
            # a movement's tracks belong on the work above it; a work without a parent yet
            # has them moved there once it gets one
            if work.parent_work:
                await self._move_movement_tracks(
                    work, await self._musicbrainz_work_hierarchy(musicbrainz, work)
                )
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
                await self._move_movement_tracks(sibling, [mb_sibling, *hierarchy])
                if sibling.parent_work and int(sibling.parent_work.item_id) != work_id:
                    await self._remove_if_empty(sibling.parent_work.item_id)

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
            await database.execute_write(
                f"UPDATE {DB_TABLE_TRACKS} SET work_id = :parent_id, movement_number = :number, "
                "movement_name = :name WHERE item_id = :track_id",
                {
                    "parent_id": int(parent.item_id),
                    "number": number,
                    "name": without_movement_number(name or mb_name),
                    "track_id": int(row["item_id"]),
                },
            )
        # a track linked by its file's tags lacks the movement's id, without which a move of
        # the work above would later take it on as performing that whole work
        await database.upsert_many(
            DB_TABLE_EXTERNAL_ID_LOOKUP,
            [
                {
                    "media_type": MediaType.TRACK.value,
                    "external_id_type": ExternalID.MB_WORK.value,
                    "external_id": mb_work.id,
                    "item_id": track_id,
                }
                for track_id in track_ids
            ],
        )
        self.logger.debug(
            'Moved %s tracks of work "%s" onto "%s"', len(rows), work.name, parent.name
        )
        if number is not None:
            await self._store_group_pieces(int(parent.item_id), number, mb_work)
        await self.mass.music.classification.update(track_ids=track_ids)
        await self.mass.music.tracks.credit_work_composers(int(parent.item_id))
        await self._remove_if_empty(work.item_id)

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
                    await database.execute_write(
                        f"UPDATE {DB_TABLE_TRACKS} SET movement_number = :number "
                        "WHERE item_id = :track_id",
                        {"number": number, "track_id": track_id},
                    )

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
        await self.mass.music.database.execute_write(
            f"UPDATE {DB_TABLE_WORKS} SET metadata = json_set(metadata, '$.movements', "
            "json(:movements)) WHERE item_id = :work_id AND json_valid(metadata)",
            {"movements": json_dumps([x.to_dict() for x in movements]), "work_id": work_id},
        )

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
        self, mb_artist: MusicBrainzArtist, album: Album | None = None
    ) -> Artist | None:
        """
        Add a MusicBrainz artist to the library on the music providers it links to, if any.

        The artist is added with its type, life dates and country, named in Latin script
        where its MusicBrainz name is not.

        :param mb_artist: The MusicBrainz artist.
        :param album: The album the artist performs on, named when the artist is not added.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return None
        try:
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
        name, sort_name = details.name, details.sort_name
        if not is_latin(name):
            name, sort_name = musicbrainz_name(details, self.preferred_language) or (
                name,
                sort_name,
            )
        artist = Artist(
            item_id=mappings[0].item_id,
            provider=mappings[0].provider_instance,
            name=name,
            sort_name=sort_name,
            provider_mappings=set(mappings),
            external_ids={(ExternalID.MB_ARTIST, details.id)},
        )
        fill_artist_from_musicbrainz(artist, details)
        return await self.mass.music.artists.add_item_to_library(artist)

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
            (DB_TABLE_TRACK_CREDITS, f"track_id IN ({tracks})"),
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
            f"AND NOT EXISTS (SELECT 1 FROM {DB_TABLE_TRACK_CREDITS} "
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

    async def _confirm_composer(self, mbid: str, details: MusicBrainzArtist | None = None) -> None:
        """Join a confirmed composer's other spellings and joined credits, and give it its period."""
        await self._merge_composer_by_aliases(mbid, details)
        # read again, as the join may have merged the artist into another
        holders = await self.mass.music.artists.get_library_items_by_external_id(
            mbid, ExternalID.MB_ARTIST, limit=None
        )
        await self.set_composer_periods(int(holder.item_id) for holder in holders)

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

    def _log_part_without_parent(self, mb_work: MusicBrainzWork) -> None:
        """Log a MusicBrainz work titled as a movement or act that is part of no other work."""
        if mb_work.parent_relation or not (
            split_movement_title(mb_work.title) or parent_work_name(mb_work.title)
        ):
            return
        log_once(
            self.logger,
            self._logged_hints,
            ("work", mb_work.id),
            logging.DEBUG,
            'MusicBrainz work "%s" looks like a part of a larger work but is not linked to it, '
            'add its "part of" relationship, or merge it into the same part listed there: '
            "%s/work/%s",
            mb_work.title,
            MUSICBRAINZ_WEB_URL,
            mb_work.id,
        )


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


def _alias_names(mb_artist: MusicBrainzArtist) -> set[str]:
    """Return a MusicBrainz artist's main name and its aliases that are not search hints."""
    aliases = mb_artist.aliases or []
    return {mb_artist.name, *(x.name for x in aliases if (x.type or "").lower() != "search hint")}
