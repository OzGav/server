"""
MusicBrainz performer enrichment for the Metadata Controller.

Provides the PerformerEnrichmentMixin, mixed into the MetaDataController, which reads the
MusicBrainz recordings of classical library tracks. The artists a track credits that the
recording names as its performers get their MusicBrainz id, and the track gets the year it
was recorded and the ids of the works it is a performance of. A credited artist is recognised
on the recording by its name, its MusicBrainz id, or the music provider artist MusicBrainz
links it to. A track whose only performers are its track artists gets their exact performing
roles. Credits are never renamed, changed or removed, and a performer the track does not
credit is only added to a track that credits no artist but its composers, or when an album's
metadata update is forced. Such a track gets the performers the recording names that it does
not credit yet, taken from the library's own artists or added to the library when MusicBrainz
links them to a loaded music provider. A classical track without a work gets the work its
file names by MusicBrainz id, or else the work of its recording, unless the recording was found
by a name search alone or it is not known how it was found. A recording performing a whole group
of pieces, such as a scene of an act, gives the work listing that group the group's pieces.
A recording read again, by the 90 day re-read or by hand, moves a track whose recording now
performs other works onto those works, unless the track's file names its own works.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from time import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole, ExternalID, MediaType, RecordingSource
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Credit, ItemMapping

from music_assistant.controllers.music.classical import PERFORMING_ROLES
from music_assistant.controllers.music.helpers import provider_mappings_from_urls
from music_assistant.helpers.compare import compare_strings, is_named_by
from music_assistant.helpers.logging import log_once
from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.musicbrainz.constants import MUSICBRAINZ_WEB_URL
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider, relation_urls

from .constants import MUSICBRAINZ_LINK_DOMAINS, REFRESH_INTERVAL

if TYPE_CHECKING:
    from collections.abc import Collection, Hashable, Sequence

    from music_assistant_models.media_items import Album, Artist, MediaItemType, Track, Work

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz.models import (
        MusicBrainzArtist,
        MusicBrainzRecording,
        MusicBrainzRelation,
        MusicBrainzWork,
    )


class Reread(Enum):
    """Why the recordings of classical tracks read before are read again, with fresh data."""

    # the 90 day re-read of the MusicBrainz link run, which calls no other service
    AUTOMATIC = auto()
    # Update metadata on the item
    UPDATE = auto()
    # Refresh item, after the item was read again from its source: a track without a file
    # takes its recording's work ids as they are now, also none
    REBUILD = auto()


# the performing roles a credit and a MusicBrainz relation may hold to name the same performer
_ROLE_FAMILIES: tuple[frozenset[ArtistRole], ...] = (
    frozenset({ArtistRole.CONDUCTOR}),
    frozenset({ArtistRole.ORCHESTRA, ArtistRole.ENSEMBLE, ArtistRole.CHOIR}),
    frozenset({ArtistRole.SOLOIST, ArtistRole.PERFORMER}),
)


class PerformerEnrichmentMixin:
    """
    MusicBrainz performer enrichment for the MetaDataController.

    Expects to be mixed with a class providing ``mass``, ``logger``, ``_logged_hints``,
    ``schedule_update_metadata``, ``_musicbrainz_provider``, ``_follow_merged_id``,
    ``_match_library_artist``, ``_add_musicbrainz_artist``, ``_add_track_work``,
    ``_musicbrainz_work_hierarchy``, ``_move_movement_tracks``, ``_store_group_pieces`` and
    ``_number_movement_tracks``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _logged_hints: set[Hashable]

        def schedule_update_metadata(  # noqa: D102
            self, item: MediaItemType, force_refresh: bool = False
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

        async def _follow_merged_id(self, item: Album | Artist | Work, mbid: str) -> None: ...

        async def _match_library_artist(self, mb_artist: MusicBrainzArtist) -> Artist | None: ...

        async def _add_musicbrainz_artist(
            self, mb_artist: MusicBrainzArtist, album: Album | None = None
        ) -> Artist | None: ...

        async def _add_track_work(self, track_id: str, work_ids: Collection[str]) -> None: ...

        async def _musicbrainz_work_hierarchy(
            self, musicbrainz: MusicbrainzProvider, work: Work
        ) -> list[MusicBrainzWork]: ...

        async def _move_movement_tracks(
            self, work: Work, hierarchy: Sequence[MusicBrainzWork]
        ) -> None: ...

        async def _store_group_pieces(
            self, work_id: int, number: int, mb_group: MusicBrainzWork
        ) -> None: ...

        async def _number_movement_tracks(
            self, work: Work, track_ids: Collection[int] | None = None
        ) -> None: ...

    async def _link_album_performers(self, album: Album, reread: Reread | None = None) -> bool:
        """
        Read the recordings of an album's classical tracks, return whether it could.

        :param album: The library album.
        :param reread: Read its classical tracks again, all of them or, automatically, those
            read over the refresh interval ago: credit the performers each still misses and move
            the tracks whose recording performs other works.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return False
        if not album.mbid:
            return True
        album_tracks = await self.mass.music.albums.get_library_album_tracks(album.item_id)
        tracks = [track for track in album_tracks if _recording_due(track, reread)]
        if not tracks:
            return True
        try:
            release = await musicbrainz.get_release_details(album.mbid, with_performers=True)
        except InvalidDataError:
            recordings = {}
        else:
            recordings = release.recordings_by_id()
            # MusicBrainz answers a merged release's old id with the release it was merged into
            await self._follow_merged_id(album, release.id)
        for track in tracks:
            assert track.mbid is not None  # for type checking
            on_release = (recording := recordings.get(track.mbid.casefold())) is not None
            if not on_release:
                if musicbrainz.rate_limited:
                    # the tracks left are read on the next run
                    return False
                # a track not on the album's release is looked up on its own
                recording = await _get_recording(musicbrainz, track.mbid)
                if recording and _lacks_work(track, recording, on_release):
                    self._log_recording_without_work(track, recording)
            await self._apply_recording(track.item_id, recording, album, reread, on_release)
        # counted over the whole album, as a run stopped by a time-out reads only some tracks
        classical = [x for x in album_tracks if x.is_classical]
        without_work = sum(
            1
            for x in classical
            if x.mbid
            and x.mbid.casefold() in recordings
            and _lacks_work(x, recordings[x.mbid.casefold()], on_release=True)
        )
        if without_work:
            log_once(
                self.logger,
                self._logged_hints,
                ("release_works", album.mbid),
                logging.INFO,
                'Classical tracks of album "%s" are not linked to a work on MusicBrainz (%s of %s), '
                "link their recordings to their works: %s/release/%s/edit-relationships",
                album.name,
                without_work,
                len(classical),
                MUSICBRAINZ_WEB_URL,
                album.mbid,
            )
        return True

    async def _link_track_performers(self, track: Track, reread: Reread | None = None) -> bool:
        """
        Read the recording of a classical track, return whether it could.

        :param track: The library track.
        :param reread: Read it again, see :meth:`_link_album_performers`.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return False
        if track.mbid:
            recording = await _get_recording(musicbrainz, track.mbid)
            if recording and _lacks_work(track, recording, on_release=False):
                self._log_recording_without_work(track, recording)
            await self._apply_recording(track.item_id, recording, reread=reread)
        return True

    async def _apply_recording(
        self,
        track_id: str,
        recording: MusicBrainzRecording | None,
        album: Album | None = None,
        reread: Reread | None = None,
        on_release: bool = False,
    ) -> None:
        """
        Identify a track's credited performers on its recording and store its recording year.

        A classical track without a work gets the work of its file's MusicBrainz work id, or
        of its recording. A recording found by a name search alone gives the track nothing.

        :param track_id: Library id of the track.
        :param recording: The track's MusicBrainz recording, None when MusicBrainz lacks it.
        :param album: The album the track is read through, named when a performer is not added.
        :param reread: Read again: credit the performers the track still misses and move it when
            its recording performs other works now.
        :param on_release: The recording is on the release of the album the track is read through.
        """
        # read again, as identifying the album's earlier tracks may have merged its artists
        track = await self.mass.music.tracks.get_library_item(track_id)
        # a classical name search is composer, album and track title, which dozens of
        # performances share (35 releases are called "Symphonies Nos. 40 & 41"), so its
        # first hit may be another orchestra's recording, with other performers and year
        if recording and not _recording_trusted(track, on_release):
            recording = None
        work_ids = {value for id_type, value in track.external_ids if id_type == ExternalID.MB_WORK}
        changed = False
        if recording:
            if not work_ids:
                work_ids = set(recording.performance_work_ids)
            elif reread and (new_ids := self._new_work_ids(track, recording, reread)) is not None:
                work_ids, changed = new_ids, True
                await self.mass.music.tracks.set_external_ids(
                    track.item_id,
                    {
                        *(x for x in track.external_ids if x[0] != ExternalID.MB_WORK),
                        *((ExternalID.MB_WORK, work_id) for work_id in new_ids),
                    },
                )
            await self._store_recording_works(track, recording)
            # read again, as its work may have moved onto the work above and left the library
            track = await self.mass.music.tracks.get_library_item(track_id)
        # a refresh checks every placement, as it read the track's source again
        if track.work and (changed or reread == Reread.REBUILD):
            await self._place_track_again(track, work_ids, changed)
        await self._add_track_work(track.item_id, work_ids)
        if recording:
            # read again, as the work may have credited its composers, who are no performers
            track = await self.mass.music.tracks.get_library_item(track_id)
            await self._store_recording_pieces(track, recording)
            relations = recording.performing_relations
            composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
            track_credits = _credits_with_main_artists(track)
            # the performers no credit names, which a credit of another name may still be
            unnamed = [
                x
                for x in relations
                if not any(_names_artist(x, credit.artist.name) for credit in track_credits)
            ]
            # an artist identified on an earlier credit may have merged into another one
            identified: dict[str, Artist] = {}
            main_artists: dict[str, tuple[Artist, list[MusicBrainzRelation]]] = {}
            for credit in track_credits:
                credited = identified.get(credit.artist.item_id, credit.artist)
                naming = _relations_naming(credit, relations)
                if not naming and credit.artist.item_id not in composers:
                    naming = await self._relations_linking(credit, credited, unnamed)
                if not naming:
                    continue
                if not (artist := await self._identify_performer(credited, naming, reread)):
                    continue
                identified[credit.artist.item_id] = artist
                if credit.role == ArtistRole.MAIN_ARTIST:
                    main_artists[credit.artist.item_id] = (artist, naming)
            await self._credit_performing_main_artists(track, main_artists)
            await self._credit_recording_performers(track, relations, album, reread)
            if relations:
                await self._log_performers_named_otherwise(track, unnamed, identified)
            elif _lacks_performer_roles(track):
                log_once(
                    self.logger,
                    self._logged_hints,
                    ("recording_performers", recording.id),
                    logging.DEBUG,
                    'Recording of "%s" names no performers on MusicBrainz, add them: %s/recording/%s',
                    track.name,
                    MUSICBRAINZ_WEB_URL,
                    recording.id,
                )
        await self._mark_recording_read(int(track.item_id), recording)

    async def _store_recording_works(self, track: Track, recording: MusicBrainzRecording) -> None:
        """
        Give a track without MusicBrainz work ids those of the works its recording performs.

        When the track's library work is one of these works, such as an aria, the tracks
        performing it move onto the library work above it, such as the act, as its movements.

        :param track: The library track.
        :param recording: The track's MusicBrainz recording.
        """
        if not (work_ids := recording.performance_work_ids) or any(
            id_type == ExternalID.MB_WORK for id_type, _ in track.external_ids
        ):
            return
        await self.mass.music.tracks.set_external_ids(
            track.item_id,
            {*track.external_ids, *((ExternalID.MB_WORK, work_id) for work_id in work_ids)},
        )
        if (
            track.work
            and (work := await self.mass.music.works.get_library_item(track.work.item_id)).mbid
            in work_ids
            and (musicbrainz := self._musicbrainz_provider())
        ):
            await self._move_movement_tracks(
                work, await self._musicbrainz_work_hierarchy(musicbrainz, work)
            )

    def _new_work_ids(
        self, track: Track, recording: MusicBrainzRecording, reread: Reread
    ) -> set[str] | None:
        """
        Return the work ids a track with work ids takes from its recording read again, if new.

        The recording's works replace the track's when none of them is one of the track's. On
        a refresh, they replace them whenever they differ, also by none. A file's own work ids
        stay; the file and the recording naming different works is logged once.

        :param track: The library track, with its stored MusicBrainz work ids.
        :param recording: The track's reliably identified MusicBrainz recording.
        :param reread: Why the recording is read again.
        """
        stored = {
            value.casefold()
            for id_type, value in track.external_ids
            if id_type == ExternalID.MB_WORK
        }
        new_ids = set(recording.performance_work_ids)
        new = {x.casefold() for x in new_ids}
        if self._read_from_file(track):
            if new and not new & stored:
                self._log_file_work_differs(track, recording)
            return None
        if reread == Reread.REBUILD:
            return new_ids if new != stored else None
        return new_ids if new and not new & stored else None

    async def _place_track_again(self, track: Track, work_ids: set[str], replaced: bool) -> None:
        """
        Take a placed track off its work when its MusicBrainz work ids no longer name that work.

        The ids name the work by its own id or one of the movements it lists. A track read from
        a file stays on the work its tags name. A track whose work ids were replaced by none
        leaves its work too. A work left empty leaves the library. A track staying is numbered
        by the movement it performs.

        :param track: The library track, placed on a work.
        :param work_ids: The track's MusicBrainz work ids.
        :param replaced: Whether this read replaced the track's stored work ids.
        """
        assert track.work is not None  # for type checking
        work = await self.mass.music.works.get_library_item(track.work.item_id)
        ids = {x.casefold() for x in work_ids}
        if work.mbid and self._read_from_file(track):
            ids.add(work.mbid.casefold())
        named = {
            x.casefold() for x in (work.mbid, *(m.mbid for m in work.metadata.movements or [])) if x
        }
        if ids & named:
            await self._number_movement_tracks(work, [int(track.item_id)])
            return
        # a work not looked up yet lists no movements, so a track lifted onto it stays
        if ids and not work.metadata.movements and not work.metadata.last_musicbrainz_lookup:
            return
        if not ids and not replaced:
            return
        self.logger.debug('Taking "%s" off work "%s"', track.name, work.name)
        await self.mass.music.tracks.set_work(int(track.item_id), None)
        await self.mass.music.classification.update(track_ids=[int(track.item_id)])
        await self.mass.music.works.remove_if_empty(work.item_id)

    def _read_from_file(self, track: Track) -> bool:
        """Return whether a library track is read from a file or a server, whose tags may give ids."""
        return any(
            (
                provider := self.mass.get_provider(
                    x.provider_instance, True, provider_type=MusicProvider
                )
            )
            and not provider.is_streaming_provider
            for x in track.provider_mappings
        )

    async def _store_recording_pieces(self, track: Track, recording: MusicBrainzRecording) -> None:
        """
        Store the pieces of the groups a recording performs whole in the work of its track.

        A group is a work MusicBrainz splits into pieces, such as a scene of an act, that the
        track's work lists as one of its movements. Its pieces come with the recording when it
        is read through its release.

        :param track: The library track.
        :param recording: The track's MusicBrainz recording.
        """
        if not track.work or not (musicbrainz := self._musicbrainz_provider()):
            return
        work = await self.mass.music.works.get_library_item(track.work.item_id)
        for relation in recording.relations or ():
            group = relation.work
            if relation.type != "performance" or not group or not group.movements:
                continue
            if (number := await _group_number(musicbrainz, work, group)) is not None:
                await self._store_group_pieces(int(work.item_id), number, group)

    async def _relations_linking(
        self,
        credit: Credit,
        credited: Artist | ItemMapping,
        relations: Sequence[MusicBrainzRelation],
    ) -> list[MusicBrainzRelation]:
        """
        Return the performing relations whose artist is a credited artist by id or provider link.

        :param credit: The track credit, whose role selects the relations to consider.
        :param credited: The library artist of the credit.
        :param relations: The performing relations of the recording no credit names.
        """
        if not (candidates := _relations_of_role(credit, relations)):
            return []
        artist = await self.mass.music.artists.get_library_item(credited.item_id)
        if artist.mbid:
            return [x for x in candidates if x.artist and x.artist.id == artist.mbid]
        mapped = {
            (x.provider_domain, x.item_id)
            for x in artist.provider_mappings
            if x.provider_domain in MUSICBRAINZ_LINK_DOMAINS
        }
        if not mapped or not (musicbrainz := self._musicbrainz_provider()):
            return []
        linking: list[MusicBrainzRelation] = []
        for relation in candidates:
            assert relation.artist is not None  # for type checking
            if mapped & await self._provider_artists(musicbrainz, relation.artist.id):
                linking.append(relation)
        return linking

    async def _provider_artists(
        self, musicbrainz: MusicbrainzProvider, mbid: str
    ) -> set[tuple[str, str]]:
        """Return the provider artists MusicBrainz links an artist to, by domain and item id."""
        try:
            details = await musicbrainz.get_artist_details(mbid)
        except InvalidDataError:
            return set()
        mappings = await provider_mappings_from_urls(
            self.mass, relation_urls(details.relations), MediaType.ARTIST, set()
        )
        return {(x.provider_domain, x.item_id) for x in mappings}

    async def _identify_performer(
        self,
        credited: Artist | ItemMapping,
        relations: Sequence[MusicBrainzRelation],
        reread: Reread | None = None,
    ) -> Artist | None:
        """
        Identify a credited artist as the one artist the relations name, return it if so.

        :param credited: The library artist of the track credit.
        :param relations: The performing relations of the recording that may name it.
        :param reread: Why the recording is read again, if it is.
        """
        mbids = {relation.artist.id for relation in relations if relation.artist}
        if len(mbids) != 1:
            return None
        mbid = mbids.pop()
        artists = self.mass.music.artists
        artist = await artists.get_library_item(credited.item_id)
        if artist.mbid and artist.mbid != mbid:
            return None
        newly_identified = not artist.mbid
        if newly_identified:
            artist.mbid = mbid
            await artists.update_item_in_library(artist.item_id, artist)
        # the recording confirms the id, so the artists holding it are the same artist and the
        # oldest of them survives
        survivor_id = await artists.merge_library_artists_by_mbid(int(artist.item_id), mbid)
        survivor = await artists.get_library_item(survivor_id)
        # the refresh fetches what is keyed on the id, such as the artist's images and bio, which
        # the automatic re-read leaves to the artist's own refresh
        if newly_identified and reread != Reread.AUTOMATIC:
            self.schedule_update_metadata(survivor, force_refresh=True)
        return survivor

    async def _credit_performing_main_artists(
        self,
        track: Track,
        identified: dict[str, tuple[Artist, list[MusicBrainzRelation]]],
    ) -> None:
        """
        Credit the track artists of a track without performing credits with their roles.

        Once one of them is identified on the recording, the others are credited as performer.

        :param track: The library track.
        :param identified: The identified library artist and its performing relations, by
            credited main artist id.
        """
        if any(credit.role in PERFORMING_ROLES for credit in track.credits):
            return
        composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
        performers = [
            credit
            for credit in _credits_with_main_artists(track)
            if credit.role == ArtistRole.MAIN_ARTIST and credit.artist.item_id not in composers
        ]
        if not any(x.artist.item_id in identified for x in performers):
            return
        new_credits: list[Credit] = []
        for performer in performers:
            if performer.artist.item_id not in identified:
                # once the track has performing credits, a track artist without one would no
                # longer count as a performer
                new_credits.append(
                    Credit(
                        artist=performer.artist,
                        role=ArtistRole.PERFORMER,
                        position=performer.position,
                    )
                )
                continue
            artist, relations = identified[performer.artist.item_id]
            for relation in relations:
                assert relation.performing_role is not None  # for type checking
                new_credits.append(
                    Credit(
                        artist=ItemMapping.from_item(artist),
                        role=relation.performing_role,
                        instrument=relation.performing_instrument,
                        position=performer.position,
                    )
                )
        await self.mass.music.tracks.add_credits(int(track.item_id), new_credits)

    async def _credit_recording_performers(
        self,
        track: Track,
        relations: Sequence[MusicBrainzRelation],
        album: Album | None = None,
        reread: Reread | None = None,
    ) -> None:
        """
        Credit a track naming no artist but its composers with its recording's performers.

        :param track: The library track.
        :param relations: The performing relations of the track's recording.
        :param album: The album the track is read through, named when a performer is not added.
        :param reread: Also credit a track naming performers, with the ones it does not name yet.
        """
        composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
        track_credits = _credits_with_main_artists(track)
        performers = [
            credit
            for credit in track_credits
            if credit.role in PERFORMING_ROLES
            or (credit.role == ArtistRole.MAIN_ARTIST and credit.artist.item_id not in composers)
        ]
        if performers and not reread:
            return
        credited = {x.artist.item_id for x in performers}
        # the added performers follow the credits the track has
        start = max((x.position for x in track_credits), default=-1) + 1 if performers else 0
        new_credits: list[Credit] = []
        for position, relation in enumerate(relations, start):
            assert relation.artist is not None  # for type checking
            assert relation.performing_role is not None  # for type checking
            if any(_names_artist(relation, x.artist.name) for x in performers):
                continue
            if not (artist := await self._match_library_artist(relation.artist)):
                artist = await self._add_musicbrainz_artist(relation.artist, album)
            if artist and artist.item_id not in credited:
                new_credits.append(
                    Credit(
                        artist=ItemMapping.from_item(artist),
                        role=relation.performing_role,
                        instrument=relation.performing_instrument,
                        position=position,
                    )
                )
        if new_credits:
            await self.mass.music.tracks.add_credits(int(track.item_id), new_credits)

    async def _mark_recording_read(
        self, track_id: int, recording: MusicBrainzRecording | None
    ) -> None:
        """Store when a track's recording was read, and the year of a recording that was found."""
        # a read without a usable recording says nothing about the year, so it keeps the stored one
        await self.mass.music.tracks.mark_recording_read(
            track_id, recording.recording_year if recording else None, keep_year=recording is None
        )

    def _log_file_work_differs(self, track: Track, recording: MusicBrainzRecording) -> None:
        """Log that a file track names another work than MusicBrainz links its recording to."""
        mb_works = [
            x.work.title for x in recording.relations or [] if x.type == "performance" and x.work
        ]
        log_once(
            self.logger,
            self._logged_hints,
            ("file_work", track.uri),
            logging.INFO,
            'Track "%s" stays on work "%s" its file names, while MusicBrainz links its recording '
            'to "%s"; check the work tags of the file or the work link of the recording: '
            "%s/recording/%s",
            track.name,
            track.work.name if track.work else "",
            '", "'.join(mb_works),
            MUSICBRAINZ_WEB_URL,
            recording.id,
        )

    def _log_recording_without_work(self, track: Track, recording: MusicBrainzRecording) -> None:
        """Log that a classical track's recording is linked to no work on MusicBrainz."""
        log_once(
            self.logger,
            self._logged_hints,
            ("recording_work", recording.id),
            logging.DEBUG,
            'Classical track "%s" is not linked to a work on MusicBrainz, '
            "link its recording to its work: %s/recording/%s",
            track.name,
            MUSICBRAINZ_WEB_URL,
            recording.id,
        )

    async def _log_performers_named_otherwise(
        self,
        track: Track,
        unnamed: Sequence[MusicBrainzRelation],
        identified: dict[str, Artist],
    ) -> None:
        """
        Log the performers a track credits whose MusicBrainz artist goes by a fuller name.

        :param track: The library track.
        :param unnamed: The performing relations of its recording no credit names.
        :param identified: The library artists identified on the recording, by credited artist id.
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
        for credit in _credits_with_main_artists(track):
            if credit.artist.item_id in identified or credit.artist.item_id in composers:
                continue
            mb_artists = {
                x.artist.id: x.artist
                for x in _relations_of_role(credit, unnamed)
                if x.artist and is_named_by(credit.artist.name, x.artist)
            }
            if len(mb_artists) != 1:
                continue
            (mb_artist,) = mb_artists.values()
            artist = await self.mass.music.artists.get_library_item(credit.artist.item_id)
            if artist.mbid or not (urls := MusicbrainzProvider.reverse_lookup_urls(artist)):
                continue
            log_once(
                self.logger,
                self._logged_hints,
                ("performer_name", artist.uri),
                logging.DEBUG,
                'Performer "%s" of "%s" is probably MusicBrainz artist "%s", '
                "add %s to their MusicBrainz page: %s/artist/%s",
                artist.name,
                track.name,
                mb_artist.name,
                urls[0],
                MUSICBRAINZ_WEB_URL,
                mb_artist.id,
            )


async def _get_recording(
    musicbrainz: MusicbrainzProvider, recording_id: str
) -> MusicBrainzRecording | None:
    """Return a recording with its related artists, works and places, None when unknown."""
    try:
        return await musicbrainz.get_recording_relations(recording_id)
    except InvalidDataError:
        return None


async def _group_number(
    musicbrainz: MusicbrainzProvider, work: Work, group: MusicBrainzWork
) -> int | None:
    """
    Return the number a work lists a MusicBrainz group of pieces under, None when it does not.

    The work's own movements tell it, else its movements on MusicBrainz, as a work not looked
    up yet has no movements.

    :param musicbrainz: The MusicBrainz provider.
    :param work: The library work.
    :param group: The MusicBrainz work of the group, with its relations.
    """
    group_id = group.id.casefold()
    for movement in work.metadata.movements or []:
        if movement.parent is None and movement.mbid and movement.mbid.casefold() == group_id:
            return movement.number
    if not work.mbid or not any(
        x.work and x.work.id.casefold() == work.mbid.casefold() for x in group.parent_relations
    ):
        return None
    try:
        mb_work = await musicbrainz.get_work_details(work.mbid)
    except InvalidDataError:
        return None
    place = mb_work.part_movement(group.id) if mb_work else None
    return place.number if place else None


def _recording_trusted(track: Track, on_release: bool) -> bool:
    """
    Return whether a track's recording is identified reliably enough to give it a work.

    :param track: The library track.
    :param on_release: The recording is on the release of the album the track is read through.
    """
    return on_release or track.metadata.recording_source not in (None, RecordingSource.SEARCH)


def _lacks_work(track: Track, recording: MusicBrainzRecording, on_release: bool) -> bool:
    """
    Return whether a track's reliably identified recording is linked to no work on MusicBrainz.

    A track whose file names its work is not counted.

    :param track: The library track.
    :param recording: The track's MusicBrainz recording.
    :param on_release: The recording is on the release of the album the track is read through.
    """
    return (
        not recording.performance_work_ids
        and _recording_trusted(track, on_release)
        and not any(id_type == ExternalID.MB_WORK for id_type, _ in track.external_ids)
    )


def _credits_with_main_artists(track: Track) -> list[Credit]:
    """Return the credits of a track, its main artists first as main artist credits."""
    return [
        *(
            Credit(artist=x, role=ArtistRole.MAIN_ARTIST, position=position)
            for position, x in enumerate(track.artists)
        ),
        *track.credits,
    ]


def _lacks_performer_roles(track: Track) -> bool:
    """Return whether a track credits no artist in a performing role."""
    return not any(x.role in PERFORMING_ROLES for x in track.credits)


def _recording_due(track: Track, reread: Reread | None = None) -> bool:
    """
    Return whether a library track is a classical track whose recording is due a read.

    A recording read before is due again once stale while the track still has no performer credit.

    :param track: The library track.
    :param reread: Why the recordings are read again: by hand all are due, automatically those
        read over the refresh interval ago.
    """
    if not track.is_classical or track.mbid is None:
        return False
    if reread in (Reread.UPDATE, Reread.REBUILD):
        return True
    last_lookup = track.metadata.last_musicbrainz_recording_lookup
    stale = last_lookup is not None and last_lookup < time() - REFRESH_INTERVAL
    return (
        last_lookup is None
        or (stale and reread == Reread.AUTOMATIC)
        or (stale and not any(x.role in PERFORMING_ROLES for x in track.credits))
    )


def _relations_naming(
    credit: Credit, relations: Sequence[MusicBrainzRelation]
) -> list[MusicBrainzRelation]:
    """Return the performing relations that may name the artist of a track credit."""
    return [
        x for x in _relations_of_role(credit, relations) if _names_artist(x, credit.artist.name)
    ]


def _relations_of_role(
    credit: Credit, relations: Sequence[MusicBrainzRelation]
) -> list[MusicBrainzRelation]:
    """Return the performing relations whose role a track credit may hold."""
    if credit.role == ArtistRole.MAIN_ARTIST:
        return list(relations)
    if credit.role in PERFORMING_ROLES:
        family = next(x for x in _ROLE_FAMILIES if credit.role in x)
        return [x for x in relations if x.performing_role in family]
    return []


def _names_artist(relation: MusicBrainzRelation, name: str) -> bool:
    """Return whether a relation's artist goes by a name, its own, sorted or credited one."""
    if not relation.artist:
        return False
    return any(
        compare_strings(name, mb_name)
        for mb_name in (relation.artist.name, relation.artist.sort_name, relation.target_credit)
        if mb_name
    )
