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
links them to a loaded music provider.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Credit, ItemMapping

from music_assistant.constants import (
    DB_TABLE_EXTERNAL_ID_LOOKUP,
    DB_TABLE_TRACKS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.music.classical import PERFORMING_ROLES
from music_assistant.controllers.music.helpers import provider_mappings_from_urls
from music_assistant.helpers.compare import compare_strings
from music_assistant.providers.musicbrainz.provider import relation_urls

from .constants import MUSICBRAINZ_LINK_DOMAINS, REFRESH_INTERVAL

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence

    from music_assistant_models.media_items import Album, Artist, MediaItemType, Track

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz.models import (
        MusicBrainzArtist,
        MusicBrainzRecording,
        MusicBrainzRelation,
    )
    from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider

# the performing roles a credit and a MusicBrainz relation may hold to name the same performer
_ROLE_FAMILIES: tuple[frozenset[ArtistRole], ...] = (
    frozenset({ArtistRole.CONDUCTOR}),
    frozenset({ArtistRole.ORCHESTRA, ArtistRole.ENSEMBLE, ArtistRole.CHOIR}),
    frozenset({ArtistRole.SOLOIST, ArtistRole.PERFORMER}),
)


class PerformerEnrichmentMixin:
    """
    MusicBrainz performer enrichment for the MetaDataController.

    Expects to be mixed with a class providing ``mass``, ``logger``,
    ``schedule_update_metadata``, ``_musicbrainz_provider``, ``_match_library_artist`` and
    ``_add_musicbrainz_artist``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger

        def schedule_update_metadata(  # noqa: D102
            self, item: MediaItemType, force_refresh: bool = False
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

        async def _match_library_artist(self, mb_artist: MusicBrainzArtist) -> Artist | None: ...

        async def _add_musicbrainz_artist(
            self, mb_artist: MusicBrainzArtist, album: Album | None = None, fresh: bool = False
        ) -> Artist | None: ...

    async def _link_album_performers(self, album: Album, reread: bool = False) -> bool:
        """
        Read the recordings of an album's classical tracks, return whether it could.

        :param album: The library album.
        :param reread: Read all its classical tracks again with fresh MusicBrainz data and credit
            the performers each still misses.
        """
        if not (musicbrainz := self._musicbrainz_provider()):
            return False
        if not album.mbid:
            return True
        tracks = [
            track
            for track in await self.mass.music.albums.get_library_album_tracks(album.item_id)
            if _recording_due(track) or (reread and track.is_classical and track.mbid)
        ]
        if not tracks:
            return True
        try:
            async with self.mass.cache.handle_refresh(reread):
                release = await musicbrainz.get_release_details(album.mbid, with_performers=True)
        except InvalidDataError:
            recordings = {}
        else:
            recordings = release.recordings_by_id()
        for track in tracks:
            assert track.mbid is not None  # for type checking
            if not (recording := recordings.get(track.mbid.casefold())):
                if musicbrainz.rate_limited:
                    # the tracks left are read on the next run
                    return False
                # a track not on the album's release is looked up on its own
                recording = await _get_recording(musicbrainz, track.mbid)
            await self._apply_recording(track.item_id, recording, album, reread)
        return True

    async def _link_track_performers(self, track: Track) -> bool:
        """Read the recording of a classical track, return whether it could."""
        if not (musicbrainz := self._musicbrainz_provider()):
            return False
        if track.mbid:
            await self._apply_recording(
                track.item_id, await _get_recording(musicbrainz, track.mbid)
            )
        return True

    async def _apply_recording(
        self,
        track_id: str,
        recording: MusicBrainzRecording | None,
        album: Album | None = None,
        reread: bool = False,
    ) -> None:
        """
        Identify a track's credited performers on its recording and store its recording year.

        :param track_id: Library id of the track.
        :param recording: The track's MusicBrainz recording, None when MusicBrainz lacks it.
        :param album: The album the track is read through, named when a performer is not added.
        :param reread: Credit the performers the track still misses with fresh MusicBrainz data.
        """
        # read again, as identifying the album's earlier tracks may have merged its artists
        track = await self.mass.music.tracks.get_library_item(track_id)
        if recording:
            await self._store_recording_works(track, recording)
            relations = recording.performing_relations
            composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
            # the performers no credit names, which a credit of another name may still be
            unnamed = [
                x
                for x in relations
                if not any(_names_artist(x, credit.artist.name) for credit in track.credits)
            ]
            # an artist identified on an earlier credit may have merged into another one
            identified: dict[str, Artist] = {}
            main_artists: dict[str, tuple[Artist, list[MusicBrainzRelation]]] = {}
            for credit in track.credits:
                credited = identified.get(credit.artist.item_id, credit.artist)
                naming = _relations_naming(credit, relations)
                if not naming and credit.artist.item_id not in composers:
                    naming = await self._relations_linking(credit, credited, unnamed)
                if not naming:
                    continue
                if not (artist := await self._identify_performer(credited, naming)):
                    continue
                identified[credit.artist.item_id] = artist
                if credit.role == ArtistRole.MAIN_ARTIST:
                    main_artists[credit.artist.item_id] = (artist, naming)
            await self._credit_performing_main_artists(track, main_artists)
            await self._credit_recording_performers(track, relations, album, reread)
        await self._mark_recording_read(
            int(track.item_id), recording.recording_year if recording else None
        )

    async def _store_recording_works(self, track: Track, recording: MusicBrainzRecording) -> None:
        """
        Give a track without MusicBrainz work ids those of the works its recording performs.

        The track's library work, when MusicBrainz has not identified it, is looked up again
        on the next link run.

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
        await self.mass.music.database.execute_write(
            f"UPDATE {DB_TABLE_WORKS} "
            "SET metadata = json_remove(metadata, '$.last_musicbrainz_lookup') "
            f"WHERE item_id = (SELECT work_id FROM {DB_TABLE_TRACKS} WHERE item_id = :track_id) "
            "AND json_valid(metadata) AND NOT EXISTS ("
            f"SELECT 1 FROM {DB_TABLE_EXTERNAL_ID_LOOKUP} "
            f"WHERE {DB_TABLE_EXTERNAL_ID_LOOKUP}.media_type = :media_type "
            f"AND {DB_TABLE_EXTERNAL_ID_LOOKUP}.external_id_type = :id_type "
            f"AND {DB_TABLE_EXTERNAL_ID_LOOKUP}.item_id = {DB_TABLE_WORKS}.item_id)",
            {
                "track_id": int(track.item_id),
                "media_type": MediaType.WORK.value,
                "id_type": ExternalID.MB_WORK.value,
            },
        )

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
        self, credited: Artist | ItemMapping, relations: Sequence[MusicBrainzRelation]
    ) -> Artist | None:
        """Identify a credited artist as the one artist the relations name, return it if so."""
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
        if newly_identified:
            # the refresh fetches what is keyed on the id, such as the artist's images and bio
            self.schedule_update_metadata(survivor, force_refresh=True)
        return survivor

    async def _credit_performing_main_artists(
        self,
        track: Track,
        identified: dict[str, tuple[Artist, list[MusicBrainzRelation]]],
    ) -> None:
        """Credit the track artists of a track without performing credits with their roles."""
        if any(credit.role in PERFORMING_ROLES for credit in track.credits):
            return
        composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
        performers = [
            credit
            for credit in track.credits
            if credit.role == ArtistRole.MAIN_ARTIST and credit.artist.item_id not in composers
        ]
        # once credited, a track artist left without a role would no longer count as a performer
        if not performers or any(x.artist.item_id not in identified for x in performers):
            return
        new_credits: list[Credit] = []
        for performer in performers:
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
        reread: bool = False,
    ) -> None:
        """
        Credit a track naming no artist but its composers with its recording's performers.

        :param track: The library track.
        :param relations: The performing relations of the track's recording.
        :param album: The album the track is read through, named when a performer is not added.
        :param reread: Also credit a track naming performers, with the ones it does not name yet.
        """
        composers = {x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER}
        performers = [
            credit
            for credit in track.credits
            if credit.role in PERFORMING_ROLES
            or (credit.role == ArtistRole.MAIN_ARTIST and credit.artist.item_id not in composers)
        ]
        if performers and not reread:
            return
        credited = {x.artist.item_id for x in performers}
        # the added performers follow the credits the track has
        start = max((x.position for x in track.credits), default=-1) + 1 if performers else 0
        new_credits: list[Credit] = []
        for position, relation in enumerate(relations, start):
            assert relation.artist is not None  # for type checking
            assert relation.performing_role is not None  # for type checking
            if any(_names_artist(relation, x.artist.name) for x in performers):
                continue
            if not (artist := await self._match_library_artist(relation.artist)):
                artist = await self._add_musicbrainz_artist(relation.artist, album, reread)
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

    async def _mark_recording_read(self, track_id: int, recording_year: int | None) -> None:
        """Store a track's recording year and when its recording was read."""
        await self.mass.music.database.execute_write(
            f"UPDATE {DB_TABLE_TRACKS} SET metadata = json_set(COALESCE(metadata, '{{}}'), "
            "'$.last_musicbrainz_recording_lookup', :now, '$.recording_year', :year) "
            "WHERE item_id = :track_id",
            {"now": int(time()), "year": recording_year, "track_id": track_id},
        )


async def _get_recording(
    musicbrainz: MusicbrainzProvider, recording_id: str
) -> MusicBrainzRecording | None:
    """Return a recording with its related artists, works and places, None when unknown."""
    try:
        return await musicbrainz.get_recording_relations(recording_id)
    except InvalidDataError:
        return None


def _recording_due(track: Track) -> bool:
    """
    Return whether a library track is a classical track whose recording is due a read.

    A recording read before is due again once stale while the track still has no performer credit.
    """
    last_lookup = track.metadata.last_musicbrainz_recording_lookup
    return (
        track.is_classical
        and track.mbid is not None
        and (
            last_lookup is None
            or (
                last_lookup < time() - REFRESH_INTERVAL
                and not any(x.role in PERFORMING_ROLES for x in track.credits)
            )
        )
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
