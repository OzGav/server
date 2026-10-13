"""
MusicBrainz performer enrichment for the Metadata Controller.

Provides the PerformerEnrichmentMixin, mixed into the MetaDataController, which reads the
MusicBrainz recordings of classical library tracks. The artists a track credits that the
recording names as its performers get their MusicBrainz id, and the track gets the year it
was recorded. A track whose only performers are its track artists gets their exact
performing roles. Credits are never renamed, changed or removed, and a performer the track
does not credit is never added.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Credit, ItemMapping

from music_assistant.constants import DB_TABLE_TRACKS
from music_assistant.controllers.music.classical import PERFORMING_ROLES
from music_assistant.helpers.compare import compare_strings

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence

    from music_assistant_models.media_items import Album, Artist, MediaItemType, Track

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz.models import (
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
    ``schedule_update_metadata`` and ``_musicbrainz_provider``.
    """

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger

        def schedule_update_metadata(  # noqa: D102
            self, item: MediaItemType, force_refresh: bool = False
        ) -> None: ...

        def _musicbrainz_provider(self) -> MusicbrainzProvider | None: ...

    async def _link_album_performers(self, album: Album) -> bool:
        """Read the recordings of an album's classical tracks, return whether it could."""
        if not (musicbrainz := self._musicbrainz_provider()):
            return False
        if not album.mbid:
            return True
        tracks = [
            track
            for track in await self.mass.music.albums.get_library_album_tracks(album.item_id)
            if _recording_due(track)
        ]
        if not tracks:
            return True
        try:
            release = await musicbrainz.get_release_details(album.mbid)
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
            await self._apply_recording(track.item_id, recording)
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

    async def _apply_recording(self, track_id: str, recording: MusicBrainzRecording | None) -> None:
        """Identify a track's credited performers on its recording and store its recording year."""
        # read again, as identifying the album's earlier tracks may have merged its artists
        track = await self.mass.music.tracks.get_library_item(track_id)
        if recording:
            relations = recording.performing_relations
            track_credits = _credits_with_main_artists(track)
            # an artist identified on an earlier credit may have merged into another one
            identified: dict[str, Artist] = {}
            main_artists: dict[str, tuple[Artist, list[MusicBrainzRelation]]] = {}
            for credit in track_credits:
                if not (naming := _relations_naming(credit, relations)):
                    continue
                credited = identified.get(credit.artist.item_id, credit.artist)
                if not (artist := await self._identify_performer(credited, naming)):
                    continue
                identified[credit.artist.item_id] = artist
                if credit.role == ArtistRole.MAIN_ARTIST:
                    main_artists[credit.artist.item_id] = (artist, naming)
            await self._credit_performing_main_artists(track, main_artists)
        await self._mark_recording_read(
            int(track.item_id), recording.recording_year if recording else None
        )

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
            for credit in _credits_with_main_artists(track)
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


def _credits_with_main_artists(track: Track) -> list[Credit]:
    """Return the credits of a track, its main artists first as main artist credits."""
    return [
        *(
            Credit(artist=x, role=ArtistRole.MAIN_ARTIST, position=position)
            for position, x in enumerate(track.artists)
        ),
        *track.credits,
    ]


def _recording_due(track: Track) -> bool:
    """Return whether a library track is a classical track whose recording is still to be read."""
    return (
        track.is_classical
        and track.mbid is not None
        and track.metadata.last_musicbrainz_recording_lookup is None
    )


def _relations_naming(
    credit: Credit, relations: Sequence[MusicBrainzRelation]
) -> list[MusicBrainzRelation]:
    """Return the performing relations that may name the artist of a track credit."""
    if credit.role == ArtistRole.MAIN_ARTIST:
        candidates = list(relations)
    elif credit.role in PERFORMING_ROLES:
        family = next(x for x in _ROLE_FAMILIES if credit.role in x)
        candidates = [x for x in relations if x.performing_role in family]
    else:
        return []
    return [x for x in candidates if _names_artist(x, credit.artist.name)]


def _names_artist(relation: MusicBrainzRelation, name: str) -> bool:
    """Return whether a relation's artist goes by a name, its own, sorted or credited one."""
    if not relation.artist:
        return False
    return any(
        compare_strings(name, mb_name)
        for mb_name in (relation.artist.name, relation.artist.sort_name, relation.target_credit)
        if mb_name
    )
