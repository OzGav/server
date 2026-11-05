"""Tests for reading the performers and recording years of classical tracks from MusicBrainz."""

from __future__ import annotations

import asyncio
import json
import pathlib
from time import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from music_assistant_models.enums import ArtistRole, ExternalID
from music_assistant_models.media_items import (
    Album,
    Artist,
    Credit,
    MediaItemMetadata,
    ProviderMapping,
    Track,
    UniqueList,
    Work,
)

from music_assistant.constants import DB_TABLE_TRACKS
from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.constants import REFRESH_INTERVAL
from music_assistant.controllers.metadata.controller import (
    _albums_with_recordings_to_read_query,
    _tracks_with_recording_to_read_query,
)
from music_assistant.controllers.metadata.performer_enrichment import _recording_due
from music_assistant.helpers.credit_roles import PERFORMING_ROLES
from music_assistant.providers.musicbrainz.api_client import _FRESH_READS
from music_assistant.providers.musicbrainz.models import MusicBrainzRelease
from music_assistant.providers.musicbrainz.provider import MusicbrainzProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from music_assistant.mass import MusicAssistant

FIXTURES = pathlib.Path(__file__).parents[2] / "providers" / "musicbrainz" / "fixtures"
PROVIDER_INSTANCE = "performer_enrichment_instance"
BEST_CLASSICAL = "best_classical_album"
KENNEDY = "Concerto no. 1 in E, RV 269 “La primavera”: I. Allegro"
SCHINDLER = "Schindler\u2019s List Theme"
CARMINA = "Carmina Burana: Fortuna Imperatrix Mundi: O Fortuna"
GRIEG = "Piano Concerto in A minor, Op. 16: Allegro molto moderato (Opening)"
GIANNI_SCHICCHI = "O mio babbino caro (Gianni Schicchi)"
PIANO_CONCERTO = "Piano Concerto No. 1 in B-flat minor, Op. 23"
KENNEDY_MBID = "2171e02c-e6c5-4109-85b7-ce4f13576a36"
ECO_MBID = "8c936585-f1e5-4729-9402-98f319d265b0"
OTHER_MBID = "00000000-0000-0000-0000-000000000001"
ECO_SPOTIFY_ID = "4fwqPcLS4wnTVT0pbfJGyl"


# --------------------------------------------------------------------------- #
#  builders                                                                    #
# --------------------------------------------------------------------------- #


def _lookup(name: str) -> dict[str, Any]:
    """Return a real MusicBrainz lookup from the fixtures, trimmed to a few tracks."""
    return cast(
        "dict[str, Any]", json.loads((FIXTURES / "performers" / f"{name}.json").read_text())
    )


def _release(name: str) -> MusicBrainzRelease:
    """Return a real MusicBrainz release from the fixtures."""
    return MusicBrainzRelease.from_raw(_lookup(name))


def _recording_id(release: str, title: str) -> str:
    """Return the id of the recording of the given title on a fixture release."""
    return next(
        track.recording.id
        for medium in _release(release).media
        for track in medium.tracks
        if track.recording and track.recording.title == title
    )


def _controller(mass: MusicAssistant, lookups: dict[str, dict[str, Any]]) -> MetaDataController:
    """
    Return a MetaDataController on the library database with MusicBrainz answering from fixtures.

    :param lookups: The lookups MusicBrainz answers, by the start of their endpoint.
    """
    ctrl = MetaDataController.__new__(MetaDataController)
    ctrl.mass = mass
    ctrl.logger = Mock()
    with patch.object(MusicbrainzProvider, "__init__", lambda *_a, **_kw: None):
        musicbrainz = MusicbrainzProvider.__new__(MusicbrainzProvider)

    async def _answer(endpoint: str, **_kwargs: Any) -> Any:
        return next((value for key, value in lookups.items() if endpoint.startswith(key)), None)

    musicbrainz._api_client = MagicMock()
    musicbrainz._api_client.throttler.cooldown_remaining = 0
    musicbrainz._api_client.get_data = AsyncMock(side_effect=_answer)
    ctrl._musicbrainz_provider = Mock(return_value=musicbrainz)  # type: ignore[method-assign]
    ctrl.schedule_update_metadata = Mock()  # type: ignore[method-assign]
    return ctrl


def _requests(ctrl: MetaDataController, works: bool = True) -> list[str]:
    """Return the endpoints MusicBrainz was asked for, without the work lookups if not works."""
    musicbrainz = cast("MusicbrainzProvider", ctrl._musicbrainz_provider())
    get_data = cast("AsyncMock", musicbrainz._api_client.get_data)
    endpoints = [call.args[0] for call in get_data.await_args_list]
    return endpoints if works else [x for x in endpoints if not x.startswith("work/")]


def _refreshed(ctrl: MetaDataController) -> list[tuple[str, bool]]:
    """Return the names of the artists a metadata refresh was scheduled for, and if forced."""
    schedule = cast("Mock", ctrl.schedule_update_metadata)
    return [(call.args[0].name, call.kwargs["force_refresh"]) for call in schedule.call_args_list]


def _mapping(item_id: str) -> ProviderMapping:
    """Create a provider mapping of the test provider."""
    return ProviderMapping(
        item_id=item_id,
        provider_domain="performer_enrichment",
        provider_instance=PROVIDER_INSTANCE,
        in_library=True,
    )


def _artist(name: str, mbid: str | None = None) -> Artist:
    """Return a provider artist, the same library artist for the same name."""
    return Artist(
        item_id=name,
        provider=PROVIDER_INSTANCE,
        name=name,
        provider_mappings={_mapping(name)},
        external_ids={(ExternalID.MB_ARTIST, mbid)} if mbid else set(),
    )


def _album(release_id: str | None = None, name: str = "Album") -> Album:
    """Return a provider album, carrying the given MusicBrainz release id."""
    return Album(
        item_id=name,
        provider=PROVIDER_INSTANCE,
        name=name,
        provider_mappings={_mapping(name)},
        external_ids={(ExternalID.MB_ALBUM, release_id)} if release_id else set(),
    )


def _track(
    name: str,
    recording_id: str | None,
    artists: Sequence[str],
    credits_: Sequence[tuple[str, ArtistRole, str | None]] = (),
    album: Album | None = None,
    number: int = 1,
    classical: bool = True,
) -> Track:
    """
    Return a provider track the way a local files sync reads it.

    :param credits_: The non main artist credits, as (artist name, role, instrument).
    """
    return Track(
        item_id=name,
        provider=PROVIDER_INSTANCE,
        name=name,
        provider_mappings={_mapping(name)},
        artists=UniqueList([_artist(x) for x in artists]),
        credits=[
            Credit(artist=_artist(artist), role=role, instrument=instrument)
            for artist, role, instrument in credits_
        ],
        album=album,
        track_number=number,
        disc_number=1,
        metadata=MediaItemMetadata(classical_tag=classical),
        external_ids={(ExternalID.MB_RECORDING, recording_id)} if recording_id else set(),
    )


async def _add(mass: MusicAssistant, track: Track) -> Track:
    """Store a provider track in the library and return it."""
    return await mass.music.tracks.add_item_to_library(track)


async def _library_album(mass: MusicAssistant, track: Track) -> Album:
    """Return the library album of a library track."""
    assert track.album is not None
    return await mass.music.albums.get_library_item(track.album.item_id)


async def _credits(mass: MusicAssistant, track: Track) -> set[tuple[str, ArtistRole, str | None]]:
    """Return the main artists and credits of a library track as (artist name, role, instrument)."""
    stored = await mass.music.tracks.get_library_item(track.item_id)
    return {(x.name, ArtistRole.MAIN_ARTIST, None) for x in stored.artists} | {
        (x.artist.name, x.role, x.instrument) for x in stored.credits
    }


async def _mbid(mass: MusicAssistant, name: str) -> str | None:
    """Return the MusicBrainz id of the library artist of the given name."""
    artists = await mass.music.artists.get_library_items_by_query(
        extra_query_parts=["artists.name = :name"], extra_query_params={"name": name}
    )
    assert len(artists) == 1
    return artists[0].mbid


async def _read_album(ctrl: MetaDataController, track: Track) -> None:
    """Run the album performers step on the album of a library track."""
    assert await ctrl._link_album_performers(await _library_album(ctrl.mass, track))


async def _kennedy_album(
    mass: MusicAssistant,
    artists: Sequence[str] = ("Nigel Kennedy", "English Chamber Orchestra"),
    credits_: Sequence[tuple[str, ArtistRole, str | None]] = (
        ("Nigel Kennedy", ArtistRole.SOLOIST, "violin"),
        ("English Chamber Orchestra", ArtistRole.ORCHESTRA, None),
        ("Antonio Vivaldi", ArtistRole.COMPOSER, None),
    ),
) -> Track:
    """Store the Four Seasons track of the fixture release, credited as given."""
    release_id = _release(BEST_CLASSICAL).id
    return await _add(
        mass,
        _track(
            "Spring", _recording_id(BEST_CLASSICAL, KENNEDY), artists, credits_, _album(release_id)
        ),
    )


# --------------------------------------------------------------------------- #
#  identifying credited performers                                             #
# --------------------------------------------------------------------------- #


async def test_credited_performers_get_their_musicbrainz_ids(music_mass: MusicAssistant) -> None:
    """The performers a track credits get the ids of the recording's performers, once refreshed."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass)
    before = await _credits(music_mass, track)

    await _read_album(ctrl, track)

    assert await _mbid(music_mass, "Nigel Kennedy") == KENNEDY_MBID
    assert await _mbid(music_mass, "English Chamber Orchestra") == ECO_MBID
    # the composer is no performer of the recording
    assert await _mbid(music_mass, "Antonio Vivaldi") is None
    assert await _credits(music_mass, track) == before
    assert sorted(_refreshed(ctrl)) == [
        ("English Chamber Orchestra", True),
        ("Nigel Kennedy", True),
    ]


async def test_credit_of_another_role_family_is_not_identified(music_mass: MusicAssistant) -> None:
    """A violinist the tags credit as conductor is not taken for the recording's violinist."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    release_id = _release(BEST_CLASSICAL).id
    track = await _add(
        music_mass,
        _track(
            "Schindler",
            _recording_id(BEST_CLASSICAL, SCHINDLER),
            ["John Williams"],
            [("Tasmin Little", ArtistRole.CONDUCTOR, None)],
            _album(release_id),
        ),
    )

    await _read_album(ctrl, track)

    assert await _mbid(music_mass, "Tasmin Little") is None


async def test_credit_matches_the_name_a_relation_credits(music_mass: MusicAssistant) -> None:
    """An artist credited under the name a relation credits it as is identified."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    release_id = _release(BEST_CLASSICAL).id
    album = _album(release_id)
    schindler = await _add(
        music_mass,
        _track(
            "Schindler",
            _recording_id(BEST_CLASSICAL, SCHINDLER),
            ["Tasmin Little"],
            [("New World Philharmonic", ArtistRole.ORCHESTRA, None)],
            album,
        ),
    )
    await _add(
        music_mass,
        _track(
            "Tchaikovsky",
            _recording_id(BEST_CLASSICAL, PIANO_CONCERTO),
            ["Peter Donohoe"],
            [("Rudolf Barshai", ArtistRole.CONDUCTOR, None)],
            album,
            number=2,
        ),
    )

    await _read_album(ctrl, schindler)

    assert await _mbid(music_mass, "New World Philharmonic") is not None
    # MusicBrainz names him Рудольф Баршай and credits him as Rudolf Barshai
    assert await _mbid(music_mass, "Rudolf Barshai") is not None


async def test_artist_with_another_musicbrainz_id_is_left_alone(
    music_mass: MusicAssistant,
) -> None:
    """An artist already holding a different MusicBrainz id keeps it and is not refreshed."""
    await music_mass.music.artists.add_item_to_library(_artist("Nigel Kennedy", OTHER_MBID))
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass)

    await _read_album(ctrl, track)

    assert await _mbid(music_mass, "Nigel Kennedy") == OTHER_MBID
    assert _refreshed(ctrl) == [("English Chamber Orchestra", True)]


async def test_identified_artist_merges_into_the_oldest_with_its_id(
    music_mass: MusicAssistant,
) -> None:
    """A performer given an id another library artist holds merges into the oldest of them."""
    older = await music_mass.music.artists.add_item_to_library(_artist("N. Kennedy", KENNEDY_MBID))
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass)

    await _read_album(ctrl, track)

    assert ("N. Kennedy", ArtistRole.SOLOIST, "violin") in await _credits(music_mass, track)
    assert not await music_mass.music.artists.get_library_items_by_query(
        extra_query_parts=["artists.name = 'Nigel Kennedy'"]
    )
    survivor = await music_mass.music.artists.get_library_item(older.item_id)
    assert survivor.mbid == KENNEDY_MBID
    assert ("N. Kennedy", True) in _refreshed(ctrl)


async def test_track_with_performing_credits_keeps_them(music_mass: MusicAssistant) -> None:
    """Credits are never changed and performers the tags do not credit are never added."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    # the tags call him the conductor only, and leave out the orchestra
    track = await _kennedy_album(
        music_mass, ["Nigel Kennedy"], [("Nigel Kennedy", ArtistRole.CONDUCTOR, None)]
    )
    before = await _credits(music_mass, track)

    await _read_album(ctrl, track)

    assert await _credits(music_mass, track) == before
    assert await _mbid(music_mass, "Nigel Kennedy") == KENNEDY_MBID


@pytest.mark.parametrize("overwrite", [True, False])
async def test_roles_and_recording_year_come_back_after_a_file_is_read_again(
    music_mass: MusicAssistant, overwrite: bool
) -> None:
    """
    A re-read or moved file drops the roles read from MusicBrainz until the next run reads them.

    The recording year is kept meanwhile.
    """
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    file_track = _track(
        "O Fortuna",
        _recording_id(BEST_CLASSICAL, CARMINA),
        ["London Symphony Orchestra Chorus"],
        [("Carl Orff", ArtistRole.COMPOSER, None)],
        _album(_release(BEST_CLASSICAL).id),
    )
    track = await _add(music_mass, file_track)
    await _read_album(ctrl, track)
    upgraded = await _credits(music_mass, track)

    await music_mass.music.tracks.update_item_in_library(
        track.item_id, file_track, overwrite=overwrite
    )

    reread = await music_mass.music.tracks.get_library_item(track.item_id)
    assert ("London Symphony Orchestra Chorus", ArtistRole.CHOIR, None) in upgraded
    assert ("London Symphony Orchestra Chorus", ArtistRole.CHOIR, None) not in await _credits(
        music_mass, track
    )
    assert reread.metadata.recording_year == 1974
    assert reread.metadata.last_musicbrainz_recording_lookup is None
    assert await _albums_due(music_mass) == {reread.album.item_id if reread.album else ""}

    await _read_album(ctrl, track)

    assert await _credits(music_mass, track) == upgraded


async def test_track_saved_with_its_credits_is_not_read_again(music_mass: MusicAssistant) -> None:
    """Saving a library track with the credits it has keeps it read."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass, ["Nigel Kennedy"], [])
    await _read_album(ctrl, track)

    stored = await music_mass.music.tracks.get_library_item(track.item_id)
    await music_mass.music.tracks.update_item_in_library(stored.item_id, stored)

    assert await _albums_due(music_mass) == set()


async def test_track_of_a_merged_artist_is_read_again(music_mass: MusicAssistant) -> None:
    """A track read while crediting an unidentified artist gets its role once that one merges."""
    identified = await music_mass.music.artists.add_item_to_library(
        _artist("Nigel Kennedy", KENNEDY_MBID)
    )
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass, ["Kennedy"], [])
    await _read_album(ctrl, track)
    assert ("Kennedy", ArtistRole.MAIN_ARTIST, None) in await _credits(music_mass, track)
    unidentified = (await music_mass.music.tracks.get_library_item(track.item_id)).artists[0]

    await music_mass.music.artists.merge_library_items(identified.item_id, unidentified.item_id)

    assert await _albums_due(music_mass) == {(await _library_album(music_mass, track)).item_id}
    await _read_album(ctrl, track)
    assert ("Nigel Kennedy", ArtistRole.SOLOIST, "violin") in await _credits(music_mass, track)


# --------------------------------------------------------------------------- #
#  recording years                                                            #
# --------------------------------------------------------------------------- #


async def test_recording_year_is_stored_and_tracks_are_marked_read(
    music_mass: MusicAssistant,
) -> None:
    """Each track read gets its recording year when MusicBrainz dates it."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    album = _album(_release(BEST_CLASSICAL).id)
    dated = await _add(
        music_mass,
        _track(
            "O Fortuna",
            _recording_id(BEST_CLASSICAL, CARMINA),
            ["London Symphony Orchestra"],
            [],
            album,
        ),
    )
    undated = await _add(
        music_mass,
        _track(
            "Gianni Schicchi",
            _recording_id(BEST_CLASSICAL, GIANNI_SCHICCHI),
            ["Amanda Roocroft"],
            [],
            album,
            number=2,
        ),
    )

    await _read_album(ctrl, dated)

    for track, year in ((dated, 1974), (undated, None)):
        stored = await music_mass.music.tracks.get_library_item(track.item_id)
        assert stored.metadata.recording_year == year
        assert stored.metadata.last_musicbrainz_recording_lookup is not None


# --------------------------------------------------------------------------- #
#  performers matched by id or provider link                                   #
# --------------------------------------------------------------------------- #


async def test_artist_of_another_name_is_matched_by_its_provider_link(
    music_mass: MusicAssistant,
) -> None:
    """A track artist MusicBrainz links to the same provider artist gets its exact role."""
    ctrl = _controller(
        music_mass,
        {"release/": _lookup(BEST_CLASSICAL), f"artist/{ECO_MBID}": _eco_with_spotify_link()},
    )
    track = await _kennedy_album(
        music_mass,
        ["Nigel Kennedy", "ECO Players"],
        [("Antonio Vivaldi", ArtistRole.COMPOSER, None)],
    )
    await _map_to_spotify(music_mass, "ECO Players", ECO_SPOTIFY_ID)

    with patch.object(music_mass.music, "get_provider_instances", _spotify_instances()):
        await _read_album(ctrl, track)

    assert await _mbid(music_mass, "ECO Players") == ECO_MBID
    assert ("ECO Players", ArtistRole.ORCHESTRA, None) in await _credits(music_mass, track)
    # only the artist no name matched is looked up, the composer is never
    assert [x.split("?")[0] for x in _requests(ctrl) if x.startswith("artist/")] == [
        f"artist/{ECO_MBID}"
    ]


async def test_artist_linked_to_another_provider_artist_is_not_matched(
    music_mass: MusicAssistant,
) -> None:
    """A track artist MusicBrainz links to another provider artist gets no role or id."""
    ctrl = _controller(
        music_mass,
        {"release/": _lookup(BEST_CLASSICAL), f"artist/{ECO_MBID}": _eco_with_spotify_link()},
    )
    track = await _kennedy_album(
        music_mass,
        ["Nigel Kennedy", "ECO Players"],
        [("Antonio Vivaldi", ArtistRole.COMPOSER, None)],
    )
    await _map_to_spotify(music_mass, "ECO Players", "anotherSpotifyArtist0")

    with patch.object(music_mass.music, "get_provider_instances", _spotify_instances()):
        await _read_album(ctrl, track)

    assert await _mbid(music_mass, "ECO Players") is None
    assert not any(x[1] in PERFORMING_ROLES for x in await _credits(music_mass, track))


async def test_artist_holding_the_relations_id_is_matched_without_a_lookup(
    music_mass: MusicAssistant,
) -> None:
    """A track artist that already has the id a relation names is that performer."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    await music_mass.music.artists.add_item_to_library(_artist("ECO Players", ECO_MBID))
    track = await _kennedy_album(
        music_mass,
        ["Nigel Kennedy", "ECO Players"],
        [("Antonio Vivaldi", ArtistRole.COMPOSER, None)],
    )

    await _read_album(ctrl, track)

    assert ("ECO Players", ArtistRole.ORCHESTRA, None) in await _credits(music_mass, track)
    assert len(_requests(ctrl, works=False)) == 1


# --------------------------------------------------------------------------- #
#  lookups and selection                                                       #
# --------------------------------------------------------------------------- #


async def test_album_is_read_with_one_request(music_mass: MusicAssistant) -> None:
    """All classical tracks of an album are read off one release lookup, and only once."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    album = _album(_release(BEST_CLASSICAL).id)
    tracks = [
        await _add(
            music_mass,
            _track(
                title, _recording_id(BEST_CLASSICAL, title), [f"Artist {number}"], [], album, number
            ),
        )
        for number, title in enumerate((KENNEDY, SCHINDLER, CARMINA, GIANNI_SCHICCHI), start=1)
    ]
    track = tracks[0]
    # a track that is not classical is left alone
    await _add(
        music_mass,
        _track("Pop", _recording_id(BEST_CLASSICAL, PIANO_CONCERTO), ["Pop"], [], album, 5, False),
    )
    assert await _albums_due(music_mass) == {(await _library_album(music_mass, track)).item_id}

    await _read_album(ctrl, track)

    assert len(_requests(ctrl, works=False)) == 1
    assert _requests(ctrl)[0].startswith(f"release/{_release(BEST_CLASSICAL).id}?")
    assert await _albums_due(music_mass) == set()


async def test_track_missing_from_the_release_is_looked_up_on_its_own(
    music_mass: MusicAssistant,
) -> None:
    """A track whose recording is not on its album's release is looked up by its recording."""
    recording = _lookup("recording_o_mio_babbino_caro")
    ctrl = _controller(
        music_mass, {"release/": _lookup("2cellos"), f"recording/{recording['id']}": recording}
    )
    track = await _add(
        music_mass,
        _track(
            "Gianni Schicchi",
            recording["id"],
            ["Amanda Roocroft"],
            [],
            _album(_release("2cellos").id),
        ),
    )

    await _read_album(ctrl, track)

    assert _requests(ctrl)[1] == f"recording/{recording['id']}?inc=artist-rels+work-rels+place-rels"
    assert ("Amanda Roocroft", ArtistRole.SOLOIST, "soprano vocals") in await _credits(
        music_mass, track
    )


async def test_track_without_an_identified_album_is_read_on_its_own(
    music_mass: MusicAssistant,
) -> None:
    """A classical track on an album without a release id is read through the track step."""
    recording = _lookup("recording_o_mio_babbino_caro")
    ctrl = _controller(music_mass, {f"recording/{recording['id']}": recording})
    track = await _add(
        music_mass,
        _track("Gianni Schicchi", recording["id"], ["Amanda Roocroft"], [], _album()),
    )
    assert await _albums_due(music_mass) == set()
    assert await _tracks_due(music_mass) == {track.item_id}

    assert await ctrl._link_track_performers(track)

    assert len(_requests(ctrl, works=False)) == 1
    assert await _mbid(music_mass, "Amanda Roocroft") is not None
    assert await _tracks_due(music_mass) == set()


async def test_recording_musicbrainz_does_not_know_is_marked_read(
    music_mass: MusicAssistant,
) -> None:
    """A recording MusicBrainz does not know is not looked up again."""
    ctrl = _controller(music_mass, {})
    track = await _add(music_mass, _track("Unknown", OTHER_MBID, ["Somebody"], [], _album()))

    assert await ctrl._link_track_performers(track)

    assert await _tracks_due(music_mass) == set()


@pytest.mark.parametrize(
    ("days_ago", "performer", "due"),
    [(91, False, True), (91, True, False), (1, False, False)],
)
async def test_recording_without_performers_is_read_again_once_stale(
    music_mass: MusicAssistant, days_ago: int, performer: bool, due: bool
) -> None:
    """A recording read long ago is read again only while its track has no performer credit."""
    credits_ = [("Orchestra", ArtistRole.ORCHESTRA, None)] if performer else []
    track = await _add(music_mass, _track("Unknown", OTHER_MBID, ["Somebody"], credits_, _album()))
    await _set_recording_read(music_mass, track, days_ago)

    assert (await _tracks_due(music_mass) == {track.item_id}) is due
    assert _recording_due(await music_mass.music.tracks.get_library_item(track.item_id)) is due


# --------------------------------------------------------------------------- #
#  artists sharing a MusicBrainz id                                            #
# --------------------------------------------------------------------------- #


async def test_artists_sharing_a_confirmed_id_merge(music_mass: MusicAssistant) -> None:
    """Artists already holding the id a recording confirms merge into the oldest, unrefreshed."""
    older = await music_mass.music.artists.add_item_to_library(
        _artist("English Chamber Orchestra", ECO_MBID)
    )
    # an id MA's own artist step set by name, which merges nothing
    newer = await music_mass.music.artists.add_item_to_library(_artist("ECO"))
    newer.mbid = ECO_MBID
    await music_mass.music.artists.update_item_in_library(newer.item_id, newer)
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(
        music_mass,
        ["Nigel Kennedy", "English Chamber Orchestra"],
        [("English Chamber Orchestra", ArtistRole.ORCHESTRA, None)],
    )

    await _read_album(ctrl, track)

    assert [x.item_id for x in await _holders(music_mass, ECO_MBID)] == [older.item_id]
    assert ("English Chamber Orchestra", True) not in _refreshed(ctrl)


async def test_fallback_lookups_stop_while_musicbrainz_is_rate_limiting(
    music_mass: MusicAssistant,
) -> None:
    """Tracks missing from the release are left for the next run while MusicBrainz limits."""
    recording = _lookup("recording_o_mio_babbino_caro")
    ctrl = _controller(
        music_mass, {"release/": _lookup("2cellos"), f"recording/{recording['id']}": recording}
    )
    track = await _add(
        music_mass,
        _track(
            "Gianni Schicchi",
            recording["id"],
            ["Amanda Roocroft"],
            [],
            _album(_release("2cellos").id),
        ),
    )
    album = await _library_album(music_mass, track)

    with patch.object(
        MusicbrainzProvider, "rate_limited", new_callable=PropertyMock, return_value=True
    ):
        assert not await ctrl._link_album_performers(album)

    assert len(_requests(ctrl)) == 1
    assert await _albums_due(music_mass) == {album.item_id}


def _eco_with_spotify_link() -> dict[str, Any]:
    """Return a MusicBrainz artist lookup of the English Chamber Orchestra, linked to Spotify."""
    return {
        "id": ECO_MBID,
        "name": "English Chamber Orchestra",
        "sort-name": "English Chamber Orchestra",
        "type": "Orchestra",
        "relations": [
            {
                "type": "free streaming",
                "target-type": "url",
                "url": {"resource": f"https://open.spotify.com/artist/{ECO_SPOTIFY_ID}"},
            }
        ],
    }


def _spotify_instances() -> Mock:
    """Return a stand-in for the music provider instance lookup with Spotify loaded."""

    def _instances(domain: str, **_kwargs: Any) -> list[Mock]:
        return [Mock(available=True, instance_id="spotify")] if domain == "spotify" else []

    return Mock(side_effect=_instances)


async def _map_to_spotify(mass: MusicAssistant, name: str, spotify_id: str) -> None:
    """Map the library artist of the given name to a Spotify artist."""
    artists = await mass.music.artists.get_library_items_by_query(
        extra_query_parts=["artists.name = :name"], extra_query_params={"name": name}
    )
    await mass.music.artists.add_provider_mappings(
        artists[0].item_id,
        [
            ProviderMapping(
                item_id=spotify_id,
                provider_domain="spotify",
                provider_instance="spotify",
                in_library=False,
            )
        ],
    )


async def _stored_work(mass: MusicAssistant, track: Track) -> Work | None:
    """Return the library work of a library track, None without one."""
    stored = await mass.music.tracks.get_library_item(track.item_id)
    return await mass.music.works.get_library_item(stored.work.item_id) if stored.work else None


async def _holders(mass: MusicAssistant, mbid: str) -> list[Artist]:
    """Return the library artists carrying a MusicBrainz artist id."""
    return await mass.music.artists.get_library_items_by_external_id(
        mbid, ExternalID.MB_ARTIST, limit=None
    )


async def _albums_due(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library albums the album performers phase would select."""
    albums = await mass.music.albums.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_albums_with_recordings_to_read_query()],
        extra_query_params={"stale": int(time() - REFRESH_INTERVAL)},
    )
    return {x.item_id for x in albums}


async def _tracks_due(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library tracks the track performers phase would select."""
    tracks = await mass.music.tracks.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_tracks_with_recording_to_read_query()],
        extra_query_params={"stale": int(time() - REFRESH_INTERVAL)},
    )
    return {x.item_id for x in tracks}


async def _set_recording_read(mass: MusicAssistant, track: Track, days_ago: int) -> None:
    """Mark the recording of a library track read the given number of days ago."""
    await mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_TRACKS} SET metadata = json_set(COALESCE(metadata, '{{}}'), "
        "'$.last_musicbrainz_recording_lookup', :read) WHERE item_id = :track_id",
        {"read": int(time()) - days_ago * 86400, "track_id": int(track.item_id)},
    )


@pytest.mark.parametrize(
    ("by_hand", "rebuild", "force_refresh", "fresh_read"),
    [
        # the Update metadata button
        (True, False, True, True),
        # Refresh item
        (False, True, True, True),
        # a forced update the server starts, such as of a newly identified performer
        (False, False, True, False),
        (True, False, False, False),
    ],
)
@pytest.mark.parametrize("kind", ["album", "track"])
async def test_only_the_buttons_read_musicbrainz_fresh(
    music_mass: MusicAssistant,
    kind: str,
    by_hand: bool,
    rebuild: bool,
    force_refresh: bool,
    fresh_read: bool,
) -> None:
    """
    Update metadata and Refresh item read an album or track past MusicBrainz's cache.

    A forced update the server starts, and an update that is not forced, read from the cache.
    """
    ctrl = _controller(music_mass, {})
    ctrl.cache = music_mass.cache
    ctrl._throttler = MagicMock()
    track = await music_mass.music.tracks.get_library_item(
        (await _kennedy_album(music_mass)).item_id
    )
    item: Album | Track = await _library_album(music_mass, track) if kind == "album" else track
    fresh: list[bool] = []

    async def _update(*_args: Any, **_kwargs: Any) -> None:
        fresh.append(_reading_fresh())

    with patch.object(ctrl, f"_update_{kind}_metadata", AsyncMock(side_effect=_update)):
        if by_hand:
            await ctrl.update_metadata_by_hand(item, force_refresh=force_refresh)
        else:
            await ctrl.update_metadata(item, force_refresh=force_refresh, rebuild=rebuild)

    assert fresh == [fresh_read]


@pytest.mark.parametrize("classical", [True, False])
async def test_track_read_again_from_its_source_has_the_performers_it_names_now(
    music_mass: MusicAssistant, classical: bool
) -> None:
    """A classical track read again from a source naming no performers loses its performers."""
    credits_ = [
        ("Nigel Kennedy", ArtistRole.SOLOIST, "violin"),
        ("English Chamber Orchestra", ArtistRole.ORCHESTRA, None),
        ("Antonio Vivaldi", ArtistRole.COMPOSER, None),
    ]
    track = await _add(
        music_mass,
        _track("Spring", None, ["Nigel Kennedy"], credits_, _album(), classical=classical),
    )
    before = await _credits(music_mass, track)

    await music_mass.music.tracks.update_item_in_library(
        track.item_id,
        _track("Spring", None, ["Nigel Kennedy"], [], _album(), classical=classical),
        overwrite=True,
    )

    kept = {x for x in before if x[1] not in PERFORMING_ROLES}
    assert kept != before
    assert await _credits(music_mass, track) == (kept if classical else before)


def _reading_fresh() -> bool:
    """Return whether the running task reads MusicBrainz past its cache."""
    return (state := _FRESH_READS.get()) is not None and state[0] is asyncio.current_task()
