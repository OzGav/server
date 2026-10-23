"""Tests for reading the performers and recording years of classical tracks from MusicBrainz."""

from __future__ import annotations

import json
import logging
import pathlib
from time import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from music_assistant_models.enums import ArtistRole, ExternalID, RecordingSource
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import (
    Album,
    Artist,
    Credit,
    ItemMapping,
    ProviderMapping,
    Track,
    UniqueList,
    Work,
)

from music_assistant.constants import DB_TABLE_TRACKS
from music_assistant.controllers.cache import BYPASS_CACHE
from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.constants import REFRESH_INTERVAL
from music_assistant.controllers.metadata.controller import (
    _albums_with_recordings_to_read_query,
    _tracks_with_recording_to_read_query,
)
from music_assistant.controllers.metadata.performer_enrichment import (
    _recording_due,
    _relations_naming,
)
from music_assistant.controllers.music.classical import PERFORMING_ROLES
from music_assistant.controllers.music.helpers import fill_track_from_recording
from music_assistant.providers.musicbrainz.models import MusicBrainzRecording, MusicBrainzRelease
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
KENNEDY_WORK = "3f62d0d8-8ccd-31ab-bfca-768a1ae5708d"
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
    ctrl._logged_hints = set()
    with patch.object(MusicbrainzProvider, "__init__", lambda *_a, **_kw: None):
        musicbrainz = MusicbrainzProvider.__new__(MusicbrainzProvider)
    musicbrainz._logged_hints = set()

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


def _record_bypass(ctrl: MetaDataController) -> list[tuple[str, bool]]:
    """Return the endpoints MusicBrainz is asked for from now on, each with its cache bypass."""
    musicbrainz = cast("MusicbrainzProvider", ctrl._musicbrainz_provider())
    get_data = cast("AsyncMock", musicbrainz._api_client.get_data)
    answer = get_data.side_effect
    recorded: list[tuple[str, bool]] = []

    async def _record(endpoint: str, **kwargs: Any) -> Any:
        recorded.append((endpoint.split("?", maxsplit=1)[0], BYPASS_CACHE.get()))
        return await answer(endpoint, **kwargs)

    get_data.side_effect = _record
    return recorded


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
        classical_tag=classical,
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
    """Return the credits of a library track as (artist name, role, instrument)."""
    stored = await mass.music.tracks.get_library_item(track.item_id)
    return {(x.artist.name, x.role, x.instrument) for x in stored.credits}


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

    # the album lookup asks for the performers of its recordings
    assert _requests(ctrl)[0].endswith("+artist-rels+recording-level-rels+work-rels+place-rels")
    assert await _mbid(music_mass, "Nigel Kennedy") == KENNEDY_MBID
    assert await _mbid(music_mass, "English Chamber Orchestra") == ECO_MBID
    # the composer is no performer of the recording
    assert await _mbid(music_mass, "Antonio Vivaldi") is None
    assert await _credits(music_mass, track) == before
    assert sorted(_refreshed(ctrl)) == [
        ("English Chamber Orchestra", True),
        ("Nigel Kennedy", True),
    ]


@pytest.mark.parametrize(
    ("role", "instrument", "relation_types"),
    [
        (ArtistRole.SOLOIST, "violin", ["instrument"]),
        (ArtistRole.CONDUCTOR, None, ["conductor"]),
        (ArtistRole.ORCHESTRA, None, []),
    ],
)
def test_a_credit_matches_relations_of_its_role_family_only(
    role: ArtistRole, instrument: str | None, relation_types: list[str]
) -> None:
    """A soloist who also conducted is matched on the relation of the role the credit holds."""
    recording = next(
        track.recording
        for track in _release(BEST_CLASSICAL).media[0].tracks
        if track.recording and track.recording.title == KENNEDY
    )
    credit = Credit(artist=_artist("Nigel Kennedy"), role=role, instrument=instrument)

    assert [x.type for x in _relations_naming(credit, recording.performing_relations)] == (
        relation_types
    )


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


async def test_artist_already_identified_is_not_refreshed(music_mass: MusicAssistant) -> None:
    """An artist already holding the recording performer's id gets no extra refresh."""
    await music_mass.music.artists.add_item_to_library(_artist("Nigel Kennedy", KENNEDY_MBID))
    await music_mass.music.artists.add_item_to_library(
        _artist("English Chamber Orchestra", ECO_MBID)
    )
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass)

    await _read_album(ctrl, track)

    assert _refreshed(ctrl) == []


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


# --------------------------------------------------------------------------- #
#  track artists as performers                                                 #
# --------------------------------------------------------------------------- #


async def test_track_artist_without_performing_credits_gets_the_exact_role(
    music_mass: MusicAssistant,
) -> None:
    """A track that only names its performer as track artist gets that performer's exact role."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    release_id = _release(BEST_CLASSICAL).id
    track = await _add(
        music_mass,
        _track(
            "Grieg",
            _recording_id(BEST_CLASSICAL, GRIEG),
            ["Cécile Ousset"],
            [("Edvard Grieg", ArtistRole.COMPOSER, None)],
            _album(release_id),
        ),
    )

    await _read_album(ctrl, track)

    # "solo" tells how she played, not what, and the uncredited conductor and orchestra are
    # not added
    assert await _credits(music_mass, track) == {
        ("Cécile Ousset", ArtistRole.MAIN_ARTIST, None),
        ("Cécile Ousset", ArtistRole.SOLOIST, "piano"),
        ("Edvard Grieg", ArtistRole.COMPOSER, None),
    }


async def test_track_artist_takes_each_role_the_recording_names(
    music_mass: MusicAssistant,
) -> None:
    """A track artist who played and conducted gets both roles, its other performers none."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass, ["Nigel Kennedy"], [])

    await _read_album(ctrl, track)

    assert await _credits(music_mass, track) == {
        ("Nigel Kennedy", ArtistRole.MAIN_ARTIST, None),
        ("Nigel Kennedy", ArtistRole.CONDUCTOR, None),
        ("Nigel Kennedy", ArtistRole.SOLOIST, "violin"),
    }


async def test_composing_track_artist_is_no_performer(music_mass: MusicAssistant) -> None:
    """A track artist who composed the piece is left out, the performing one gets a role."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    release_id = _release(BEST_CLASSICAL).id
    track = await _add(
        music_mass,
        _track(
            "Gianni Schicchi",
            _recording_id(BEST_CLASSICAL, GIANNI_SCHICCHI),
            ["Giacomo Puccini", "Amanda Roocroft"],
            [("Giacomo Puccini", ArtistRole.COMPOSER, None)],
            _album(release_id),
        ),
    )

    await _read_album(ctrl, track)

    assert ("Amanda Roocroft", ArtistRole.SOLOIST, "soprano vocals") in await _credits(
        music_mass, track
    )


async def test_track_naming_only_its_composer_gets_the_recordings_performers(
    music_mass: MusicAssistant,
) -> None:
    """A track naming no artist but its composer is credited with the recording's performers."""
    ctrl = _controller(
        music_mass,
        {"release/": _lookup(BEST_CLASSICAL), f"artist/{ECO_MBID}": _eco_with_spotify_link()},
    )
    await music_mass.music.artists.add_item_to_library(_artist("Nigel Kennedy"))
    track = await _kennedy_album(
        music_mass, ["Antonio Vivaldi"], [("Antonio Vivaldi", ArtistRole.COMPOSER, None)]
    )

    with patch.object(music_mass.music, "get_provider_instances", _spotify_instances()):
        await _read_album(ctrl, track)

    # the library's own artist is taken by its name, the orchestra is added on its Spotify link
    assert await _credits(music_mass, track) == {
        ("Antonio Vivaldi", ArtistRole.MAIN_ARTIST, None),
        ("Antonio Vivaldi", ArtistRole.COMPOSER, None),
        ("Nigel Kennedy", ArtistRole.CONDUCTOR, None),
        ("Nigel Kennedy", ArtistRole.SOLOIST, "violin"),
        ("English Chamber Orchestra", ArtistRole.ORCHESTRA, None),
    }
    assert await _mbid(music_mass, "English Chamber Orchestra") == ECO_MBID


async def test_track_artist_its_new_work_credits_as_composer_gets_the_recordings_performers(
    music_mass: MusicAssistant,
) -> None:
    """A track artist the work the recording gives the track composed is no performer."""
    ctrl = _controller(
        music_mass,
        {"release/": _lookup(BEST_CLASSICAL), f"artist/{ECO_MBID}": _eco_with_spotify_link()},
    )
    vivaldi = await music_mass.music.artists.add_item_to_library(_artist("Antonio Vivaldi"))
    await music_mass.music.artists.add_item_to_library(_artist("Nigel Kennedy"))
    await music_mass.music.works.add_item_to_library(
        Work(
            item_id="La primavera",
            provider=PROVIDER_INSTANCE,
            name="La primavera",
            provider_mappings=set(),
            composers=UniqueList([vivaldi]),
            external_ids={(ExternalID.MB_WORK, KENNEDY_WORK)},
        )
    )
    # the file names its composer as its artist only
    track = await _kennedy_album(music_mass, ["Antonio Vivaldi"], [])

    with patch.object(music_mass.music, "get_provider_instances", _spotify_instances()):
        await _read_album(ctrl, track)

    assert {
        ("Antonio Vivaldi", ArtistRole.COMPOSER, None),
        ("Nigel Kennedy", ArtistRole.SOLOIST, "violin"),
        ("English Chamber Orchestra", ArtistRole.ORCHESTRA, None),
    } <= await _credits(music_mass, track)


async def test_performer_linked_to_no_music_provider_is_not_added(
    music_mass: MusicAssistant,
) -> None:
    """A performer neither in the library nor linked to a loaded music provider is left out."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(
        music_mass, ["Antonio Vivaldi"], [("Antonio Vivaldi", ArtistRole.COMPOSER, None)]
    )
    before = await _credits(music_mass, track)

    await _read_album(ctrl, track)

    assert await _credits(music_mass, track) == before
    assert not await music_mass.music.artists.get_library_items_by_query(
        extra_query_parts=["artists.name = :name"],
        extra_query_params={"name": "English Chamber Orchestra"},
    )


@pytest.mark.parametrize(
    ("release", "title", "artists"),
    [
        # MusicBrainz names the duo's members, not the duo
        ("2cellos", "Where the Streets Have No Name", ["2CELLOS"]),
        # one of the track artists is not on the recording
        (BEST_CLASSICAL, GIANNI_SCHICCHI, ["Amanda Roocroft", "Someone Else"]),
    ],
)
async def test_unmatched_track_artist_gets_no_roles_and_nobody_is_added(
    music_mass: MusicAssistant, release: str, title: str, artists: list[str]
) -> None:
    """Unless every track artist is on the recording, the track's credits stay as tagged."""
    ctrl = _controller(music_mass, {"release/": _lookup(release)})
    track = await _add(
        music_mass,
        _track("Track", _recording_id(release, title), artists, [], _album(_release(release).id)),
    )
    before = await _credits(music_mass, track)

    await _read_album(ctrl, track)

    assert await _credits(music_mass, track) == before


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


async def test_roles_and_recording_year_come_back_after_a_file_is_read_again(
    music_mass: MusicAssistant,
) -> None:
    """A re-read file drops the roles read from MusicBrainz until the next run reads them again."""
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

    await music_mass.music.tracks.update_item_in_library(track.item_id, file_track, overwrite=True)

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


async def test_moved_file_without_its_roles_is_read_again(music_mass: MusicAssistant) -> None:
    """A source replacing the credits without the roles read from MusicBrainz is read again."""
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

    await music_mass.music.tracks.update_item_in_library(track.item_id, file_track)

    assert await _albums_due(music_mass) == {(await _library_album(music_mass, track)).item_id}


# --------------------------------------------------------------------------- #
#  recording years                                                             #
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
#  recording works and performers matched by id or provider link               #
# --------------------------------------------------------------------------- #


async def test_track_gets_the_work_ids_of_its_recording(music_mass: MusicAssistant) -> None:
    """A track without MusicBrainz work ids takes those of the works its recording performs."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _kennedy_album(music_mass)

    await _read_album(ctrl, track)

    assert await _work_ids(music_mass, track) == {KENNEDY_WORK}


async def test_track_keeps_the_work_ids_it_has(music_mass: MusicAssistant) -> None:
    """The work ids a file's tags give a track are kept as they are."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    file_track = _track(
        "Spring",
        _recording_id(BEST_CLASSICAL, KENNEDY),
        ["Nigel Kennedy"],
        [],
        _album(_release(BEST_CLASSICAL).id),
    )
    file_track.external_ids.add((ExternalID.MB_WORK, OTHER_MBID))
    track = await _add(music_mass, file_track)

    await _read_album(ctrl, track)

    assert await _work_ids(music_mass, track) == {OTHER_MBID}


async def test_track_without_a_work_joins_the_library_work_of_its_recording(
    music_mass: MusicAssistant,
) -> None:
    """A track without a work joins the library work its recording on the release performs."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    work = await music_mass.music.works.add_item_to_library(
        Work(
            item_id="La primavera",
            provider=PROVIDER_INSTANCE,
            name="La primavera",
            provider_mappings=set(),
            external_ids={(ExternalID.MB_WORK, KENNEDY_WORK)},
        )
    )
    track = await _kennedy_album(music_mass)

    await _read_album(ctrl, track)

    stored = await _stored_work(music_mass, track)
    assert stored is not None
    assert stored.item_id == work.item_id
    assert len(_requests(ctrl)) == 1


async def test_track_without_a_work_gets_its_work_added_from_musicbrainz(
    music_mass: MusicAssistant,
) -> None:
    """The work of a track's recording is added from MusicBrainz and filled in right away."""
    ctrl = _controller(
        music_mass,
        {
            "release/": _lookup(BEST_CLASSICAL),
            f"work/{KENNEDY_WORK}": _work_lookup(KENNEDY_WORK, KENNEDY),
        },
    )
    track = await _kennedy_album(music_mass)

    await _read_album(ctrl, track)

    stored = await _stored_work(music_mass, track)
    assert stored is not None
    assert (stored.name, stored.mbid) == (KENNEDY, KENNEDY_WORK)
    assert [x.name for x in stored.composers] == ["Antonio Vivaldi"]
    assert stored.metadata.last_musicbrainz_lookup is not None


async def test_tracks_with_the_same_file_work_id_share_one_added_work(
    music_mass: MusicAssistant,
) -> None:
    """Tracks whose files name the same work id without a work name join one added work."""
    recording = _lookup("recording_o_mio_babbino_caro")
    ctrl = _controller(
        music_mass,
        {
            f"recording/{recording['id']}": recording,
            f"work/{OTHER_MBID}": _work_lookup(OTHER_MBID, "Gianni Schicchi"),
        },
    )
    tracks = []
    for number, recording_id in enumerate((recording["id"], KENNEDY_MBID), start=1):
        file_track = _track(f"Part {number}", recording_id, ["Amanda Roocroft"], [], _album())
        file_track.track_number = number
        file_track.external_ids.add((ExternalID.MB_WORK, OTHER_MBID))
        tracks.append(await _add(music_mass, file_track))

    for track in tracks:
        assert await ctrl._link_track_performers(track)

    works = [await _stored_work(music_mass, track) for track in tracks]
    assert works[0] is not None
    assert works[1] is not None
    assert works[0].item_id == works[1].item_id
    assert works[0].name == "Gianni Schicchi"
    holders = await music_mass.music.works.get_library_items_by_external_id(
        OTHER_MBID, ExternalID.MB_WORK, limit=None
    )
    assert len(holders) == 1


async def test_track_naming_several_works_gets_no_work(music_mass: MusicAssistant) -> None:
    """A track whose file names several works that are no parts of one work gets no work."""
    ctrl = _controller(music_mass, {"work/": _work_lookup(OTHER_MBID, "Gianni Schicchi")})
    file_track = _track("Part", OTHER_MBID, ["Amanda Roocroft"], [], _album())
    file_track.external_ids.update(
        {(ExternalID.MB_WORK, OTHER_MBID), (ExternalID.MB_WORK, ECO_MBID)}
    )
    track = await _add(music_mass, file_track)

    assert await ctrl._link_track_performers(track)

    assert await _stored_work(music_mass, track) is None


@pytest.mark.parametrize(
    ("sources", "trusted"),
    [
        ((RecordingSource.SEARCH,), False),
        ((None,), False),
        ((RecordingSource.SEARCH, RecordingSource.RELEASE), True),
    ],
)
async def test_only_a_reliably_found_recording_gives_work_and_performers(
    music_mass: MusicAssistant, sources: tuple[RecordingSource | None, ...], trusted: bool
) -> None:
    """A recording found by a name search alone, or of unknown source, gives the track nothing."""
    recording = _lookup("recording_o_mio_babbino_caro")
    ctrl = _controller(
        music_mass,
        {
            f"recording/{recording['id']}": recording,
            "work/": _work_lookup(OTHER_MBID, "Gianni Schicchi"),
        },
    )
    await music_mass.music.artists.add_item_to_library(_artist("Amanda Roocroft"))
    track = await _add(
        music_mass,
        _track(
            "Gianni Schicchi",
            None,
            ["Giacomo Puccini"],
            [("Giacomo Puccini", ArtistRole.COMPOSER, None)],
            _album(),
        ),
    )
    # each link of the track's recording id, as the MusicBrainz link step stores it
    for source in sources:
        library_track = await music_mass.music.tracks.get_library_item(track.item_id)
        fill_track_from_recording(library_track, MusicBrainzRecording.from_raw(recording), source)
        await music_mass.music.tracks.update_item_in_library(track.item_id, library_track)
    track = await music_mass.music.tracks.get_library_item(track.item_id)
    assert track.metadata.recording_source == sources[-1]

    assert await ctrl._link_track_performers(track)

    work_ids = {"433001da-5c32-3d5b-9794-5b3dfabb0457"} if trusted else set()
    assert await _work_ids(music_mass, track) == work_ids
    assert (await _stored_work(music_mass, track) is not None) == trusted
    credited = {name for name, _, _ in await _credits(music_mass, track)}
    assert ("Amanda Roocroft" in credited) == trusted
    stored = await music_mass.music.tracks.get_library_item(track.item_id)
    assert stored.metadata.last_musicbrainz_recording_lookup is not None


async def test_track_with_its_own_recording_id_gets_the_work_of_its_recording(
    music_mass: MusicAssistant,
) -> None:
    """A recording id a file's tags give, or AcoustID finds, is reliable enough for a work."""
    recording = _lookup("recording_o_mio_babbino_caro")
    ctrl = _controller(
        music_mass,
        {
            f"recording/{recording['id']}": recording,
            "work/": _work_lookup("433001da-5c32-3d5b-9794-5b3dfabb0457", "Gianni Schicchi"),
        },
    )
    tagged = await _add(
        music_mass, _track("Tagged", recording["id"], ["Amanda Roocroft"], [], _album())
    )
    fingerprinted = await _add(
        music_mass, _track("Fingerprinted", None, ["Amanda Roocroft"], [], _album(), 2)
    )
    await music_mass.music.tracks.set_identifiers(
        "Fingerprinted", PROVIDER_INSTANCE, mbid=KENNEDY_MBID
    )

    assert await ctrl._link_track_performers(tagged)

    sources = [
        (await music_mass.music.tracks.get_library_item(x.item_id)).metadata.recording_source
        for x in (tagged, fingerprinted)
    ]
    assert sources == [RecordingSource.TAG, RecordingSource.ACOUSTID]
    stored = await _stored_work(music_mass, tagged)
    assert stored is not None
    assert stored.mbid == "433001da-5c32-3d5b-9794-5b3dfabb0457"


async def test_track_that_is_not_classical_gets_no_work(music_mass: MusicAssistant) -> None:
    """A track that is not classical is never given a work."""
    ctrl = _controller(music_mass, {"work/": _work_lookup(KENNEDY_WORK, KENNEDY)})
    track = await _add(music_mass, _track("Pop", None, ["Pop"], [], _album(), classical=False))
    assert not track.is_classical

    await ctrl._add_track_work(track.item_id, {KENNEDY_WORK})

    assert await _stored_work(music_mass, track) is None
    assert _requests(ctrl) == []


async def test_track_with_a_work_keeps_it(music_mass: MusicAssistant) -> None:
    """A track with a work keeps it when its recording performs another one."""
    ctrl = _controller(
        music_mass,
        {
            "release/": _lookup(BEST_CLASSICAL),
            f"work/{KENNEDY_WORK}": _work_lookup(KENNEDY_WORK, KENNEDY),
        },
    )
    work = await music_mass.music.works.add_item_to_library(
        Work(item_id="Seasons", provider=PROVIDER_INSTANCE, name="Seasons", provider_mappings=set())
    )
    spring = _track(
        "Spring",
        _recording_id(BEST_CLASSICAL, KENNEDY),
        ["Nigel Kennedy"],
        [],
        _album(_release(BEST_CLASSICAL).id),
    )
    spring.work = ItemMapping.from_item(work)
    track = await _add(music_mass, spring)

    await _read_album(ctrl, track)

    stored = await _stored_work(music_mass, track)
    assert stored is not None
    assert stored.item_id == work.item_id
    assert not any(x.startswith("work/") for x in _requests(ctrl))


async def test_file_track_of_a_movement_moves_onto_its_parent_when_its_work_is_linked(
    music_mass: MusicAssistant,
) -> None:
    """A file track whose work is a movement by its tags becomes a movement of the work above."""
    concerto_id = "00000000-0000-0000-0000-000000000002"
    ctrl = _controller(
        music_mass,
        {
            f"work/{KENNEDY_WORK}": _part_lookup(KENNEDY_WORK, KENNEDY, concerto_id, 1),
            f"work/{concerto_id}": _work_lookup(
                concerto_id, "La primavera", [(KENNEDY_WORK, KENNEDY, 1)]
            ),
        },
    )
    concerto = await _add_work(music_mass, "La primavera", concerto_id)
    movement = await _add_work(music_mass, KENNEDY, KENNEDY_WORK, concerto)
    track = await _add_track_on(music_mass, movement)

    await ctrl._link_work_to_musicbrainz(movement)

    stored_track = await music_mass.music.tracks.get_library_item(track.item_id)
    assert stored_track.work is not None
    assert stored_track.work.item_id == concerto.item_id
    assert stored_track.movement_number == 1
    with pytest.raises(MediaNotFoundError):
        await music_mass.music.works.get_library_item(movement.item_id)


async def test_file_track_of_a_whole_act_becomes_the_act_of_its_opera(
    music_mass: MusicAssistant,
) -> None:
    """A file track whose work is an act by its tags shows as that act of the opera."""
    opera_id = "00000000-0000-0000-0000-000000000003"
    act_id = "00000000-0000-0000-0000-000000000004"
    ctrl = _controller(
        music_mass,
        {
            f"work/{act_id}": _part_lookup(act_id, "Opera: Act I", opera_id, 1),
            f"work/{opera_id}": _work_lookup(opera_id, "Opera", [(act_id, "Opera: Act I", 1)]),
        },
    )
    opera = await _add_work(music_mass, "Opera", opera_id)
    act = await _add_work(music_mass, "Opera: Act I", act_id, opera)
    track = await _add_track_on(music_mass, act)

    await ctrl._link_work_to_musicbrainz(act)

    stored_track = await music_mass.music.tracks.get_library_item(track.item_id)
    assert stored_track.work is not None
    assert (stored_track.work.item_id, stored_track.movement_number) == (opera.item_id, 1)


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


# TEMP(classical 5a catch-up): remove before release
async def test_read_tracks_without_work_ids_are_read_once_more(music_mass: MusicAssistant) -> None:
    """The tracks read before their work ids were kept are queued again, once."""
    ctrl = _controller(music_mass, {})
    without_ids = await _add(music_mass, _track("Plain", OTHER_MBID, ["Somebody"], [], _album()))
    tagged = _track("Tagged", KENNEDY_MBID, ["Somebody"], [], _album(), 2)
    tagged.external_ids.add((ExternalID.MB_WORK, KENNEDY_WORK))
    with_ids = await _add(music_mass, tagged)
    for track in (without_ids, with_ids):
        assert await ctrl._link_track_performers(track)
    assert await _tracks_due(music_mass) == set()

    await ctrl._queue_recording_works_catch_up()

    assert await _tracks_due(music_mass) == {without_ids.item_id}
    assert await ctrl._link_track_performers(without_ids)
    await ctrl._queue_recording_works_catch_up()
    assert await _tracks_due(music_mass) == set()


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
    await music_mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_TRACKS} SET metadata = json_set(COALESCE(metadata, '{{}}'), "
        "'$.last_musicbrainz_recording_lookup', :read) WHERE item_id = :track_id",
        {"read": int(time()) - days_ago * 86400, "track_id": int(track.item_id)},
    )

    assert (await _tracks_due(music_mass) == {track.item_id}) is due
    assert _recording_due(await music_mass.music.tracks.get_library_item(track.item_id)) is due


async def test_link_run_reads_the_performers_before_the_works() -> None:
    """The performer phases read albums and tracks through their own steps, before the works."""
    ctrl = MetaDataController.__new__(MetaDataController)
    ctrl.mass = MagicMock()

    phases = ctrl._musicbrainz_link_phases({})

    assert [x.name for x in phases[-4:]] == [
        "album performers",
        "track performers",
        "works",
        "composers",
    ]
    assert phases[-4].link == ctrl._link_album_performers
    assert phases[-3].link == ctrl._link_track_performers


# --------------------------------------------------------------------------- #
#  reading an album's performers again on a forced update                      #
# --------------------------------------------------------------------------- #


async def test_forced_album_update_credits_a_performer_skipped_before(
    music_mass: MusicAssistant,
) -> None:
    """A re-read adds the performers a track misses, with fresh data, and duplicates none."""
    ctrl = _controller(
        music_mass,
        {"release/": _lookup(BEST_CLASSICAL), f"artist/{ECO_MBID}": _eco_with_spotify_link()},
    )
    bypassed = _record_bypass(ctrl)
    # the tags credit the soloist only, and leave out the orchestra
    track = await _kennedy_album(
        music_mass,
        ["Nigel Kennedy"],
        [
            ("Nigel Kennedy", ArtistRole.SOLOIST, "violin"),
            ("Antonio Vivaldi", ArtistRole.COMPOSER, None),
        ],
    )
    before = await _credits(music_mass, track)
    album = await _library_album(music_mass, track)

    with patch.object(music_mass.music, "get_provider_instances", _spotify_instances()):
        await _read_album(ctrl, track)
        assert await _credits(music_mass, track) == before
        assert await ctrl._link_album_performers(album, reread=True)
        assert await ctrl._link_album_performers(album, reread=True)

    assert await _credits(music_mass, track) == {
        *before,
        ("English Chamber Orchestra", ArtistRole.ORCHESTRA, None),
    }
    stored = await music_mass.music.tracks.get_library_item(track.item_id)
    orchestra = next(x for x in stored.credits if x.role == ArtistRole.ORCHESTRA)
    assert orchestra.position > max(
        x.position for x in stored.credits if x.role != ArtistRole.ORCHESTRA
    )
    release = f"release/{_release(BEST_CLASSICAL).id}"
    assert [x for x in bypassed if not x[0].startswith("work/")] == [
        (release, False),
        (release, True),
        (f"artist/{ECO_MBID}", True),
        (release, True),
    ]


async def test_forced_update_of_an_album_without_classical_tracks_reads_nothing(
    music_mass: MusicAssistant,
) -> None:
    """Only an identified album with classical tracks is read again."""
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)})
    track = await _add(
        music_mass,
        _track(
            "Pop",
            _recording_id(BEST_CLASSICAL, KENNEDY),
            ["Pop"],
            [],
            _album(_release(BEST_CLASSICAL).id),
            classical=False,
        ),
    )

    assert await ctrl._link_album_performers(await _library_album(music_mass, track), reread=True)

    assert _requests(ctrl) == []


@pytest.mark.parametrize("force_refresh", [True, False])
async def test_album_metadata_update_reads_the_performers_again_when_forced(
    music_mass: MusicAssistant, force_refresh: bool
) -> None:
    """The Update metadata button reads an album's performers again, a scheduled update not."""
    ctrl = _controller(music_mass, {})
    ctrl.cache = music_mass.cache
    ctrl._throttler = MagicMock()
    track = await _kennedy_album(music_mass)
    album = await _library_album(music_mass, track)

    with (
        patch.object(ctrl, "_update_album_metadata", AsyncMock()),
        patch.object(ctrl, "_link_album_performers", AsyncMock(return_value=True)) as read,
    ):
        await ctrl.update_metadata(album, force_refresh=force_refresh)

    if force_refresh:
        read.assert_awaited_once_with(album, reread=True)
    else:
        read.assert_not_awaited()


async def test_performer_without_a_link_tells_to_update_the_album_once_added(
    music_mass: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A performer left out for want of a link names the album and how to read it again."""
    eco = {**_eco_with_spotify_link(), "relations": []}
    ctrl = _controller(music_mass, {"release/": _lookup(BEST_CLASSICAL), f"artist/{ECO_MBID}": eco})
    ctrl.logger = logging.getLogger("test_performer_enrichment")
    caplog.set_level(logging.INFO, logger="test_performer_enrichment")
    track = await _kennedy_album(
        music_mass, ["Antonio Vivaldi"], [("Antonio Vivaldi", ArtistRole.COMPOSER, None)]
    )

    def _instances(domain: str, **_kwargs: Any) -> list[Mock]:
        if domain != "tidal":
            return []
        instance = Mock(available=True, instance_id=domain)
        instance.manifest.name = "Tidal"
        return [instance]

    with patch.object(music_mass.music, "get_provider_instances", Mock(side_effect=_instances)):
        await _read_album(ctrl, track)

    assert (
        'Artist "English Chamber Orchestra" has no link to Tidal on MusicBrainz, so it is not '
        'added to album "Album"; add one at '
        f"https://musicbrainz.org/artist/{ECO_MBID}, then use Update metadata on the album"
    ) in caplog.messages


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


async def test_artist_identified_by_the_artist_step_does_not_merge(
    music_mass: MusicAssistant,
) -> None:
    """An id MA's own artist step finds by name merges no artists, being no reliable link."""
    ctrl = _controller(music_mass, {})
    older = await music_mass.music.artists.add_item_to_library(_artist("N. Kennedy", KENNEDY_MBID))
    newer = await music_mass.music.artists.add_item_to_library(_artist("Nigel Kennedy"))
    ctrl._get_artist_mbid = AsyncMock(return_value=(KENNEDY_MBID, True))  # type: ignore[method-assign]
    ctrl._link_artist_to_musicbrainz = AsyncMock(return_value=False)  # type: ignore[method-assign]

    await ctrl.link_item_to_musicbrainz(newer)

    assert {x.item_id for x in await _holders(music_mass, KENNEDY_MBID)} == {
        older.item_id,
        newer.item_id,
    }


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


# --------------------------------------------------------------------------- #
#  hints on what MusicBrainz lacks                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("linked", [False, True])
async def test_album_tracks_without_a_work_are_logged_once_per_album(
    music_mass: MusicAssistant, caplog: pytest.LogCaptureFixture, linked: bool
) -> None:
    """The classical tracks of an album whose recordings name no work are logged in one line."""
    lookup = _lookup(BEST_CLASSICAL) if linked else _without_relations(BEST_CLASSICAL, "work")
    ctrl = _hinting_controller(music_mass, {"release/": lookup}, caplog)
    track = await _kennedy_album(music_mass)
    album = await _library_album(music_mass, track)

    assert await ctrl._link_album_performers(album, reread=True)
    assert await ctrl._link_album_performers(album, reread=True)

    expected = (
        'Classical tracks of album "Album" are not linked to a work on MusicBrainz (1 of 1), '
        "link their recordings to their works: "
        f"https://musicbrainz.org/release/{lookup['id']}/edit-relationships"
    )
    assert _hints(caplog, "not linked to a work") == ([] if linked else [(logging.INFO, expected)])


async def test_album_hint_counts_the_tracks_read_in_earlier_runs(
    music_mass: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """The hint counts every classical track of the album, also those an earlier run read."""
    lookup = _lookup(BEST_CLASSICAL)
    for medium in lookup["media"]:
        for release_track in medium["tracks"]:
            recording = release_track["recording"]
            recording["relations"] = [
                x for x in recording["relations"] if x["target-type"] != "work"
            ]
    track = await _kennedy_album(music_mass)
    # an earlier run, stopped before it logged its hint, read the first track
    await _read_album(_controller(music_mass, {"release/": lookup}), track)
    await _add(
        music_mass,
        _track(
            "Gianni Schicchi",
            _recording_id(BEST_CLASSICAL, "O mio babbino caro (Gianni Schicchi)"),
            ["Amanda Roocroft"],
            album=_album(_release(BEST_CLASSICAL).id),
            number=2,
        ),
    )
    ctrl = _hinting_controller(music_mass, {"release/": lookup}, caplog)

    await _read_album(ctrl, track)

    assert [x for _, x in _hints(caplog, "not linked to a work")] == [
        'Classical tracks of album "Album" are not linked to a work on MusicBrainz (2 of 2), '
        "link their recordings to their works: "
        f"https://musicbrainz.org/release/{lookup['id']}/edit-relationships"
    ]


@pytest.mark.parametrize("linked", [False, True])
async def test_single_track_without_a_work_is_logged_once(
    music_mass: MusicAssistant, caplog: pytest.LogCaptureFixture, linked: bool
) -> None:
    """A classical track read on its own whose recording names no work is logged with it."""
    recording = _lookup("recording_o_mio_babbino_caro")
    if not linked:
        recording["relations"] = [x for x in recording["relations"] if x["target-type"] != "work"]
    ctrl = _hinting_controller(
        music_mass,
        {
            f"recording/{recording['id']}": recording,
            "work/": _work_lookup("433001da-5c32-3d5b-9794-5b3dfabb0457", "Gianni Schicchi"),
        },
        caplog,
    )
    track = await _add(
        music_mass, _track("Tagged", recording["id"], ["Amanda Roocroft"], [], _album())
    )

    assert await ctrl._link_track_performers(track)
    assert await ctrl._link_track_performers(track)

    expected = (
        'Classical track "Tagged" is not linked to a work on MusicBrainz, link its recording to '
        f"its work: https://musicbrainz.org/recording/{recording['id']}"
    )
    assert _hints(caplog, "not linked to a work") == ([] if linked else [(logging.DEBUG, expected)])


@pytest.mark.parametrize(
    "artists",
    [
        ("Nigel Kennedy", "English Chamber Orchestra"),
        # a track naming only its composer shows no performer at all
        ("Antonio Vivaldi",),
    ],
)
async def test_recording_without_performers_is_logged_once(
    music_mass: MusicAssistant, caplog: pytest.LogCaptureFixture, artists: tuple[str, ...]
) -> None:
    """A recording naming no performers is logged for a track without performer credits."""
    lookup = _without_relations(BEST_CLASSICAL, "artist")
    ctrl = _hinting_controller(music_mass, {"release/": lookup}, caplog)
    track = await _kennedy_album(
        music_mass, artists, [("Antonio Vivaldi", ArtistRole.COMPOSER, None)]
    )
    album = await _library_album(music_mass, track)

    assert await ctrl._link_album_performers(album, reread=True)
    assert await ctrl._link_album_performers(album, reread=True)

    recording_id = _recording_id(BEST_CLASSICAL, KENNEDY)
    expected = (
        'Recording of "Spring" names no performers on MusicBrainz, add them: '
        f"https://musicbrainz.org/recording/{recording_id}"
    )
    assert _hints(caplog, "names no performers") == [(logging.DEBUG, expected)]


@pytest.mark.parametrize("on_spotify", [True, False])
async def test_performer_under_a_fuller_musicbrainz_name_is_logged_once(
    music_mass: MusicAssistant, caplog: pytest.LogCaptureFixture, on_spotify: bool
) -> None:
    """A performer the recording names by a fuller name is logged with the link to add there."""
    ctrl = _hinting_controller(music_mass, {"release/": _lookup(BEST_CLASSICAL)}, caplog)
    track = await _kennedy_album(
        music_mass,
        ["Kennedy", "English Chamber Orchestra"],
        [("Antonio Vivaldi", ArtistRole.COMPOSER, None)],
    )
    if on_spotify:
        await _map_to_spotify(music_mass, "Kennedy", "kennedySpotifyId00000")
    album = await _library_album(music_mass, track)

    with patch.object(music_mass.music, "get_provider_instances", _spotify_instances()):
        assert await ctrl._link_album_performers(album)
        assert await ctrl._link_album_performers(album, reread=True)

    expected = (
        'Performer "Kennedy" of "Spring" is probably MusicBrainz artist "Nigel Kennedy", add '
        "https://open.spotify.com/artist/kennedySpotifyId00000 to their MusicBrainz page: "
        f"https://musicbrainz.org/artist/{KENNEDY_MBID}"
    )
    assert _hints(caplog, "is probably MusicBrainz artist") == (
        [(logging.DEBUG, expected)] if on_spotify else []
    )


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


def _hinting_controller(
    mass: MusicAssistant, lookups: dict[str, dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> MetaDataController:
    """Return a controller as _controller does, logging its hints to caplog."""
    ctrl = _controller(mass, lookups)
    ctrl.logger = logging.getLogger("test_performer_hints")
    caplog.set_level(logging.DEBUG, logger="test_performer_hints")
    return ctrl


def _hints(caplog: pytest.LogCaptureFixture, text: str) -> list[tuple[int, str]]:
    """Return the level and message of each logged record holding the given text."""
    return [(x.levelno, x.getMessage()) for x in caplog.records if text in x.getMessage()]


def _without_relations(release: str, target_type: str) -> dict[str, Any]:
    """Return a fixture release lookup whose Four Seasons recording lacks relations of a type."""
    lookup = _lookup(release)
    for medium in lookup["media"]:
        for track in medium["tracks"]:
            if track["recording"]["title"] == KENNEDY:
                relations = track["recording"]["relations"]
                track["recording"]["relations"] = [
                    x for x in relations if x["target-type"] != target_type
                ]
    return lookup


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


async def _work_ids(mass: MusicAssistant, track: Track) -> set[str]:
    """Return the MusicBrainz work ids a library track carries."""
    stored = await mass.music.tracks.get_library_item(track.item_id)
    return {value for id_type, value in stored.external_ids if id_type == ExternalID.MB_WORK}


def _work_lookup(
    mbid: str, title: str, parts: Sequence[tuple[str, str, int]] = ()
) -> dict[str, Any]:
    """Return a MusicBrainz work lookup of a work listing the given parts, by id, title and place."""
    relations = [
        {
            "type": "parts",
            "direction": "forward",
            "ordering-key": position,
            "work": {"id": part_id, "title": part_title},
        }
        for part_id, part_title, position in parts
    ]
    return {"id": mbid, "title": title, "relations": relations}


def _part_lookup(mbid: str, title: str, parent_id: str, position: int) -> dict[str, Any]:
    """Return a MusicBrainz work lookup of a work that is a part of the given work."""
    relation = {
        "type": "parts",
        "direction": "backward",
        "ordering-key": position,
        "work": {"id": parent_id, "title": ""},
    }
    return {"id": mbid, "title": title, "relations": [relation]}


async def _add_work(mass: MusicAssistant, name: str, mbid: str, parent: Work | None = None) -> Work:
    """Store a library work with a MusicBrainz id, below the given parent work."""
    return await mass.music.works.add_item_to_library(
        Work(
            item_id=name,
            provider=PROVIDER_INSTANCE,
            name=name,
            provider_mappings=set(),
            external_ids={(ExternalID.MB_WORK, mbid)},
            parent_work=ItemMapping.from_item(parent) if parent else None,
        )
    )


async def _add_track_on(mass: MusicAssistant, work: Work) -> Track:
    """Store the Four Seasons track of the fixture release on a work, the way a file names it."""
    spring = _track(
        "Spring",
        _recording_id(BEST_CLASSICAL, KENNEDY),
        ["Nigel Kennedy"],
        [("Antonio Vivaldi", ArtistRole.COMPOSER, None)],
        _album(_release(BEST_CLASSICAL).id),
    )
    spring.work = ItemMapping.from_item(work)
    return await _add(mass, spring)


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
    """Return the ids of the library albums the album performers phase would select now."""
    albums = await mass.music.albums.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_albums_with_recordings_to_read_query()],
        extra_query_params={"stale": int(time() - REFRESH_INTERVAL)},
    )
    return {x.item_id for x in albums}


async def _tracks_due(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library tracks the track performers phase would select now."""
    tracks = await mass.music.tracks.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_tracks_with_recording_to_read_query()],
        extra_query_params={"stale": int(time() - REFRESH_INTERVAL)},
    )
    return {x.item_id for x in tracks}
