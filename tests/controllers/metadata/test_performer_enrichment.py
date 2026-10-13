"""Tests for reading the performers and recording years of classical tracks from MusicBrainz."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from music_assistant_models.enums import ArtistRole, ExternalID
from music_assistant_models.media_items import (
    Album,
    Artist,
    Credit,
    ProviderMapping,
    Track,
    UniqueList,
)

from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.controller import (
    _albums_with_recordings_to_read_query,
    _tracks_with_recording_to_read_query,
)
from music_assistant.controllers.metadata.performer_enrichment import _relations_naming
from music_assistant.controllers.music.classical import PERFORMING_ROLES
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


def _requests(ctrl: MetaDataController) -> list[str]:
    """Return the endpoints MusicBrainz was asked for."""
    musicbrainz = cast("MusicbrainzProvider", ctrl._musicbrainz_provider())
    get_data = cast("AsyncMock", musicbrainz._api_client.get_data)
    return [call.args[0] for call in get_data.await_args_list]


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
    assert len(_requests(ctrl)) == 1


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

    assert len(_requests(ctrl)) == 1
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

    assert len(_requests(ctrl)) == 1
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


async def test_link_run_reads_the_performers_after_the_other_phases() -> None:
    """The performer phases come last and read albums and tracks through their own steps."""
    ctrl = MetaDataController.__new__(MetaDataController)
    ctrl.mass = MagicMock()

    phases = ctrl._musicbrainz_link_phases({})

    assert [x.name for x in phases[-2:]] == ["album performers", "track performers"]
    assert phases[-2].link == ctrl._link_album_performers
    assert phases[-1].link == ctrl._link_track_performers


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


async def _work_ids(mass: MusicAssistant, track: Track) -> set[str]:
    """Return the MusicBrainz work ids a library track carries."""
    stored = await mass.music.tracks.get_library_item(track.item_id)
    return {value for id_type, value in stored.external_ids if id_type == ExternalID.MB_WORK}


async def _holders(mass: MusicAssistant, mbid: str) -> list[Artist]:
    """Return the library artists carrying a MusicBrainz artist id."""
    return await mass.music.artists.get_library_items_by_external_id(
        mbid, ExternalID.MB_ARTIST, limit=None
    )


async def _albums_due(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library albums the album performers phase would select now."""
    albums = await mass.music.albums.get_library_items_by_query(
        limit=0, extra_query_parts=[_albums_with_recordings_to_read_query()]
    )
    return {x.item_id for x in albums}


async def _tracks_due(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library tracks the track performers phase would select now."""
    tracks = await mass.music.tracks.get_library_items_by_query(
        limit=0, extra_query_parts=[_tracks_with_recording_to_read_query()]
    )
    return {x.item_id for x in tracks}
