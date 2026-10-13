"""Tests for the role-typed artist credits of library tracks and albums."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from music_assistant_models.enums import AlbumType, ArtistRole, ExternalID
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import (
    Album,
    Artist,
    Credit,
    ItemMapping,
    ProviderMapping,
    Track,
    UniqueList,
)

from music_assistant.constants import DB_TABLE_TRACK_CREDITS
from music_assistant.mass import MusicAssistant
from music_assistant.models.music_provider import MusicProvider

PROVIDER_INSTANCE = "credits_instance"
OTHER_PROVIDER_INSTANCE = "other_credits_instance"
# whether a credit is one the track held before an update or one the update brings
STORED = "stored"
NEW = "new"


@pytest.fixture(name="mass")
def mass_fixture(music_mass_module: MusicAssistant) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture."""
    return music_mass_module


@pytest.fixture(name="no_audio_analysis")
def no_audio_analysis_fixture(mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the audio analysis the database-only server runs none of."""
    monkeypatch.setattr(
        mass,
        "streams",
        Mock(audio_analysis=Mock(delete_audio_analysis=AsyncMock())),
        raising=False,
    )


def _mapping(provider_instance: str = PROVIDER_INSTANCE) -> ProviderMapping:
    """Create a provider mapping with a unique provider item id."""
    return ProviderMapping(
        item_id=uuid4().hex,
        provider_domain="credits",
        provider_instance=provider_instance,
        in_library=True,
    )


async def _add_artist(mass: MusicAssistant, name: str) -> Artist:
    """Store a library artist under a name unique to the calling test."""
    return await mass.music.artists.add_item_to_library(
        Artist(
            item_id="0",
            provider="library",
            name=f"{name} {uuid4().hex[:8]}",
            provider_mappings={_mapping()},
        )
    )


async def _add_album(
    mass: MusicAssistant,
    artists: list[Artist],
    name: str | None = None,
) -> Album:
    """Store a library album with the given main artists."""
    return await mass.music.albums.add_item_to_library(
        Album(
            item_id="0",
            provider="library",
            name=name or f"Album {uuid4().hex[:8]}",
            album_type=AlbumType.ALBUM,
            provider_mappings={_mapping()},
            artists=UniqueList(artists),
        )
    )


async def _add_track(
    mass: MusicAssistant,
    artists: list[Artist],
    item_credits: list[Credit] | None = None,
    name: str | None = None,
    album: Album | None = None,
) -> Track:
    """Store a library track with the given main artists and credits."""
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=name or f"Track {uuid4().hex[:8]}",
            provider_mappings={_mapping()},
            artists=UniqueList(artists),
            credits=item_credits or [],
            album=album,
            disc_number=1 if album else 0,
            track_number=1 if album else 0,
        )
    )


def _credit_rows(item: Track) -> list[tuple[str, ArtistRole, str | None, int]]:
    """Return the (artist id, role, instrument, position) of each credit, in order."""
    return [(x.artist.item_id, x.role, x.instrument, x.position) for x in item.credits]


def _artist_ids(item: Track | Album) -> list[str]:
    """Return the item ids of the (main) artists of an item, in order."""
    return [x.item_id for x in item.artists]


async def _add_spellings(mass: MusicAssistant) -> tuple[Artist, Artist]:
    """Store two library artists, oldest first, that hold the same MusicBrainz id."""
    mbid = str(uuid4())
    spellings = []
    for name in ("Edward Elgar", "Elgar"):
        artist = await _add_artist(mass, name)
        await mass.music.artists.set_external_ids(artist.item_id, {(ExternalID.MB_ARTIST, mbid)})
        spellings.append(await mass.music.artists.get_library_item(artist.item_id))
    return spellings[0], spellings[1]


def _credit_artist(artist: Artist, mbid: str | None = None) -> Artist:
    """Return the provider artist a credit names for the given library artist."""
    mapping = next(iter(artist.provider_mappings))
    credited = Artist(
        item_id=mapping.item_id,
        provider=mapping.provider_instance,
        name=artist.name,
        provider_mappings={mapping},
    )
    if mbid:
        credited.mbid = mbid
    return credited


async def test_track_credits_round_trip(mass: MusicAssistant) -> None:
    """
    Track credits are stored and returned with their role, instrument and position.

    The main artists are not repeated as credits, but keep a credit in another role.
    """
    main_1 = await _add_artist(mass, "Main One")
    main_2 = await _add_artist(mass, "Main Two")
    composer = await _add_artist(mass, "Composer")
    violinist = await _add_artist(mass, "Violinist")
    pianist = await _add_artist(mass, "Pianist")
    track = await _add_track(
        mass,
        [main_1, main_2],
        [
            Credit(artist=pianist, role=ArtistRole.SOLOIST, instrument="piano", position=1),
            Credit(
                artist=violinist,
                role=ArtistRole.SOLOIST,
                instrument="violin",
                position=0,
            ),
            Credit(artist=main_1, role=ArtistRole.CONDUCTOR),
            Credit(artist=composer, role=ArtistRole.COMPOSER),
        ],
    )

    stored = await mass.music.tracks.get_library_item(track.item_id)

    assert _artist_ids(stored) == [main_1.item_id, main_2.item_id]
    assert _credit_rows(stored) == [
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
        (main_1.item_id, ArtistRole.CONDUCTOR, None, 0),
        (violinist.item_id, ArtistRole.SOLOIST, "violin", 0),
        (pianist.item_id, ArtistRole.SOLOIST, "piano", 1),
    ]
    assert isinstance(stored.credits[0].artist, ItemMapping)
    assert stored.credits[0].artist.provider == "library"
    assert stored.credits[0].artist.name == composer.name
    assert stored.composers == [stored.credits[0].artist]


async def test_credit_of_a_provider_artist_adds_the_artist(
    mass: MusicAssistant,
) -> None:
    """A credit naming a provider artist resolves (and adds) it as a library artist."""
    main = await _add_artist(mass, "Main")
    provider_item_id = uuid4().hex
    lyricist = ItemMapping(
        media_type=main.media_type,
        item_id=provider_item_id,
        provider=PROVIDER_INSTANCE,
        name=f"Lyricist {provider_item_id[:8]}",
    )
    track = await _add_track(mass, [main], [Credit(artist=lyricist, role=ArtistRole.LYRICIST)])

    stored = await mass.music.tracks.get_library_item(track.item_id)
    stored_lyricist = stored.credits[0].artist
    assert (stored_lyricist.provider, stored_lyricist.name) == (
        "library",
        lyricist.name,
    )
    assert stored.credits[0].role == ArtistRole.LYRICIST
    assert await mass.music.artists.get_library_item(stored_lyricist.item_id)


async def test_credit_without_instrument_is_stored_once(mass: MusicAssistant) -> None:
    """The same credit without an instrument, given twice, is stored once at its first position."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    track = await _add_track(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])

    await mass.music.tracks.add_credits(
        int(track.item_id), [Credit(artist=composer, role=ArtistRole.COMPOSER, position=5)]
    )

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _credit_rows(stored) == [(composer.item_id, ArtistRole.COMPOSER, None, 0)]


async def test_artists_win_over_main_artist_credits(mass: MusicAssistant) -> None:
    """The main artists come from the artists, not from main artist credits."""
    main = await _add_artist(mass, "Main")
    other = await _add_artist(mass, "Other")
    track = await _add_track(mass, [main], [Credit(artist=other, role=ArtistRole.MAIN_ARTIST)])

    stored = await mass.music.tracks.get_library_item(track.item_id)

    assert _artist_ids(stored) == [main.item_id]
    assert not stored.credits


@pytest.mark.parametrize(
    ("stored", "update", "source", "overwrite", "classical", "expected"),
    [
        # a merging update from a provider the track is not on adds its credits
        (
            [ArtistRole.COMPOSER],
            [ArtistRole.CONDUCTOR],
            "other",
            False,
            False,
            {(STORED, ArtistRole.COMPOSER), (NEW, ArtistRole.CONDUCTOR)},
        ),
        # one from one of the two providers of a track adds them too
        (
            [ArtistRole.COMPOSER],
            [ArtistRole.CONDUCTOR],
            "shared",
            False,
            False,
            {(STORED, ArtistRole.COMPOSER), (NEW, ArtistRole.CONDUCTOR)},
        ),
        # one from the only provider of a track, like a renamed file, replaces them
        (
            [ArtistRole.COMPOSER, ArtistRole.LYRICIST],
            [ArtistRole.COMPOSER],
            "same",
            False,
            False,
            {(NEW, ArtistRole.COMPOSER)},
        ),
        # one from a streaming service, which names only some credits, adds them
        (
            [ArtistRole.CONDUCTOR],
            [ArtistRole.COMPOSER],
            "streaming",
            False,
            False,
            {(NEW, ArtistRole.COMPOSER), (STORED, ArtistRole.CONDUCTOR)},
        ),
        # an overwriting update replaces them, unless it brings none
        (
            [ArtistRole.COMPOSER],
            [ArtistRole.LYRICIST],
            "same",
            True,
            False,
            {(NEW, ArtistRole.LYRICIST)},
        ),
        ([ArtistRole.COMPOSER], [], "same", True, False, {(STORED, ArtistRole.COMPOSER)}),
        # a refreshed classical track read from its files gets the performers they name now,
        # one from a streaming service keeps a performer it no longer names
        (
            [ArtistRole.CONDUCTOR],
            [ArtistRole.SOLOIST],
            "same",
            True,
            True,
            {(NEW, ArtistRole.SOLOIST)},
        ),
        (
            [ArtistRole.CONDUCTOR],
            [ArtistRole.SOLOIST],
            "streaming",
            True,
            True,
            {(STORED, ArtistRole.CONDUCTOR), (NEW, ArtistRole.SOLOIST)},
        ),
    ],
)
async def test_update_adds_or_replaces_credits(
    mass: MusicAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stored: list[ArtistRole],
    update: list[ArtistRole],
    source: str,
    overwrite: bool,
    classical: bool,
    expected: set[tuple[str, ArtistRole]],
) -> None:
    """
    An update adds its credits to the stored ones, or replaces them when it names them all.

    An update from the only provider of a track names them all, unless it is a streaming
    service, which names only some. An overwriting update replaces them unless it brings
    none, and a streaming service keeps the stored performers. The main artists of a merging
    update are added, those of an overwriting update replace the stored ones.

    :param stored: The roles of the stored credits.
    :param update: The roles of the credits of the update.
    :param source: Where the update comes from: the only provider of the track ("same"), the
        only one being a streaming service ("streaming"), a provider the track is not on
        ("other"), or one of the two providers of the track ("shared").
    :param overwrite: Whether the update overwrites the track.
    :param classical: Whether the track is classical.
    :param expected: The (stored or new, role) of the credits the track ends up with.
    """
    artists = {
        (origin, role): await _add_artist(mass, f"{origin} {role.value}")
        for origin, roles in ((STORED, stored), (NEW, update))
        for role in roles
    }
    main = await _add_artist(mass, "Main")
    new_main = await _add_artist(mass, "New Main")
    track = await mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=f"Track {uuid4().hex[:8]}",
            provider_mappings={_mapping()},
            artists=UniqueList([main]),
            credits=[Credit(artist=artists[STORED, x], role=x) for x in stored],
            classical_tag=classical,
        )
    )
    assert track.is_classical == classical
    if source == "shared":
        await mass.music.tracks.add_provider_mapping(
            track.item_id, _mapping(OTHER_PROVIDER_INSTANCE)
        )
    streaming = Mock(spec=MusicProvider, is_streaming_provider=source == "streaming")
    monkeypatch.setattr(mass, "get_provider", Mock(return_value=streaming))
    item = await mass.music.tracks.get_library_item(track.item_id)
    item.provider_mappings = {
        _mapping(OTHER_PROVIDER_INSTANCE) if source == "other" else _mapping()
    }
    item.artists = UniqueList([new_main])
    item.credits = [Credit(artist=artists[NEW, x], role=x) for x in update]

    result = await mass.music.tracks.update_item_in_library(
        track.item_id, item, overwrite=overwrite
    )

    origins = {x.name: origin for (origin, _), x in artists.items()}
    assert {(origins[x.artist.name], x.role) for x in result.credits} == expected
    assert _artist_ids(result) == (
        [new_main.item_id] if overwrite else [main.item_id, new_main.item_id]
    )


async def _add_artist_with_two_spellings(mass: MusicAssistant) -> Artist:
    """Store a library artist mapped to two provider spellings and holding a Discogs id."""
    name = f"Pyotr Ilyich Tchaikovsky {uuid4().hex[:8]}"
    return await mass.music.artists.add_item_to_library(
        Artist(
            item_id="0",
            provider="library",
            name=name,
            sort_name=f"Tchaikovsky, Pyotr Ilyich {name[-8:]}",
            provider_mappings={_mapping(), _mapping()},
            external_ids={(ExternalID.DISCOGS, uuid4().hex)},
        )
    )


def _short_spelling(artist: Artist) -> Artist:
    """Return the provider artist of one spelling of the given artist, under a shorter name."""
    mapping = min(artist.provider_mappings, key=lambda x: x.item_id)
    return Artist(
        item_id=mapping.item_id,
        provider=mapping.provider_instance,
        name="Tchaikovsky",
        provider_mappings={mapping},
    )


async def _assert_artist_unchanged(mass: MusicAssistant, artist: Artist) -> None:
    """Assert that the stored library artist still matches the given one."""
    stored = await mass.music.artists.get_library_item(artist.item_id)
    assert stored.name == artist.name
    assert stored.sort_name == artist.sort_name
    assert stored.external_ids == artist.external_ids
    assert {x.item_id for x in stored.provider_mappings} == {
        x.item_id for x in artist.provider_mappings
    }


async def test_track_update_with_overwrite_keeps_the_library_artist(
    mass: MusicAssistant,
) -> None:
    """An overwriting track update links its artists without rewriting the library artist."""
    artist = await _add_artist_with_two_spellings(mass)
    track = await _add_track(mass, [artist])
    spelling = _short_spelling(artist)
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.artists = UniqueList(
        [
            ItemMapping(
                media_type=spelling.media_type,
                item_id=spelling.item_id,
                provider=spelling.provider,
                name=spelling.name,
            )
        ]
    )
    update.credits = [Credit(artist=spelling, role=ArtistRole.COMPOSER)]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)

    await _assert_artist_unchanged(mass, artist)
    assert _artist_ids(stored) == [artist.item_id]
    assert _credit_rows(stored) == [
        (artist.item_id, ArtistRole.COMPOSER, None, 0),
    ]


async def test_album_update_with_overwrite_keeps_the_library_artist(
    mass: MusicAssistant,
) -> None:
    """An overwriting album update links its artists without rewriting the library artist."""
    artist = await _add_artist_with_two_spellings(mass)
    album = await _add_album(mass, [artist])
    update = await mass.music.albums.get_library_item(album.item_id)
    update.artists = UniqueList([_short_spelling(artist)])

    stored = await mass.music.albums.update_item_in_library(album.item_id, update, overwrite=True)

    await _assert_artist_unchanged(mass, artist)
    assert _artist_ids(stored) == [artist.item_id]


async def test_update_adds_a_new_mbid_to_the_library_artist(mass: MusicAssistant) -> None:
    """An update whose artist carries a new MusicBrainz id adds it to the library artist."""
    artist = await _add_artist_with_two_spellings(mass)
    track = await _add_track(mass, [artist])
    spelling = _short_spelling(artist)
    spelling.mbid = mbid = str(uuid4())
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.artists = UniqueList([spelling])

    await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)

    stored = await mass.music.artists.get_library_item(artist.item_id)
    assert stored.external_ids == {*artist.external_ids, (ExternalID.MB_ARTIST, mbid)}
    assert stored.name == artist.name
    assert stored.sort_name == artist.sort_name


@pytest.mark.parametrize("as_credit", [False, True])
async def test_new_spelling_joins_the_library_artist(mass: MusicAssistant, as_credit: bool) -> None:
    """
    A new spelling of a library artist, matched by MusicBrainz id, joins it as a mapping.

    That goes for a new spelling as the main artist of an album update and as a track credit.
    The library artist keeps its name, sort name and ids.
    """
    artist = await _add_artist_with_two_spellings(mass)
    mbid = str(uuid4())
    await mass.music.artists.set_external_ids(
        artist.item_id, {*artist.external_ids, (ExternalID.MB_ARTIST, mbid)}
    )
    stored_ids = (await mass.music.artists.get_library_item(artist.item_id)).external_ids
    new_mapping = _mapping()
    new_spelling = Artist(
        item_id=new_mapping.item_id,
        provider=new_mapping.provider_instance,
        name="Tchaikovsky",
        provider_mappings={new_mapping},
        external_ids={(ExternalID.DISCOGS, uuid4().hex)},
    )
    new_spelling.mbid = mbid

    if as_credit:
        main = await _add_artist(mass, "Main")
        track = await _add_track(
            mass, [main], [Credit(artist=new_spelling, role=ArtistRole.COMPOSER)]
        )
        stored = await mass.music.tracks.get_library_item(track.item_id)
        linked = [x.artist.item_id for x in stored.credits]
    else:
        album = await _add_album(mass, [artist])
        update = await mass.music.albums.get_library_item(album.item_id)
        update.artists = UniqueList([new_spelling])
        stored_album = await mass.music.albums.update_item_in_library(
            album.item_id, update, overwrite=True
        )
        linked = _artist_ids(stored_album)

    assert linked == [artist.item_id]
    stored_artist = await mass.music.artists.get_library_item(artist.item_id)
    assert {x.item_id for x in stored_artist.provider_mappings} == {
        *(x.item_id for x in artist.provider_mappings),
        new_mapping.item_id,
    }
    assert stored_artist.external_ids == stored_ids
    assert stored_artist.name == artist.name
    assert stored_artist.sort_name == artist.sort_name


async def test_summaries_carry_no_credits(mass: MusicAssistant) -> None:
    """Track and album listings show the main artists only and leave out the credits."""
    main = await _add_artist(mass, "Summary Main")
    composer = await _add_artist(mass, "Summary Composer")
    album = await _add_album(mass, [main])
    track = await _add_track(
        mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)], album=album
    )

    track_summary = (await mass.music.tracks.library_items(search=track.name))[0]
    album_summary = (await mass.music.albums.library_items(search=album.name))[0]

    for summary in (track_summary, album_summary):
        assert _artist_ids(summary) == [main.item_id]
    assert not track_summary.credits


async def test_merging_artists_keeps_credit_roles(mass: MusicAssistant) -> None:
    """Merging an artist moves its credits over with their role and instrument."""
    main = await _add_artist(mass, "Main")
    soloist = await _add_artist(mass, "Soloist")
    duplicate = await _add_artist(mass, "Soloist Duplicate")
    track = await _add_track(
        mass,
        [main],
        [
            Credit(
                artist=duplicate,
                role=ArtistRole.SOLOIST,
                instrument="cello",
                position=2,
            )
        ],
    )

    await mass.music.artists.merge_library_items(soloist.item_id, duplicate.item_id)

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _credit_rows(stored) == [
        (soloist.item_id, ArtistRole.SOLOIST, "cello", 2),
    ]


async def test_credit_merges_artists_sharing_its_mbid(mass: MusicAssistant) -> None:
    """A credit with an MBID joins the library artists holding that MBID into the oldest one."""
    main = await _add_artist(mass, "Main")
    oldest, spelling = await _add_spellings(mass)
    credited = _credit_artist(spelling, oldest.mbid)

    track = await _add_track(mass, [main], [Credit(artist=credited, role=ArtistRole.COMPOSER)])

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _credit_rows(stored) == [
        (oldest.item_id, ArtistRole.COMPOSER, None, 0),
    ]
    merged = await mass.music.artists.get_library_item(oldest.item_id)
    assert merged.name == oldest.name
    assert spelling.provider_mappings <= merged.provider_mappings
    with pytest.raises(MediaNotFoundError):
        await mass.music.artists.get_library_item(spelling.item_id)


async def test_credit_without_mbid_keeps_artists_apart(mass: MusicAssistant) -> None:
    """A credit without an MBID leaves library artists that share one alone."""
    main = await _add_artist(mass, "Main")
    oldest, spelling = await _add_spellings(mass)

    track = await _add_track(
        mass, [main], [Credit(artist=_credit_artist(spelling), role=ArtistRole.COMPOSER)]
    )

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _credit_rows(stored) == [
        (spelling.item_id, ArtistRole.COMPOSER, None, 0),
    ]
    assert await mass.music.artists.get_library_item(oldest.item_id)
    assert await mass.music.artists.get_library_item(spelling.item_id)


async def test_merging_tracks_keeps_credit_roles(mass: MusicAssistant) -> None:
    """Merging a track keeps the credits of both tracks with their role."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    conductor = await _add_artist(mass, "Conductor")
    target = await _add_track(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])
    source = await _add_track(mass, [main], [Credit(artist=conductor, role=ArtistRole.CONDUCTOR)])

    merged = await mass.music.tracks.merge_library_items(target.item_id, source.item_id)

    assert _artist_ids(merged) == [main.item_id]
    assert _credit_rows(merged) == [
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
        (conductor.item_id, ArtistRole.CONDUCTOR, None, 0),
    ]


@pytest.mark.usefixtures("no_audio_analysis")
async def test_removing_a_composer_keeps_their_tracks(mass: MusicAssistant) -> None:
    """Removing an artist only credited as composer keeps the credited tracks and their albums."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    composer_credit = Credit(artist=composer, role=ArtistRole.COMPOSER)
    album = await _add_album(mass, [main])
    track = await _add_track(mass, [main], [composer_credit], album=album)

    await mass.music.artists.remove_item_from_library(composer.item_id)

    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    stored_album = await mass.music.albums.get_library_item(album.item_id)
    assert _artist_ids(stored_track) == [main.item_id]
    assert _artist_ids(stored_album) == [main.item_id]
    assert not stored_track.credits
    assert not await mass.music.database.get_rows(
        DB_TABLE_TRACK_CREDITS, {"artist_id": int(composer.item_id)}
    )


@pytest.mark.usefixtures("no_audio_analysis")
async def test_removing_a_track_removes_its_credits(mass: MusicAssistant) -> None:
    """Removing a track removes its credits and keeps the credited artists."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    track = await _add_track(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])

    await mass.music.tracks.remove_item_from_library(track.item_id)

    assert not await mass.music.database.get_rows(
        DB_TABLE_TRACK_CREDITS, {"track_id": int(track.item_id)}
    )
    assert await mass.music.artists.get_library_item(composer.item_id)


async def test_artist_listings_ignore_other_credits(mass: MusicAssistant) -> None:
    """The tracks, albums and appears-on of an artist only count main artist credits."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    composer_credit = Credit(artist=composer, role=ArtistRole.COMPOSER)
    album = await _add_album(mass, [main])
    track = await _add_track(mass, [main], [composer_credit], album=album)
    artists_ctrl = mass.music.artists

    assert [x.item_id for x in await artists_ctrl.get_library_artist_tracks(main.item_id)] == [
        track.item_id
    ]
    assert [x.item_id for x in await artists_ctrl.get_library_artist_albums(main.item_id)] == [
        album.item_id
    ]
    assert not await artists_ctrl.get_library_artist_tracks(composer.item_id)
    assert not await artists_ctrl.get_library_artist_albums(composer.item_id)
    assert not await artists_ctrl.get_library_artist_appears_on(composer.item_id)
    album_artists = await artists_ctrl.library_items(
        search=composer.name, album_artists_only=True, limit=0
    )
    assert composer.item_id not in {x.item_id for x in album_artists}


async def test_artist_name_search_ignores_other_credits(mass: MusicAssistant) -> None:
    """An 'artist - title' search matches the main artists only."""
    main = await _add_artist(mass, "Searched Main")
    composer = await _add_artist(mass, "Searched Composer")
    composer_credit = Credit(artist=composer, role=ArtistRole.COMPOSER)
    title = f"Searched Title {uuid4().hex[:8]}"
    album = await _add_album(mass, [main], name=title)
    track = await _add_track(mass, [main], [composer_credit], name=title, album=album)

    for ctrl, item in ((mass.music.tracks, track), (mass.music.albums, album)):
        by_main = await ctrl.library_items(search=f"{main.name} - {title}")
        by_composer = await ctrl.library_items(search=f"{composer.name} - {title}")
        assert [x.item_id for x in by_main] == [item.item_id]
        assert not by_composer


async def test_artist_name_sort_ignores_other_credits(mass: MusicAssistant) -> None:
    """Sorting on artist name sorts on the main artists only."""
    title = f"Sorted Title {uuid4().hex[:8]}"
    # the composer is stored first, so an unfiltered sort would likely pick it up
    composer_credit = Credit(artist=await _add_artist(mass, "Zoe"), role=ArtistRole.COMPOSER)
    abe = await _add_artist(mass, "Abe")
    lou = await _add_artist(mass, "Lou")
    abe_album = await _add_album(mass, [abe], name=f"{title} 1")
    lou_album = await _add_album(mass, [lou], name=f"{title} 2")
    abe_track = await _add_track(mass, [abe], [composer_credit], name=f"{title} 1", album=abe_album)
    lou_track = await _add_track(mass, [lou], name=f"{title} 2", album=lou_album)

    tracks = await mass.music.tracks.library_items(search=title, order_by="track_artist_name")
    albums = await mass.music.albums.library_items(search=title, order_by="album_artist_name")

    assert [x.item_id for x in tracks] == [abe_track.item_id, lou_track.item_id]
    assert [x.item_id for x in albums] == [abe_album.item_id, lou_album.item_id]
