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

from music_assistant.constants import DB_TABLE_TRACK_ARTISTS
from music_assistant.mass import MusicAssistant

PROVIDER_INSTANCE = "credits_instance"
OTHER_PROVIDER_INSTANCE = "other_credits_instance"


@pytest.fixture(name="mass")
def mass_fixture(music_mass_module: MusicAssistant) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture."""
    return music_mass_module


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
    item_credits: list[Credit] | None = None,
    name: str | None = None,
) -> Album:
    """Store a library album with the given main artists and credits."""
    return await mass.music.albums.add_item_to_library(
        Album(
            item_id="0",
            provider="library",
            name=name or f"Album {uuid4().hex[:8]}",
            album_type=AlbumType.ALBUM,
            provider_mappings={_mapping()},
            artists=UniqueList(artists),
            credits=item_credits or [],
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


def _credit_rows(item: Track | Album) -> list[tuple[str, ArtistRole, str | None, int]]:
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
    """Track credits are stored and returned with their role, instrument and position."""
    main_1 = await _add_artist(mass, "Main One")
    main_2 = await _add_artist(mass, "Main Two")
    composer = await _add_artist(mass, "Composer")
    conductor = await _add_artist(mass, "Conductor")
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
            Credit(artist=conductor, role=ArtistRole.CONDUCTOR),
            Credit(artist=composer, role=ArtistRole.COMPOSER),
        ],
    )

    stored = await mass.music.tracks.get_library_item(track.item_id)

    assert _artist_ids(stored) == [main_1.item_id, main_2.item_id]
    assert _credit_rows(stored) == [
        (main_1.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (main_2.item_id, ArtistRole.MAIN_ARTIST, None, 1),
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
        (conductor.item_id, ArtistRole.CONDUCTOR, None, 0),
        (violinist.item_id, ArtistRole.SOLOIST, "violin", 0),
        (pianist.item_id, ArtistRole.SOLOIST, "piano", 1),
    ]
    assert isinstance(stored.credits[0].artist, ItemMapping)
    assert stored.credits[0].artist.provider == "library"
    assert stored.credits[0].artist.name == main_1.name
    assert stored.composers == [stored.credits[2].artist]


async def test_album_credits_round_trip(mass: MusicAssistant) -> None:
    """Album credits are stored and returned next to the album artists."""
    main = await _add_artist(mass, "Album Main")
    composer = await _add_artist(mass, "Album Composer")
    album = await _add_album(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])

    stored = await mass.music.albums.get_library_item(album.item_id)

    assert _artist_ids(stored) == [main.item_id]
    assert _credit_rows(stored) == [
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
    ]


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
    stored_lyricist = stored.credits[1].artist
    assert (stored_lyricist.provider, stored_lyricist.name) == (
        "library",
        lyricist.name,
    )
    assert stored.credits[1].role == ArtistRole.LYRICIST
    assert await mass.music.artists.get_library_item(stored_lyricist.item_id)


async def test_artists_follow_main_artist_position(mass: MusicAssistant) -> None:
    """The artists of a track are its main artist credits, ordered by position."""
    main_1 = await _add_artist(mass, "First")
    main_2 = await _add_artist(mass, "Second")
    track = await _add_track(mass, [main_1, main_2])
    for artist_id, position in ((main_1.item_id, 1), (main_2.item_id, 0)):
        await mass.music.database.update(
            DB_TABLE_TRACK_ARTISTS,
            {"track_id": int(track.item_id), "artist_id": int(artist_id)},
            {"position": position},
        )

    stored = await mass.music.tracks.get_library_item(track.item_id)

    assert _artist_ids(stored) == [main_2.item_id, main_1.item_id]
    assert [x.artist.item_id for x in stored.credits if x.role == ArtistRole.MAIN_ARTIST] == [
        main_2.item_id,
        main_1.item_id,
    ]


async def test_artist_with_two_roles_is_listed_once_as_artist(
    mass: MusicAssistant,
) -> None:
    """An artist credited as main artist and conductor is one artist but two credits."""
    maestro = await _add_artist(mass, "Maestro")
    track = await _add_track(mass, [maestro], [Credit(artist=maestro, role=ArtistRole.CONDUCTOR)])

    stored = await mass.music.tracks.get_library_item(track.item_id)

    assert _artist_ids(stored) == [maestro.item_id]
    assert _credit_rows(stored) == [
        (maestro.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (maestro.item_id, ArtistRole.CONDUCTOR, None, 0),
    ]


async def test_artists_win_over_main_artist_credits(mass: MusicAssistant) -> None:
    """Main artist credits come from the artists, not from the given credits."""
    main = await _add_artist(mass, "Main")
    other = await _add_artist(mass, "Other")
    track = await _add_track(mass, [main], [Credit(artist=other, role=ArtistRole.MAIN_ARTIST)])

    stored = await mass.music.tracks.get_library_item(track.item_id)

    assert _artist_ids(stored) == [main.item_id]
    assert _credit_rows(stored) == [(main.item_id, ArtistRole.MAIN_ARTIST, None, 0)]


async def test_update_without_overwrite_adds_credits(mass: MusicAssistant) -> None:
    """A merging update from another provider keeps the stored credits and adds the new ones."""
    main_1 = await _add_artist(mass, "Main One")
    main_2 = await _add_artist(mass, "Main Two")
    composer_1 = await _add_artist(mass, "Composer One")
    composer_2 = await _add_artist(mass, "Composer Two")
    track = await _add_track(mass, [main_1], [Credit(artist=composer_1, role=ArtistRole.COMPOSER)])
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping(OTHER_PROVIDER_INSTANCE)}
    update.artists = UniqueList([main_2])
    update.credits = [
        Credit(artist=composer_1, role=ArtistRole.COMPOSER, position=5),
        Credit(artist=composer_2, role=ArtistRole.COMPOSER, position=1),
    ]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)

    assert _artist_ids(stored) == [main_1.item_id, main_2.item_id]
    assert _credit_rows(stored) == [
        (main_1.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (main_2.item_id, ArtistRole.MAIN_ARTIST, None, 1),
        (composer_1.item_id, ArtistRole.COMPOSER, None, 0),
        (composer_2.item_id, ArtistRole.COMPOSER, None, 1),
    ]


async def test_update_from_the_only_provider_replaces_credits(mass: MusicAssistant) -> None:
    """A merging update from the only provider of a track, like a renamed file, replaces them."""
    main = await _add_artist(mass, "Renamed Main")
    old_composer = await _add_artist(mass, "Old Composer")
    lyricist = await _add_artist(mass, "Old Lyricist")
    new_composer = await _add_artist(mass, "New Composer")
    track = await _add_track(
        mass,
        [main],
        [
            Credit(artist=old_composer, role=ArtistRole.COMPOSER),
            Credit(artist=lyricist, role=ArtistRole.LYRICIST),
        ],
    )
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping()}
    update.credits = [Credit(artist=new_composer, role=ArtistRole.COMPOSER)]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)

    assert _credit_rows(stored) == [
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (new_composer.item_id, ArtistRole.COMPOSER, None, 0),
    ]


async def test_update_of_a_track_on_two_providers_adds_credits(mass: MusicAssistant) -> None:
    """A merging update from one of the providers of a track keeps the stored credits."""
    main = await _add_artist(mass, "Shared Main")
    composer = await _add_artist(mass, "Shared Composer")
    conductor = await _add_artist(mass, "Shared Conductor")
    track = await _add_track(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])
    await mass.music.tracks.add_provider_mapping(track.item_id, _mapping(OTHER_PROVIDER_INSTANCE))
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping()}
    update.credits = [Credit(artist=conductor, role=ArtistRole.CONDUCTOR)]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)

    assert _credit_rows(stored) == [
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
        (conductor.item_id, ArtistRole.CONDUCTOR, None, 0),
    ]


async def test_update_with_overwrite_replaces_credits(mass: MusicAssistant) -> None:
    """An overwriting update replaces all credits of a track."""
    main_1 = await _add_artist(mass, "Main One")
    main_2 = await _add_artist(mass, "Main Two")
    composer = await _add_artist(mass, "Composer")
    lyricist = await _add_artist(mass, "Lyricist")
    track = await _add_track(mass, [main_1], [Credit(artist=composer, role=ArtistRole.COMPOSER)])
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.artists = UniqueList([main_2])
    update.credits = [Credit(artist=lyricist, role=ArtistRole.LYRICIST)]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)

    assert _artist_ids(stored) == [main_2.item_id]
    assert _credit_rows(stored) == [
        (main_2.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (lyricist.item_id, ArtistRole.LYRICIST, None, 0),
    ]


async def test_update_with_overwrite_without_credits_keeps_credits(mass: MusicAssistant) -> None:
    """An overwriting update that brings no credits keeps the stored ones."""
    main = await _add_artist(mass, "Keep Main")
    composer = await _add_artist(mass, "Keep Composer")
    track = await _add_track(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.credits = []

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)

    assert _credit_rows(stored) == [
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
    ]


async def test_album_update_with_overwrite_replaces_credits(
    mass: MusicAssistant,
) -> None:
    """An overwriting update replaces all credits of an album."""
    main_1 = await _add_artist(mass, "Album Main One")
    main_2 = await _add_artist(mass, "Album Main Two")
    composer = await _add_artist(mass, "Album Composer")
    conductor = await _add_artist(mass, "Album Conductor")
    album = await _add_album(mass, [main_1], [Credit(artist=composer, role=ArtistRole.COMPOSER)])
    update = await mass.music.albums.get_library_item(album.item_id)
    update.artists = UniqueList([main_2])
    update.credits = [Credit(artist=conductor, role=ArtistRole.CONDUCTOR)]

    stored = await mass.music.albums.update_item_in_library(album.item_id, update, overwrite=True)

    assert _artist_ids(stored) == [main_2.item_id]
    assert _credit_rows(stored) == [
        (main_2.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (conductor.item_id, ArtistRole.CONDUCTOR, None, 0),
    ]


async def test_rewriting_artists_keeps_other_credits(mass: MusicAssistant) -> None:
    """Replacing the artists of a track leaves its other credits in place."""
    main_1 = await _add_artist(mass, "Main One")
    main_2 = await _add_artist(mass, "Main Two")
    composer = await _add_artist(mass, "Composer")
    track = await _add_track(mass, [main_1], [Credit(artist=composer, role=ArtistRole.COMPOSER)])

    await mass.music.tracks._set_track_artists(int(track.item_id), [main_2], overwrite=True)

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _credit_rows(stored) == [
        (main_2.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
    ]


async def test_summaries_carry_no_credits(mass: MusicAssistant) -> None:
    """Track and album listings show the main artists only and leave out the credits."""
    main = await _add_artist(mass, "Summary Main")
    composer = await _add_artist(mass, "Summary Composer")
    album = await _add_album(mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)])
    track = await _add_track(
        mass, [main], [Credit(artist=composer, role=ArtistRole.COMPOSER)], album=album
    )

    track_summary = (await mass.music.tracks.library_items(search=track.name))[0]
    album_summary = (await mass.music.albums.library_items(search=album.name))[0]

    for summary in (track_summary, album_summary):
        assert _artist_ids(summary) == [main.item_id]
        assert not summary.credits


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
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
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
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
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
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
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
        (main.item_id, ArtistRole.MAIN_ARTIST, None, 0),
        (composer.item_id, ArtistRole.COMPOSER, None, 0),
        (conductor.item_id, ArtistRole.CONDUCTOR, None, 0),
    ]


async def test_removing_a_composer_keeps_their_tracks_and_albums(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing an artist only credited as composer keeps the credited items."""
    # the database-only server runs no audio analysis
    monkeypatch.setattr(
        mass,
        "streams",
        Mock(audio_analysis=Mock(delete_audio_analysis=AsyncMock())),
        raising=False,
    )
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    composer_credit = Credit(artist=composer, role=ArtistRole.COMPOSER)
    album = await _add_album(mass, [main], [composer_credit])
    track = await _add_track(mass, [main], [composer_credit], album=album)

    await mass.music.artists.remove_item_from_library(composer.item_id)

    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    stored_album = await mass.music.albums.get_library_item(album.item_id)
    assert _credit_rows(stored_track) == [(main.item_id, ArtistRole.MAIN_ARTIST, None, 0)]
    assert _credit_rows(stored_album) == [(main.item_id, ArtistRole.MAIN_ARTIST, None, 0)]


async def test_artist_listings_ignore_other_credits(mass: MusicAssistant) -> None:
    """The tracks, albums and appears-on of an artist only count main artist credits."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    composer_credit = Credit(artist=composer, role=ArtistRole.COMPOSER)
    album = await _add_album(mass, [main], [composer_credit])
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
    album = await _add_album(mass, [main], [composer_credit], name=title)
    track = await _add_track(mass, [main], [composer_credit], name=title)

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
    abe_track = await _add_track(mass, [abe], [composer_credit], name=f"{title} 1")
    lou_track = await _add_track(mass, [lou], name=f"{title} 2")
    abe_album = await _add_album(mass, [abe], [composer_credit], name=f"{title} 1")
    lou_album = await _add_album(mass, [lou], name=f"{title} 2")

    tracks = await mass.music.tracks.library_items(search=title, order_by="track_artist_name")
    albums = await mass.music.albums.library_items(search=title, order_by="album_artist_name")

    assert [x.item_id for x in tracks] == [abe_track.item_id, lou_track.item_id]
    assert [x.item_id for x in albums] == [abe_album.item_id, lou_album.item_id]
