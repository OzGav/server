"""Tests for the classical classification of library tracks, albums and artists."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from music_assistant_models.enums import AlbumType, ArtistRole, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    Credit,
    Genre,
    ItemMapping,
    ProviderMapping,
    Track,
    UniqueList,
    Work,
)
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.constants import (
    DB_TABLE_ALBUMS,
    DB_TABLE_ARTISTS,
    DB_TABLE_GENRE_MEDIA_ITEM_MAPPING,
    DB_TABLE_GENRES,
    DB_TABLE_TRACKS,
)
from music_assistant.helpers.json import json_loads
from music_assistant.mass import MusicAssistant
from music_assistant.models.music_provider import MusicProvider

PROVIDER_INSTANCE = "classification_instance"


def _patch_mass(mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch) -> MusicAssistant:
    """Stub the parts a database-only server lacks for removing tracks."""
    monkeypatch.setattr(
        mass,
        "streams",
        Mock(audio_analysis=Mock(delete_audio_analysis=AsyncMock())),
        raising=False,
    )
    monkeypatch.setattr(mass.cache, "delete", AsyncMock())
    return mass


@pytest.fixture(name="mass")
def mass_fixture(
    music_mass_module: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture."""
    return _patch_mass(music_mass_module, monkeypatch)


def _mapping() -> ProviderMapping:
    """Create a provider mapping with a unique provider item id."""
    return ProviderMapping(
        item_id=uuid4().hex,
        provider_domain="classification",
        provider_instance=PROVIDER_INSTANCE,
        in_library=True,
    )


def _token() -> str:
    """Return a word unique to the calling test, to keep its names apart from other tests."""
    return uuid4().hex[:10]


async def _add_artist(mass: MusicAssistant, name: str = "Artist", **kwargs: object) -> Artist:
    """Store a library artist under a name unique to the calling test."""
    return await mass.music.artists.add_item_to_library(
        Artist(
            item_id="0",
            provider="library",
            name=f"{name} {_token()}",
            provider_mappings={_mapping()},
            **kwargs,  # type: ignore[arg-type]
        )
    )


async def _add_album(mass: MusicAssistant, **kwargs: object) -> Album:
    """Store a library album."""
    return await mass.music.albums.add_item_to_library(
        Album(
            item_id="0",
            provider="library",
            name=f"Album {_token()}",
            album_type=AlbumType.ALBUM,
            provider_mappings={_mapping()},
            artists=UniqueList([await _add_artist(mass, "Album Artist")]),
            **kwargs,  # type: ignore[arg-type]
        )
    )


def _track(
    artists: list[Artist | ItemMapping],
    album: Album | None = None,
    track_number: int = 1,
    classical_tag: bool | None = None,
    **kwargs: object,
) -> Track:
    """Create a track as a source hands it over to be added to the library."""
    return Track(
        item_id="0",
        provider="library",
        name=f"Track {_token()}",
        provider_mappings={_mapping()},
        artists=UniqueList(artists),
        album=album,
        disc_number=1 if album else 0,
        track_number=track_number if album else 0,
        metadata=MediaItemMetadata(classical_tag=classical_tag),
        **kwargs,  # type: ignore[arg-type]
    )


async def _add_track(
    mass: MusicAssistant,
    album: Album | None = None,
    track_number: int = 1,
    artists: list[Artist | ItemMapping] | None = None,
    classical_tag: bool | None = None,
    **kwargs: object,
) -> Track:
    """Store a library track, by default by a new artist."""
    artists = artists or [await _add_artist(mass)]
    return await mass.music.tracks.add_item_to_library(
        _track(artists, album, track_number, classical_tag, **kwargs)
    )


async def _add_work(mass: MusicAssistant) -> Work:
    """Add a work to the library."""
    return await mass.music.works.add_item_to_library(
        Work(
            item_id=uuid4().hex,
            provider="works_source",
            name=f"Work {_token()}",
            provider_mappings=set(),
        )
    )


async def _classical_genre_id(mass: MusicAssistant) -> int:
    """Return the library id of the curated classical genre."""
    rows = await mass.music.database.get_rows_from_query(
        f"SELECT item_id FROM {DB_TABLE_GENRES} "
        "WHERE translation_key = 'classical' AND content_type IS NULL",
        limit=1,
    )
    return int(rows[0]["item_id"])


async def _is_classical(mass: MusicAssistant, item: Track | Album | Artist) -> bool:
    """Return the classical flag of a library item as the library returns it."""
    ctrl = mass.music.get_controller(item.media_type)
    library_item = await ctrl.get_library_item(item.item_id)
    assert isinstance(library_item, Track | Album | Artist)
    return library_item.is_classical


@pytest.mark.parametrize("classical_tag", [False, True])
async def test_track_is_classical_by_its_tag(mass: MusicAssistant, classical_tag: bool) -> None:
    """
    A track the source marks as classical is classical, as is its artist, and keeps the tag.

    A track without any classical signal is not classical, nor is its artist.
    """
    track = await _add_track(mass, classical_tag=classical_tag)

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert stored.metadata.classical_tag is classical_tag
    assert stored.is_classical is classical_tag
    assert await _is_classical(mass, track.artists[0]) is classical_tag  # type: ignore[arg-type]


async def test_track_in_classical_genre_is_classical(mass: MusicAssistant) -> None:
    """A track mapped to the curated classical genre is classical, other genres are not."""
    classical = await _add_track(mass)
    rock = await _add_track(mass)

    await mass.music.genres.sync_media_item_genres(
        MediaType.TRACK, classical.item_id, {"Classical"}
    )
    await mass.music.genres.sync_media_item_genres(MediaType.TRACK, rock.item_id, {"Rock"})

    assert await _is_classical(mass, classical)
    assert not await _is_classical(mass, rock)


@pytest.mark.parametrize(
    ("classical_tracks", "total_tracks", "expected"),
    [(2, 3, True), (2, 4, False), (1, 3, False)],
)
async def test_album_is_classical_by_majority(
    mass: MusicAssistant, classical_tracks: int, total_tracks: int, expected: bool
) -> None:
    """An album is classical when more than half of its tracks are, and then all its tracks are."""
    album = await _add_album(mass)
    tracks = [
        await _add_track(mass, album, number, classical_tag=number <= classical_tracks)
        for number in range(1, total_tracks + 1)
    ]

    assert await _is_classical(mass, album) is expected
    # the untagged tracks follow their album
    for track in tracks[classical_tracks:]:
        assert await _is_classical(mass, track) is expected


async def test_album_in_classical_genre_makes_its_tracks_classical(mass: MusicAssistant) -> None:
    """An album mapped to the classical genre is classical along with all its tracks."""
    album = await _add_album(mass)
    track = await _add_track(mass, album)

    await mass.music.genres.add_media_mapping(
        await _classical_genre_id(mass), MediaType.ALBUM, album.item_id
    )

    assert await _is_classical(mass, album)
    assert await _is_classical(mass, track)


async def test_album_genre_copied_down_from_its_tracks_does_not_count(
    mass: MusicAssistant,
) -> None:
    """A classical genre an album only inherits from one of its tracks leaves a mixed album be."""
    album = await _add_album(mass)
    tracks = [await _add_track(mass, album, number) for number in (1, 2, 3)]
    genre_id = await _classical_genre_id(mass)
    await mass.music.genres.add_media_mapping(genre_id, MediaType.TRACK, tracks[0].item_id)
    await mass.music.database.insert(
        DB_TABLE_GENRE_MEDIA_ITEM_MAPPING,
        {
            "genre_id": genre_id,
            "media_id": int(album.item_id),
            "media_type": MediaType.ALBUM.value,
            "is_derived": 1,
        },
    )

    await mass.music.classification.update(album_ids=[int(album.item_id)])

    assert not await _is_classical(mass, album)
    assert not await _is_classical(mass, tracks[1])


async def test_album_majority_ignores_tracks_classical_through_another_album(
    mass: MusicAssistant,
) -> None:
    """A track classical only through another album does not count toward an album's majority."""
    album = await _add_album(mass)
    classical_album = await _add_album(mass)
    await mass.music.genres.add_media_mapping(
        await _classical_genre_id(mass), MediaType.ALBUM, classical_album.item_id
    )
    track = await _add_track(mass, album)
    await mass.music.tracks.update_item_in_library(
        track.item_id, _track([], classical_album), overwrite=False
    )

    assert await _is_classical(mass, track)
    assert not await _is_classical(mass, album)


async def test_artist_is_classical_through_any_credit(mass: MusicAssistant) -> None:
    """Every artist credited on a classical track is classical, whatever the role."""
    main = await _add_artist(mass, "Main")
    composer = await _add_artist(mass, "Composer")
    conductor = await _add_artist(mass, "Conductor")
    pop_artist = await _add_artist(mass, "Pop")
    await _add_track(
        mass,
        artists=[main],
        credits=[
            Credit(artist=composer, role=ArtistRole.COMPOSER),
            Credit(artist=conductor, role=ArtistRole.CONDUCTOR),
        ],
        classical_tag=True,
    )
    await _add_track(mass, artists=[pop_artist])

    assert await _is_classical(mass, main)
    assert await _is_classical(mass, composer)
    assert await _is_classical(mass, conductor)
    assert not await _is_classical(mass, pop_artist)


async def test_incoming_is_classical_is_ignored(mass: MusicAssistant) -> None:
    """The flags are computed by the server, never taken from the incoming item."""
    artist = await _add_artist(mass, is_classical=True)
    album = await _add_album(mass, is_classical=True)
    track = await _add_track(mass, album, artists=[artist], is_classical=True)
    plain = await _add_track(mass)
    await mass.music.tracks.update_item_in_library(
        plain.item_id, _track(list(plain.artists), is_classical=True), overwrite=True
    )

    assert not await _is_classical(mass, artist)
    assert not await _is_classical(mass, album)
    assert not await _is_classical(mass, track)
    assert not await _is_classical(mass, plain)


@pytest.mark.parametrize("streaming", [False, True])
async def test_track_overwrite_takes_the_classical_tag_only_from_a_file(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """A re-read file without the classical tag clears it, a streaming overwrite keeps it."""
    track = await _add_track(mass, classical_tag=True)
    source = Mock(spec=MusicProvider, is_streaming_provider=streaming)
    monkeypatch.setattr(mass, "get_provider", Mock(return_value=source))

    await mass.music.tracks.update_item_in_library(
        track.item_id, _track(list(track.artists)), overwrite=True
    )

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert stored.metadata.classical_tag is streaming
    assert stored.is_classical is streaming


async def test_track_classical_tag_is_stored_only_in_its_column(mass: MusicAssistant) -> None:
    """The classical tag of an added or updated track is kept out of its stored metadata."""
    track = await _add_track(mass, classical_tag=True)
    await mass.music.tracks.update_item_in_library(
        track.item_id, _track(list(track.artists), classical_tag=True)
    )

    row = await mass.music.database.get_row(DB_TABLE_TRACKS, {"item_id": int(track.item_id)})
    assert row is not None
    assert row["classical_tag"]
    assert "classical_tag" not in json_loads(row["metadata"])


async def test_flags_follow_genre_mapping_changes(mass: MusicAssistant) -> None:
    """Adding, removing and excluding the classical genre on a track updates the flags."""
    genre_id = await _classical_genre_id(mass)
    track = await _add_track(mass)
    artist = track.artists[0]
    assert isinstance(artist, Artist | ItemMapping)

    await mass.music.genres.add_media_mapping(genre_id, MediaType.TRACK, track.item_id)
    added = await _is_classical(mass, track), await _is_classical(mass, artist)  # type: ignore[arg-type]
    await mass.music.genres.remove_media_mapping(genre_id, MediaType.TRACK, track.item_id)
    removed = await _is_classical(mass, track), await _is_classical(mass, artist)  # type: ignore[arg-type]
    await mass.music.genres.sync_media_item_genres(MediaType.TRACK, track.item_id, {"Baroque"})
    synced = await _is_classical(mass, track)
    await mass.music.genres.exclude_genre_from_media_item(genre_id, MediaType.TRACK, track.item_id)
    excluded = await _is_classical(mass, track)

    assert added == (True, True)
    assert removed == (False, False)
    assert synced
    assert not excluded


async def test_flags_follow_genre_merge(mass: MusicAssistant) -> None:
    """Merging a genre into the classical genre makes its tracks classical."""
    name = f"Klassik {_token()}"
    genre = await mass.music.genres.add_item_to_library(
        Genre(item_id="0", provider="library", name=name, provider_mappings=set())
    )
    track = await _add_track(mass)
    await mass.music.genres.add_media_mapping(genre.item_id, MediaType.TRACK, track.item_id)
    before = await _is_classical(mass, track)

    await mass.music.genres.merge_genres([genre.item_id], await _classical_genre_id(mass))

    assert not before
    assert await _is_classical(mass, track)


async def test_removing_a_track_reclassifies_its_album(mass: MusicAssistant) -> None:
    """A track leaving an album can drop the album, and its other tracks, below the majority."""
    album = await _add_album(mass)
    first = await _add_track(mass, album, 1, classical_tag=True)
    await _add_track(mass, album, 2, classical_tag=True)
    last = await _add_track(mass, album, 3)
    assert await _is_classical(mass, last)

    await mass.music.tracks.remove_item_from_library(first.item_id)

    assert not await _is_classical(mass, album)
    assert not await _is_classical(mass, last)
    assert not await _is_classical(mass, first.artists[0])  # type: ignore[arg-type]


async def test_flags_follow_credit_changes(mass: MusicAssistant) -> None:
    """An artist that loses its only credit on a classical track is no longer classical."""
    composer = await _add_artist(mass, "Composer")
    conductor = await _add_artist(mass, "Conductor")
    track = await _add_track(
        mass, credits=[Credit(artist=composer, role=ArtistRole.COMPOSER)], classical_tag=True
    )
    assert await _is_classical(mass, composer)

    await mass.music.tracks.update_item_in_library(
        track.item_id,
        _track(
            list(track.artists),
            credits=[Credit(artist=conductor, role=ArtistRole.CONDUCTOR)],
            classical_tag=True,
        ),
        overwrite=True,
    )

    assert not await _is_classical(mass, composer)
    assert await _is_classical(mass, conductor)


async def test_flags_follow_main_artist_changes(mass: MusicAssistant) -> None:
    """An artist that stops being the main artist of its only classical track is not classical."""
    old_main = await _add_artist(mass, "Old Main")
    new_main = await _add_artist(mass, "New Main")
    track = await _add_track(mass, artists=[old_main], classical_tag=True)
    assert await _is_classical(mass, old_main)

    await mass.music.tracks.update_item_in_library(
        track.item_id, _track([new_main], classical_tag=True), overwrite=True
    )

    assert not await _is_classical(mass, old_main)
    assert await _is_classical(mass, new_main)


async def test_flags_follow_merges(mass: MusicAssistant) -> None:
    """Merged tracks and artists carry the classification of what was merged into them."""
    target = await _add_track(mass)
    source = await _add_track(mass, classical_tag=True)
    artist = await _add_artist(mass)
    duplicate = source.artists[0]

    await mass.music.tracks.merge_library_items(target.item_id, source.item_id)
    await mass.music.artists.merge_library_items(artist.item_id, duplicate.item_id)

    assert await _is_classical(mass, target)
    assert await _is_classical(mass, target.artists[0])  # type: ignore[arg-type]
    assert await _is_classical(mass, artist)


async def test_update_all_restores_every_flag(mass: MusicAssistant) -> None:
    """The full recompute corrects flags that went stale in either direction."""
    album = await _add_album(mass)
    await mass.music.genres.add_media_mapping(
        await _classical_genre_id(mass), MediaType.ALBUM, album.item_id
    )
    track = await _add_track(mass, album)
    plain = await _add_track(mass)
    for table, item_id, value in (
        (DB_TABLE_ALBUMS, album.item_id, 0),
        (DB_TABLE_TRACKS, track.item_id, 0),
        (DB_TABLE_ARTISTS, track.artists[0].item_id, 0),
        (DB_TABLE_TRACKS, plain.item_id, 1),
        (DB_TABLE_ARTISTS, plain.artists[0].item_id, 1),
    ):
        await mass.music.database.update(table, {"item_id": int(item_id)}, {"is_classical": value})

    await mass.music.classification.update_all()

    assert await _is_classical(mass, album)
    assert await _is_classical(mass, track)
    assert await _is_classical(mass, track.artists[0])  # type: ignore[arg-type]
    assert not await _is_classical(mass, plain)
    assert not await _is_classical(mass, plain.artists[0])  # type: ignore[arg-type]


class TestHasClassicalContent:
    """The classical content probe, on a library of its own."""

    @pytest.fixture(name="mass")
    async def mass_fixture(
        self, music_mass_class: MusicAssistant, monkeypatch: pytest.MonkeyPatch
    ) -> MusicAssistant:
        """Return the class-scoped database-only Music Assistant fixture."""
        return _patch_mass(music_mass_class, monkeypatch)

    async def test_has_classical_content(self, mass: MusicAssistant) -> None:
        """The probe is false for a library without classical music, true with it."""
        empty = await mass.music.has_classical_content()
        for genre in ("Rock", "Pop"):
            album = await _add_album(mass)
            track = await _add_track(mass, album)
            await mass.music.genres.sync_media_item_genres(MediaType.TRACK, track.item_id, {genre})
        await mass.music.classification.update_all()
        non_classical = await mass.music.has_classical_content()
        flags = await mass.music.database.get_rows_from_query(
            f"SELECT is_classical FROM {DB_TABLE_TRACKS} UNION ALL "
            f"SELECT is_classical FROM {DB_TABLE_ALBUMS} UNION ALL "
            f"SELECT is_classical FROM {DB_TABLE_ARTISTS}",
            limit=0,
        )
        classical = await _add_track(mass, classical_tag=True)
        with_track = await mass.music.has_classical_content()
        await mass.music.tracks.remove_item_from_library(classical.item_id)
        without_track = await mass.music.has_classical_content()
        await _add_work(mass)
        with_only_a_work = await mass.music.has_classical_content()

        assert not empty
        assert not non_classical
        assert flags
        assert not any(row["is_classical"] for row in flags)
        assert with_track
        assert not without_track
        assert not with_only_a_work
