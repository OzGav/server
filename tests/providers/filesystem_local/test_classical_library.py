"""Tests for the classical tags of local files as they land in the library."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from music_assistant_models.enums import ArtistRole, ExternalID, Period
from music_assistant_models.media_items import Artist, ItemMapping, ProviderMapping, Track

from music_assistant.helpers.tags import AudioTags
from music_assistant.mass import MusicAssistant
from music_assistant.providers.filesystem_local import LocalFileSystemProvider
from music_assistant.providers.filesystem_local.helpers import FileSystemItem
from tests.providers.filesystem_local.conftest import make_provider


@pytest.fixture(name="mass")
def mass_fixture(
    music_mass_module: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture."""
    # the database-only server runs no audio analysis
    monkeypatch.setattr(
        music_mass_module,
        "streams",
        Mock(audio_analysis=Mock(delete_audio_analysis=AsyncMock())),
        raising=False,
    )
    return music_mass_module


@pytest.fixture(name="provider")
def provider_fixture(mass: MusicAssistant, tmp_path: Path) -> LocalFileSystemProvider:
    """Return a Local files source reading from an empty folder."""
    provider = make_provider(mass, tmp_path)
    # the database-only server has no cache database
    provider.cache = Mock(get=AsyncMock(return_value=None), set=AsyncMock())
    return provider


def _token() -> str:
    """Return a word unique to the calling test, to keep its names apart from other tests."""
    return uuid4().hex[:10]


def _tags(path: str, **tags: Any) -> AudioTags:
    """Create the tags of a track file, with a title, artist and album unless given."""
    return AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=1411,
        duration=300.0,
        tags={"title": "Movement", "artist": "Performer", "album": "Album", **tags},
        has_cover_image=False,
        filename=path,
    )


async def _add_track(provider: LocalFileSystemProvider, path: str, tags: AudioTags) -> Track:
    """Parse a track file and add it to the library, the way a sync does."""
    absolute_path = Path(provider.base_path, path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.touch()
    file_item = FileSystemItem(
        filename=path.rsplit("/", 1)[-1],
        relative_path=path,
        absolute_path=str(absolute_path),
        is_dir=False,
        checksum=uuid4().hex,
    )
    track = await provider._parse_track(file_item, tags)
    await provider._set_track_work(track, tags)
    return await provider.mass.music.tracks.add_item_to_library(track)


async def test_classical_track_lands_in_the_library(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """A tagged classical track carries its credits, work, movement and classical tag."""
    token = _token()
    composer_mbid = str(uuid4())
    tags = _tags(
        f"{token}/Album/01.flac",
        title="Symphony No. 5: I. Allegro con brio",
        album=f"Symphonies {token}",
        artist=f"Orchestra {token}",
        composer=f"Composer {token}",
        composersort=[f"{token}, Composer"],
        musicbrainzcomposerid=[composer_mbid],
        conductor=[f"Conductor {token}"],
        performer=[f"Orchestra {token} (orchestra)", f"Pianist {token} (piano)"],
        work=[f"Symphony No. 5 {token}"],
        movementname="Allegro con brio",
        movement="1",
        movementtotal="4",
        isclassical="1",
        genre="Classical; Baroque",
    )

    track = await _add_track(provider, f"{token}/Album/01.flac", tags)

    roles = {(c.role, c.artist.name, c.instrument) for c in track.credits}
    assert (ArtistRole.COMPOSER, f"Composer {token}", None) in roles
    assert (ArtistRole.CONDUCTOR, f"Conductor {token}", None) in roles
    assert (ArtistRole.ORCHESTRA, f"Orchestra {token}", None) in roles
    assert (ArtistRole.SOLOIST, f"Pianist {token}", "piano") in roles
    assert track.work is not None
    assert track.work.name == f"Symphony No. 5 {token}"
    assert (track.movement_number, track.movement_total, track.movement_name) == (
        1,
        4,
        "Allegro con brio",
    )
    assert track.classical_tag
    composer = await mass.music.artists.get_library_item_by_external_id(
        composer_mbid, ExternalID.MB_ARTIST
    )
    assert composer is not None
    assert composer.sort_name == f"{token}, Composer"
    assert composer.period == Period.BAROQUE


async def test_renamed_file_replaces_the_credits(provider: LocalFileSystemProvider) -> None:
    """A file renamed with new tags leaves only its new credits on the library track."""
    token = _token()
    shared = {
        "title": f"Ave Maria {token}",
        "album": f"Songs {token}",
        "artist": f"Singer {token}",
        "musicbrainzrecordingid": str(uuid4()),
    }
    old = await _add_track(
        provider,
        f"{token}/Album/01.mp3",
        _tags(
            f"{token}/Album/01.mp3",
            **shared,
            composer=f"Gounod {token}",
            lyricist=f"Lyricist {token}",
        ),
    )

    new = await _add_track(
        provider,
        f"{token}/Album/01.flac",
        _tags(f"{token}/Album/01.flac", **shared, composer=f"Charles Gounod {token}"),
    )

    assert new.item_id == old.item_id
    assert [(c.role, c.artist.name) for c in new.credits if c.role != ArtistRole.MAIN_ARTIST] == [
        (ArtistRole.COMPOSER, f"Charles Gounod {token}")
    ]


async def test_tracks_of_one_work_share_it_when_added_together(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """Movements of a new work processed at the same time link to one library work."""
    token = _token()
    paths = [f"{token}/Album/{number:02d}.flac" for number in range(1, 5)]
    tracks = await asyncio.gather(
        *(
            _add_track(
                provider,
                path,
                _tags(
                    path,
                    title=f"Movement {number}",
                    album=f"Album {token}",
                    composer=f"Composer {token}",
                    work=[f"Quartet {token}"],
                    movement=str(number),
                ),
            )
            for number, path in enumerate(paths, start=1)
        )
    )

    assert all(track.work for track in tracks)
    assert len({track.work.item_id for track in tracks if track.work}) == 1
    works = await mass.music.works.library_items(search=f"Quartet {token}")
    assert len(works) == 1


async def _parse_track(provider: LocalFileSystemProvider, path: str, tags: AudioTags) -> Track:
    """Parse a track file without adding it to the library."""
    absolute_path = Path(provider.base_path, path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.touch()
    file_item = FileSystemItem(
        filename=absolute_path.name,
        relative_path=path,
        absolute_path=str(absolute_path),
        is_dir=False,
        checksum=uuid4().hex,
    )
    return await provider._parse_track(file_item, tags)


def _work_ids(track: Track) -> set[str]:
    """Return the MusicBrainz work ids a track carries itself."""
    return {value for id_type, value in track.external_ids if id_type == ExternalID.MB_WORK}


def _period(artist: Artist | ItemMapping) -> Period | None:
    """Return the period of a parsed artist."""
    assert isinstance(artist, Artist)
    return artist.period


def _credits(track: Track) -> list[tuple[ArtistRole, str, str | None, int]]:
    """Return the role, artist name, instrument and position of each credit of a track."""
    return [(x.role, x.artist.name, x.instrument, x.position) for x in track.credits]


async def test_credits_follow_the_tag_order_per_role(provider: LocalFileSystemProvider) -> None:
    """Each role numbers its credits in tag order, and a credit is never repeated."""
    tags = _tags(
        "Album/01.flac",
        composer="Composer A; Composer B",
        conductor=["Conductor"],
        performer=[
            "Orchestra (orchestra)",
            "Choir (choir vocals)",
            "Quartet (string quartet)",
            "Pianist (piano)",
            "Pianist (harpsichord)",
            "Session Player",
            "Conductor (conductor)",
        ],
        lyricist=["Lyricist"],
        arranger=["Arranger"],
    )

    track = await _parse_track(provider, "Album/01.flac", tags)

    assert _credits(track) == [
        (ArtistRole.COMPOSER, "Composer A", None, 0),
        (ArtistRole.COMPOSER, "Composer B", None, 1),
        (ArtistRole.CONDUCTOR, "Conductor", None, 0),
        (ArtistRole.ORCHESTRA, "Orchestra", None, 0),
        (ArtistRole.CHOIR, "Choir", None, 0),
        (ArtistRole.ENSEMBLE, "Quartet", None, 0),
        (ArtistRole.SOLOIST, "Pianist", "piano", 0),
        (ArtistRole.SOLOIST, "Pianist", "harpsichord", 1),
        (ArtistRole.PERFORMER, "Session Player", None, 0),
        (ArtistRole.LYRICIST, "Lyricist", None, 0),
        (ArtistRole.ARRANGER, "Arranger", None, 0),
    ]
    # one artist with several credits is one artist
    pianists = {id(x.artist) for x in track.credits if x.artist.name == "Pianist"}
    assert len(pianists) == 1


async def test_musicbrainz_placeholder_credits_are_skipped(
    provider: LocalFileSystemProvider,
) -> None:
    """Placeholder names such as [traditional] and [anonymous] give no credit."""
    tags = _tags(
        "Album/01.flac",
        composer="[traditional]",
        lyricist=["[anonymous]", "Real Lyricist"],
    )

    track = await _parse_track(provider, "Album/01.flac", tags)

    assert _credits(track) == [(ArtistRole.LYRICIST, "Real Lyricist", None, 0)]


async def test_roon_credits_are_read_without_performer_tags(
    provider: LocalFileSystemProvider,
) -> None:
    """Roon's personnel, soloist and ensemble tags credit the performers of a file."""
    tags = _tags(
        "Album/01.flac",
        personnel=["Berliner Philharmoniker - Orchestra", "Wolfgang Schulz - Flute"],
        soloist=["Martha Argerich - Piano", "Gidon Kremer"],
        ensemble=["Kremerata Baltica"],
    )

    track = await _parse_track(provider, "Album/01.flac", tags)

    assert _credits(track) == [
        (ArtistRole.ORCHESTRA, "Berliner Philharmoniker", None, 0),
        (ArtistRole.SOLOIST, "Wolfgang Schulz", "Flute", 0),
        (ArtistRole.SOLOIST, "Martha Argerich", "Piano", 1),
        (ArtistRole.SOLOIST, "Gidon Kremer", None, 2),
        (ArtistRole.ENSEMBLE, "Kremerata Baltica", None, 0),
    ]


async def test_composers_carry_their_musicbrainz_ids_and_sort_names(
    provider: LocalFileSystemProvider,
) -> None:
    """Composer ids and sort names pair up with the composers by position."""
    first_id, second_id = str(uuid4()), str(uuid4())
    tags = _tags(
        "Album/01.flac",
        composer="Wolfgang Amadeus Mozart; Antonio Salieri",
        composersort=["Mozart, Wolfgang Amadeus", "Salieri, Antonio"],
        musicbrainzcomposerid=[first_id, second_id],
    )

    track = await _parse_track(provider, "Album/01.flac", tags)

    assert [(x.name, x.sort_name, x.mbid) for x in track.composers] == [
        ("Wolfgang Amadeus Mozart", "Mozart, Wolfgang Amadeus", first_id),
        ("Antonio Salieri", "Salieri, Antonio", second_id),
    ]


async def test_credit_reuses_the_track_artist(provider: LocalFileSystemProvider) -> None:
    """A songwriter credited as composer is the same artist as the track artist."""
    tags = _tags("Album/01.flac", artist="Singer", composer="Singer; Cowriter")

    track = await _parse_track(provider, "Album/01.flac", tags)

    assert track.composers[0] is track.artists[0]
    assert track.composers[1].name == "Cowriter"


async def test_numbered_movement_title_links_the_parent_work(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """A movement work title links the parent work, the movement work id stays on the track."""
    token = _token()
    movement_work_id = str(uuid4())
    path = f"{token}/Album/02.flac"
    tags = _tags(
        path,
        album=f"Album {token}",
        composer=f"Composer {token}",
        work=[f"Orchestersuite Nr. 3 D-Dur, BWV 1068 {token}: II. Air"],
        musicbrainzworkid=[movement_work_id],
    )

    track = await _add_track(provider, path, tags)

    assert track.work is not None
    assert track.work.name == f"Orchestersuite Nr. 3 D-Dur, BWV 1068 {token}"
    assert (track.movement_number, track.movement_name) == (2, "II. Air")
    assert _work_ids(track) == {movement_work_id}
    work = await mass.music.works.get_library_item(track.work.item_id)
    assert work.get_external_id(ExternalID.MB_WORK) is None
    assert [x.name for x in work.composers] == [f"Composer {token}"]


async def test_movement_tags_link_the_general_work_and_keep_the_movement_id(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """With movement tags the general work links with its id, the movement id stays on the track."""
    token = _token()
    work_id, movement_work_id = str(uuid4()), str(uuid4())
    path = f"{token}/Album/01.flac"
    tags = _tags(
        path,
        album=f"Album {token}",
        composer=f"Composer {token}",
        work=[f"Symphony {token}", f"Symphony {token}: I. Allegro"],
        musicbrainzworkid=[work_id, movement_work_id],
        movementname="Allegro",
        movement="1/4",
    )

    track = await _add_track(provider, path, tags)

    assert track.work is not None
    work = await mass.music.works.get_library_item(track.work.item_id)
    assert (work.name, work.get_external_id(ExternalID.MB_WORK)) == (f"Symphony {token}", work_id)
    assert (track.movement_number, track.movement_total, track.movement_name) == (1, 4, "Allegro")
    assert _work_ids(track) == {movement_work_id}


async def test_movement_tags_win_over_a_split_movement_title(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """A movement title links the parent work, the movement tags keep their own values."""
    token = _token()
    movement_work_id = str(uuid4())
    path = f"{token}/Album/03.flac"
    tags = _tags(
        path,
        album=f"Album {token}",
        composer=f"Composer {token}",
        work=[f"Symphony {token}: III. Scherzo"],
        musicbrainzworkid=[movement_work_id],
        movementname="Scherzo. Allegro",
        movementtotal="4",
    )

    track = await _add_track(provider, path, tags)

    assert track.work is not None
    work = await mass.music.works.get_library_item(track.work.item_id)
    assert (work.name, work.get_external_id(ExternalID.MB_WORK)) == (f"Symphony {token}", None)
    # the tags name the movement and its total, the title only fills in its number
    assert (track.movement_name, track.movement_number, track.movement_total) == (
        "Scherzo. Allegro",
        3,
        4,
    )
    assert _work_ids(track) == {movement_work_id}


async def test_work_ids_without_matching_titles_stay_on_the_track(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """Work ids that do not pair up with the work titles stay on the track, not on the work."""
    token = _token()
    work_ids = {"51bb8773-8492-3773-ab88-73a89c922c3d", "b58b7e13-17df-4122-82de-8031081d9647"}
    path = f"{token}/Album/02.flac"
    tags = _tags(
        path,
        album=f"Album {token}",
        composer=f"Composer {token}",
        work=[f"Orchestersuite Nr. 3 D-Dur, BWV 1068 {token}: II. Air"],
        musicbrainzworkid=sorted(work_ids),
    )
    other_path = f"{token}/Other/02.flac"
    other_tags = _tags(
        other_path,
        album=f"Other {token}",
        composer=f"Composer {token}",
        work=[f"Orchestersuite Nr. 3 D-Dur, BWV 1068 {token}"],
    )

    track = await _add_track(provider, path, tags)
    other_track = await _add_track(provider, other_path, other_tags)

    assert track.work is not None
    assert other_track.work is not None
    assert track.work.item_id == other_track.work.item_id
    work = await mass.music.works.get_library_item(track.work.item_id)
    assert work.get_external_id(ExternalID.MB_WORK) is None
    assert _work_ids(track) == work_ids


async def test_pop_track_with_a_work_tag_stays_non_classical(
    provider: LocalFileSystemProvider, mass: MusicAssistant
) -> None:
    """A pop song tagged with its work gets the work but is not classical."""
    token = _token()
    path = f"{token}/Album/01.flac"
    tags = _tags(
        path,
        album=f"Album {token}",
        artist=f"Band {token}",
        composer=f"Songwriter {token}",
        work=[f"Song {token}"],
        musicbrainzworkid=[str(uuid4())],
        genre="Pop",
    )

    track = await _add_track(provider, path, tags)

    assert track.work is not None
    assert track.work.name == f"Song {token}"
    full_track = await mass.music.tracks.get_library_item(track.item_id)
    assert not full_track.classical_tag
    assert not full_track.is_classical


async def test_composer_period_comes_from_the_nfo_files_first(
    provider: LocalFileSystemProvider,
) -> None:
    """The artist.nfo genres outrank the album.nfo genres, which outrank the track genres."""
    token = _token()
    composer = f"Composer {token}"
    album = f"Album {token}"
    base = Path(provider.base_path)
    (base / composer).mkdir()
    (base / composer / "artist.nfo").write_text(
        f"<artist><title>{composer}</title><genre>Romantic</genre></artist>"
    )
    (base / token / album).mkdir(parents=True)
    (base / token / album / "album.nfo").write_text(
        f"<album><title>{album}</title><genre>Baroque</genre></album>"
    )

    def _tags_for(path: str, artist: str) -> AudioTags:
        return _tags(path, album=album, artist=artist, composer=composer, genre="Renaissance")

    # the composer is the track artist, so its own folder and artist.nfo are read
    with_artist_nfo = await _parse_track(
        provider, f"{token}/{album}/01.flac", _tags_for(f"{token}/{album}/01.flac", composer)
    )
    # credited as composer only, the album.nfo genres come first
    with_album_nfo = await _parse_track(
        provider,
        f"{token}/{album}/02.flac",
        _tags_for(f"{token}/{album}/02.flac", f"Pianist {token}"),
    )
    with_track_genres = await _parse_track(
        provider, f"{token}/Other/03.flac", _tags_for(f"{token}/Other/03.flac", f"Pianist {token}")
    )

    assert _period(with_artist_nfo.composers[0]) == Period.ROMANTIC
    assert _period(with_album_nfo.composers[0]) == Period.BAROQUE
    assert _period(with_track_genres.composers[0]) == Period.RENAISSANCE


async def test_only_composers_get_a_period(provider: LocalFileSystemProvider) -> None:
    """Performers get no period, and the plain genre Classical sets none."""
    baroque = await _parse_track(
        provider,
        "Album/01.flac",
        _tags("Album/01.flac", composer="Bach", conductor=["Karajan"], genre="Baroque"),
    )
    classical = await _parse_track(
        provider, "Album/02.flac", _tags("Album/02.flac", composer="Haydn", genre="Classical")
    )

    assert _period(baroque.composers[0]) == Period.BAROQUE
    assert _period(baroque.conductors[0]) is None
    assert _period(baroque.artists[0]) is None
    assert _period(classical.composers[0]) is None


async def test_artist_overwrite_without_a_period_keeps_it(mass: MusicAssistant) -> None:
    """A source that has no period for an artist leaves the stored period in place."""
    token = _token()

    def _artist(period: Period | None) -> Artist:
        return Artist(
            item_id=f"Composer {token}",
            provider="filesystem_local--test",
            name=f"Composer {token}",
            period=period,
            provider_mappings={
                ProviderMapping(
                    item_id=f"Composer {token}",
                    provider_domain="filesystem_local",
                    provider_instance="filesystem_local--test",
                )
            },
        )

    await mass.music.artists.add_item_to_library(_artist(Period.BAROQUE))
    kept = await mass.music.artists.add_item_to_library(_artist(None), overwrite_existing=True)
    replaced = await mass.music.artists.add_item_to_library(
        _artist(Period.ROMANTIC), overwrite_existing=True
    )

    assert kept.period == Period.BAROQUE
    assert replaced.period == Period.ROMANTIC
