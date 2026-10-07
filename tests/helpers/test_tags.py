"""Tests for parsing audio file tags (ID3, MP4/AAC, Vorbis, APEv2, etc.)."""

import pathlib
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import mutagen
import pytest
from music_assistant_models.errors import InvalidDataError
from mutagen.apev2 import APEv2
from mutagen.flac import FLAC
from mutagen.id3 import (
    GRP1,
    ID3,
    IPLS,
    MVIN,
    MVNM,
    TCOM,
    TDOR,
    TDRC,
    TEXT,
    TIPL,
    TIT1,
    TMCL,
    TPE3,
    TSOC,
    TXXX,
    UFID,
)
from mutagen.mp4 import MP4, MP4FreeForm

from music_assistant.constants import UNKNOWN_ARTIST
from music_assistant.helpers import tags
from music_assistant.helpers.tags import (
    _parse_apev2_tags,
    _parse_id3_tags,
    _parse_mp4_tags,
    _parse_vorbis_tags,
    clean_mbid,
    parse_tags_mutagen,
    split_artists,
    write_replaygain_track_gain,
)

RESOURCES_DIR = pathlib.Path(__file__).parent.parent.resolve().joinpath("fixtures")

FILE_MP3 = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.mp3"))
FILE_MP3_ID3V24_MULTIVALUE = str(RESOURCES_DIR.joinpath("MultiArtist-ID3v24-NullSeparated.mp3"))
FILE_M4A = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.m4a"))
FILE_FLAC = str(RESOURCES_DIR.joinpath("MultipleArtists.flac"))
FILE_FLAC_SEMICOLON = str(RESOURCES_DIR.joinpath("ArtistWithSemicolon.flac"))
FILE_WV = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.wv"))


@pytest.mark.parametrize(
    ("stderr", "expected_detail"),
    [
        (
            b"[Vorbis parser @ 0x123] Invalid Setup header\n"
            b"[ogg @ 0x456] Header processing failed: Unknown error occurred\n",
            "Invalid Setup header",
        ),
        (b"broken.ogg: Unknown error occurred\n", "Invalid or unsupported media file"),
    ],
)
def test_parse_tags_reports_actionable_ffprobe_error(
    monkeypatch: pytest.MonkeyPatch, stderr: bytes, expected_detail: str
) -> None:
    """Test that tag parsing reports a useful FFprobe failure."""
    process_error = subprocess.CalledProcessError(
        returncode=1,
        cmd=("ffprobe",),
        output=b'{"error":{"code":-1,"string":"Unknown error occurred"}}',
        stderr=stderr,
    )
    check_output = MagicMock(side_effect=process_error)
    monkeypatch.setattr(subprocess, "check_output", check_output)

    with pytest.raises(InvalidDataError) as err:
        tags.parse_tags("broken.ogg")

    assert str(err.value) == f"Unable to retrieve info for broken.ogg ({expected_detail})"
    assert check_output.call_args.kwargs == {"stderr": subprocess.PIPE}
    args = check_output.call_args.args[0]
    assert args[args.index("-loglevel") + 1] == "error"


def test_parse_rejects_file_without_audio_channels() -> None:
    """A file for which ffprobe reports zero channels is corrupt and must be skipped."""
    raw = {
        "format": {"filename": "corrupt.mp3", "format_name": "mp3", "duration": "0"},
        "streams": [{"codec_type": "audio", "channels": 0, "sample_rate": "44100"}],
    }
    with pytest.raises(InvalidDataError, match="No audio channels found"):
        tags.AudioTags.parse(raw)


async def test_parse_metadata_from_id3tags() -> None:
    """Test parsing of parsing metadata from ID3 tags."""
    filename = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.mp3"))
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album == "MyAlbum"
    assert _tags.title == "MyTitle"
    # FFmpeg versions differ on whether MP3 encoder delay/padding counts toward duration.
    assert _tags.duration in (1.0, 1.032)
    assert _tags.album_artists == ("MyArtist",)
    assert _tags.artists == ("MyArtist", "MyArtist2")
    assert _tags.genres == ("Genre1", "Genre2")
    assert _tags.musicbrainz_albumartistids == ("abcdefg",)
    assert _tags.musicbrainz_artistids == ("abcdefg",)
    assert _tags.musicbrainz_releasegroupid == "abcdefg"
    assert _tags.musicbrainz_recordingid == "abcdefg"
    assert _tags.synchronized_lyrics == [
        ("My synchronized lyrics start here", 0),
        ("continue on this line", 671110),
        ("and end here.", 5999999),
    ]
    # test parsing disc/track number
    _tags.tags["disc"] = ""
    assert _tags.disc is None
    _tags.tags["disc"] = "1"
    assert _tags.disc == 1
    _tags.tags["disc"] = "1/1"  # type: ignore[unreachable]
    assert _tags.disc == 1
    # test parsing album year
    _tags.tags["date"] = "blah"
    assert _tags.year is None
    _tags.tags.pop("date", None)
    assert _tags.year is None
    _tags.tags["date"] = "2022"
    assert _tags.year == 2022
    _tags.tags["date"] = "2022-05-05"
    assert _tags.year == 2022
    _tags.tags["date"] = ""
    assert _tags.year is None


async def test_parse_id3v24_null_separated_artists() -> None:
    """Test parsing ID3v2.4 tags with null-separated multi-value TPE1/TPE2."""
    _tags = await tags.async_parse_tags(FILE_MP3_ID3V24_MULTIVALUE)
    # Null-separated artists in TPE1 should be parsed as multiple artists
    assert _tags.artists == ("Artist One", "Artist Two", "Artist Three")
    # Null-separated album artists in TPE2 should be parsed as multiple album artists
    assert _tags.album_artists == ("Album Artist A", "Album Artist B")
    # MB IDs should match
    assert _tags.musicbrainz_artistids == ("mb-artist-1", "mb-artist-2", "mb-artist-3")
    assert _tags.musicbrainz_albumartistids == ("mb-albumartist-1", "mb-albumartist-2")


async def test_parse_metadata_from_mp4tags() -> None:
    """Test parsing of metadata from MP4/AAC tags."""
    filename = FILE_M4A
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album == "MyAlbum"
    assert _tags.title == "MyTitle"
    assert _tags.album_artists == ("MyArtist",)
    assert _tags.artists == ("MyArtist", "MyArtist2")
    assert _tags.genres == ("Genre1", "Genre2")
    assert _tags.musicbrainz_albumartistids == ("abcdefg",)
    assert _tags.musicbrainz_artistids == ("abcdefg",)
    assert _tags.musicbrainz_releasegroupid == "abcdefg"
    assert _tags.musicbrainz_recordingid == "abcdefg"
    # test track/disc from MP4 tuples
    assert _tags.track == 5
    assert _tags.disc == 1
    # test total track/disc
    assert _tags.tags.get("tracktotal") == "12"
    assert _tags.tags.get("disctotal") == "2"
    # test year
    assert _tags.year == 2022
    # test sort tags (artistsort/albumartistsort returned as lists to match ID3 behavior)
    assert _tags.tags.get("titlesort") == "MyTitle Sort"
    assert _tags.tags.get("artistsort") == ["MyArtist Sort"]
    assert _tags.tags.get("albumsort") == "MyAlbum Sort"
    assert _tags.tags.get("albumartistsort") == ["MyAlbumArtist Sort"]


def test_parse_metadata_from_apev2tags() -> None:
    """
    Test parsing of metadata from APEv2 tags (WavPack).

    Uses parse_tags_mutagen directly since the minimal WavPack fixture
    does not contain valid audio data for ffprobe to parse.
    """
    result = parse_tags_mutagen(FILE_WV)
    assert result.get("album") == "MyAlbum"
    assert result.get("title") == "MyTitle"
    assert result.get("albumartist") == "MyArtist"
    assert result.get("artist") == "MyArtist"
    assert result.get("artists") == ["MyArtist", "MyArtist2"]
    assert result.get("genre") == ["Genre1", "Genre2"]
    assert result.get("musicbrainzalbumartistid") == ["abcdefg"]
    assert result.get("musicbrainzartistid") == ["abcdefg"]
    assert result.get("musicbrainzreleasegroupid") == "abcdefg"
    assert result.get("musicbrainzrecordingid") == "abcdefg"
    # test track/disc (APEv2 uses "5/12" format like ID3)
    assert result.get("track") == "5/12"
    assert result.get("disc") == "1/2"
    # test year
    assert result.get("date") == "2022"
    # test sort tags (artistsort/albumartistsort returned as lists to match ID3 behavior)
    assert result.get("titlesort") == "MyTitle Sort"
    assert result.get("artistsort") == ["MyArtist Sort"]
    assert result.get("albumsort") == "MyAlbum Sort"
    assert result.get("albumartistsort") == ["MyAlbumArtist Sort"]


def test_id3_musicbrainz_ufid_strips_trailing_null() -> None:
    """
    Strip a trailing NUL terminator from the MusicBrainz UFID frame data.

    Some taggers (e.g. Picard) append a NUL to the UFID data; without stripping
    it the recording MBID is malformed and breaks import and MusicBrainz
    lookups.

    See https://github.com/music-assistant/support/issues/5906
    """
    ufid = UFID(  # type: ignore[no-untyped-call]
        owner="http://musicbrainz.org",
        data=b"1e74cd4c-cfa7-4bdb-99da-41869f5f1171\x00",
    )

    mock_tags = MagicMock()
    mock_tags.get = lambda key: ufid if key == "UFID:http://musicbrainz.org" else None
    result = _parse_id3_tags(mock_tags)

    assert result["musicbrainzrecordingid"] == "1e74cd4c-cfa7-4bdb-99da-41869f5f1171"


async def test_parse_metadata_from_flac_with_multiple_artist_fields() -> None:
    """Test parsing of FLAC file with multiple ARTIST fields (per Vorbis spec)."""
    _tags = await tags.async_parse_tags(FILE_FLAC)
    assert _tags.album == "Test Album"
    assert _tags.title == "Test Track"
    # Multiple ARTIST fields should be treated as authoritative list
    assert _tags.artists == ("Artist One", "Artist Two", "Artist Three")
    # Multiple ALBUMARTIST fields should be treated as authoritative list
    assert _tags.album_artists == ("Album Artist 1", "Album Artist 2")
    assert _tags.genres == ("Rock", "Pop")
    assert _tags.year == 2024
    # MusicBrainz IDs
    assert _tags.musicbrainz_artistids == ("mb-artist-id-1", "mb-artist-id-2", "mb-artist-id-3")
    assert _tags.musicbrainz_albumartistids == ("mb-albumartist-id-1", "mb-albumartist-id-2")
    assert _tags.musicbrainz_recordingid == "mb-track-id"
    # Track/disc from Vorbis comments
    assert _tags.track == 5
    assert _tags.disc == 1


async def test_parse_metadata_from_filename() -> None:
    """Test parsing of parsing metadata from filename."""
    filename = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle without Tags.mp3"))
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album is None
    assert _tags.title == "MyTitle without Tags"
    assert _tags.duration in (1.0, 1.032)
    assert _tags.album_artists == ()
    assert _tags.artists == ("MyArtist",)
    assert _tags.genres == ()
    assert _tags.musicbrainz_albumartistids == ()
    assert _tags.musicbrainz_artistids == ()
    assert _tags.musicbrainz_releasegroupid is None
    assert _tags.musicbrainz_recordingid is None


async def test_parse_metadata_from_invalid_filename() -> None:
    """Test parsing of parsing metadata from (invalid) filename."""
    filename = str(RESOURCES_DIR.joinpath("test.mp3"))
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album is None
    assert _tags.title == "test"
    assert _tags.duration in (1.0, 1.032)
    assert _tags.album_artists == ()
    assert _tags.artists == (UNKNOWN_ARTIST,)
    assert _tags.genres == ()
    assert _tags.musicbrainz_albumartistids == ()
    assert _tags.musicbrainz_artistids == ()
    assert _tags.musicbrainz_releasegroupid is None
    assert _tags.musicbrainz_recordingid is None


def test_split_artists_with_expected_count() -> None:
    """Test splitting artists guided by expected count (from MB IDs)."""
    # With expected_count=3, should split on extra splitters to reach target
    result = split_artists("Shabson, Krgovich & Harris", expected_count=3)
    assert result == ("Shabson", "Krgovich", "Harris")

    # With expected_count=3, ampersands should split
    result = split_artists("Shabson & Krgovich & Harris", expected_count=3)
    assert result == ("Shabson", "Krgovich", "Harris")

    # With expected_count=3, commas should split
    result = split_artists("Shabson, Krgovich, Harris", expected_count=3)
    assert result == ("Shabson", "Krgovich", "Harris")

    # With expected_count=1, should NOT split at all
    result = split_artists("Shabson & Krgovich", expected_count=1)
    assert result == ("Shabson & Krgovich",)

    # With expected_count=None (no MB IDs), should NOT split on extra splitters
    result = split_artists("Shabson & Krgovich", expected_count=None)
    assert result == ("Shabson & Krgovich",)

    # With expected_count=0 (no MB IDs), should NOT split on extra splitters
    result = split_artists("Shabson & Krgovich", expected_count=0)
    assert result == ("Shabson & Krgovich",)


def test_split_artists_featuring() -> None:
    """Test that featuring splitters always work regardless of expected_count."""
    # "feat." should always split, even with no expected_count
    result = split_artists("John Lennon feat. Yoko Ono", expected_count=None)
    assert result == ("John Lennon", "Yoko Ono")

    # "feat." should split even with expected_count=1 (featuring overrides)
    # Actually, expected_count=1 means single artist, so we return as-is
    result = split_artists("John Lennon feat. Yoko Ono", expected_count=1)
    assert result == ("John Lennon feat. Yoko Ono",)

    # "featuring" should work
    result = split_artists("Artist A featuring Artist B", expected_count=None)
    assert result == ("Artist A", "Artist B")

    # "ft." should work
    result = split_artists("Artist A ft. Artist B", expected_count=None)
    assert result == ("Artist A", "Artist B")

    # " presents " should split without disturbing an ampersand inside an artist name
    result = split_artists("Above & Beyond presents OceanLab", expected_count=None)
    assert result == ("Above & Beyond", "OceanLab")


def test_split_artists_no_oversplit() -> None:
    """Test that split_artists stops at expected_count and doesn't over-split."""
    # Hall & Oates is a duo, with 2 MB IDs we should split on feat. first
    # and get exactly 2 artists
    result = split_artists("Hall & Oates feat. David Ruffin", expected_count=2)
    assert result == ("Hall & Oates", "David Ruffin")

    # With 3 MB IDs, we should split further
    result = split_artists("Hall & Oates feat. David Ruffin", expected_count=3)
    assert result == ("Hall", "Oates", "David Ruffin")

    # Simon & Garfunkel with 1 MB ID (the duo) should stay as one
    result = split_artists("Simon & Garfunkel", expected_count=1)
    assert result == ("Simon & Garfunkel",)

    # Simon & Garfunkel with 2 MB IDs (Paul + Art) should split
    result = split_artists("Simon & Garfunkel", expected_count=2)
    assert result == ("Simon", "Garfunkel")


def test_split_artists_with_not_split() -> None:
    """Test that 'with' is only split when we have MB ID evidence."""
    # "with" should NOT split without expected_count (could be artist name)
    result = split_artists("Jerk With a Bomb", expected_count=None)
    assert result == ("Jerk With a Bomb",)

    # "with" should NOT split with expected_count=1
    result = split_artists("Jerk With a Bomb", expected_count=1)
    assert result == ("Jerk With a Bomb",)

    # "with" SHOULD split when expected_count=2 indicates multiple artists
    result = split_artists("Artist A with Artist B", expected_count=2)
    assert result == ("Artist A", "Artist B")


def _create_mock_vorbis_tags(tag_dict: dict[str, list[str]]) -> MagicMock:
    """
    Create a mock VCommentDict with the given tags.

    :param tag_dict: Dictionary mapping tag names to lists of values.
    """
    mock = MagicMock()
    mock.get = lambda key: tag_dict.get(key.upper())
    return mock


def test_parse_vorbis_tags_multiple_artist_fields() -> None:
    """Test that multiple ARTIST fields are treated as authoritative artist list."""
    # Per Vorbis spec: multiple ARTIST fields should list all artists
    mock_tags = _create_mock_vorbis_tags(
        {
            "TITLE": ["My Song"],
            "ALBUM": ["My Album"],
            "ARTIST": ["Artist 1", "Artist 2", "Artist 3"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Multiple ARTIST fields should be stored as "artists" (plural)
    assert result.get("artists") == ["Artist 1", "Artist 2", "Artist 3"]
    # Single "artist" key should NOT be set when multiple artists are present
    assert "artist" not in result
    assert result.get("title") == "My Song"
    assert result.get("album") == "My Album"


def test_parse_vorbis_tags_single_artist_field() -> None:
    """Test that a single ARTIST field is stored as singular artist."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "TITLE": ["My Song"],
            "ARTIST": ["Single Artist"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Single ARTIST should use singular key for normal parsing logic
    assert result.get("artist") == "Single Artist"
    assert "artists" not in result


def test_parse_vorbis_tags_multiple_albumartist_fields() -> None:
    """Test that multiple ALBUMARTIST fields are treated as authoritative list."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ALBUMARTIST": ["Album Artist 1", "Album Artist 2"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Multiple ALBUMARTIST fields should be stored as "albumartists" (plural)
    assert result.get("albumartists") == ["Album Artist 1", "Album Artist 2"]
    assert "albumartist" not in result


def test_parse_vorbis_tags_single_albumartist_field() -> None:
    """Test that a single ALBUMARTIST field is stored as singular."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ALBUMARTIST": ["Single Album Artist"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    assert result.get("albumartist") == "Single Album Artist"
    assert "albumartists" not in result


def test_parse_vorbis_tags_explicit_artists_tag_takes_precedence() -> None:
    """Test that explicit ARTISTS tag takes precedence over multiple ARTIST fields."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ARTIST": ["Artist A", "Artist B"],  # Multiple ARTIST fields
            "ARTISTS": [
                "Explicit Artist 1",
                "Explicit Artist 2",
                "Explicit Artist 3",
            ],  # Explicit tag
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # ARTISTS tag should take precedence
    assert result.get("artists") == ["Explicit Artist 1", "Explicit Artist 2", "Explicit Artist 3"]


def test_parse_vorbis_tags_musicbrainz_ids() -> None:
    """Test that MusicBrainz IDs are parsed correctly from Vorbis tags."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ARTIST": ["Artist 1", "Artist 2"],
            "MUSICBRAINZ_ARTISTID": ["mb-id-1", "mb-id-2"],
            "MUSICBRAINZ_ALBUMID": ["mb-album-id"],
            "MUSICBRAINZ_TRACKID": ["mb-track-id"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    assert result.get("musicbrainzartistid") == ["mb-id-1", "mb-id-2"]
    assert result.get("musicbrainzalbumid") == "mb-album-id"
    assert result.get("musicbrainzrecordingid") == "mb-track-id"


def test_parse_vorbis_multi_value_releasetype() -> None:
    """Repeated RELEASETYPE Vorbis fields are joined into a single value."""
    mock_tags = _create_mock_vorbis_tags({"RELEASETYPE": ["album", "live"]})
    result = _parse_vorbis_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def _create_mock_apev2_tags(tag_dict: dict[str, str]) -> MagicMock:
    r"""
    Create a mock APEv2 tags object.

    :param tag_dict: Dictionary mapping tag names to values (use \x00 for multi-value).
    """
    mock = MagicMock()
    mock.__contains__ = lambda _, key: key in tag_dict
    mock.__getitem__ = lambda _, key: tag_dict[key]
    mock.keys = lambda: tag_dict.keys()
    return mock


def test_parse_apev2_tags_multi_value_artists() -> None:
    """Test that APEv2 multi-value fields (null-separated) are parsed correctly."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Title": "My Song",
            "Album": "My Album",
            "Artist": "Single Artist",
            "Artists": "Artist 1\x00Artist 2\x00Artist 3",  # Null-separated
        }
    )

    result = _parse_apev2_tags(mock_tags)

    assert result.get("title") == "My Song"
    assert result.get("album") == "My Album"
    assert result.get("artist") == "Single Artist"
    assert result.get("artists") == ["Artist 1", "Artist 2", "Artist 3"]


def test_parse_apev2_tags_musicbrainz_ids() -> None:
    """Test that MusicBrainz IDs are parsed correctly from APEv2 tags."""
    mock_tags = _create_mock_apev2_tags(
        {
            "MUSICBRAINZ_ARTISTID": "mb-id-1\x00mb-id-2",  # Multi-value
            "MUSICBRAINZ_ALBUMID": "mb-album-id",
            "MUSICBRAINZ_TRACKID": "mb-track-id",  # Recording ID in APEv2
            "MUSICBRAINZ_RELEASEGROUPID": "mb-rg-id",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    assert result.get("musicbrainzartistid") == ["mb-id-1", "mb-id-2"]
    assert result.get("musicbrainzalbumid") == "mb-album-id"
    assert result.get("musicbrainzrecordingid") == "mb-track-id"
    assert result.get("musicbrainzreleasegroupid") == "mb-rg-id"


def test_parse_apev2_multi_value_musicbrainz_albumtype() -> None:
    """Null-separated MUSICBRAINZ_ALBUMTYPE values are joined into a single value."""
    mock_tags = _create_mock_apev2_tags({"MUSICBRAINZ_ALBUMTYPE": "album\x00live"})
    result = _parse_apev2_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def test_parse_apev2_tags_genre_multi_value() -> None:
    """Test that APEv2 genre with multiple values is parsed correctly."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Genre": "Rock\x00Pop\x00Jazz",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    assert result.get("genre") == ["Rock", "Pop", "Jazz"]


def test_parse_apev2_tags_null_separated_artists() -> None:
    """Test that APEv2 null-separated Artist field is parsed as multiple artists."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Artist": "ave;new\x00佐倉紗織",
            "Album Artist": "Album Artist A\x00Album Artist B",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    # Multiple null-separated values should be stored as "artists" (plural)
    assert result.get("artists") == ["ave;new", "佐倉紗織"]
    assert result.get("albumartists") == ["Album Artist A", "Album Artist B"]
    # Singular keys should not be set
    assert "artist" not in result
    assert "albumartist" not in result


def test_parse_apev2_tags_single_artist() -> None:
    """Test that APEv2 single Artist field is parsed as singular."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Artist": "Single Artist",
            "Album Artist": "Single Album Artist",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    # Single value should be stored as "artist" (singular)
    assert result.get("artist") == "Single Artist"
    assert result.get("albumartist") == "Single Album Artist"
    # Plural keys should not be set
    assert "artists" not in result
    assert "albumartists" not in result


def test_parse_mp4_multi_value_musicbrainz_albumtype() -> None:
    """Multi-value MP4 freeform album type entries are joined into a single value."""
    mock_tags = MagicMock()
    mock_tags.__contains__ = lambda _, key: key == "----:com.apple.iTunes:MusicBrainz Album Type"
    mock_tags.__getitem__ = lambda _, _k: [b"album", b"live"]
    result = _parse_mp4_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def test_parse_id3_multi_value_musicbrainz_albumtype() -> None:
    """Multi-value TXXX:MusicBrainz Album Type frame entries are joined into a single value."""
    frame = MagicMock()
    frame.text = ["album", "live"]
    mock_tags = MagicMock()
    mock_tags.get = lambda key: frame if key == "TXXX:MusicBrainz Album Type" else None
    result = _parse_id3_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def test_vorbis_multiple_artist_fields_semicolon_in_name() -> None:
    """
    Test that multiple ARTIST fields in Vorbis with semicolons are handled correctly.

    Regression test for the "ave;new" edge case per the Vorbis spec:
    - Japanese artist "ave;new" has a semicolon in their name
    - Vorbis allows multiple ARTIST (singular) fields for multi-artist tracks
    - The semicolon within "ave;new" must NOT cause additional splitting

    Correct Vorbis tagging (per https://xiph.org/vorbis/doc/v-comment.html):
        ARTIST=ave;new
        ARTIST=佐倉紗織
        MUSICBRAINZ_ARTISTID=2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
        MUSICBRAINZ_ARTISTID=822c07bd-1f8a-4fef-acdb-8acfe82fbef5

    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    # Simulate Vorbis tags with multiple ARTIST fields (correct per Vorbis spec)
    mock_tags = _create_mock_vorbis_tags(
        {
            "TITLE": ["Call My Dears"],
            "ALBUM": ["Lovable"],
            "ARTIST": ["ave;new", "佐倉紗織"],  # Multiple ARTIST fields
            "ARTISTSORT": ["ave;new feat.Sakura, Saori"],
            "MUSICBRAINZ_ARTISTID": [
                "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
                "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
            ],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Multiple ARTIST fields should be stored as "artists" (plural key)
    assert result.get("artists") == ["ave;new", "佐倉紗織"]
    # MusicBrainz Artist IDs should be preserved
    assert result.get("musicbrainzartistid") == [
        "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
    ]

    # Now test that AudioTags.artists property correctly handles the multiple fields
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=None,
        duration=180.0,
        tags=result,
        has_cover_image=False,
        filename="01 - ave;new feat.佐倉紗織 - Call My Dears.flac",
    )

    # The artists property must return exactly 2 artists
    assert audio_tags.artists == ("ave;new", "佐倉紗織")
    # The semicolon in "ave;new" must NOT cause it to be split
    assert "ave" not in audio_tags.artists
    assert "new" not in audio_tags.artists
    # MusicBrainz Artist IDs should match the artist count
    assert audio_tags.musicbrainz_artistids == (
        "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
    )
    assert len(audio_tags.artists) == len(audio_tags.musicbrainz_artistids)


async def test_flac_multiple_artist_fields_semicolon_e2e() -> None:
    """
    End-to-end test: FLAC with multiple ARTIST fields, one containing semicolon.

    Tests real file parsing to ensure the full pipeline correctly handles
    artist names with semicolons when using multiple ARTIST fields per Vorbis spec.

    See: https://xiph.org/vorbis/doc/v-comment.html
    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    audio_tags = await tags.async_parse_tags(FILE_FLAC_SEMICOLON)

    # Verify the artists are correctly parsed without splitting on semicolons
    assert audio_tags.artists == ("ave;new", "佐倉紗織")
    assert "ave" not in audio_tags.artists
    assert "new" not in audio_tags.artists

    # Verify MB Artist IDs match
    assert audio_tags.musicbrainz_artistids == (
        "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
    )
    assert len(audio_tags.artists) == len(audio_tags.musicbrainz_artistids)

    # Verify other tags
    assert audio_tags.title == "Call My Dears"
    assert audio_tags.album == "Lovable"


def test_id3_artist_tag_semicolon_single_mbid() -> None:
    """
    Test that single ARTIST tag with semicolon is not split when 1 MB ID exists.

    Regression test for formats without multi-value ARTISTS tag support (ID3, etc.):
    - Artist name "ave;new" contains a semicolon
    - Single MUSICBRAINZ_ARTISTID confirms this is one artist
    - The semicolon must NOT cause the name to be split into "ave" and "new"

    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    # Simulate ID3 tags: single ARTIST field with semicolon, single MB ID
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="mp3",
        bit_rate=None,
        duration=180.0,
        tags={
            "title": "Colorful",
            "album": "Lovable",
            "artist": "ave;new",
            "musicbrainzartistid": "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        },
        has_cover_image=False,
        filename="01 - ave;new - Colorful.mp3",
    )

    # Single MB ID = single artist, no splitting
    assert audio_tags.artists == ("ave;new",)
    assert audio_tags.musicbrainz_artistids == ("2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",)
    # Verify the semicolon did NOT cause incorrect splitting
    assert "ave" not in audio_tags.artists
    assert "new" not in audio_tags.artists


def test_artists_tag_semicolon_single_mbid() -> None:
    """
    Test that ARTISTS tag with semicolon is not split when 1 MB ID exists.

    Regression test for the ARTISTS (plural) tag path:
    - Artist name "ave;new" contains a semicolon
    - Single MUSICBRAINZ_ARTISTID confirms this is one artist
    - The semicolon must NOT cause the name to be split

    Based on real tags from ave;new's "Lovable" album track "eve".
    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=None,
        duration=180.0,
        tags={
            "title": "eve",
            "album": "Lovable",
            "artist": "ave;new",
            "artists": "ave;new",  # ARTISTS tag with semicolon
            "artistsort": "ave;new",
            "musicbrainzartistid": "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
            "musicbrainzrecordingid": "0389384e-3015-45ba-8a09-d949ff68f9d9",
        },
        has_cover_image=False,
        filename="04 - ave;new - eve.flac",
    )

    # Single MB ID = single artist, ARTISTS tag should NOT be split on semicolon
    assert audio_tags.artists == ("ave;new",)
    assert audio_tags.musicbrainz_artistids == ("2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",)
    # Verify the semicolon did NOT cause incorrect splitting
    assert "ave" not in audio_tags.artists
    assert "new" not in audio_tags.artists


def test_id3_artist_tag_semicolon_multiple_mbids() -> None:
    """
    Test that ARTIST tag with semicolon IS split when multiple MB IDs exist.

    When multiple MusicBrainz Artist IDs are present, the semicolon should be
    treated as a separator between artists.
    """
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="mp3",
        bit_rate=None,
        duration=180.0,
        # musicbrainzartistid can be list[str] from mutagen (dict type is str for ffprobe compat)
        tags={
            "artist": "Artist A;Artist B",
            "musicbrainzartistid": ["id-a", "id-b"],
        },
        has_cover_image=False,
        filename="test.mp3",
    )

    # Multiple MB IDs = semicolon should split
    assert audio_tags.artists == ("Artist A", "Artist B")
    assert audio_tags.musicbrainz_artistids == ("id-a", "id-b")


def test_id3_albumartist_tag_semicolon_single_mbid() -> None:
    """Test that ALBUMARTIST tag with semicolon is not split when 1 MB Album Artist ID exists."""
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="mp3",
        bit_rate=None,
        duration=180.0,
        tags={
            "albumartist": "ave;new",
            "musicbrainzalbumartistid": "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        },
        has_cover_image=False,
        filename="test.mp3",
    )

    # Single MB Album Artist ID = single artist, no splitting
    assert audio_tags.album_artists == ("ave;new",)
    assert audio_tags.musicbrainz_albumartistids == ("2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",)


def _read_replaygain_track_gain(path: str) -> str | None:
    """Read REPLAYGAIN_TRACK_GAIN from a file using mutagen (format-agnostic)."""
    audio = mutagen.File(path)
    if audio is None or audio.tags is None:
        return None
    tag_key_mp4 = "----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN"
    if tag_key_mp4 in audio.tags:
        val = audio.tags[tag_key_mp4][0]
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)
    if "TXXX:REPLAYGAIN_TRACK_GAIN" in audio.tags:
        return str(audio.tags["TXXX:REPLAYGAIN_TRACK_GAIN"].text[0])
    if "REPLAYGAIN_TRACK_GAIN" in audio.tags:
        return str(audio.tags["REPLAYGAIN_TRACK_GAIN"][0])
    return None


@pytest.mark.parametrize(
    "source",
    [FILE_MP3, FILE_M4A, FILE_FLAC, FILE_WV],
)
async def test_write_replaygain_track_gain_roundtrip(tmp_path: pathlib.Path, source: str) -> None:
    """Write a REPLAYGAIN_TRACK_GAIN tag and verify the value is read back."""
    dest = tmp_path / pathlib.Path(source).name
    shutil.copy(source, dest)

    assert await write_replaygain_track_gain(str(dest), -5.3) is True
    assert _read_replaygain_track_gain(str(dest)) == "-5.30 dB"

    # verify overwrite replaces the previous value
    assert await write_replaygain_track_gain(str(dest), -2.1) is True
    assert _read_replaygain_track_gain(str(dest)) == "-2.10 dB"


async def test_write_replaygain_track_gain_missing_file(tmp_path: pathlib.Path) -> None:
    """Return False if the file does not exist or cannot be opened."""
    assert await write_replaygain_track_gain(str(tmp_path / "nope.mp3"), -5.0) is False


async def test_write_replaygain_track_gain_read_only(tmp_path: pathlib.Path) -> None:
    """Return False if the file cannot be written to."""
    dest = tmp_path / "readonly.mp3"
    shutil.copy(FILE_MP3, dest)
    dest.chmod(0o444)
    try:
        assert await write_replaygain_track_gain(str(dest), -5.0) is False
    finally:
        # restore permissions so tmp_path cleanup can remove the file
        dest.chmod(0o644)


VALID_MBID = "73c69a4b-1f9e-4c8c-b8bb-3ba903af1c3f"


def test_clean_mbid() -> None:
    """Test cleaning/canonicalizing MusicBrainz identifiers from file tags."""
    assert clean_mbid(VALID_MBID) == VALID_MBID
    # uppercase hex digits are canonicalized to lowercase
    assert clean_mbid(VALID_MBID.upper()) == VALID_MBID
    # trailing NUL bytes and surrounding whitespace are stripped
    assert clean_mbid(f"{VALID_MBID}\x00") == VALID_MBID
    assert clean_mbid(f"  {VALID_MBID} \n") == VALID_MBID
    # non-UUID values are rejected
    assert clean_mbid("CAAE0466 1G4B0N3 07800NE1") is None
    assert clean_mbid("abcdefg") is None
    assert clean_mbid("") is None
    assert clean_mbid(None) is None
    # non-string values (e.g. repeated NFO elements parsed as a list) are rejected
    assert clean_mbid([VALID_MBID, VALID_MBID]) is None  # type: ignore[arg-type]


def test_parse_id3_ufid_frame_binary_data() -> None:
    """A UFID frame with non-UTF-8 binary data must not break parsing of other tags."""
    ufid = UFID(  # type: ignore[no-untyped-call]
        owner="http://musicbrainz.org",
        data=bytes.fromhex("73c69a4b1f9e4c8cb8bb3ba903af1c3f"),
    )
    title_frame = MagicMock()
    title_frame.text = ["MyTitle"]
    frames = {"UFID:http://musicbrainz.org": ufid, "TIT2": title_frame}

    mock_tags = MagicMock()
    mock_tags.get = frames.get
    result = _parse_id3_tags(mock_tags)

    assert result.get("title") == "MyTitle"
    # the garbled identifier is rejected downstream by clean_mbid
    assert clean_mbid(result.get("musicbrainzrecordingid")) is None


async def test_parse_ufid_frame_with_dirty_payload(tmp_path: pathlib.Path) -> None:
    """
    A UFID payload with a NUL terminator must still yield a usable recording id.

    Regression test for https://github.com/music-assistant/support/issues/5906
    where such files failed to import with "Invalid MusicBrainz identifier".
    """
    dest = tmp_path / "ufid.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(  # type: ignore[no-untyped-call]
        UFID(owner="http://musicbrainz.org", data=f"{VALID_MBID}\x00".encode("ascii"))  # type: ignore[no-untyped-call]
    )
    id3.save()

    _tags = await tags.async_parse_tags(str(dest))
    assert clean_mbid(_tags.musicbrainz_recordingid) == VALID_MBID


def _tags_with(raw_tags: dict[str, Any]) -> tags.AudioTags:
    """Build an AudioTags instance carrying the given raw tags."""
    return tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=None,
        duration=180.0,
        tags=raw_tags,
        has_cover_image=False,
        filename="track.flac",
    )


@pytest.mark.parametrize(
    ("raw_tags", "expected"),
    [
        # Vorbis comments and iTunes atoms, as used by FLAC, Ogg, Opus, WavPack and M4A
        ({"originaldate": "1978-06-01", "date": "2015-03-07"}, "1978-06-01"),
        ({"originalyear": "1978", "date": "2015-03-07"}, "1978-01-01"),
        # ffmpeg hands ID3 frames over under their raw frame name
        ({"tdor": "1978-06-01", "date": "2015-03-07"}, "1978-06-01"),
        ({"tory": "1978", "date": "2015-03-07"}, "1978-01-01"),
        # a full date beats a bare year, whichever tags they arrive in
        ({"originaldate": "1978-06-01", "originalyear": "1977"}, "1978-06-01"),
        ({"tdor": "1978-06-01", "tory": "1977"}, "1978-06-01"),
        ({"originaldate": "1978-06-01", "tory": "1977"}, "1978-06-01"),
        # a date tagged to the month keeps the month
        ({"originaldate": "1978-06"}, "1978-06-01"),
        # without an original date the release's own date is used
        ({"date": "2015-03-07"}, "2015-03-07"),
        ({"date": "2015"}, "2015-01-01"),
        # an unusable original date falls through instead of blocking
        ({"originaldate": "0000", "originalyear": "1978"}, "1978-01-01"),
        # nothing usable
        ({}, None),
        ({"date": "not a date"}, None),
    ],
)
def test_release_date(raw_tags: dict[str, str], expected: str | None) -> None:
    """The original release date wins over the date of the release the file came from."""
    release_date = _tags_with(raw_tags).release_date
    assert release_date == (
        datetime.fromisoformat(expected).replace(tzinfo=UTC) if expected else None
    )


@pytest.mark.parametrize(
    ("raw_tags", "expected"),
    [
        # the release's own date wins, whichever tag format it arrives in
        ({"date": "2015-03-07", "originaldate": "1978-06-01"}, 2015),
        ({"date": "2015-03-07", "tdor": "1978-06-01"}, 2015),
        ({"date": "2015-03-07", "originalyear": "1978"}, 2015),
        # without it, the original release is the best the file offers
        ({"originaldate": "1978-06-01"}, 1978),
        ({"tory": "1978"}, 1978),
        ({"originaldate": "1978-06-01", "originalyear": "1977"}, 1978),
        ({}, None),
    ],
)
def test_album_year(raw_tags: dict[str, str], expected: int | None) -> None:
    """The album year is the date of the release itself, not of the original."""
    assert _tags_with(raw_tags).year == expected


def test_a_compilation_dates_the_album_and_its_tracks_apart() -> None:
    """A track keeps its original release date while the album keeps the compilation's."""
    audio_tags = _tags_with({"date": "2015-03-07", "originaldate": "1978-06-01"})

    assert audio_tags.release_date == datetime(1978, 6, 1, tzinfo=UTC)
    assert audio_tags.year == 2015


async def test_original_release_date_is_read_from_an_id3_file(tmp_path: pathlib.Path) -> None:
    """The ID3 frame holding the original release date is the one ffmpeg does not map."""
    dest = tmp_path / "original_date.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.setall("TDRC", [TDRC(encoding=3, text=["2015-03-07"])])  # type: ignore[no-untyped-call]
    id3.setall("TDOR", [TDOR(encoding=3, text=["1978-06-01"])])  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.release_date == datetime(1978, 6, 1, tzinfo=UTC)
    assert _tags.year == 2015


async def test_original_release_date_is_read_from_an_m4a_file(tmp_path: pathlib.Path) -> None:
    """The original date sits in an iTunes freeform atom, which ffprobe drops."""
    dest = tmp_path / "original_date.m4a"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["\xa9day"] = ["2015-03-07"]
    mp4["----:com.apple.iTunes:originaldate"] = [MP4FreeForm(b"1978-06-01")]  # type: ignore[no-untyped-call]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.release_date == datetime(1978, 6, 1, tzinfo=UTC)
    assert _tags.year == 2015


async def test_audiobook_credits_are_read_from_an_m4b_file(tmp_path: pathlib.Path) -> None:
    """The narrator and writer atoms of an m4b name every person, not just the first."""
    dest = tmp_path / "book.m4b"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:NARRATOR"] = [
        MP4FreeForm(b"Jane Reader"),  # type: ignore[no-untyped-call]
        MP4FreeForm(b"John Voice"),  # type: ignore[no-untyped-call]
    ]
    mp4["----:com.apple.iTunes:WRITER"] = [
        MP4FreeForm(b"Jane Austen"),  # type: ignore[no-untyped-call]
        MP4FreeForm(b"John Writer"),  # type: ignore[no-untyped-call]
    ]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.authors == ("Jane Austen", "John Writer")
    assert _tags.narrators == ("Jane Reader", "John Voice")


async def test_audiobook_narrator_falls_back_to_the_composer_atom(tmp_path: pathlib.Path) -> None:
    """Audible style m4b files name the narrator in the composer atom."""
    dest = tmp_path / "book.m4b"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["\xa9wrt"] = ["Jane Reader", "John Voice"]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.narrators == ("Jane Reader", "John Voice")


async def test_audiobook_credits_are_read_from_an_mp3_file(tmp_path: pathlib.Path) -> None:
    """A narrator frame wins over the composer frame, and both names survive."""
    dest = tmp_path / "book.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="NARRATOR", text=["Jane Reader", "John Voice"]))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="WRITER", text=["Jane Austen", "John Writer"]))  # type: ignore[no-untyped-call]
    id3.add(TCOM(encoding=3, text=["Someone Else"]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.authors == ("Jane Austen", "John Writer")
    assert _tags.narrators == ("Jane Reader", "John Voice")


async def test_audiobook_narrator_falls_back_to_the_composer_frame(tmp_path: pathlib.Path) -> None:
    """The composer frame is null separated, so both narrators have to come through."""
    dest = tmp_path / "book.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TCOM(encoding=3, text=["Jane Reader", "John Voice"]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.narrators == ("Jane Reader", "John Voice")


async def test_audiobook_credits_are_read_from_a_flac_file(tmp_path: pathlib.Path) -> None:
    """Vorbis comments repeat a field per name."""
    dest = tmp_path / "book.flac"
    shutil.copy(FILE_FLAC, dest)
    flac = FLAC(str(dest))  # type: ignore[no-untyped-call]
    flac["NARRATOR"] = ["Jane Reader", "John Voice"]
    flac["WRITER"] = ["Jane Austen", "John Writer"]
    flac.save()

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.authors == ("Jane Austen", "John Writer")
    assert _tags.narrators == ("Jane Reader", "John Voice")


def test_audiobook_credits_are_read_from_a_wavpack_file(tmp_path: pathlib.Path) -> None:
    """
    APEv2 separates the names with a null byte.

    Uses parse_tags_mutagen directly since the minimal WavPack fixture
    does not contain valid audio data for ffprobe to parse.
    """
    dest = tmp_path / "book.wv"
    shutil.copy(FILE_WV, dest)
    ape = APEv2(str(dest))  # type: ignore[no-untyped-call]
    ape["NARRATOR"] = ["Jane Reader", "John Voice"]
    ape["WRITER"] = ["Jane Austen", "John Writer"]
    ape.save(str(dest))

    result = parse_tags_mutagen(str(dest))

    assert result.get("narrators") == ["Jane Reader", "John Voice"]
    assert result.get("writers") == ["Jane Austen", "John Writer"]


async def test_audiobook_author_falls_back_to_the_album_artist(tmp_path: pathlib.Path) -> None:
    """Taggers without a writer field reach for the album artist."""
    dest = tmp_path / "book.m4b"
    shutil.copy(FILE_M4A, dest)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.authors == ("MyArtist",)
    assert _tags.narrators == ()


async def test_audiobook_author_is_not_invented_from_the_filename() -> None:
    """An author becomes a real artist, so an untagged book may not name one."""
    filename = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle without Tags.mp3"))

    _tags = await tags.async_parse_tags(filename)

    assert _tags.artists == ("MyArtist",)
    assert _tags.authors == ()


async def test_audiobook_author_is_never_the_unknown_artist() -> None:
    """The unknown artist placeholder may not end up in the library as an author."""
    filename = str(RESOURCES_DIR.joinpath("test.mp3"))

    _tags = await tags.async_parse_tags(filename)

    assert _tags.artists == (UNKNOWN_ARTIST,)
    assert _tags.authors == ()


async def test_audiobook_series_is_read_from_an_m4b_file(tmp_path: pathlib.Path) -> None:
    """The series atoms of an m4b are freeform, which ffprobe only sometimes surfaces."""
    dest = tmp_path / "book.m4b"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:Series"] = [MP4FreeForm(b"The Expanse")]  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:Series-Part"] = [MP4FreeForm(b"3")]  # type: ignore[no-untyped-call]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert (_tags.series, _tags.series_part) == ("The Expanse", 3.0)


async def test_audiobook_series_is_read_from_an_mp3_file(tmp_path: pathlib.Path) -> None:
    """MP3 keeps the series in user defined frames."""
    dest = tmp_path / "book.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="SERIES", text=["The Expanse"]))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="SERIES-PART", text=["3"]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert (_tags.series, _tags.series_part) == ("The Expanse", 3.0)


async def test_audiobook_series_is_read_from_a_flac_file(tmp_path: pathlib.Path) -> None:
    """Vorbis comments name the series fields directly."""
    dest = tmp_path / "book.flac"
    shutil.copy(FILE_FLAC, dest)
    flac = FLAC(str(dest))  # type: ignore[no-untyped-call]
    flac["SERIES"] = ["The Expanse"]
    flac["SERIES-PART"] = ["3"]
    flac.save()

    _tags = await tags.async_parse_tags(str(dest))

    assert (_tags.series, _tags.series_part) == ("The Expanse", 3.0)


def test_audiobook_series_is_read_from_a_wavpack_file(tmp_path: pathlib.Path) -> None:
    """
    APEv2 names the series fields directly too.

    Uses parse_tags_mutagen directly since the minimal WavPack fixture
    does not contain valid audio data for ffprobe to parse.
    """
    dest = tmp_path / "book.wv"
    shutil.copy(FILE_WV, dest)
    ape = APEv2(str(dest))  # type: ignore[no-untyped-call]
    ape["SERIES"] = "The Expanse"
    ape["SERIES-PART"] = "3"
    ape.save(str(dest))

    result = parse_tags_mutagen(str(dest))

    assert result.get("series") == "The Expanse"
    assert result.get("seriespart") == "3"


async def test_audiobook_series_sequence_may_be_fractional(tmp_path: pathlib.Path) -> None:
    """A novella between two books is tagged 1.5."""
    dest = tmp_path / "book.m4b"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:Series"] = [MP4FreeForm(b"The Expanse")]  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:Series-Part"] = [MP4FreeForm(b"1.5")]  # type: ignore[no-untyped-call]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.series_part == 1.5


async def test_audiobook_series_sequence_keeps_a_non_numeric_value(
    tmp_path: pathlib.Path,
) -> None:
    """Not every tagger numbers the parts."""
    dest = tmp_path / "book.m4b"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:Series"] = [MP4FreeForm(b"Discworld")]  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:Series-Part"] = [MP4FreeForm(b"Guards")]  # type: ignore[no-untyped-call]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.series_part == "Guards"


async def test_audiobook_without_a_series_tag_has_none() -> None:
    """No series tag must not end up clearing what is stored."""
    _tags = await tags.async_parse_tags(FILE_M4A)

    assert (_tags.series, _tags.series_part) == (None, None)


COMPOSER_MBID = "24f1766e-9635-4d58-a4d4-9413f9f98a4c"
PARENT_WORK_MBID = "6bc0d3d3-f6e1-3f87-9a4e-2e7a8a4f45d2"
WORK_MBID = "d6f6f1d4-3b70-4b0f-9d0a-7c8c1f2c4e11"


async def test_classical_tags_are_read_from_an_id3v24_file(tmp_path: pathlib.Path) -> None:
    """Picard's ID3v2.4 frames give every classical credit, the work and the movement."""
    dest = tmp_path / "classical.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TCOM(encoding=3, text=["Ludwig van Beethoven", "Franz Liszt"]))  # type: ignore[no-untyped-call]
    id3.add(TSOC(encoding=3, text=["Beethoven, Ludwig van", "Liszt, Franz"]))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="MusicBrainz Composer Id", text=[COMPOSER_MBID]))  # type: ignore[no-untyped-call]
    id3.add(TPE3(encoding=3, text=["Carlos Kleiber", "Herbert von Karajan"]))  # type: ignore[no-untyped-call]
    id3.add(
        TMCL(  # type: ignore[no-untyped-call]
            encoding=3,
            people=[["piano", "Martha Argerich"], ["performer", "Wiener Philharmoniker"]],
        )
    )
    id3.add(TIPL(encoding=3, people=[["arranger", "Franz Liszt"], ["producer", "Someone"]]))  # type: ignore[no-untyped-call]
    id3.add(TEXT(encoding=3, text=["Friedrich Schiller"]))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="WORK", text=["Symphonies", "Symphony No. 5"]))  # type: ignore[no-untyped-call]
    id3.add(
        TXXX(  # type: ignore[no-untyped-call]
            encoding=3, desc="MusicBrainz Work Id", text=[PARENT_WORK_MBID, WORK_MBID]
        )
    )
    id3.add(MVNM(encoding=3, text=["Allegro con brio"]))  # type: ignore[no-untyped-call]
    id3.add(MVIN(encoding=3, text=["1/4"]))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="is_classical", text=["1"]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.composers == ("Ludwig van Beethoven", "Franz Liszt")
    assert _tags.composer_sort_names == ("Beethoven, Ludwig van", "Liszt, Franz")
    assert _tags.musicbrainz_composerids == (COMPOSER_MBID,)
    assert _tags.conductors == ("Carlos Kleiber", "Herbert von Karajan")
    assert _tags.performers == (("Martha Argerich", "piano"), ("Wiener Philharmoniker", None))
    assert _tags.arrangers == ("Franz Liszt",)
    assert _tags.lyricists == ("Friedrich Schiller",)
    assert _tags.works == ("Symphonies", "Symphony No. 5")
    assert _tags.work == "Symphony No. 5"
    assert _tags.musicbrainz_workids == (PARENT_WORK_MBID, WORK_MBID)
    assert _tags.musicbrainz_workid == WORK_MBID
    assert (_tags.movement_name, _tags.movement_number, _tags.movement_total) == (
        "Allegro con brio",
        1,
        4,
    )
    assert _tags.is_classical


async def test_classical_tags_are_read_from_an_id3v23_file(tmp_path: pathlib.Path) -> None:
    """ID3v2.3 keeps performers and arrangers in one people list and the composer sort in TXXX."""
    dest = tmp_path / "classical.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TCOM(encoding=3, text=["Ludwig van Beethoven", "Franz Liszt"]))  # type: ignore[no-untyped-call]
    id3.add(TXXX(encoding=3, desc="COMPOSERSORT", text=["Beethoven, Ludwig van"]))  # type: ignore[no-untyped-call]
    id3.add(
        IPLS(  # type: ignore[no-untyped-call]
            encoding=3,
            people=[["arranger", "Franz Liszt"], ["piano", "Glenn Gould"], ["engineer", "X"]],
        )
    )
    id3.add(MVIN(encoding=3, text=["2/4"]))  # type: ignore[no-untyped-call]
    id3.update_to_v23()  # type: ignore[no-untyped-call]
    id3.save(v2_version=3, v23_sep="; ")

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.composers == ("Ludwig van Beethoven", "Franz Liszt")
    assert _tags.composer_sort_names == ("Beethoven, Ludwig van",)
    assert _tags.performers == (("Glenn Gould", "piano"),)
    assert _tags.arrangers == ("Franz Liszt",)
    assert (_tags.movement_number, _tags.movement_total) == (2, 4)


async def test_conductor_is_not_read_as_performer_from_an_id3_file(tmp_path: pathlib.Path) -> None:
    """The conductor frame, which ffprobe calls performer, never credits a performer."""
    dest = tmp_path / "classical.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TPE3(encoding=3, text=["Carlos Kleiber"]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.conductors == ("Carlos Kleiber",)
    assert _tags.performers == ()


@pytest.mark.parametrize(
    ("frames", "expected_work", "expected_grouping"),
    [
        # TIT1 is the grouping, as it always was
        ({"TIT1": "My Grouping"}, None, "My Grouping"),
        # Picard's iTunes compatible style puts the work in TIT1 and the grouping in GRP1
        ({"TIT1": "Symphony No. 5", "GRP1": "My Grouping"}, "Symphony No. 5", "My Grouping"),
        # an explicit work tag wins over TIT1
        ({"TIT1": "Symphony No. 5", "WORK": "The Work"}, "The Work", "Symphony No. 5"),
        ({"TIT1": "Other", "GRP1": "My Grouping", "WORK": "The Work"}, "The Work", "My Grouping"),
        # Picard's ID3v2.3 files carry the work in TIT1 next to its MusicBrainz work id
        (
            {"TIT1": "Orchestersuite Nr. 3 D-Dur, BWV 1068: II. Air", "WORKID": WORK_MBID},
            "Orchestersuite Nr. 3 D-Dur, BWV 1068: II. Air",
            None,
        ),
    ],
)
async def test_mp3_work_and_grouping(
    tmp_path: pathlib.Path,
    frames: dict[str, str],
    expected_work: str | None,
    expected_grouping: str | None,
) -> None:
    """The work comes from TXXX:WORK, else from TIT1 when GRP1 or a MusicBrainz work id marks it."""
    dest = tmp_path / "classical.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    if "TIT1" in frames:
        id3.add(TIT1(encoding=3, text=[frames["TIT1"]]))  # type: ignore[no-untyped-call]
    if "GRP1" in frames:
        id3.add(GRP1(encoding=3, text=[frames["GRP1"]]))  # type: ignore[no-untyped-call]
    if "WORK" in frames:
        id3.add(TXXX(encoding=3, desc="WORK", text=[frames["WORK"]]))  # type: ignore[no-untyped-call]
    if "WORKID" in frames:
        id3.add(TXXX(encoding=3, desc="MusicBrainz Work Id", text=[frames["WORKID"]]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.work == expected_work
    assert _tags.get("grouping") == expected_grouping


async def test_classical_tags_are_read_from_a_flac_file(tmp_path: pathlib.Path) -> None:
    """Vorbis comments repeat a field per value and name the instrument in parentheses."""
    dest = tmp_path / "classical.flac"
    shutil.copy(FILE_FLAC, dest)
    flac = FLAC(str(dest))  # type: ignore[no-untyped-call]
    flac["COMPOSER"] = ["Ludwig van Beethoven", "Franz Liszt"]
    flac["COMPOSERSORT"] = ["Beethoven, Ludwig van", "Liszt, Franz"]
    flac["MUSICBRAINZ_COMPOSERID"] = [COMPOSER_MBID]
    flac["CONDUCTOR"] = ["Carlos Kleiber"]
    flac["PERFORMER"] = ["Martha Argerich (piano)", "Wiener Philharmoniker", "AC/DC (guest band)"]
    flac["LYRICIST"] = ["Friedrich Schiller"]
    flac["ARRANGER"] = ["Franz Liszt; Ferruccio Busoni"]
    flac["WORK"] = ["Symphonies", "Symphony No. 5"]
    flac["MUSICBRAINZ_WORKID"] = [PARENT_WORK_MBID, WORK_MBID]
    flac["MOVEMENTNAME"] = ["Andante con moto"]
    flac["MOVEMENT"] = ["2"]
    flac["MOVEMENTTOTAL"] = ["4"]
    flac["IS_CLASSICAL"] = ["Yes"]
    flac.save()

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.composers == ("Ludwig van Beethoven", "Franz Liszt")
    assert _tags.composer_sort_names == ("Beethoven, Ludwig van", "Liszt, Franz")
    assert _tags.musicbrainz_composerids == (COMPOSER_MBID,)
    assert _tags.conductors == ("Carlos Kleiber",)
    assert _tags.performers == (
        ("Martha Argerich", "piano"),
        ("Wiener Philharmoniker", None),
        ("AC/DC", "guest band"),
    )
    assert _tags.lyricists == ("Friedrich Schiller",)
    assert _tags.arrangers == ("Franz Liszt", "Ferruccio Busoni")
    assert _tags.works == ("Symphonies", "Symphony No. 5")
    assert _tags.work == "Symphony No. 5"
    assert _tags.musicbrainz_workids == (PARENT_WORK_MBID, WORK_MBID)
    assert _tags.musicbrainz_workid == WORK_MBID
    assert (_tags.movement_name, _tags.movement_number, _tags.movement_total) == (
        "Andante con moto",
        2,
        4,
    )
    assert _tags.is_classical


def test_classical_tags_are_read_from_a_wavpack_file(tmp_path: pathlib.Path) -> None:
    """
    APEv2 uses Picard's mixed case keys and separates values with a null byte.

    Uses parse_tags_mutagen directly since the minimal WavPack fixture
    does not contain valid audio data for ffprobe to parse.
    """
    dest = tmp_path / "classical.wv"
    shutil.copy(FILE_WV, dest)
    ape = APEv2(str(dest))  # type: ignore[no-untyped-call]
    ape["Composer"] = ["Ludwig van Beethoven", "Franz Liszt"]
    ape["Composersort"] = "Beethoven, Ludwig van"
    ape["Musicbrainz_Composerid"] = COMPOSER_MBID
    ape["Conductor"] = "Carlos Kleiber"
    ape["Performer"] = ["Martha Argerich (piano)", "Wiener Philharmoniker"]
    ape["Lyricist"] = "Friedrich Schiller"
    ape["Arranger"] = "Franz Liszt"
    ape["Work"] = ["Symphonies", "Symphony No. 5"]
    ape["Musicbrainz_Workid"] = [PARENT_WORK_MBID, WORK_MBID]
    ape["MOVEMENTNAME"] = "Scherzo"
    ape["MOVEMENT"] = "3"
    ape["MOVEMENTTOTAL"] = "4"
    ape["Is_Classical"] = "true"
    ape.save(str(dest))

    _tags = _tags_with(parse_tags_mutagen(str(dest)))

    assert _tags.composers == ("Ludwig van Beethoven", "Franz Liszt")
    assert _tags.composer_sort_names == ("Beethoven, Ludwig van",)
    assert _tags.musicbrainz_composerids == (COMPOSER_MBID,)
    assert _tags.conductors == ("Carlos Kleiber",)
    assert _tags.performers == (("Martha Argerich", "piano"), ("Wiener Philharmoniker", None))
    assert _tags.lyricists == ("Friedrich Schiller",)
    assert _tags.arrangers == ("Franz Liszt",)
    assert _tags.work == "Symphony No. 5"
    assert _tags.musicbrainz_workids == (PARENT_WORK_MBID, WORK_MBID)
    assert (_tags.movement_name, _tags.movement_number, _tags.movement_total) == ("Scherzo", 3, 4)
    assert _tags.is_classical


async def test_classical_tags_are_read_from_an_m4a_file(tmp_path: pathlib.Path) -> None:
    """MP4 has atoms for the composer, work and movement and freeform tags for the rest."""
    dest = tmp_path / "classical.m4a"
    shutil.copy(FILE_M4A, dest)
    mp4 = MP4(str(dest))  # type: ignore[no-untyped-call]
    mp4["\xa9wrt"] = ["Ludwig van Beethoven", "Franz Liszt"]
    mp4["soco"] = ["Beethoven, Ludwig van", "Liszt, Franz"]
    mp4["----:com.apple.iTunes:MusicBrainz Composer Id"] = [
        MP4FreeForm(COMPOSER_MBID.encode())  # type: ignore[no-untyped-call]
    ]
    mp4["----:com.apple.iTunes:CONDUCTOR"] = [MP4FreeForm(b"Carlos Kleiber")]  # type: ignore[no-untyped-call]
    mp4["----:com.apple.iTunes:LYRICIST"] = [MP4FreeForm(b"Friedrich Schiller")]  # type: ignore[no-untyped-call]
    mp4["\xa9wrk"] = ["Symphony No. 5"]
    mp4["----:com.apple.iTunes:MusicBrainz Work Id"] = [
        MP4FreeForm(PARENT_WORK_MBID.encode()),  # type: ignore[no-untyped-call]
        MP4FreeForm(WORK_MBID.encode()),  # type: ignore[no-untyped-call]
    ]
    mp4["\xa9mvn"] = ["Allegro"]
    mp4["\xa9mvi"] = [4]
    mp4["\xa9mvc"] = [4]
    mp4["----:com.apple.iTunes:IS_CLASSICAL"] = [MP4FreeForm(b"1")]  # type: ignore[no-untyped-call]
    mp4.save()  # type: ignore[no-untyped-call]

    _tags = await tags.async_parse_tags(str(dest))

    assert _tags.composers == ("Ludwig van Beethoven", "Franz Liszt")
    assert _tags.composer_sort_names == ("Beethoven, Ludwig van", "Liszt, Franz")
    assert _tags.musicbrainz_composerids == (COMPOSER_MBID,)
    assert _tags.conductors == ("Carlos Kleiber",)
    assert _tags.lyricists == ("Friedrich Schiller",)
    assert _tags.work == "Symphony No. 5"
    assert _tags.musicbrainz_workids == (PARENT_WORK_MBID, WORK_MBID)
    assert _tags.musicbrainz_workid == WORK_MBID
    assert (_tags.movement_name, _tags.movement_number, _tags.movement_total) == ("Allegro", 4, 4)
    assert _tags.is_classical


async def test_classical_tags_leave_artists_and_narrators_unchanged(tmp_path: pathlib.Path) -> None:
    """The composer stays the audiobook narrator fallback and the artists are untouched."""
    dest = tmp_path / "classical.mp3"
    shutil.copy(FILE_MP3, dest)
    id3 = ID3(str(dest))  # type: ignore[no-untyped-call]
    id3.add(TCOM(encoding=3, text=["Ludwig van Beethoven"]))  # type: ignore[no-untyped-call]
    id3.add(TPE3(encoding=3, text=["Carlos Kleiber"]))  # type: ignore[no-untyped-call]
    id3.add(TMCL(encoding=3, people=[["piano", "Martha Argerich"]]))  # type: ignore[no-untyped-call]
    id3.save(v2_version=4)

    _tags = await tags.async_parse_tags(str(dest))
    _untouched = await tags.async_parse_tags(FILE_MP3)

    assert _tags.artists == _untouched.artists
    assert _tags.narrators == ("Ludwig van Beethoven",)


@pytest.mark.parametrize(
    ("raw_tags", "expected"),
    [
        # the movement tag holds the number, alone or with the total
        ({"movement": "3", "movementname": "IV. Finale"}, (3, None)),
        ({"movement": "2/4"}, (2, 4)),
        ({"movement": "2", "movementtotal": "4"}, (2, 4)),
        # without a number tag, a leading Roman or Arabic numeral gives it away
        ({"movementname": "II. Andante"}, (2, None)),
        ({"movementname": "iv: Finale", "movementtotal": "4"}, (4, 4)),
        ({"movementname": "XIV - Fugue"}, (14, None)),
        ({"movementname": "XX Finale"}, (20, None)),
        ({"movementname": "3. Scherzo"}, (3, None)),
        ({"movementname": "12 Allegro"}, (12, None)),
        # an inferred number beyond the known total is not the movement number
        ({"movementname": "V. Finale", "movementtotal": "4"}, (None, 4)),
        # words that merely start with a numeral letter, and numerals beyond XX, are no number
        ({"movementname": "Vivace"}, (None, None)),
        ({"movementname": "XXI. Finale"}, (None, None)),
        ({"movementname": "Andante"}, (None, None)),
        ({}, (None, None)),
    ],
)
def test_movement_number(raw_tags: dict[str, str], expected: tuple[int | None, int | None]) -> None:
    """The movement number comes from its tag, or from the movement name as a last resort."""
    audio_tags = _tags_with(raw_tags)
    assert (audio_tags.movement_number, audio_tags.movement_total) == expected


@pytest.mark.parametrize(
    ("raw_tags", "expected_name", "expected_number"),
    [
        # Classical Extras writes the movement name where Picard keeps its number
        ({"movement": "II. Andante"}, "II. Andante", 2),
        # Roon names the movement in PART
        ({"part": "3. Scherzo"}, "3. Scherzo", 3),
        ({"movementname": "Allegro", "part": "Other"}, "Allegro", None),
        # the work in front of a Classical Extras hierarchy separator is left off
        ({"movementname": "Symphony No. 5:: I. Allegro con brio"}, "I. Allegro con brio", 1),
    ],
)
def test_movement_name_fallbacks(
    raw_tags: dict[str, str], expected_name: str, expected_number: int | None
) -> None:
    """The movement name falls back to the Classical Extras and Roon tags."""
    audio_tags = _tags_with(raw_tags)
    assert (audio_tags.movement_name, audio_tags.movement_number) == (
        expected_name,
        expected_number,
    )


@pytest.mark.parametrize(
    ("raw_tags", "expected"),
    [
        ({"work": "Symphony No. 5", "groupheading": "Other"}, ("Symphony No. 5",)),
        # Classical Extras names the work in groupheading or top_work
        ({"groupheading": "Symphony No. 5::", "topwork": "Other"}, ("Symphony No. 5",)),
        ({"topwork": "Der Ring des Nibelungen"}, ("Der Ring des Nibelungen",)),
        ({"work": "Symphony No. 5:: I. Allegro"}, ("Symphony No. 5",)),
        ({"work": "Symphonies; Symphony No. 5"}, ("Symphonies", "Symphony No. 5")),
        ({}, ()),
    ],
)
def test_works(raw_tags: dict[str, str], expected: tuple[str, ...]) -> None:
    """The work falls back to the Classical Extras tags and drops their hierarchy separator."""
    assert _tags_with(raw_tags).works == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("True", True), ("YES", True), ("0", False), ("no", False), ("", False)],
)
def test_is_classical(value: str, expected: bool) -> None:
    """Only a true value marks the file as classical."""
    assert _tags_with({"isclassical": value}).is_classical is expected


def test_is_classical_without_the_tag() -> None:
    """A file without the tag is not marked as classical."""
    assert not _tags_with({}).is_classical


def test_roon_credits_are_read_without_a_performer_tag() -> None:
    """Roon's personnel, soloist and ensemble tags, written as "Name - Role", fill in for a missing performer tag."""
    audio_tags = _tags_with(
        {
            "personnel": "Martha Argerich - Piano; Gidon Kremer (violin); Andreas Spreer",
            "soloist": "Martha Argerich - Piano",
            "ensemble": "Wiener Philharmoniker",
            "section": "Act I",
        }
    )

    assert audio_tags.performers == (
        ("Martha Argerich", "Piano"),
        ("Gidon Kremer", "violin"),
        ("Andreas Spreer", None),
    )
    assert audio_tags.soloists == (("Martha Argerich", "Piano"),)
    assert audio_tags.ensembles == ("Wiener Philharmoniker",)
    assert audio_tags.section == "Act I"


def test_roon_credits_are_ignored_with_a_performer_tag() -> None:
    """Picard's performer tag already credits everyone Roon's tags would."""
    audio_tags = _tags_with(
        {
            "performer": "Gidon Kremer (violin)",
            "personnel": "Martha Argerich - Piano",
            "soloist": "Martha Argerich - Piano",
            "ensemble": "Wiener Philharmoniker",
        }
    )

    assert audio_tags.performers == (("Gidon Kremer", "violin"),)
    assert audio_tags.soloists == ()
    assert audio_tags.ensembles == ()


def test_work_and_composer_mbids() -> None:
    """Invalid identifiers are dropped and ID3v2.3's slash joined identifiers are split."""
    audio_tags = _tags_with(
        {
            "musicbrainzworkid": ["not-an-id", f"{PARENT_WORK_MBID}/{WORK_MBID}"],
            "musicbrainzcomposerid": "not-an-id",
        }
    )

    assert audio_tags.musicbrainz_workids == (PARENT_WORK_MBID, WORK_MBID)
    assert audio_tags.musicbrainz_workid == WORK_MBID
    assert audio_tags.musicbrainz_composerids == ()
