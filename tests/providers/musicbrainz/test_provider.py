"""Tests for the MusicBrainz provider."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import (
    AlbumType,
    ExternalID,
    LinkType,
    MediaType,
    ProviderFeature,
    RecordingSource,
    WorkType,
)
from music_assistant_models.errors import InvalidDataError, RateLimited
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    MediaItemLink,
    ProviderMapping,
    Track,
    UniqueList,
    WorkMovement,
)

from music_assistant.constants import VARIOUS_ARTISTS_MBID
from music_assistant.providers.musicbrainz.api_client import MusicBrainzAPIClient
from music_assistant.providers.musicbrainz.constants import (
    RELEASE_BROWSE_MAX_PAGES,
    SUPPORTED_FEATURES,
)
from music_assistant.providers.musicbrainz.models import (
    MusicBrainzArtist,
    MusicBrainzBarcodeRelease,
    MusicBrainzRecording,
    MusicBrainzRelation,
    MusicBrainzReleaseGroup,
    MusicBrainzWork,
)
from music_assistant.providers.musicbrainz.provider import (
    MusicbrainzProvider,
    _edition_rank,
    _length_matches,
    relation_urls,
)
from tests.common import use_real_create_task

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _provider(
    response: Any, release_group_response: Any = None
) -> tuple[MusicbrainzProvider, AsyncMock]:
    """
    Return a MusicbrainzProvider whose API client answers with the given response.

    :param response: Answer to every request but the release group lookup.
    :param release_group_response: Answer to the release group lookup.
    """
    with patch.object(MusicbrainzProvider, "__init__", lambda *_a, **_kw: None):
        provider = MusicbrainzProvider.__new__(MusicbrainzProvider)
    provider._logged_hints = set()

    async def _answer(endpoint: str, **_kwargs: Any) -> Any:
        return release_group_response if endpoint == "release-group" else response

    get_data = AsyncMock(side_effect=_answer)
    api_client = MagicMock()
    api_client.get_data = get_data
    provider._api_client = api_client
    return provider, get_data


def _recordings(*first_release_dates: str | None) -> dict[str, Any]:
    """Return an isrc lookup or search response with a recording per given release date."""
    return {
        "isrc": "GBAYE8600477",
        "recordings": [
            {"id": f"stub-{i}", "title": "stub"}
            if date is None
            else {"id": f"stub-{i}", "title": "stub", "first-release-date": date}
            for i, date in enumerate(first_release_dates)
        ],
    }


# ---------------------------------------------------------------------------
# get_release_year_by_isrc
# ---------------------------------------------------------------------------


async def test_release_year_parses_all_date_precisions() -> None:
    """Accept a year, year-month or full date as the first release date."""
    for release_date in ("1986", "1986-06", "1986-06-23"):
        provider, _ = _provider(_recordings(release_date))
        assert await provider.get_release_year_by_isrc("GBAYE8600477") == 1986


async def test_release_year_searches_recordings_by_the_normalized_isrc() -> None:
    """Search recordings by ISRC in a single request, which carries the first release date."""
    provider, get_data = _provider(_recordings("1986"))

    await provider.get_release_year_by_isrc("GB-AYE-86-00477")

    get_data.assert_awaited_once_with("recording", query="isrc:GBAYE8600477", limit="100")


async def test_release_year_is_none_for_a_truncated_search() -> None:
    """Refuse to date a song when the search does not return every recording."""
    response = {**_recordings("1986"), "count": 2}
    provider, _ = _provider(response)
    assert await provider.get_release_year_by_isrc("GBAYE8600477") is None


async def test_release_year_returns_earliest_of_multiple_recordings() -> None:
    """Date the song by the oldest recording the ISRC covers."""
    provider, _ = _provider(_recordings("2009-05-01", "1986-06", "1994"))
    assert await provider.get_release_year_by_isrc("GBAYE8600477") == 1986


async def test_release_year_is_none_without_a_usable_date() -> None:
    """Return None when MusicBrainz has no parseable first release date."""
    for response in (
        None,
        {"isrc": "GBAYE8600477"},
        {"isrc": "GBAYE8600477", "recordings": []},
        _recordings(None),
        _recordings("????-06"),
    ):
        provider, _ = _provider(response)
        assert await provider.get_release_year_by_isrc("GBAYE8600477") is None


async def test_release_year_rejects_a_malformed_isrc() -> None:
    """Never put an ISRC that cannot be part of a URL path in the request."""
    provider, get_data = _provider(_recordings("1986"))

    assert await provider.get_release_year_by_isrc("../artist/1") is None
    get_data.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_recordings_by_isrc
# ---------------------------------------------------------------------------


_YELLOW_SUBMARINE = {
    "isrc": "GBAYE0601498",
    "recordings": [
        {
            "id": "b2181aae-5cba-496c-bb0c-b4cc0109ebf8",
            "title": "Yellow Submarine",
            "length": 160000,
            "first-release-date": "1966-08-05",
            "disambiguation": "original stereo studio mix",
            "video": False,
        }
    ],
}


async def test_recordings_by_isrc_parses_a_realistic_payload() -> None:
    """Parse id, title and first-release-date from a real isrc lookup response."""
    provider, _ = _provider(_YELLOW_SUBMARINE)

    recordings = await provider.get_recordings_by_isrc("GBAYE0601498")

    assert len(recordings) == 1
    recording = recordings[0]
    assert recording.id == "b2181aae-5cba-496c-bb0c-b4cc0109ebf8"
    assert recording.title == "Yellow Submarine"
    assert recording.first_release_date == "1966-08-05"


async def test_recordings_by_isrc_returns_all_recordings() -> None:
    """Return every recording an ISRC covers, not just the first."""
    provider, _ = _provider(_recordings("2009-05-01", "1986-06", "1994"))

    recordings = await provider.get_recordings_by_isrc("GBAYE8600477")

    assert len(recordings) == 3
    assert [r.first_release_date for r in recordings] == ["2009-05-01", "1986-06", "1994"]


async def test_recordings_by_isrc_skips_a_malformed_entry() -> None:
    """Skip a recording missing a required field while keeping its valid siblings."""
    response = {
        "isrc": "GBAYE8600477",
        "recordings": [
            {"title": "no id here"},
            {"id": "good-1", "title": "stub", "first-release-date": "1986"},
        ],
    }
    provider, _ = _provider(response)

    recordings = await provider.get_recordings_by_isrc("GBAYE8600477")

    assert len(recordings) == 1
    assert recordings[0].id == "good-1"


async def test_recordings_by_isrc_is_empty_without_usable_data() -> None:
    """Return an empty list for every shape of "MusicBrainz has nothing" response."""
    for response in (
        None,
        {"isrc": "GBAYE8600477"},
        {"isrc": "GBAYE8600477", "recordings": []},
    ):
        provider, _ = _provider(response)
        assert await provider.get_recordings_by_isrc("GBAYE8600477") == []


async def test_recordings_by_isrc_rejects_a_malformed_isrc() -> None:
    """Never put an ISRC that cannot be part of a URL path in the request."""
    provider, get_data = _provider(_YELLOW_SUBMARINE)

    assert await provider.get_recordings_by_isrc("../artist/1") == []
    get_data.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_release_year_by_track_name
# ---------------------------------------------------------------------------


def _credit(name: str, artist_id: str) -> dict[str, Any]:
    """Return one artist credit of a release."""
    return {"name": name, "artist": {"id": artist_id, "name": name, "sort-name": name}}


def _release(
    date: str,
    *,
    title: str = "A Night at the Opera",
    primary_type: str = "Album",
    secondary_types: list[str] | None = None,
    status: str = "Official",
    credit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one release of a searched recording."""
    release: dict[str, Any] = {
        "id": f"release-{date}",
        "title": title,
        "date": date,
        "status": status,
        "release-group": {
            "id": f"rg-{title}-{primary_type}",
            "title": title,
            "primary-type": primary_type,
        },
    }
    if secondary_types:
        release["release-group"]["secondary-types"] = secondary_types
    if credit is not None:
        release["artist-credit"] = [credit]
    return release


def _search_result(*recordings: dict[str, Any]) -> dict[str, Any]:
    """Return a recording search response holding the given recordings."""
    return {"count": len(recordings), "recordings": list(recordings)}


def _recording(
    *releases: dict[str, Any],
    title: str = "Bohemian Rhapsody",
    artist: str = "Queen",
    artist_id: str = "artist-1",
    first_release: str | None = None,
) -> dict[str, Any]:
    """Return one searched recording credited to the given artist."""
    recording: dict[str, Any] = {
        "id": f"recording-{title}-{releases[0]['date'] if releases else 'none'}",
        "title": title,
        "artist-credit": [{"artist": {"id": artist_id, "name": artist, "sort-name": artist}}],
        "releases": list(releases),
    }
    if first_release is not None:
        recording["first-release-date"] = first_release
    return recording


async def test_release_year_by_track_name_returns_the_earliest_studio_release() -> None:
    """Date a song by the oldest studio album any matching recording appeared on."""
    provider, get_data = _provider(
        _search_result(
            _recording(_release("2011-05-16", title="The Platinum Collection")),
            _recording(_release("1992-08-25", title="Classic Queen")),
            _recording(_release("1975-11-21")),
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975
    assert get_data.await_args_list[0].args == ("recording",)
    assert get_data.await_args_list[0].kwargs == {
        "query": '"Bohemian Rhapsody" AND artist:"Queen"',
        "limit": "100",
    }


async def test_release_year_by_track_name_ignores_untrustworthy_releases() -> None:
    """Never date a song by a compilation, a live album, a bootleg or an unrelated single."""
    provider, _ = _provider(
        _search_result(
            _recording(
                # every untrusted release predates the studio album, so each filter has to
                # hold on its own for the studio year to win
                _release("1968-10-26", title="Greatest Hits", secondary_types=["Compilation"]),
                _release("1969-06-22", title="Live Killers", secondary_types=["Live"]),
                _release("1970-01-01", title="Some Other Song", primary_type="Single"),
                _release("1971-03-01", title="Bootleg Tape", status="Bootleg"),
                _release("1972-05-05", title="A Tribute", primary_type="Other"),
                _release("1975-11-21"),
            )
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_dates_a_song_by_its_soundtrack() -> None:
    """Date a song written for a film by that film's soundtrack."""
    provider, _ = _provider(
        _search_result(
            _recording(
                _release(
                    "1994-05-31",
                    title="The Lion King: Original Motion Picture Soundtrack",
                    secondary_types=["Soundtrack"],
                ),
                _release("2013-09-13", title="The Diving Board"),
                title="Circle of Life",
                artist="Elton John",
            )
        )
    )

    assert await provider.get_release_year_by_track_name("Elton John", "Circle of Life") == 1994


async def test_release_year_by_track_name_ignores_a_soundtrack_compilation() -> None:
    """Never date a song by a film compilation of songs released before it."""
    provider, _ = _provider(
        _search_result(
            _recording(
                # the compilation predates the studio album, so the secondary type filter
                # has to hold on its own for the studio year to win
                _release(
                    "1968-10-26",
                    title="Music From the Motion Picture",
                    secondary_types=["Compilation", "Soundtrack"],
                ),
                _release("1975-11-21"),
            )
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_ignores_a_various_artists_soundtrack() -> None:
    """Never date a song by a film compilation credited to Various Artists."""
    provider, _ = _provider(
        _search_result(
            _recording(
                # most film soundtracks are compilations of several artists, and the credit
                # filter is what keeps them out now that soundtracks are allowed through
                _release(
                    "1968-10-26",
                    title="Music From the Motion Picture",
                    secondary_types=["Soundtrack"],
                    credit=_credit("Various Artists", VARIOUS_ARTISTS_MBID),
                ),
                _release("1975-11-21", credit=_credit("Queen", "artist-1")),
            )
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_ignores_a_various_artists_release() -> None:
    """Never date a song by a hits compilation that carries no Compilation type."""
    provider, _ = _provider(
        _search_result(
            _recording(
                # the compilation predates the studio album, so the credit filter has to
                # hold on its own for the studio year to win
                _release(
                    "1968-10-26",
                    title="Hits of the 60s",
                    credit=_credit("Various Artists", VARIOUS_ARTISTS_MBID),
                ),
                _release("1975-11-21", credit=_credit("Queen", "artist-1")),
            )
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_identifies_various_artists_by_id() -> None:
    """
    Recognise the Various Artists entity by its id rather than by its name.

    MusicBrainz localizes the name it credits that entity under, and unrelated artists
    are named after it.
    """
    provider, _ = _provider(
        _search_result(
            _recording(
                _release(
                    "1968-10-26",
                    title="Artisti Vari Compilation",
                    credit=_credit("Artisti Vari", VARIOUS_ARTISTS_MBID),
                ),
                _release(
                    "1975-11-21",
                    credit=_credit("Various Artist", "artist-named-like-various"),
                ),
            )
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_group_by_track_name_drops_a_various_artists_release() -> None:
    """Offer no artwork candidate when a song is only listed on a hits compilation."""
    provider, _ = _provider(
        _search_result(
            _recording(
                _release(
                    "1981-10-26",
                    title="Hits of the 80s",
                    credit=_credit("Various Artists", VARIOUS_ARTISTS_MBID),
                )
            )
        )
    )

    result = await provider.get_release_group_by_track_name("Queen", "Bohemian Rhapsody")

    assert result is not None
    artist, release_groups = result
    assert artist.name == "Queen"
    assert release_groups == []


async def test_release_group_by_track_name_offers_a_soundtrack_as_artwork() -> None:
    """Offer the soundtrack of a song written for a film as an artwork candidate."""
    provider, _ = _provider(
        _search_result(
            _recording(
                _release(
                    "1994-05-31",
                    title="The Lion King: Original Motion Picture Soundtrack",
                    secondary_types=["Soundtrack"],
                ),
                title="Circle of Life",
                artist="Elton John",
            )
        )
    )

    result = await provider.get_release_group_by_track_name("Elton John", "Circle of Life")

    assert result is not None
    _, release_groups = result
    assert [rg.title for rg in release_groups] == [
        "The Lion King: Original Motion Picture Soundtrack"
    ]


async def test_release_year_by_track_name_ignores_an_undated_release_group() -> None:
    """Date a song by the oldest release group that has a date, not by an undated one."""
    provider, _ = _provider(
        _search_result(
            _recording(_release("", title="Unknown Pressing")),
            _recording(_release("1975-11-21")),
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_accepts_a_single_named_after_the_song() -> None:
    """Date a song by its own single when no studio album carries it."""
    provider, _ = _provider(
        _search_result(
            _recording(_release("1975-10-31", title="Bohemian Rhapsody", primary_type="Single"))
        )
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_is_none_without_a_confident_match() -> None:
    """Return no year at all rather than guessing from a name that does not match."""
    for response in (
        None,
        {"count": 0, "recordings": []},
        _search_result(_recording(_release("1975-11-21"), artist="Not Queen")),
        _search_result(_recording(_release("1975-11-21"), title="Another Song")),
        _search_result(_recording()),
        _search_result(_recording(_release(""))),
    ):
        provider, _ = _provider(response)
        assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") is None


def _release_groups(*groups: tuple[str, str]) -> dict[str, Any]:
    """
    Return a release group search response holding the given release groups.

    :param groups: Release groups as (id, first release date) pairs.
    """
    return {
        "count": len(groups),
        "release-groups": [
            {"id": group_id, "title": group_id, "first-release-date": date}
            for group_id, date in groups
        ],
    }


async def test_release_year_by_track_name_prefers_the_release_group_first_release() -> None:
    """Date a much reissued song by its album's first release, not by the reissue found."""
    provider, _ = _provider(
        _search_result(_recording(_release("2021-11-12"))),
        _release_groups(("rg-A Night at the Opera-Album", "1975-11-21")),
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_keeps_a_close_release_date() -> None:
    """Keep the found release when the album barely predates it, as a single ahead of it would."""
    provider, _ = _provider(
        _search_result(_recording(_release("1975-11-21"))),
        _release_groups(("rg-A Night at the Opera-Album", "1974-10-31")),
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_corrects_only_beyond_the_threshold() -> None:
    """Correct the year only once the album predates the found release by enough years."""
    for first_release_year, expected in ((1970, 1975), (1969, 1969)):
        provider, _ = _provider(
            _search_result(_recording(_release("1975-11-21"))),
            _release_groups(("rg-A Night at the Opera-Album", f"{first_release_year}-10-31")),
        )

        assert (
            await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == expected
        )


async def test_release_year_by_track_name_keeps_the_found_release_when_the_lookup_fails() -> None:
    """Never lose the year the search already supplied when the release group lookup fails."""
    search_result = _search_result(_recording(_release("1975-11-21")))
    provider, get_data = _provider(search_result)

    async def _answer(endpoint: str, **_kwargs: Any) -> Any:
        if endpoint == "release-group":
            raise RateLimited("rate limited")
        return search_result

    get_data.side_effect = _answer

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_looks_up_every_release_group_at_once() -> None:
    """Resolve all release groups of a song with a single, escaped, request."""
    provider, get_data = _provider(
        _search_result(
            _recording(_release("2011-05-16", title="The Platinum Collection")),
            _recording(_release("1975-11-21")),
        ),
        _release_groups(("rg-A Night at the Opera-Album", "1975-11-21")),
    )

    await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody")

    # one release group lookup, however many groups the search turned up
    assert [call.args for call in get_data.await_args_list] == [("recording",), ("release-group",)]
    assert get_data.await_args_list[1].kwargs["query"] == (
        r"rgid:(rg\-A Night at the Opera\-Album OR rg\-The Platinum Collection\-Album)"
    )


async def test_release_year_by_track_name_falls_back_to_the_found_release() -> None:
    """Keep the found release when MusicBrainz does not know when the album first came out."""
    for release_group_response in (None, _release_groups(), {"release-groups": [{"id": "rg-x"}]}):
        provider, _ = _provider(
            _search_result(_recording(_release("1975-11-21"))), release_group_response
        )

        assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_year_by_track_name_dates_an_undated_release_group() -> None:
    """Date a song whose found releases carry no date at all by its album's first release."""
    provider, _ = _provider(
        _search_result(_recording(_release(""))),
        _release_groups(("rg-A Night at the Opera-Album", "1975-11-21")),
    )

    assert await provider.get_release_year_by_track_name("Queen", "Bohemian Rhapsody") == 1975


async def test_release_group_by_track_name_costs_a_single_request() -> None:
    """Never spend the release group lookup on callers that only want the release groups."""
    provider, get_data = _provider(
        _search_result(_recording(_release("1975-11-21"))),
        _release_groups(("rg-A Night at the Opera-Album", "1975-11-21")),
    )

    await provider.get_release_group_by_track_name("Queen", "Bohemian Rhapsody")

    assert [call.args for call in get_data.await_args_list] == [("recording",)]


async def test_release_group_by_track_name_returns_the_artist_and_oldest_groups_first() -> None:
    """Hand out the artist of the oldest recording, and their release groups oldest first."""
    provider, _ = _provider(
        _search_result(
            _recording(
                _release("2011-05-16", title="The Platinum Collection"),
                first_release="2011-05-16",
                artist_id="artist-reissue",
            ),
            _recording(
                _release("1975-11-21"),
                first_release="1975-11-21",
                artist_id="artist-original",
            ),
        )
    )

    result = await provider.get_release_group_by_track_name("Queen", "Bohemian Rhapsody")

    assert result is not None
    artist, release_groups = result
    # MusicBrainz can hold several artist entries under one name, and the oldest recording
    # is the one that identifies the original
    assert artist.id == "artist-original"
    assert [group.title for group in release_groups] == [
        "A Night at the Opera",
        "The Platinum Collection",
    ]


async def test_release_group_by_track_name_returns_the_artist_without_release_groups() -> None:
    """Still identify the artist when no matched recording carries a usable release group."""
    provider, _ = _provider(
        _search_result(_recording(_release("1981-10-26", secondary_types=["Compilation"])))
    )

    result = await provider.get_release_group_by_track_name("Queen", "Bohemian Rhapsody")

    assert result is not None
    artist, release_groups = result
    assert artist.name == "Queen"
    assert release_groups == []


async def test_release_year_by_track_name_escapes_lucene_specials() -> None:
    """Escape characters that would otherwise change the meaning of the search query."""
    provider, get_data = _provider(_search_result())

    await provider.get_release_year_by_track_name("AC/DC", "T.N.T. (live!)")

    get_data.assert_awaited_once_with(
        "recording",
        query='"T.N.T. \\(live\\!\\)" AND artist:"AC\\/DC"',
        limit="100",
    )


# ---------------------------------------------------------------------------
# get_artist_by_track_titles
# ---------------------------------------------------------------------------


def _title_search_provider(*responses: Any) -> tuple[MusicbrainzProvider, AsyncMock]:
    """Return a MusicbrainzProvider answering each successive request with the next response."""
    provider, get_data = _provider(None)
    get_data.side_effect = list(responses)
    return provider, get_data


def _credited_recording(title: str, artist: str, artist_id: str = "artist-1") -> dict[str, Any]:
    """Return one searched recording of the given title, credited as MusicBrainz credits it."""
    return {
        "id": f"recording-{title}",
        "title": title,
        "artist-credit": [_credit(artist, artist_id)],
    }


async def test_artist_by_track_titles_returns_the_first_credited_recording() -> None:
    """Search title by title until a recording of that title credits the artist."""
    provider, get_data = _title_search_provider(
        _search_result(_credited_recording("Under Pressure", "David Bowie")),
        _search_result(_credited_recording("Bohemian Rhapsody", "Queen", "artist-queen")),
    )

    artist = await provider.get_artist_by_track_titles(
        "Queen", ["Under Pressure", "Bohemian Rhapsody"]
    )

    assert artist is not None
    assert artist.id == "artist-queen"
    assert [call.kwargs for call in get_data.await_args_list] == [
        {"query": '"Under Pressure" AND artist:"Queen"', "limit": "100"},
        {"query": '"Bohemian Rhapsody" AND artist:"Queen"', "limit": "100"},
    ]


async def test_artist_by_track_titles_requires_the_searched_title() -> None:
    """A recording of another title does not identify the artist."""
    provider, _ = _title_search_provider(
        _search_result(_credited_recording("Bohemian Rhapsody (Live Aid)", "Queen"))
    )

    assert await provider.get_artist_by_track_titles("Queen", ["Radio Ga Ga"]) is None


async def test_artist_by_track_titles_prefers_an_exact_title() -> None:
    """A recording of exactly the searched title wins over a near one listed before it."""
    provider, _ = _title_search_provider(
        _search_result(
            _credited_recording("Creeps", "Radiohead", "artist-near"),
            _credited_recording("Creep", "Radiohead", "artist-exact"),
        )
    )

    artist = await provider.get_artist_by_track_titles("Radiohead", ["Creep"])

    assert artist is not None
    assert artist.id == "artist-exact"


async def test_artist_by_track_titles_skips_a_title_namesakes_share() -> None:
    """A title two artists of the name are credited with is passed over for the next one."""
    provider, get_data = _title_search_provider(
        _search_result(
            _credited_recording("All My Loving", "Bella", "artist-trio"),
            _credited_recording("All My Loving", "Bella", "artist-band"),
        ),
        _search_result(_credited_recording("Dance With Me", "Bella", "artist-trio")),
    )

    artist = await provider.get_artist_by_track_titles("Bella", ["All My Loving", "Dance With Me"])

    assert artist is not None
    assert artist.id == "artist-trio"
    assert get_data.await_count == 2


async def test_artist_by_track_titles_leaves_namesakes_unidentified() -> None:
    """When every title is shared by namesakes, no artist is picked."""
    provider, _ = _title_search_provider(
        _search_result(
            _credited_recording("Little Lies", "Aerosol", "artist-a"),
            _credited_recording("Little Lies", "Aerosol", "artist-b"),
        )
    )

    assert await provider.get_artist_by_track_titles("Aerosol", ["Little Lies"]) is None


async def test_artist_by_track_titles_matches_an_alias() -> None:
    """The artist is also recognised by one of its aliases."""
    recording = {"id": "recording-creep", "title": "Creep", "artist-credit": [_radiohead_credit()]}
    provider, _ = _title_search_provider(_search_result(recording))

    artist = await provider.get_artist_by_track_titles("レディオヘッド", ["Creep"])

    assert artist is not None
    assert artist.id == RADIOHEAD_MBID


async def test_artist_by_track_titles_stops_at_the_first_match() -> None:
    """No further title is searched once one identifies the artist."""
    provider, get_data = _title_search_provider(
        _search_result(
            _credited_recording("Bohemian Rhapsody", "Queen", "artist-first"),
            {**_credited_recording("Bohemian Rhapsody", "Queen", "artist-first"), "id": "live"},
        ),
        _search_result(_credited_recording("Radio Ga Ga", "Queen")),
    )

    artist = await provider.get_artist_by_track_titles(
        "Queen", ["Bohemian Rhapsody", "Radio Ga Ga"]
    )

    assert artist is not None
    assert artist.id == "artist-first"
    get_data.assert_awaited_once()


async def test_artist_by_track_titles_escapes_lucene_specials() -> None:
    """Escape characters that would otherwise change the meaning of the search query."""
    provider, get_data = _title_search_provider(_search_result())

    assert await provider.get_artist_by_track_titles("AC/DC", ["T.N.T. (live!)"]) is None

    get_data.assert_awaited_once_with(
        "recording", query='"T.N.T. \\(live\\!\\)" AND artist:"AC\\/DC"', limit="100"
    )


# ---------------------------------------------------------------------------
# get_releases_by_barcode
# ---------------------------------------------------------------------------


def _barcode_release(release_id: str, release_group_id: str, barcode: str) -> dict[str, Any]:
    """
    Return one release stub as a barcode search actually returns it.

    Includes the summary ``media`` object (format and track count, no tracklist) a listing
    carries, so the listing model is exercised realistically.
    """
    return {
        "id": release_id,
        "status-id": "status-id",
        "count": 1,
        "title": "( )",
        "status": "Official",
        "barcode": barcode,
        "artist-credit": [_credit("Sigur Rós", "artist-1")],
        "release-group": {"id": release_group_id, "title": "( )", "primary-type": "Album"},
        "media": [{"format": "CD", "disc-count": 1, "track-count": 14}],
        "track-count": 14,
    }


async def test_releases_by_barcode_parses_releases() -> None:
    """Return every release MusicBrainz has on file for a barcode (summary media and all)."""
    response = {
        "count": 2,
        "releases": [
            _barcode_release("rel-1", "rg-1", "0888072439412"),
            _barcode_release("rel-2", "rg-1", "0888072439412"),
        ],
    }
    provider, _ = _provider(response)

    releases = await provider.get_releases_by_barcode("888072439412")

    assert [release.id for release in releases] == ["rel-1", "rel-2"]
    assert {release.release_group.id for release in releases} == {"rg-1"}


async def test_releases_by_barcode_queries_every_compatible_form() -> None:
    """Query the UPC-12 and its zero-padded EAN-13/GTIN forms in a single request."""
    provider, get_data = _provider({"releases": []})

    await provider.get_releases_by_barcode("888072439412")

    get_data.assert_awaited_once()
    call = get_data.await_args
    assert call is not None
    assert call.args == ("release",)
    assert call.kwargs["limit"] == "100"
    query = call.kwargs["query"]
    assert "barcode:888072439412" in query
    assert "barcode:0888072439412" in query


async def test_releases_by_barcode_skips_an_invalid_barcode() -> None:
    """A structurally invalid barcode is treated as absent, without any request."""
    provider, get_data = _provider({"releases": []})

    assert await provider.get_releases_by_barcode("not-a-barcode") == []
    get_data.assert_not_awaited()


async def test_releases_by_barcode_is_empty_when_not_found() -> None:
    """An unknown barcode yields an empty list rather than an error."""
    provider, _ = _provider(None)

    assert await provider.get_releases_by_barcode("888072439412") == []


async def test_releases_by_barcode_abstains_on_malformed_entry() -> None:
    """One unparsable release makes the whole lookup abstain rather than look complete."""
    response = {
        "releases": [
            {"id": "broken"},
            _barcode_release("rel-2", "rg-2", "0888072439412"),
        ]
    }
    provider, _ = _provider(response)

    with pytest.raises(InvalidDataError):
        await provider.get_releases_by_barcode("888072439412")


async def test_releases_by_barcode_abstains_on_truncated_result() -> None:
    """A truncated page abstains rather than treating a partial set as complete."""
    response = {
        "count": 5,
        "releases": [_barcode_release("rel-1", "rg-1", "0888072439412")],
    }
    provider, _ = _provider(response)

    with pytest.raises(InvalidDataError):
        await provider.get_releases_by_barcode("888072439412")


# ---------------------------------------------------------------------------
# identity lookups: models, reverse URL lookup, browse and resolvers
# ---------------------------------------------------------------------------

RADIOHEAD_MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
SPOTIFY_ARTIST_URL = "https://open.spotify.com/artist/4Z8W4fKeB5YxbusRsdQVPb"
SPOTIFY_ALBUM_URL = "https://open.spotify.com/album/7eyQXxuf2nGj9d2367Gi5f"
SPOTIFY_TRACK_URL = "https://open.spotify.com/track/2Ex8hBvUhZjXjJpZjJZ0aA"
QOBUZ_ALBUM_URL = "https://www.qobuz.com/us-en/album/in-rainbows-radiohead/0634904032432"
BARCODE = "634904032463"
ISRC = "GBSTK0700001"


def _release_browse_route(offset: int = 0) -> str:
    """Return the routing key of a release group browse, of a later page given its offset."""
    return f"release?release-group&offset={offset}" if offset else "release?release-group"


def _routed_provider(routes: dict[str, Any]) -> tuple[MusicbrainzProvider, AsyncMock]:
    """
    Return a MusicbrainzProvider whose API client answers each request from a routing table.

    A lookup is keyed by its endpoint path (without inc parameters), a url lookup by the
    looked-up resource, a search by ``<endpoint>?query`` and a release group browse by
    ``release?release-group``, a later page of it by ``release?release-group&offset=<N>``.
    Anything not routed is answered like a 404.
    """
    with patch.object(MusicbrainzProvider, "__init__", lambda *_a, **_kw: None):
        provider = MusicbrainzProvider.__new__(MusicbrainzProvider)
    provider.logger = MagicMock()
    provider._logged_hints = set()

    async def _answer(endpoint: str, **kwargs: Any) -> Any:
        if endpoint == "url":
            return routes.get(kwargs["resource"])
        if "query" in kwargs:
            return routes.get(f"{endpoint}?query")
        if "release-group" in kwargs:
            return routes.get(_release_browse_route(int(kwargs.get("offset", 0))))
        return routes.get(endpoint.split("?", maxsplit=1)[0])

    get_data = AsyncMock(side_effect=_answer)
    api_client = MagicMock()
    api_client.get_data = get_data
    provider._api_client = api_client
    return provider, get_data


def _requested(get_data: AsyncMock) -> list[str]:
    """Return the endpoint paths (or url resources) requested, in order."""
    requested: list[str] = []
    for call in get_data.await_args_list:
        endpoint = call.args[0]
        if endpoint == "url":
            requested.append(call.kwargs["resource"])
        elif "query" in call.kwargs:
            requested.append(f"{endpoint}?query")
        elif "release-group" in call.kwargs:
            requested.append("release?release-group")
        else:
            requested.append(endpoint.split("?")[0])
    return requested


def _url_relation(
    resource: str, ended: bool = False, type_: str = "free streaming"
) -> dict[str, Any]:
    """Return one URL relation as MusicBrainz lists it."""
    return {"type": type_, "ended": ended, "url": {"id": f"url-{resource}", "resource": resource}}


def _radiohead_credit() -> dict[str, Any]:
    """Return the Radiohead artist credit, with an alias."""
    return {
        "name": "Radiohead",
        "artist": {
            "id": RADIOHEAD_MBID,
            "name": "Radiohead",
            "sort-name": "Radiohead",
            "aliases": [{"name": "レディオヘッド", "sort-name": "Radiohead"}],
        },
    }


def _edition(
    release_id: str,
    *,
    title: str = "In Rainbows",
    status: str | None = "Official",
    media_format: str | None = "Digital Media",
    track_counts: tuple[int, ...] = (10,),
    country: str | None = "XW",
    date: str | None = "2016-05-06",
    spotify_id: str | None = None,
    credit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one release as a barcode search or release group browse lists it."""
    edition: dict[str, Any] = {
        "id": release_id,
        "title": title,
        "status": status,
        "date": date,
        "country": country,
        "barcode": BARCODE,
        "release-group": {"id": "rg-in-rainbows", "title": "In Rainbows", "primary-type": "Album"},
        "artist-credit": [credit or _radiohead_credit()],
        "media": [
            {"format": media_format, "track-count": count, "position": position}
            for position, count in enumerate(track_counts, start=1)
        ],
        "relations": [],
    }
    if spotify_id:
        edition["relations"].append(_url_relation(f"https://open.spotify.com/album/{spotify_id}"))
    return edition


def _release_lookup(release_id: str, **overrides: Any) -> dict[str, Any]:
    """Return a full release lookup response, as the mirror answers it."""
    release: dict[str, Any] = {
        "id": release_id,
        "title": "In Rainbows",
        "status": "Official",
        "status-id": "4e304316-386d-3409-af2e-78857eec5cfe",
        "date": "2016-05-06",
        "country": "XW",
        "barcode": BARCODE,
        "asin": None,
        "disambiguation": "",
        "release-group": {
            "id": "rg-in-rainbows",
            "title": "In Rainbows",
            "primary-type": "Album",
            "secondary-types": [],
            "first-release-date": "2007-10-10",
            "genres": [{"id": "g1", "name": "alternative rock", "count": 5, "disambiguation": ""}],
        },
        "artist-credit": [_radiohead_credit()],
        "label-info": [
            {"catalog-number": "XLDA324", "label": {"id": "label-xl", "name": "XL Recordings"}}
        ],
        "relations": [
            _url_relation(SPOTIFY_ALBUM_URL),
            _url_relation(QOBUZ_ALBUM_URL, ended=True, type_="purchase for download"),
        ],
        "media": [
            {
                "position": 1,
                "format": None,
                "track-count": 10,
                "track-offset": 0,
                "title": "",
                "tracks": [
                    {
                        "id": "track-15-step",
                        "number": "1",
                        "position": 1,
                        "length": 237000,
                        "title": "15 Step",
                        "recording": {
                            "id": "rec-15-step",
                            "title": "15 Step",
                            "length": 238000,
                            "isrcs": [ISRC],
                            "disambiguation": "",
                            "video": False,
                        },
                    }
                ],
            }
        ],
        "genres": [],
    }
    release.update(overrides)
    return release


def _recording_lookup(
    recording_id: str,
    *,
    title: str = "15 Step",
    length: int | None = 238000,
    credit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one recording as an isrc lookup or recording lookup lists it."""
    return {
        "id": recording_id,
        "title": title,
        "length": length,
        "disambiguation": "",
        "video": False,
        "isrcs": [ISRC],
        "artist-credit": [credit or _radiohead_credit()],
        "relations": [_url_relation(SPOTIFY_TRACK_URL)],
    }


def _artist_lookup(artist_id: str = RADIOHEAD_MBID, name: str = "Radiohead") -> dict[str, Any]:
    """Return an artist lookup response."""
    return {
        "id": artist_id,
        "name": name,
        "sort-name": name,
        "type": "Group",
        "genres": [{"id": "g1", "name": "alternative rock", "count": 5, "disambiguation": ""}],
        "relations": [_url_relation(SPOTIFY_ARTIST_URL)],
    }


def _url_lookup(entity: str, *ids: str) -> dict[str, Any]:
    """Return a url lookup response linking the url to the given entities."""
    return {
        "id": "url-1",
        "resource": "resource",
        "relations": [
            {"type": "free streaming", "ended": False, entity: {"id": entity_id, "name": "x"}}
            for entity_id in ids
        ],
    }


def _mapping(domain: str, item_id: str, url: str | None = None) -> ProviderMapping:
    """Return a provider mapping of the given domain."""
    return ProviderMapping(
        item_id=item_id, provider_domain=domain, provider_instance=f"{domain}_1", url=url
    )


def _artist_item(
    name: str = "Radiohead",
    *,
    mappings: set[ProviderMapping] | None = None,
    external_ids: set[tuple[ExternalID, str]] | None = None,
) -> Artist:
    """Return a library artist."""
    return Artist(
        item_id="1",
        provider="library",
        name=name,
        provider_mappings=mappings or set(),
        external_ids=external_ids or set(),
    )


def _album_item(
    name: str = "In Rainbows",
    *,
    artist: str | None = "Radiohead",
    mappings: set[ProviderMapping] | None = None,
    external_ids: set[tuple[ExternalID, str]] | None = None,
) -> Album:
    """Return a library album."""
    artists: UniqueList[Artist | ItemMapping] = UniqueList()
    if artist:
        artists.append(
            ItemMapping(media_type=MediaType.ARTIST, item_id="a", provider="library", name=artist)
        )
    return Album(
        item_id="1",
        provider="library",
        name=name,
        artists=artists,
        provider_mappings=mappings or set(),
        external_ids=external_ids or set(),
    )


def _track_item(
    name: str = "15 Step",
    *,
    artist: str | None = "Radiohead",
    album: str | None = "In Rainbows",
    duration: int = 237,
    mappings: set[ProviderMapping] | None = None,
    external_ids: set[tuple[ExternalID, str]] | None = None,
) -> Track:
    """Return a library track."""
    artists: UniqueList[Artist | ItemMapping] = UniqueList()
    if artist:
        artists.append(
            ItemMapping(media_type=MediaType.ARTIST, item_id="a", provider="library", name=artist)
        )
    return Track(
        item_id="1",
        provider="library",
        name=name,
        duration=duration,
        artists=artists,
        album=ItemMapping(media_type=MediaType.ALBUM, item_id="b", provider="library", name=album)
        if album
        else None,
        provider_mappings=mappings or set(),
        external_ids=external_ids or set(),
    )


async def test_release_details_parses_a_full_release_lookup() -> None:
    """Parse tracklist, recordings, labels, links and identity off a release lookup."""
    provider, get_data = _routed_provider({"release/rel-1": _release_lookup("rel-1")})

    release = await provider.get_release_details("rel-1")

    get_data.assert_awaited_once_with(
        "release/rel-1?inc=artist-credits+aliases+labels+release-groups"
        "+recordings+isrcs+url-rels+genres"
    )
    assert release.count is None
    assert release.barcode == BARCODE
    assert release.asin is None
    assert release.release_group is not None
    assert release.release_group.first_release_date == "2007-10-10"
    assert [genre.name for genre in release.release_group.genres or []] == ["alternative rock"]
    assert release.artist_credit[0].artist.aliases is not None
    assert release.label_info is not None
    assert release.label_info[0].label is not None
    assert (release.label_info[0].label.name, release.label_info[0].catalog_number) == (
        "XL Recordings",
        "XLDA324",
    )
    assert [(r.url.resource if r.url else None, r.ended) for r in release.relations or []] == [
        (SPOTIFY_ALBUM_URL, False),
        (QOBUZ_ALBUM_URL, True),
    ]
    medium = release.media[0]
    assert medium.format is None
    assert medium.track_count == 10
    track = medium.tracks[0]
    assert (track.position, track.length, track.title) == (1, 237000, "15 Step")
    assert track.recording is not None
    assert (track.recording.id, track.recording.isrcs) == ("rec-15-step", [ISRC])


def test_models_parse_genres_and_ignore_their_extra_keys() -> None:
    """MusicBrainz genres carry more keys than a tag; only name and count are kept."""
    genres = [{"id": "g1", "name": "rock", "count": 3, "disambiguation": ""}]
    artist = MusicBrainzArtist.from_raw({**_artist_lookup(), "genres": genres})
    recording = MusicBrainzRecording.from_raw({**_recording_lookup("rec-1"), "genres": genres})
    release_group = MusicBrainzReleaseGroup.from_raw(
        {"id": "rg", "title": "In Rainbows", "genres": genres}
    )

    for item in (artist, recording, release_group):
        assert item.genres is not None
        assert [(genre.name, genre.count) for genre in item.genres] == [("rock", 3)]
    assert recording.relations is not None
    assert recording.relations[0].url is not None
    assert recording.relations[0].url.resource == SPOTIFY_TRACK_URL


def test_barcode_release_parses_the_edition_data_of_a_listing() -> None:
    """A listed release carries the status, country, media and links an edition is told by."""
    release = MusicBrainzBarcodeRelease.from_raw(
        _edition("rel-1", track_counts=(10, 8), spotify_id="7eyQXxuf2nGj9d2367Gi5f")
    )

    assert (release.status, release.country, release.date) == ("Official", "XW", "2016-05-06")
    assert [(medium.format, medium.track_count) for medium in release.media] == [
        ("Digital Media", 10),
        ("Digital Media", 8),
    ]
    assert relation_urls(release.relations) == [SPOTIFY_ALBUM_URL]
    assert release.artist_credit is not None
    assert release.artist_credit[0].artist.id == RADIOHEAD_MBID


async def test_artist_and_recording_details_request_links_and_genres() -> None:
    """Request the aliases, genres and URL relations the identity lookups build on."""
    provider, get_data = _routed_provider(
        {
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
            "recording/rec-1": _recording_lookup("rec-1"),
        }
    )

    artist = await provider.get_artist_details(RADIOHEAD_MBID)
    recording = await provider.get_recording_details("rec-1")

    assert [call.args[0] for call in get_data.await_args_list] == [
        f"artist/{RADIOHEAD_MBID}?inc=aliases+tags+genres+url-rels",
        "recording/rec-1?inc=artists+releases+isrcs+url-rels+genres",
    ]
    assert relation_urls(artist.relations) == [SPOTIFY_ARTIST_URL]
    assert relation_urls(recording.relations) == [SPOTIFY_TRACK_URL]


async def test_mbid_by_url_returns_the_single_linked_entity() -> None:
    """Resolve a streaming service URL to the one entity it is linked to, per entity kind."""
    provider, get_data = _routed_provider(
        {
            SPOTIFY_ARTIST_URL: _url_lookup("artist", RADIOHEAD_MBID),
            SPOTIFY_ALBUM_URL: _url_lookup("release", "rel-1"),
            SPOTIFY_TRACK_URL: _url_lookup("recording", "rec-1"),
        }
    )

    assert await provider.get_mbid_by_url(SPOTIFY_ARTIST_URL, MediaType.ARTIST) == RADIOHEAD_MBID
    assert await provider.get_mbid_by_url(SPOTIFY_ALBUM_URL, MediaType.ALBUM) == "rel-1"
    assert await provider.get_mbid_by_url(SPOTIFY_TRACK_URL, MediaType.TRACK) == "rec-1"
    assert [call.kwargs["inc"] for call in get_data.await_args_list] == [
        "artist-rels",
        "release-rels",
        "recording-rels",
    ]
    assert get_data.await_args_list[0].args == ("url",)
    assert get_data.await_args_list[0].kwargs["resource"] == SPOTIFY_ARTIST_URL


async def test_mbid_by_url_ignores_ended_relations() -> None:
    """A link MusicBrainz marks as ended no longer identifies the entity it once pointed to."""
    reassigned = _url_lookup("artist", "former-owner", RADIOHEAD_MBID)
    reassigned["relations"][0]["ended"] = True
    provider, _ = _routed_provider({SPOTIFY_ARTIST_URL: reassigned})

    assert await provider.get_mbid_by_url(SPOTIFY_ARTIST_URL, MediaType.ARTIST) == RADIOHEAD_MBID

    dead = _url_lookup("artist", "former-owner")
    dead["relations"][0]["ended"] = True
    provider, _ = _routed_provider({SPOTIFY_ARTIST_URL: dead})

    assert await provider.get_mbid_by_url(SPOTIFY_ARTIST_URL, MediaType.ARTIST) is None


async def test_mbid_by_url_is_none_for_an_unknown_or_ambiguous_url() -> None:
    """An unknown URL, or one linked to several entities, identifies nothing."""
    provider, _ = _routed_provider(
        {
            SPOTIFY_ARTIST_URL: _url_lookup("artist", RADIOHEAD_MBID, "artist-other"),
            SPOTIFY_ALBUM_URL: _url_lookup("release", "rel-1", "rel-1"),
        }
    )

    assert (
        await provider.get_mbid_by_url("https://open.spotify.com/artist/unknown", MediaType.ARTIST)
        is None
    )
    assert await provider.get_mbid_by_url(SPOTIFY_ARTIST_URL, MediaType.ARTIST) is None
    # the same entity linked twice is no ambiguity
    assert await provider.get_mbid_by_url(SPOTIFY_ALBUM_URL, MediaType.ALBUM) == "rel-1"


async def test_browse_releases_by_release_group_lists_every_edition() -> None:
    """Browse a release group's releases with their media and links in one request."""
    listing = {
        "release-count": 2,
        "release-offset": 0,
        "releases": [_edition("rel-1", spotify_id="a"), _edition("rel-2", media_format="CD")],
    }
    provider, get_data = _routed_provider({"release?release-group": listing})

    releases = await provider.browse_releases_by_release_group("rg-in-rainbows")

    get_data.assert_awaited_once_with(
        "release",
        **{"release-group": "rg-in-rainbows"},
        inc="url-rels+media+release-groups",
        limit="100",
    )
    assert [release.id for release in releases] == ["rel-1", "rel-2"]
    assert relation_urls(releases[0].relations) == ["https://open.spotify.com/album/a"]
    assert releases[1].media[0].format == "CD"


async def test_browse_releases_by_release_group_is_empty_without_a_complete_listing() -> None:
    """An unknown group, or one with more releases than listed, yields no releases at all."""
    for listing in (
        None,
        {"release-count": 0, "releases": []},
        {"release-count": 30, "release-offset": 0, "releases": [_edition("rel-1")]},
        {"release-count": 1, "release-offset": 0, "releases": [{"title": "no id"}]},
    ):
        provider, _ = _routed_provider({"release?release-group": listing})
        assert await provider.browse_releases_by_release_group("rg-in-rainbows") == []


async def test_browse_releases_by_release_group_skips_a_malformed_edition_when_partial() -> None:
    """Where a partial listing will do, one edition that cannot be parsed costs only itself."""
    listing = {
        "release-count": 3,
        "release-offset": 0,
        "releases": [_edition("rel-1"), {"title": "no id"}, _edition("rel-2")],
    }
    provider, _ = _routed_provider({"release?release-group": listing})

    releases = await provider.browse_releases_by_release_group("rg-in-rainbows", complete=False)

    assert [release.id for release in releases] == ["rel-1", "rel-2"]


async def test_browse_releases_by_release_group_pages_through_a_reissued_group() -> None:
    """Where a partial listing will do, a group with more releases than a page is browsed on."""
    first_page = [_edition(f"rel-{index}") for index in range(100)]
    second_page = [_edition(f"rel-{100 + index}") for index in range(50)]
    provider, get_data = _routed_provider(
        {
            _release_browse_route(): {
                "release-count": 150,
                "release-offset": 0,
                "releases": first_page,
            },
            _release_browse_route(100): {
                "release-count": 150,
                "release-offset": 100,
                "releases": second_page,
            },
        }
    )

    releases = await provider.browse_releases_by_release_group("rg-in-rainbows", complete=False)

    assert [release.id for release in releases] == [f"rel-{index}" for index in range(150)]
    # the first page is requested as a complete browse requests it, so both share its cache entry
    assert [request.kwargs.get("offset") for request in get_data.await_args_list] == [None, "100"]


async def test_browse_releases_by_release_group_stops_at_the_page_cap() -> None:
    """A group reissued hundreds of times is cut off after a few pages, not browsed to the end."""
    provider, get_data = _routed_provider(
        {
            _release_browse_route(offset): {
                "release-count": 1000,
                "release-offset": offset,
                "releases": [_edition(f"rel-{offset + index}") for index in range(100)],
            }
            for offset in range(0, 1000, 100)
        }
    )

    releases = await provider.browse_releases_by_release_group("rg-in-rainbows", complete=False)

    assert len(releases) == RELEASE_BROWSE_MAX_PAGES * 100
    assert get_data.await_count == RELEASE_BROWSE_MAX_PAGES


# resolve_release


async def test_resolve_release_by_musicbrainz_id() -> None:
    """A known release id is looked up directly, nothing else is asked."""
    provider, get_data = _routed_provider({"release/rel-1": _release_lookup("rel-1")})
    album = _album_item(
        external_ids={(ExternalID.MB_ALBUM, "rel-1"), (ExternalID.BARCODE, BARCODE)},
        mappings={_mapping("spotify", "7eyQXxuf2nGj9d2367Gi5f")},
    )

    release = await provider.resolve_release(album)

    assert release is not None
    assert release.id == "rel-1"
    assert _requested(get_data) == ["release/rel-1"]


async def test_resolve_release_by_streaming_service_link() -> None:
    """A provider mapping is reverse-looked up through the URL MusicBrainz links the album with."""
    provider, get_data = _routed_provider(
        {
            SPOTIFY_ALBUM_URL: _url_lookup("release", "rel-1"),
            "release/rel-1": _release_lookup("rel-1"),
        }
    )
    album = _album_item(
        mappings={
            _mapping("spotify", "7eyQXxuf2nGj9d2367Gi5f"),
            _mapping("qobuz", "0634904032432"),
        },
        external_ids={(ExternalID.BARCODE, BARCODE)},
    )

    release = await provider.resolve_release(album)

    assert release is not None
    assert release.id == "rel-1"
    assert _requested(get_data) == [SPOTIFY_ALBUM_URL, "release/rel-1"]


async def test_resolve_release_by_barcode_prefers_the_official_worldwide_digital_edition() -> None:
    """Of a barcode's releases, the official digital worldwide one is fetched first and taken."""
    candidates = {
        "count": 3,
        "releases": [
            _edition("rel-cd", media_format="CD", country="GB", date="2007-12-31"),
            _edition("rel-promo", status="Promotion"),
            _edition("rel-xw"),
        ],
    }
    provider, get_data = _routed_provider(
        {"release?query": candidates, "release/rel-xw": _release_lookup("rel-xw")}
    )
    album = _album_item(external_ids={(ExternalID.BARCODE, BARCODE)})

    release = await provider.resolve_release(album)

    assert release is not None
    assert release.id == "rel-xw"
    assert _requested(get_data) == ["release?query", "release/rel-xw"]


async def test_resolve_release_by_barcode_rejects_another_album_and_stops_after_two() -> None:
    """A release that is not the album by title or artist is skipped; two are fetched at most."""
    candidates = {
        "count": 3,
        "releases": [
            _edition("rel-1", title="OK Computer"),
            _edition("rel-2", credit=_credit("Muse", "artist-muse")),
            _edition("rel-3"),
        ],
    }
    provider, get_data = _routed_provider(
        {
            "release?query": candidates,
            "release/rel-1": _release_lookup("rel-1", title="OK Computer"),
            "release/rel-2": _release_lookup(
                "rel-2", **{"artist-credit": [_credit("Muse", "artist-muse")]}
            ),
            "release/rel-3": _release_lookup("rel-3"),
        }
    )
    album = _album_item(external_ids={(ExternalID.BARCODE, BARCODE)})

    assert await provider.resolve_release(album) is None
    assert _requested(get_data) == ["release?query", "release/rel-1", "release/rel-2"]


async def test_resolve_release_reports_the_barcode_release_that_is_not_the_album() -> None:
    """The best ranked release a barcode finds that is not the album is reported, the album not."""
    candidates = {
        "count": 2,
        "releases": [_edition("rel-cd", media_format="CD"), _edition("rel-xw")],
    }
    lookups = {
        "release/rel-cd": _release_lookup("rel-cd", title="OK Computer"),
        "release/rel-xw": _release_lookup("rel-xw", title="OK Computer"),
    }
    provider, _ = _routed_provider({"release?query": candidates, **lookups})
    album = _album_item(external_ids={(ExternalID.BARCODE, BARCODE)})
    mismatches: list[tuple[str, str]] = []

    release = await provider.resolve_release(
        album, on_barcode_mismatch=lambda barcode, found: mismatches.append((barcode, found.id))
    )

    assert release is None
    assert mismatches == [(BARCODE, "rel-xw")]

    lookups["release/rel-xw"] = _release_lookup("rel-xw")
    provider, _ = _routed_provider({"release?query": candidates, **lookups})
    mismatches.clear()

    release = await provider.resolve_release(
        album, on_barcode_mismatch=lambda barcode, found: mismatches.append((barcode, found.id))
    )

    assert release is not None
    assert mismatches == []


async def test_resolve_release_tries_at_most_three_barcodes() -> None:
    """However many barcodes an album carries, three are searched at most."""
    provider, get_data = _routed_provider({})
    barcodes = ("6349040324603", "6349040324610", "6349040324627", "6349040324634", "6349040324641")
    album = _album_item(external_ids={(ExternalID.BARCODE, barcode) for barcode in barcodes})

    assert await provider.resolve_release(album) is None
    assert _requested(get_data) == ["release?query"] * 3


def test_edition_rank_prefers_the_worldwide_release() -> None:
    """Of two official digital editions, the worldwide one ranks ahead of a regional one."""
    worldwide = MusicBrainzBarcodeRelease.from_raw(_edition("rel-xw", country="XW"))
    regional = MusicBrainzBarcodeRelease.from_raw(_edition("rel-us", country="US"))

    assert _edition_rank(worldwide) < _edition_rank(regional)


async def test_resolve_release_by_barcode_accepts_various_artists_and_artistless_albums() -> None:
    """A compilation credited to Various Artists, or an album without artists, matches on title."""
    various = _credit("Various Artists", VARIOUS_ARTISTS_MBID)
    provider, _ = _routed_provider(
        {
            "release?query": {"count": 1, "releases": [_edition("rel-1", credit=various)]},
            "release/rel-1": _release_lookup("rel-1", **{"artist-credit": [various]}),
        }
    )

    for album in (
        _album_item(artist="Some Compilation Artist", external_ids={(ExternalID.BARCODE, BARCODE)}),
        _album_item(artist=None, external_ids={(ExternalID.BARCODE, BARCODE)}),
    ):
        release = await provider.resolve_release(album)
        assert release is not None
        assert release.id == "rel-1"


async def test_resolve_release_by_release_group_needs_a_single_digital_edition() -> None:
    """A release group identifies the album only when one official digital edition fits."""
    listing: dict[str, Any] = {
        "release-count": 3,
        "release-offset": 0,
        "releases": [
            _edition("rel-cd", media_format="CD"),
            _edition("rel-10", spotify_id="a"),
            _edition("rel-18", track_counts=(10, 8), spotify_id="b"),
        ],
    }
    album = _album_item(external_ids={(ExternalID.MB_RELEASEGROUP, "rg-in-rainbows")})

    provider, get_data = _routed_provider({"release?release-group": listing})
    assert await provider.resolve_release(album) is None
    assert _requested(get_data) == ["release?release-group"]

    provider, get_data = _routed_provider(
        {"release?release-group": listing, "release/rel-18": _release_lookup("rel-18")}
    )
    release = await provider.resolve_release(album, library_track_count=18)
    assert release is not None
    assert release.id == "rel-18"
    assert _requested(get_data) == ["release?release-group", "release/rel-18"]

    listing["releases"].pop()
    listing["release-count"] = 2
    provider, _ = _routed_provider(
        {"release?release-group": listing, "release/rel-10": _release_lookup("rel-10")}
    )
    release = await provider.resolve_release(album)
    assert release is not None
    assert release.id == "rel-10"


async def test_resolve_release_by_release_group_rejects_a_promotional_edition() -> None:
    """A release group whose only digital edition is not official identifies nothing."""
    listing = {
        "release-count": 2,
        "release-offset": 0,
        "releases": [
            _edition("rel-cd", media_format="CD"),
            _edition("rel-promo", status="Promotion"),
        ],
    }
    provider, get_data = _routed_provider(
        {"release?release-group": listing, "release/rel-promo": _release_lookup("rel-promo")}
    )
    album = _album_item(external_ids={(ExternalID.MB_RELEASEGROUP, "rg-in-rainbows")})

    assert await provider.resolve_release(album) is None
    assert _requested(get_data) == ["release?release-group"]


async def test_resolve_release_tries_the_release_group_after_an_unknown_barcode() -> None:
    """A barcode MusicBrainz does not know falls through to the release group."""
    provider, get_data = _routed_provider(
        {
            "release?release-group": {"release-count": 1, "releases": [_edition("rel-1")]},
            "release/rel-1": _release_lookup("rel-1"),
        }
    )
    album = _album_item(
        external_ids={(ExternalID.BARCODE, BARCODE), (ExternalID.MB_RELEASEGROUP, "rg")}
    )

    release = await provider.resolve_release(album)

    assert release is not None
    assert _requested(get_data) == ["release?query", "release?release-group", "release/rel-1"]


async def test_resolve_release_is_none_without_any_evidence() -> None:
    """An album without ids, barcodes, streaming links or a track count is never searched."""
    provider, get_data = _routed_provider({})

    assert await provider.resolve_release(_album_item()) is None
    get_data.assert_not_awaited()


async def test_resolve_release_by_name_as_the_last_resort() -> None:
    """An album nothing else identifies is found by title, primary artist and track count."""
    candidates = {
        "count": 3,
        "releases": [
            _edition("rel-cd", media_format="CD", country="GB"),
            _edition("rel-18", track_counts=(10, 8)),
            _edition("rel-xw"),
        ],
    }
    provider, get_data = _routed_provider(
        {"release?query": candidates, "release/rel-xw": _release_lookup("rel-xw")}
    )

    release = await provider.resolve_release(_album_item(), library_track_count=10)

    assert release is not None
    assert release.id == "rel-xw"
    assert _requested(get_data) == ["release?query", "release/rel-xw"]


async def test_resolve_release_by_name_needs_title_artist_and_track_count() -> None:
    """A hit with another title, primary artist or track count is not looked up."""
    candidates = {
        "count": 3,
        "releases": [
            _edition("rel-1", title="OK Computer"),
            _edition("rel-2", credit=_credit("Muse", "artist-muse")),
            _edition("rel-3", track_counts=(12,)),
        ],
    }
    provider, get_data = _routed_provider({"release?query": candidates})

    assert await provider.resolve_release(_album_item(), library_track_count=10) is None
    assert _requested(get_data) == ["release?query"]


async def test_resolve_release_by_name_abstains_on_several_release_groups() -> None:
    """Two albums of the same name by the same artist cannot be told apart by name."""
    other_album = _edition("rel-2")
    other_album["release-group"] = {"id": "rg-other", "title": "In Rainbows"}
    provider, get_data = _routed_provider(
        {"release?query": {"count": 2, "releases": [_edition("rel-1"), other_album]}}
    )

    assert await provider.resolve_release(_album_item(), library_track_count=10) is None
    assert _requested(get_data) == ["release?query"]


async def test_resolve_release_by_name_abstains_on_an_incomplete_search() -> None:
    """A truncated search or one with an unparsable hit may hide another album."""
    for result in (
        {"count": 101, "releases": [_edition("rel-1")]},
        {"count": 2, "releases": [_edition("rel-1"), {"title": "In Rainbows"}]},
    ):
        provider, get_data = _routed_provider(
            {"release?query": result, "release/rel-1": _release_lookup("rel-1")}
        )

        assert await provider.resolve_release(_album_item(), library_track_count=10) is None
        assert _requested(get_data) == ["release?query"]


async def test_resolve_release_by_name_looks_up_two_editions_at_most() -> None:
    """A name search spends one search and at most two release lookups."""
    candidates = {"count": 3, "releases": [_edition(f"rel-{index}") for index in range(3)]}
    provider, get_data = _routed_provider(
        {
            "release?query": candidates,
            **{
                f"release/rel-{index}": _release_lookup(f"rel-{index}", title="OK Computer")
                for index in range(3)
            },
        }
    )

    assert await provider.resolve_release(_album_item(), library_track_count=10) is None
    assert _requested(get_data) == ["release?query", "release/rel-0", "release/rel-1"]


async def test_resolve_release_by_name_needs_an_artist() -> None:
    """An album without artists is never searched by name."""
    provider, get_data = _routed_provider({})

    assert await provider.resolve_release(_album_item(artist=None), library_track_count=10) is None
    get_data.assert_not_awaited()


# resolve_recording


async def test_resolve_recording_by_musicbrainz_id_or_link() -> None:
    """A known recording id is looked up directly, a streaming link reverse-looked up."""
    provider, get_data = _routed_provider(
        {
            "recording/rec-1": _recording_lookup("rec-1"),
            SPOTIFY_TRACK_URL: _url_lookup("recording", "rec-1"),
        }
    )

    by_id = await provider.resolve_recording(
        _track_item(external_ids={(ExternalID.MB_RECORDING, "rec-1"), (ExternalID.ISRC, ISRC)})
    )
    by_link = await provider.resolve_recording(
        _track_item(mappings={_mapping("spotify", "2Ex8hBvUhZjXjJpZjJZ0aA")})
    )

    assert by_id is not None
    assert (by_id[0].id, by_id[1]) == ("rec-1", None)
    assert by_link is not None
    assert (by_link[0].id, by_link[1]) == ("rec-1", RecordingSource.URL)
    assert _requested(get_data) == ["recording/rec-1", SPOTIFY_TRACK_URL, "recording/rec-1"]


async def test_resolve_recording_by_isrc_picks_the_matching_recording() -> None:
    """Of an ISRC's recordings, only one close in length and title counts, the artist's first."""
    isrc_lookup = {
        "isrc": ISRC,
        "recordings": [
            _recording_lookup("rec-long", length=300000),
            _recording_lookup("rec-live", title="15 Step (Live)"),
            _recording_lookup("rec-cover", credit=_credit("Some Cover Band", "artist-cover")),
            _recording_lookup("rec-radiohead"),
        ],
    }
    provider, get_data = _routed_provider(
        {f"isrc/{ISRC}": isrc_lookup, "recording/rec-radiohead": _recording_lookup("rec-radiohead")}
    )

    recording = await provider.resolve_recording(
        _track_item(external_ids={(ExternalID.ISRC, ISRC)})
    )

    assert recording is not None
    assert (recording[0].id, recording[1]) == ("rec-radiohead", RecordingSource.ISRC)
    assert _requested(get_data) == [f"isrc/{ISRC}", "recording/rec-radiohead"]


def test_length_matches_up_to_the_tolerance() -> None:
    """A recording within the length tolerance of the track is it, one beyond it is not."""
    track = _track_item(duration=230)

    assert _length_matches(MusicBrainzRecording(id="r", title="15 Step", length=237999), track)
    assert not _length_matches(MusicBrainzRecording(id="r", title="15 Step", length=238001), track)


async def test_resolve_recording_by_isrc_needs_a_credit_for_a_named_artist() -> None:
    """A recording crediting none of the track's artists is not the track; artistless tracks take the first fit."""
    isrc_lookup = {
        "isrc": ISRC,
        "recordings": [
            _recording_lookup("rec-cover", credit=_credit("Some Cover Band", "artist-cover")),
            _recording_lookup("rec-other", credit=_credit("Other Band", "artist-other")),
        ],
    }
    provider, _ = _routed_provider(
        {f"isrc/{ISRC}": isrc_lookup, "recording/rec-cover": _recording_lookup("rec-cover")}
    )
    isrc_only = {(ExternalID.ISRC, ISRC)}

    # no album, so the name search cannot answer either: the ISRC leg alone decides
    assert await provider.resolve_recording(_track_item(album=None, external_ids=isrc_only)) is None

    recording = await provider.resolve_recording(_track_item(artist=None, external_ids=isrc_only))

    assert recording is not None
    assert recording[0].id == "rec-cover"


async def test_resolve_recording_by_isrc_of_one_recording_needs_no_title() -> None:
    """An ISRC naming one recording identifies a track titled differently, by length and artist."""
    work_title = "Symphony no. 5 in C minor, op. 67: I. Allegro con brio"
    isrc_lookup = {"isrc": ISRC, "recordings": [_recording_lookup("rec-1", title=work_title)]}
    provider, _ = _routed_provider(
        {f"isrc/{ISRC}": isrc_lookup, "recording/rec-1": _recording_lookup("rec-1")}
    )
    isrc_only = {(ExternalID.ISRC, ISRC)}

    recording = await provider.resolve_recording(
        _track_item("I. Allegro con brio", album=None, external_ids=isrc_only)
    )

    assert recording is not None
    assert recording[0].id == "rec-1"


@pytest.mark.parametrize(
    ("recordings", "track"),
    [
        # two recordings share the ISRC, so the title still decides
        (
            [_recording_lookup("rec-1", title="Symphony no. 5: I"), _recording_lookup("rec-2")],
            _track_item("I. Allegro con brio", album=None, external_ids={(ExternalID.ISRC, ISRC)}),
        ),
        # far off in length
        (
            [_recording_lookup("rec-1", title="Symphony no. 5: I", length=300000)],
            _track_item("I. Allegro con brio", album=None, external_ids={(ExternalID.ISRC, ISRC)}),
        ),
        # crediting none of the track's artists
        (
            [
                _recording_lookup(
                    "rec-1", title="Symphony no. 5: I", credit=_credit("Other", "artist-other")
                )
            ],
            _track_item("I. Allegro con brio", album=None, external_ids={(ExternalID.ISRC, ISRC)}),
        ),
        # a track without artists cannot be checked against the credit
        (
            [_recording_lookup("rec-1", title="Symphony no. 5: I")],
            _track_item(
                "I. Allegro con brio",
                artist=None,
                album=None,
                external_ids={(ExternalID.ISRC, ISRC)},
            ),
        ),
    ],
)
async def test_resolve_recording_by_isrc_without_a_title_match_rejects(
    recordings: list[dict[str, Any]], track: Track
) -> None:
    """Without the title, a shared ISRC alone is not enough for several recordings or a doubt."""
    provider, _ = _routed_provider(
        {
            f"isrc/{ISRC}": {"isrc": ISRC, "recordings": recordings},
            "recording/rec-1": _recording_lookup("rec-1"),
            "recording/rec-2": _recording_lookup("rec-2"),
        }
    )

    assert await provider.resolve_recording(track) is None


@pytest.mark.parametrize(
    ("album_ids", "release_recording", "expected"),
    [
        ({(ExternalID.MB_ALBUM, "rel-1")}, "rec-1", "REC-1"),
        (set(), "rec-1", None),
        ({(ExternalID.MB_ALBUM, "rel-1")}, "rec-other", None),
    ],
)
async def test_resolve_recording_by_isrc_on_the_album_release_needs_no_artist_credit(
    album_ids: set[tuple[ExternalID, str]], release_recording: str, expected: str | None
) -> None:
    """A recording crediting other artists is the track when it is on the album's release."""
    release = _release_lookup("rel-1")
    release["media"][0]["tracks"][0]["recording"]["id"] = release_recording
    performer_credit = _credit("The Royal Festival Orchestra", "artist-orchestra")
    isrc_lookup = {
        "isrc": ISRC,
        "recordings": [
            _recording_lookup("REC-1", title="Dance of the Hours", credit=performer_credit)
        ],
    }
    provider, get_data = _routed_provider(
        {
            f"isrc/{ISRC}": isrc_lookup,
            "release/rel-1": release,
            "recording/REC-1": _recording_lookup("REC-1"),
        }
    )
    provider.mass = MagicMock()
    provider.mass.music.albums.get_library_item = AsyncMock(
        return_value=_album_item(external_ids=album_ids)
    )
    track = _track_item(
        "Ponchielli: Dance of the Hours",
        artist="Amilcare Ponchielli",
        external_ids={(ExternalID.ISRC, ISRC)},
    )

    recording = await provider.resolve_recording(track)

    assert (recording[0].id if recording else None) == expected
    provider.mass.music.albums.get_library_item.assert_awaited_once_with("b")
    expected_requests = [f"isrc/{ISRC}"]
    if album_ids:
        expected_requests.append("release/rel-1")
    if expected:
        expected_requests.append("recording/REC-1")
    # the name search finds nothing here, so the ISRC leg alone decides
    assert [r for r in _requested(get_data) if r != "recording?query"] == expected_requests


@pytest.mark.parametrize(
    ("album_ids", "expected"),
    [
        (
            {(ExternalID.MB_ALBUM, "rel-1")},
            f'ISRC {ISRC} of "15 Step" (album "In Rainbows") is not on MusicBrainz, add it to '
            "the track's recording: https://musicbrainz.org/release/rel-1",
        ),
        (
            set(),
            f'ISRC {ISRC} of "15 Step" (album "In Rainbows") is not on MusicBrainz, add it to '
            "the track's recording",
        ),
    ],
)
async def test_resolve_recording_logs_an_isrc_musicbrainz_lacks(
    album_ids: set[tuple[ExternalID, str]], expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    """An ISRC without any recording is logged, with the album's release when it is known."""
    provider, _ = _routed_provider({})
    provider.logger = logging.getLogger("test_musicbrainz")
    caplog.set_level(logging.DEBUG, logger="test_musicbrainz")
    provider.mass = MagicMock()
    provider.mass.music.albums.get_library_item = AsyncMock(
        return_value=_album_item(external_ids=album_ids)
    )

    await provider.resolve_recording(_track_item(external_ids={(ExternalID.ISRC, ISRC)}))

    assert [record.getMessage() for record in caplog.records] == [expected]
    provider.mass.music.albums.get_library_item.assert_awaited_once_with("b")


async def test_resolve_recording_logs_a_recording_waiting_for_its_album(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A recording crediting other artists is logged when the album is not identified."""
    performer_credit = _credit("The Royal Festival Orchestra", "artist-orchestra")
    isrc_lookup = {
        "isrc": ISRC,
        "recordings": [
            _recording_lookup("rec-1", title="Dance of the Hours", credit=performer_credit)
        ],
    }
    provider, _ = _routed_provider({f"isrc/{ISRC}": isrc_lookup})
    provider.logger = logging.getLogger("test_musicbrainz")
    caplog.set_level(logging.DEBUG, logger="test_musicbrainz")
    provider.mass = MagicMock()
    provider.mass.music.albums.get_library_item = AsyncMock(return_value=_album_item())
    track = _track_item(
        "Ponchielli: Dance of the Hours",
        artist="Amilcare Ponchielli",
        external_ids={(ExternalID.ISRC, ISRC)},
    )

    assert await provider.resolve_recording(track) is None

    assert [record.getMessage() for record in caplog.records] == [
        f'ISRC {ISRC} of "Ponchielli: Dance of the Hours" is on MusicBrainz recording '
        "https://musicbrainz.org/recording/rec-1, which credits other artists, so it needs "
        'album "In Rainbows" identified, add its streaming link or barcode to the release of '
        "that recording"
    ]


async def test_resolve_recording_logs_nothing_for_an_isrc_of_other_recordings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ISRC whose recordings are not the track is not reported as missing."""
    isrc_lookup = {"isrc": ISRC, "recordings": [_recording_lookup("rec-1", length=300000)]}
    provider, _ = _routed_provider({f"isrc/{ISRC}": isrc_lookup})
    provider.logger = logging.getLogger("test_musicbrainz")
    caplog.set_level(logging.DEBUG, logger="test_musicbrainz")

    track = _track_item(album=None, external_ids={(ExternalID.ISRC, ISRC)})
    assert await provider.resolve_recording(track) is None

    assert caplog.records == []


async def test_resolve_recording_falls_back_to_a_name_search() -> None:
    """A track with an album and artist but no ids is searched by name as the last resort."""
    search_result = {
        "count": 1,
        "recordings": [
            {
                **_recording_lookup("rec-1"),
                "releases": [
                    {
                        "id": "rel-1",
                        "title": "In Rainbows",
                        "release-group": {"id": "rg", "title": "In Rainbows"},
                    }
                ],
            }
        ],
    }
    provider, get_data = _routed_provider(
        {"recording?query": search_result, "recording/rec-1": _recording_lookup("rec-1")}
    )

    recording = await provider.resolve_recording(_track_item())

    assert recording is not None
    assert (recording[0].id, recording[1]) == ("rec-1", RecordingSource.SEARCH)
    assert _requested(get_data) == ["recording?query", "recording/rec-1"]
    assert await provider.resolve_recording(_track_item(album=None)) is None


# resolve_artist


async def test_resolve_artist_by_musicbrainz_id_or_various_artists_name() -> None:
    """A known id, or the Various Artists name, costs only the artist lookup itself."""
    provider, get_data = _routed_provider(
        {
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
            f"artist/{VARIOUS_ARTISTS_MBID}": _artist_lookup(
                VARIOUS_ARTISTS_MBID, "Various Artists"
            ),
        }
    )
    known = _artist_item(
        external_ids={(ExternalID.MB_ARTIST, RADIOHEAD_MBID)},
        mappings={_mapping("spotify", "4Z8W4fKeB5YxbusRsdQVPb")},
    )

    radiohead = await provider.resolve_artist(known, [_album_item()], [_track_item()])
    various = await provider.resolve_artist(_artist_item("Various Artists"), [], [])

    assert radiohead is not None

    assert radiohead.id == RADIOHEAD_MBID
    assert relation_urls(radiohead.relations) == [SPOTIFY_ARTIST_URL]
    assert various is not None
    assert various.id == VARIOUS_ARTISTS_MBID
    assert _requested(get_data) == [f"artist/{RADIOHEAD_MBID}", f"artist/{VARIOUS_ARTISTS_MBID}"]


async def test_resolve_artist_reverse_looks_up_at_most_three_links_in_a_fixed_order() -> None:
    """Streaming links are tried Spotify, Deezer, Tidal first and never more than three."""
    provider, get_data = _routed_provider({})
    artist = _artist_item(
        mappings={
            _mapping("ytmusic", "UCr_iyUANcn9OX_yy9piYoLw"),
            _mapping("apple_music", "657515", "https://music.apple.com/gb/artist/radiohead/657515"),
            _mapping("tidal", "64518"),
            _mapping("deezer", "399"),
            _mapping("spotify", "4Z8W4fKeB5YxbusRsdQVPb"),
        }
    )

    assert await provider.resolve_artist(artist, [], []) is None
    assert _requested(get_data) == [
        SPOTIFY_ARTIST_URL,
        "https://www.deezer.com/artist/399",
        "https://tidal.com/artist/64518",
    ]


async def test_resolve_artist_by_apple_music_link_uses_the_mapping_storefront() -> None:
    """An Apple Music mapping is looked up on the storefront its URL names."""
    apple_url = "https://music.apple.com/gb/artist/657515"
    provider, get_data = _routed_provider(
        {
            apple_url: _url_lookup("artist", RADIOHEAD_MBID),
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    artist = _artist_item(
        mappings={
            _mapping("apple_music", "657515", "https://music.apple.com/gb/artist/radiohead/657515")
        }
    )

    resolved = await provider.resolve_artist(artist, [], [])

    assert resolved is not None

    assert resolved.id == RADIOHEAD_MBID
    assert _requested(get_data) == [apple_url, f"artist/{RADIOHEAD_MBID}"]


async def test_resolve_artist_through_reference_items() -> None:
    """A reference album's release group, or a reference track's ISRC, names the artist."""
    provider, get_data = _routed_provider(
        {
            "release-group/rg": {
                "id": "rg",
                "title": "In Rainbows",
                "artist-credit": [_radiohead_credit()],
            },
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    ref_album = _album_item(external_ids={(ExternalID.MB_RELEASEGROUP, "rg")})

    resolved = await provider.resolve_artist(_artist_item(), [ref_album], [])

    assert resolved is not None

    assert resolved.id == RADIOHEAD_MBID
    assert _requested(get_data) == ["release-group/rg", f"artist/{RADIOHEAD_MBID}"]

    provider, get_data = _routed_provider(
        {
            f"isrc/{ISRC}": {"isrc": ISRC, "recordings": [_recording_lookup("rec-1")]},
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    ref_track = _track_item(external_ids={(ExternalID.ISRC, ISRC)})

    # the alias matches too
    resolved = await provider.resolve_artist(_artist_item("レディオヘッド"), [], [ref_track])

    assert resolved is not None

    assert resolved.id == RADIOHEAD_MBID
    assert _requested(get_data) == [f"isrc/{ISRC}", f"artist/{RADIOHEAD_MBID}"]


async def test_resolve_artist_by_barcode_matches_the_release_credit() -> None:
    """A reference album's barcode names the artist through the release's credit."""
    provider, get_data = _routed_provider(
        {
            "release?query": {"count": 1, "releases": [_edition("rel-1")]},
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    ref_album = _album_item(external_ids={(ExternalID.BARCODE, BARCODE)})

    resolved = await provider.resolve_artist(_artist_item(), [ref_album], [])

    assert resolved is not None

    assert resolved.id == RADIOHEAD_MBID
    assert _requested(get_data) == ["release?query", f"artist/{RADIOHEAD_MBID}"]
    # a release credited to someone else is no evidence
    provider, _ = _routed_provider(
        {
            "release?query": {
                "count": 1,
                "releases": [_edition("rel-1", credit=_credit("Muse", "m"))],
            }
        }
    )
    assert await provider.resolve_artist(_artist_item(), [ref_album], []) is None


async def test_resolve_artist_by_barcode_checks_every_release_of_the_barcode() -> None:
    """The matching credit may sit on any of the releases carrying the barcode."""
    provider, get_data = _routed_provider(
        {
            "release?query": {
                "count": 2,
                "releases": [
                    _edition("rel-1", credit=_credit("Muse", "m")),
                    _edition("rel-2"),
                ],
            },
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    ref_album = _album_item(external_ids={(ExternalID.BARCODE, BARCODE)})

    resolved = await provider.resolve_artist(_artist_item(), [ref_album], [])

    assert resolved is not None
    assert resolved.id == RADIOHEAD_MBID
    assert _requested(get_data) == ["release?query", f"artist/{RADIOHEAD_MBID}"]


async def test_identify_artist_tells_a_name_search_apart() -> None:
    """Only an artist that a name search alone identifies is reported as found by name."""
    provider, _ = _routed_provider(
        {
            SPOTIFY_ARTIST_URL: _url_lookup("artist", RADIOHEAD_MBID),
            "recording?query": {
                "recordings": [
                    {
                        **_recording_lookup("rec-1"),
                        "releases": [
                            {
                                "title": "In Rainbows",
                                "release-group": {"id": "rg", "title": "In Rainbows"},
                            }
                        ],
                    }
                ]
            },
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    by_link = _artist_item(mappings={_mapping("spotify", "4Z8W4fKeB5YxbusRsdQVPb")})

    linked = await provider.identify_artist(by_link, [], [])
    searched = await provider.identify_artist(_artist_item(), [], [_track_item()])

    assert linked is not None
    assert (linked.artist.id, linked.by_name) == (RADIOHEAD_MBID, False)
    assert searched is not None
    assert (searched.artist.id, searched.by_name) == (RADIOHEAD_MBID, True)


def test_reverse_lookup_urls_take_one_canonical_url_per_provider_first() -> None:
    """A provider's first mapping is looked up canonically; its other mappings come after."""
    artist = _artist_item(
        mappings={
            _mapping(
                "spotify",
                "7eyQXxuf2nGj9d2367Gi5f",
                "https://open.spotify.com/artist/7eyQXxuf2nGj9d2367Gi5f",
            ),
            _mapping("spotify", "4Z8W4fKeB5YxbusRsdQVPb", SPOTIFY_ARTIST_URL),
            _mapping("deezer", "399"),
        }
    )

    assert MusicbrainzProvider.reverse_lookup_urls(artist) == [
        SPOTIFY_ARTIST_URL,
        "https://www.deezer.com/artist/399",
        "https://open.spotify.com/artist/7eyQXxuf2nGj9d2367Gi5f",
    ]


def test_reverse_lookup_urls_include_the_mappings_own_public_links() -> None:
    """Mappings without a canonical form contribute their own URL, public catalog hosts only."""
    artist = _artist_item(
        mappings={
            _mapping("apple_music", "657515", "https://music.apple.com/gb/artist/radiohead/657515"),
            _mapping("soundcloud", "radiohead", "https://soundcloud.com/radiohead"),
            _mapping("bandcamp", "radiohead", "https://radiohead.bandcamp.com/"),
            # a local server's URL never goes out
            _mapping(
                "plex", "1", "http://192.168.1.5:32400/library/metadata/1?X-Plex-Token=s3cret"
            ),
            _mapping("jellyfin", "2", "plex://artist/2"),
        }
    )

    assert MusicbrainzProvider.reverse_lookup_urls(artist) == [
        "https://music.apple.com/gb/artist/657515",
        "https://radiohead.bandcamp.com/",
        "https://soundcloud.com/radiohead",
    ]


def test_reverse_lookup_urls_include_qobuz_artists_only() -> None:
    """Qobuz artists are looked up by their Qobuz page, albums and name-only performers never."""
    artist = _artist_item(
        mappings={_mapping("qobuz", "43840", "https://open.qobuz.com/artist/43840")}
    )
    album = _album_item(
        mappings={_mapping("qobuz", "ftrty0qq43i3b", "https://open.qobuz.com/album/ftrty0qq43i3b")}
    )

    assert MusicbrainzProvider.reverse_lookup_urls(artist) == [
        "https://open.qobuz.com/artist/43840"
    ]
    assert MusicbrainzProvider.reverse_lookup_urls(album) == []
    assert (
        MusicbrainzProvider.reverse_lookup_urls(
            _artist_item(mappings={_mapping("qobuz", "Herbert von Karajan")})
        )
        == []
    )
    assert (
        MusicbrainzProvider.reverse_lookup_urls(_artist_item(mappings={_mapping("qobuz", "311")}))
        == []
    )


async def test_resolve_artist_by_qobuz_link() -> None:
    """A Qobuz artist mapping is reverse-looked up through its Qobuz page."""
    qobuz_url = "https://open.qobuz.com/artist/43840"
    provider, get_data = _routed_provider(
        {
            qobuz_url: _url_lookup("artist", RADIOHEAD_MBID),
            f"artist/{RADIOHEAD_MBID}": _artist_lookup(),
        }
    )
    artist = _artist_item(mappings={_mapping("qobuz", "43840", qobuz_url)})

    resolved = await provider.resolve_artist(artist, [], [])

    assert resolved is not None
    assert resolved.id == RADIOHEAD_MBID
    assert _requested(get_data) == [qobuz_url, f"artist/{RADIOHEAD_MBID}"]


def test_reverse_lookup_urls_are_capped_at_three() -> None:
    """Never more than three URLs, the canonical ones taking precedence."""
    artist = _artist_item(
        mappings={
            _mapping("soundcloud", "radiohead", "https://soundcloud.com/radiohead"),
            _mapping("tidal", "64518"),
            _mapping("deezer", "399"),
            _mapping("spotify", "4Z8W4fKeB5YxbusRsdQVPb"),
        }
    )

    assert MusicbrainzProvider.reverse_lookup_urls(artist) == [
        SPOTIFY_ARTIST_URL,
        "https://www.deezer.com/artist/399",
        "https://tidal.com/artist/64518",
    ]


async def test_resolve_artist_spends_a_bounded_number_of_requests_per_leg() -> None:
    """However many reference items there are, each lookup leg tries three at most, in order."""
    provider, get_data = _routed_provider({})
    ref_albums = [
        _album_item(
            f"Album {i}",
            external_ids={(ExternalID.MB_RELEASEGROUP, f"rg-{i}"), (ExternalID.BARCODE, BARCODE)},
        )
        for i in range(10)
    ]
    ref_tracks = [
        _track_item(
            f"Track {i}",
            external_ids={(ExternalID.MB_RECORDING, f"rec-{i}"), (ExternalID.ISRC, ISRC)},
        )
        for i in range(10)
    ]

    assert await provider.resolve_artist(_artist_item(), ref_albums, ref_tracks) is None

    assert _requested(get_data) == [
        "release-group/rg-0",
        "release-group/rg-1",
        "release-group/rg-2",
        "recording/rec-0",
        "recording/rec-1",
        "recording/rec-2",
        "release?query",
        "release?query",
        "release?query",
        f"isrc/{ISRC}",
        f"isrc/{ISRC}",
        f"isrc/{ISRC}",
        # the name search tries a strict and a loose query per track
        *["recording?query"] * 6,
    ]


def test_relation_urls_skips_ended_links_and_duplicates() -> None:
    """Only current links count, each once, whatever relation type they carry."""
    relations = [
        MusicBrainzRelation.from_dict({"type": "free streaming", "url": {"resource": "a"}}),
        MusicBrainzRelation.from_dict({"type": "streaming", "url": {"resource": "a"}}),
        MusicBrainzRelation.from_dict(
            {"type": "streaming", "url": {"resource": "b"}, "ended": True}
        ),
        MusicBrainzRelation.from_dict({"type": "member of band"}),
        MusicBrainzRelation.from_dict({"type": "discogs", "url": {"resource": "c"}}),
    ]

    assert relation_urls(relations) == ["a", "c"]
    assert relation_urls(None) == []


@pytest.mark.parametrize(
    ("primary_type", "secondary_types", "expected"),
    [
        ("Album", None, AlbumType.ALBUM),
        ("Album", [], AlbumType.ALBUM),
        ("Single", None, AlbumType.SINGLE),
        ("EP", None, AlbumType.EP),
        ("Album", ["Compilation"], AlbumType.COMPILATION),
        ("Album", ["Soundtrack"], AlbumType.SOUNDTRACK),
        ("Album", ["Live"], AlbumType.LIVE),
        ("Album", ["Live", "Compilation"], AlbumType.COMPILATION),
        ("Album", ["Remix"], AlbumType.ALBUM),
        ("Other", None, AlbumType.UNKNOWN),
        (None, None, AlbumType.UNKNOWN),
    ],
)
def test_album_type_from_release_group(
    primary_type: str | None, secondary_types: list[str] | None, expected: AlbumType
) -> None:
    """Map a release group's types to an album type, a secondary type taking precedence."""
    release_group = MusicBrainzReleaseGroup(
        id="rg", title="x", primary_type=primary_type, secondary_types=secondary_types
    )
    assert MusicbrainzProvider.album_type_from_release_group(release_group) == expected


# ---------------------------------------------------------------------------
# album, track and artist metadata
# ---------------------------------------------------------------------------

DISCOGS_RELEASE_URL = "https://www.discogs.com/release/1119453"
DISCOGS_ARTIST_URL = "https://www.discogs.com/artist/3840"


def test_metadata_features_cover_artists_albums_and_tracks() -> None:
    """MusicBrainz offers metadata for every library item type it identifies."""
    assert {
        ProviderFeature.ARTIST_METADATA,
        ProviderFeature.ALBUM_METADATA,
        ProviderFeature.TRACK_METADATA,
    } <= SUPPORTED_FEATURES


async def test_album_metadata_surfaces_genres_label_release_date_and_links() -> None:
    """An album gets the genres of its release and group, its label, date and Discogs link."""
    provider, get_data = _routed_provider(
        {
            "release/rel-1": _release_lookup(
                "rel-1",
                genres=[{"id": "g2", "name": "art rock", "count": 2, "disambiguation": ""}],
                relations=[
                    _url_relation(SPOTIFY_ALBUM_URL),
                    _url_relation(DISCOGS_RELEASE_URL, type_="discogs"),
                    _url_relation("https://www.discogs.com/release/1", type_="discogs", ended=True),
                ],
            )
        }
    )
    album = _album_item(external_ids={(ExternalID.MB_ALBUM, "rel-1")})

    metadata = await provider.get_album_metadata(album)

    assert metadata is not None
    assert metadata.genres == {"alternative rock", "art rock"}
    assert metadata.label == "XL Recordings"
    assert metadata.release_date == datetime(2016, 5, 6, tzinfo=UTC)
    assert metadata.links == {MediaItemLink(type=LinkType.DISCOGS, url=DISCOGS_RELEASE_URL)}
    assert _requested(get_data) == ["release/rel-1"]


async def test_album_metadata_release_date_needs_a_full_date() -> None:
    """A release dated to the month or year gives no release date."""
    for date in ("2016-05", "2016", "", None):
        provider, _ = _routed_provider({"release/rel-1": _release_lookup("rel-1", date=date)})
        metadata = await provider.get_album_metadata(
            _album_item(external_ids={(ExternalID.MB_ALBUM, "rel-1")})
        )
        assert metadata is not None
        assert metadata.release_date is None


async def test_album_metadata_is_none_without_an_id_or_anything_to_tell() -> None:
    """No MusicBrainz id, an unknown id or a bare release yields no metadata."""
    provider, get_data = _routed_provider({})
    assert await provider.get_album_metadata(_album_item()) is None
    get_data.assert_not_awaited()

    unknown = _album_item(external_ids={(ExternalID.MB_ALBUM, "rel-unknown")})
    assert await provider.get_album_metadata(unknown) is None

    bare = _release_lookup(
        "rel-1", date="2016", relations=[], **{"label-info": [], "release-group": None}
    )
    provider, _ = _routed_provider({"release/rel-1": bare})
    known = _album_item(external_ids={(ExternalID.MB_ALBUM, "rel-1")})
    assert await provider.get_album_metadata(known) is None


async def test_track_metadata_surfaces_the_recording_genres() -> None:
    """A track gets the genres of its recording, nothing when the recording has none."""
    genres = [{"id": "g1", "name": "alternative rock", "count": 5, "disambiguation": ""}]
    provider, _ = _routed_provider(
        {
            "recording/rec-1": {**_recording_lookup("rec-1"), "genres": genres},
            "recording/rec-2": _recording_lookup("rec-2"),
        }
    )

    tagged = await provider.get_track_metadata(
        _track_item(external_ids={(ExternalID.MB_RECORDING, "rec-1")})
    )
    assert tagged is not None
    assert tagged.genres == {"alternative rock"}

    plain = _track_item(external_ids={(ExternalID.MB_RECORDING, "rec-2")})
    assert await provider.get_track_metadata(plain) is None
    assert await provider.get_track_metadata(_track_item()) is None


async def test_artist_metadata_includes_genres_and_current_typed_links() -> None:
    """An artist gets its MusicBrainz genres and its current Discogs and homepage links."""
    lookup = _artist_lookup()
    lookup["relations"] = [
        _url_relation(SPOTIFY_ARTIST_URL),
        _url_relation(DISCOGS_ARTIST_URL, type_="discogs"),
        _url_relation("https://radiohead.com/", type_="official homepage"),
        _url_relation("https://old.example.com/", type_="official homepage", ended=True),
    ]
    provider, _ = _routed_provider({f"artist/{RADIOHEAD_MBID}": lookup})

    metadata = await provider.get_artist_metadata(
        _artist_item(external_ids={(ExternalID.MB_ARTIST, RADIOHEAD_MBID)})
    )

    assert metadata is not None
    assert metadata.genres == {"alternative rock"}
    assert metadata.links == {
        MediaItemLink(type=LinkType.DISCOGS, url=DISCOGS_ARTIST_URL),
        MediaItemLink(type=LinkType.WEBSITE, url="https://radiohead.com/"),
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"country": "PL"}, "PL"),
        ({"country": "XW"}, None),
        ({"area": {"name": "England", "iso-3166-2-codes": ["GB-ENG"]}}, "GB"),
        ({"area": {"name": "Warsaw"}}, None),
        ({}, None),
    ],
)
def test_artist_country_is_read_from_the_country_or_its_subdivision(
    raw: dict[str, Any], expected: str | None
) -> None:
    """An artist's country is its ISO code, else that of the country subdivision it is from."""
    artist = MusicBrainzArtist.from_raw({**_artist_lookup(), **raw})

    assert artist.country_code == expected


async def test_artist_metadata_includes_the_country() -> None:
    """An artist gets the country MusicBrainz reports for them."""
    provider, _ = _routed_provider(
        {f"artist/{RADIOHEAD_MBID}": {**_artist_lookup(), "country": "GB"}}
    )

    metadata = await provider.get_artist_metadata(
        _artist_item(external_ids={(ExternalID.MB_ARTIST, RADIOHEAD_MBID)})
    )

    assert metadata is not None
    assert metadata.country == "GB"


# ---------------------------------------------------------------------------
# browse_release_groups_by_artist
# ---------------------------------------------------------------------------


def _release_group(
    group_id: str,
    *,
    primary_type: str | None = "Album",
    secondary_types: list[str] | None = None,
    first_release_date: str | None = "2007-10-10",
) -> dict[str, Any]:
    """Return one release group as an artist browse lists it."""
    return {
        "id": group_id,
        "title": "In Rainbows",
        "primary-type": primary_type,
        "secondary-types": secondary_types or [],
        "first-release-date": first_release_date,
    }


def _browse_page(groups: list[dict[str, Any]], offset: int, count: int) -> dict[str, Any]:
    """Return one page of an artist's release group browse."""
    return {"release-group-count": count, "release-group-offset": offset, "release-groups": groups}


def _browsing_provider(pages: dict[str, Any]) -> tuple[MusicbrainzProvider, MagicMock]:
    """Return a MusicbrainzProvider and its mock API client, answering a browse per offset."""
    with patch.object(MusicbrainzProvider, "__init__", lambda *_a, **_kw: None):
        provider = MusicbrainzProvider.__new__(MusicbrainzProvider)
    provider.logger = MagicMock()
    provider._logged_hints = set()

    async def _answer(_endpoint: str, **kwargs: Any) -> Any:
        return pages.get(kwargs["offset"])

    api_client = MagicMock()
    api_client.get_data = AsyncMock(return_value=None)
    api_client.get_browse_data = AsyncMock(side_effect=_answer)
    provider._api_client = api_client
    return provider, api_client


async def test_browse_release_groups_pages_through_a_discography() -> None:
    """A discography longer than a page is browsed page by page, through the browse cache."""
    first_page = [_release_group(f"rg-{index}") for index in range(100)]
    second_page = [_release_group(f"rg-{100 + index}") for index in range(50)]
    provider, api_client = _browsing_provider(
        {"0": _browse_page(first_page, 0, 150), "100": _browse_page(second_page, 100, 150)}
    )

    groups = await provider.browse_release_groups_by_artist(RADIOHEAD_MBID)

    assert len(groups) == 150
    assert [
        (*request.args, request.kwargs) for request in api_client.get_browse_data.await_args_list
    ] == [
        ("release-group", {"artist": RADIOHEAD_MBID, "limit": "100", "offset": "0"}),
        ("release-group", {"artist": RADIOHEAD_MBID, "limit": "100", "offset": "100"}),
    ]
    api_client.get_data.assert_not_awaited()


async def test_browse_release_groups_stops_at_the_page_cap() -> None:
    """A catalog-sized discography is cut off after ten pages."""
    pages = {
        str(offset): _browse_page(
            [_release_group(f"rg-{offset + index}") for index in range(100)], offset, 2000
        )
        for offset in range(0, 2000, 100)
    }
    provider, api_client = _browsing_provider(pages)

    groups = await provider.browse_release_groups_by_artist(RADIOHEAD_MBID)

    assert len(groups) == 1000
    assert api_client.get_browse_data.await_count == 10


async def test_browse_release_groups_keeps_albums_eps_and_singles_only() -> None:
    """Broadcasts and untyped groups are left out, while secondary types are kept."""
    listing = [
        _release_group("rg-album"),
        _release_group("rg-live", secondary_types=["Live"]),
        _release_group("rg-ep", primary_type="EP"),
        _release_group("rg-single", primary_type="Single"),
        _release_group("rg-broadcast", primary_type="Broadcast"),
        _release_group("rg-other", primary_type="Other"),
        _release_group("rg-untyped", primary_type=None),
    ]
    provider, _ = _browsing_provider({"0": _browse_page(listing, 0, len(listing))})

    groups = await provider.browse_release_groups_by_artist(RADIOHEAD_MBID)

    assert [group.id for group in groups] == ["rg-album", "rg-live", "rg-ep", "rg-single"]
    assert MusicbrainzProvider.album_type_from_release_group(groups[1]) == AlbumType.LIVE


async def test_browse_release_groups_sorts_newest_first_with_undated_last() -> None:
    """The most recent release group comes first; groups without a date close the list."""
    listing = [
        _release_group("rg-undated", first_release_date=None),
        _release_group("rg-1997", first_release_date="1997-05-21"),
        _release_group("rg-2016", first_release_date="2016-05-08"),
        _release_group("rg-2007", first_release_date="2007"),
        _release_group("rg-blank", first_release_date=""),
    ]
    provider, _ = _browsing_provider({"0": _browse_page(listing, 0, len(listing))})

    groups = await provider.browse_release_groups_by_artist(RADIOHEAD_MBID)

    assert [group.id for group in groups] == [
        "rg-2016",
        "rg-2007",
        "rg-1997",
        "rg-undated",
        "rg-blank",
    ]
    assert [group.first_release_year for group in groups] == [2016, 2007, 1997, None, None]


async def test_browse_release_groups_skips_a_malformed_entry() -> None:
    """One entry that cannot be parsed does not sink the rest of the discography."""
    listing = [_release_group("rg-1"), {"id": "rg-no-title"}, _release_group("rg-2")]
    provider, _ = _browsing_provider({"0": _browse_page(listing, 0, len(listing))})

    groups = await provider.browse_release_groups_by_artist(RADIOHEAD_MBID)

    assert [group.id for group in groups] == ["rg-1", "rg-2"]


async def test_browse_release_groups_is_empty_for_an_unknown_artist() -> None:
    """An unknown artist, or one without release groups, has no discography."""
    for response in (None, {"release-group-count": 0, "release-groups": []}):
        provider, api_client = _browsing_provider({"0": response})
        assert await provider.browse_release_groups_by_artist("unknown") == []
        assert api_client.get_browse_data.await_count == 1


# ---------------------------------------------------------------------------
# api client
# ---------------------------------------------------------------------------


class _RequestContext:
    """Minimal async context manager standing in for an aiohttp request."""

    def __init__(self, response: MagicMock) -> None:
        self._response = response

    async def __aenter__(self) -> MagicMock:
        return self._response

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _api_client(payload: Any) -> tuple[MusicBrainzAPIClient, MagicMock]:
    """Return an API client whose (mock) server answers every request with the payload."""
    mass = MagicMock()
    mass.version = "test"
    mass.cache.get_with_freshness = AsyncMock(return_value=(None, False, False))
    mass.cache.set = AsyncMock()
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value=payload)
    mass.http_session.get = MagicMock(return_value=_RequestContext(response))
    use_real_create_task(mass)
    return MusicBrainzAPIClient(mass), mass


async def test_api_client_caches_a_browse_for_a_week_and_a_lookup_for_a_month() -> None:
    """Both entry points request the same way and share the throttler, each with its own cache."""
    client, mass = _api_client({"release-groups": []})

    with patch.object(client.throttler, "acquire", wraps=client.throttler.acquire) as acquire:
        browsed = await client.get_browse_data("release-group", artist=RADIOHEAD_MBID, offset="0")
        looked_up = await client.get_data(f"artist/{RADIOHEAD_MBID}")
    await asyncio.sleep(0)

    assert browsed == looked_up == {"release-groups": []}
    assert acquire.call_count == 2
    assert [request.args[0] for request in mass.http_session.get.call_args_list] == [
        "https://musicbrainz-mirror.music-assistant.io/ws/2/release-group",
        f"https://musicbrainz-mirror.music-assistant.io/ws/2/artist/{RADIOHEAD_MBID}",
    ]
    assert mass.http_session.get.call_args_list[0].kwargs["params"] == {
        "artist": RADIOHEAD_MBID,
        "offset": "0",
        "fmt": "json",
    }
    assert {
        store.kwargs["key"]: store.kwargs["expiration"] for store in mass.cache.set.await_args_list
    } == {
        f"get_browse_data.release-group.artist{RADIOHEAD_MBID}.offset0": 86400 * 7,
        f"get_data.artist/{RADIOHEAD_MBID}": 86400 * 30,
    }


# ---------------------------------------------------------------------------
# get_work_details
# ---------------------------------------------------------------------------

WORK_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "works"
SYMPHONY_5_MBID = "d03bff61-26fc-301b-98ac-4d8e85771cbc"
BEETHOVEN_MBID = "1f9df192-a621-4f54-8850-2c5373b7eac9"
MOZART_MBID = "b972f589-fb0e-474e-b64a-803b0364fa75"


def _work_lookup(name: str) -> dict[str, Any]:
    """Return a real MusicBrainz work lookup response, trimmed to a few relations."""
    data: dict[str, Any] = json.loads((WORK_FIXTURES_DIR / f"{name}.json").read_text())
    return data


def _work(name: str) -> MusicBrainzWork:
    """Return the parsed MusicBrainz work of a work lookup fixture."""
    return MusicBrainzWork.from_raw(_work_lookup(name))


async def test_work_details_requests_aliases_and_artist_work_and_series_relations() -> None:
    """Request the names and relations a work is enriched from and parse the work off the lookup."""
    provider, get_data = _provider(_work_lookup("symphony_5"))

    work = await provider.get_work_details(SYMPHONY_5_MBID)

    get_data.assert_awaited_once_with(
        f"work/{SYMPHONY_5_MBID}?inc=aliases+artist-rels+work-rels+series-rels"
    )
    assert work is not None
    assert (work.id, work.title, work.type) == (
        SYMPHONY_5_MBID,
        "Symphony no. 5 in C minor, op. 67",
        "Symphony",
    )


async def test_work_details_is_none_when_not_found() -> None:
    """A work MusicBrainz does not know yields None."""
    provider, _ = _provider(None)

    assert await provider.get_work_details(SYMPHONY_5_MBID) is None


async def test_work_details_rejects_a_malformed_work() -> None:
    """A lookup without the work's title is invalid data."""
    provider, _ = _provider({"id": SYMPHONY_5_MBID})

    with pytest.raises(InvalidDataError):
        await provider.get_work_details(SYMPHONY_5_MBID)


def test_work_parses_attributes_and_relation_targets() -> None:
    """Parse the work attributes and the artist, work and series targets of its relations."""
    work = _work("symphony_5")

    assert work.languages == ["zxx"]
    assert [(attribute.type, attribute.value) for attribute in work.attributes] == [
        ("Key", "C minor")
    ]
    composer = work.relations[0]
    assert (composer.type, composer.direction, composer.begin, composer.end) == (
        "composer",
        "backward",
        "1804",
        "1808",
    )
    assert composer.artist is not None
    assert (composer.artist.id, composer.artist.sort_name, composer.artist.type) == (
        BEETHOVEN_MBID,
        "Beethoven, Ludwig van",
        "Person",
    )
    part = next(relation for relation in work.relations if relation.type == "parts")
    assert (part.direction, part.attributes, part.ordering_key) == ("forward", ["movement"], 1)
    assert part.work is not None
    assert part.work.title == "Symphony no. 5 in C minor, op. 67: I. Allegro con brio"
    assert part.work.type is None
    series = next(relation for relation in work.relations if relation.series)
    assert series.attribute_values == {"number": "op. 67"}
    assert series.url is None


def test_work_parent_relation_is_the_backward_parts_relation() -> None:
    """A movement points at its parent work and its position in it; a top-level work has none."""
    parent = _work("symphony_5_movement_1").parent_relation

    assert parent is not None
    assert parent.work is not None
    assert (parent.work.id, parent.work.type, parent.ordering_key) == (
        SYMPHONY_5_MBID,
        "Symphony",
        1,
    )
    assert _work("symphony_5").parent_relation is None


def test_work_movements_are_its_parts_without_its_title() -> None:
    """The parts of a work are its movements in order, named without the work's title."""
    assert _work("symphony_5").movements == [
        WorkMovement(
            number=1, name="I. Allegro con brio", mbid="b6f9ecc3-24d1-38ed-b8a0-091f7cd0c6b2"
        ),
        WorkMovement(
            number=2, name="II. Andante con moto", mbid="3bc742da-f67b-36c9-b31e-6ed09fdbdadd"
        ),
    ]
    assert _work("symphony_5_movement_1").movements == []


def test_work_movements_follow_the_ordering_key_else_their_position() -> None:
    """Parts come by ordering key, those without one numbered by their position."""
    data = _work_lookup("symphony_5")
    parts = [x for x in data["relations"] if x["type"] == "parts"]
    parts[0]["ordering-key"], parts[1]["ordering-key"] = 2, 1
    parts[0]["work"]["title"] = "Finale"
    parts.append({"type": "parts", "direction": "forward", "work": {"id": "x", "title": "Coda"}})
    data["relations"].append(parts[-1])

    assert [(x.number, x.name) for x in MusicBrainzWork.from_raw(data).movements] == [
        (1, "II. Andante con moto"),
        (2, "Finale"),
        (3, "Coda"),
    ]


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        # a part without a key after keys with a gap, as in Lucia di Lammermoor's Atto III
        ([1, 2, 4, None], [1, 2, 4, 5]),
        # parts sharing a key, as in Aida's Atto II
        ([5, 6, 6, 7], [5, 6, 7, 8]),
        ([None, None, None], [1, 2, 3]),
        ([2, 5, 9], [2, 5, 9]),
    ],
)
def test_work_movements_each_have_their_own_number(
    keys: list[int | None], expected: list[int]
) -> None:
    """Parts keep their ordering keys, unless missing or taken, then follow the part before."""
    parent = MusicBrainzWork(
        id="p",
        title="Opera: Atto III",
        relations=[
            MusicBrainzRelation(
                type="parts",
                direction="forward",
                work=MusicBrainzWork(id=f"part {index}", title=f"Opera: Atto III, no. {index}"),
                ordering_key=key,
            )
            for index, key in enumerate(keys)
        ],
    )

    assert [(x.number, x.mbid) for x in parent.movements] == [
        (number, f"part {index}") for index, number in enumerate(expected)
    ]


@pytest.mark.parametrize(
    ("title", "parent_title", "expected"),
    [
        ("Tosca: Atto III. “E lucevan le stelle”", "Tosca: Atto III", "“E lucevan le stelle”"),
        (
            "La bohème: Atto I, no. 6. “Che gelida manina!” (Rodolfo)",
            "La bohème: Atto I. In soffitta",
            "no. 6. “Che gelida manina!” (Rodolfo)",
        ),
        (
            "Spartacus: Adagio of Spartacus & Phrygia",
            "Spartacus, op. 82",
            "Adagio of Spartacus & Phrygia",
        ),
        (
            "La bohème: Atto I. “Già dell'Apocalisse appariscono i segni”",
            "La bohème: Atto I. In soffitta",
            "“Già dell'Apocalisse appariscono i segni”",
        ),
        (
            "Luisa Miller: Atto II. “Oh! fede negar potessi” (Rodolfo)",
            "Luisa Miller: Atto II. “L'intrigo”",
            "“Oh! fede negar potessi” (Rodolfo)",
        ),
        (
            "Don Carlo: Atto I, scena 1. N°2 “Il duolo della terra”",
            "Don Carlo: Atto I (II)",
            "scena 1. N°2 “Il duolo della terra”",
        ),
        ("Turandot: Atto I, no. 2", "Turandot: Atto II", "Turandot: Atto I, no. 2"),
        (
            "Ein Sommernachtstraum, op. 21 (op. 61 no. 1): Ouvertüre",
            "Ein Sommernachtstraum, op. 61",
            "Ein Sommernachtstraum, op. 21 (op. 61 no. 1): Ouvertüre",
        ),
        ("Main Theme", "Schindler's List", "Main Theme"),
    ],
)
def test_part_name_leaves_out_the_start_it_shares_with_its_work(
    title: str, parent_title: str, expected: str
) -> None:
    """A part's name leaves out the start its title shares with its work up to where they part."""
    part = MusicBrainzWork(id="w", title=title)
    parent = MusicBrainzWork(
        id="p",
        title=parent_title,
        relations=[MusicBrainzRelation(type="parts", direction="forward", work=part)],
    )

    movement = parent.part_movement("w")

    assert movement is not None
    assert movement.name == expected


@pytest.mark.parametrize("work_type", [None, "Opera"])
def test_part_movement_is_numbered_by_its_place_in_the_work_s_list(
    work_type: str | None,
) -> None:
    """A part is numbered by its place in its work's list, not by the number in its title."""
    keyed = MusicBrainzWork(id="keyed", title="Opera: Atto I, no. 6. Aria")
    unkeyed = MusicBrainzWork(id="unkeyed", title="Opera: Atto I, no. 9. Duet")
    parent = MusicBrainzWork(
        id="p",
        title="Opera: Atto I",
        type=work_type,
        relations=[
            MusicBrainzRelation(type="parts", direction="forward", work=unkeyed),
            MusicBrainzRelation(type="parts", direction="forward", work=keyed, ordering_key=1),
        ],
    )

    movements = [parent.part_movement(x) for x in ("keyed", "unkeyed")]

    assert [(x.number, x.name) if x else None for x in movements] == [
        (1, "no. 6. Aria"),
        (2, "no. 9. Duet"),
    ]


@pytest.mark.parametrize(("work_type", "part_type"), [("Opera", None), (None, "Operetta")])
def test_staged_work_lists_its_parts_as_movements(
    work_type: str | None, part_type: str | None
) -> None:
    """An opera lists its acts, and a work whose parts are operas lists those, as movements."""
    part = MusicBrainzWork(id="p", title="Work: Atto I", type=part_type)
    work = MusicBrainzWork(
        id="w",
        title="Work",
        type=work_type,
        relations=[MusicBrainzRelation(type="parts", direction="forward", work=part)],
    )

    assert [(x.number, x.name, x.mbid) for x in work.movements] == [(1, "Atto I", "p")]


def test_work_arrangement_relations_point_at_its_arrangements() -> None:
    """The forward arrangement relations of an original name its arrangements."""
    work = _work("symphony_5")

    assert [x.work.id for x in work.arrangement_relations if x.work] == [
        "f47df2ea-0671-4466-88f8-38a9d936989b"
    ]
    assert work.arrangement_of_relations == []


@pytest.mark.parametrize("relation_type", ["arrangement", "orchestration"])
def test_work_arrangement_of_relations_point_at_its_originals(relation_type: str) -> None:
    """The backward arrangement and orchestration relations of a work name its originals."""
    data = _work_lookup("symphony_5_movement_1")
    original = {"id": SYMPHONY_5_MBID, "title": "Symphony no. 5 in C minor, op. 67"}
    data["relations"].append({"type": relation_type, "direction": "backward", "work": original})
    work = MusicBrainzWork.from_raw(data)

    assert [x.work.id for x in work.arrangement_of_relations if x.work] == [SYMPHONY_5_MBID]
    # the arrangement of the movement itself stays a forward relation
    assert len(work.arrangement_relations) == 1


def test_work_composer_relations_carry_the_composer_and_years() -> None:
    """Only composer relations count, each with its composer and its years."""
    composers = _work("st_matthew_passion").composer_relations

    assert len(composers) == 1
    assert composers[0].artist is not None
    assert composers[0].artist.name == "Johann Sebastian Bach"
    assert (composers[0].begin_year, composers[0].end_year) == (None, 1727)


@pytest.mark.parametrize(
    ("fixture", "year"),
    [
        ("symphony_5", 1808),
        ("st_matthew_passion", 1727),
        ("gute_nacht", 1827),
    ],
)
def test_work_composition_year_prefers_the_year_composing_ended(fixture: str, year: int) -> None:
    """Date a work by when composing it ended, else by when it began, at any date precision."""
    assert _work(fixture).composition_year == year


def _composed(begin: str | None, end: str | None) -> MusicBrainzWork:
    """Return a work with a single composer relation of the given dates."""
    composer = {"id": BEETHOVEN_MBID, "name": "Ludwig van Beethoven", "sort-name": "Beethoven"}
    relation = {"type": "composer", "direction": "backward", "begin": begin, "end": end}
    return MusicBrainzWork.from_raw(
        {"id": "w", "title": "t", "relations": [{**relation, "artist": composer}]}
    )


def test_work_composer_relations_leave_out_additional_composers() -> None:
    """A composer who completed the work does not count as its composer or date it."""
    completer = {"id": "c", "name": "Franz Xaver Süßmayr", "sort-name": "Süßmayr, Franz Xaver"}
    work = MusicBrainzWork.from_raw(
        {
            "id": "w",
            "title": "t",
            "relations": [
                {
                    "type": "composer",
                    "direction": "backward",
                    "end": None,
                    "artist": {
                        "id": MOZART_MBID,
                        "name": "Wolfgang Amadeus Mozart",
                        "sort-name": "Mozart, Wolfgang Amadeus",
                    },
                },
                {
                    "type": "composer",
                    "direction": "backward",
                    "end": "1792",
                    "attributes": ["additional"],
                    "artist": completer,
                },
            ],
        }
    )

    assert [x.artist.name for x in work.composer_relations if x.artist] == [
        "Wolfgang Amadeus Mozart"
    ]
    assert work.composition_year is None


@pytest.mark.parametrize("name", ["[traditional]", "[anonymous]", "[unknown]"])
def test_work_composer_relations_leave_out_placeholder_artists(name: str) -> None:
    """A MusicBrainz placeholder credited as composer does not count as one or date the work."""
    placeholder = {"id": "p", "name": name, "sort-name": name}
    relation = {"type": "composer", "direction": "backward", "end": "1850"}
    work = MusicBrainzWork.from_raw(
        {"id": "w", "title": "t", "relations": [{**relation, "artist": placeholder}]}
    )

    assert work.composer_relations == []
    assert work.composition_year is None


def test_work_composition_year_falls_back_to_the_begin_year() -> None:
    """A composer relation with only a begin date dates the work by that year."""
    assert _composed("1790", None).composition_year == 1790


def test_work_composition_year_is_none_without_a_dated_composer() -> None:
    """A work whose composer relation carries no dates has no composition year."""
    assert _composed(None, None).composition_year is None


@pytest.mark.parametrize(
    ("languages", "language"),
    [
        (["zxx"], None),
        ([], None),
        (["deu"], "deu"),
        (["deu", "ita"], "deu"),
    ],
)
def test_work_language_ignores_no_lyrics(languages: list[str], language: str | None) -> None:
    """A work without lyrics has no language; with several languages the first one counts."""
    assert MusicBrainzWork(id="w", title="t", languages=languages).language == language


@pytest.mark.parametrize(
    ("fixture", "numbers"),
    [
        ("st_matthew_passion", ["BWV 244"]),
        ("symphony_5", ["op. 67"]),
        ("goldberg_variations", ["BWV 988"]),
        ("gute_nacht", []),
    ],
)
def test_work_catalog_numbers_come_from_catalog_series(fixture: str, numbers: list[str]) -> None:
    """Take the numbers of catalog series only, not of other series the work is part of."""
    assert _work(fixture).catalog_numbers == numbers


@pytest.mark.parametrize(
    ("mb_type", "work_type"),
    [
        ("Symphony", WorkType.SYMPHONY),
        ("Song-cycle", WorkType.SONG_CYCLE),
        ("Symphonic poem", WorkType.SYMPHONIC_POEM),
        ("Étude", WorkType.ETUDE),
        ("Incidental music", WorkType.INCIDENTAL_MUSIC),
        ("Song", WorkType.OTHER),
        ("Musical", WorkType.OTHER),
        (None, None),
    ],
)
def test_work_type_maps_musicbrainz_types(mb_type: str | None, work_type: WorkType | None) -> None:
    """Map MusicBrainz work types; an unknown type is other and a missing type stays unknown."""
    assert MusicBrainzWork(id="w", title="t", type=mb_type).work_type == work_type


@pytest.mark.parametrize(
    ("name", "expected"),
    [("symphony_5", "C minor"), ("goldberg_variations", "G major"), ("gute_nacht", None)],
)
def test_work_musical_key_comes_from_its_key_attribute(name: str, expected: str | None) -> None:
    """The key of a work is its Key attribute, None for a work without one."""
    assert _work(name).musical_key == expected


# ---------------------------------------------------------------------------
# get_artists_by_name
# ---------------------------------------------------------------------------


async def test_artists_by_name_keeps_only_those_carrying_the_name() -> None:
    """Return the artists whose name or an alias is the name, punctuation and accents aside."""
    # MusicBrainz writes the hyphen in Saint-Saëns as a Unicode hyphen
    provider, get_data = _provider(
        {
            "artists": [
                {
                    "id": "camille",
                    "name": "Camille Saint\u2010Saëns",
                    "sort-name": "Saint\u2010Saëns, Camille",
                    "aliases": [{"name": "Saint Saens", "sort-name": "Saint Saens"}],
                },
                {"id": "marie", "name": "Marie Saint-Saëns", "sort-name": "Saint-Saëns, Marie"},
                {"id": "broken", "sort-name": "no name"},
            ]
        }
    )

    artists = await provider.get_artists_by_name("Saint-Saëns")

    assert [x.id for x in artists] == ["camille"]
    assert get_data.await_args is not None
    assert get_data.await_args.kwargs["query"] == 'artist:"Saint\\-Saëns" OR alias:"Saint\\-Saëns"'


async def test_artists_by_name_is_empty_without_a_result() -> None:
    """Return no artists when the search answers nothing."""
    provider, _ = _provider(None)

    assert await provider.get_artists_by_name("Johann Strauss") == []
