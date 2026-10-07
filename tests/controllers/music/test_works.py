"""Tests for the library works and the work links of library tracks."""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.enums import ArtistRole, ExternalID, MediaType, WorkType
from music_assistant_models.media_items import (
    Artist,
    ItemMapping,
    MediaItem,
    ProviderMapping,
    Track,
    TrackSummary,
    UniqueList,
    Work,
    WorkSummary,
)

from music_assistant.constants import DB_TABLE_WORK_ARRANGEMENTS, DB_TABLE_WORK_ARTISTS
from music_assistant.controllers.webserver.helpers.auth_middleware import current_user
from music_assistant.mass import MusicAssistant

PROVIDER_INSTANCE = "works_instance"
# works are added the way a tag parser hands them over, from their source without mappings
WORK_SOURCE = "works_source"


@pytest.fixture(name="mass")
def mass_fixture(
    music_mass_module: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture."""
    # the database-only server runs no audio analysis and no cache database
    monkeypatch.setattr(
        music_mass_module,
        "streams",
        Mock(audio_analysis=Mock(delete_audio_analysis=AsyncMock())),
        raising=False,
    )
    monkeypatch.setattr(music_mass_module.cache, "delete", AsyncMock())
    return music_mass_module


def _mapping() -> ProviderMapping:
    """Create a provider mapping with a unique provider item id."""
    return ProviderMapping(
        item_id=uuid4().hex,
        provider_domain="works",
        provider_instance=PROVIDER_INSTANCE,
        in_library=True,
    )


def _token() -> str:
    """Return a word unique to the calling test, to keep its names apart from other tests."""
    return uuid4().hex[:10]


def _mbid() -> str:
    """Return a new MusicBrainz identifier."""
    return str(uuid4())


async def _add_artist(mass: MusicAssistant, name: str) -> Artist:
    """Store a library artist under a name unique to the calling test."""
    return await mass.music.artists.add_item_to_library(
        Artist(
            item_id="0",
            provider="library",
            name=f"{name} {_token()}",
            provider_mappings={_mapping()},
        )
    )


def _work(
    name: str,
    composers: list[Artist | ItemMapping] | None = None,
    mbid: str | None = None,
    **kwargs: object,
) -> Work:
    """Create a work as a source hands it over to be added to the library."""
    return Work(
        item_id=uuid4().hex,
        provider=WORK_SOURCE,
        name=name,
        provider_mappings=set(),
        composers=UniqueList(composers or []),
        external_ids={(ExternalID.MB_WORK, mbid)} if mbid else set(),
        **kwargs,  # type: ignore[arg-type]
    )


async def _add_work(
    mass: MusicAssistant,
    name: str,
    composers: list[Artist | ItemMapping] | None = None,
    mbid: str | None = None,
    **kwargs: object,
) -> Work:
    """Add a work to the library."""
    return await mass.music.works.add_item_to_library(_work(name, composers, mbid, **kwargs))


async def _add_track(
    mass: MusicAssistant,
    work: Work | ItemMapping | None = None,
    movement_number: int | None = None,
    movement_name: str | None = None,
) -> Track:
    """Store a library track, optionally linked to a work."""
    artist = await _add_artist(mass, "Performer")
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=f"Track {_token()}",
            provider_mappings={_mapping()},
            artists=UniqueList([artist]),
            work=ItemMapping.from_item(work) if isinstance(work, Work) else work,
            movement_number=movement_number,
            movement_total=4 if movement_number else None,
            movement_name=movement_name,
        )
    )


def _ids(items: Iterable[MediaItem | ItemMapping]) -> list[str]:
    """Return the item ids of the given items, in order."""
    return [x.item_id for x in items]


async def test_work_round_trip(mass: MusicAssistant) -> None:
    """A work is stored and returned with all its fields and links."""
    first = await _add_artist(mass, "Composer One")
    second = await _add_artist(mass, "Composer Two")
    parent = await _add_work(mass, f"Parent {_token()}", [first])
    original = await _add_work(mass, f"Original {_token()}", [first])
    original_mbid = _mbid()
    by_mbid = await _add_work(mass, f"By Mbid {_token()}", [first], mbid=original_mbid)
    mbid = _mbid()
    work = await _add_work(
        mass,
        f"Symphony No. 5 {_token()}",
        [second, first],
        mbid=mbid,
        version="Revised",
        catalog_numbers=["Op. 67", "K. 1"],
        work_type=WorkType.SYMPHONY,
        composition_year=1808,
        language="de",
        musical_key="C minor",
        parent_work=ItemMapping.from_item(parent),
        arrangement_of=UniqueList(
            [
                ItemMapping.from_item(original),
                ItemMapping(
                    media_type=MediaType.WORK,
                    item_id="elsewhere",
                    provider=WORK_SOURCE,
                    name="Known by its MusicBrainz id",
                    external_ids={(ExternalID.MB_WORK, original_mbid)},
                ),
                ItemMapping(
                    media_type=MediaType.WORK,
                    item_id="unknown",
                    provider=WORK_SOURCE,
                    name="Not in the library",
                ),
            ]
        ),
    )

    stored = await mass.music.works.get_library_item(work.item_id)

    assert stored.provider == "library"
    assert _ids(stored.composers) == [second.item_id, first.item_id]
    assert stored.composers[0].name == second.name
    assert stored.version == "Revised"
    assert stored.catalog_numbers == ["Op. 67", "K. 1"]
    assert stored.work_type == WorkType.SYMPHONY
    assert stored.composition_year == 1808
    assert stored.language == "de"
    assert stored.musical_key == "C minor"
    assert stored.parent_work is not None
    assert stored.parent_work.item_id == parent.item_id
    assert stored.parent_work.name == parent.name
    assert stored.parent_work.media_type == MediaType.WORK
    assert _ids(stored.arrangement_of) == [original.item_id, by_mbid.item_id]
    assert stored.get_external_id(ExternalID.MB_WORK) == mbid
    assert stored.uri == f"library://work/{work.item_id}"


async def test_work_summary(mass: MusicAssistant) -> None:
    """A work summary carries the slim composer, parent and arrangement links."""
    composer = await _add_artist(mass, "Summary Composer")
    parent = await _add_work(mass, f"Summary Parent {_token()}", [composer])
    original = await _add_work(mass, f"Summary Original {_token()}", [composer])
    token = _token()
    await _add_work(
        mass,
        f"Summary Work {token}",
        [composer],
        catalog_numbers=["BWV 1041"],
        work_type=WorkType.CONCERTO,
        composition_year=1730,
        parent_work=ItemMapping.from_item(parent),
        arrangement_of=UniqueList([ItemMapping.from_item(original)]),
    )

    [summary] = await mass.music.works.library_items(search=token)

    assert isinstance(summary, WorkSummary)
    assert _ids(summary.composers) == [composer.item_id]
    assert summary.catalog_numbers == ["BWV 1041"]
    assert summary.work_type == WorkType.CONCERTO
    assert summary.composition_year == 1730
    assert summary.parent_work is not None
    assert summary.parent_work.item_id == parent.item_id
    assert _ids(summary.arrangement_of) == [original.item_id]


async def test_parent_name_never_goes_stale(mass: MusicAssistant) -> None:
    """The parent work link shows the current name of the parent work."""
    parent = await _add_work(mass, f"Old Name {_token()}")
    child = await _add_work(mass, f"Child {_token()}", parent_work=ItemMapping.from_item(parent))
    parent.name = f"New Name {_token()}"
    await mass.music.works.update_item_in_library(parent.item_id, parent, overwrite=True)

    stored = await mass.music.works.get_library_item(child.item_id)

    assert stored.parent_work is not None
    assert stored.parent_work.name == parent.name


async def test_provider_composer_is_added_to_the_library(mass: MusicAssistant) -> None:
    """A composer that is not in the library yet is added along with the work."""
    composer = Artist(
        item_id=uuid4().hex,
        provider=PROVIDER_INSTANCE,
        name=f"New Composer {_token()}",
        provider_mappings={_mapping()},
    )

    work = await _add_work(mass, f"Work {_token()}", [composer])

    [stored_composer] = work.composers
    assert stored_composer.provider == "library"
    library_artist = await mass.music.artists.get_library_item(stored_composer.item_id)
    assert library_artist.name == composer.name


async def test_match_on_musicbrainz_id(mass: MusicAssistant) -> None:
    """A work with a known MusicBrainz id is the stored work, whatever its title."""
    composer = await _add_artist(mass, "Composer")
    mbid = _mbid()
    work = await _add_work(mass, f"Title {_token()}", [composer], mbid=mbid)

    again = await _add_work(mass, f"Other Title {_token()}", mbid=mbid)

    assert again.item_id == work.item_id


async def test_different_musicbrainz_ids_stay_apart(mass: MusicAssistant) -> None:
    """Works with the same title and composer but other MusicBrainz ids are different works."""
    composer = await _add_artist(mass, "Composer")
    name = f"Requiem {_token()}"
    work = await _add_work(mass, name, [composer], mbid=_mbid())

    other = await _add_work(mass, name, [composer], mbid=_mbid())

    assert other.item_id != work.item_id


async def test_match_on_composer_and_title(mass: MusicAssistant) -> None:
    """A work matches on composer and title, ignoring case and surrounding whitespace."""
    composer = await _add_artist(mass, "Composer")
    name = f"Piano Sonata No. 14 {_token()}"
    work = await _add_work(mass, name, [composer], catalog_numbers=["Op. 27 No. 2"])

    again = await _add_work(mass, f"  {name.upper()} ", [composer], catalog_numbers=["Op. 27"])

    assert again.item_id == work.item_id
    # the stored name keeps the spelling it was added with
    assert again.name == name
    assert again.catalog_numbers == ["Op. 27 No. 2", "Op. 27"]


async def test_title_match_needs_the_same_composer(mass: MusicAssistant) -> None:
    """Works with the same title by different composers stay apart."""
    first = await _add_artist(mass, "First Composer")
    second = await _add_artist(mass, "Second Composer")
    name = f"Symphony No. 5 {_token()}"
    work = await _add_work(mass, name, [first])

    other = await _add_work(mass, name, [second])

    assert other.item_id != work.item_id


async def test_title_match_ignores_punctuation_and_accents(mass: MusicAssistant) -> None:
    """Titles match like artist and album names, ignoring punctuation and accents."""
    composer = await _add_artist(mass, "Composer")
    token = _token()
    work = await _add_work(mass, f"Symphony No. 5 {token}", [composer])
    mere = await _add_work(mass, f"Ma mère l'Oye {token}", [composer])

    other = await _add_work(mass, f"Symphony No 5 {token}", [composer])
    mere_again = await _add_work(mass, f"Ma mere l'Oye {token}", [composer])

    assert other.item_id == work.item_id
    assert mere_again.item_id == mere.item_id


async def test_composerless_work_matches_composerless_work(mass: MusicAssistant) -> None:
    """A work without a composer matches a stored work without a composer by title."""
    name = f"Greensleeves {_token()}"
    work = await _add_work(mass, name)

    again = await _add_work(mass, name.lower())

    assert again.item_id == work.item_id


async def test_composerless_work_never_matches_a_composed_work(mass: MusicAssistant) -> None:
    """Works with and without a composer never match each other on their title."""
    composer = await _add_artist(mass, "Composer")
    name = f"Requiem {_token()}"
    composed = await _add_work(mass, name, [composer])

    composerless = await _add_work(mass, name)
    composed_again = await _add_work(mass, name, [composer])

    assert composerless.item_id != composed.item_id
    assert composed_again.item_id == composed.item_id


async def test_update_with_overwrite_keeps_composers_without_new_ones(
    mass: MusicAssistant,
) -> None:
    """An overwrite without composers, links or field values keeps the stored ones."""
    composer = await _add_artist(mass, "Composer")
    parent = await _add_work(mass, f"Parent {_token()}")
    original = await _add_work(mass, f"Original {_token()}")
    work = await _add_work(
        mass,
        f"Work {_token()}",
        [composer],
        catalog_numbers=["Op. 1"],
        musical_key="D major",
        parent_work=ItemMapping.from_item(parent),
        arrangement_of=UniqueList([ItemMapping.from_item(original)]),
    )
    update = _work(f"Renamed {_token()}")

    await mass.music.works.update_item_in_library(work.item_id, update, overwrite=True)

    stored = await mass.music.works.get_library_item(work.item_id)
    assert stored.name == update.name
    assert stored.musical_key == "D major"
    assert _ids(stored.composers) == [composer.item_id]
    assert stored.catalog_numbers == ["Op. 1"]
    assert stored.parent_work is not None
    assert stored.parent_work.item_id == parent.item_id
    assert _ids(stored.arrangement_of) == [original.item_id]


async def test_update_with_overwrite_replaces_composers(mass: MusicAssistant) -> None:
    """An overwrite with composers replaces the stored composers, in the new order."""
    first = await _add_artist(mass, "First")
    second = await _add_artist(mass, "Second")
    third = await _add_artist(mass, "Third")
    work = await _add_work(mass, f"Work {_token()}", [first, second])

    await mass.music.works.update_item_in_library(
        work.item_id, _work(work.name, [third, first]), overwrite=True
    )

    stored = await mass.music.works.get_library_item(work.item_id)
    assert _ids(stored.composers) == [third.item_id, first.item_id]


async def test_update_without_overwrite_adds_composers(mass: MusicAssistant) -> None:
    """A merging update keeps the stored composers and fields and adds new composers after them."""
    first = await _add_artist(mass, "First")
    second = await _add_artist(mass, "Second")
    work = await _add_work(mass, f"Work {_token()}", [first], composition_year=1801)

    await mass.music.works.update_item_in_library(
        work.item_id, _work(work.name, [second], composition_year=1900, language="it")
    )

    stored = await mass.music.works.get_library_item(work.item_id)
    assert _ids(stored.composers) == [first.item_id, second.item_id]
    assert stored.composition_year == 1801
    assert stored.language == "it"


async def test_library_items_paging_and_sorting(mass: MusicAssistant) -> None:
    """Works are listed without provider mappings, paged and sorted."""
    token = _token()
    names = [f"{letter} {token}" for letter in ("Alpha", "Bravo", "Charlie")]
    for name in reversed(names):
        await _add_work(mass, name)

    by_name = await mass.music.works.library_items(search=token, order_by="name")
    by_name_desc = await mass.music.works.library_items(search=token, order_by="name_desc")
    page = await mass.music.works.library_items(search=token, order_by="name", limit=1, offset=1)
    full = await mass.music.works.library_items(search=token, summary=False)

    assert [x.name for x in by_name] == names
    assert [x.name for x in by_name_desc] == names[::-1]
    assert [x.name for x in page] == names[1:2]
    assert {x.name for x in full} == set(names)
    assert all(isinstance(x, Work) and not isinstance(x, WorkSummary) for x in full)


@pytest.mark.parametrize(
    "order_by",
    [
        "play_count",
        "last_played_desc",
        "duration",
        "year",
        "random_play_count",
        "random",
        "timestamp_added",
        "favorite_timestamp",
        "sort_name",
    ],
)
async def test_library_items_other_sort_keys_fall_back(mass: MusicAssistant, order_by: str) -> None:
    """Sort keys works do not offer still list the works."""
    token = _token()
    work = await _add_work(mass, f"Sorted {token}")

    for summary in (True, False):
        items = await mass.music.works.library_items(
            search=token, order_by=order_by, summary=summary
        )
        assert _ids(items) == [work.item_id]


async def test_library_items_sort_by_title_in_natural_order(mass: MusicAssistant) -> None:
    """Sorting by title keeps leading articles and orders numbers naturally."""
    token = _token()
    no_10 = await _add_work(mass, f"{token} Symphony No. 10")
    no_2 = await _add_work(mass, f"{token} Symphony No. 2")
    nozze = await _add_work(mass, f"{token} Le nozze di Figaro")

    by_name = await mass.music.works.library_items(search=token)
    by_name_desc = await mass.music.works.library_items(search=token, order_by="name_desc")

    assert _ids(by_name) == _ids([nozze, no_2, no_10])
    assert _ids(by_name_desc) == _ids([no_10, no_2, nozze])


async def test_library_items_sort_by_composer_in_catalogue_order(mass: MusicAssistant) -> None:
    """Sorting by composer orders each composer's works by catalogue number, numbers naturally."""
    token = _token()
    bach = await _add_artist(mass, f"Bach {token}")
    mozart = await _add_artist(mass, f"Mozart {token}")
    op_67 = await _add_work(mass, f"Symphony {token}", [bach], catalog_numbers=["Op. 67"])
    op_9 = await _add_work(mass, f"Quartet {token}", [bach], catalog_numbers=["Op. 9"])
    uncatalogued = await _add_work(mass, f"Aria {token}", [bach])
    k_525 = await _add_work(mass, f"Serenade {token}", [mozart], catalog_numbers=["K. 525"])
    composerless = await _add_work(mass, f"Chant {token}")

    by_composer = await mass.music.works.library_items(search=token, order_by="composer")
    by_composer_desc = await mass.music.works.library_items(search=token, order_by="composer_desc")

    assert _ids(by_composer) == _ids([op_9, op_67, uncatalogued, k_525, composerless])
    assert _ids(by_composer_desc) == _ids([k_525, op_9, op_67, uncatalogued, composerless])


async def test_library_items_sort_by_composition_year(mass: MusicAssistant) -> None:
    """Sorting by year composed lists works without a year last, in both directions."""
    token = _token()
    early = await _add_work(mass, f"Early {token}", composition_year=1721)
    late = await _add_work(mass, f"Late {token}", composition_year=1808)
    undated = await _add_work(mass, f"Undated {token}")

    by_year = await mass.music.works.library_items(search=token, order_by="composition_year")
    by_year_desc = await mass.music.works.library_items(
        search=token, order_by="composition_year_desc", summary=False
    )

    assert _ids(by_year) == _ids([early, late, undated])
    assert _ids(by_year_desc) == _ids([late, early, undated])


async def test_library_count_and_played_only(mass: MusicAssistant) -> None:
    """Works are counted without provider mappings and are never listed as played."""
    before = await mass.music.works.library_count()
    token = _token()
    await _add_work(mass, f"Counted {token}")

    assert await mass.music.works.library_count() == before + 1
    assert await mass.music.works.library_items(search=token, played_only=True) == []


async def test_library_search(mass: MusicAssistant) -> None:
    """The library search of works finds them by (part of) their title."""
    token = _token()
    work = await _add_work(mass, f"Goldberg Variations {token}")

    found = await mass.music.works.search(f"variations {token}", "library")

    assert _ids(found) == [work.item_id]


async def test_track_work_round_trip(mass: MusicAssistant) -> None:
    """A track carries its work and movement, in full items and summaries."""
    work = await _add_work(mass, f"Work {_token()}")
    track = await _add_track(mass, work, movement_number=2, movement_name="II. Andante")

    stored = await mass.music.tracks.get_library_item(track.item_id)
    [summary] = [
        x
        for x in await mass.music.tracks.library_items(search=track.name)
        if x.item_id == track.item_id
    ]

    for item in (stored, summary):
        assert item.work is not None
        assert item.work.item_id == work.item_id
        assert item.work.name == work.name
        assert item.work.media_type == MediaType.WORK
        assert item.movement_number == 2
        assert item.movement_total == 4
        assert item.movement_name == "II. Andante"
    assert isinstance(summary, TrackSummary)


async def test_track_work_resolved_by_musicbrainz_id(mass: MusicAssistant) -> None:
    """A track work from a source is linked through its MusicBrainz id, otherwise ignored."""
    mbid = _mbid()
    work = await _add_work(mass, f"Work {_token()}", mbid=mbid)

    linked = await _add_track(
        mass,
        ItemMapping(
            media_type=MediaType.WORK,
            item_id="source-work",
            provider=WORK_SOURCE,
            name=work.name,
            external_ids={(ExternalID.MB_WORK, mbid)},
        ),
    )
    unknown = await _add_track(
        mass,
        ItemMapping(
            media_type=MediaType.WORK,
            item_id="source-work",
            provider=WORK_SOURCE,
            name="Unknown work",
            external_ids={(ExternalID.MB_WORK, _mbid())},
        ),
    )

    assert linked.work is not None
    assert linked.work.item_id == work.item_id
    assert unknown.work is None


async def test_track_overwrite_without_work_keeps_work(mass: MusicAssistant) -> None:
    """An overwrite without a work or movement keeps the stored ones."""
    work = await _add_work(mass, f"Work {_token()}")
    track = await _add_track(mass, work, movement_number=1, movement_name="I. Allegro")
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.work = None
    update.movement_number = None
    update.movement_total = None
    update.movement_name = None

    await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert stored.work is not None
    assert stored.work.item_id == work.item_id
    assert stored.movement_number == 1
    assert stored.movement_name == "I. Allegro"


async def test_track_overwrite_replaces_work(mass: MusicAssistant) -> None:
    """An overwrite with another work replaces the work, a merging update keeps it."""
    work = await _add_work(mass, f"Work {_token()}")
    other = await _add_work(mass, f"Other {_token()}")
    track = await _add_track(mass, work, movement_number=1)
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.work = ItemMapping.from_item(other)
    update.movement_number = 3

    await mass.music.tracks.update_item_in_library(track.item_id, update)
    merged = await mass.music.tracks.get_library_item(track.item_id)
    await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)
    overwritten = await mass.music.tracks.get_library_item(track.item_id)

    assert merged.work is not None
    assert merged.work.item_id == work.item_id
    assert merged.movement_number == 1
    assert overwritten.work is not None
    assert overwritten.work.item_id == other.item_id
    assert overwritten.movement_number == 3


async def test_removing_a_work_unlinks_it(mass: MusicAssistant) -> None:
    """Removing a work keeps its tracks and other works, without their links to it."""
    composer = await _add_artist(mass, "Composer")
    original = await _add_work(mass, f"Original {_token()}", [composer])
    work = await _add_work(
        mass,
        f"Work {_token()}",
        [composer],
        arrangement_of=UniqueList([ItemMapping.from_item(original)]),
    )
    child = await _add_work(mass, f"Child {_token()}", parent_work=ItemMapping.from_item(work))
    arrangement = await _add_work(
        mass,
        f"Arrangement {_token()}",
        arrangement_of=UniqueList([ItemMapping.from_item(work)]),
    )
    track = await _add_track(mass, work, movement_number=2, movement_name="II. Adagio")

    await mass.music.works.remove_item_from_library(work.item_id)

    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert stored_track.work is None
    assert stored_track.movement_number == 2
    assert stored_track.movement_name == "II. Adagio"
    assert (await mass.music.works.get_library_item(child.item_id)).parent_work is None
    assert not (await mass.music.works.get_library_item(arrangement.item_id)).arrangement_of
    assert await mass.music.works.get_library_item(original.item_id)
    for table, column in (
        (DB_TABLE_WORK_ARTISTS, "work_id"),
        (DB_TABLE_WORK_ARRANGEMENTS, "work_id"),
        (DB_TABLE_WORK_ARRANGEMENTS, "source_work_id"),
    ):
        assert not await mass.music.database.get_rows(table, {column: int(work.item_id)})


async def test_removing_a_composer_keeps_their_works(mass: MusicAssistant) -> None:
    """Removing an artist drops them as composer and keeps the work."""
    composer = await _add_artist(mass, "Composer")
    other = await _add_artist(mass, "Other Composer")
    work = await _add_work(mass, f"Work {_token()}", [composer, other])

    await mass.music.artists.remove_item_from_library(composer.item_id)

    stored = await mass.music.works.get_library_item(work.item_id)
    assert _ids(stored.composers) == [other.item_id]
    assert not await mass.music.database.get_rows(
        DB_TABLE_WORK_ARTISTS, {"artist_id": int(composer.item_id)}
    )


async def test_merging_artists_keeps_work_composers(mass: MusicAssistant) -> None:
    """Merging a composer into another artist moves their works to that artist."""
    composer = await _add_artist(mass, "Composer")
    duplicate = await _add_artist(mass, "Duplicate")
    work = await _add_work(mass, f"Work {_token()}", [duplicate])
    shared = await _add_work(mass, f"Shared {_token()}", [composer, duplicate])

    await mass.music.artists.merge_library_items(composer.item_id, duplicate.item_id)

    stored = await mass.music.works.get_library_item(work.item_id)
    stored_shared = await mass.music.works.get_library_item(shared.item_id)
    assert _ids(stored.composers) == [composer.item_id]
    assert _ids(stored_shared.composers) == [composer.item_id]
    rows = await mass.music.database.get_rows(DB_TABLE_WORK_ARTISTS, {"work_id": int(work.item_id)})
    assert [row["role"] for row in rows] == [ArtistRole.COMPOSER.value]


async def test_cleanup_removes_works_without_tracks(mass: MusicAssistant) -> None:
    """The library clean-up removes works without tracks unless another work refers to them."""
    played = await _add_work(mass, f"With Track {_token()}")
    await _add_track(mass, played)
    empty = await _add_work(mass, f"Empty {_token()}")
    parent = await _add_work(mass, f"Parent {_token()}")
    child = await _add_work(mass, f"Child {_token()}", parent_work=ItemMapping.from_item(parent))
    await _add_track(mass, child)
    source = await _add_work(mass, f"Source {_token()}")
    arrangement = await _add_work(
        mass,
        f"Arrangement {_token()}",
        arrangement_of=UniqueList([ItemMapping.from_item(source)]),
    )
    await _add_track(mass, arrangement)
    lone_source = await _add_work(mass, f"Lone Source {_token()}")
    await _add_work(
        mass,
        f"Lone Arrangement {_token()}",
        arrangement_of=UniqueList([ItemMapping.from_item(lone_source)]),
    )

    await mass.music._cleanup_database()

    remaining = {x.item_id for x in await mass.music.works.library_items(limit=100000)}
    assert {played.item_id, parent.item_id, child.item_id} <= remaining
    assert {source.item_id, arrangement.item_id} <= remaining
    assert empty.item_id not in remaining
    # the lone source is orphaned once its trackless arrangement is removed
    assert lone_source.item_id not in remaining


async def test_favorite_work(mass: MusicAssistant) -> None:
    """A work is liked and unliked through the per-user favorites."""
    work = await _add_work(mass, f"Favorite {_token()}")
    user = User(user_id=f"fan-{_token()}", username="fan", role=UserRole.USER)

    token = current_user.set(user)
    try:
        await mass.music.add_item_to_favorites(work)
        liked = await mass.music.get_item(
            MediaType.WORK, work.item_id, "library", allow_update_metadata=False
        )
        assert isinstance(liked, Work)
        favorites = await mass.music.works.library_items(favorite=True)
        favorite_count = await mass.music.works.library_count(favorite_only=True)
        await mass.music.remove_item_from_favorites(MediaType.WORK, work.item_id)
        unliked = await mass.music.works.get_library_item(work.item_id)
    finally:
        current_user.reset(token)

    assert liked.favorite is True
    assert _ids(favorites) == [work.item_id]
    assert favorite_count == 1
    assert unliked.favorite is None
