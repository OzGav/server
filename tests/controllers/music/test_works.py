"""Tests for the library works and the work links of library tracks."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.enums import (
    ArtistRole,
    ExternalID,
    MediaType,
    SortDirection,
    SortField,
    WorkType,
)
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import (
    Artist,
    Credit,
    ItemMapping,
    MediaItem,
    MediaItemMetadata,
    ProviderMapping,
    Track,
    TrackSummary,
    UniqueList,
    Work,
    WorkMovement,
    WorkSummary,
)

from music_assistant.constants import DB_TABLE_WORK_ARRANGEMENTS, DB_TABLE_WORK_ARTISTS
from music_assistant.controllers.webserver.helpers.auth_middleware import current_user
from music_assistant.mass import MusicAssistant
from music_assistant.models.music_provider import MusicProvider

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
    item_credits: list[Credit] | None = None,
    work_ids: list[str] | None = None,
) -> Track:
    """Store a library track, optionally linked to a work and carrying MusicBrainz work ids."""
    artist = await _add_artist(mass, "Performer")
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id="0",
            provider="library",
            name=f"Track {_token()}",
            provider_mappings={_mapping()},
            artists=UniqueList([artist]),
            # a library work is linked by its mapping, a source's work is handed over in full
            work=ItemMapping.from_item(work)
            if isinstance(work, Work) and work.provider == "library"
            else work,
            movement_number=movement_number,
            movement_total=4 if movement_number else None,
            movement_name=movement_name,
            credits=item_credits or [],
            external_ids={(ExternalID.MB_WORK, work_id) for work_id in work_ids or []},
        )
    )


def _composer_ids(track: Track) -> list[str]:
    """Return the item ids of the composers credited on a track, in order."""
    return [x.artist.item_id for x in track.credits if x.role == ArtistRole.COMPOSER]


def _linked(works: Iterable[Work | ItemMapping]) -> list[tuple[str, list[str]]]:
    """Return the id and composer ids of each linked work."""
    linked = []
    for work in works:
        assert isinstance(work, Work)
        linked.append((work.item_id, [x.item_id for x in work.composers]))
    return linked


def _ids(items: Iterable[MediaItem | ItemMapping]) -> list[str]:
    """Return the item ids of the given items, in order."""
    return [x.item_id for x in items]


async def _library_work_ids(mass: MusicAssistant) -> set[str]:
    """Return the ids of every work stored in the library, listed or not."""
    rows = await mass.music.database.get_rows_from_query("SELECT item_id FROM works", limit=0)
    return {str(row["item_id"]) for row in rows}


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
    work = await _add_work(
        mass,
        f"Summary Work {token}",
        [composer],
        catalog_numbers=["BWV 1041"],
        work_type=WorkType.CONCERTO,
        composition_year=1730,
        parent_work=ItemMapping.from_item(parent),
        arrangement_of=UniqueList([ItemMapping.from_item(original)]),
    )

    # a part is listed through its parent, so its summary is fetched on its own
    [summary] = await mass.music.works.get_library_items_by_query(
        extra_query_parts=["works.item_id = :item_id"],
        extra_query_params={"item_id": int(work.item_id)},
        summary=True,
    )

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


@pytest.mark.parametrize("same_id", [True, False], ids=["same id, other title", "other id"])
async def test_works_match_on_musicbrainz_id(mass: MusicAssistant, same_id: bool) -> None:
    """
    A work with a known MusicBrainz id is the stored work, whatever its title.

    A work of the same title and composer with another MusicBrainz id is a different work.
    """
    composer = await _add_artist(mass, "Composer")
    name, mbid = f"Requiem {_token()}", _mbid()
    work = await _add_work(mass, name, [composer], mbid=mbid)

    if same_id:
        again = await _add_work(mass, f"Other Title {_token()}", mbid=mbid)
    else:
        again = await _add_work(mass, name, [composer], mbid=_mbid())

    assert (again.item_id == work.item_id) is same_id


async def test_catalog_numbers_differing_only_in_spelling_are_kept_once(
    mass: MusicAssistant,
) -> None:
    """
    Catalog numbers differing in case, dots or spacing count once, in their first spelling.

    This holds for a new work given a number several times and for an update of a stored work.
    """
    composer = await _add_artist(mass, "Composer")
    name = f"Serenade No. 13 {_token()}"
    mbid = _mbid()
    work = await mass.music.works.create_library_item(
        _work(name, [composer], mbid, catalog_numbers=["K.525", "K.525", "op. 67", "K.525"])
    )
    assert work.catalog_numbers == ["K.525", "op. 67"]

    again = await _add_work(
        mass, name, [composer], mbid, catalog_numbers=["K. 525", "Op. 67", "KV 525"]
    )

    assert again.catalog_numbers == ["K.525", "op. 67", "KV 525"]


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

    by_name = await mass.music.works.library_items(search=token, sort_field=SortField.NAME)
    by_name_desc = await mass.music.works.library_items(
        search=token, sort_field=SortField.NAME, sort_direction=SortDirection.DESC
    )
    page = await mass.music.works.library_items(
        search=token, sort_field=SortField.NAME, limit=1, offset=1
    )
    full = await mass.music.works.library_items(search=token, summary=False)

    assert [x.name for x in by_name] == names
    assert [x.name for x in by_name_desc] == names[::-1]
    assert [x.name for x in page] == names[1:2]
    assert {x.name for x in full} == set(names)
    assert all(isinstance(x, Work) and not isinstance(x, WorkSummary) for x in full)


@pytest.mark.parametrize(
    "sort_field", [SortField.SORT_NAME, SortField.TIMESTAMP_ADDED, SortField.RANDOM]
)
async def test_library_items_reject_sort_fields_works_do_not_offer(
    mass: MusicAssistant, sort_field: SortField
) -> None:
    """Sort fields works do not offer are rejected like on every other listing."""
    with pytest.raises(InvalidDataError):
        await mass.music.works.library_items(sort_field=sort_field)


async def test_library_items_sort_by_title_in_natural_order(mass: MusicAssistant) -> None:
    """Sorting by title keeps leading articles and orders numbers naturally."""
    token = _token()
    no_10 = await _add_work(mass, f"{token} Symphony No. 10")
    no_2 = await _add_work(mass, f"{token} Symphony No. 2")
    nozze = await _add_work(mass, f"{token} Le nozze di Figaro")

    by_name = await mass.music.works.library_items(search=token, sort_field=SortField.NAME)
    by_name_desc = await mass.music.works.library_items(
        search=token, sort_field=SortField.NAME, sort_direction=SortDirection.DESC
    )

    assert _ids(by_name) == _ids([nozze, no_2, no_10])
    assert _ids(by_name_desc) == _ids([no_10, no_2, nozze])


async def test_library_items_sort_by_composer_in_catalogue_order(mass: MusicAssistant) -> None:
    """
    Sorting by composer orders each composer's works by catalogue number, numbers naturally.

    Works are listed this way when no sort is requested.
    """
    token = _token()
    bach = await _add_artist(mass, f"Bach {token}")
    mozart = await _add_artist(mass, f"Mozart {token}")
    op_67 = await _add_work(mass, f"Symphony {token}", [bach], catalog_numbers=["Op. 67"])
    op_9 = await _add_work(mass, f"Quartet {token}", [bach], catalog_numbers=["Op. 9"])
    uncatalogued = await _add_work(mass, f"Aria {token}", [bach])
    k_525 = await _add_work(mass, f"Serenade {token}", [mozart], catalog_numbers=["K. 525"])
    composerless = await _add_work(mass, f"Chant {token}")

    by_composer = await mass.music.works.library_items(search=token, sort_field=SortField.COMPOSER)
    by_composer_desc = await mass.music.works.library_items(
        search=token, sort_field=SortField.COMPOSER, sort_direction=SortDirection.DESC
    )
    by_default = await mass.music.works.library_items(search=token)

    assert _ids(by_composer) == _ids([op_9, op_67, uncatalogued, k_525, composerless])
    assert _ids(by_default) == _ids(by_composer)
    assert _ids(by_composer_desc) == _ids([k_525, op_9, op_67, uncatalogued, composerless])


async def test_server_lists_works_newest_first(mass: MusicAssistant) -> None:
    """The server can list works newest first, a sort only it uses."""
    token = _token()
    older = await _add_work(mass, f"B {token}", date_added=datetime(2020, 1, 1, tzinfo=UTC))
    newer = await _add_work(mass, f"C {token}", date_added=datetime(2021, 1, 1, tzinfo=UTC))
    oldest = await _add_work(mass, f"A {token}", date_added=datetime(2019, 1, 1, tzinfo=UTC))

    newest_first = await mass.music.works.get_library_items_by_query(
        search=token, sort_field=SortField.TIMESTAMP_ADDED, sort_direction=SortDirection.DESC
    )

    assert _ids(newest_first) == _ids([newer, older, oldest])


async def test_library_items_sort_by_composition_year(mass: MusicAssistant) -> None:
    """Sorting by year composed lists the oldest first by default, works without a year last."""
    token = _token()
    early = await _add_work(mass, f"Early {token}", composition_year=1721)
    late = await _add_work(mass, f"Late {token}", composition_year=1808)
    undated = await _add_work(mass, f"Undated {token}")

    by_year = await mass.music.works.library_items(search=token, sort_field=SortField.YEAR)
    by_year_desc = await mass.music.works.library_items(
        search=token, sort_field=SortField.YEAR, sort_direction=SortDirection.DESC, summary=False
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


async def test_track_work_with_its_musicbrainz_id_is_added(mass: MusicAssistant) -> None:
    """A provider work carrying its MusicBrainz id is added with the track, shared by others."""
    composer = await _add_artist(mass, "Composer")
    name = f"Symphony No. 5 {_token()}, Op. 67"
    mbid = _mbid()

    first = await _add_track(mass, _work(name, [composer], mbid), movement_number=1)
    second = await _add_track(mass, _work(name, [composer], mbid), movement_number=2)

    assert first.work is not None
    assert second.work is not None
    assert first.work.item_id == second.work.item_id
    work = await mass.music.works.get_library_item(first.work.item_id)
    assert work.name == name
    assert _ids(work.composers) == [composer.item_id]
    assert _composer_ids(first) == [composer.item_id]


async def test_track_work_named_without_an_id_is_not_added(mass: MusicAssistant) -> None:
    """A work read from a pop track's title is not added and does not make the track classical."""
    before = await _library_work_ids(mass)

    track = await _add_track(mass, _work(f"Suite {_token()}"), movement_name="Judy Blue Eyes")

    assert track.work is None
    assert await _library_work_ids(mass) == before
    assert (track.classical_tag, track.is_classical) == (False, False)


async def test_track_work_named_without_an_id_is_the_work_of_the_track_s_own_id(
    mass: MusicAssistant,
) -> None:
    """A track carrying one MusicBrainz work id gets that work, whatever work its source names."""
    mbid = _mbid()
    name = f"Moonlight Sonata {_token()}"

    track = await _add_track(mass, _work(name), movement_name="First Movement", work_ids=[mbid])

    assert track.work is not None
    work = await mass.music.works.get_library_item(track.work.item_id)
    assert (work.mbid, work.name) == (mbid, name)
    assert not track.classical_tag


async def test_provider_work_matches_a_composer_the_provider_names_differently(
    mass: MusicAssistant,
) -> None:
    """Tracks share a provider work whose composer the provider spells unlike the library."""
    mapping = _mapping()
    composer = await mass.music.artists.add_item_to_library(
        Artist(item_id="0", provider="library", name="Schubert", provider_mappings={mapping})
    )
    provider_composer = Artist(
        item_id=mapping.item_id,
        provider=mapping.provider_instance,
        name="Franz Schubert",
        provider_mappings={mapping},
    )
    name = f"Trout Quintet {_token()}"
    mbid = _mbid()

    first = await _add_track(mass, _work(name, [provider_composer], mbid))
    second = await _add_track(mass, _work(name, [provider_composer], mbid))

    assert first.work is not None
    assert second.work is not None
    assert first.work.item_id == second.work.item_id
    work = await mass.music.works.get_library_item(first.work.item_id)
    assert _ids(work.composers) == [composer.item_id]


async def test_work_takes_a_new_name(mass: MusicAssistant) -> None:
    """A library work renamed is found and sorted by its new name and catalog numbers."""
    work = await _add_work(mass, "Planets Suite, Op 23", catalog_numbers=["Op 23"])

    await mass.music.works.rename_library_work(int(work.item_id), "The Planets, Op. 32", ["Op. 32"])

    stored = await mass.music.works.get_library_item(work.item_id)
    assert (stored.name, stored.sort_name) == ("The Planets, Op. 32", "planets, op. 32, the")
    assert stored.catalog_numbers == ["Op. 32"]
    found = await mass.music.works.library_items(search="The Planets, Op. 32")
    assert work.item_id in {x.item_id for x in found}


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
    """
    An overwrite with another work replaces the work.

    A merging update keeps the stored work and does not add the work it describes.
    """
    work = await _add_work(mass, f"Work {_token()}")
    other = await _add_work(mass, f"Other {_token()}")
    track = await _add_track(mass, work, movement_number=1)
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.work = _work(f"Provider work {_token()}")
    update.movement_number = 3
    before = await _library_work_ids(mass)

    await mass.music.tracks.update_item_in_library(track.item_id, update)
    merged = await mass.music.tracks.get_library_item(track.item_id)
    assert await _library_work_ids(mass) == before
    update.work = ItemMapping.from_item(other)
    await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)
    overwritten = await mass.music.tracks.get_library_item(track.item_id)

    assert merged.work is not None
    assert merged.work.item_id == work.item_id
    assert merged.movement_number == 1
    assert overwritten.work is not None
    assert overwritten.work.item_id == other.item_id
    assert overwritten.movement_number == 3


async def _add_act(mass: MusicAssistant) -> tuple[Work, str, str]:
    """Add an act listing two numbers by MusicBrainz id, return it with their ids."""
    first, second = _mbid(), _mbid()
    act = await _add_work(
        mass,
        f"Act {_token()}",
        mbid=_mbid(),
        metadata=MediaItemMetadata(
            movements=[
                WorkMovement(number=1, name="No. 1", mbid=first),
                WorkMovement(number=2, name="No. 2", mbid=second),
            ]
        ),
    )
    return act, first, second


async def _overwrite_with_tag_work(mass: MusicAssistant, track: Track, mbid: str) -> Track:
    """Overwrite a library track as its file names a work by MusicBrainz id, return it stored."""
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.work = _work(f"Tag work {_token()}", mbid=mbid)
    update.movement_number = 5
    update.movement_name = "Tag name"
    update.external_ids = set()
    await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)
    return await mass.music.tracks.get_library_item(track.item_id)


def _track_work_ids(track: Track) -> set[str]:
    """Return the MusicBrainz work ids a track carries."""
    return {value for id_type, value in track.external_ids if id_type == ExternalID.MB_WORK}


async def test_track_overwrite_naming_its_movement_keeps_its_place(mass: MusicAssistant) -> None:
    """An overwrite naming the movement a track performs keeps its work, number and name."""
    act, _first, second = await _add_act(mass)
    track = await _add_track(mass, act, movement_number=2, movement_name="No. 2")
    before = await _library_work_ids(mass)

    stored = await _overwrite_with_tag_work(mass, track, second.upper())

    assert stored.work is not None
    assert stored.work.item_id == act.item_id
    assert (stored.movement_number, stored.movement_name) == (2, "No. 2")
    assert await _library_work_ids(mass) == before
    assert {x.casefold() for x in _track_work_ids(stored)} == {second}


async def test_track_overwrite_naming_its_own_work_takes_the_movement_of_the_update(
    mass: MusicAssistant,
) -> None:
    """An overwrite naming the track's own work keeps it there with the update's movement."""
    act, _first, _second = await _add_act(mass)
    track = await _add_track(mass, act, movement_number=2, movement_name="No. 2")
    assert act.mbid is not None

    stored = await _overwrite_with_tag_work(mass, track, act.mbid)

    assert stored.work is not None
    assert stored.work.item_id == act.item_id
    assert (stored.movement_number, stored.movement_name) == (5, "Tag name")
    assert not _track_work_ids(stored)


async def test_track_overwrite_naming_its_stored_work_id_keeps_its_place(
    mass: MusicAssistant,
) -> None:
    """An overwrite naming a work id the track carries keeps it on a work listing no movements."""
    work = await _add_work(mass, f"Act {_token()}", mbid=_mbid())
    held = _mbid()
    track = await _add_track(mass, work, movement_number=2, work_ids=[held])
    before = await _library_work_ids(mass)

    stored = await _overwrite_with_tag_work(mass, track, held)

    assert stored.work is not None
    assert stored.work.item_id == work.item_id
    assert stored.movement_number == 2
    assert await _library_work_ids(mass) == before
    assert _track_work_ids(stored) == {held}


@pytest.mark.parametrize("sibling", [True, False])
async def test_track_overwrite_naming_another_work_moves_it(
    mass: MusicAssistant, sibling: bool
) -> None:
    """An overwrite naming another movement of the work, or an unknown work, moves the track."""
    act, first, _second = await _add_act(mass)
    track = await _add_track(mass, act, movement_number=2, movement_name="No. 2")
    mbid = first if sibling else _mbid()

    stored = await _overwrite_with_tag_work(mass, track, mbid)

    assert stored.work is not None
    assert stored.work.item_id != act.item_id
    assert (await mass.music.works.get_library_item(stored.work.item_id)).mbid == mbid
    assert (stored.movement_number, stored.movement_name) == (5, "Tag name")


@pytest.mark.parametrize("streaming", [True, False])
async def test_track_overwrite_keeps_musicbrainz_work_ids_only_from_streaming(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """A streaming overwrite keeps the stored work ids, a file overwrite takes its tags' ids."""
    stored_id, tag_id = _mbid(), _mbid()
    track = await _add_track(mass, work_ids=[stored_id])
    source = Mock(spec=MusicProvider, is_streaming_provider=streaming)
    monkeypatch.setattr(mass, "get_provider", Mock(return_value=source))
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping()}
    update.external_ids = (
        {(ExternalID.MB_RECORDING, _mbid())} if streaming else {(ExternalID.MB_WORK, tag_id)}
    )

    await mass.music.tracks.update_item_in_library(track.item_id, update, overwrite=True)

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _track_work_ids(stored) == ({stored_id} if streaming else {tag_id})


async def test_track_of_a_work_gets_its_composers(mass: MusicAssistant) -> None:
    """A track of a work credits the work's composers in order, unless it credits its own."""
    first = await _add_artist(mass, "First")
    second = await _add_artist(mass, "Second")
    own = await _add_artist(mass, "Own")
    work = await _add_work(mass, f"Work {_token()}", [first, second])

    bare = await _add_track(mass, work)
    credited = await _add_track(
        mass, work, item_credits=[Credit(artist=own, role=ArtistRole.COMPOSER)]
    )
    workless = await _add_track(mass)

    assert _composer_ids(bare) == [first.item_id, second.item_id]
    assert [x.position for x in bare.credits if x.role == ArtistRole.COMPOSER] == [0, 1]
    assert _composer_ids(credited) == [own.item_id]
    assert _composer_ids(workless) == []


async def test_reread_track_keeps_its_work_composers(mass: MusicAssistant) -> None:
    """A track read again from its only provider, which replaces its credits, keeps them."""
    composer = await _add_artist(mass, "Composer")
    conductor = await _add_artist(mass, "Conductor")
    work = await _add_work(mass, f"Work {_token()}", [composer])
    track = await _add_track(mass, work)
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping()}
    update.credits = [Credit(artist=conductor, role=ArtistRole.CONDUCTOR)]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)

    assert _composer_ids(stored) == [composer.item_id]
    assert [x.artist.item_id for x in stored.credits if x.role == ArtistRole.CONDUCTOR] == [
        conductor.item_id
    ]


async def test_new_work_composers_reach_its_tracks(mass: MusicAssistant) -> None:
    """Composers added to a work later are credited on its tracks and classified with them."""
    composer = await _add_artist(mass, "Composer")
    work = await _add_work(mass, f"Work {_token()}", work_type=WorkType.SYMPHONY)
    track = await _add_track(mass, work)
    await mass.music.works.update_item_in_library(work.item_id, _work(work.name, [composer]))

    await mass.music.tracks.credit_work_composers(int(work.item_id))

    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert _composer_ids(stored) == [composer.item_id]
    assert (await mass.music.artists.get_library_item(composer.item_id)).is_classical


async def test_work_with_composers_keeps_them_for_a_track_crediting_others(
    mass: MusicAssistant,
) -> None:
    """A work with composers keeps them when a track of it credits another composer."""
    composer = await _add_artist(mass, "Composer")
    other = await _add_artist(mass, "Other")
    work = await _add_work(mass, f"Work {_token()}", [composer])

    await _add_track(mass, work, item_credits=[Credit(artist=other, role=ArtistRole.COMPOSER)])

    stored = await mass.music.works.get_library_item(work.item_id)
    assert _ids(stored.composers) == [composer.item_id]


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

    remaining = await _library_work_ids(mass)
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


# --------------------------------------------------------------------------- #
#  merging works                                                               #
# --------------------------------------------------------------------------- #


async def test_merge_moves_everything_onto_the_oldest_work(mass: MusicAssistant) -> None:
    """A merge keeps the target's own fields, fills its empty ones and moves every link over."""
    first = await _add_artist(mass, "Composer")
    second = await _add_artist(mass, "Second Composer")
    lyricist = await _add_artist(mass, "Lyricist")
    mbid = _mbid()
    target = await _add_work(
        mass, f"Target {_token()}", [first], catalog_numbers=["Op. 1"], work_type=WorkType.SYMPHONY
    )
    source = await _add_work(
        mass,
        f"Source {_token()}",
        [second],
        mbid=mbid,
        catalog_numbers=["op.1", "K. 2"],
        composition_year=1800,
    )
    await mass.music.database.insert(
        DB_TABLE_WORK_ARTISTS,
        {
            "work_id": int(source.item_id),
            "artist_id": int(lyricist.item_id),
            "role": ArtistRole.LYRICIST.value,
            "position": 0,
        },
    )
    track = await _add_track(mass, source, movement_number=2, movement_name="Adagio")
    child = await _add_work(mass, f"Child {_token()}", parent_work=ItemMapping.from_item(source))
    arranged = await _add_work(
        mass, f"Arranged {_token()}", arrangement_of=UniqueList([ItemMapping.from_item(source)])
    )
    original = await _add_work(mass, f"Original {_token()}")
    await mass.music.works.add_arrangement(int(source.item_id), int(original.item_id))
    user = User(user_id=f"fan-{_token()}", username="fan", role=UserRole.USER)
    token = current_user.set(user)
    try:
        await mass.music.add_item_to_favorites(source)
        merged = await mass.music.works.merge_library_items(target.item_id, source.item_id)
        favorite = (await mass.music.works.get_library_item(target.item_id)).favorite
    finally:
        current_user.reset(token)

    assert source.item_id not in await _library_work_ids(mass)
    assert merged.name == target.name
    assert merged.mbid == mbid
    assert merged.composition_year == 1800
    assert merged.work_type == WorkType.SYMPHONY
    assert merged.catalog_numbers == ["Op. 1", "K. 2"]
    assert _ids(merged.composers) == [first.item_id, second.item_id]
    rows = await mass.music.database.get_rows(
        DB_TABLE_WORK_ARTISTS, {"work_id": int(target.item_id), "role": "lyricist"}
    )
    assert [row["artist_id"] for row in rows] == [int(lyricist.item_id)]
    assert _ids(merged.parts) == [child.item_id]
    assert _ids(merged.arrangements) == [arranged.item_id]
    assert _ids(merged.arrangement_of) == [original.item_id]
    assert favorite is True
    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert stored_track.work is not None
    assert stored_track.work.item_id == target.item_id
    assert (stored_track.movement_number, stored_track.movement_name) == (2, "Adagio")
    # the target's work type makes the moved track classical
    assert stored_track.is_classical


async def test_merge_credits_the_composers_on_moved_tracks(mass: MusicAssistant) -> None:
    """A moved track that credits no composer gets the composers of the merged work."""
    composer = await _add_artist(mass, "Composer")
    target = await _add_work(mass, f"Target {_token()}", [composer])
    source = await _add_work(mass, f"Source {_token()}")
    track = await _add_track(mass, source)

    await mass.music.works.merge_library_items(target.item_id, source.item_id)

    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert [x.artist.item_id for x in stored_track.credits if x.role == ArtistRole.COMPOSER] == [
        composer.item_id
    ]


async def test_merge_drops_links_between_the_two_works(mass: MusicAssistant) -> None:
    """A merged work that was a part or an arrangement of the target leaves no self link."""
    target = await _add_work(mass, f"Target {_token()}")
    source = await _add_work(
        mass,
        f"Source {_token()}",
        parent_work=ItemMapping.from_item(target),
        arrangement_of=UniqueList([ItemMapping.from_item(target)]),
    )

    merged = await mass.music.works.merge_library_items(target.item_id, source.item_id)

    assert merged.parent_work is None
    assert merged.arrangement_of == []
    assert merged.arrangements == []
    rows = await mass.music.database.get_rows_from_query(
        f"SELECT * FROM {DB_TABLE_WORK_ARRANGEMENTS} "
        "WHERE work_id = :work_id OR source_work_id = :work_id",
        {"work_id": int(target.item_id)},
    )
    assert rows == []


async def test_merge_of_different_musicbrainz_works_is_refused(mass: MusicAssistant) -> None:
    """Two works holding different MusicBrainz work ids are never merged."""
    target = await _add_work(mass, f"Target {_token()}", mbid=_mbid())
    source = await _add_work(mass, f"Source {_token()}", mbid=_mbid())

    with pytest.raises(InvalidDataError):
        await mass.music.works.merge_library_items(target.item_id, source.item_id)

    assert {target.item_id, source.item_id} <= await _library_work_ids(mass)


@pytest.mark.parametrize(
    ("with_tracks", "given", "survivor"),
    [
        # none has tracks, the newest is given
        ((), 2, 0),
        # the oldest has none, the two newer ones have one each
        ((1, 2), 0, 1),
    ],
)
async def test_merge_by_musicbrainz_id_keeps_the_oldest_with_tracks(
    mass: MusicAssistant, with_tracks: tuple[int, ...], given: int, survivor: int
) -> None:
    """
    Works sharing a MusicBrainz id merge into the oldest that has tracks of its own.

    Without tracks on any of them they merge into the oldest, whichever of them is given.
    """
    mbid = _mbid()
    # an add would match the stored work holding the id, so the duplicates are created as such
    works = [
        await mass.music.works.create_library_item(_work(f"{name} {_token()}", mbid=mbid))
        for name in ("Oldest", "Newer", "Newest")
    ]
    for index in with_tracks:
        await _add_track(mass, works[index])

    kept = await mass.music.works.merge_library_works_by_mbid(int(works[given].item_id), mbid)

    assert kept == int(works[survivor].item_id)
    remaining = await _library_work_ids(mass)
    assert {x.item_id for x in works} & remaining == {works[survivor].item_id}


# --------------------------------------------------------------------------- #
#  related works and parts                                                     #
# --------------------------------------------------------------------------- #


async def test_full_work_lists_its_related_works_with_their_composers(
    mass: MusicAssistant,
) -> None:
    """The full work carries its originals, arrangements and parts, each with its composers."""
    composer = await _add_artist(mass, "Mussorgsky")
    arranger = await _add_artist(mass, "Ravel")
    original = await _add_work(mass, f"Pictures {_token()}", [composer])
    arrangement = await _add_work(
        mass,
        f"Pictures orchestrated {_token()}",
        [arranger],
        arrangement_of=UniqueList([ItemMapping.from_item(original)]),
    )
    part = await _add_work(
        mass, f"Promenade {_token()}", [composer], parent_work=ItemMapping.from_item(original)
    )

    stored = await mass.music.get_item(
        MediaType.WORK, original.item_id, "library", allow_update_metadata=False
    )
    stored_arrangement = await mass.music.works.get_library_item(arrangement.item_id)

    assert isinstance(stored, Work)
    assert _linked(stored.arrangements) == [(arrangement.item_id, [arranger.item_id])]
    assert _linked(stored.parts) == [(part.item_id, [composer.item_id])]
    assert _linked(stored_arrangement.arrangement_of) == [(original.item_id, [composer.item_id])]
    assert all(isinstance(x, Work) for x in (*stored.arrangements, *stored.parts))


async def test_parts_are_listed_through_their_work(mass: MusicAssistant) -> None:
    """
    A part is no row of its own; its work is listed and found by a search for the part.

    A work whose parts have no tracks in the library is not listed.
    """
    token = _token()
    opera = await _add_work(mass, f"Opera {token}")
    aria = await _add_work(mass, f"Aria {token}", parent_work=ItemMapping.from_item(opera))
    await _add_track(mass, aria)
    lone = await _add_work(mass, f"Lone {token}")
    unheard = await _add_work(mass, f"Unheard opera {token}")
    await _add_work(mass, f"Unheard aria {token}", parent_work=ItemMapping.from_item(unheard))

    listed = await mass.music.works.library_items(search=token)
    by_part = await mass.music.works.library_items(search=f"Aria {token}")

    assert sorted(_ids(listed)) == sorted([opera.item_id, lone.item_id])
    assert _ids(by_part) == [opera.item_id]


async def test_streaming_update_drops_the_joined_composer_it_replaces(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A joined name replaced on a streaming update loses its artist, track and work credits."""
    beethoven = await _add_artist(mass, "Ludwig van Beethoven")
    joined = await _add_artist(mass, "Ludwig van Beethoven & Conducted By William Bowles")
    composers = AsyncMock(return_value=[])
    monkeypatch.setattr(mass.music.artists, "get_library_composers", composers)
    work = await _add_work(mass, f"Für Elise {_token()}", [joined])
    track = await _add_track(
        mass, work, item_credits=[Credit(artist=joined, role=ArtistRole.COMPOSER)]
    )
    performer = track.artists[0]
    track.artists = UniqueList([joined])
    await mass.music.tracks.update_item_in_library(track.item_id, track)
    composers.return_value = [beethoven]
    streaming = Mock(spec=MusicProvider, is_streaming_provider=True)
    monkeypatch.setattr(mass, "get_provider", Mock(return_value=streaming))
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping()}
    update.credits = [Credit(artist=joined, role=ArtistRole.COMPOSER)]

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)
    stored_work = await mass.music.works.get_library_item(work.item_id)

    assert _composer_ids(stored) == [beethoven.item_id]
    assert _ids(stored_work.composers) == []
    assert sorted(_ids(stored.artists)) == sorted([performer.item_id, beethoven.item_id])


async def test_streaming_update_drops_the_performing_credit_of_the_composer(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A performer that becomes the composer on a streaming update loses its stored credit."""
    vivaldi = await _add_artist(mass, "Antonio Vivaldi")
    composers = AsyncMock(return_value=[])
    monkeypatch.setattr(mass.music.artists, "get_library_composers", composers)
    work = await _add_work(mass, f"Le quattro stagioni {_token()}", work_type=WorkType.CONCERTO)
    track = await _add_track(
        mass, work, item_credits=[Credit(artist=vivaldi, role=ArtistRole.ORCHESTRA)]
    )
    composers.return_value = [vivaldi]
    streaming = Mock(spec=MusicProvider, is_streaming_provider=True)
    monkeypatch.setattr(mass, "get_provider", Mock(return_value=streaming))
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.provider_mappings = {_mapping()}
    update.credits = [Credit(artist=vivaldi, role=ArtistRole.ORCHESTRA)]
    update.metadata.performers = {"Dr, Composer", f"{vivaldi.name}, Orchestra"}

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)

    assert [(x.artist.item_id, x.role) for x in stored.credits] == [
        (vivaldi.item_id, ArtistRole.COMPOSER)
    ]
