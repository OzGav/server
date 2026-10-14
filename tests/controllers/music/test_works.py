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
    ProviderMapping,
    Track,
    TrackSummary,
    UniqueList,
    Work,
    WorkSummary,
)

from music_assistant.constants import DB_TABLE_WORK_ARRANGEMENTS, DB_TABLE_WORK_ARTISTS
from music_assistant.controllers.webserver.helpers.auth_middleware import current_user
from music_assistant.helpers.work_titles import parse_catalog_numbers
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
            # a library work is linked by its mapping, a source's work is handed over in full
            work=ItemMapping.from_item(work)
            if isinstance(work, Work) and work.provider == "library"
            else work,
            movement_number=movement_number,
            movement_total=4 if movement_number else None,
            movement_name=movement_name,
            credits=item_credits or [],
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


async def test_catalog_numbers_differing_only_in_spelling_are_kept_once(
    mass: MusicAssistant,
) -> None:
    """Catalog numbers differing in case, dots or spacing count once, in their first spelling."""
    composer = await _add_artist(mass, "Composer")
    name = f"Serenade No. 13 {_token()}"
    await _add_work(mass, name, [composer], catalog_numbers=["K.525", "op. 67"])

    again = await _add_work(mass, name, [composer], catalog_numbers=["K. 525", "Op. 67", "KV 525"])

    assert again.catalog_numbers == ["K.525", "op. 67", "KV 525"]


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


async def test_numbered_piece_of_a_library_opus_is_a_movement_of_it(mass: MusicAssistant) -> None:
    """A track of a numbered piece of an opus the library holds is a movement of that work."""
    mozart = await _add_artist(mass, "Wolfgang Amadeus Mozart")
    sonata = await _add_work(
        mass, "Piano Sonata No. 11 in A Major, K. 331", [mozart], catalog_numbers=["K. 331"]
    )
    name = "Piano Sonata No. 11 in A major, K 331, No. 3 Rondo Alla Turca"

    track = await _add_track(mass, _work(name, [mozart], catalog_numbers=["K 331, No. 3"]))

    assert track.work is not None
    assert track.work.item_id == sonata.item_id
    assert (track.movement_number, track.movement_name) == (3, "Rondo Alla Turca")


@pytest.mark.parametrize(
    ("stored", "composer_name"),
    [("Nocturne, Op. 15, No. 2", "Frédéric Chopin"), ("Nocturnes, Op. 15", "Other Composer")],
    ids=["no_bare_opus_work", "other_composer"],
)
async def test_numbered_piece_without_its_library_opus_is_a_work_of_its_own(
    mass: MusicAssistant, stored: str, composer_name: str
) -> None:
    """A numbered piece is a work of its own without a library work of its bare opus number."""
    chopin = await _add_artist(mass, "Frédéric Chopin")
    composer = (
        chopin if composer_name == "Frédéric Chopin" else await _add_artist(mass, composer_name)
    )
    work = await _add_work(mass, stored, [composer], catalog_numbers=parse_catalog_numbers(stored))
    name = "Nocturne, Op. 15, No. 7"

    track = await _add_track(mass, _work(name, [chopin], catalog_numbers=["Op. 15, No. 7"]))

    assert track.work is not None
    assert track.work.item_id != work.item_id
    assert track.work.name == name
    assert track.movement_number is None


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


async def test_library_items_default_sort_is_composer(mass: MusicAssistant) -> None:
    """Without a requested sort works are listed by composer, then catalogue number."""
    token = _token()
    bach = await _add_artist(mass, f"Bach {token}")
    op_9 = await _add_work(mass, f"Zeta {token}", [bach], catalog_numbers=["Op. 9"])
    composerless = await _add_work(mass, f"Alpha {token}")

    items = await mass.music.works.library_items(search=token)

    assert _ids(items) == _ids([op_9, composerless])


@pytest.mark.parametrize(
    "sort_field",
    [
        SortField.SORT_NAME,
        SortField.TIMESTAMP_ADDED,
        SortField.LAST_PLAYED,
        SortField.PLAY_COUNT,
        SortField.DURATION,
        SortField.FAVORITE_TIMESTAMP,
        SortField.RANDOM,
        SortField.RANDOM_PLAY_COUNT,
    ],
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
    """Sorting by composer orders each composer's works by catalogue number, numbers naturally."""
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

    assert _ids(by_composer) == _ids([op_9, op_67, uncatalogued, k_525, composerless])
    assert _ids(by_composer_desc) == _ids([k_525, op_9, op_67, uncatalogued, composerless])


async def test_newest_first_sort_is_internal_only(mass: MusicAssistant) -> None:
    """The server can list works newest first, the works listing does not offer that sort."""
    token = _token()
    older = await _add_work(mass, f"B {token}", date_added=datetime(2020, 1, 1, tzinfo=UTC))
    newer = await _add_work(mass, f"C {token}", date_added=datetime(2021, 1, 1, tzinfo=UTC))
    oldest = await _add_work(mass, f"A {token}", date_added=datetime(2019, 1, 1, tzinfo=UTC))

    newest_first = await mass.music.works.get_library_items_by_query(
        search=token, sort_field=SortField.TIMESTAMP_ADDED, sort_direction=SortDirection.DESC
    )

    assert _ids(newest_first) == _ids([newer, older, oldest])
    with pytest.raises(InvalidDataError):
        await mass.music.works.library_items(search=token, sort_field=SortField.TIMESTAMP_ADDED)


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


async def test_track_work_described_by_its_provider_is_added(mass: MusicAssistant) -> None:
    """A full work on a provider track is added with the track, shared by its other tracks."""
    composer = await _add_artist(mass, "Composer")
    name = f"Symphony No. 5 {_token()}, Op. 67"

    first = await _add_track(mass, _work(name, [composer]), movement_number=1)
    second = await _add_track(mass, _work(name, [composer]), movement_number=2)

    assert first.work is not None
    assert second.work is not None
    assert first.work.item_id == second.work.item_id
    work = await mass.music.works.get_library_item(first.work.item_id)
    assert work.name == name
    assert _ids(work.composers) == [composer.item_id]
    assert _composer_ids(first) == [composer.item_id]


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

    first = await _add_track(mass, _work(name, [provider_composer]))
    second = await _add_track(mass, _work(name, [provider_composer]))

    assert first.work is not None
    assert second.work is not None
    assert first.work.item_id == second.work.item_id
    work = await mass.music.works.get_library_item(first.work.item_id)
    assert _ids(work.composers) == [composer.item_id]


async def test_provider_work_is_added_under_its_parent(mass: MusicAssistant) -> None:
    """A provider work naming the work it is part of is added under that work, shared too."""
    composer = await _add_artist(mass, "Composer")
    opera = f"Opera {_token()}"

    def act(number: str) -> Work:
        return _work(
            f"{opera}, Act {number}",
            [composer],
            parent_work=ItemMapping(
                media_type=MediaType.WORK, item_id=opera, provider=WORK_SOURCE, name=opera
            ),
        )

    first = await _add_track(mass, act("I"))
    second = await _add_track(mass, act("II"))

    assert first.work is not None
    assert second.work is not None
    act_one = await mass.music.works.get_library_item(first.work.item_id)
    act_two = await mass.music.works.get_library_item(second.work.item_id)
    assert act_one.parent_work is not None
    assert act_two.parent_work is not None
    assert act_one.parent_work.item_id == act_two.parent_work.item_id
    parent = await mass.music.works.get_library_item(act_one.parent_work.item_id)
    assert parent.name == opera
    assert _ids(parent.composers) == [composer.item_id]


async def _add_part(mass: MusicAssistant, name: str, parent: Work, composer: Artist) -> Work:
    """Add a work that is a part of a library work."""
    return await _add_work(mass, name, [composer], parent_work=ItemMapping.from_item(parent))


@pytest.mark.parametrize(
    ("stored", "added"),
    [
        ("Matthäus-Passion, BWV 244, Part 1", "St. Matthew Passion, BWV 244, Part I"),
        ("Messiah, HWV 56, Pt. 2", "The Messiah, Part II"),
        ("La bohème, Act II", "La Bohème, Act 2"),
        ("Matthäus-Passion, BWV 244, Anhang", "St. Matthew Passion, BWV 244, Anhang"),
    ],
)
async def test_part_matches_the_same_part_of_its_work(
    mass: MusicAssistant, stored: str, added: str
) -> None:
    """A part matches the part of its work with the same number, or the same unnumbered label."""
    composer = await _add_artist(mass, "Composer")
    parent = await _add_work(mass, f"Work {_token()}", [composer])
    part = await _add_part(mass, stored, parent, composer)

    assert (await _add_part(mass, added, parent, composer)).item_id == part.item_id


@pytest.mark.parametrize(
    ("stored", "added"),
    [
        ("Matthäus-Passion, BWV 244, Part 1", "Matthäus-Passion, BWV 244, Part 2"),
        ("Opera, Act 1", "Opera, Part 1"),
        ("Passion, Anhang", "Passion, Appendix"),
        ("Opera, Act 1", "Opera, Scene 1"),
    ],
)
async def test_part_never_matches_another_part(
    mass: MusicAssistant, stored: str, added: str
) -> None:
    """A part never matches a part of its work with another number or label."""
    composer = await _add_artist(mass, "Composer")
    parent = await _add_work(mass, f"Work {_token()}", [composer])
    part = await _add_part(mass, stored, parent, composer)

    assert (await _add_part(mass, added, parent, composer)).item_id != part.item_id


async def test_part_never_matches_the_part_of_another_work(mass: MusicAssistant) -> None:
    """A part only matches the parts of its own work."""
    composer = await _add_artist(mass, "Composer")
    first = await _add_work(mass, f"First {_token()}", [composer])
    second = await _add_work(mass, f"Second {_token()}", [composer])
    part = await _add_part(mass, "First Opera, Act I", first, composer)

    assert (await _add_part(mass, "Second Opera, Act 1", second, composer)).item_id != (
        part.item_id
    )


async def test_part_never_matches_a_different_musicbrainz_work(mass: MusicAssistant) -> None:
    """A part never matches the same part number of its work holding another MusicBrainz id."""
    composer = await _add_artist(mass, "Composer")
    parent = await _add_work(mass, f"Work {_token()}", [composer])
    part = await _add_work(
        mass, "Opera, Act 1", [composer], _mbid(), parent_work=ItemMapping.from_item(parent)
    )
    added = await _add_work(
        mass, "Opera, Act I", [composer], _mbid(), parent_work=ItemMapping.from_item(parent)
    )

    assert added.item_id != part.item_id


async def test_work_matches_the_work_with_its_composer_and_catalog_numbers(
    mass: MusicAssistant,
) -> None:
    """A work matches the one top-level work with the same composers and catalog numbers."""
    composer = await _add_artist(mass, "Composer")
    stored = await _add_work(
        mass, "Concerto in E major, op. 8 no. 1", [composer], catalog_numbers=["op. 8 no. 1"]
    )

    added = await _add_work(
        mass, "Violin Concerto in E Major, Op. 8 No. 1", [composer], catalog_numbers=["Op. 8 No.1"]
    )

    assert added.item_id == stored.item_id


@pytest.mark.parametrize(
    ("stored_numbers", "added_numbers"),
    [
        (["Op. 8"], ["Op. 8 No. 1"]),
        (["Op. 8 No. 1", "RV 269"], ["Op. 8 No. 1"]),
        (["Op. 8 No. 1"], ["Op. 8 No. 2"]),
        ([], []),
    ],
)
async def test_work_never_matches_other_catalog_numbers(
    mass: MusicAssistant, stored_numbers: list[str], added_numbers: list[str]
) -> None:
    """A work never matches a work with only some, other or no catalog numbers."""
    composer = await _add_artist(mass, "Composer")
    stored = await _add_work(mass, "Concerto", [composer], catalog_numbers=stored_numbers)

    added = await _add_work(mass, "Violin Concerto", [composer], catalog_numbers=added_numbers)

    assert added.item_id != stored.item_id


async def test_catalog_match_needs_the_same_composer(mass: MusicAssistant) -> None:
    """A work never matches another composer's work with the same catalog numbers."""
    stored = await _add_work(
        mass, "Symphony", [await _add_artist(mass, "Composer")], catalog_numbers=["Op. 1"]
    )

    added = await _add_work(
        mass, "Sinfonie", [await _add_artist(mass, "Other")], catalog_numbers=["Op. 1"]
    )

    assert added.item_id != stored.item_id


async def test_catalog_match_never_joins_different_musicbrainz_works(
    mass: MusicAssistant,
) -> None:
    """A work never matches a work with its catalog numbers holding another MusicBrainz id."""
    composer = await _add_artist(mass, "Composer")
    stored = await _add_work(mass, "Symphony", [composer], _mbid(), catalog_numbers=["Op. 1"])

    added = await _add_work(mass, "Sinfonie", [composer], _mbid(), catalog_numbers=["Op. 1"])

    assert added.item_id != stored.item_id


async def test_catalog_match_needs_a_single_candidate(mass: MusicAssistant) -> None:
    """A work matches none of several works with its composers and catalog numbers."""
    composer = await _add_artist(mass, "Composer")
    # works the library keeps apart, as an add would have matched the first
    stored = [
        await mass.music.works.create_library_item(
            _work(name, [composer], catalog_numbers=["BWV 244"])
        )
        for name in ("Matthäus-Passion, BWV 244", "St. Matthew Passion, BWV 244")
    ]

    added = await _add_work(mass, "Passion selon saint Matthieu", [composer])

    assert added.item_id not in _ids(stored)


async def test_catalog_match_skips_a_work_titled_as_a_part(mass: MusicAssistant) -> None:
    """A whole work never matches a top-level work titled as an act or part of it."""
    composer = await _add_artist(mass, "Composer")
    part = await mass.music.works.create_library_item(
        _work("Swan Lake, Act II, Op. 20", [composer], catalog_numbers=["Op. 20"])
    )

    added = await _add_work(mass, "Le Lac des cygnes, Op. 20", [composer])

    assert added.item_id != part.item_id


async def test_provider_works_in_two_languages_share_their_work_and_parts(
    mass: MusicAssistant,
) -> None:
    """The parts of a work its provider names in two languages join under one work."""
    composer = await _add_artist(mass, "Bach")

    def part(name: str) -> Work:
        parent = name.rsplit(", ", 1)[0]
        return _work(
            name,
            [composer],
            catalog_numbers=["BWV 244"],
            parent_work=ItemMapping(
                media_type=MediaType.WORK, item_id=parent, provider=WORK_SOURCE, name=parent
            ),
        )

    tracks = [
        await _add_track(mass, part(name))
        for name in (
            "Matthäus-Passion, BWV 244, Part I",
            "Matthäus-Passion, BWV 244, Part II",
            "Matthäus-Passion, BWV 244, Anhang",
            "St. Matthew Passion, BWV 244, Part 1",
            "St. Matthew Passion, BWV 244, Part II",
            "St. Matthew Passion, BWV 244, Anhang",
        )
    ]

    work_ids = [x.work.item_id for x in tracks if x.work]
    assert work_ids[3:] == work_ids[:3]
    assert len(set(work_ids)) == 3
    parts = [await mass.music.works.get_library_item(x) for x in work_ids[:3]]
    parent_ids = {x.parent_work.item_id for x in parts if x.parent_work}
    assert len(parent_ids) == 1
    parent = await mass.music.works.get_library_item(parent_ids.pop())
    assert parent.name == "Matthäus-Passion, BWV 244"
    assert parent.parent_work is None


async def test_merging_track_update_adds_no_provider_work(mass: MusicAssistant) -> None:
    """A merging update keeps the stored work and leaves the work it describes out."""
    work = await _add_work(mass, f"Work {_token()}")
    track = await _add_track(mass, work)
    update = await mass.music.tracks.get_library_item(track.item_id)
    update.work = _work(f"Other {_token()}")
    before = await _library_work_ids(mass)

    stored = await mass.music.tracks.update_item_in_library(track.item_id, update)

    assert stored.work is not None
    assert stored.work.item_id == work.item_id
    assert await _library_work_ids(mass) == before


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


async def test_merge_by_musicbrainz_id_keeps_the_oldest(mass: MusicAssistant) -> None:
    """Works sharing a MusicBrainz id merge into the oldest, whichever of them is given."""
    mbid = _mbid()
    # an add would match the stored work holding the id, so the duplicates are created as such
    oldest, newer, newest = [
        await mass.music.works.create_library_item(_work(f"{name} {_token()}", mbid=mbid))
        for name in ("Oldest", "Newer", "Newest")
    ]

    survivor = await mass.music.works.merge_library_works_by_mbid(int(newest.item_id), mbid)

    assert survivor == int(oldest.item_id)
    remaining = await _library_work_ids(mass)
    assert oldest.item_id in remaining
    assert not {newer.item_id, newest.item_id} & remaining


async def test_merge_by_musicbrainz_id_keeps_the_oldest_with_tracks(mass: MusicAssistant) -> None:
    """Works sharing a MusicBrainz id merge into the oldest that has tracks of its own."""
    mbid = _mbid()
    without_tracks, oldest_tagged, newest_tagged = [
        await mass.music.works.create_library_item(_work(f"{name} {_token()}", mbid=mbid))
        for name in ("Without tracks", "Oldest tagged", "Newest tagged")
    ]
    await _add_track(mass, oldest_tagged)
    await _add_track(mass, newest_tagged)

    survivor = await mass.music.works.merge_library_works_by_mbid(int(without_tracks.item_id), mbid)

    assert survivor == int(oldest_tagged.item_id)
    remaining = await _library_work_ids(mass)
    assert oldest_tagged.item_id in remaining
    assert not {without_tracks.item_id, newest_tagged.item_id} & remaining


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
    """A part is no row of its own; its work is listed and found by a search for the part."""
    token = _token()
    opera = await _add_work(mass, f"Opera {token}")
    aria = await _add_work(mass, f"Aria {token}", parent_work=ItemMapping.from_item(opera))
    await _add_track(mass, aria)
    lone = await _add_work(mass, f"Lone {token}")

    listed = await mass.music.works.library_items(search=token)
    by_part = await mass.music.works.library_items(search=f"Aria {token}")

    assert sorted(_ids(listed)) == sorted([opera.item_id, lone.item_id])
    assert _ids(by_part) == [opera.item_id]


async def test_work_with_parts_needs_a_track_the_user_sees(mass: MusicAssistant) -> None:
    """A work whose parts have no tracks in the library is not listed."""
    token = _token()
    opera = await _add_work(mass, f"Opera {token}")
    await _add_work(mass, f"Aria {token}", parent_work=ItemMapping.from_item(opera))

    assert await mass.music.works.library_items(search=token) == []


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

    assert _composer_ids(stored) == [beethoven.item_id]
    assert _ids(stored.artists) == [performer.item_id, beethoven.item_id]


async def test_movement_of_a_numbered_piece_keeps_its_own_work(mass: MusicAssistant) -> None:
    """A track that is a numbered movement of a numbered piece never joins the bare opus."""
    beethoven = await _add_artist(mass, "Ludwig van Beethoven")
    opus = await _add_work(mass, "Piano Sonatas, Op. 27", [beethoven], catalog_numbers=["Op. 27"])
    name = "Piano Sonata No. 14 in C-sharp minor, Op. 27 No. 2"

    track = await _add_track(
        mass,
        _work(name, [beethoven], catalog_numbers=["Op. 27 No. 2"]),
        movement_number=1,
        movement_name="I. Adagio sostenuto",
    )

    assert track.work is not None
    assert track.work.item_id != opus.item_id
    assert (track.movement_number, track.movement_name) == (1, "I. Adagio sostenuto")
