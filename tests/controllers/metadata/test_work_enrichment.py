"""Tests for filling library works in from MusicBrainz during the MusicBrainz link run."""

from __future__ import annotations

import json
import logging
import pathlib
import re
from time import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, PropertyMock, patch
from uuid import uuid4

import pytest
from music_assistant_models.enums import ArtistEntityType, ArtistRole, ExternalID, Period, WorkType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import (
    Artist,
    Credit,
    ItemMapping,
    ProviderMapping,
    Track,
    UniqueList,
    Work,
)
from music_assistant_models.media_items.metadata import LifeSpan, MediaItemMetadata

from music_assistant.constants import DB_TABLE_TRACK_ARTISTS, DB_TABLE_WORK_ARTISTS
from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.constants import (
    ARTIST_LOOKUP_RETRY_INTERVAL,
    REFRESH_INTERVAL,
)
from music_assistant.controllers.metadata.controller import (
    _composers_to_identify_query,
    _performers_to_identify_query,
    _works_to_enrich_query,
)
from music_assistant.controllers.metadata.helpers import composer_period, musicbrainz_name
from music_assistant.helpers.compare import is_named_by
from music_assistant.helpers.work_titles import parse_catalog_numbers
from music_assistant.providers.musicbrainz.models import (
    MusicBrainzAlias,
    MusicBrainzArtist,
    MusicBrainzLifeSpan,
    MusicBrainzRelation,
    MusicBrainzUrl,
    MusicBrainzWork,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from music_assistant.mass import MusicAssistant

FIXTURES = pathlib.Path(__file__).parents[2] / "providers" / "musicbrainz" / "fixtures" / "works"
PROVIDER_INSTANCE = "work_enrichment_instance"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


_BEETHOVEN = MediaItemMetadata(
    artist_entity_type=ArtistEntityType.PERSON,
    life_span=LifeSpan(begin="1770-12-17", end="1827-03-26", ended=True),
)
_HANDEL = MediaItemMetadata(
    artist_entity_type=ArtistEntityType.PERSON,
    life_span=LifeSpan(begin="1685-02-23", end="1759-04-14", ended=True),
)


@pytest.fixture(name="mass")
def mass_fixture(music_mass_module: MusicAssistant) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture."""
    return music_mass_module


# --------------------------------------------------------------------------- #
#  builders                                                                    #
# --------------------------------------------------------------------------- #


def _musicbrainz_works(*names: str) -> dict[str, MusicBrainzWork]:
    """Return real MusicBrainz work lookups by fixture name, with ids unique to the calling test."""
    # the tests of this module share the library database, so every MusicBrainz id is replaced
    # by a fresh one, the same throughout the returned works
    texts = {name: (FIXTURES / f"{name}.json").read_text() for name in names}
    fresh_ids: dict[str, str] = {}
    works = {}
    for name, text in texts.items():
        unique = _UUID.sub(lambda match: fresh_ids.setdefault(match.group(), str(uuid4())), text)
        works[name] = MusicBrainzWork.from_raw(json.loads(unique))
    return works


def _controller(
    mass: MusicAssistant,
    works: Iterable[MusicBrainzWork],
    artists: Iterable[MusicBrainzArtist] = (),
) -> MetaDataController:
    """Return a MetaDataController on the library database with MusicBrainz stubbed."""
    # MusicBrainz knows the given works and artists only
    ctrl = MetaDataController.__new__(MetaDataController)
    ctrl.mass = mass
    ctrl.logger = Mock()
    ctrl._logged_hints = set()
    by_id = {work.id: work for work in works}
    # MusicBrainz lists a part on its parent too
    for work in by_id.values():
        if (parent := work.parent_relation) and parent.work and parent.work.id in by_id:
            relations = by_id[parent.work.id].relations
            if not any(x.work and x.work.id == work.id for x in relations):
                relations.append(MusicBrainzRelation(type="parts", direction="forward", work=work))
    artists_by_id = {artist.id: artist for artist in artists}

    def _artist_details(mbid: str) -> MusicBrainzArtist:
        if mbid not in artists_by_id:
            raise InvalidDataError
        return artists_by_id[mbid]

    musicbrainz = Mock()
    musicbrainz.get_work_details = AsyncMock(side_effect=by_id.get)
    musicbrainz.get_artist_details = AsyncMock(side_effect=_artist_details)
    ctrl._musicbrainz_provider = Mock(return_value=musicbrainz)  # type: ignore[method-assign]
    ctrl.schedule_update_metadata = Mock()  # type: ignore[method-assign]
    return ctrl


def _token() -> str:
    """Return a word unique to the calling test, to keep its names apart from other tests."""
    return uuid4().hex[:10]


def _mapping() -> ProviderMapping:
    """Create a provider mapping with a unique provider item id."""
    return ProviderMapping(
        item_id=uuid4().hex,
        provider_domain="work_enrichment",
        provider_instance=PROVIDER_INSTANCE,
        in_library=True,
    )


async def _add_artist(
    mass: MusicAssistant, name: str, mbid: str | None = None, **kwargs: Any
) -> Artist:
    """Store a library artist, with the given MusicBrainz id if any."""
    return await mass.music.artists.add_item_to_library(
        Artist(
            item_id=uuid4().hex,
            provider=PROVIDER_INSTANCE,
            name=name,
            provider_mappings=kwargs.pop("provider_mappings", {_mapping()}),
            external_ids={(ExternalID.MB_ARTIST, mbid)} if mbid else set(),
            **kwargs,
        )
    )


async def _add_work(
    mass: MusicAssistant,
    name: str,
    composers: Sequence[Artist] = (),
    mbid: str | None = None,
    **kwargs: Any,
) -> Work:
    """Store a library work, the way a local files sync does."""
    return await mass.music.works.add_item_to_library(
        Work(
            item_id=name,
            provider=PROVIDER_INSTANCE,
            name=name,
            provider_mappings=set(),
            composers=UniqueList(composers),
            external_ids={(ExternalID.MB_WORK, mbid)} if mbid else set(),
            **kwargs,
        )
    )


async def _add_track(
    mass: MusicAssistant,
    work: Work,
    work_ids: Sequence[str] = (),
    composer: Artist | None = None,
) -> Track:
    """Store a library track of a work, carrying the given MusicBrainz work ids."""
    token = _token()
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id=token,
            provider=PROVIDER_INSTANCE,
            name=f"Track {token}",
            provider_mappings={_mapping()},
            artists=UniqueList([await _add_artist(mass, f"Orchestra {token}")]),
            credits=[Credit(artist=composer, role=ArtistRole.COMPOSER)] if composer else [],
            work=ItemMapping.from_item(work),
            external_ids={(ExternalID.MB_WORK, work_id) for work_id in work_ids},
        )
    )


async def _enrich(ctrl: MetaDataController, work: Work) -> Work:
    """Run the link step on a library work and return the work as stored afterwards."""
    await ctrl.link_item_to_musicbrainz(await ctrl.mass.music.works.get_library_item(work.item_id))
    return await ctrl.mass.music.works.get_library_item(work.item_id)


async def _library_work_ids(mass: MusicAssistant) -> set[str]:
    """Return the ids of every work stored in the library."""
    rows = await mass.music.database.get_rows_from_query("SELECT item_id FROM works", limit=0)
    return {str(row["item_id"]) for row in rows}


def _linked(works: Iterable[Work | ItemMapping]) -> list[tuple[str, list[str]]]:
    """Return the id and composer ids of each linked work."""
    linked = []
    for work in works:
        assert isinstance(work, Work)
        linked.append((work.item_id, [x.item_id for x in work.composers]))
    return linked


def _composers(track: Track) -> list[str]:
    """Return the names of the composers credited on a track."""
    return [c.artist.name for c in track.credits if c.role == ArtistRole.COMPOSER]


def _composer_ids(track: Track) -> list[str]:
    """Return the library ids of the composers credited on a track."""
    return [c.artist.item_id for c in track.credits if c.role == ArtistRole.COMPOSER]


async def _due_works(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library works the MusicBrainz link run would select now."""
    selected = await mass.music.works.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_works_to_enrich_query()],
        extra_query_params={"stale": int(time() - REFRESH_INTERVAL)},
    )
    return {x.item_id for x in selected}


def _unique_composer(mb_work: MusicBrainzWork, token: str) -> MusicBrainzArtist:
    """Return the composer of a MusicBrainz work, with the token added to its names."""
    artist = mb_work.composer_relations[0].artist
    assert artist is not None
    artist.name = f"{artist.name} {token}"
    artist.sort_name = f"{artist.sort_name} {token}"
    return artist


def _linked_to_spotify(mb_artist: MusicBrainzArtist, spotify_id: str) -> MusicBrainzArtist:
    """Return the MusicBrainz artist lookup of a composer, linked to a Spotify artist."""
    url = MusicBrainzUrl(resource=f"https://open.spotify.com/artist/{spotify_id}")
    return MusicBrainzArtist(
        id=mb_artist.id,
        name=mb_artist.name,
        sort_name=mb_artist.sort_name,
        relations=[MusicBrainzRelation(type="free streaming", url=url)],
    )


def _spotify_instances() -> Mock:
    """Return a stand-in for the music provider instance lookup with Spotify loaded."""

    def _instances(domain: str, **_kwargs: Any) -> list[Mock]:
        return [Mock(available=True, instance_id="spotify")] if domain == "spotify" else []

    return Mock(side_effect=_instances)


def _scheduled(ctrl: MetaDataController) -> list[str]:
    """Return the names of the artists scheduled for a metadata refresh."""
    schedule = cast("Mock", ctrl.schedule_update_metadata)
    return [call.args[0].name for call in schedule.call_args_list]


# --------------------------------------------------------------------------- #
#  identifying the work                                                        #
# --------------------------------------------------------------------------- #


async def test_movement_id_identifies_the_parent_work_and_fills_it(mass: MusicAssistant) -> None:
    """The id of a track's movement leads to the parent work, whose fields fill the work."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    symphony, movement = works["symphony_5"], works["symphony_5_movement_1"]
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    work = await _add_work(mass, f"Symphony no. 5 {_token()}", [composer])
    track = await _add_track(mass, work, [movement.id], composer)
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert stored.mbid == symphony.id
    assert stored.work_type == WorkType.SYMPHONY
    assert stored.composition_year == 1808
    assert stored.language is None
    assert stored.musical_key == "C minor"
    assert stored.catalog_numbers == ["op. 67"]
    assert stored.metadata.last_musicbrainz_lookup is not None
    # the work type makes its tracks classical
    assert (await mass.music.tracks.get_library_item(track.item_id)).is_classical


async def test_work_stores_the_movements_musicbrainz_lists(mass: MusicAssistant) -> None:
    """The parts MusicBrainz lists for the work are stored as its movements."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    work = await _add_work(mass, f"Symphony no. 5 {_token()}")
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])

    stored = await _enrich(_controller(mass, works.values()), work)

    assert stored.metadata.movements == works["symphony_5"].movements
    assert [(x.number, x.name) for x in stored.metadata.movements or []] == [
        (1, "I. Allegro con brio"),
        (2, "II. Andante con moto"),
    ]


async def test_work_type_comes_from_the_top_level_work(mass: MusicAssistant) -> None:
    """A work identified by its own id takes the type of the work it is a part of."""
    song = _musicbrainz_works("gute_nacht")["gute_nacht"]
    parent = song.parent_relation
    assert parent is not None
    assert parent.work is not None
    cycle = MusicBrainzWork(id=parent.work.id, title=parent.work.title, type="Song-cycle")
    work = await _add_work(mass, f"Gute Nacht {_token()}", mbid=song.id)
    ctrl = _controller(mass, [song, cycle])

    stored = await _enrich(ctrl, work)

    assert stored.mbid == song.id
    assert stored.work_type == WorkType.SONG_CYCLE
    assert stored.language == "deu"
    assert stored.composition_year == 1827


async def test_values_the_work_already_has_are_kept(mass: MusicAssistant) -> None:
    """MusicBrainz only fills fields that are empty, the stored ones win."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    name = f"Symphony no. 5 {_token()}"
    work = await _add_work(
        mass,
        name,
        [composer],
        work_type=WorkType.OTHER,
        composition_year=1807,
        language="eng",
        musical_key="c-moll",
        catalog_numbers=["Op. 67"],
    )
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert stored.mbid == works["symphony_5"].id
    assert stored.name == name
    assert (
        stored.work_type,
        stored.composition_year,
        stored.language,
        stored.musical_key,
        stored.catalog_numbers,
    ) == (WorkType.OTHER, 1807, "eng", "c-moll", ["Op. 67"])


async def test_parent_held_by_another_work_merges_into_the_oldest(mass: MusicAssistant) -> None:
    """When another library work already holds the parent's id, the two works merge."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    symphony = works["symphony_5"]
    holder_name = f"Fifth {_token()}"
    holder = await _add_work(mass, holder_name, mbid=symphony.id)
    await _add_track(mass, holder)
    work = await _add_work(mass, f"Symphony no. 5 {_token()}")
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values())

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(work.item_id))

    assert await _library_work_ids(mass) & {holder.item_id, work.item_id} == {holder.item_id}
    kept = await mass.music.works.get_library_item(holder.item_id)
    assert (kept.name, kept.mbid, kept.work_type) == (holder_name, symphony.id, WorkType.SYMPHONY)
    assert kept.metadata.last_musicbrainz_lookup is not None
    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert stored_track.work is not None
    assert stored_track.work.item_id == holder.item_id


async def test_works_sharing_a_musicbrainz_id_merge_into_the_oldest(mass: MusicAssistant) -> None:
    """Library works carrying the same MusicBrainz id merge when one of them is enriched."""
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    older = await _add_work(mass, f"Fifth {_token()}", mbid=symphony.id)
    newer = await _add_work(mass, f"Fifth again {_token()}", mbid=symphony.id)
    ctrl = _controller(mass, [symphony])

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(newer.item_id))

    assert await _library_work_ids(mass) & {older.item_id, newer.item_id} == {older.item_id}
    assert (await mass.music.works.get_library_item(older.item_id)).work_type == (WorkType.SYMPHONY)


@pytest.mark.parametrize(
    ("name", "identified"),
    [("Symphony No. 5 in C Minor, Op. 67", True), ("Fifth Symphony", False)],
)
async def test_track_work_without_a_parent_needs_the_same_title(
    mass: MusicAssistant, name: str, identified: bool
) -> None:
    """A track's MusicBrainz work that is no part of another identifies a work of its title."""
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    work = await _add_work(mass, name, [composer])
    await _add_track(mass, work, [symphony.id], composer)
    ctrl = _controller(mass, [symphony])

    stored = await _enrich(ctrl, work)

    assert (stored.mbid == symphony.id) is identified
    assert (stored.work_type == WorkType.SYMPHONY) is identified
    assert stored.metadata.last_musicbrainz_lookup is not None


async def test_track_work_of_the_same_title_is_the_work_not_its_parent(
    mass: MusicAssistant,
) -> None:
    """A library work titled as the track's MusicBrainz work is that work, not its parent."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    movement = works["symphony_5_movement_1"]
    work = await _add_work(mass, movement.title)
    await _add_track(mass, work, [movement.id])
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert stored.mbid == movement.id
    assert stored.work_type == WorkType.SYMPHONY


@pytest.mark.parametrize(("suffix", "level"), [("", 0), (": Teil I", 1)])
async def test_work_takes_the_level_of_its_title_and_the_nearest_composers(
    mass: MusicAssistant, suffix: str, level: int
) -> None:
    """A work is the level above its movement that has its title, else the movement's parent."""
    token = _token()
    name = f"St Matthew Passion {token}"
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    passion = MusicBrainzWork(
        id=str(uuid4()),
        title=name,
        type="Oratorio",
        relations=[MusicBrainzRelation(type="composer", direction="backward", artist=mb_composer)],
    )
    part = MusicBrainzWork(
        id=str(uuid4()),
        title=f"{name}: Teil I",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=passion)],
    )
    movement = MusicBrainzWork(
        id=str(uuid4()),
        title=f"{name}: Teil I: 1. Kommt, ihr Töchter",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=part)],
    )
    composer = await _add_artist(mass, mb_composer.name, mb_composer.id)
    work = await _add_work(mass, f"{name}{suffix}")
    await _add_track(mass, work, [movement.id])
    ctrl = _controller(mass, [passion, part, movement])

    stored = await _enrich(ctrl, work)

    assert stored.mbid == [passion.id, part.id][level]
    # a part is typed by the work it belongs to
    assert stored.work_type == WorkType.ORATORIO
    assert [x.item_id for x in stored.composers] == [composer.item_id]


async def test_work_musicbrainz_does_not_know_is_marked_looked_up(mass: MusicAssistant) -> None:
    """A work whose id MusicBrainz does not know is left as it is and not looked up again."""
    work = await _add_work(mass, f"Unknown {_token()}", mbid=str(uuid4()))
    ctrl = _controller(mass, [])

    stored = await _enrich(ctrl, work)

    assert stored.work_type is None
    assert stored.metadata.last_musicbrainz_lookup is not None
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


# --------------------------------------------------------------------------- #
#  arrangements                                                                #
# --------------------------------------------------------------------------- #


async def test_arrangement_in_the_library_is_linked_to_its_original(mass: MusicAssistant) -> None:
    """An original's forward arrangement relation links the library work arranged from it."""
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    [relation] = symphony.arrangement_relations
    assert relation.work is not None
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    original = await _add_work(mass, f"Symphony no. 5 {_token()}", [composer], mbid=symphony.id)
    transcription = await _add_work(mass, f"S. 464 no. 5 {_token()}", mbid=relation.work.id)
    ctrl = _controller(mass, [symphony])

    stored = await _enrich(ctrl, original)

    assert _linked(stored.arrangements) == [(transcription.item_id, [])]
    arranged = await mass.music.works.get_library_item(transcription.item_id)
    assert _linked(arranged.arrangement_of) == [(original.item_id, [composer.item_id])]


@pytest.mark.parametrize("relation_type", ["arrangement", "orchestration"])
async def test_arrangement_is_linked_to_its_original_in_the_library(
    mass: MusicAssistant, relation_type: str
) -> None:
    """A work's backward arrangement or orchestration relation links it to its original."""
    token = _token()
    original_mb = MusicBrainzWork(id=str(uuid4()), title=f"Pictures {token}")
    arrangement_mb = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Pictures, orchestrated {token}",
        relations=[MusicBrainzRelation(type=relation_type, direction="backward", work=original_mb)],
    )
    original = await _add_work(mass, original_mb.title, mbid=original_mb.id)
    arrangement = await _add_work(mass, arrangement_mb.title, mbid=arrangement_mb.id)
    ctrl = _controller(mass, [arrangement_mb])

    stored = await _enrich(ctrl, arrangement)

    assert [x.item_id for x in stored.arrangement_of] == [original.item_id]
    assert stored.arrangements == []


async def test_arrangement_missing_from_the_library_is_not_added(mass: MusicAssistant) -> None:
    """An arrangement MusicBrainz names that the library lacks adds no work."""
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    original = await _add_work(mass, f"Symphony no. 5 {_token()}", mbid=symphony.id)
    ctrl = _controller(mass, [symphony])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, original)

    assert stored.arrangements == []
    assert await _library_work_ids(mass) == before


# --------------------------------------------------------------------------- #
#  parent works                                                                #
# --------------------------------------------------------------------------- #


def _passion(token: str, mb_composer: MusicBrainzArtist) -> list[MusicBrainzWork]:
    """Return a movement of a part of a passion, followed by the part and the passion."""
    passion = MusicBrainzWork(
        id=str(uuid4()),
        title=f"St Matthew Passion {token}",
        type="Oratorio",
        relations=[MusicBrainzRelation(type="composer", direction="backward", artist=mb_composer)],
    )
    part = MusicBrainzWork(
        id=str(uuid4()),
        title=f"St Matthew Passion {token}: Teil I",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=passion)],
    )
    movement = MusicBrainzWork(
        id=str(uuid4()),
        title=f"St Matthew Passion {token}: Teil I: 1. Kommt, ihr Töchter",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=part)],
    )
    return [movement, part, passion]


def _work_levels(*levels: tuple[str, str | None]) -> list[MusicBrainzWork]:
    """Return MusicBrainz works of the given titles and types, each a part of the next."""
    works: list[MusicBrainzWork] = []
    for title, work_type in reversed(levels):
        relations = (
            [MusicBrainzRelation(type="parts", direction="backward", work=works[0])]
            if works
            else []
        )
        works.insert(
            0, MusicBrainzWork(id=str(uuid4()), title=title, type=work_type, relations=relations)
        )
    return works


async def test_excerpt_gets_the_nearest_typed_work_added_as_its_parent(
    mass: MusicAssistant,
) -> None:
    """A part links to the nearest typed work above it, which is added without tracks."""
    token = _token()
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    movement, part, passion = _passion(token, mb_composer)
    composer = await _add_artist(mass, mb_composer.name, mb_composer.id)
    name = f"Kommt, ihr Töchter {token}"
    work = await _add_work(mass, name, mbid=movement.id)
    track = await _add_track(mass, work)
    ctrl = _controller(mass, [movement, part, passion])

    stored = await _enrich(ctrl, work)

    assert stored.name == name
    assert stored.parent_work is not None
    parent = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert (parent.name, parent.mbid, parent.work_type) == (
        passion.title,
        passion.id,
        WorkType.ORATORIO,
    )
    assert [x.item_id for x in parent.composers] == [composer.item_id]
    assert parent.metadata.last_musicbrainz_lookup is not None
    assert parent.parent_work is None
    assert [x.item_id for x in parent.parts] == [work.item_id]
    assert not await mass.music.works.get_library_items_by_external_id(
        part.id, ExternalID.MB_WORK, limit=None
    )
    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert stored_track.work is not None
    assert stored_track.work.item_id == work.item_id
    # the parent is filled in already, so the link run does not look it up again
    assert parent.item_id not in await _due_works(mass)


async def test_parent_is_the_library_work_of_its_composer_and_catalogue_number(
    mass: MusicAssistant,
) -> None:
    """A parent the library holds without its id, under another title, is not added again."""
    passion = _musicbrainz_works("st_matthew_passion")["st_matthew_passion"]
    token = _token()
    composer = await _add_artist(mass, _unique_composer(passion, token).name)
    library_passion = await _add_work(
        mass, f"St Matthew Passion {token}, BWV 244", [composer], catalog_numbers=["BWV 244"]
    )
    await _add_track(mass, library_passion)
    movement = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Matthäus-Passion, BWV 244: Kommt {token}",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=passion)],
    )
    work = await _add_work(mass, f"Kommt {token}", [composer], mbid=movement.id)
    ctrl = _controller(mass, [movement, passion])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    assert stored.parent_work.item_id == library_passion.item_id
    assert (await mass.music.works.get_library_item(library_passion.item_id)).mbid == passion.id
    assert await _library_work_ids(mass) == before


async def test_nearest_typed_level_is_the_parent_rather_than_the_top(
    mass: MusicAssistant,
) -> None:
    """An aria links to its opera, not to the set of operas above it."""
    token = _token()
    aria, opera, triptych = _work_levels(
        (f"Gianni Schicchi {token}: O mio babbino caro", "Aria"),
        (f"Gianni Schicchi {token}", "Opera"),
        (f"Il trittico {token}", "Opera"),
    )
    work = await _add_work(mass, aria.title, mbid=aria.id)
    ctrl = _controller(mass, [aria, opera, triptych])

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    parent = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert (parent.name, parent.mbid) == (opera.title, opera.id)
    assert not await mass.music.works.get_library_items_by_external_id(
        triptych.id, ExternalID.MB_WORK, limit=None
    )


async def test_work_without_a_typed_level_above_gets_no_parent(mass: MusicAssistant) -> None:
    """A sonata in an untyped set of sonatas gets no parent and no work is added."""
    token = _token()
    sonata, sonatas = _work_levels(
        (f"Moonlight Sonata {token}", "Sonata"), (f"Two Piano Sonatas, op. 27 {token}", None)
    )
    work = await _add_work(mass, sonata.title, mbid=sonata.id)
    ctrl = _controller(mass, [sonata, sonatas])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is None
    assert stored.mbid == sonata.id
    assert await _library_work_ids(mass) == before


async def test_added_parent_merges_into_a_newer_work_with_tracks(mass: MusicAssistant) -> None:
    """An added parent without tracks merges into the newer work with tracks that is the same."""
    token = _token()
    aria, opera = _work_levels(
        (f"Gianni Schicchi {token}: O mio babbino caro", "Aria"),
        (f"Gianni Schicchi {token}", "Opera"),
    )
    excerpt = await _add_work(mass, aria.title, mbid=aria.id)
    ctrl = _controller(mass, [aria, opera])
    added_parent = (await _enrich(ctrl, excerpt)).parent_work
    assert added_parent is not None
    tagged_name = f"Schicchi {token}"
    tagged = await _add_work(mass, tagged_name)
    await _add_track(mass, tagged, [aria.id])

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(tagged.item_id))

    remaining = await _library_work_ids(mass)
    assert remaining & {added_parent.item_id, tagged.item_id} == {tagged.item_id}
    kept = await mass.music.works.get_library_item(tagged.item_id)
    assert (kept.name, kept.mbid) == (tagged_name, opera.id)
    stored_excerpt = await mass.music.works.get_library_item(excerpt.item_id)
    assert stored_excerpt.parent_work is not None
    assert stored_excerpt.parent_work.item_id == tagged.item_id


async def test_parent_work_takes_no_composer_the_library_lacks(mass: MusicAssistant) -> None:
    """A parent work added from MusicBrainz only gets composers the library already holds."""
    token = _token()
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    hierarchy = _passion(token, mb_composer)
    work = await _add_work(mass, f"Kommt {token}", mbid=hierarchy[0].id)
    ctrl = _controller(mass, hierarchy)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    parent = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert parent.composers == []
    assert not await mass.music.artists.get_library_items_by_query(search=token)


async def test_parent_work_takes_the_composers_of_the_work_when_unmatched(
    mass: MusicAssistant,
) -> None:
    """A parent whose MusicBrainz composers the library lacks gets the work's own composers."""
    token = _token()
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    hierarchy = _passion(token, mb_composer)
    composer = await _add_artist(mass, f"J. S. Bach {token}")
    work = await _add_work(mass, f"Kommt {token}", [composer], mbid=hierarchy[0].id)
    ctrl = _controller(mass, hierarchy)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    parent = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert [x.item_id for x in parent.composers] == [composer.item_id]


async def test_library_work_holding_the_top_level_id_becomes_the_parent(
    mass: MusicAssistant,
) -> None:
    """The library work that is the top-level MusicBrainz work is the parent, none is added."""
    token = _token()
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    movement, part, passion = _passion(token, mb_composer)
    holder = await _add_work(mass, f"Matthäus-Passion {token}", mbid=passion.id)
    work = await _add_work(mass, f"Kommt {token}", mbid=movement.id)
    ctrl = _controller(mass, [movement, part, passion])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    assert stored.parent_work.item_id == holder.item_id
    assert await _library_work_ids(mass) == before


async def test_parent_work_with_its_own_id_is_kept(mass: MusicAssistant) -> None:
    """A work whose parent work has its own MusicBrainz id keeps it and adds none."""
    token = _token()
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    hierarchy = _passion(token, mb_composer)
    tagged_parent = await _add_work(mass, f"Passion {token}", mbid=str(uuid4()))
    work = await _add_work(
        mass,
        f"Kommt {token}",
        mbid=hierarchy[0].id,
        parent_work=ItemMapping.from_item(tagged_parent),
    )
    ctrl = _controller(mass, hierarchy)
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    assert stored.parent_work.item_id == tagged_parent.item_id
    assert await _library_work_ids(mass) == before


async def test_enriching_again_changes_nothing(mass: MusicAssistant) -> None:
    """A work enriched a second time gets no second parent and no second arrangement link."""
    token = _token()
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
    movement, part, passion = _passion(token, mb_composer)
    original = await _add_work(mass, f"Chorale {token}", mbid=str(uuid4()))
    movement.relations.append(
        MusicBrainzRelation(
            type="arrangement",
            direction="backward",
            work=MusicBrainzWork(id=str(original.mbid), title=original.name),
        )
    )
    work = await _add_work(mass, f"Kommt {token}", mbid=movement.id)
    ctrl = _controller(mass, [movement, part, passion])

    first = await _enrich(ctrl, work)
    works_after_first = await _library_work_ids(mass)
    second = await _enrich(ctrl, work)

    assert await _library_work_ids(mass) == works_after_first
    assert second.parent_work == first.parent_work
    assert [x.item_id for x in second.arrangement_of] == [original.item_id]


# --------------------------------------------------------------------------- #
#  acts and the levels of an opera                                             #
# --------------------------------------------------------------------------- #


def _boheme(token: str) -> tuple[list[MusicBrainzWork], list[MusicBrainzWork]]:
    """Return an aria of each of two acts of an opera, each followed by its act and the opera."""
    first_aria, first_act, opera = _work_levels(
        (f"La bohème {token}: Atto I, no. 6. Che gelida manina", "Aria"),
        (f"La bohème {token}: Atto I", None),
        (f"La bohème {token}", "Opera"),
    )
    second_act = MusicBrainzWork(
        id=str(uuid4()),
        title=f"La bohème {token}: Atto II",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=opera)],
    )
    second_aria = MusicBrainzWork(
        id=str(uuid4()),
        title=f"La bohème {token}: Atto II. Quando me'n vo'",
        type="Aria",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=second_act)],
    )
    return [first_aria, first_act, opera], [second_aria, second_act, opera]


async def _add_act(mass: MusicAssistant, name: str, parent: Work) -> Work:
    """Store a library act of a work, the way a streaming provider names one."""
    return await _add_work(mass, name, parent_work=ItemMapping.from_item(parent))


async def test_act_takes_the_act_level_and_its_parent_the_opera(mass: MusicAssistant) -> None:
    """An act whose track performs an aria is the act above the aria, its opera the opera."""
    token = _token()
    _, (aria, act_level, opera) = _boheme(token)
    parent = await _add_work(mass, f"La bohème {token}")
    act = await _add_act(mass, f"La bohème {token}, Act II", parent)
    await _add_track(mass, act, [aria.id])
    ctrl = _controller(mass, [aria, act_level, opera])
    assert act.item_id in await _due_works(mass)

    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, act)

    assert stored.mbid == act_level.id
    assert stored.parent_work is not None
    assert stored.parent_work.item_id == parent.item_id
    stored_parent = await mass.music.works.get_library_item(parent.item_id)
    assert (stored_parent.mbid, stored_parent.work_type) == (opera.id, WorkType.OPERA)
    assert await _library_work_ids(mass) == before


@pytest.mark.parametrize(("catalog_number", "identified"), [("Op. 105", True), ("Op. 104", False)])
async def test_work_without_parts_is_recognised_by_its_catalog_numbers(
    mass: MusicAssistant, catalog_number: str, identified: bool
) -> None:
    """A work MusicBrainz titles in another language and without parts matches its catalogue."""
    token = _token()
    (symphony,) = _work_levels((f"Sinfonia nro 7 {token} C-duuri, op. 105", "Symphony"))
    work = await _add_work(
        mass,
        f"Symphony No. 7 {token} in C Major, {catalog_number}",
        catalog_numbers=[catalog_number],
    )
    await _add_track(mass, work, [symphony.id])
    ctrl = _controller(mass, [symphony])

    stored = await _enrich(ctrl, work)

    assert stored.mbid == (symphony.id if identified else None)


@pytest.mark.parametrize(
    ("tracks", "identified"),
    [
        (["piece", "piece"], True),
        (["piece", None], False),
        (["piece"], False),
        (["piece", "other"], False),
    ],
    ids=["all-tracks-the-same-work", "a-track-without-id", "single-track", "different-works"],
)
async def test_work_without_parts_is_recognised_by_all_its_tracks_performing_it(
    mass: MusicAssistant, tracks: list[str | None], identified: bool
) -> None:
    """A work whose several tracks all perform one work without parts is that work."""
    token = _token()
    (piece,) = _work_levels((f"Miserere mei, Deus {token}", None))
    (other,) = _work_levels((f"Jubilate Deo {token}", None))
    work = await _add_work(mass, f"Miserere {token}")
    by_name = {"piece": piece, "other": other}
    for name in tracks:
        await _add_track(mass, work, [by_name[name].id] if name else [])
    ctrl = _controller(mass, [piece, other])

    stored = await _enrich(ctrl, work)

    assert stored.mbid == (piece.id if identified else None)


async def test_act_never_takes_the_top_of_the_hierarchy(mass: MusicAssistant) -> None:
    """An act whose track performs a part directly below the ballet stays unidentified."""
    token = _token()
    waltz, ballet = _work_levels(
        (f"The Nutcracker {token}, op. 71: Waltz of the Flowers", None),
        (f"The Nutcracker {token}, op. 71", "Ballet"),
    )
    parent = await _add_work(mass, f"The Nutcracker {token}, Op. 71")
    act = await _add_act(mass, f"The Nutcracker {token}, Op. 71, Act II", parent)
    await _add_track(mass, act, [waltz.id])
    ctrl = _controller(mass, [waltz, ballet])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, act)

    assert stored.mbid is None
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert stored.parent_work is not None
    assert stored.parent_work.item_id == parent.item_id
    assert (await mass.music.works.get_library_item(parent.item_id)).mbid is None
    assert await _library_work_ids(mass) == before


async def test_work_spanning_parts_of_a_larger_work_is_not_one_of_them(
    mass: MusicAssistant,
) -> None:
    """A passion whose tracks perform numbers of both its halves takes neither half's id."""
    token = _token()
    first_number, first_half, passion = _work_levels(
        (f"Matthäus-Passion {token}: Teil I: 1. Kommt", None),
        (f"Matthäus-Passion {token}: Teil I", None),
        (f"Matthäus-Passion {token}", "Oratorio"),
    )
    second_half = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Matthäus-Passion {token}: Teil II",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=passion)],
    )
    second_number = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Matthäus-Passion {token}: Teil II: 30. Ach",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=second_half)],
    )
    holder = await _add_work(mass, f"St Matthew Passion {token}, Part I", mbid=first_half.id)
    work = await _add_work(mass, f"St. Matthew Passion {token}")
    await _add_track(mass, work, [first_number.id])
    await _add_track(mass, work, [second_number.id])
    ctrl = _controller(mass, [first_number, first_half, second_number, second_half, passion])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, work)

    assert (stored.mbid, stored.parent_work) == (None, None)
    assert await _library_work_ids(mass) == before
    assert (await mass.music.works.get_library_item(holder.item_id)).parent_work is None


async def test_part_never_takes_the_level_of_its_parent(mass: MusicAssistant) -> None:
    """A part whose parent holds the level above the track's part stays unidentified."""
    token = _token()
    number, opera = _work_levels(
        (f"Carmen {token}: Habanera", "Aria"), (f"Carmen {token}", "Opera")
    )
    deeper = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Carmen {token}: Habanera, part 1",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=number)],
    )
    parent = await _add_work(mass, number.title, mbid=number.id)
    act = await _add_act(mass, f"Carmen {token}: Habanera (excerpt)", parent)
    await _add_track(mass, act, [deeper.id])
    ctrl = _controller(mass, [deeper, number, opera])

    stored = await _enrich(ctrl, act)

    assert stored.mbid is None


async def test_part_links_to_the_act_the_library_holds(mass: MusicAssistant) -> None:
    """An aria links to the library work holding its act rather than to the opera."""
    token = _token()
    (aria, act_level, opera), _ = _boheme(token)
    holder = await _add_work(mass, f"La bohème {token}, Act I", mbid=act_level.id)
    work = await _add_work(mass, aria.title, mbid=aria.id)
    ctrl = _controller(mass, [aria, act_level, opera])

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    assert stored.parent_work.item_id == holder.item_id
    assert not await mass.music.works.get_library_items_by_external_id(
        opera.id, ExternalID.MB_WORK, limit=None
    )


async def test_identified_act_takes_its_arias_from_the_opera(mass: MusicAssistant) -> None:
    """The arias of an act found among the parts of its opera move under the act."""
    token = _token()
    (first_aria, first_act, opera), (second_aria, second_act, _) = _boheme(token)
    other_aria = MusicBrainzWork(
        id=str(uuid4()),
        title=f"La bohème {token}: Atto I, no. 9. O soave fanciulla",
        type="Aria",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=first_act)],
    )
    library_opera = await _add_work(mass, f"La bohème {token}", mbid=opera.id)
    arias = [
        await _add_work(mass, x.title, mbid=x.id, parent_work=ItemMapping.from_item(library_opera))
        for x in (first_aria, other_aria, second_aria)
    ]
    act = await _add_act(mass, f"La bohème {token}, Act I", library_opera)
    await _add_track(mass, act, [first_aria.id])
    ctrl = _controller(mass, [first_aria, other_aria, second_aria, first_act, second_act, opera])

    stored = await _enrich(ctrl, act)

    assert stored.mbid == first_act.id
    parents = [(await mass.music.works.get_library_item(x.item_id)).parent_work for x in arias]
    assert [x.item_id if x else None for x in parents] == [
        act.item_id,
        act.item_id,
        library_opera.item_id,
    ]
    # moved once, the arias stay where they are
    assert (await _enrich(ctrl, act)).mbid == first_act.id
    stored_act = await mass.music.works.get_library_item(act.item_id)
    assert {x.item_id for x in stored_act.parts} == {arias[0].item_id, arias[1].item_id}


async def test_act_moves_from_its_parent_without_id_to_the_opera_the_library_holds(
    mass: MusicAssistant,
) -> None:
    """An act leaves a parent without an id for the library opera, and that empty parent goes."""
    token = _token()
    (aria, act_level, opera), _ = _boheme(token)
    other_aria = MusicBrainzWork(
        id=str(uuid4()),
        title=f"La bohème {token}: Atto I, no. 9. O soave fanciulla",
        type="Aria",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=act_level)],
    )
    library_opera = await _add_work(mass, f"La bohème {token}", mbid=opera.id)
    title_parent = await _add_work(mass, f"La bohème {token}, SC 67")
    act = await _add_act(mass, f"La bohème {token}, SC 67 / Act 1", title_parent)
    await _add_track(mass, act, [aria.id])
    arias = [
        await _add_work(mass, x.title, mbid=x.id, parent_work=ItemMapping.from_item(act))
        for x in (aria, other_aria)
    ]
    ctrl = _controller(mass, [aria, other_aria, act_level, opera])

    stored = await _enrich(ctrl, act)

    assert stored.mbid == act_level.id
    assert stored.parent_work is not None
    assert stored.parent_work.item_id == library_opera.item_id
    assert {x.item_id for x in stored.parts} == {x.item_id for x in arias}
    assert title_parent.item_id not in await _library_work_ids(mass)


@pytest.mark.parametrize("kept_by", ["track", "part"])
async def test_parent_without_id_holding_more_stays_when_its_act_moves(
    mass: MusicAssistant, kept_by: str
) -> None:
    """A parent without an id that still holds a track or another part stays in the library."""
    token = _token()
    (aria, act_level, opera), _ = _boheme(token)
    library_opera = await _add_work(mass, f"La bohème {token}", mbid=opera.id)
    title_parent = await _add_work(mass, f"La bohème {token}, SC 67")
    act = await _add_act(mass, f"La bohème {token}, SC 67 / Act 1", title_parent)
    await _add_track(mass, act, [aria.id])
    if kept_by == "track":
        await _add_track(mass, title_parent)
    else:
        await _add_act(mass, f"La bohème {token}, SC 67 / Act 2", title_parent)
    ctrl = _controller(mass, [aria, act_level, opera])

    stored = await _enrich(ctrl, act)

    assert stored.parent_work is not None
    assert stored.parent_work.item_id == library_opera.item_id
    assert title_parent.item_id in await _library_work_ids(mass)


# --------------------------------------------------------------------------- #
#  composers                                                                   #
# --------------------------------------------------------------------------- #


async def test_sole_composer_gets_the_musicbrainz_id_and_merges(mass: MusicAssistant) -> None:
    """The only composer gets the id of the only MusicBrainz composer and merges, oldest first."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    older = await _add_artist(mass, f"L. van Beethoven {token}", beethoven.id)
    composer = await _add_artist(mass, f"Beethoven {token}")
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert [(x.item_id, x.name) for x in stored.composers] == [
        (older.item_id, f"L. van Beethoven {token}")
    ]
    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert _composers(stored_track) == [f"L. van Beethoven {token}"]
    same_id = await mass.music.artists.get_library_items_by_external_id(
        beethoven.id, ExternalID.MB_ARTIST, limit=None
    )
    assert [x.item_id for x in same_id] == [older.item_id]
    assert _scheduled(ctrl) == [f"L. van Beethoven {token}"]


async def test_sole_composer_gets_its_period_from_the_musicbrainz_life_dates(
    mass: MusicAssistant,
) -> None:
    """An identified sole composer takes its life dates from MusicBrainz, without a refresh."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    beethoven.type = "Person"
    beethoven.life_span = MusicBrainzLifeSpan(begin="1770-12-17", end="1827-03-26", ended=True)
    composer = await _add_artist(mass, f"Beethoven {token}")
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values(), [beethoven])
    cast("Mock", ctrl._musicbrainz_provider()).get_artists_by_name = AsyncMock(return_value=[])

    await _enrich(ctrl, work)

    stored = await mass.music.artists.get_library_item(composer.item_id)
    assert stored.mbid == beethoven.id
    assert stored.period == composer_period(1770, 1827)
    assert _scheduled(ctrl) == []


async def test_sole_composer_of_another_name_gets_no_musicbrainz_id(mass: MusicAssistant) -> None:
    """A composer whose name MusicBrainz's composer does not carry is not given its id."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, f"Bach {token}")
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert stored.mbid == works["symphony_5"].id
    assert [(x.item_id, x.mbid) for x in stored.composers] == [(composer.item_id, None)]
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


async def test_library_composers_named_by_a_name_leave_out_the_others(
    mass: MusicAssistant,
) -> None:
    """Asked for the composers a name may name, the composers it cannot name are left out."""
    token = _token()
    composers = [
        await _add_artist(mass, f"Ralph Vaughan-Williams{token}", uuid4().hex),
        await _add_artist(mass, f"Frédéric Chopin{token}", uuid4().hex),
    ]
    for composer in composers:
        work = await _add_work(
            mass, f"Symphony {composer.name}", [composer], work_type=WorkType.SYMPHONY
        )
        await _add_track(mass, work, [], composer)

    artists = mass.music.artists
    for name, composer in zip(
        (f"Vaughan-Williams{token}", f"Frederic Chopin{token}"), composers, strict=True
    ):
        found = await artists.get_library_composers(with_mbid=True, named_by=name)
        assert [x.item_id for x in found] == [composer.item_id]


async def test_sole_composer_sharing_a_name_with_another_composer_gets_no_id(
    mass: MusicAssistant,
) -> None:
    """A family name such as "Bach" is left alone while another composer of it is confirmed."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    _unique_composer(works["symphony_5"], token)
    relative = await _add_artist(mass, f"Karl van Beethoven {token}", uuid4().hex)
    relative_work = await _add_work(
        mass, f"Symphony no. 1 {token}", [relative], work_type=WorkType.SYMPHONY
    )
    await _add_track(mass, relative_work, [], relative)
    assert relative.item_id in {
        x.item_id for x in await mass.music.artists.get_library_composers(with_mbid=True)
    }
    composer = await _add_artist(mass, f"Beethoven {token}")
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert [(x.item_id, x.mbid) for x in stored.composers] == [(composer.item_id, None)]


@pytest.mark.parametrize(
    ("name", "mb_name", "mb_sort_name", "expected"),
    [
        ("Mahler", "Gustav Mahler", "Mahler, Gustav", True),
        ("Prokofiev", "Sergei Prokofiev", "Prokofiev, Sergei Sergeyevich", True),
        ("Sergeyevich Prokofiev", "Sergei Prokofiev", "Prokofiev, Sergei Sergeyevich", True),
        ("antonín DVOŘÁK", "Antonín Dvořák", "Dvořák, Antonín", True),
        ("Vaughan-Williams", "Ralph Vaughan Williams", "Vaughan Williams, Ralph", True),
        ("Beethoven", "Johann Sebastian Bach", "Bach, Johann Sebastian", False),
        ("Gustav Holst", "Gustav Mahler", "Mahler, Gustav", False),
        ("...", "Gustav Mahler", "Mahler, Gustav", False),
    ],
)
def test_is_named_by_needs_every_word_of_the_name(
    name: str, mb_name: str, mb_sort_name: str, expected: bool
) -> None:
    """Every word of the name, case, accents and hyphens aside, is in the name or sort name."""
    mb_artist = MusicBrainzArtist(id=str(uuid4()), name=mb_name, sort_name=mb_sort_name)

    assert is_named_by(name, mb_artist) is expected


def _tchaikovsky(*aliases: MusicBrainzAlias) -> MusicBrainzArtist:
    """Return a MusicBrainz composer named in Cyrillic, with the given aliases."""
    return MusicBrainzArtist(
        id=str(uuid4()),
        name="Пётр Ильич Чайковский",
        sort_name="Чайковский, Пётр Ильич",
        aliases=list(aliases),
    )


def _classical_artist(name: str, mbid: str | None = None) -> Artist:
    """Return a classical library artist, with the given MusicBrainz id if any."""
    return Artist(
        item_id=str(uuid4().int % 10**6),
        provider="library",
        name=name,
        provider_mappings=set(),
        external_ids={(ExternalID.MB_ARTIST, mbid)} if mbid else set(),
    )


async def _merge_by_aliases(
    mass: MusicAssistant,
    mb_artist: MusicBrainzArtist,
    unconfirmed: list[Artist],
    confirmed: list[Artist],
    holders: Sequence[Artist] = (),
    namesakes: Sequence[MusicBrainzArtist] | None = None,
    rename: AsyncMock | None = None,
) -> AsyncMock:
    """
    Merge the library artists a composer's aliases name, returning the stand-in merge.

    :param mb_artist: The MusicBrainz composer.
    :param unconfirmed: The classical library artists without a MusicBrainz id.
    :param confirmed: The classical library artists with one.
    :param holders: The library artists holding the composer's MusicBrainz id.
    :param namesakes: The MusicBrainz artists a name search finds, the composer alone if None.
    :param rename: The stand-in rename of the library artists, a new one if None.
    """
    artists = mass.music.artists
    merge = AsyncMock()

    with (
        patch.object(
            artists,
            "get_library_composers",
            AsyncMock(side_effect=lambda with_mbid=False: confirmed if with_mbid else unconfirmed),
        ),
        patch.object(artists, "update_item_in_library", AsyncMock()),
        patch.object(artists, "merge_library_artists_by_mbid", merge),
        patch.object(artists, "set_library_artist_name", rename or AsyncMock()),
        patch.object(
            artists, "get_library_items_by_external_id", AsyncMock(return_value=list(holders))
        ),
    ):
        ctrl = _controller(mass, [])
        musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
        musicbrainz.get_artists_by_name = AsyncMock(
            return_value=[mb_artist] if namesakes is None else list(namesakes)
        )
        await ctrl._merge_composer_by_aliases(mb_artist.id, mb_artist)
    return merge


def _tchaikovsky_alias() -> MusicBrainzAlias:
    """Return the English artist name alias of Tchaikovsky."""
    return MusicBrainzAlias(
        name="Pyotr Ilyich Tchaikovsky", sort_name="Tchaikovsky, Pyotr Ilyich", type="Artist name"
    )


async def test_artist_named_by_an_alias_takes_the_composer_id_and_merges(
    mass: MusicAssistant,
) -> None:
    """A classical artist without an id carrying an alias of the composer merges into it."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias())
    candidate = _classical_artist("Pyotr Ilyich Tchaikovsky")

    merge = await _merge_by_aliases(mass, mb_artist, [candidate], [])

    assert candidate.mbid == mb_artist.id
    merge.assert_awaited_once_with(int(candidate.item_id), mb_artist.id)


@pytest.mark.parametrize(
    ("name", "aliases"),
    [
        # only part of an alias, as a family name is
        ("Tchaikovsky", []),
        ("Tschaikowsky", [MusicBrainzAlias("Tschaikowsky", "Tschaikowsky", type="Search hint")]),
    ],
)
async def test_name_that_is_no_name_of_the_composer_is_not_merged(
    mass: MusicAssistant, name: str, aliases: list[MusicBrainzAlias]
) -> None:
    """A name that is only part of an alias, or only a search hint, merges nothing."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias(), *aliases)
    candidate = _classical_artist(name)

    merge = await _merge_by_aliases(mass, mb_artist, [candidate], [])

    assert candidate.mbid is None
    merge.assert_not_awaited()


async def test_name_another_library_composer_carries_is_not_merged(mass: MusicAssistant) -> None:
    """A library artist whose name another confirmed composer also carries is left apart."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias())
    candidate = _classical_artist("Pyotr Ilyich Tchaikovsky")
    other = _classical_artist("Pyotr Ilyich Tchaikovsky Jr", str(uuid4()))

    merge = await _merge_by_aliases(mass, mb_artist, [candidate], [other])

    assert candidate.mbid is None
    merge.assert_not_awaited()


@pytest.mark.parametrize("namesake_type", [None, "Person"])
async def test_name_another_musicbrainz_artist_carries_is_not_merged(
    mass: MusicAssistant, namesake_type: str | None
) -> None:
    """A name that another MusicBrainz artist carries too, as father and son can, is left apart."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias())
    namesake = MusicBrainzArtist(
        id=str(uuid4()),
        name="Pyotr Ilyich Tchaikovsky",
        sort_name="Tchaikovsky, Pyotr Ilyich",
        type=namesake_type,
    )
    candidate = _classical_artist("Pyotr Ilyich Tchaikovsky")

    merge = await _merge_by_aliases(
        mass, mb_artist, [candidate], [], namesakes=[mb_artist, namesake]
    )

    assert candidate.mbid is None
    merge.assert_not_awaited()


async def test_name_a_musicbrainz_group_carries_too_is_merged(mass: MusicAssistant) -> None:
    """A band of the same name is no composer, so it does not keep the name apart."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias())
    band = MusicBrainzArtist(
        id=str(uuid4()),
        name="Pyotr Ilyich Tchaikovsky",
        sort_name="Pyotr Ilyich Tchaikovsky",
        type="Group",
    )
    candidate = _classical_artist("Pyotr Ilyich Tchaikovsky")

    merge = await _merge_by_aliases(mass, mb_artist, [candidate], [], namesakes=[mb_artist, band])

    assert candidate.mbid == mb_artist.id
    merge.assert_awaited_once_with(int(candidate.item_id), mb_artist.id)


async def test_artists_sharing_the_id_merge_whatever_their_names(mass: MusicAssistant) -> None:
    """Artists already holding the composer's id merge, also by a name that is no composer name."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias())
    holders = [
        _classical_artist("Tchaikovsky", mb_artist.id),
        _classical_artist("Pyotr Ilyich Tchaikovsky", mb_artist.id),
    ]

    merge = await _merge_by_aliases(mass, mb_artist, [], holders, holders, namesakes=[])

    merge.assert_awaited_once_with(int(holders[0].item_id), mb_artist.id)


async def test_artists_sharing_the_id_merge_with_the_artists_carrying_its_name(
    mass: MusicAssistant,
) -> None:
    """A holder by another name leaves the join of a library artist carrying an alias intact."""
    mb_artist = _tchaikovsky(_tchaikovsky_alias())
    holders = [
        _classical_artist("Пётр Ильич Чайковский", mb_artist.id),
        _classical_artist("Tchaikovsky", mb_artist.id),
    ]
    candidate = _classical_artist("Pyotr Ilyich Tchaikovsky")

    merge = await _merge_by_aliases(mass, mb_artist, [candidate], holders, holders)

    assert candidate.mbid == mb_artist.id
    merge.assert_awaited_once_with(int(candidate.item_id), mb_artist.id)


@pytest.mark.parametrize(
    ("mb_artist", "expected"),
    [
        (
            MusicBrainzArtist(id="ravel", name="Maurice Ravel", sort_name="Ravel, Maurice"),
            ("Maurice Ravel", "Ravel, Maurice"),
        ),
        (
            MusicBrainzArtist(
                id="saint-saens",
                name="Camille Saint\u2010Saëns",
                sort_name="Saint\u2010Saëns, Camille",
            ),
            ("Camille Saint-Saëns", "Saint-Saëns, Camille"),
        ),
        (
            MusicBrainzArtist(
                id="prokofiev",
                name="Сергей Сергеевич Прокофьев",
                sort_name="Prokofiev, Sergei Sergeyevich",
                aliases=[
                    MusicBrainzAlias("Sergej Prokofjew", "Prokofjew, Sergej", locale="de"),
                    MusicBrainzAlias(
                        "Sergei Prokofiev", "Prokofiev, Sergei", locale="en", primary=True
                    ),
                ],
            ),
            ("Sergei Prokofiev", "Prokofiev, Sergei"),
        ),
        (
            MusicBrainzArtist(
                id="tchaikovsky",
                name="Пётр Ильич Чайковский",
                sort_name="Чайковский, Пётр Ильич",
                aliases=[
                    MusicBrainzAlias(
                        "Pyotr Ilyich Tchaikovsky", "", locale="en", type="Artist name"
                    ),
                ],
            ),
            ("Pyotr Ilyich Tchaikovsky", "pyotr ilyich tchaikovsky"),
        ),
        (
            MusicBrainzArtist(
                id="prokofiev",
                name="Сергей Сергеевич Прокофьев",
                sort_name="Prokofiev, Sergei Sergeyevich",
                aliases=[MusicBrainzAlias("Sergei Prokofiev", "", locale="en", primary=True)],
            ),
            ("Sergei Prokofiev", "Prokofiev, Sergei Sergeyevich"),
        ),
        (_tchaikovsky(_tchaikovsky_alias()), None),
        (
            MusicBrainzArtist(
                id="strauss-ii",
                name="Johann Strauss",
                sort_name="Strauss, Johann",
                aliases=[
                    MusicBrainzAlias(
                        "Johann Strauss II", "Strauss, Johann, II", locale="en", primary=True
                    )
                ],
            ),
            ("Johann Strauss II", "Strauss, Johann, II"),
        ),
        (
            MusicBrainzArtist(
                id="strauss-i",
                name="Johann Strauss",
                sort_name="Strauss, Johann",
                aliases=[MusicBrainzAlias("Johann Strauss I", "", locale="en", primary=True)],
            ),
            ("Johann Strauss I", "Strauss, Johann"),
        ),
        (
            MusicBrainzArtist(
                id="chopin",
                name="Fryderyk Chopin",
                sort_name="Chopin, Fryderyk",
                aliases=[
                    MusicBrainzAlias(
                        "Frédéric Chopin", "Chopin, Frédéric", locale="en", primary=True
                    )
                ],
            ),
            ("Fryderyk Chopin", "Chopin, Fryderyk"),
        ),
    ],
    ids=[
        "latin",
        "plain hyphen",
        "primary english alias",
        "english artist name alias",
        "latin main sort name",
        "no english alias",
        "english alias extending the main name",
        "extending alias without sort name",
        "english alias not extending the main name",
    ],
)
def test_musicbrainz_name_is_in_latin_script(
    mb_artist: MusicBrainzArtist, expected: tuple[str, str] | None
) -> None:
    """A composer's MusicBrainz name is its main name, else its English alias, in Latin script."""
    assert musicbrainz_name(mb_artist) == expected


def _tchaikovsky_in(*locales: tuple[str, str, str]) -> MusicBrainzArtist:
    """Return Tchaikovsky with a primary alias of the given name and sort name in each locale."""
    return _tchaikovsky(
        *(
            MusicBrainzAlias(name, sort_name, locale=x, primary=True)
            for x, name, sort_name in locales
        )
    )


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("de", ("Peter Tschaikowsky", "Tschaikowsky, Peter")),
        ("fr", ("Pyotr Ilyich Tchaikovsky", "Tchaikovsky, Pyotr Ilyich")),
        ("ja", ("Pyotr Ilyich Tchaikovsky", "Tchaikovsky, Pyotr Ilyich")),
    ],
    ids=["alias in the language", "no alias in the language", "alias in another script"],
)
def test_musicbrainz_name_is_in_the_language_given_first(
    language: str, expected: tuple[str, str]
) -> None:
    """A composer's name is its primary alias in the given language, else in English."""
    mb_artist = _tchaikovsky_in(
        ("ja", "ピョートル・チャイコフスキー", "チャイコフスキー"),
        ("de", "Peter Tschaikowsky", "Tschaikowsky, Peter"),
        ("en", "Pyotr Ilyich Tchaikovsky", "Tchaikovsky, Pyotr Ilyich"),
    )

    assert musicbrainz_name(mb_artist, language) == expected


async def test_confirmed_composer_takes_its_name_in_the_metadata_language(
    mass: MusicAssistant,
) -> None:
    """A confirmed composer is renamed after its MusicBrainz name in the user's language."""
    mb_artist = _tchaikovsky_in(
        ("de", "Peter Tschaikowsky", "Tschaikowsky, Peter"),
        ("en", "Pyotr Ilyich Tchaikovsky", "Tchaikovsky, Pyotr Ilyich"),
    )
    holder = _classical_artist("Пётр Ильич Чайковский", mb_artist.id)
    rename = AsyncMock()

    with patch.object(
        MetaDataController, "preferred_language", new_callable=PropertyMock, return_value="de"
    ):
        await _merge_by_aliases(mass, mb_artist, [], [holder], [holder], rename=rename)

    rename.assert_awaited_once_with(
        int(holder.item_id), "Peter Tschaikowsky", "Tschaikowsky, Peter"
    )


async def test_confirmed_composer_takes_its_musicbrainz_name(mass: MusicAssistant) -> None:
    """A composer confirmed on MusicBrainz is renamed after it, its holders joined first."""
    mb_artist = _tchaikovsky(
        MusicBrainzAlias(
            "Pyotr Ilyich Tchaikovsky",
            "Tchaikovsky, Pyotr Ilyich",
            locale="en",
            primary=True,
            type="Artist name",
        )
    )
    holder = _classical_artist("Пётр Ильич Чайковский", mb_artist.id)
    rename = AsyncMock()

    await _merge_by_aliases(mass, mb_artist, [], [holder], [holder], rename=rename)

    rename.assert_awaited_once_with(
        int(holder.item_id), "Pyotr Ilyich Tchaikovsky", "Tchaikovsky, Pyotr Ilyich"
    )
    assert (holder.name, holder.sort_name) == (
        "Pyotr Ilyich Tchaikovsky",
        "Tchaikovsky, Pyotr Ilyich",
    )


async def test_composers_sharing_the_id_are_joined_under_the_musicbrainz_name(
    mass: MusicAssistant,
) -> None:
    """Two library artists holding a composer's id are joined and stored by its MusicBrainz name."""
    token = _token()
    mb_artist = MusicBrainzArtist(
        id=str(uuid4()), name=f"Maurice Ravel {token}", sort_name=f"Ravel {token}, Maurice"
    )
    oldest = await _add_classical_composer(mass, f"Ravel {token}", mb_artist.id)
    later = await _add_classical_composer(mass, f"M. Ravel {token}")
    later.mbid = mb_artist.id
    await mass.music.artists.update_item_in_library(later.item_id, later)
    assert later.item_id != oldest.item_id
    ctrl = _controller(mass, [], [mb_artist])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[])

    await ctrl._merge_composer_by_aliases(mb_artist.id)

    holders = await mass.music.artists.get_library_items_by_external_id(
        mb_artist.id, ExternalID.MB_ARTIST, limit=None
    )
    assert [(x.item_id, x.name, x.sort_name) for x in holders] == [
        (oldest.item_id, mb_artist.name, mb_artist.sort_name)
    ]
    rows = await mass.music.database.get_rows_from_query(
        "SELECT search_name, search_sort_name FROM artists WHERE item_id = :item_id",
        {"item_id": int(oldest.item_id)},
    )
    assert (rows[0]["search_name"], rows[0]["search_sort_name"]) == (
        create_safe_string(mb_artist.name, True, True),
        create_safe_string(mb_artist.sort_name, True, True),
    )


async def test_composer_identified_by_the_artist_step_takes_its_musicbrainz_name(
    mass: MusicAssistant,
) -> None:
    """A composer holding an id from outside the composer steps is stored by its English name."""
    token = _token()
    mb_artist = MusicBrainzArtist(
        id=str(uuid4()),
        name=f"Антон Григорьевич Рубинштейн {token}",
        sort_name=f"Рубинштейн {token}, Антон Григорьевич",
        aliases=[
            MusicBrainzAlias(
                f"Anton Rubinstein {token}", f"Rubinstein {token}, Anton", locale="en", primary=True
            )
        ],
    )
    composer = await _add_classical_composer(mass, f"Rubenstein {token}", mb_artist.id)
    assert composer.is_classical
    ctrl = _controller(mass, [], [mb_artist])

    with patch.object(
        MetaDataController, "link_providers_via_musicbrainz", new_callable=PropertyMock
    ) as link_providers:
        link_providers.return_value = False
        await ctrl.link_item_to_musicbrainz(composer)

    stored = await mass.music.artists.get_library_item(composer.item_id)
    assert (stored.name, stored.sort_name) == (
        f"Anton Rubinstein {token}",
        f"Rubinstein {token}, Anton",
    )


async def test_several_composers_are_not_guessed(mass: MusicAssistant) -> None:
    """With two composers on the work, none of them gets MusicBrainz's composer id."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    _unique_composer(works["symphony_5"], token)
    composers = [
        await _add_artist(mass, f"Beethoven {token}"),
        await _add_artist(mass, f"Ries {token}"),
    ]
    work = await _add_work(mass, f"Symphony no. 5 {token}", composers)
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert all(x.mbid is None for x in stored.composers)
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


async def test_library_composers_are_added_to_a_work_without_them_and_its_tracks(
    mass: MusicAssistant,
) -> None:
    """A work without composers gets MusicBrainz's, credited on its tracks without a composer."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    movement_id = works["symphony_5_movement_1"].id
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, f"L. van Beethoven {token}", beethoven.id)
    arranger = await _add_artist(mass, f"Liszt {token}")
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    bare = await _add_track(mass, work, [movement_id])
    credited = await _add_track(mass, work, [movement_id], arranger)
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [composer.item_id]
    assert _composers(await mass.music.tracks.get_library_item(bare.item_id)) == [composer.name]
    assert _composers(await mass.music.tracks.get_library_item(credited.item_id)) == [arranger.name]
    assert _scheduled(ctrl) == [composer.name]


async def test_work_musicbrainz_knows_is_not_given_its_tracks_composers(
    mass: MusicAssistant,
) -> None:
    """A work with a MusicBrainz work id, its own or a track's, waits for MusicBrainz's composers."""
    token = _token()
    composer = await _add_artist(mass, f"Composer {token}")
    identified = await _add_work(mass, f"Identified {token}", mbid=str(uuid4()))
    by_track = await _add_work(mass, f"By track {token}")

    await _add_track(mass, identified, composer=composer)
    await _add_track(mass, by_track, [str(uuid4())])
    await _add_track(mass, by_track, composer=composer)

    assert not (await mass.music.works.get_library_item(identified.item_id)).composers
    assert not (await mass.music.works.get_library_item(by_track.item_id)).composers


async def test_composer_without_a_musicbrainz_id_is_found_by_its_exact_name(
    mass: MusicAssistant,
) -> None:
    """A library artist of the MusicBrainz composer's exact name composes the work as it is."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, beethoven.name)
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert [(x.item_id, x.mbid) for x in stored.composers] == [(composer.item_id, None)]
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


async def test_composer_the_library_lacks_is_not_added(mass: MusicAssistant) -> None:
    """A MusicBrainz composer the library holds no artist for leaves the work without one."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values())

    stored = await _enrich(ctrl, work)

    assert stored.mbid == works["symphony_5"].id
    assert stored.composers == []
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert _composers(await mass.music.tracks.get_library_item(track.item_id)) == []
    assert not await mass.music.artists.get_library_items_by_external_id(
        beethoven.id, ExternalID.MB_ARTIST, limit=None
    )
    assert not await mass.music.artists.get_library_items_by_query(search=token)


async def test_composer_the_library_lacks_is_added_through_its_provider_link(
    mass: MusicAssistant,
) -> None:
    """A MusicBrainz composer linked to a loaded music provider is added and credited."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    spotify_id = uuid4().hex[:22]
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(beethoven, spotify_id)])

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    assert [(x.name, x.mbid) for x in stored.composers] == [(beethoven.name, beethoven.id)]
    composer = await mass.music.artists.get_library_item(stored.composers[0].item_id)
    assert [(x.provider_instance, x.item_id, x.in_library) for x in composer.provider_mappings] == [
        ("spotify", spotify_id, False)
    ]
    assert _composers(await mass.music.tracks.get_library_item(track.item_id)) == [beethoven.name]
    assert _scheduled(ctrl) == [beethoven.name]


async def test_composer_without_a_loaded_provider_link_is_not_added(mass: MusicAssistant) -> None:
    """A MusicBrainz composer linked to no loaded music provider is not added."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(beethoven, uuid4().hex[:22])])

    stored = await _enrich(ctrl, work)

    assert stored.composers == []
    assert not await mass.music.artists.get_library_items_by_query(search=token)


async def test_composer_without_a_loaded_provider_link_tells_where_to_add_one(
    mass: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A composer left out for want of a link is logged with its MusicBrainz page."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(beethoven, uuid4().hex[:22])])
    ctrl.logger = logging.getLogger("test_work_enrichment")
    caplog.set_level(logging.DEBUG, logger="test_work_enrichment")

    def _instances(domain: str, **_kwargs: Any) -> list[Mock]:
        if domain not in ("qobuz", "tidal"):
            return []
        instance = Mock(available=True, instance_id=domain)
        instance.manifest.name = domain.capitalize()
        return [instance]

    with patch.object(mass.music, "get_provider_instances", Mock(side_effect=_instances)):
        await _enrich(ctrl, work)

    assert (
        f'Artist "{beethoven.name}" has no link to Tidal, Qobuz on MusicBrainz, so it is not '
        f"added; add one at https://musicbrainz.org/artist/{beethoven.id}"
    ) in caplog.messages


async def test_library_artist_on_the_linked_provider_artist_composes_the_work(
    mass: MusicAssistant,
) -> None:
    """A library artist mapped to the provider artist MusicBrainz links to is not duplicated."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    spotify_id = uuid4().hex[:22]
    artist = await _add_artist(mass, f"L. van Beethoven {token}")
    await mass.music.artists.add_provider_mappings(
        artist.item_id,
        [
            ProviderMapping(
                item_id=spotify_id,
                provider_domain="spotify",
                provider_instance="spotify",
                in_library=False,
            )
        ],
    )
    work = await _add_work(mass, f"Symphony no. 5 {token}")
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(beethoven, spotify_id)])

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [artist.item_id]
    assert len(await mass.music.artists.get_library_items_by_query(search=token)) == 1


async def test_artist_of_the_name_holding_another_musicbrainz_id_is_no_match(
    mass: MusicAssistant,
) -> None:
    """A library artist of the same name but another MusicBrainz id is someone else."""
    token = _token()
    namesake = await _add_artist(mass, f"Engelbert Humperdinck {token}", str(uuid4()))
    mb_artist = MusicBrainzArtist(id=str(uuid4()), name=namesake.name, sort_name=namesake.name)

    assert await _controller(mass, [])._match_library_artist(mb_artist) is None


async def test_namesake_composer_is_replaced_by_the_musicbrainz_composer(
    mass: MusicAssistant,
) -> None:
    """A work and its tracks credited to a namesake of MusicBrainz's composer take the composer."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    composer = _unique_composer(works["symphony_5"], token)
    namesake_mbid = str(uuid4())
    namesake = await _add_artist(mass, composer.name, namesake_mbid)
    work = await _add_work(mass, f"Hansel and Gretel {token}", [namesake])
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id], namesake)
    other_work = await _add_work(mass, f"Release Me {token}", [namesake])
    other_track = await _add_track(mass, other_work, composer=namesake)
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(composer, uuid4().hex[:22])])

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    assert [x.mbid for x in stored.composers] == [composer.id]
    added_id = stored.composers[0].item_id
    assert added_id != namesake.item_id
    assert _composer_ids(await mass.music.tracks.get_library_item(track.item_id)) == [added_id]
    assert _composer_ids(await mass.music.tracks.get_library_item(other_track.item_id)) == [
        namesake.item_id
    ]
    kept = await mass.music.artists.get_library_item(namesake.item_id)
    assert (kept.name, kept.mbid) == (composer.name, namesake_mbid)


async def test_namesake_listed_as_track_artist_gives_way_to_the_musicbrainz_composer(
    mass: MusicAssistant,
) -> None:
    """A namesake credited as artist of the work's tracks leaves them, losing its period."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    composer = _unique_composer(works["symphony_5"], token)
    namesake = await _add_artist(mass, composer.name, str(uuid4()), period=Period.CONTEMPORARY)
    work = await _add_work(mass, f"Hansel and Gretel {token}", [namesake], work_type=WorkType.OPERA)
    track = await mass.music.tracks.add_item_to_library(
        Track(
            item_id=token,
            provider=PROVIDER_INSTANCE,
            name=f"Overture {token}",
            provider_mappings={_mapping()},
            artists=UniqueList([namesake]),
            credits=[Credit(artist=namesake, role=ArtistRole.COMPOSER)],
            work=ItemMapping.from_item(work),
            external_ids={(ExternalID.MB_WORK, works["symphony_5_movement_1"].id)},
        )
    )
    assert (await mass.music.artists.get_library_item(namesake.item_id)).is_classical
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(composer, uuid4().hex[:22])])

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    added_id = stored.composers[0].item_id
    stored_track = await mass.music.tracks.get_library_item(track.item_id)
    assert [x.item_id for x in stored_track.artists] == [added_id]
    assert _composer_ids(stored_track) == [added_id]
    kept = await mass.music.artists.get_library_item(namesake.item_id)
    assert (kept.is_classical, kept.period) == (False, None)


async def test_namesake_composer_stays_when_the_musicbrainz_composer_cannot_be_added(
    mass: MusicAssistant,
) -> None:
    """A namesake composer stays when the library cannot take MusicBrainz's composer."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    composer = _unique_composer(works["symphony_5"], token)
    namesake = await _add_artist(mass, composer.name, str(uuid4()))
    work = await _add_work(mass, f"Hansel and Gretel {token}", [namesake])
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id], namesake)
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(composer, uuid4().hex[:22])])

    stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [namesake.item_id]
    assert _composer_ids(await mass.music.tracks.get_library_item(track.item_id)) == [
        namesake.item_id
    ]


async def test_composer_holding_the_musicbrainz_id_is_kept(mass: MusicAssistant) -> None:
    """A composer holding the MusicBrainz composer's id stays on the work and its tracks."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    mb_composer = _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, mb_composer.name, mb_composer.id)
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    track = await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values(), [_linked_to_spotify(mb_composer, uuid4().hex[:22])])

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [composer.item_id]
    assert _composer_ids(await mass.music.tracks.get_library_item(track.item_id)) == [
        composer.item_id
    ]
    assert len(await mass.music.artists.get_library_items_by_query(search=token)) == 1


# --------------------------------------------------------------------------- #
#  composers identified by their own links                                     #
# --------------------------------------------------------------------------- #


async def _add_classical_composer(
    mass: MusicAssistant, name: str, mbid: str | None = None, **kwargs: Any
) -> Artist:
    """Store a library artist credited as composer on a classical track only."""
    composer = await _add_artist(mass, name, mbid, **kwargs)
    work = await _add_work(mass, f"Symphony {_token()}", [composer], work_type=WorkType.SYMPHONY)
    track = await _add_track(mass, work, composer=composer)
    assert (await mass.music.tracks.get_library_item(track.item_id)).is_classical
    return await mass.music.artists.get_library_item(composer.item_id)


async def _due_composers(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library artists the composers phase would select now."""
    selected = await mass.music.artists.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_composers_to_identify_query()],
        extra_query_params={"stale": int(time() - REFRESH_INTERVAL)},
    )
    return {x.item_id for x in selected}


async def _identify_composer(
    ctrl: MetaDataController, composer: Artist, mb_artist: MusicBrainzArtist | None
) -> tuple[Artist, AsyncMock]:
    """
    Run the composers phase on a library artist, asserting no music provider is asked.

    Returns the stored artist and the stand-in alias join.

    :param ctrl: The controller to run the phase on.
    :param composer: The library artist.
    :param mb_artist: The MusicBrainz artist its streaming service links lead to, if any.
    """
    mass = ctrl.mass
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.resolve_artist = AsyncMock(return_value=mb_artist)
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[])
    alias_join = AsyncMock()
    provider, provider_item, top_tracks = Mock(), AsyncMock(), AsyncMock()
    with (
        patch.object(mass.music, "get_provider_instances", _spotify_instances()),
        patch.object(
            MetaDataController,
            "link_providers_via_musicbrainz",
            new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(ctrl, "_merge_composer_by_aliases", alias_join),
        patch.object(mass, "get_provider", Mock(return_value=provider)),
        patch.object(mass.music.artists, "get_provider_item", provider_item),
        patch.object(mass.music.artists, "top_tracks", top_tracks),
    ):
        assert await ctrl._identify_composer(composer)
    musicbrainz.resolve_artist.assert_awaited_once_with(composer, [], [])
    assert provider.mock_calls == []
    provider_item.assert_not_awaited()
    top_tracks.assert_not_awaited()
    return await mass.music.artists.get_library_item(composer.item_id), alias_join


async def test_composers_phase_selects_unidentified_composers_of_classical_tracks(
    mass: MusicAssistant,
) -> None:
    """A composer without an id is due until looked up; an identified or performing one is not."""
    stale = int(time()) - REFRESH_INTERVAL - 1
    never = await _add_classical_composer(mass, f"Haydn {_token()}")
    stale_unknown = await _add_classical_composer(
        mass, f"Liszt {_token()}", metadata=MediaItemMetadata(last_musicbrainz_lookup=stale)
    )
    fresh = await _add_classical_composer(
        mass, f"Rossini {_token()}", metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time()))
    )
    identified = await _add_classical_composer(mass, f"Fauré {_token()}", mbid=str(uuid4()))
    work = await _add_work(mass, f"Symphony {_token()}", work_type=WorkType.SYMPHONY)
    track = await _add_track(mass, work)
    performer = track.artists[0]

    due = await _due_composers(mass)

    assert {never.item_id, stale_unknown.item_id} <= due
    assert not {fresh.item_id, identified.item_id, performer.item_id} & due


async def test_composer_is_identified_by_its_own_streaming_link(mass: MusicAssistant) -> None:
    """The artist's own links give its id, links, life dates, country and period, then the join."""
    token = _token()
    composer = await _add_classical_composer(mass, f"Shostakovich {token}")
    spotify_id = uuid4().hex[:22]
    mb_artist = _linked_to_spotify(
        MusicBrainzArtist(id=str(uuid4()), name=composer.name, sort_name=composer.name), spotify_id
    )
    mb_artist.type = "Person"
    mb_artist.life_span = MusicBrainzLifeSpan(begin="1906-09-25", end="1975-08-09", ended=True)
    mb_artist.country = "RU"
    ctrl = _controller(mass, [], [mb_artist])

    stored, alias_join = await _identify_composer(ctrl, composer, mb_artist)

    assert stored.mbid == mb_artist.id
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert ("spotify", spotify_id) in {
        (x.provider_instance, x.item_id) for x in stored.provider_mappings
    }
    assert stored.metadata.artist_entity_type == ArtistEntityType.PERSON
    assert stored.metadata.life_span == LifeSpan(begin="1906-09-25", end="1975-08-09", ended=True)
    assert stored.metadata.country == "RU"
    assert stored.period == composer_period(1906, 1975)
    alias_join.assert_awaited_once_with(mb_artist.id, mb_artist)
    assert _scheduled(ctrl) == []


async def test_composer_its_links_do_not_identify_is_marked_looked_up(mass: MusicAssistant) -> None:
    """A composer MusicBrainz does not know by its links is only marked as looked up."""
    composer = await _add_classical_composer(mass, f"Mussorgsky {_token()}")
    ctrl = _controller(mass, [])

    stored, alias_join = await _identify_composer(ctrl, composer, None)

    assert stored.mbid is None
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert stored.provider_mappings == composer.provider_mappings
    alias_join.assert_not_awaited()
    assert _scheduled(ctrl) == []
    assert composer.item_id not in await _due_composers(mass)


async def test_placeholder_composer_is_marked_looked_up_without_a_lookup(
    mass: MusicAssistant,
) -> None:
    """A placeholder credited as composer is never looked up on MusicBrainz nor logged."""
    composer = await _add_classical_composer(mass, f"[traditional {_token()}]")
    ctrl = _controller(mass, [])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())

    assert await ctrl._identify_composer(composer)

    stored = await mass.music.artists.get_library_item(composer.item_id)
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert musicbrainz.mock_calls == []
    assert cast("Mock", ctrl.logger).mock_calls == []
    assert composer.item_id not in await _due_composers(mass)


def _albinoni(token: str) -> MusicBrainzArtist:
    """Return a MusicBrainz composer carrying a shorter spelling of its name as an alias."""
    return MusicBrainzArtist(
        id=str(uuid4()),
        name=f"Tomaso Giovanni Albinoni {token}",
        sort_name=f"Albinoni, Tomaso Giovanni {token}",
        type="Person",
        aliases=[
            MusicBrainzAlias(
                f"Tomaso Albinoni {token}", f"Albinoni, Tomaso {token}", type="Artist name"
            )
        ],
    )


def _tidal_mapping(tidal_id: str) -> ProviderMapping:
    """Create a Tidal provider mapping."""
    return ProviderMapping(item_id=tidal_id, provider_domain="tidal", provider_instance="tidal")


async def _identify_by_name(
    mass: MusicAssistant, composer: Artist, persons: Sequence[MusicBrainzArtist]
) -> list[str]:
    """
    Run the composers phase on a composer its links do not identify, returning the logged lines.

    :param composer: The library composer.
    :param persons: The MusicBrainz artists a name search finds.
    """
    ctrl = _controller(mass, [], persons)
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.resolve_artist = AsyncMock(return_value=None)
    musicbrainz.get_artists_by_name = AsyncMock(return_value=list(persons))
    assert await ctrl._identify_composer(composer)
    logger = cast("Mock", ctrl.logger)
    return [call.args[0] % call.args[1:] for call in logger.debug.call_args_list] + [
        call.args[1] % call.args[2:] for call in logger.log.call_args_list
    ]


async def test_composer_takes_the_id_of_the_one_person_of_its_name(mass: MusicAssistant) -> None:
    """A composer named like the one person a library composer holds the id of merges into it."""
    token = _token()
    albinoni = _albinoni(token)
    confirmed = await _add_classical_composer(mass, albinoni.name, albinoni.id)
    composer = await _add_classical_composer(mass, f"Tomaso Albinoni {token}")

    lines = await _identify_by_name(mass, composer, [albinoni])

    holders = await mass.music.artists.get_library_items_by_external_id(
        albinoni.id, ExternalID.MB_ARTIST, limit=None
    )
    assert [x.item_id for x in holders] == [confirmed.item_id]
    with pytest.raises(MediaNotFoundError):
        await mass.music.artists.get_library_item(composer.item_id)
    assert not [x for x in lines if "is not on MusicBrainz" in x]


async def test_composer_named_like_two_persons_is_not_joined(mass: MusicAssistant) -> None:
    """A name two MusicBrainz persons carry leaves the composer apart, with a search link."""
    token = _token()
    albinoni = _albinoni(token)
    namesake = MusicBrainzArtist(
        id=str(uuid4()), name=f"Tomaso Albinoni {token}", sort_name="Albinoni", type="Person"
    )
    await _add_classical_composer(mass, albinoni.name, albinoni.id)
    composer = await _add_classical_composer(
        mass, f"Tomaso Albinoni {token}", provider_mappings={_tidal_mapping("1")}
    )

    lines = await _identify_by_name(mass, composer, [albinoni, namesake])

    assert (await mass.music.artists.get_library_item(composer.item_id)).mbid is None
    assert (
        f'Composer "Tomaso Albinoni {token}" is not on MusicBrainz through '
        "https://tidal.com/artist/1, add https://tidal.com/artist/1 to their "
        f"MusicBrainz page: https://musicbrainz.org/search?type=artist&query=Tomaso+Albinoni+{token}"
    ) in lines


async def test_composer_named_like_a_person_no_library_composer_holds_is_not_joined(
    mass: MusicAssistant,
) -> None:
    """A person no library composer holds gives no id, only a link to their MusicBrainz page."""
    token = _token()
    albinoni = _albinoni(token)
    composer = await _add_classical_composer(
        mass, f"Tomaso Albinoni {token}", provider_mappings={_tidal_mapping("2")}
    )

    lines = await _identify_by_name(mass, composer, [albinoni])

    assert (await mass.music.artists.get_library_item(composer.item_id)).mbid is None
    assert not await mass.music.artists.get_library_items_by_external_id(
        albinoni.id, ExternalID.MB_ARTIST, limit=None
    )
    assert (
        f'Composer "Tomaso Albinoni {token}" is not on MusicBrainz through '
        "https://tidal.com/artist/2, add https://tidal.com/artist/2 to their "
        f"MusicBrainz page: https://musicbrainz.org/artist/{albinoni.id}"
    ) in lines


async def test_composer_named_like_another_confirmed_composer_is_not_joined(
    mass: MusicAssistant,
) -> None:
    """A name another confirmed composer carries every word of leaves the composer apart."""
    token = _token()
    albinoni = _albinoni(token)
    await _add_classical_composer(mass, albinoni.name, albinoni.id)
    await _add_classical_composer(mass, f"Tomaso Albinoni {token} the Younger", str(uuid4()))
    composer = await _add_classical_composer(mass, f"Tomaso Albinoni {token}")

    await _identify_by_name(mass, composer, [albinoni])

    assert (await mass.music.artists.get_library_item(composer.item_id)).mbid is None


async def _add_joined_track(mass: MusicAssistant, name: str, joined: Artist) -> Track:
    """Store a library track crediting a joined name as composer and main artist."""
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id=_token(),
            provider=PROVIDER_INSTANCE,
            name=name,
            provider_mappings={_mapping()},
            artists=UniqueList([joined]),
            credits=[Credit(artist=joined, role=ArtistRole.COMPOSER)],
        )
    )


async def test_confirmed_composer_splits_the_joined_credits_of_stored_tracks(
    mass: MusicAssistant,
) -> None:
    """A composer confirmed after a track was stored replaces the joined name it starts."""
    token = _token()
    chopin = await _add_classical_composer(mass, f"Frédéric Chopin {token}")
    joined = await _add_artist(mass, f"Frédéric Chopin {token} & Sarah Ainsworth {token}")
    band = await _add_artist(mass, f"Frédéric Chopin {token} & Friends")
    nocturne = await _add_joined_track(mass, f"Chopin {token}: Nocturne No. 2", joined)
    party = await _add_joined_track(mass, "Party Mix", band)
    mb_chopin = MusicBrainzArtist(id=str(uuid4()), name=chopin.name, sort_name=chopin.name)
    chopin.mbid = mb_chopin.id
    await mass.music.artists.update_item_in_library(chopin.item_id, chopin)
    ctrl = _controller(mass, [], [mb_chopin])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[mb_chopin])

    await ctrl._merge_composer_by_aliases(mb_chopin.id)

    stored = await mass.music.tracks.get_library_item(nocturne.item_id)
    assert _composer_ids(stored) == [chopin.item_id]
    assert [x.item_id for x in stored.artists] == [chopin.item_id]
    untouched = await mass.music.tracks.get_library_item(party.item_id)
    assert _composer_ids(untouched) == [band.item_id]
    assert [x.item_id for x in untouched.artists] == [band.item_id]


async def _split_by_alias(
    mass: MusicAssistant, token: str, other: str | None = None
) -> tuple[Artist, Track]:
    """
    Confirm "Fryderyk Chopin" known as "Frédéric Chopin", return it and its joined track.

    :param token: The word unique to the test.
    :param other: The name of another confirmed composer added after the track, if any.
    """
    mb_chopin = MusicBrainzArtist(
        id=str(uuid4()),
        name=f"Fryderyk Chopin {token}",
        sort_name=f"Chopin {token}, Fryderyk",
        aliases=[
            MusicBrainzAlias(
                f"Frédéric Chopin {token}", f"Chopin {token}, Frédéric", type="Artist name"
            )
        ],
    )
    chopin = await _add_classical_composer(mass, mb_chopin.name, mb_chopin.id)
    joined = await _add_artist(mass, f"Frédéric Chopin {token} & Sarah Ainsworth {token}")
    nocturne = await _add_joined_track(mass, f"Chopin {token}: Nocturne No. 2", joined)
    if other:
        await _add_classical_composer(mass, other, str(uuid4()))
    ctrl = _controller(mass, [], [mb_chopin])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[mb_chopin])

    await ctrl._merge_composer_by_aliases(mb_chopin.id)

    return chopin, await mass.music.tracks.get_library_item(nocturne.item_id)


async def test_confirmed_composer_splits_joined_credits_naming_it_by_an_alias(
    mass: MusicAssistant,
) -> None:
    """A joined name starting with a MusicBrainz alias of the composer is split to it."""
    chopin, stored = await _split_by_alias(mass, _token())

    assert _composer_ids(stored) == [chopin.item_id]
    assert [x.item_id for x in stored.artists] == [chopin.item_id]


async def test_alias_split_leaves_a_name_another_confirmed_composer_carries(
    mass: MusicAssistant,
) -> None:
    """A joined name another confirmed composer carries every word of stays as it is."""
    token = _token()
    joined = f"Frédéric Chopin {token} & Sarah Ainsworth {token}"

    _chopin, stored = await _split_by_alias(mass, token, f"Frédéric Chopin {token} the Younger")

    assert _composers(stored) == [joined]
    assert [x.name for x in stored.artists] == [joined]


# --------------------------------------------------------------------------- #
#  classical performers identified by their own links                          #
# --------------------------------------------------------------------------- #


async def _add_classical_performer(
    mass: MusicAssistant,
    name: str,
    mbid: str | None = None,
    *,
    domain: str = "spotify",
    classical: bool = True,
    composer: bool = False,
    **kwargs: Any,
) -> Artist:
    """
    Store a library artist credited as conductor on a track, mapped to a provider of the domain.

    :param classical: Whether the track is a classical one.
    :param composer: Also credit the artist as composer of a classical track.
    """
    mapping = ProviderMapping(
        item_id=uuid4().hex[:22], provider_domain=domain, provider_instance=domain
    )
    performer = await _add_artist(mass, name, mbid, provider_mappings={mapping}, **kwargs)
    work_type = WorkType.SYMPHONY if classical else WorkType.OTHER
    work = await _add_work(mass, f"Symphony {_token()}", work_type=work_type)
    track = await _add_track(mass, work, composer=performer if composer else None)
    await mass.music.tracks.add_credits(
        int(track.item_id), [Credit(artist=performer, role=ArtistRole.CONDUCTOR)]
    )
    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert stored.is_classical is classical
    return await mass.music.artists.get_library_item(performer.item_id)


async def _due_performers(mass: MusicAssistant) -> set[str]:
    """Return the ids of the library artists the performers phase would select now."""
    selected = await mass.music.artists.get_library_items_by_query(
        limit=0,
        extra_query_parts=[_performers_to_identify_query()],
        extra_query_params={"stale": int(time() - ARTIST_LOOKUP_RETRY_INTERVAL)},
    )
    return {x.item_id for x in selected}


async def _identify_classical_performer(
    ctrl: MetaDataController, performer: Artist, mb_artist: MusicBrainzArtist | None
) -> Artist | None:
    """
    Run the performers phase on a library artist, asserting no music provider is asked.

    Returns the stored artist, None when it merged into another.

    :param ctrl: The controller to run the phase on.
    :param performer: The library artist.
    :param mb_artist: The MusicBrainz artist its streaming service links lead to, if any.
    """
    mass = ctrl.mass
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.resolve_artist = AsyncMock(return_value=mb_artist)
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[])
    provider, provider_item, top_tracks = Mock(), AsyncMock(), AsyncMock()
    with (
        patch.object(mass.music, "get_provider_instances", _spotify_instances()),
        patch.object(
            MetaDataController,
            "link_providers_via_musicbrainz",
            new_callable=PropertyMock,
            return_value=True,
        ),
        patch.object(mass, "get_provider", Mock(return_value=provider)),
        patch.object(mass.music.artists, "get_provider_item", provider_item),
        patch.object(mass.music.artists, "top_tracks", top_tracks),
    ):
        assert await ctrl._identify_classical_performer(performer)
    musicbrainz.resolve_artist.assert_awaited_once_with(performer, [], [])
    musicbrainz.get_artists_by_name.assert_not_awaited()
    assert provider.mock_calls == []
    provider_item.assert_not_awaited()
    top_tracks.assert_not_awaited()
    try:
        return await mass.music.artists.get_library_item(performer.item_id)
    except MediaNotFoundError:
        return None


async def test_performers_phase_selects_unidentified_performers_of_classical_tracks(
    mass: MusicAssistant,
) -> None:
    """A performer without an id that its own links may identify is due until looked up."""
    stale = int(time()) - ARTIST_LOOKUP_RETRY_INTERVAL - 1
    never = await _add_classical_performer(mass, f"Abbado {_token()}")
    stale_unknown = await _add_classical_performer(
        mass, f"Karajan {_token()}", metadata=MediaItemMetadata(last_musicbrainz_lookup=stale)
    )
    fresh = await _add_classical_performer(
        mass, f"Solti {_token()}", metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time()))
    )
    identified = await _add_classical_performer(mass, f"Rattle {_token()}", str(uuid4()))
    composing = await _add_classical_performer(mass, f"Bernstein {_token()}", composer=True)
    unlinked = await _add_classical_performer(mass, f"Jansons {_token()}", domain="soundcloud")
    not_classical = await _add_classical_performer(mass, f"Zimmer {_token()}", classical=False)

    due = await _due_performers(mass)

    assert {never.item_id, stale_unknown.item_id} <= due
    assert (
        not {
            fresh.item_id,
            identified.item_id,
            composing.item_id,
            unlinked.item_id,
            not_classical.item_id,
        }
        & due
    )


async def test_performer_is_identified_by_its_own_streaming_link(mass: MusicAssistant) -> None:
    """The artist's own links give its id and life dates, and it merges into the id's holder."""
    token = _token()
    mb_artist = MusicBrainzArtist(
        id=str(uuid4()), name=f"Claudio Abbado {token}", sort_name=f"Abbado, Claudio {token}"
    )
    mb_artist.type = "Person"
    mb_artist.life_span = MusicBrainzLifeSpan(begin="1933-06-26", end="2014-01-20", ended=True)
    holder = await _add_artist(mass, f"C. Abbado {token}", mb_artist.id)
    performer = await _add_classical_performer(mass, f"Claudio Abbado {token}")
    ctrl = _controller(mass, [], [mb_artist])

    assert await _identify_classical_performer(ctrl, performer, mb_artist) is None

    stored = await mass.music.artists.get_library_item(holder.item_id)
    assert stored.metadata.life_span == LifeSpan(begin="1933-06-26", end="2014-01-20", ended=True)
    holders = await mass.music.artists.get_library_items_by_external_id(
        mb_artist.id, ExternalID.MB_ARTIST, limit=None
    )
    assert [x.item_id for x in holders] == [holder.item_id]
    assert performer.item_id not in await _due_performers(mass)


async def test_performer_its_links_do_not_identify_is_marked_looked_up(
    mass: MusicAssistant,
) -> None:
    """A performer MusicBrainz does not know by its links is only marked as looked up."""
    performer = await _add_classical_performer(mass, f"Gergiev {_token()}")
    ctrl = _controller(mass, [])

    stored = await _identify_classical_performer(ctrl, performer, None)

    assert stored is not None
    assert stored.mbid is None
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert stored.provider_mappings == performer.provider_mappings
    assert performer.item_id not in await _due_performers(mass)


async def test_placeholder_performer_is_marked_looked_up_without_a_lookup(
    mass: MusicAssistant,
) -> None:
    """A placeholder credited as performer is never looked up on MusicBrainz."""
    performer = await _add_classical_performer(mass, f"[unknown {_token()}]")
    ctrl = _controller(mass, [])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())

    assert await ctrl._identify_classical_performer(performer)

    stored = await mass.music.artists.get_library_item(performer.item_id)
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert musicbrainz.mock_calls == []
    assert performer.item_id not in await _due_performers(mass)


# --------------------------------------------------------------------------- #
#  composer periods                                                            #
# --------------------------------------------------------------------------- #


async def test_composer_with_life_dates_gets_a_period_right_away(mass: MusicAssistant) -> None:
    """A classical composer whose life dates are stored gets a period without a refresh."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, f"Beethoven {token}", beethoven.id, metadata=_BEETHOVEN)
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    await _enrich(ctrl, work)

    assert (await mass.music.artists.get_library_item(composer.item_id)).period == (
        Period.CLASSICAL
    )
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


async def test_composer_with_a_period_is_left_alone(mass: MusicAssistant) -> None:
    """A composer who already has a period keeps it and is not refreshed."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, f"Beethoven {token}", beethoven.id, period=Period.ROMANTIC)
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    await _enrich(ctrl, work)

    assert (await mass.music.artists.get_library_item(composer.item_id)).period == (Period.ROMANTIC)
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


async def test_identified_composer_without_life_dates_gets_a_normal_refresh(
    mass: MusicAssistant,
) -> None:
    """A composer who already had a MusicBrainz id but no life dates gets an unforced refresh."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    token = _token()
    beethoven = _unique_composer(works["symphony_5"], token)
    composer = await _add_artist(mass, f"Beethoven {token}", beethoven.id)
    work = await _add_work(mass, f"Symphony no. 5 {token}", [composer])
    await _add_track(mass, work, [works["symphony_5_movement_1"].id], composer)
    ctrl = _controller(mass, works.values())

    await _enrich(ctrl, work)

    assert _scheduled(ctrl) == [f"Beethoven {token}"]


async def test_composer_period_needs_a_classical_composer(mass: MusicAssistant) -> None:
    """Only a classical artist credited as a composer gets a period."""
    token = _token()
    classical_work = await _add_work(mass, f"Messiah {token}", work_type=WorkType.ORATORIO)
    plain_work = await _add_work(mass, f"Song {token}")
    composer = await _add_artist(mass, f"Händel {token}", metadata=_HANDEL)
    songwriter = await _add_artist(mass, f"Songwriter {token}", metadata=_HANDEL)
    performer = (await _add_track(mass, classical_work, composer=composer)).artists[0]
    await _add_track(mass, plain_work, composer=songwriter)
    ctrl = _controller(mass, [])
    artists = mass.music.artists
    performer = await artists.get_library_item(performer.item_id)
    performer.metadata = _HANDEL

    assert await ctrl._composer_period(await artists.get_library_item(composer.item_id)) == (
        Period.BAROQUE
    )
    assert await ctrl._composer_period(await artists.get_library_item(songwriter.item_id)) is None
    assert await ctrl._composer_period(performer) is None


# --------------------------------------------------------------------------- #
#  selection                                                                   #
# --------------------------------------------------------------------------- #


async def test_works_query_selects_works_with_a_musicbrainz_id_not_looked_up(
    mass: MusicAssistant,
) -> None:
    """A work is due while not looked up and it or one of its tracks carries a work id."""
    token = _token()
    own_id = await _add_work(mass, f"Own {token}", mbid=str(uuid4()))
    track_id = await _add_work(mass, f"Track {token}")
    await _add_track(mass, track_id, [str(uuid4())])
    without_id = await _add_work(mass, f"Without {token}")
    await _add_track(mass, without_id)
    looked_up = await _add_work(
        mass,
        f"Looked up {token}",
        mbid=str(uuid4()),
        metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time())),
    )
    created = {x.item_id for x in (own_id, track_id, without_id, looked_up)}

    assert await _due_works(mass) & created == {own_id.item_id, track_id.item_id}


async def test_works_query_retries_an_unidentified_work_once_stale(mass: MusicAssistant) -> None:
    """A work MusicBrainz did not identify is due again after the refresh interval."""
    token = _token()
    long_ago = int(time() - REFRESH_INTERVAL) - 60
    stale = await _add_work(
        mass, f"Stale {token}", metadata=MediaItemMetadata(last_musicbrainz_lookup=long_ago)
    )
    await _add_track(mass, stale, [str(uuid4())])
    recent = await _add_work(
        mass, f"Recent {token}", metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time()))
    )
    await _add_track(mass, recent, [str(uuid4())])
    identified = await _add_work(
        mass,
        f"Identified {token}",
        mbid=str(uuid4()),
        metadata=MediaItemMetadata(last_musicbrainz_lookup=long_ago),
    )
    created = {x.item_id for x in (stale, recent, identified)}

    assert await _due_works(mass) & created == {stale.item_id}


# TEMP(classical 5b catch-up): remove before release
async def test_identified_works_without_composers_are_looked_up_once_more(
    mass: MusicAssistant,
) -> None:
    """The identified works without composers are queued for another lookup, once."""
    token = _token()
    looked_up = MediaItemMetadata(last_musicbrainz_lookup=int(time()))
    composer = await _add_artist(mass, f"Composer {token}")
    without = await _add_work(mass, f"Without {token}", mbid=str(uuid4()), metadata=looked_up)
    composed = await _add_work(
        mass, f"Composed {token}", [composer], mbid=str(uuid4()), metadata=looked_up
    )
    ctrl = _controller(mass, [])
    created = {without.item_id, composed.item_id}
    assert await _due_works(mass) & created == set()

    await ctrl._queue_composerless_works_catch_up()

    assert await _due_works(mass) & created == {without.item_id}
    await _enrich(ctrl, without)
    await ctrl._queue_composerless_works_catch_up()
    assert await _due_works(mass) & created == set()


# TEMP(classical 5b work join): remove before release
async def test_works_that_are_the_same_work_are_joined_once(music_mass: MusicAssistant) -> None:
    """The stored works and parts that are one work are joined into it, once."""
    mass = music_mass
    works = mass.music.works
    bach = await _add_artist(mass, "Bach")
    tchaikovsky = await _add_artist(mass, "Tchaikovsky")

    async def create(name: str, composer: Artist, parent: Work | None = None) -> Work:
        # stored the way an earlier version added them, apart from each other
        work = await works.create_library_item(
            Work(
                item_id=name,
                provider=PROVIDER_INSTANCE,
                name=name,
                provider_mappings=set(),
                composers=UniqueList([composer]),
                catalog_numbers=parse_catalog_numbers(name),
                parent_work=ItemMapping.from_item(parent) if parent else None,
            )
        )
        await _add_track(mass, work)
        return work

    english = await create("St. Matthew Passion, BWV 244", bach)
    german = await create("Matthäus-Passion, BWV 244", bach)
    part_one = await create("St. Matthew Passion, BWV 244, Part I", bach, english)
    await create("Matthäus-Passion, BWV 244, Part 1", bach, german)
    part_two = await create("Matthäus-Passion, BWV 244, Part II", bach, german)
    anhang = await create("St. Matthew Passion, BWV 244, Anhang", bach)
    await create("Matthäus-Passion, BWV 244, Anhang", bach)
    act_one = await create("Swan Lake, Act I, Op. 20, TH 12", tchaikovsky)
    act_two = await create("Swan Lake, Act II, Op. 20, TH 12", tchaikovsky)
    ctrl = _controller(mass, [])

    await ctrl._join_same_works_once()

    rows = await mass.music.database.get_rows_from_query(
        "SELECT item_id, parent_work_id FROM works", limit=0
    )
    parents = {str(row["item_id"]): str(row["parent_work_id"]) for row in rows}
    assert set(parents) - {act_one.item_id, act_two.item_id} == {
        english.item_id,
        part_one.item_id,
        part_two.item_id,
        anhang.item_id,
        parents[act_one.item_id],
    }
    assert {parents[x.item_id] for x in (part_one, part_two, anhang)} == {english.item_id}
    assert parents[act_one.item_id] == parents[act_two.item_id] != "None"
    swan_lake = await works.get_library_item(parents[act_one.item_id])
    assert swan_lake.name == "Swan Lake, Op. 20, TH 12"

    duplicate = await create("Matthäus-Passion, BWV 244, Part 2", bach, english)
    await ctrl._join_same_works_once()
    assert duplicate.item_id in await _library_work_ids(mass)


# TEMP(classical 5c work composers): remove before release
async def test_works_without_composers_get_their_tracks_composers_once(
    music_mass: MusicAssistant,
) -> None:
    """The works without composers take the composers their tracks credit, once."""
    mass = music_mass
    first = await _add_artist(mass, "First")
    second = await _add_artist(mass, "Second")
    composer = await _add_artist(mass, "Composer")
    without = await _add_work(mass, f"Without {_token()}")
    composed = await _add_work(mass, f"Composed {_token()}", [composer])
    await _add_track(mass, without, composer=first)
    await _add_track(mass, without, composer=second)
    await _add_track(mass, without, composer=first)
    bare = await _add_track(mass, without)
    await _add_track(mass, composed, composer=first)
    # stored the way an earlier version left them, without composers
    await mass.music.database.delete(DB_TABLE_WORK_ARTISTS, {"work_id": int(without.item_id)})
    await mass.music.database.delete(
        DB_TABLE_TRACK_ARTISTS, {"track_id": int(bare.item_id), "role": ArtistRole.COMPOSER.value}
    )
    ctrl = _controller(mass, [])

    await ctrl._credit_track_composers_once()

    stored = await mass.music.works.get_library_item(without.item_id)
    assert [x.item_id for x in stored.composers] == [first.item_id, second.item_id]
    stored_bare = await mass.music.tracks.get_library_item(bare.item_id)
    assert [x.artist.item_id for x in stored_bare.credits if x.role == ArtistRole.COMPOSER] == [
        first.item_id,
        second.item_id,
    ]
    kept = await mass.music.works.get_library_item(composed.item_id)
    assert [x.item_id for x in kept.composers] == [composer.item_id]

    await mass.music.database.delete(DB_TABLE_WORK_ARTISTS, {"work_id": int(without.item_id)})
    await ctrl._credit_track_composers_once()
    assert not (await mass.music.works.get_library_item(without.item_id)).composers
