"""Tests for filling library works in from MusicBrainz during the MusicBrainz link run."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
from time import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from music_assistant_models.enums import ArtistEntityType, ArtistRole, ExternalID, Period, WorkType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
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

from music_assistant.constants import DB_TABLE_TRACKS
from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.constants import REFRESH_INTERVAL
from music_assistant.controllers.metadata.controller import _works_to_enrich_query
from music_assistant.controllers.metadata.work_enrichment import _fill_work_from_musicbrainz
from music_assistant.helpers.compare import is_named_by
from music_assistant.providers.musicbrainz.api_client import _FRESH_READS
from music_assistant.providers.musicbrainz.models import (
    MusicBrainzAlias,
    MusicBrainzArtist,
    MusicBrainzRelation,
    MusicBrainzSeries,
    MusicBrainzWork,
    MusicBrainzWorkAttribute,
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
def mass_fixture(
    music_mass_module: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> MusicAssistant:
    """Return the module-scoped database-only Music Assistant fixture, setting composer periods."""
    monkeypatch.delattr(music_mass_module.music.artists, "set_composer_periods")
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
                relations.append(
                    MusicBrainzRelation(
                        type="parts",
                        direction="forward",
                        work=work,
                        ordering_key=parent.ordering_key,
                    )
                )
    artists_by_id = {artist.id: artist for artist in artists}

    def _artist_details(mbid: str) -> MusicBrainzArtist:
        if mbid not in artists_by_id:
            raise InvalidDataError
        return artists_by_id[mbid]

    musicbrainz = Mock()
    musicbrainz.get_work_details = AsyncMock(side_effect=lambda work_id: by_id.get(work_id))
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
            provider_mappings={_mapping()},
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
    name: str | None = None,
    movement: str | None = "Movement",
) -> Track:
    """
    Store a library track of a work, carrying the given MusicBrainz work ids.

    :param movement: The movement of the work the track holds, None for the whole work.
    """
    token = _token()
    return await mass.music.tracks.add_item_to_library(
        Track(
            item_id=token,
            provider=PROVIDER_INSTANCE,
            name=name or f"Track {token}",
            provider_mappings={_mapping()},
            artists=UniqueList([await _add_artist(mass, f"Orchestra {token}")]),
            credits=[Credit(artist=composer, role=ArtistRole.COMPOSER)] if composer else [],
            work=ItemMapping.from_item(work),
            movement_name=movement,
            external_ids={(ExternalID.MB_WORK, work_id) for work_id in work_ids},
        )
    )


async def _add_titled_track(
    mass: MusicAssistant,
    title: str,
    work_name: str,
    work_ids: Sequence[str],
    movement_number: int | None = None,
    movement_name: str | None = None,
) -> Track:
    """
    Store a classical library track whose source names its work by title only, as a service does.

    The track gets its MusicBrainz work ids afterwards, as from its recording.
    """
    token = _token()
    track = await mass.music.tracks.add_item_to_library(
        Track(
            item_id=token,
            provider=PROVIDER_INSTANCE,
            name=title,
            provider_mappings={_mapping()},
            artists=UniqueList([await _add_artist(mass, f"Orchestra {token}")]),
            work=Work(
                item_id=work_name,
                provider=PROVIDER_INSTANCE,
                name=work_name,
                provider_mappings=set(),
            ),
            movement_number=movement_number,
            movement_name=movement_name,
            metadata=MediaItemMetadata(classical_tag=True),
        )
    )
    await mass.music.tracks.set_external_ids(
        track.item_id, {(ExternalID.MB_WORK, work_id) for work_id in work_ids}
    )
    return track


async def _enrich(ctrl: MetaDataController, work: Work) -> Work:
    """Run the link step on a library work and return the work as stored afterwards."""
    await ctrl.link_item_to_musicbrainz(await ctrl.mass.music.works.get_library_item(work.item_id))
    return await ctrl.mass.music.works.get_library_item(work.item_id)


async def _enrich_removed(ctrl: MetaDataController, work: Work) -> Work:
    """Run the link step on a library work that leaves the library during it, return the work."""
    library_work = await ctrl.mass.music.works.get_library_item(work.item_id)
    await ctrl.link_item_to_musicbrainz(library_work)
    with pytest.raises(MediaNotFoundError):
        await ctrl.mass.music.works.get_library_item(work.item_id)
    return library_work


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


def _scheduled(ctrl: MetaDataController) -> list[str]:
    """Return the names of the artists scheduled for a metadata refresh."""
    schedule = cast("Mock", ctrl.schedule_update_metadata)
    return [call.args[0].name for call in schedule.call_args_list]


def _symphony_5(token: str) -> tuple[MusicBrainzWork, MusicBrainzWork, MusicBrainzArtist]:
    """Return Beethoven's fifth, its first movement and its composer named with the token."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    symphony = works["symphony_5"]
    return symphony, works["symphony_5_movement_1"], _unique_composer(symphony, token)


async def _add_symphony_5(
    mass: MusicAssistant,
    symphony: MusicBrainzWork,
    movement: MusicBrainzWork,
    composers: Sequence[Artist] = (),
    credit: Artist | None = None,
    artists: Iterable[MusicBrainzArtist] = (),
    name: str | None = None,
    **kwargs: Any,
) -> tuple[Work, Track, MetaDataController]:
    """
    Store a library work of the symphony with a track of its movement, return them and a controller.

    :param composers: The library composers of the work.
    :param credit: The composer the track credits, if any.
    :param artists: The MusicBrainz artists the controller knows, besides the two works.
    :param name: The name of the work, a unique "Symphony no. 5" if None.
    """
    work = await _add_work(
        mass, name or f"Symphony no. 5 {_token()}", composers, symphony.id, **kwargs
    )
    track = await _add_track(mass, work, [movement.id], credit)
    return work, track, _controller(mass, [symphony, movement], artists)


async def _track_work(mass: MusicAssistant, track: Track) -> tuple[Track, Work]:
    """Return a library track as stored, with the library work it is on."""
    stored = await mass.music.tracks.get_library_item(track.item_id)
    assert stored.work is not None
    return stored, await mass.music.works.get_library_item(stored.work.item_id)


async def _clear_track_work(mass: MusicAssistant, track: Track) -> None:
    """Take a stored track off its work, leaving the rest of it as stored."""
    await mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_TRACKS} SET work_id = NULL WHERE item_id = :track_id",
        {"track_id": int(track.item_id)},
    )


# --------------------------------------------------------------------------- #
#  filling the work in                                                         #
# --------------------------------------------------------------------------- #


async def test_work_is_filled_in_from_its_musicbrainz_work(mass: MusicAssistant) -> None:
    """The MusicBrainz work a work's own id names fills its fields."""
    symphony, movement, _ = _symphony_5(_token())
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    work, track, ctrl = await _add_symphony_5(mass, symphony, movement, [composer], composer)

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


@pytest.mark.parametrize(
    ("disambiguation", "stored_version", "expected"),
    [
        ("1888 version, most often performed", "", "1888 version, most often performed"),
        (None, "", ""),
        # the note MusicBrainz gives now replaces the one stored, also when it has none
        ("original 1874/75 version", "1888 version", "original 1874/75 version"),
        ("", "1888 version", ""),
    ],
)
async def test_work_takes_the_musicbrainz_note_as_its_version(
    mass: MusicAssistant, disambiguation: str | None, stored_version: str, expected: str
) -> None:
    """A work's version is the note MusicBrainz tells same-named works apart by."""
    works = _musicbrainz_works("symphony_5")
    works["symphony_5"].disambiguation = disambiguation
    work = await _add_work(
        mass, f"Symphony no. 5 {_token()}", mbid=works["symphony_5"].id, version=stored_version
    )

    stored = await _enrich(_controller(mass, works.values()), work)

    assert stored.version == expected


async def test_work_stores_the_movements_musicbrainz_lists(mass: MusicAssistant) -> None:
    """The parts MusicBrainz lists for the work are stored as its movements."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    work = await _add_work(mass, f"Symphony no. 5 {_token()}", mbid=works["symphony_5"].id)
    await _add_track(mass, work, [works["symphony_5_movement_1"].id])

    stored = await _enrich(_controller(mass, works.values()), work)

    assert stored.metadata.movements == works["symphony_5"].movements
    assert [(x.number, x.name) for x in stored.metadata.movements or []] == [
        (1, "I. Allegro con brio"),
        (2, "II. Andante con moto"),
    ]


def _nutcracker_act(token: str) -> list[MusicBrainzWork]:
    """Return an act MusicBrainz titles in Cyrillic script, followed by its parts."""
    act = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Щелкунчик {token}: Действие II",
        aliases=[MusicBrainzAlias(f"The Nutcracker {token}: Act II", "", "en", "Work name", True)],
    )
    parts = [
        # an English alias that is not the primary one
        (
            f"Щелкунчик {token}: Действие II, Картина III, no. 10. Сцена",
            [MusicBrainzAlias(f"The Nutcracker {token}: Act II, Scene III. Scene", "", "en")],
        ),
        (
            f"Щелкунчик {token}: Действие II, Картина III, no. 11. Сцена",
            [MusicBrainzAlias(f"Nutcracker {token}", "", None, "Search hint")],
        ),
        ("Pas de deux", []),
    ]
    return [
        act,
        *(
            MusicBrainzWork(
                id=str(uuid4()),
                title=title,
                aliases=aliases,
                relations=[
                    MusicBrainzRelation(
                        type="parts", direction="backward", work=act, ordering_key=number
                    )
                ],
            )
            for number, (title, aliases) in enumerate(parts, 1)
        ),
    ]


async def test_movements_titled_in_another_script_take_a_latin_alias(mass: MusicAssistant) -> None:
    """Movements titled in another script are named by a Latin alias of their part, if any."""
    token = _token()
    act, *parts = _nutcracker_act(token)
    work = await _add_work(mass, act.title, mbid=act.id)
    ctrl = _controller(mass, [act, *parts])

    stored = await _enrich(ctrl, work)

    assert stored.name == f"The Nutcracker {token}: Act II"
    assert [(x.number, x.name) for x in stored.metadata.movements or []] == [
        (1, "Scene III. Scene"),
        (2, "Картина III, no. 11. Сцена"),
        (3, "Pas de deux"),
    ]
    # a movement titled in Latin script is not looked up
    musicbrainz = ctrl._musicbrainz_provider()
    assert musicbrainz is not None
    get_work_details = cast("AsyncMock", musicbrainz.get_work_details)
    assert parts[2].id not in {x.args[0] for x in get_work_details.await_args_list}


async def test_movement_track_moving_onto_its_work_takes_the_latin_alias_of_its_part(
    mass: MusicAssistant,
) -> None:
    """A track moving onto the work above its movement is named by the movement's Latin alias."""
    token = _token()
    act, *parts = _nutcracker_act(token)
    holder = await _add_work(mass, f"The Nutcracker {token}: Act II", mbid=act.id)
    work = await _add_work(mass, parts[0].title, mbid=parts[0].id)
    track = await _add_track(mass, work, [parts[0].id], movement=None)
    ctrl = _controller(mass, [act, *parts])

    await _enrich_removed(ctrl, work)

    stored, stored_work = await _track_work(mass, track)
    assert stored_work.item_id == holder.item_id
    assert (stored.movement_number, stored.movement_name) == (1, "Scene III. Scene")


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
    symphony, movement, _ = _symphony_5(_token())
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    name = f"Symphony no. 5 {_token()}"
    # named as MusicBrainz names it, which a work looked up for the first time takes
    symphony.title, symphony.aliases = name, []
    work, _, ctrl = await _add_symphony_5(
        mass,
        symphony,
        movement,
        [composer],
        composer,
        name=name,
        work_type=WorkType.OTHER,
        composition_year=1807,
        language="eng",
        musical_key="c-moll",
        catalog_numbers=["Op. 67"],
    )

    stored = await _enrich(ctrl, work)

    assert stored.mbid == symphony.id
    assert stored.name == name
    assert (
        stored.work_type,
        stored.composition_year,
        stored.language,
        stored.musical_key,
        stored.catalog_numbers,
    ) == (WorkType.OTHER, 1807, "eng", "c-moll", ["Op. 67"])


async def test_works_sharing_a_musicbrainz_id_merge_into_the_oldest(mass: MusicAssistant) -> None:
    """Library works carrying the same MusicBrainz id merge when one of them is enriched."""
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    older = await _add_work(mass, f"Fifth {_token()}", mbid=symphony.id)
    newer = await _add_work(mass, f"Fifth again {_token()}", mbid=symphony.id)
    ctrl = _controller(mass, [symphony])

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(newer.item_id))

    assert await _library_work_ids(mass) & {older.item_id, newer.item_id} == {older.item_id}
    survivor = await mass.music.works.get_library_item(older.item_id)
    assert survivor.work_type == WorkType.SYMPHONY
    assert survivor.metadata.last_musicbrainz_lookup is not None


# a tagged file holds the movement's id on its work only, not on the track; the track names
# the movement in its own words, else MusicBrainz names it, either without its number
@pytest.mark.parametrize("id_on_track", [True, False])
@pytest.mark.parametrize(
    ("movement_name", "shown_name"),
    [("02. Allegro con brio", "Allegro con brio"), (None, "Allegro con brio")],
)
async def test_movement_tracks_move_onto_the_work_added_above_it(
    mass: MusicAssistant, id_on_track: bool, movement_name: str | None, shown_name: str
) -> None:
    """A movement's tracks become movements of the work added above it, the movement goes."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    symphony, movement = works["symphony_5"], works["symphony_5_movement_1"]
    work = await _add_work(mass, movement.title, mbid=movement.id)
    track = await _add_track(
        mass, work, [movement.id] if id_on_track else [], movement=movement_name
    )
    ctrl = _controller(mass, works.values())

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(work.item_id))

    stored, parent = await _track_work(mass, track)
    assert (parent.mbid, parent.work_type) == (symphony.id, WorkType.SYMPHONY)
    assert (stored.movement_number, stored.movement_name) == (1, shown_name)
    assert not await mass.music.works.get_library_items_by_external_id(
        movement.id, ExternalID.MB_WORK, limit=None
    )


def _tosca_act(token: str) -> list[MusicBrainzWork]:
    """Return two arias of an act of an opera, followed by the act and the opera."""
    first_aria, act, opera = _work_levels(
        (f"Tosca {token}: Atto I. Recondita armonia", "Aria"),
        (f"Tosca {token}: Atto I", None),
        (f"Tosca {token}", "Opera"),
    )
    second_aria = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Tosca {token}: Atto I. Mario! Mario! Mario!",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=act)],
    )
    return [first_aria, second_aria, act, opera]


def _preludes(token: str) -> tuple[MusicBrainzWork, MusicBrainzWork, MusicBrainzWork]:
    """Return two MusicBrainz preludes, each a part of the set of preludes returned last."""
    seventh, preludes = _work_levels(
        (f"Prélude no. 7 in A major, op. 28 {token}: Andantino", None),
        (f"24 Préludes, op. 28 {token}", None),
    )
    fourth = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Prélude no. 4 in E minor, op. 28 {token}: Largo",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=preludes)],
    )
    return seventh, fourth, preludes


@pytest.mark.parametrize(
    "name",
    [
        # named after the key of a part the tracks perform, not the set
        "Prelude In A Major, Op 28",
        # named as a file's tag names it
        "Preludes, Op 28",
    ],
)
async def test_work_looked_up_first_takes_the_musicbrainz_name(
    mass: MusicAssistant, name: str
) -> None:
    """A work looked up for the first time takes MusicBrainz's name, in English."""
    token = _token()
    seventh, fourth, preludes = _preludes(token)
    preludes.aliases = [
        MusicBrainzAlias(f"24 Preludes, Op. 28 {token}", f"24 Preludes {token}", "en", None, True)
    ]
    work = await _add_work(mass, f"{name} {token}", mbid=preludes.id)
    await _add_track(mass, work, [seventh.id])
    await _add_track(mass, work, [fourth.id])
    ctrl = _controller(mass, [seventh, fourth, preludes])

    stored = await _enrich(ctrl, work)

    assert (stored.mbid, stored.name) == (preludes.id, f"24 Preludes, Op. 28 {token}")


async def test_renamed_work_takes_the_catalog_numbers_of_its_new_name(mass: MusicAssistant) -> None:
    """A work renamed after MusicBrainz no longer carries the catalog number of its old name."""
    token = _token()
    (concerto,) = _work_levels((f"Violin Concerto in E major, BWV 1042 {token}", "Concerto"))
    work = await _add_work(
        mass,
        f"Violin Concerto in E major, BWV 1041 {token}",
        mbid=concerto.id,
        catalog_numbers=["BWV 1041"],
    )
    await _add_track(mass, work, [concerto.id], movement=None)
    ctrl = _controller(mass, [concerto])

    stored = await _enrich(ctrl, work)

    assert (stored.name, stored.catalog_numbers) == (concerto.title, ["BWV 1042"])


@pytest.mark.parametrize("named", [False, True])
async def test_work_lists_the_catalog_numbers_its_name_gives_first(
    mass: MusicAssistant, named: bool
) -> None:
    """A work's catalog numbers from MusicBrainz start with those its name gives, keeping all."""
    token = _token()
    title = f"Toccata and Fugue in D minor, BWV 565 {token}"
    (toccata,) = _work_levels((title, None))
    toccata.relations = [
        MusicBrainzRelation(
            type="part of",
            series=MusicBrainzSeries(id=str(uuid4()), name=name, type="Catalogue"),
            attribute_values={"number": number},
        )
        for name, number in (("Bach-Compendium", "BC J 37"), ("BWV", "BWV 565"))
    ]
    # named as MusicBrainz names it, or by a file's tag
    work = await _add_work(mass, title if named else f"Toccata {token}", mbid=toccata.id)
    await _add_track(mass, work, [toccata.id], movement=None)
    ctrl = _controller(mass, [toccata])

    stored = await _enrich(ctrl, work)

    assert (stored.name, stored.catalog_numbers) == (title, ["BWV 565", "BC J 37"])


@pytest.mark.parametrize(
    ("title", "parent", "catalogue", "expected"),
    [
        ("Pavane in F-sharp minor, op. 50", None, None, ["op. 50"]),
        # a range numbers no single work
        ("Ouvertures, HWV 348\u2013350", None, None, []),
        ("Quartets, TWV 40:2-13", None, None, []),
        # an act carries no number of the opera it belongs to
        ("Tosca, op. 1: Act I", "Tosca, op. 1", None, []),
        # MusicBrainz's numbers win over those of the name
        ("Pavane in F-sharp minor, op. 50", None, "BWV 1", ["BWV 1"]),
    ],
)
def test_work_without_musicbrainz_catalog_numbers_takes_those_of_its_name(
    title: str, parent: str | None, catalogue: str | None, expected: list[str]
) -> None:
    """A work takes the catalog numbers of its name when MusicBrainz gives it none."""
    levels = _work_levels((title, None), *([(parent, None)] if parent else []))
    if catalogue:
        levels[0].relations.append(_catalogue(catalogue))
    work = Work(item_id="1", provider="library", name=title, provider_mappings=set())

    _fill_work_from_musicbrainz(work, levels)

    assert work.catalog_numbers == expected


@pytest.mark.parametrize("held", [True, False])
async def test_part_of_two_works_follows_the_held_one_else_the_one_without_a_comment(
    mass: MusicAssistant, held: bool
) -> None:
    """A part of a work and of an edition follows the one the library holds, else the plain one."""
    token = _token()
    (ballet,) = _work_levels((f"Swan Lake, op. 20 {token}", "Ballet"))
    original = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Swan Lake, op. 20 {token}: Act II, no. 13",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=ballet)],
    )
    (edition,) = _work_levels((f"Swan Lake, op. 20 {token}: Act II", None))
    edition.disambiguation = "ed. Drigo"
    waltz = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Swan Lake, op. 20 {token}: Act II, no. 13: I. Tempo di valse",
        relations=[
            MusicBrainzRelation(type="parts", direction="backward", work=edition),
            MusicBrainzRelation(type="parts", direction="backward", work=original),
        ],
    )
    if held:
        await _add_work(mass, f"Le Lac des cygnes {token}, Act II", mbid=edition.id)
    ctrl = _controller(mass, [waltz, original, edition, ballet])

    parent = await ctrl._parent_to_follow(waltz)

    assert parent is not None
    assert parent.work is not None
    assert parent.work.id == (edition.id if held else original.id)


async def test_work_lookup_cut_short_is_looked_up_again(mass: MusicAssistant) -> None:
    """A work whose lookup fails partway is not marked looked up, so the next run does it again."""
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    work = await _add_work(mass, f"Symphony no. 5 {_token()}", mbid=symphony.id)
    ctrl = _controller(mass, [symphony])

    with (
        patch.object(ctrl, "_move_movement_tracks", AsyncMock(side_effect=TimeoutError)),
        pytest.raises(TimeoutError),
    ):
        await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(work.item_id))

    assert (
        await mass.music.works.get_library_item(work.item_id)
    ).metadata.last_musicbrainz_lookup is None
    assert work.item_id in await _due_works(mass)
    stored = await _enrich(ctrl, work)
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert work.item_id not in await _due_works(mass)


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


@pytest.mark.parametrize("held", [True, False])
async def test_arrangement_in_the_library_is_linked_to_its_original(
    mass: MusicAssistant, held: bool
) -> None:
    """
    An original's forward arrangement relation links the library work arranged from it.

    An arrangement the library lacks adds no work.
    """
    symphony = _musicbrainz_works("symphony_5")["symphony_5"]
    [relation] = symphony.arrangement_relations
    assert relation.work is not None
    composer = await _add_artist(mass, f"Beethoven {_token()}")
    original = await _add_work(mass, f"Symphony no. 5 {_token()}", [composer], mbid=symphony.id)
    transcription = (
        await _add_work(mass, f"S. 464 no. 5 {_token()}", mbid=relation.work.id) if held else None
    )
    ctrl = _controller(mass, [symphony])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, original)

    assert await _library_work_ids(mass) == before
    if transcription:
        assert _linked(stored.arrangements) == [(transcription.item_id, [])]
        arranged = await mass.music.works.get_library_item(transcription.item_id)
        assert _linked(arranged.arrangement_of) == [(original.item_id, [composer.item_id])]
    else:
        assert stored.arrangements == []


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


# --------------------------------------------------------------------------- #
#  parent works                                                                #
# --------------------------------------------------------------------------- #


def _passion(token: str) -> list[MusicBrainzWork]:
    """Return a movement of a part of a passion by Bach, followed by the part and the passion."""
    mb_composer = MusicBrainzArtist(
        id=str(uuid4()), name=f"Bach {token}", sort_name=f"Bach {token}"
    )
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


@pytest.mark.parametrize(
    ("title", "has_parent", "logged"),
    [
        ("Orchestral Suite no. 3, BWV 1068: II. Air", False, True),
        ("La bohème, Act II", False, True),
        ("Orchestral Suite no. 3, BWV 1068: II. Air", True, False),
        # a whole work despite its colon
        ("Romeo and Juliet: Suite no. 2", False, False),
    ],
)
async def test_movement_without_a_parent_on_musicbrainz_is_logged_once(
    mass: MusicAssistant,
    caplog: pytest.LogCaptureFixture,
    title: str,
    has_parent: bool,
    logged: bool,
) -> None:
    """A MusicBrainz work titled as a movement or act but part of no work is logged with it."""
    levels = _work_levels((title, None), ("Orchestral Suite no. 3", "Suite"))
    mb_work = levels[0] if has_parent else MusicBrainzWork(id=str(uuid4()), title=title)
    work = await _add_work(mass, f"{title} {_token()}", mbid=mb_work.id)
    ctrl = _controller(mass, [*levels[1:], mb_work])
    ctrl.logger = logging.getLogger("test_work_hints")
    caplog.set_level(logging.DEBUG, logger="test_work_hints")

    await _enrich(ctrl, work)
    await _enrich(ctrl, work)

    expected = (
        f'MusicBrainz work "{title}" looks like a part of a larger work but is not linked to it, '
        'add its "part of" relationship, or merge it into the same part listed there: '
        f"https://musicbrainz.org/work/{mb_work.id}"
    )
    hints = [(x.levelno, x.getMessage()) for x in caplog.records if "looks like a part" in x.msg]
    assert hints == ([(logging.DEBUG, expected)] if logged else [])


async def test_every_level_above_a_movement_is_added_below_the_level_above_it(
    mass: MusicAssistant,
) -> None:
    """
    A movement gets each level above it added, named in the user's language, and moves up.

    Each added level takes the note MusicBrainz tells same-named works apart by as its version.
    """
    token = _token()
    movement, part, passion = _passion(token)
    mb_composer = passion.composer_relations[0].artist
    assert mb_composer is not None
    part_name, passion_name = f"St Matthew Passion {token}: Part I", f"St Matthew Passion {token}"
    part.aliases = [MusicBrainzAlias(part_name, part_name, "en", None, True)]
    passion.title = f"Matthäus-Passion {token}"
    passion.disambiguation = "first version"
    passion.aliases = [
        MusicBrainzAlias(passion_name, passion_name, "en", None, True),
        MusicBrainzAlias(f"La Passion {token}", f"La Passion {token}", "fr", None, True),
    ]
    composer = await _add_artist(mass, mb_composer.name, mb_composer.id)
    name = f"Kommt, ihr Töchter {token}"
    work = await _add_work(mass, name, mbid=movement.id)
    track = await _add_track(mass, work)
    ctrl = _controller(mass, [movement, part, passion])

    await _enrich_removed(ctrl, work)

    # the movement's track moves onto the part added above it, and the movement goes
    _, parent = await _track_work(mass, track)
    assert parent.parent_work is not None
    top = await mass.music.works.get_library_item(parent.parent_work.item_id)
    # a level without a type of its own takes the type of the level above
    assert [
        (x.name, x.mbid, x.work_type, x.parent_work is None, x.version) for x in (parent, top)
    ] == [
        (part_name, part.id, WorkType.ORATORIO, False, ""),
        (passion_name, passion.id, WorkType.ORATORIO, True, "first version"),
    ]
    assert [x.item_id for x in parent.composers] == [composer.item_id]
    assert [x.item_id for x in top.composers] == [composer.item_id]
    assert [x.item_id for x in top.parts] == [parent.item_id]
    # the levels are filled in already, so the link run does not look them up again
    assert not {parent.item_id, top.item_id} & await _due_works(mass)


@pytest.mark.parametrize(
    ("levels", "added"),
    [
        # an aria gets its opera and the typed set of operas, not the untyped group above
        (
            [
                ("Gianni Schicchi {t}: O mio babbino caro", "Aria"),
                ("Gianni Schicchi {t}", "Opera"),
                ("Il trittico {t}", "Opera"),
                ("Puccini's operas {t}", None),
            ],
            2,
        ),
        # a movement of a work without a type gets every level above it
        (
            [
                ("Pictures at an Exhibition {t}: Promenade", None),
                ("Pictures at an Exhibition {t}", None),
                ("Piano cycles {t}", None),
            ],
            2,
        ),
    ],
)
async def test_levels_are_added_up_to_the_topmost_typed_one(
    mass: MusicAssistant, levels: list[tuple[str, str | None]], added: int
) -> None:
    """
    An aria gets its opera and the typed set of operas added, not the untyped group above.

    A movement of a work without a type gets every level above it added, up to the top.
    """
    token = _token()
    mb_works = _work_levels(*((title.format(t=token), work_type) for title, work_type in levels))
    work = await _add_work(mass, mb_works[0].title, mbid=mb_works[0].id)
    ctrl = _controller(mass, mb_works)

    stored = await _enrich(ctrl, work)

    for mb_work in mb_works[1 : added + 1]:
        assert stored.parent_work is not None
        stored = await mass.music.works.get_library_item(stored.parent_work.item_id)
        assert (stored.name, stored.mbid) == (mb_work.title, mb_work.id)
    assert stored.parent_work is None
    for mb_work in mb_works[added + 1 :]:
        assert not await mass.music.works.get_library_items_by_external_id(
            mb_work.id, ExternalID.MB_WORK, limit=None
        )


@pytest.mark.parametrize("collection_held", [False, True])
async def test_work_without_a_typed_level_above_gets_no_parent(
    mass: MusicAssistant, collection_held: bool
) -> None:
    """
    A sonata in an untyped set of sonatas gets no parent and no work is added.

    This holds when the library holds a level above the set, which the sonata does not skip to.
    """
    token = _token()
    sonata, sonatas, collection = _work_levels(
        (f"Moonlight Sonata {token}", "Sonata"),
        (f"Two Piano Sonatas, op. 27 {token}", None),
        (f"Piano Sonatas {token}", None),
    )
    if collection_held:
        await _add_work(mass, collection.title, mbid=collection.id)
    work = await _add_work(mass, sonata.title, mbid=sonata.id)
    ctrl = _controller(mass, [sonata, sonatas, collection])
    before = await _library_work_ids(mass)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is None
    assert stored.mbid == sonata.id
    assert await _library_work_ids(mass) == before


@pytest.mark.parametrize("linked", ["concerto", "movement"])
async def test_typed_work_goes_under_the_untyped_set_the_library_holds(
    mass: MusicAssistant, linked: str
) -> None:
    """A concerto goes under its held set, linked itself or added as the level of a movement."""
    token = _token()
    movement, concerto, concertos = _work_levels(
        (f"Concerto grosso no. 1 {token}: I. Largo", None),
        (f"Concerto grosso no. 1 {token}", "Concerto"),
        (f"12 Concerti grossi, op. 6 {token}", None),
    )
    held_set = await _add_work(mass, concertos.title, mbid=concertos.id)
    mb_work = concerto if linked == "concerto" else movement
    work = await _add_work(mass, mb_work.title, mbid=mb_work.id)
    ctrl = _controller(mass, [movement, concerto, concertos])

    await _enrich(ctrl, work)

    (library_concerto,) = await mass.music.works.get_library_items_by_external_id(
        concerto.id, ExternalID.MB_WORK, limit=None
    )
    assert library_concerto.parent_work is not None
    assert library_concerto.parent_work.item_id == held_set.item_id


async def test_added_level_stops_at_a_held_level_and_takes_in_its_parts(
    mass: MusicAssistant,
) -> None:
    """A level added below a held one takes in its parts, and no level above is added."""
    token = _token()
    piece, book, collection, cycle = _work_levels(
        (f"Book I {token}: No. 2", None),
        (f"Book I {token}", None),
        (f"Collection {token}", None),
        (f"Cycle {token}", None),
    )
    sonata = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Book I {token}: No. 1, Sonata",
        type="Sonata",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=book)],
    )
    held = await _add_work(mass, collection.title, mbid=collection.id)
    ctrl = _controller(mass, [piece, sonata, book, collection, cycle])
    # linked while its book was missing, so it stays a work of its own
    sonata_work = await _add_work(mass, sonata.title, mbid=sonata.id)
    assert (await _enrich(ctrl, sonata_work)).parent_work is None
    work = await _add_work(mass, piece.title, mbid=piece.id)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    added = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert added.mbid == book.id
    assert added.parent_work is not None
    assert added.parent_work.item_id == held.item_id
    stored_sonata = await mass.music.works.get_library_item(sonata_work.item_id)
    assert stored_sonata.parent_work is not None
    assert stored_sonata.parent_work.item_id == added.item_id
    assert not await mass.music.works.get_library_items_by_external_id(
        cycle.id, ExternalID.MB_WORK, limit=None
    )


async def test_added_level_leaves_the_work_being_linked_to_its_own_link(
    mass: MusicAssistant,
) -> None:
    """A level added above a piece takes in its other parts, the piece's tracks follow."""
    token = _token()
    first, collection = _work_levels(
        (f"Sonata no. 1 {token}", "Sonata"), (f"Two Sonatas {token}", None)
    )
    second = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Sonata no. 2 {token}",
        relations=[MusicBrainzRelation(type="parts", direction="backward", work=collection)],
    )
    ctrl = _controller(mass, [first, second, collection])
    first_work = await _add_work(mass, first.title, mbid=first.id)
    assert (await _enrich(ctrl, first_work)).parent_work is None
    second_work = await _add_work(mass, second.title, mbid=second.id)
    track = await _add_track(mass, second_work, [second.id])

    await _enrich_removed(ctrl, second_work)

    stored_first = await mass.music.works.get_library_item(first_work.item_id)
    assert stored_first.parent_work is not None
    assert (await _track_work(mass, track))[1].item_id == stored_first.parent_work.item_id


async def test_added_parent_merges_into_a_newer_work_with_tracks(mass: MusicAssistant) -> None:
    """An added parent without tracks merges into a newer work of its id holding tracks."""
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
    # added apart from the parent, as a file tagged with the opera's id is
    tagged = await mass.music.works.create_library_item(
        Work(
            item_id=tagged_name,
            provider=PROVIDER_INSTANCE,
            name=tagged_name,
            provider_mappings=set(),
            external_ids={(ExternalID.MB_WORK, opera.id)},
        )
    )
    await _add_track(mass, tagged, [aria.id])

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(tagged.item_id))

    remaining = await _library_work_ids(mass)
    assert remaining & {added_parent.item_id, tagged.item_id} == {tagged.item_id}
    kept = await mass.music.works.get_library_item(tagged.item_id)
    assert (kept.name, kept.mbid) == (opera.title, opera.id)
    stored_excerpt = await mass.music.works.get_library_item(excerpt.item_id)
    assert stored_excerpt.parent_work is not None
    assert stored_excerpt.parent_work.item_id == tagged.item_id


@pytest.mark.parametrize("composed", [False, True])
async def test_parent_work_takes_no_composer_the_library_lacks(
    mass: MusicAssistant, composed: bool
) -> None:
    """
    A parent work added from MusicBrainz only gets composers the library already holds.

    A parent whose MusicBrainz composers the library lacks gets the work's own composers.
    """
    token = _token()
    hierarchy = _passion(token)
    composers = [await _add_artist(mass, f"J. S. Bach {token}")] if composed else []
    work = await _add_work(mass, f"Kommt {token}", composers, mbid=hierarchy[0].id)
    ctrl = _controller(mass, hierarchy)

    stored = await _enrich(ctrl, work)

    assert stored.parent_work is not None
    parent = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert [x.item_id for x in parent.composers] == [x.item_id for x in composers]
    found = await mass.music.artists.get_library_items_by_query(search=token)
    assert [x.item_id for x in found] == [x.item_id for x in composers]


async def test_parent_work_with_its_own_id_is_kept(mass: MusicAssistant) -> None:
    """A work whose parent work has its own MusicBrainz id keeps it and adds none."""
    token = _token()
    hierarchy = _passion(token)
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
    movement, part, passion = _passion(token)
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
    # the aria's place in its act differs from the number its title gives
    first_aria.relations[0].ordering_key = 13
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


@pytest.mark.parametrize("looked_up", ["opera", "act"])
async def test_opera_stores_its_acts_as_movements(mass: MusicAssistant, looked_up: str) -> None:
    """An opera looked up itself, or added above its act, stores all its acts as movements."""
    token = _token()
    (aria, first_act, opera), (second_aria, second_act, _) = _boheme(token)
    mb_work = opera if looked_up == "opera" else first_act
    work = await _add_work(mass, mb_work.title, mbid=mb_work.id)
    ctrl = _controller(mass, [aria, first_act, second_aria, second_act, opera])

    stored = await _enrich(ctrl, work)

    if looked_up == "act":
        assert stored.parent_work is not None
        stored = await mass.music.works.get_library_item(stored.parent_work.item_id)
    assert stored.mbid == opera.id
    assert [(x.number, x.name, x.mbid) for x in stored.metadata.movements or []] == [
        (1, "Act I", first_act.id),
        (2, "Act II", second_act.id),
    ]


async def test_movement_named_as_an_act_without_parts_shows_its_own_name(
    mass: MusicAssistant,
) -> None:
    """A movement named after an act but without parts of its own shows its own name."""
    token = _token()
    aria, act, opera = _work_levels(
        (f"Manon Lescaut {token}: Atto III. Ah! non v'avvicinate!", "Aria"),
        (f"Manon Lescaut {token}: Atto III", None),
        (f"Manon Lescaut {token}", "Opera"),
    )
    intermezzo = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Manon Lescaut {token}: Atto III. Intermezzo",
        relations=[
            MusicBrainzRelation(type="parts", direction="backward", work=opera, ordering_key=1)
        ],
    )
    act.relations[0].ordering_key = 2
    work = await _add_work(mass, opera.title, mbid=opera.id)
    ctrl = _controller(mass, [aria, act, intermezzo, opera])

    stored = await _enrich(ctrl, work)

    assert [(x.number, x.name) for x in stored.metadata.movements or []] == [
        (1, "Intermezzo"),
        (2, "Act III"),
    ]


async def test_act_named_in_another_script_shows_as_an_act(mass: MusicAssistant) -> None:
    """An act titled in Cyrillic, with an English name "Act 1", shows as "Act I"."""
    token = _token()
    dance, act, ballet = _work_levels(
        (f"Щелкунчик {token}: Действие I, no. 2. Марш", None),
        (f"Щелкунчик {token}: Действие I", None),
        (f"Щелкунчик {token}", "Ballet"),
    )
    ballet.aliases = [
        MusicBrainzAlias(f"The Nutcracker {token}", f"The Nutcracker {token}", "en", None, True)
    ]
    act.aliases = [
        MusicBrainzAlias(
            f"The Nutcracker {token}: Act 1", f"The Nutcracker {token}: Act 1", "en", None, True
        )
    ]
    work = await _add_work(mass, ballet.title, mbid=ballet.id)
    ctrl = _controller(mass, [dance, act, ballet])

    stored = await _enrich(ctrl, work)

    assert [x.name for x in stored.metadata.movements or []] == ["Act I"]


async def test_acts_that_would_be_shown_alike_keep_their_names(mass: MusicAssistant) -> None:
    """Acts of an opera that would both be shown as "Act I" keep MusicBrainz's names."""
    token = _token()
    opera = MusicBrainzWork(id=str(uuid4()), title=f"Don Carlo {token}", type="Opera")
    acts = ["Atto I (Fontainebleau)", "Atto I (II)", "Atto II (III)"]
    works = [opera]
    for number, act_name in enumerate(acts, 1):
        act = MusicBrainzWork(
            id=str(uuid4()),
            title=f"{opera.title}: {act_name}",
            relations=[
                MusicBrainzRelation(
                    type="parts", direction="backward", work=opera, ordering_key=number
                )
            ],
        )
        aria = MusicBrainzWork(
            id=str(uuid4()),
            title=f"{act.title}, scena 1. Aria",
            type="Aria",
            relations=[MusicBrainzRelation(type="parts", direction="backward", work=act)],
        )
        works += [act, aria]
    work = await _add_work(mass, opera.title, mbid=opera.id)
    ctrl = _controller(mass, works)

    stored = await _enrich(ctrl, work)

    assert [(x.number, x.name) for x in stored.metadata.movements or []] == [
        (1, "Atto I (Fontainebleau)"),
        (2, "Atto I (II)"),
        (3, "Act II"),
    ]


@pytest.mark.parametrize("with_track", [False, True])
async def test_aria_goes_to_the_act_the_library_holds(
    mass: MusicAssistant, with_track: bool
) -> None:
    """
    An aria links to the library work holding its act rather than to the opera.

    An aria's track becomes a movement of the act holding its id instead, and the aria work goes.
    """
    token = _token()
    (aria, act_level, opera), _ = _boheme(token)
    act = await _add_work(mass, f"La bohème {token}, Act I", mbid=act_level.id)
    work = await _add_work(mass, aria.title, mbid=aria.id)
    track = (
        await _add_track(mass, work, [aria.id], name="Che Gelida Manina (Puccini)", movement=None)
        if with_track
        else None
    )
    ctrl = _controller(mass, [aria, act_level, opera])

    await ctrl.link_item_to_musicbrainz(await mass.music.works.get_library_item(work.item_id))

    if track:
        stored, stored_work = await _track_work(mass, track)
        assert stored_work.item_id == act.item_id
        assert (stored.movement_number, stored.movement_name) == (13, "Che gelida manina")
        assert not await mass.music.works.get_library_items_by_external_id(
            aria.id, ExternalID.MB_WORK, limit=None
        )
        assert aria.id in {x[1] for x in stored.external_ids}
    else:
        parent = (await mass.music.works.get_library_item(work.item_id)).parent_work
        assert parent is not None
        assert parent.item_id == act.item_id
    assert not await mass.music.works.get_library_items_by_external_id(
        opera.id, ExternalID.MB_WORK, limit=None
    )


async def test_movement_below_a_level_further_up_moves_onto_the_level_right_above_it(
    mass: MusicAssistant,
) -> None:
    """
    A movement under its top work has its tracks moved onto the part added between them.

    The part is the only work added.
    """
    token = _token()
    movement, part, passion = _passion(token)
    library_passion = await _add_work(mass, passion.title, mbid=passion.id)
    work = await _add_work(
        mass, f"Kommt {token}", mbid=movement.id, parent_work=ItemMapping.from_item(library_passion)
    )
    track = await _add_track(
        mass,
        work,
        [movement.id],
        name="Erster Teil, 3. Chorus Kommt, ihr Töchter",
        movement="3. Chorus Kommt, ihr Töchter",
    )
    ctrl = _controller(mass, [movement, part, passion])
    before = await _library_work_ids(mass)

    await _enrich_removed(ctrl, work)

    stored_track, library_part = await _track_work(mass, track)
    assert library_part.mbid == part.id
    assert library_part.parent_work is not None
    assert library_part.parent_work.item_id == library_passion.item_id
    assert await _library_work_ids(mass) == before - {work.item_id} | {library_part.item_id}
    # numbered as MusicBrainz numbers the movement, whatever number the track's title gives
    assert (stored_track.movement_number, stored_track.movement_name) == (
        1,
        "Chorus Kommt, ihr Töchter",
    )


async def test_movement_work_keeps_a_track_performing_more_works(mass: MusicAssistant) -> None:
    """Only a track performing the movement alone moves, so the movement work stays."""
    token = _token()
    (aria, act_level, opera), (second_aria, _, _) = _boheme(token)
    act = await _add_work(mass, f"La bohème {token}, Act I", mbid=act_level.id)
    work = await _add_work(mass, aria.title, mbid=aria.id)
    alone = await _add_track(mass, work, [aria.id])
    medley = await _add_track(mass, work, [aria.id, second_aria.id])
    ctrl = _controller(mass, [aria, act_level, opera])

    stored = await _enrich(ctrl, work)

    assert stored.mbid == aria.id
    _, moved = await _track_work(mass, alone)
    _, kept = await _track_work(mass, medley)
    assert (moved.item_id, kept.item_id) == (act.item_id, work.item_id)


async def test_track_moved_onto_its_act_stays_when_the_act_moves_onto_the_opera(
    mass: MusicAssistant,
) -> None:
    """A track linked by its tags alone keeps the aria it performs once moved onto the act."""
    token = _token()
    (aria, act_level, opera), _ = _boheme(token)
    await _add_work(mass, opera.title, mbid=opera.id)
    act = await _add_work(mass, f"La bohème {token}, Act I", mbid=act_level.id)
    work = await _add_work(mass, aria.title, mbid=aria.id)
    track = await _add_track(mass, work)
    ctrl = _controller(mass, [aria, act_level, opera])

    await ctrl._move_movement_tracks(work, [aria, act_level, opera])
    await ctrl._move_movement_tracks(act, [act_level, opera])

    stored, stored_work = await _track_work(mass, track)
    assert (stored_work.item_id, stored.movement_number) == (act.item_id, 13)
    assert (ExternalID.MB_WORK, aria.id) in stored.external_ids


async def test_identified_act_takes_its_arias_from_the_opera(mass: MusicAssistant) -> None:
    """
    The arias of an act found among the parts of its opera move under the act.

    The tracks of such an aria become movements of the act, and the aria work goes.
    """
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
    track = await _add_track(mass, arias[0], [first_aria.id], name="Che Gelida Manina")
    act = await _add_work(
        mass,
        f"La bohème {token}, Act I",
        mbid=first_act.id,
        parent_work=ItemMapping.from_item(library_opera),
    )
    await _add_track(mass, act, [first_aria.id])
    ctrl = _controller(mass, [first_aria, other_aria, second_aria, first_act, second_act, opera])

    stored = await _enrich(ctrl, act)

    assert stored.mbid == first_act.id
    stored_track, track_work = await _track_work(mass, track)
    assert (track_work.item_id, stored_track.movement_number) == (act.item_id, 13)
    assert arias[0].item_id not in await _library_work_ids(mass)
    parents = [(await mass.music.works.get_library_item(x.item_id)).parent_work for x in arias[1:]]
    assert [x.item_id if x else None for x in parents] == [act.item_id, library_opera.item_id]
    # moved once, the arias stay where they are
    assert (await _enrich(ctrl, act)).mbid == first_act.id
    stored_act = await mass.music.works.get_library_item(act.item_id)
    assert {x.item_id for x in stored_act.parts} == {arias[1].item_id}


async def test_work_looked_up_first_takes_its_parts_from_anywhere_in_the_library(
    mass: MusicAssistant,
) -> None:
    """A work looked up for the first time takes the library works of its parts, wherever."""
    token = _token()
    prelude, preludes = _work_levels(
        (f"Prélude in E minor, op. 28 no. 4 {token}", "Prelude"),
        (f"24 Préludes, op. 28 {token}", None),
    )
    piece = await _add_work(mass, prelude.title, mbid=prelude.id)
    track = await _add_track(mass, piece, [prelude.id], movement=None)
    ctrl = _controller(mass, [prelude, preludes])
    assert (await _enrich(ctrl, piece)).parent_work is None
    holder = await _add_work(mass, preludes.title, mbid=preludes.id)

    await _enrich(ctrl, holder)

    assert (await _track_work(mass, track))[1].item_id == holder.item_id
    assert piece.item_id not in await _library_work_ids(mass)


async def test_part_taken_in_by_another_work_leaves_its_empty_parent(
    mass: MusicAssistant,
) -> None:
    """A part listed under two works moves to the one looked up later, its emptied parents go."""
    token = _token()
    work_level = MusicBrainzWork(id=str(uuid4()), title=f"Concerto {token}", type="Concerto")
    edition = MusicBrainzWork(
        id=str(uuid4()), title=f"Concerto {token}", type="Concerto", disambiguation="1750 version"
    )
    part = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Concerto {token}: I. Allegro",
        relations=[
            MusicBrainzRelation(type="parts", direction="backward", work=work_level),
            MusicBrainzRelation(type="parts", direction="backward", work=edition),
        ],
    )
    edition.relations = [MusicBrainzRelation(type="parts", direction="forward", work=part)]
    # the old parent's own parent holds nothing else, so it goes too
    grandparent = await _add_work(mass, f"Concertos {token}", mbid=str(uuid4()))
    old_parent = await _add_work(
        mass, work_level.title, mbid=work_level.id, parent_work=ItemMapping.from_item(grandparent)
    )
    part_work = await _add_work(
        mass, part.title, mbid=part.id, parent_work=ItemMapping.from_item(old_parent)
    )
    edition_work = await _add_work(mass, f"{edition.title} (1750)", mbid=edition.id)
    ctrl = _controller(mass, [work_level, edition, part])

    await _enrich(ctrl, edition_work)

    stored = await mass.music.works.get_library_item(part_work.item_id)
    assert stored.parent_work is not None
    assert stored.parent_work.item_id == edition_work.item_id
    remaining = await _library_work_ids(mass)
    assert old_parent.item_id not in remaining
    assert grandparent.item_id not in remaining


# --------------------------------------------------------------------------- #
#  composers                                                                   #
# --------------------------------------------------------------------------- #


async def test_sole_composer_gets_the_musicbrainz_id_and_merges(mass: MusicAssistant) -> None:
    """The only composer gets the id of the only MusicBrainz composer and merges, oldest first."""
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    older = await _add_artist(mass, f"L. van Beethoven {token}", beethoven.id)
    composer = await _add_artist(mass, f"Beethoven {token}")
    work, track, ctrl = await _add_symphony_5(mass, symphony, movement, [composer], composer)

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


@pytest.mark.parametrize(
    "names",
    [
        # a sole composer whose name MusicBrainz's composer does not carry
        ["Bach"],
        # two composers on the work, the first named as MusicBrainz's composer
        ["Beethoven", "Ries"],
    ],
)
async def test_composer_musicbrainz_does_not_confirm_gets_no_musicbrainz_id(
    mass: MusicAssistant, names: list[str]
) -> None:
    """
    A sole composer whose name MusicBrainz's composer does not carry is not given its id.

    With several composers on the work, none of them is guessed to be MusicBrainz's composer.
    """
    token = _token()
    symphony, movement, _ = _symphony_5(token)
    composers = [await _add_artist(mass, f"{name} {token}") for name in names]
    credit = composers[0] if len(composers) == 1 else None
    work, _, ctrl = await _add_symphony_5(mass, symphony, movement, composers, credit)

    stored = await _enrich(ctrl, work)

    assert stored.mbid == symphony.id
    assert [(x.item_id, x.mbid) for x in stored.composers] == [(x.item_id, None) for x in composers]
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


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


_SEARCH_HINT = MusicBrainzAlias("Tschaikowsky", "Tschaikowsky", type="Search hint")


async def test_library_composers_are_added_to_a_work_without_them_and_its_tracks(
    mass: MusicAssistant,
) -> None:
    """A work without composers gets MusicBrainz's, credited on its tracks without a composer."""
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    composer = await _add_artist(mass, f"L. van Beethoven {token}", beethoven.id)
    arranger = await _add_artist(mass, f"Liszt {token}")
    work, bare, ctrl = await _add_symphony_5(mass, symphony, movement)
    credited = await _add_track(mass, work, [movement.id], arranger)

    stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [composer.item_id]
    assert _composers(await mass.music.tracks.get_library_item(bare.item_id)) == [composer.name]
    assert _composers(await mass.music.tracks.get_library_item(credited.item_id)) == [arranger.name]


async def test_composer_without_a_musicbrainz_id_is_found_by_its_exact_name(
    mass: MusicAssistant,
) -> None:
    """A library artist of the MusicBrainz composer's exact name composes the work as it is."""
    symphony, movement, beethoven = _symphony_5(_token())
    composer = await _add_artist(mass, beethoven.name)
    work, _, ctrl = await _add_symphony_5(mass, symphony, movement)

    stored = await _enrich(ctrl, work)

    assert [(x.item_id, x.mbid) for x in stored.composers] == [(composer.item_id, None)]
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


async def test_composer_the_library_lacks_is_not_added(mass: MusicAssistant) -> None:
    """A MusicBrainz composer the library holds no artist for leaves the work without one."""
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    work, track, ctrl = await _add_symphony_5(mass, symphony, movement)

    stored = await _enrich(ctrl, work)

    assert stored.mbid == symphony.id
    assert stored.composers == []
    assert stored.metadata.last_musicbrainz_lookup is not None
    assert _composers(await mass.music.tracks.get_library_item(track.item_id)) == []
    assert not await mass.music.artists.get_library_items_by_external_id(
        beethoven.id, ExternalID.MB_ARTIST, limit=None
    )
    assert not await mass.music.artists.get_library_items_by_query(search=token)


# --------------------------------------------------------------------------- #
#  composer periods                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("stored", "period"),
    [
        # life dates stored, no period yet
        ({"metadata": _BEETHOVEN}, Period.CLASSICAL),
        # a period stored already
        ({"period": Period.ROMANTIC}, Period.ROMANTIC),
    ],
)
async def test_composer_with_life_dates_gets_a_period_right_away(
    mass: MusicAssistant, stored: dict[str, Any], period: Period
) -> None:
    """
    A classical composer whose life dates are stored gets a period without a refresh.

    A composer who already has a period keeps it and is not refreshed.
    """
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    composer = await _add_artist(mass, f"Beethoven {token}", beethoven.id, **stored)
    work, _, ctrl = await _add_symphony_5(mass, symphony, movement, [composer], composer)

    await _enrich(ctrl, work)

    assert (await mass.music.artists.get_library_item(composer.item_id)).period == period
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


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
    """A work is due while not looked up and it carries its MusicBrainz work id."""
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

    assert await _due_works(mass) & created == {own_id.item_id}


# --------------------------------------------------------------------------- #
#  looking works up again                                                      #
# --------------------------------------------------------------------------- #


def _catalogue(number: str) -> MusicBrainzRelation:
    """Return the relation of a MusicBrainz work to the BWV catalogue, under the given number."""
    return MusicBrainzRelation(
        type="part of",
        series=MusicBrainzSeries(id=str(uuid4()), name="BWV", type="Catalogue"),
        attribute_values={"number": number},
    )


async def _reload(ctrl: MetaDataController, work: Work) -> Work:
    """Look a library work up again and return the work as stored afterwards."""
    library_work = await ctrl.mass.music.works.get_library_item(work.item_id)
    reloaded = await ctrl._link_work_to_musicbrainz(library_work, reload=True)
    return await ctrl.mass.music.works.get_library_item(reloaded.item_id)


async def test_work_looked_up_again_takes_musicbrainz_current_values(mass: MusicAssistant) -> None:
    """
    A work looked up again takes MusicBrainz's name, version, type, key and catalogue numbers.

    It also takes the movements MusicBrainz lists now.
    """
    token = _token()
    part_ids = [str(uuid4()) for _ in range(3)]

    def _parts(*titles: str) -> list[MusicBrainzRelation]:
        return [
            MusicBrainzRelation(
                type="parts",
                direction="forward",
                work=MusicBrainzWork(id=part_id, title=title),
                ordering_key=number,
            )
            for number, (part_id, title) in enumerate(zip(part_ids, titles, strict=False), 1)
        ]

    mb_work = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Sonata {token}, BWV 1",
        type="Sonata",
        disambiguation="first version",
        attributes=[MusicBrainzWorkAttribute(type="Key", value="C minor")],
        relations=[_catalogue("BWV 1"), *_parts("I. Allegro", "II. Adagio")],
    )
    work = await _add_work(mass, mb_work.title, mbid=mb_work.id)
    ctrl = _controller(mass, [mb_work])
    await _enrich(ctrl, work)
    # what changed on MusicBrainz since
    mb_work.title = f"Concerto {token}, BWV 2"
    mb_work.type = "Concerto"
    mb_work.disambiguation = "second version"
    mb_work.attributes = [MusicBrainzWorkAttribute(type="Key", value="D major")]
    mb_work.relations = [
        _catalogue("BWV 2"),
        *_parts("I. Allegro con brio", "II. Adagio", "III. Finale"),
    ]

    # a lookup that is not a first one only fills what the work lacks
    kept = await _enrich(ctrl, work)
    assert (kept.name, kept.musical_key, kept.catalog_numbers) == (
        f"Sonata {token}, BWV 1",
        "C minor",
        ["BWV 1"],
    )

    stored = await _reload(ctrl, work)

    assert stored.name == f"Concerto {token}, BWV 2"
    assert stored.version == "second version"
    assert stored.work_type == WorkType.CONCERTO
    assert stored.musical_key == "D major"
    assert stored.catalog_numbers == ["BWV 2"]
    assert stored.metadata.movements == mb_work.movements
    assert len(mb_work.movements or []) == 3


async def test_refreshing_any_level_looks_up_the_whole_work_tree(mass: MusicAssistant) -> None:
    """Refreshing an aria looks up its opera and every library work under it again."""
    token = _token()
    aria, act, opera = _work_levels(
        (f"Opera {token}: Act I: Aria", None),
        (f"Opera {token}: Act I", None),
        (f"Opera {token}", "Opera"),
    )
    work = await _add_work(mass, aria.title, mbid=aria.id)
    ctrl = _controller(mass, [aria, act, opera])
    await _enrich(ctrl, work)
    library_opera = await mass.music.works.get_library_item_by_external_id(
        opera.id, ExternalID.MB_WORK
    )
    assert library_opera is not None
    opera.title = f"Die Oper {token}"
    lookups = cast("AsyncMock", ctrl._musicbrainz_provider().get_work_details)  # type: ignore[union-attr]
    lookups.reset_mock()

    refreshed = await ctrl.refresh_work_tree(await mass.music.works.get_library_item(work.item_id))

    assert refreshed.item_id == work.item_id
    renamed = await mass.music.works.get_library_item(library_opera.item_id)
    assert renamed.name == f"Die Oper {token}"
    # each work of the tree is looked up again, the levels above each one with it
    assert {call.args[0] for call in lookups.await_args_list} == {aria.id, act.id, opera.id}


async def test_work_merged_on_musicbrainz_takes_the_new_id_and_joins_its_holder(
    mass: MusicAssistant,
) -> None:
    """A work whose id MusicBrainz answers with another work takes that id and joins its holder."""
    token = _token()
    new = MusicBrainzWork(id=str(uuid4()), title=f"Ave Maria {token}")
    old_id = str(uuid4())
    merged = await _add_work(mass, f"Ave Maria {token}", mbid=old_id)
    track = await _add_track(mass, merged, [old_id])
    holder = await _add_work(mass, f"Ave Maria (holder) {token}", mbid=new.id)
    ctrl = _controller(mass, [new])
    cast("AsyncMock", ctrl._musicbrainz_provider().get_work_details).side_effect = (  # type: ignore[union-attr]
        lambda work_id: new if work_id in (old_id, new.id) else None
    )

    stored = await _reload(ctrl, merged)

    assert stored.mbid == new.id
    holders = await mass.music.works.get_library_items_by_external_id(
        new.id, ExternalID.MB_WORK, limit=None
    )
    assert [x.item_id for x in holders] == [stored.item_id]
    assert not await mass.music.works.get_library_items_by_external_id(
        old_id, ExternalID.MB_WORK, limit=None
    )
    # the work holding the track survives
    assert stored.item_id == merged.item_id
    with pytest.raises(MediaNotFoundError):
        await mass.music.works.get_library_item(holder.item_id)
    assert (await _track_work(mass, track))[1].item_id == merged.item_id


@pytest.mark.parametrize("new_parent", [True, False])
async def test_work_looked_up_again_follows_its_parent_on_musicbrainz(
    mass: MusicAssistant, new_parent: bool
) -> None:
    """A part moves under the level MusicBrainz names now, or loses its parent; the old one goes."""
    token = _token()
    part, old_parent = _work_levels((f"Part {token}", None), (f"Old set {token}", "Suite"))
    work = await _add_work(mass, part.title, mbid=part.id)
    ctrl = _controller(mass, [part, old_parent])
    await _enrich(ctrl, work)
    old = await mass.music.works.get_library_item_by_external_id(old_parent.id, ExternalID.MB_WORK)
    assert old is not None
    parent = MusicBrainzWork(id=str(uuid4()), title=f"New set {token}", type="Suite")
    part.relations = (
        [MusicBrainzRelation(type="parts", direction="backward", work=parent)] if new_parent else []
    )
    ctrl = _controller(mass, [part, parent])

    stored = await _reload(ctrl, work)

    if new_parent:
        assert stored.parent_work is not None
        assert stored.parent_work.name == f"New set {token}"
    else:
        assert stored.parent_work is None
    with pytest.raises(MediaNotFoundError):
        await mass.music.works.get_library_item(old.item_id)


@pytest.mark.parametrize("still_listed", [True, False])
async def test_work_looked_up_again_drops_an_arrangement_link_musicbrainz_removed(
    mass: MusicAssistant, still_listed: bool
) -> None:
    """
    An arrangement link MusicBrainz no longer lists goes on a lookup again, not before.

    A link MusicBrainz still lists stays.
    """
    token = _token()
    original_mb = MusicBrainzWork(id=str(uuid4()), title=f"Pictures {token}")
    arrangement_mb = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Pictures, orchestrated {token}",
        relations=[MusicBrainzRelation(type="arrangement", direction="backward", work=original_mb)],
    )
    original = await _add_work(mass, original_mb.title, mbid=original_mb.id)
    arrangement = await _add_work(mass, arrangement_mb.title, mbid=arrangement_mb.id)
    ctrl = _controller(mass, [arrangement_mb])
    assert (await _enrich(ctrl, arrangement)).arrangement_of
    if not still_listed:
        arrangement_mb.relations = []

    assert (await _enrich(ctrl, arrangement)).arrangement_of
    reloaded = await _reload(ctrl, arrangement)
    assert [x.item_id for x in reloaded.arrangement_of] == (
        [original.item_id] if still_listed else []
    )


async def test_work_looked_up_again_numbers_its_tracks_by_its_new_list(
    mass: MusicAssistant,
) -> None:
    """The tracks of a work's movements are numbered again when MusicBrainz reorders them."""
    token = _token()
    first = MusicBrainzWork(id=str(uuid4()), title=f"Suite {token}: Prelude")
    second = MusicBrainzWork(id=str(uuid4()), title=f"Suite {token}: Gigue")
    suite = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Suite {token}",
        type="Suite",
        relations=[
            MusicBrainzRelation(type="parts", direction="forward", work=first, ordering_key=1),
            MusicBrainzRelation(type="parts", direction="forward", work=second, ordering_key=2),
        ],
    )
    work = await _add_work(mass, suite.title, mbid=suite.id)
    track = await _add_track(mass, work, [second.id])
    await mass.music.database.execute_write(
        f"UPDATE {DB_TABLE_TRACKS} SET movement_number = 2 WHERE item_id = :track_id",
        {"track_id": int(track.item_id)},
    )
    ctrl = _controller(mass, [suite, first, second])
    await _enrich(ctrl, work)
    for relation in suite.relations:
        relation.ordering_key = 3 - (relation.ordering_key or 0)

    await _reload(ctrl, work)

    assert (await mass.music.tracks.get_library_item(track.item_id)).movement_number == 1


@pytest.mark.parametrize("by_hand", [True, False])
async def test_only_a_work_refreshed_by_hand_is_read_past_the_cache(
    mass: MusicAssistant, by_hand: bool
) -> None:
    """Refresh item on a work reads MusicBrainz fresh; the 90 day re-read reads through the cache."""
    mb_work = MusicBrainzWork(id=str(uuid4()), title=f"Sonata {_token()}")
    work = await _add_work(mass, mb_work.title, mbid=mb_work.id)
    ctrl = _controller(mass, [mb_work])
    fresh: list[bool] = []

    async def _details(work_id: str) -> MusicBrainzWork | None:
        state = _FRESH_READS.get()
        fresh.append(state is not None and state[0] is asyncio.current_task())
        return mb_work if work_id == mb_work.id else None

    cast("AsyncMock", ctrl._musicbrainz_provider().get_work_details).side_effect = _details  # type: ignore[union-attr]
    library_work = await mass.music.works.get_library_item(work.item_id)

    if by_hand:
        await ctrl.refresh_work_tree(library_work)
    else:
        assert await ctrl._reread_work_on_musicbrainz(library_work)

    assert fresh
    assert set(fresh) == {by_hand}
