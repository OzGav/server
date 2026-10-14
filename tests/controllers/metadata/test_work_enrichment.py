"""Tests for filling library works in from MusicBrainz during the MusicBrainz link run."""

from __future__ import annotations

import asyncio
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

from music_assistant.constants import (
    DB_TABLE_TRACKS,
    DB_TABLE_WORKS,
)
from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.constants import (
    ARTIST_LOOKUP_RETRY_INTERVAL,
    REFRESH_INTERVAL,
)
from music_assistant.controllers.metadata.controller import (
    _classical_artists_to_reread_query,
    _composers_to_identify_query,
    _performers_to_identify_query,
    _works_to_enrich_query,
    _works_to_reread_query,
)
from music_assistant.controllers.metadata.helpers import composer_period, musicbrainz_name
from music_assistant.helpers.compare import is_named_by
from music_assistant.providers.musicbrainz.api_client import _FRESH_READS
from music_assistant.providers.musicbrainz.models import (
    MusicBrainzAlias,
    MusicBrainzArtist,
    MusicBrainzLifeSpan,
    MusicBrainzRelation,
    MusicBrainzSeries,
    MusicBrainzUrl,
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
            classical_tag=True,
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


async def test_track_joining_a_held_movement_work_moves_onto_the_work_above_it(
    mass: MusicAssistant,
) -> None:
    """A track joining a movement work the library holds becomes a movement of the work above."""
    works = _musicbrainz_works("symphony_5", "symphony_5_movement_1")
    symphony, movement = works["symphony_5"], works["symphony_5_movement_1"]
    holder = await _add_work(mass, symphony.title, mbid=symphony.id, work_type=WorkType.SYMPHONY)
    movement_work = await _add_work(
        mass, movement.title, mbid=movement.id, parent_work=ItemMapping.from_item(holder)
    )
    # a tagged file carries the movement's id on its work only, and moves along with it
    file_track = await _add_track(mass, movement_work, movement=None)
    track = await _add_track(mass, holder, [movement.id], movement=None)
    await _clear_track_work(mass, track)
    ctrl = _controller(mass, works.values())

    await ctrl._add_track_work(track.item_id, [movement.id])

    stored, stored_work = await _track_work(mass, track)
    assert (stored_work.item_id, stored.movement_number, stored.movement_name) == (
        holder.item_id,
        1,
        "Allegro con brio",
    )
    assert (await _track_work(mass, file_track))[1].item_id == holder.item_id
    assert movement_work.item_id not in await _library_work_ids(mass)


async def test_work_added_for_a_track_is_named_in_the_user_s_language(
    mass: MusicAssistant,
) -> None:
    """A work added from a track's MusicBrainz id takes its English name, not its original one."""
    token = _token()
    (variations,) = _work_levels((f"Goldberg-Variationen, BWV 988 {token}", "Variations"))
    english = f"Goldberg Variations {token}"
    variations.aliases = [MusicBrainzAlias(english, english, "en", None, True)]
    track = await _add_titled_track(
        mass, f"Goldberg {token}: Aria", f"Goldberg {token}", [variations.id]
    )
    ctrl = _controller(mass, [variations])

    await ctrl._add_track_work(track.item_id, [variations.id])

    _, work = await _track_work(mass, track)
    assert (work.mbid, work.name) == (variations.id, english)


async def test_track_is_placed_by_its_musicbrainz_id_whatever_its_title_names(
    mass: MusicAssistant,
) -> None:
    """A track titled as another work gets the work its MusicBrainz id names, and no other."""
    token = _token()
    aria, opera = _work_levels(
        (f"Gianni Schicchi {token}: “O mio babbino caro”", "Aria"),
        (f"Gianni Schicchi {token}", "Opera"),
    )
    title_work = f"Tosca {token}, Act II"
    track = await _add_titled_track(mass, f"{title_work}: Vissi d'arte", title_work, [aria.id])
    ctrl = _controller(mass, [aria, opera])

    await ctrl._add_track_work(track.item_id, [aria.id])

    assert (await _track_work(mass, track))[1].mbid == opera.id
    rows = await mass.music.database.get_rows_from_query(
        "SELECT item_id FROM works WHERE name LIKE :name", {"name": f"Tosca {token}%"}, limit=0
    )
    assert rows == []


async def test_track_titled_with_its_act_is_numbered_as_musicbrainz_numbers_its_piece(
    mass: MusicAssistant,
) -> None:
    """A track whose title names its act and a number takes MusicBrainz's number of its piece."""
    token = _token()
    opera = MusicBrainzWork(id=str(uuid4()), title=f"Gianni Schicchi {token}", type="Opera")
    aria = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Gianni Schicchi {token}: “O mio babbino caro” (Lauretta)",
        relations=[
            MusicBrainzRelation(type="parts", direction="backward", work=opera, ordering_key=11)
        ],
    )
    work_name = f"Gianni Schicchi {token}, Act I"
    track = await _add_titled_track(
        mass, f"{work_name}: No. 3, O mio babbino caro", work_name, [aria.id], movement_number=3
    )
    ctrl = _controller(mass, [aria, opera])

    await ctrl._add_track_work(track.item_id, [aria.id])

    stored, work = await _track_work(mass, track)
    assert work.mbid == opera.id
    assert (stored.movement_number, stored.movement_name) == (
        11,
        "“O mio babbino caro” (Lauretta)",
    )


@pytest.mark.parametrize("held", [False, True])
async def test_track_on_the_work_its_id_names_has_no_movement(
    mass: MusicAssistant, held: bool
) -> None:
    """A track placed on the work its one id names keeps no movement its title gives."""
    token = _token()
    (vocalise,) = _work_levels((f"Vocalise, op. 34 no. 14 {token}", "Song"))
    if held:
        await _add_work(mass, vocalise.title, mbid=vocalise.id)
    work_name = f"14 Romances, Op. 34 {token}"
    track = await _add_titled_track(
        mass,
        f"{work_name}: XIV. Vocalise",
        work_name,
        [vocalise.id],
        movement_number=14,
        movement_name="XIV. Vocalise",
    )
    ctrl = _controller(mass, [vocalise])

    await ctrl._add_track_work(track.item_id, [vocalise.id])

    stored, work = await _track_work(mass, track)
    assert work.mbid == vocalise.id
    assert (stored.movement_number, stored.movement_name) == (None, None)


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


async def test_track_performing_several_parts_of_an_act_is_a_recording_of_the_act(
    mass: MusicAssistant,
) -> None:
    """A track performing two arias of one act is placed on the act, added with its opera."""
    token = _token()
    first_aria, second_aria, act, opera = _tosca_act(token)
    work_ids = [first_aria.id, second_aria.id]
    work_name = f"Tosca {token}, Act I"
    track = await _add_titled_track(
        mass, f"{work_name}: Duet", work_name, work_ids, movement_number=3, movement_name="Duet"
    )
    ctrl = _controller(mass, [first_aria, second_aria, act, opera])

    await ctrl._add_track_work(track.item_id, set(work_ids))

    stored, work = await _track_work(mass, track)
    assert work.mbid == act.id
    assert work.parent_work is not None
    assert (await mass.music.works.get_library_item(work.parent_work.item_id)).mbid == opera.id
    assert (stored.movement_number, stored.movement_name) == (None, "Duet")
    assert {x for id_type, x in stored.external_ids if id_type == ExternalID.MB_WORK} == set(
        work_ids
    )


@pytest.mark.parametrize(
    ("aria_count", "movement"),
    [
        # the act and two of its arias, a recording of the act
        (2, False),
        # the act and one of its arias, a movement of the act
        (1, True),
    ],
)
async def test_track_naming_an_act_and_its_parts_is_placed_on_the_act(
    mass: MusicAssistant, aria_count: int, movement: bool
) -> None:
    """
    A track holding the id of an act and of two of its arias is a recording of the act.

    A track whose tags name one aria together with its act is a movement of the act.
    """
    token = _token()
    first_aria, second_aria, act, opera = _tosca_act(token)
    holder = await _add_work(mass, act.title, mbid=act.id)
    work_ids = [act.id, first_aria.id, second_aria.id][: 1 + aria_count]
    track = await _add_titled_track(mass, f"Tosca {token}: Act I", f"Tosca {token}", work_ids)
    ctrl = _controller(mass, [first_aria, second_aria, act, opera])

    await ctrl._add_track_work(track.item_id, set(work_ids))

    stored, work = await _track_work(mass, track)
    assert work.item_id == holder.item_id
    assert (stored.movement_number is not None) is movement


async def test_track_performing_parts_of_two_acts_gets_no_work(mass: MusicAssistant) -> None:
    """A track performing an aria of each of two acts is left without a work."""
    token = _token()
    (first_aria, first_act, opera), (second_aria, second_act, _) = _boheme(token)
    track = await _add_titled_track(
        mass, f"La bohème {token}: Arias", f"La bohème {token}", [first_aria.id, second_aria.id]
    )
    ctrl = _controller(mass, [first_aria, first_act, second_aria, second_act, opera])

    await ctrl._add_track_work(track.item_id, {first_aria.id, second_aria.id})

    assert (await mass.music.tracks.get_library_item(track.item_id)).work is None


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


async def test_added_track_work_of_an_aria_is_its_act(mass: MusicAssistant) -> None:
    """A classical track without a work performing an aria joins the act the library holds."""
    token = _token()
    (aria, act_level, opera), _ = _boheme(token)
    act = await _add_work(
        mass, f"La bohème {token}, Act I", mbid=act_level.id, work_type=WorkType.OPERA
    )
    track = await _add_track(mass, act, [aria.id])
    # the track keeps its classical flag, as a track whose work was removed does
    await _clear_track_work(mass, track)
    ctrl = _controller(mass, [aria, act_level, opera])
    before = await _library_work_ids(mass)

    await ctrl._add_track_work(track.item_id, [aria.id])

    stored, stored_work = await _track_work(mass, track)
    assert (stored_work.item_id, stored.movement_number) == (act.item_id, 13)
    assert await _library_work_ids(mass) == before


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


async def test_sole_composer_gets_its_period_from_the_musicbrainz_life_dates(
    mass: MusicAssistant,
) -> None:
    """An identified sole composer takes its life dates from MusicBrainz, without a refresh."""
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    beethoven.type = "Person"
    beethoven.life_span = MusicBrainzLifeSpan(begin="1770-12-17", end="1827-03-26", ended=True)
    composer = await _add_artist(mass, f"Beethoven {token}")
    work, _, ctrl = await _add_symphony_5(
        mass, symphony, movement, [composer], composer, [beethoven]
    )
    cast("Mock", ctrl._musicbrainz_provider()).get_artists_by_name = AsyncMock(return_value=[])

    await _enrich(ctrl, work)

    stored = await mass.music.artists.get_library_item(composer.item_id)
    assert stored.mbid == beethoven.id
    assert stored.period == composer_period(1770, 1827)
    assert _scheduled(ctrl) == []


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
    token = _token()
    symphony, movement, _ = _symphony_5(token)
    relative = await _add_artist(mass, f"Karl van Beethoven {token}", uuid4().hex)
    relative_work = await _add_work(
        mass, f"Symphony no. 1 {token}", [relative], work_type=WorkType.SYMPHONY
    )
    await _add_track(mass, relative_work, [], relative)
    assert relative.item_id in {
        x.item_id for x in await mass.music.artists.get_library_composers(with_mbid=True)
    }
    composer = await _add_artist(mass, f"Beethoven {token}")
    work, _, ctrl = await _add_symphony_5(mass, symphony, movement, [composer], composer)

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


_SEARCH_HINT = MusicBrainzAlias("Tschaikowsky", "Tschaikowsky", type="Search hint")


@pytest.mark.parametrize(
    ("candidate", "aliases", "others", "namesakes", "holders", "merged_into"),
    [
        # carrying an alias of the composer
        ("Pyotr Ilyich Tchaikovsky", [], [], [], [], "candidate"),
        # only part of an alias, as a family name is
        ("Tchaikovsky", [], [], [], [], None),
        # only a search hint
        ("Tschaikowsky", [_SEARCH_HINT], [], [], [], None),
        # another confirmed library composer carries the name too
        ("Pyotr Ilyich Tchaikovsky", [], ["Pyotr Ilyich Tchaikovsky Jr"], [], [], None),
        # another MusicBrainz artist carries it too, as father and son can
        ("Pyotr Ilyich Tchaikovsky", [], [], [None], [], None),
        ("Pyotr Ilyich Tchaikovsky", [], [], ["Person"], [], None),
        # a band of the same name is no composer
        ("Pyotr Ilyich Tchaikovsky", [], [], ["Group"], [], "candidate"),
        # holders of the id, one by a name that is no composer name
        (None, [], [], [], ["Tchaikovsky", "Pyotr Ilyich Tchaikovsky"], "holder"),
        # holders by other names
        (
            "Pyotr Ilyich Tchaikovsky",
            [],
            [],
            [],
            ["Пётр Ильич Чайковский", "Tchaikovsky"],
            "candidate",
        ),
    ],
)
async def test_artist_named_by_an_alias_takes_the_composer_id_and_merges(
    mass: MusicAssistant,
    candidate: str | None,
    aliases: list[MusicBrainzAlias],
    others: list[str],
    namesakes: list[str | None],
    holders: list[str],
    merged_into: str | None,
) -> None:
    """
    A classical artist without an id carrying an alias of the composer takes its id and merges.

    A name that is only part of an alias, or only a search hint, merges nothing. Nor does a name
    another confirmed library composer, or another MusicBrainz person, also carries; a band of
    the name is no composer and does not keep it apart. Artists already holding the composer's id
    merge whatever their names, and a holder by another name leaves the join intact.

    :param namesakes: The types of the other MusicBrainz artists a name search finds.
    :param merged_into: Whether the candidate or the first holder is merged into, if either.
    """
    mb_artist = _tchaikovsky(_tchaikovsky_alias(), *aliases)
    unconfirmed = [_classical_artist(candidate)] if candidate else []
    holding = [_classical_artist(x, mb_artist.id) for x in holders]
    confirmed = [*holding, *(_classical_artist(x, str(uuid4())) for x in others)]
    found = [
        mb_artist,
        *(
            MusicBrainzArtist(
                id=str(uuid4()),
                name="Pyotr Ilyich Tchaikovsky",
                sort_name="Tchaikovsky, Pyotr Ilyich",
                type=x,
            )
            for x in namesakes
        ),
    ]

    merge = await _merge_by_aliases(mass, mb_artist, unconfirmed, confirmed, holding, found)

    if merged_into == "holder":
        merge.assert_awaited_once_with(int(holding[0].item_id), mb_artist.id)
    elif merged_into == "candidate":
        assert unconfirmed[0].mbid == mb_artist.id
        merge.assert_awaited_once_with(int(unconfirmed[0].item_id), mb_artist.id)
    else:
        assert unconfirmed[0].mbid is None
        merge.assert_not_awaited()


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
            ("Frédéric Chopin", "Chopin, Frédéric"),
        ),
        (
            MusicBrainzArtist(
                id="gorecki",
                name="Henryk Mikołaj Górecki",
                sort_name="Górecki, Henryk Mikołaj",
                aliases=[
                    MusicBrainzAlias(
                        "Henryk Mikolaj Gorecki",
                        "Gorecki, Henryk Mikolaj",
                        locale="en",
                        primary=True,
                    )
                ],
            ),
            ("Henryk Mikołaj Górecki", "Górecki, Henryk Mikołaj"),
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
        "english alias over the latin main name",
        "latin main name over its english alias without accents",
    ],
)
def test_musicbrainz_name_is_in_latin_script(
    mb_artist: MusicBrainzArtist, expected: tuple[str, str] | None
) -> None:
    """An artist's MusicBrainz name is its primary alias, else its main name, in Latin script."""
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


@pytest.mark.parametrize(
    ("language", "name", "sort_name"),
    [
        ("en", "Pyotr Ilyich Tchaikovsky", "Tchaikovsky, Pyotr Ilyich"),
        ("de", "Peter Tschaikowsky", "Tschaikowsky, Peter"),
    ],
)
async def test_composers_sharing_the_id_are_joined_under_the_musicbrainz_name(
    mass: MusicAssistant, language: str, name: str, sort_name: str
) -> None:
    """
    Two library artists holding a composer's id are joined and stored by its MusicBrainz name.

    The name is the one MusicBrainz gives in the user's language.
    """
    token = _token()
    name, sort_name = f"{name} {token}", f"{sort_name} {token}"
    mb_artist = _tchaikovsky_in(
        ("de", f"Peter Tschaikowsky {token}", f"Tschaikowsky, Peter {token}"),
        ("en", f"Pyotr Ilyich Tchaikovsky {token}", f"Tchaikovsky, Pyotr Ilyich {token}"),
    )
    oldest = await _add_classical_composer(mass, f"Tchaikovsky {token}", mb_artist.id)
    later = await _add_classical_composer(mass, f"P. I. Tchaikovsky {token}")
    later.mbid = mb_artist.id
    await mass.music.artists.update_item_in_library(later.item_id, later)
    assert later.item_id != oldest.item_id
    ctrl = _controller(mass, [], [mb_artist])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[])

    with patch.object(
        MetaDataController, "preferred_language", new_callable=PropertyMock, return_value=language
    ):
        await ctrl._merge_composer_by_aliases(mb_artist.id)

    holders = await mass.music.artists.get_library_items_by_external_id(
        mb_artist.id, ExternalID.MB_ARTIST, limit=None
    )
    assert [(x.item_id, x.name, x.sort_name) for x in holders] == [
        (oldest.item_id, name, sort_name)
    ]
    rows = await mass.music.database.get_rows_from_query(
        "SELECT search_name, search_sort_name FROM artists WHERE item_id = :item_id",
        {"item_id": int(oldest.item_id)},
    )
    assert (rows[0]["search_name"], rows[0]["search_sort_name"]) == (
        create_safe_string(name, True, True),
        create_safe_string(sort_name, True, True),
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


async def test_composer_the_library_lacks_is_added_through_its_provider_link(
    mass: MusicAssistant,
) -> None:
    """
    A MusicBrainz composer linked to a loaded music provider is added and credited.

    It takes the name MusicBrainz gives it in Latin script.
    """
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    latin = (beethoven.name, beethoven.sort_name)
    beethoven.name = f"Людвиг ван Бетховен {token}"
    spotify_id = uuid4().hex[:22]
    linked = _linked_to_spotify(beethoven, spotify_id)
    linked.aliases = [MusicBrainzAlias(*latin, locale="en", primary=True)]
    linked.life_span = MusicBrainzLifeSpan(begin="1770-12-17", end="1827-03-26", ended=True)
    work, track, ctrl = await _add_symphony_5(mass, symphony, movement, artists=[linked])

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    assert [(x.name, x.mbid) for x in stored.composers] == [(latin[0], beethoven.id)]
    composer = await mass.music.artists.get_library_item(stored.composers[0].item_id)
    assert (composer.name, composer.sort_name) == latin
    assert [(x.provider_instance, x.item_id, x.in_library) for x in composer.provider_mappings] == [
        ("spotify", spotify_id, False)
    ]
    assert composer.metadata.life_span == LifeSpan(begin="1770-12-17", end="1827-03-26", ended=True)
    assert _composers(await mass.music.tracks.get_library_item(track.item_id)) == [latin[0]]


async def test_composer_without_a_loaded_provider_link_is_not_added(
    mass: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A MusicBrainz composer linked to no loaded music provider is not added.

    It is logged with its MusicBrainz page, to add a link to a loaded provider there.
    """
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    work, _, ctrl = await _add_symphony_5(
        mass, symphony, movement, artists=[_linked_to_spotify(beethoven, uuid4().hex[:22])]
    )
    ctrl.logger = logging.getLogger("test_work_enrichment")
    caplog.set_level(logging.DEBUG, logger="test_work_enrichment")

    def _instances(domain: str, **_kwargs: Any) -> list[Mock]:
        if domain not in ("qobuz", "tidal"):
            return []
        instance = Mock(available=True, instance_id=domain)
        instance.manifest.name = domain.capitalize()
        return [instance]

    with patch.object(mass.music, "get_provider_instances", Mock(side_effect=_instances)):
        stored = await _enrich(ctrl, work)

    assert stored.composers == []
    assert not await mass.music.artists.get_library_items_by_query(search=token)
    assert (
        f'Artist "{beethoven.name}" has no link to Tidal, Qobuz on MusicBrainz, so it is not '
        f"added; add one at https://musicbrainz.org/artist/{beethoven.id}"
    ) in caplog.messages


async def test_library_artist_on_the_linked_provider_artist_composes_the_work(
    mass: MusicAssistant,
) -> None:
    """A library artist mapped to the provider artist MusicBrainz links to is not duplicated."""
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
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
    work, _, ctrl = await _add_symphony_5(
        mass, symphony, movement, artists=[_linked_to_spotify(beethoven, spotify_id)]
    )

    with patch.object(mass.music, "get_provider_instances", _spotify_instances()):
        stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [artist.item_id]
    assert len(await mass.music.artists.get_library_items_by_query(search=token)) == 1


async def test_namesake_composer_is_replaced_by_the_musicbrainz_composer(
    mass: MusicAssistant,
) -> None:
    """A work and its tracks credited to a namesake of MusicBrainz's composer take the composer."""
    token = _token()
    symphony, movement, composer = _symphony_5(token)
    namesake_mbid = str(uuid4())
    namesake = await _add_artist(mass, composer.name, namesake_mbid)
    work, track, ctrl = await _add_symphony_5(
        mass,
        symphony,
        movement,
        [namesake],
        namesake,
        [_linked_to_spotify(composer, uuid4().hex[:22])],
        name=f"Hansel and Gretel {token}",
    )
    other_work = await _add_work(mass, f"Release Me {token}", [namesake])
    other_track = await _add_track(mass, other_work, composer=namesake)

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
    token = _token()
    symphony, movement, composer = _symphony_5(token)
    namesake = await _add_artist(mass, composer.name, str(uuid4()), period=Period.CONTEMPORARY)
    work = await _add_work(
        mass, f"Hansel and Gretel {token}", [namesake], symphony.id, work_type=WorkType.OPERA
    )
    track = await mass.music.tracks.add_item_to_library(
        Track(
            item_id=token,
            provider=PROVIDER_INSTANCE,
            name=f"Overture {token}",
            provider_mappings={_mapping()},
            artists=UniqueList([namesake]),
            credits=[Credit(artist=namesake, role=ArtistRole.COMPOSER)],
            work=ItemMapping.from_item(work),
            movement_name="Overture",
            external_ids={(ExternalID.MB_WORK, movement.id)},
        )
    )
    assert (await mass.music.artists.get_library_item(namesake.item_id)).is_classical
    ctrl = _controller(mass, [symphony, movement], [_linked_to_spotify(composer, uuid4().hex[:22])])

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
    token = _token()
    symphony, movement, composer = _symphony_5(token)
    namesake = await _add_artist(mass, composer.name, str(uuid4()))
    work, track, ctrl = await _add_symphony_5(
        mass,
        symphony,
        movement,
        [namesake],
        namesake,
        [_linked_to_spotify(composer, uuid4().hex[:22])],
        name=f"Hansel and Gretel {token}",
    )

    stored = await _enrich(ctrl, work)

    assert [x.item_id for x in stored.composers] == [namesake.item_id]
    assert _composer_ids(await mass.music.tracks.get_library_item(track.item_id)) == [
        namesake.item_id
    ]


async def test_composer_holding_the_musicbrainz_id_is_kept(mass: MusicAssistant) -> None:
    """A composer holding the MusicBrainz composer's id stays on the work and its tracks."""
    token = _token()
    symphony, movement, mb_composer = _symphony_5(token)
    composer = await _add_artist(mass, mb_composer.name, mb_composer.id)
    work, track, ctrl = await _add_symphony_5(
        mass,
        symphony,
        movement,
        [composer],
        composer,
        [_linked_to_spotify(mb_composer, uuid4().hex[:22])],
    )

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
    """
    A composer without an id is due until looked up, one its tags identified until read once.

    A performer is never due.
    """
    stale = int(time()) - REFRESH_INTERVAL - 1
    never = await _add_classical_composer(mass, f"Haydn {_token()}")
    stale_unknown = await _add_classical_composer(
        mass, f"Liszt {_token()}", metadata=MediaItemMetadata(last_musicbrainz_lookup=stale)
    )
    fresh = await _add_classical_composer(
        mass, f"Rossini {_token()}", metadata=MediaItemMetadata(last_musicbrainz_lookup=int(time()))
    )
    identified = await _add_classical_composer(mass, f"Fauré {_token()}", mbid=str(uuid4()))
    identified_read = await _add_classical_composer(
        mass,
        f"Ravel {_token()}",
        mbid=str(uuid4()),
        metadata=MediaItemMetadata(last_musicbrainz_lookup=stale),
    )
    work = await _add_work(mass, f"Symphony {_token()}", work_type=WorkType.SYMPHONY)
    track = await _add_track(mass, work)
    performer = track.artists[0]

    due = await _due_composers(mass)

    assert {never.item_id, stale_unknown.item_id, identified.item_id} <= due
    assert not {fresh.item_id, identified_read.item_id, performer.item_id} & due


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


async def test_composer_linked_before_it_is_classical_gets_its_period_once_classical(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artist linked before it counts as classical takes its life dates, its period later."""
    token = _token()
    albinoni = _albinoni(token)
    albinoni.life_span = MusicBrainzLifeSpan(begin="1671-06-08", end="1751-01-17", ended=True)
    artist = await _add_artist(mass, albinoni.name, albinoni.id)
    ctrl = _controller(mass, [], [albinoni])
    monkeypatch.setattr(mass.metadata, "set_composer_periods", ctrl.set_composer_periods)

    with patch.object(
        MetaDataController,
        "link_providers_via_musicbrainz",
        new_callable=PropertyMock,
        return_value=False,
    ):
        await ctrl.link_item_to_musicbrainz(artist)

    linked = await mass.music.artists.get_library_item(artist.item_id)
    assert linked.metadata.life_span is not None
    assert linked.metadata.life_span.begin == "1671-06-08"
    assert linked.period is None

    work = await _add_work(mass, f"Concerto {token}", [linked], work_type=WorkType.CONCERTO)
    await _add_track(mass, work, composer=linked)

    stored = await mass.music.artists.get_library_item(artist.item_id)
    assert stored.period == composer_period(1671, 1751)


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


@pytest.mark.parametrize("by_alias", [False, True], ids=["main name", "alias"])
async def test_confirmed_composer_splits_the_joined_credits_of_stored_tracks(
    mass: MusicAssistant, by_alias: bool
) -> None:
    """
    A composer confirmed after a track was stored replaces the joined name it starts.

    The joined name may start with the composer's MusicBrainz name or with an alias of it. A
    name that only starts like the composer's stays.
    """
    token = _token()
    band = await _add_artist(mass, f"Frédéric Chopin {token} & Friends")
    party = await _add_joined_track(mass, "Party Mix", band)

    chopin, stored = await _split_joined_credits(mass, token, by_alias)

    assert _composer_ids(stored) == [chopin.item_id]
    assert [x.item_id for x in stored.artists] == [chopin.item_id]
    untouched = await mass.music.tracks.get_library_item(party.item_id)
    assert _composer_ids(untouched) == [band.item_id]
    assert [x.item_id for x in untouched.artists] == [band.item_id]


@pytest.mark.parametrize(
    ("composer", "own_id", "looked_up", "split"),
    [
        (True, False, False, True),
        (False, False, False, False),
        # an id the artist came with, as from a file's tags, on its first lookup
        (True, True, False, True),
        (True, True, True, False),
    ],
)
async def test_composer_the_artist_link_confirms_splits_the_joined_credits(
    mass: MusicAssistant, composer: bool, own_id: bool, looked_up: bool, split: bool
) -> None:
    """A classical composer the artist link step confirms takes the joined credits naming it."""
    token = _token()
    mb_chopin = MusicBrainzArtist(
        id=str(uuid4()), name=f"Frédéric Chopin {token}", sort_name=f"Frédéric Chopin {token}"
    )
    if composer:
        chopin = await _add_classical_composer(mass, mb_chopin.name)
    else:
        chopin = await _add_artist(mass, mb_chopin.name)
    joined = await _add_artist(mass, f"Frédéric Chopin {token} & Sarah Ainsworth {token}")
    # stored while the composer is not confirmed, so saving it splits nothing
    nocturne = await _add_joined_track(mass, f"Chopin {token}: Nocturne No. 2", joined)
    if own_id:
        chopin.mbid = mb_chopin.id
        if looked_up:
            chopin.metadata.last_musicbrainz_lookup = int(time())
        await mass.music.artists.update_item_in_library(chopin.item_id, chopin)
    assert _composer_ids(await mass.music.tracks.get_library_item(nocturne.item_id)) == [
        joined.item_id
    ]
    ctrl = _controller(mass, [], [mb_chopin])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[mb_chopin])

    with patch.object(
        MetaDataController, "_get_artist_mbid", AsyncMock(return_value=(mb_chopin.id, False))
    ):
        await ctrl.link_item_to_musicbrainz(
            await mass.music.artists.get_library_item(chopin.item_id)
        )

    stored = await mass.music.tracks.get_library_item(nocturne.item_id)
    assert _composer_ids(stored) == [chopin.item_id if split else joined.item_id]


async def _split_joined_credits(
    mass: MusicAssistant, token: str, by_alias: bool = True, other: str | None = None
) -> tuple[Artist, Track]:
    """
    Confirm Chopin after storing a track crediting him with Sarah Ainsworth, return both.

    :param token: The word unique to the test.
    :param by_alias: Name him "Fryderyk Chopin" on MusicBrainz, the track's "Frédéric Chopin"
        an alias of it.
    :param other: The name of another confirmed composer added after the track, if any.
    """
    name = f"Frédéric Chopin {token}"
    mb_chopin = MusicBrainzArtist(id=str(uuid4()), name=name, sort_name=name)
    if by_alias:
        mb_chopin.name, mb_chopin.sort_name = (
            f"Fryderyk Chopin {token}",
            f"Chopin {token}, Fryderyk",
        )
        mb_chopin.aliases = [
            MusicBrainzAlias(name, f"Chopin {token}, Frédéric", type="Artist name")
        ]
    chopin = await _add_classical_composer(mass, mb_chopin.name)
    joined = await _add_artist(mass, f"Frédéric Chopin {token} & Sarah Ainsworth {token}")
    nocturne = await _add_joined_track(mass, f"Chopin {token}: Nocturne No. 2", joined)
    chopin.mbid = mb_chopin.id
    await mass.music.artists.update_item_in_library(chopin.item_id, chopin)
    if other:
        await _add_classical_composer(mass, other, str(uuid4()))
    ctrl = _controller(mass, [], [mb_chopin])
    musicbrainz = cast("Mock", ctrl._musicbrainz_provider())
    musicbrainz.get_artists_by_name = AsyncMock(return_value=[mb_chopin])

    await ctrl._merge_composer_by_aliases(mb_chopin.id)

    return chopin, await mass.music.tracks.get_library_item(nocturne.item_id)


async def test_alias_split_leaves_a_name_another_confirmed_composer_carries(
    mass: MusicAssistant,
) -> None:
    """A joined name another confirmed composer carries every word of stays as it is."""
    token = _token()
    joined = f"Frédéric Chopin {token} & Sarah Ainsworth {token}"

    _chopin, stored = await _split_joined_credits(
        mass, token, other=f"Frédéric Chopin {token} the Younger"
    )

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


@pytest.mark.parametrize(("days_ago", "due"), [(10, False), (100, True)])
async def test_works_and_classical_artists_are_read_again_after_90_days(
    mass: MusicAssistant, days_ago: int, due: bool
) -> None:
    """A work or classical artist with a MusicBrainz id is read again 90 days after its lookup."""
    token = _token()
    composer = await _add_classical_composer(mass, f"Composer {token}", str(uuid4()))
    work = await _add_work(mass, f"Work {token}", mbid=str(uuid4()))
    pop = await _add_artist(mass, f"Singer {token}", str(uuid4()))
    read = int(time()) - days_ago * 86400
    for table, item in ((DB_TABLE_WORKS, work), ("artists", composer), ("artists", pop)):
        await mass.music.database.execute_write(
            f"UPDATE {table} SET metadata = json_set(COALESCE(metadata, '{{}}'), "
            "'$.last_musicbrainz_lookup', :read) WHERE item_id = :item_id",
            {"read": read, "item_id": int(item.item_id)},
        )
    stale = {"stale": int(time() - REFRESH_INTERVAL)}

    works = await mass.music.works.get_library_items_by_query(
        limit=0, extra_query_parts=[_works_to_reread_query()], extra_query_params=stale
    )
    artists = await mass.music.artists.get_library_items_by_query(
        limit=0, extra_query_parts=[_classical_artists_to_reread_query()], extra_query_params=stale
    )

    assert (work.item_id in {x.item_id for x in works}) is due
    selected = {x.item_id for x in artists}
    assert (composer.item_id in selected) is due
    # an artist that is not classical is left to its own refresh
    assert pop.item_id not in selected


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
