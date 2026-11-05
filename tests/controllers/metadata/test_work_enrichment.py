"""Tests for filling library works in from MusicBrainz during the MusicBrainz link run."""

from __future__ import annotations

import json
import pathlib
import re
from time import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock
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

from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.constants import REFRESH_INTERVAL
from music_assistant.controllers.metadata.controller import _works_to_enrich_query
from music_assistant.controllers.metadata.helpers import musicbrainz_name
from music_assistant.controllers.metadata.work_enrichment import _fill_work_from_musicbrainz
from music_assistant.controllers.music.helpers import composer_period
from music_assistant.helpers.compare import is_named_by
from music_assistant.providers.musicbrainz.models import (
    MusicBrainzAlias,
    MusicBrainzArtist,
    MusicBrainzLifeSpan,
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


async def _enrich(ctrl: MetaDataController, work: Work) -> Work:
    """Run the link step on a library work and return the work as stored afterwards."""
    await ctrl.link_item_to_musicbrainz(await ctrl.mass.music.works.get_library_item(work.item_id))
    return await ctrl.mass.music.works.get_library_item(work.item_id)


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


async def test_work_looked_up_first_takes_the_musicbrainz_name(mass: MusicAssistant) -> None:
    """A work looked up for the first time takes MusicBrainz's name, in English."""
    token = _token()
    seventh, fourth, preludes = _preludes(token)
    preludes.aliases = [
        MusicBrainzAlias(f"24 Preludes, Op. 28 {token}", f"24 Preludes {token}", "en", None, True)
    ]
    # named after the key of a part the tracks perform, not the set
    work = await _add_work(mass, f"Prelude In A Major, Op 28 {token}", mbid=preludes.id)
    await _add_track(mass, work, [seventh.id])
    await _add_track(mass, work, [fourth.id])
    ctrl = _controller(mass, [seventh, fourth, preludes])

    stored = await _enrich(ctrl, work)

    assert (stored.mbid, stored.name) == (preludes.id, f"24 Preludes, Op. 28 {token}")


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
    assert stored.sort_name == concerto.title.lower()
    found = await mass.music.works.library_items(search=concerto.title)
    assert work.item_id in {x.item_id for x in found}


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


async def test_work_musicbrainz_does_not_know_is_marked_looked_up(mass: MusicAssistant) -> None:
    """A work whose id MusicBrainz does not know is left as it is and not looked up again."""
    work = await _add_work(mass, f"Unknown {_token()}", mbid=str(uuid4()))
    ctrl = _controller(mass, [])

    stored = await _enrich(ctrl, work)

    assert stored.work_type is None
    assert stored.metadata.last_musicbrainz_lookup is not None
    cast("Mock", ctrl.schedule_update_metadata).assert_not_called()


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


def _tchaikovsky_alias() -> MusicBrainzAlias:
    """Return the English artist name alias of Tchaikovsky."""
    return MusicBrainzAlias(
        name="Pyotr Ilyich Tchaikovsky", sort_name="Tchaikovsky, Pyotr Ilyich", type="Artist name"
    )


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
    assert musicbrainz_name(mb_artist, "en") == expected


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
    assert (await mass.music.artists.get_library_item(composer.item_id)).is_classical


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


async def test_composer_with_life_dates_gets_a_period_right_away(mass: MusicAssistant) -> None:
    """A classical composer whose life dates are stored gets a period without a refresh."""
    token = _token()
    symphony, movement, beethoven = _symphony_5(token)
    composer = await _add_artist(mass, f"Beethoven {token}", beethoven.id, metadata=_BEETHOVEN)
    work, _, ctrl = await _add_symphony_5(mass, symphony, movement, [composer], composer)

    await _enrich(ctrl, work)

    stored = await mass.music.artists.get_library_item(composer.item_id)
    assert stored.period == Period.CLASSICAL
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
    """A work looked up again takes MusicBrainz's name, version, type, key and catalogue numbers."""
    token = _token()
    mb_work = MusicBrainzWork(
        id=str(uuid4()),
        title=f"Sonata {token}, BWV 1",
        type="Sonata",
        disambiguation="first version",
        attributes=[MusicBrainzWorkAttribute(type="Key", value="C minor")],
        relations=[_catalogue("BWV 1")],
    )
    work = await _add_work(mass, mb_work.title, mbid=mb_work.id)
    ctrl = _controller(mass, [mb_work])
    await _enrich(ctrl, work)
    # what changed on MusicBrainz since
    mb_work.title = f"Concerto {token}, BWV 2"
    mb_work.type = "Concerto"
    mb_work.disambiguation = "second version"
    mb_work.attributes = [MusicBrainzWorkAttribute(type="Key", value="D major")]
    mb_work.relations = [_catalogue("BWV 2")]

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
