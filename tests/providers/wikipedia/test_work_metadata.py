"""Tests for the notes on a classical work, and an artist's bio, from the Wikipedia provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from music_assistant_models.enums import ExternalID, LinkType
from music_assistant_models.media_items import Artist, MediaItemLink, MediaItemMetadata, Work

from music_assistant.providers.musicbrainz.models import (
    MusicBrainzArtist,
    MusicBrainzRelation,
    MusicBrainzUrl,
    MusicBrainzWork,
)
from music_assistant.providers.wikipedia import SUPPORTED_FEATURES, WikipediaMetadataProvider
from tests.common import use_real_create_task

WORK_MBID = "d03bff61-26fc-301b-98ac-4d8e85771cbc"
ARTIST_MBID = "1f9df192-a621-4f54-8850-2c5373b7eac9"
QID = "Q200298"
EN_TITLE = "Symphony No. 5 (Beethoven)"
DE_TITLE = "5. Sinfonie (Beethoven)"
EN_INTRO = "The Symphony No. 5 in C minor, Op. 67, is a symphony by Ludwig van Beethoven."
DE_INTRO = "Die 5. Sinfonie in c-Moll op. 67 ist eine Sinfonie von Ludwig van Beethoven."


def _url_relation(relation_type: str, resource: str) -> MusicBrainzRelation:
    """Return a MusicBrainz URL relation of the given type."""
    return MusicBrainzRelation(type=relation_type, url=MusicBrainzUrl(resource=resource))


def _work(mbid: str | None = WORK_MBID) -> Work:
    """Return a library work, identified on MusicBrainz unless no id is given."""
    return Work(
        item_id="1",
        provider="library",
        name="Symphony no. 5",
        provider_mappings=set(),
        external_ids={(ExternalID.MB_WORK, mbid)} if mbid else set(),
    )


async def _provider(
    relations: list[MusicBrainzRelation],
    preferred_language: str = "en",
    sitelinks: dict[str, str] | None = None,
    intros: dict[str, str] | None = None,
) -> tuple[WikipediaMetadataProvider, AsyncMock]:
    """
    Return a Wikipedia provider whose MusicBrainz work and artist link to Wikipedia as given.

    :param relations: The URL relations of the work and the artist on MusicBrainz.
    :param preferred_language: The user's preferred language.
    :param sitelinks: Wikidata's article titles by language.
    :param intros: Wikipedia's article intros by language.
    """
    mass = MagicMock()
    mass.metadata.preferred_language = preferred_language
    mass.cache.get_with_freshness = AsyncMock(return_value=(None, False, False))
    mass.cache.set = AsyncMock()
    use_real_create_task(mass)
    musicbrainz = MagicMock()
    musicbrainz.get_work_details = AsyncMock(
        return_value=MusicBrainzWork(id=WORK_MBID, title="Symphony no. 5", relations=relations)
    )
    musicbrainz.get_artist_details = AsyncMock(
        return_value=MusicBrainzArtist(
            id=ARTIST_MBID, name="Beethoven", sort_name="Beethoven", relations=relations
        )
    )
    mass.get_provider.return_value = musicbrainz
    manifest = MagicMock()
    manifest.domain = "wikipedia"
    config = MagicMock()
    config.get_value.return_value = None
    provider = WikipediaMetadataProvider(mass, manifest, config, SUPPORTED_FEATURES)
    await provider.handle_async_init()

    async def _answer(url: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
        assert params is not None
        if url.startswith("https://www.wikidata.org"):
            links = {f"{lang}wiki": {"title": title} for lang, title in (sitelinks or {}).items()}
            return {"entities": {QID: {"sitelinks": links}}}
        lang = url.removeprefix("https://").split(".", 1)[0]
        if not (extract := (intros or {}).get(lang)):
            return None
        return {"query": {"pages": {"1": {"title": params["titles"], "extract": extract}}}}

    get_json = AsyncMock(side_effect=_answer)
    provider._get_json = get_json  # type: ignore[method-assign]
    return provider, get_json


async def test_work_notes_are_the_intro_in_the_preferred_language() -> None:
    """A work's notes are the Wikipedia intro in the user's language, found through Wikidata."""
    provider, _ = await _provider(
        [
            _url_relation("wikipedia", f"https://en.wikipedia.org/wiki/{EN_TITLE}"),
            _url_relation("wikidata", f"https://www.wikidata.org/wiki/{QID}"),
        ],
        preferred_language="de",
        sitelinks={"de": DE_TITLE},
        intros={"de": DE_INTRO, "en": EN_INTRO},
    )

    assert await provider.get_work_metadata(_work()) == MediaItemMetadata(
        description=DE_INTRO,
        description_language="de",
        links={
            MediaItemLink(
                type=LinkType.WIKIPEDIA, url="https://de.wikipedia.org/wiki/5._Sinfonie_(Beethoven)"
            )
        },
    )


async def test_work_notes_fall_back_to_english() -> None:
    """Without an article in the user's language, a work's notes are the English intro."""
    provider, _ = await _provider(
        [_url_relation("wikipedia", f"https://en.wikipedia.org/wiki/{EN_TITLE}")],
        preferred_language="de",
        intros={"en": EN_INTRO},
    )

    metadata = await provider.get_work_metadata(_work())

    assert metadata is not None
    assert (metadata.description, metadata.description_language) == (EN_INTRO, "en")
    assert {x.url for x in metadata.links or ()} == {
        f"https://en.wikipedia.org/wiki/{EN_TITLE.replace(' ', '_')}"
    }


async def test_work_without_musicbrainz_id_has_no_notes() -> None:
    """A work not identified on MusicBrainz is not looked up."""
    provider, get_json = await _provider(
        [_url_relation("wikipedia", f"https://en.wikipedia.org/wiki/{EN_TITLE}")],
        intros={"en": EN_INTRO},
    )

    assert await provider.get_work_metadata(_work(mbid=None)) is None
    get_json.assert_not_awaited()


async def test_work_without_a_wiki_link_has_no_notes() -> None:
    """A work MusicBrainz links to no Wikipedia article or Wikidata item has no notes."""
    provider, get_json = await _provider(
        [_url_relation("score", "https://imslp.org/wiki/Symphony_No.5,_Op.67_(Beethoven)")],
        intros={"en": EN_INTRO},
    )

    assert await provider.get_work_metadata(_work()) is None
    get_json.assert_not_awaited()


async def test_artist_bio_is_the_intro_in_the_preferred_language() -> None:
    """An artist's bio is the Wikipedia intro in the user's language, as a work's notes are."""
    provider, _ = await _provider(
        [_url_relation("wikidata", f"https://www.wikidata.org/wiki/{QID}")],
        preferred_language="de",
        sitelinks={"de": DE_TITLE, "en": EN_TITLE},
        intros={"de": DE_INTRO, "en": EN_INTRO},
    )
    artist = Artist(
        item_id="1",
        provider="library",
        name="Beethoven",
        provider_mappings=set(),
        external_ids={(ExternalID.MB_ARTIST, ARTIST_MBID)},
    )

    assert await provider.get_artist_metadata(artist) == MediaItemMetadata(
        description=DE_INTRO, description_language="de"
    )
