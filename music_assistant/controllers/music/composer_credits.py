"""
Composer credit clean-up of classical tracks for the Music controller.

A composer credited joined with others, as in "Chopin & X", is credited on its own when the
joined name starts with the name of one confirmed library composer. A composer a label credits
as artist next to a placeholder composer, as in "Dr, Composer", becomes the composer of its track.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant_models.enums import ArtistRole
from music_assistant_models.media_items import Credit, UniqueList, Work

from music_assistant.constants import (
    DB_TABLE_TRACK_ARTISTS,
    DB_TABLE_TRACK_CREDITS,
    DB_TABLE_WORK_ARTISTS,
)
from music_assistant.helpers.compare import is_named_by
from music_assistant.helpers.credit_roles import PERFORMING_ROLES, role_word
from music_assistant.helpers.util import is_placeholder_artist
from music_assistant.helpers.work_titles import joined_composer_name

if TYPE_CHECKING:
    from music_assistant_models.media_items import Artist, Track

    from music_assistant import MusicAssistant


async def split_joined_composers(mass: MusicAssistant, track: Track) -> set[str]:
    """
    Credit the library composer a joined composer name starts with, as in "Chopin & X".

    The composer also takes the place of the joined name among the main artists.
    Return the joined names that were replaced.

    :param mass: The MusicAssistant instance.
    :param track: The track to add or update with.
    """
    if not (candidates := _joined_composer_candidates(track)):
        return set()
    replacements: dict[str, Artist] = {}
    for name, candidate in candidates.items():
        library_composers = await mass.music.artists.get_library_composers(
            with_mbid=True, named_by=candidate
        )
        matches = [x for x in library_composers if is_named_by(candidate, x)]
        # a name several composers share, such as "Strauss", stays as it is
        if len(matches) == 1:
            replacements[name] = matches[0]
    return _replace_joined_composers(track, replacements)


async def credit_performing_composer(
    mass: MusicAssistant, track: Track, cur_item: Track | None = None
) -> set[str]:
    """
    Credit a track with a placeholder composer to the composer it credits as artist.

    Some labels give a placeholder composer, as in "Dr, Composer", and credit the real one as
    performer, as in "Antonio Vivaldi, Orchestra". A main artist or performer named like
    exactly one confirmed library composer becomes the composer of the track, and of its work
    when that names none.
    The performing credits of that artist are dropped, the main artist is kept.
    Return the names whose performing credits were replaced.

    :param mass: The MusicAssistant instance.
    :param track: The track to add or update with.
    :param cur_item: The library track as stored, on an update.
    """
    work = track.work if isinstance(track.work, Work) else None
    if (
        any(x.role == ArtistRole.COMPOSER for x in track.credits)
        or (work and work.composers)
        or not _names_placeholder_composer(track)
    ):
        return set()
    credited = {x.name: x for x in track.artists} | {
        x.artist.name: x.artist for x in track.credits if x.role in PERFORMING_ROLES
    }
    if not credited:
        return set()
    artists = mass.music.artists
    library_composers = await artists.get_library_composers(with_mbid=True)
    composer_ids = {x.item_id: x for x in library_composers}
    matches: dict[str, Artist] = {}
    for name, artist in credited.items():
        # the library artist of the provider artist itself wins over a name match
        known = await artists.get_library_item_by_prov_id(artist.item_id, artist.provider)
        if known and known.item_id in composer_ids:
            matches[name] = composer_ids[known.item_id]
            continue
        found = [x for x in library_composers if is_named_by(name, x)]
        if len(found) == 1:
            matches[name] = found[0]
    # artists naming different composers leave the composer unknown
    if len({x.item_id for x in matches.values()}) != 1:
        return set()
    composer = next(iter(matches.values()))
    # a track the library credits another composer keeps it
    if cur_item and any(
        x.role == ArtistRole.COMPOSER and x.artist.item_id != composer.item_id
        for x in cur_item.credits
    ):
        return set()
    track.credits = [
        x for x in track.credits if x.role not in PERFORMING_ROLES or x.artist.name not in matches
    ]
    track.credits.append(Credit(artist=composer, role=ArtistRole.COMPOSER))
    if work:
        work.composers = UniqueList([composer])
    return set(matches)


async def remove_joined_artists(
    mass: MusicAssistant,
    track: Track,
    work_id: int | None,
    joined_names: set[str],
    main_artists: bool,
) -> None:
    """
    Remove the stored credits of a library track and its work for joined names.

    :param mass: The MusicAssistant instance.
    :param track: The library track as stored.
    :param work_id: The library id of the track's work, if any.
    :param joined_names: The joined names to remove.
    :param main_artists: Whether to remove the main artists of the joined names too.
    """
    if main_artists:
        for artist in track.artists:
            if artist.name in joined_names:
                await mass.music.database.delete(
                    DB_TABLE_TRACK_ARTISTS,
                    {"track_id": int(track.item_id), "artist_id": int(artist.item_id)},
                )
    for credit in track.credits:
        if credit.role != ArtistRole.COMPOSER or credit.artist.name not in joined_names:
            continue
        match = {"artist_id": int(credit.artist.item_id), "role": ArtistRole.COMPOSER.value}
        await mass.music.database.delete(
            DB_TABLE_TRACK_CREDITS, {"track_id": int(track.item_id), **match}
        )
        if work_id:
            await mass.music.database.delete(DB_TABLE_WORK_ARTISTS, {"work_id": work_id, **match})


def _names_placeholder_composer(track: Track) -> bool:
    """Return whether the source of a track names a placeholder composer, as "Dr, Composer"."""
    for performer in track.metadata.performers or ():
        name, *roles = performer.split(", ")
        if is_placeholder_artist(name) and any(role_word(x) == "composer" for x in roles):
            return True
    return False


def _joined_composer_candidates(track: Track) -> dict[str, str]:
    """Return the joined composer names of a track with the composer each starts with."""
    work = track.work if isinstance(track.work, Work) else None
    composers = [x.artist for x in track.credits if x.role == ArtistRole.COMPOSER]
    return {
        artist.name: candidate
        for artist in [*composers, *(work.composers if work else [])]
        if (candidate := joined_composer_name(artist.name, track.name))
    }


def _replace_joined_composers(track: Track, replacements: dict[str, Artist]) -> set[str]:
    """Credit each composer in place of its joined name on a track, return the names."""
    if not replacements:
        return set()
    work = track.work if isinstance(track.work, Work) else None
    track.credits = [
        replace(x, artist=replacements[x.artist.name])
        if x.role == ArtistRole.COMPOSER and x.artist.name in replacements
        else x
        for x in track.credits
    ]
    if work:
        work.composers = UniqueList(replacements.get(x.name, x) for x in work.composers)
    track.artists = UniqueList(replacements.get(x.name, x) for x in track.artists)
    return set(replacements)
