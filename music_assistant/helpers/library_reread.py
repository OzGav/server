"""Helper to read a music provider's library albums again, for details it did not read before."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from music_assistant_models.enums import MediaType
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    ProviderUnavailableError,
    ResourceTemporarilyUnavailable,
    RetriesExhausted,
)

from music_assistant.constants import DB_TABLE_PROVIDER_MAPPINGS
from music_assistant.helpers.throttle_retry import RequestPriority, set_request_priority

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from music_assistant_models.media_items import Album

    from music_assistant.models.music_provider import MusicProvider


async def reread_library_albums(
    provider: MusicProvider, reread_album: Callable[[int, str], Awaitable[None]]
) -> bool:
    """
    Read every library album of a provider instance again, behind the requests a user waits for.

    An album the provider no longer has is skipped. Never raises.

    :param provider: The provider instance whose library albums are read.
    :param reread_album: Reads one album, given its library id and its id on the provider.
    :return: Whether every album was read, False when the run stopped on an error.
    """
    set_request_priority(RequestPriority.LOW)
    try:
        rows = await provider.mass.music.database.get_rows_from_query(
            f"SELECT item_id, provider_item_id FROM {DB_TABLE_PROVIDER_MAPPINGS} "
            "WHERE media_type = :media_type AND provider_instance = :instance",
            {"media_type": MediaType.ALBUM.value, "instance": provider.instance_id},
            limit=0,
        )
        for row in rows:
            try:
                await reread_album(int(row["item_id"]), row["provider_item_id"])
            except MediaNotFoundError:
                continue
    except (
        LoginFailed,
        ProviderUnavailableError,
        ResourceTemporarilyUnavailable,
        RetriesExhausted,
    ) as err:
        provider.logger.info("Reading the library albums again stops until the next start: %s", err)
        return False
    except Exception as err:
        provider.logger.warning(
            "Error reading the library albums again: %s",
            err,
            exc_info=err if provider.logger.isEnabledFor(logging.DEBUG) else None,
        )
        return False
    return True


async def update_library_album(provider: MusicProvider, library_id: int, album: Album) -> None:
    """
    Update a library album with a provider album read again, its genres mapped right away.

    :param provider: The provider instance the album comes from.
    :param library_id: The library id of the album.
    :param album: The album as the provider gives it now.
    """
    library_album = await provider.mass.music.albums.update_item_in_library(library_id, album)
    # as a library sync does, so the album is in its genres before the next genre scan
    await provider.mass.music.genres.sync_media_item_genres(
        MediaType.ALBUM, library_id, set(library_album.metadata.genres or ())
    )
