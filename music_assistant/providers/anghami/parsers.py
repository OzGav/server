"""Parsers that turn Anghami SDK API objects into Music Assistant media items."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID, ImageType, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    MediaItemImage,
    Playlist,
    ProviderMapping,
    Track,
    UniqueList,
)

from music_assistant.helpers.lyrics import convert_to_lrc_lyrics

from .constants import ALBUM_TYPES

if TYPE_CHECKING:
    from music_assistant_models.media_items import MediaItem

    from .provider import AnghamiProvider

# the id fields of a mixed content entry, for the media types this provider serves
CONTENT_ID_FIELDS = {
    "songId": MediaType.TRACK,
    "albumId": MediaType.ALBUM,
    "artistId": MediaType.ARTIST,
    "playlistId": MediaType.PLAYLIST,
}


def content_ref(content: dict[str, Any]) -> tuple[MediaType, str] | None:
    """
    Return the media type and id of a mixed content entry, or None for video content.

    :param content: A content entry from a search, library, playlist or browse list.
    """
    for field, media_type in CONTENT_ID_FIELDS.items():
        if ref := content.get(field):
            return media_type, ref["value"]
    return None


def song_album_id(song_obj: dict[str, Any]) -> str:
    """
    Return the album id of a song, or an empty string when it has none.

    :param song_obj: The song object from the API.
    """
    album_id: str = song_obj.get("albumId", {}).get("value", "")
    return album_id


def parse_artist(provider: AnghamiProvider, artist_obj: dict[str, Any]) -> Artist:
    """
    Parse an Anghami artist into a Music Assistant Artist.

    :param provider: The provider instance the artist belongs to.
    :param artist_obj: The artist object from the API.
    """
    artist_id = artist_obj["id"]["value"]
    artist = Artist(
        item_id=artist_id,
        provider=provider.instance_id,
        name=artist_obj["name"]["value"],
        provider_mappings={_provider_mapping(provider, artist_id)},
    )
    if bio := artist_obj.get("bio", {}).get("value"):
        artist.metadata.description = bio
    if genres := artist_obj.get("genres"):
        artist.metadata.genres = set(genres)
    _add_image(provider, artist, _largest_image(artist_obj.get("artwork")))
    return artist


def parse_album(provider: AnghamiProvider, album_obj: dict[str, Any]) -> Album:
    """
    Parse an Anghami album into a Music Assistant Album.

    :param provider: The provider instance the album belongs to.
    :param album_obj: The album object from the API.
    """
    album_id = album_obj["id"]["value"]
    album = Album(
        item_id=album_id,
        provider=provider.instance_id,
        name=album_obj["title"]["value"],
        artists=UniqueList(_artist_mapping(provider, ref) for ref in album_obj.get("artists", [])),
        provider_mappings={_provider_mapping(provider, album_id)},
    )
    if album_type := ALBUM_TYPES.get(album_obj.get("albumType", "")):
        album.album_type = album_type
    if release_date := album_obj.get("releaseDate"):
        with suppress(ValueError):
            album.year = int(release_date[:4])
            album.metadata.release_date = datetime.fromisoformat(release_date)
    if upc := album_obj.get("upc"):
        album.external_ids.add((ExternalID.BARCODE, upc))
    if genres := album_obj.get("genres"):
        album.metadata.genres = set(genres)
    album.metadata.label = album_obj.get("label") or None
    album.metadata.popularity = album_obj.get("popularity")
    _add_image(provider, album, _largest_image(album_obj.get("artwork")))
    return album


def parse_track(
    provider: AnghamiProvider, song_obj: dict[str, Any], album: Album | None = None
) -> Track:
    """
    Parse an Anghami song into a Music Assistant Track.

    :param provider: The provider instance the track belongs to.
    :param song_obj: The song object from the API.
    :param album: The album of the song, when known.
    """
    song_id = song_obj["id"]["value"]
    track = Track(
        item_id=song_id,
        provider=provider.instance_id,
        name=song_obj["title"]["value"],
        duration=song_obj.get("durationMs", 0) // 1000,
        disc_number=song_obj.get("discNumber", 0),
        track_number=song_obj.get("trackNumber", 0),
        artists=UniqueList(_artist_mapping(provider, ref) for ref in song_obj.get("artists", [])),
        album=album,
        provider_mappings={_provider_mapping(provider, song_id)},
    )
    if isrc := song_obj.get("isrc"):
        track.external_ids.add((ExternalID.ISRC, isrc))
    if genres := song_obj.get("genres"):
        track.metadata.genres = set(genres)
    track.metadata.explicit = song_obj.get("isExplicit", False)
    track.metadata.popularity = song_obj.get("popularity")
    _add_image(provider, track, _largest_image(song_obj.get("artwork")))
    return track


def parse_playlist(provider: AnghamiProvider, playlist_obj: dict[str, Any]) -> Playlist:
    """
    Parse an Anghami playlist into a Music Assistant Playlist.

    :param provider: The provider instance the playlist belongs to.
    :param playlist_obj: The playlist object from the API.
    """
    playlist_id = playlist_obj["id"]["value"]
    playlist = Playlist(
        item_id=playlist_id,
        provider=provider.instance_id,
        name=playlist_obj["name"]["value"],
        owner=playlist_obj.get("ownerName", ""),
        is_editable=False,
        provider_mappings={_provider_mapping(provider, playlist_id)},
    )
    if description := playlist_obj.get("description", {}).get("value"):
        playlist.metadata.description = description
    _add_image(provider, playlist, playlist_obj.get("coverArtwork"))
    return playlist


def parse_playlist_content(provider: AnghamiProvider, content: dict[str, Any]) -> Playlist:
    """
    Parse the playlist entry of a mixed content list into a Music Assistant Playlist.

    :param provider: The provider instance the playlist belongs to.
    :param content: The content entry, which carries only display fields.
    """
    playlist_id = content["playlistId"]["value"]
    playlist = Playlist(
        item_id=playlist_id,
        provider=provider.instance_id,
        name=content.get("title", ""),
        owner=content.get("subtitle", ""),
        is_editable=False,
        provider_mappings={_provider_mapping(provider, playlist_id)},
    )
    _add_image(provider, playlist, content.get("artwork"))
    return playlist


def parse_lyrics(lyrics_obj: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Return the plain and the time synced (LRC) lyrics of a lyrics response.

    :param lyrics_obj: The lyrics response from the API.
    """
    if synced := lyrics_obj.get("syncedLyrics", {}).get("lines"):
        lines = [(line.get("text", ""), int(line.get("startTimeMs", 0))) for line in synced]
        return "\n".join(text for text, _ in lines), convert_to_lrc_lyrics(lines)
    if plain := lyrics_obj.get("lyrics", {}).get("lines"):
        return "\n".join(plain), None
    return None, None


def image_from_obj(
    provider: AnghamiProvider, image_obj: dict[str, Any] | None
) -> MediaItemImage | None:
    """
    Return the Music Assistant image of an Anghami image, or None when there is none.

    :param provider: The provider instance the image belongs to.
    :param image_obj: The image object from the API.
    """
    if not image_obj or not image_obj.get("url"):
        return None
    return MediaItemImage(
        type=ImageType.THUMB,
        path=image_obj["url"],
        provider=provider.instance_id,
        remotely_accessible=True,
    )


def _provider_mapping(provider: AnghamiProvider, item_id: str) -> ProviderMapping:
    """Return the provider mapping of an item of this provider."""
    return ProviderMapping(
        item_id=item_id,
        provider_domain=provider.domain,
        provider_instance=provider.instance_id,
    )


def _artist_mapping(provider: AnghamiProvider, artist_ref: dict[str, Any]) -> ItemMapping:
    """Return the item mapping of an artist reference on a song or album."""
    return ItemMapping(
        media_type=MediaType.ARTIST,
        item_id=artist_ref["id"]["value"],
        provider=provider.instance_id,
        name=artist_ref["name"]["value"],
    )


def _largest_image(images: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Return the widest image of an artwork list."""
    if not images:
        return None
    return max(images, key=lambda image: image.get("width", 0))


def _add_image(
    provider: AnghamiProvider, item: MediaItem, image_obj: dict[str, Any] | None
) -> None:
    """Add an Anghami image to a media item as its thumb."""
    if image := image_from_obj(provider, image_obj):
        item.metadata.add_image(image)
