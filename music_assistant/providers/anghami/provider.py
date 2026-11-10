"""Anghami music provider implementation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import (
    MediaNotFoundError,
    ProviderPermissionDenied,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemType,
    Playlist,
    RecommendationFolder,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.cache import use_cache
from music_assistant.models.music_provider import MusicProvider

from .api import AnghamiClient, client_credentials
from .constants import (
    CONTENT_TYPES,
    DRM_SCHEME_NONE,
    LIKED_ALBUMS,
    LIKED_SONGS,
    MAX_PAGE_SIZE,
    RECOMMENDATION_ROWS,
    ROW_CHARTS,
    ROW_FEATURED,
    ROW_NEW_RELEASES,
    STREAM_EXPIRY_MARGIN,
)
from .parsers import (
    content_ref,
    image_from_obj,
    parse_album,
    parse_artist,
    parse_lyrics,
    parse_playlist,
    parse_playlist_content,
    parse_track,
    song_album_id,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence


class AnghamiProvider(MusicProvider):
    """Provider for the Anghami streaming service."""

    client: AnghamiClient

    @property
    def max_concurrent_streams(self) -> int:
        """Anghami caps the streams one account can play at once."""
        return 1

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        # fails early with a clear error on a build without the Anghami credentials
        client_credentials()
        self.client = AnghamiClient(self)
        user = (await self.client.get("/v1/user/me"))["user"]
        self.logger.info("Signed in to Anghami as %s", user.get("displayName"))

    @use_cache(3600 * 3)
    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """
        Perform a search on Anghami.

        :param search_query: Search query.
        :param media_types: A list of media_types to include.
        :param limit: Number of items to return in the search (per type).
        """
        result = SearchResults()
        content_types = [
            CONTENT_TYPES[media_type] for media_type in media_types if media_type in CONTENT_TYPES
        ]
        if not content_types:
            return result
        data = await self.client.get(
            "/v1/discovery/search",
            {
                "q": search_query,
                "types": content_types,
                "page_size": min(limit * len(content_types), MAX_PAGE_SIZE),
            },
        )
        items = await self._resolve_contents(data.get("results", []))
        result.tracks = [item for item in items if isinstance(item, Track)][:limit]
        result.albums = [item for item in items if isinstance(item, Album)][:limit]
        result.artists = [item for item in items if isinstance(item, Artist)][:limit]
        result.playlists = [item for item in items if isinstance(item, Playlist)][:limit]
        return result

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Browse Anghami, including the featured sections and charts opened from the recommendations.

        :param path: The path to browse, in the form ``<instance_id>://<sub_path>``.
        """
        section, _, sub_id = path.split("://", 1)[1].partition("/")
        if section == ROW_FEATURED and sub_id:
            return await self._get_featured_section_items(sub_id)
        if section == ROW_CHARTS and sub_id:
            return await self._get_chart_items(sub_id)
        return await super().browse(path)

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's available recommendation rows, without items."""
        return [
            RecommendationFolder(
                item_id=row,
                provider=self.instance_id,
                name=name,
                translation_key=row,
                icon=icon,
            )
            for row, (name, icon) in RECOMMENDATION_ROWS.items()
        ]

    async def get_recommendation_items(
        self, item_id: str
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Get the items for a single recommendation row.

        :param item_id: The item_id of the row, as returned by get_recommendations.
        """
        if item_id == ROW_FEATURED:
            return UniqueList(await self._get_featured_folders())
        if item_id == ROW_NEW_RELEASES:
            return UniqueList(await self._get_new_releases())
        if item_id == ROW_CHARTS:
            return UniqueList(await self._get_chart_folders())
        return UniqueList()

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        """Retrieve the artists the user follows on Anghami."""
        async for item in self._iter_contents("/v1/library/following/artists"):
            if isinstance(item, Artist):
                yield item

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Retrieve the albums the user liked on Anghami."""
        async for item in self._iter_contents("/v1/library/likes", {"content_type": LIKED_ALBUMS}):
            if isinstance(item, Album):
                yield item

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Retrieve the songs the user liked on Anghami."""
        async for item in self._iter_contents("/v1/library/likes", {"content_type": LIKED_SONGS}):
            if isinstance(item, Track):
                yield item

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Retrieve the playlists the user created or saved on Anghami."""
        async for page in self.client.paginate("/v1/playlists", "playlists"):
            for playlist_obj in page:
                yield parse_playlist(self, playlist_obj)

    @use_cache(86400 * 30)
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """
        Get full artist details by id.

        :param prov_artist_id: The Anghami artist id.
        """
        data = await self.client.get(f"/v1/music/artists/{prov_artist_id}")
        return parse_artist(self, data["artist"])

    @use_cache(86400 * 7)
    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """
        Get the discography of an artist.

        :param prov_artist_id: The Anghami artist id.
        """
        return [
            parse_album(self, album_obj)
            async for page in self.client.paginate(
                f"/v1/music/artists/{prov_artist_id}/discography", "albums"
            )
            for album_obj in page
        ]

    @use_cache(86400 * 7)
    async def get_artist_toptracks(self, prov_artist_id: str) -> list[Track]:
        """
        Get the most popular songs of an artist.

        :param prov_artist_id: The Anghami artist id.
        """
        data = await self.client.get(
            f"/v1/music/artists/{prov_artist_id}/top-tracks", {"page_size": 25}
        )
        return await self._parse_songs(data.get("tracks", []))

    @use_cache(86400 * 7)
    async def get_similar_artists(self, prov_artist_id: str, limit: int = 25) -> list[Artist]:
        """
        Get the artists Anghami lists as related to an artist.

        :param prov_artist_id: The Anghami artist id.
        :param limit: Maximum number of artists to return.
        """
        data = await self.client.get(
            f"/v1/music/artists/{prov_artist_id}/related",
            {"page_size": min(limit, MAX_PAGE_SIZE)},
        )
        return [parse_artist(self, artist_obj) for artist_obj in data.get("artists", [])]

    @use_cache(86400 * 30)
    async def get_album(self, prov_album_id: str) -> Album:
        """
        Get full album details by id.

        :param prov_album_id: The Anghami album id.
        """
        data = await self.client.get(f"/v1/music/albums/{prov_album_id}")
        return parse_album(self, data["album"])

    @use_cache(86400 * 7)
    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """
        Get the tracks of an album.

        :param prov_album_id: The Anghami album id.
        """
        album = await self.get_album(prov_album_id)
        return [
            parse_track(self, song_obj, album)
            async for page in self.client.paginate(
                f"/v1/music/albums/{prov_album_id}/tracks", "tracks"
            )
            for song_obj in page
        ]

    @use_cache(86400 * 7)
    async def get_track(self, prov_track_id: str) -> Track:
        """
        Get full track details by id, including its lyrics when Anghami has them.

        :param prov_track_id: The Anghami song id.
        """
        song_obj = (await self.client.get(f"/v1/music/songs/{prov_track_id}"))["song"]
        track = (await self._parse_songs([song_obj]))[0]
        if song_obj.get("hasLyrics"):
            try:
                lyrics_obj = await self.client.get(f"/v1/music/songs/{prov_track_id}/lyrics")
            except (MediaNotFoundError, ProviderPermissionDenied) as err:
                self.logger.debug("No lyrics for Anghami song %s: %s", prov_track_id, err)
            else:
                track.metadata.lyrics, track.metadata.lrc_lyrics = parse_lyrics(lyrics_obj)
        return track

    @use_cache(3600 * 3)
    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """
        Get full playlist details by id.

        :param prov_playlist_id: The Anghami playlist id.
        """
        data = await self.client.get(f"/v1/playlists/{prov_playlist_id}", {"page_size": 1})
        return parse_playlist(self, data["playlist"])

    @use_cache(3600 * 3)
    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """
        Get the tracks of a playlist, all of them on the first page.

        :param prov_playlist_id: The Anghami playlist id.
        :param page: Zero-based page index, later pages are empty.
        """
        if page > 0:
            return []
        tracks: list[Track] = []
        async for items in self.client.paginate(f"/v1/playlists/{prov_playlist_id}", "items"):
            for item in await self._resolve_contents([entry["content"] for entry in items]):
                if isinstance(item, Track):
                    # a copy, as a song listed twice resolves to the same track object
                    tracks.append(replace(item, position=len(tracks) + 1))
        return tracks

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """
        Acquire a stream for a song, which Anghami counts as a billable stream.

        :param item_id: The Anghami song id.
        :param media_type: The media type of the item, always a track.
        """
        stream = await self.client.acquire_stream(item_id)
        if (drm_scheme := stream.get("drmScheme", DRM_SCHEME_NONE)) != DRM_SCHEME_NONE:
            raise UnplayableMediaError(
                f"Anghami song {item_id} is protected with {drm_scheme}, "
                "protected streams are not supported"
            )
        manifest_url = stream["manifestUrl"]
        details = StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=_stream_audio_format(stream.get("availableQualities", [])),
            media_type=media_type,
            # a DASH manifest is read by ffmpeg directly, like any other HTTP source
            stream_type=StreamType.HLS
            if urlparse(manifest_url).path.endswith(".m3u8")
            else StreamType.HTTP,
            path=manifest_url,
            can_seek=True,
            allow_seek=True,
        )
        if expires_in := stream.get("expiresInSeconds"):
            details.expiration = max(expires_in - STREAM_EXPIRY_MARGIN, 0)
        return details

    async def _iter_contents(
        self, path: str, params: dict[str, Any] | None = None
    ) -> AsyncGenerator[Track | Album | Artist | Playlist]:
        """Yield the full media items of a paginated content list, resolved page by page."""
        async for page in self.client.paginate(path, "items", params):
            for item in await self._resolve_contents(page):
                yield item

    async def _resolve_contents(
        self, contents: list[dict[str, Any]]
    ) -> list[Track | Album | Artist | Playlist]:
        """Return the full media items of mixed content entries, in their original order."""
        refs = [ref for content in contents if (ref := content_ref(content))]
        ids = {
            media_type: [item_id for ref_type, item_id in refs if ref_type == media_type]
            for media_type in CONTENT_TYPES
        }
        songs = await self.client.batch_get("song", ids[MediaType.TRACK])
        # the albums of the songs come along in the batch of the listed albums
        albums = await self._get_albums(
            ids[MediaType.ALBUM] + [song_album_id(song_obj) for song_obj in songs.values()]
        )
        artists = await self.client.batch_get("artist", ids[MediaType.ARTIST])
        items: dict[tuple[MediaType, str], Track | Album | Artist | Playlist] = {}
        for song_id, song_obj in songs.items():
            items[MediaType.TRACK, song_id] = parse_track(
                self, song_obj, albums.get(song_album_id(song_obj))
            )
        for album_id in ids[MediaType.ALBUM]:
            if album_id in albums:
                items[MediaType.ALBUM, album_id] = albums[album_id]
        for artist_id, artist_obj in artists.items():
            items[MediaType.ARTIST, artist_id] = parse_artist(self, artist_obj)
        # a playlist entry is listed with its display fields, its tracks load when it is opened
        for content in contents:
            if (ref := content_ref(content)) and ref[0] == MediaType.PLAYLIST:
                items[ref] = parse_playlist_content(self, content)
        return [items[ref] for ref in refs if ref in items]

    async def _parse_songs(self, songs: list[dict[str, Any]]) -> list[Track]:
        """Parse songs into tracks, with their albums fetched in one batch."""
        albums = await self._get_albums([song_album_id(song_obj) for song_obj in songs])
        return [
            parse_track(self, song_obj, albums.get(song_album_id(song_obj))) for song_obj in songs
        ]

    async def _get_albums(self, album_ids: list[str]) -> dict[str, Album]:
        """Return the albums for the given ids in one batch, keyed by id."""
        album_objs = await self.client.batch_get("album", [aid for aid in album_ids if aid])
        return {
            album_id: parse_album(self, album_obj) for album_id, album_obj in album_objs.items()
        }

    @use_cache(3600 * 3)
    async def _get_featured_sections(self) -> list[dict[str, Any]]:
        """Return the featured sections that hold music."""
        data = await self.client.get("/v1/discovery/browse/featured", {"page_size": MAX_PAGE_SIZE})
        return [
            section
            for section in data.get("sections", [])
            if any(content_ref(content) for content in section.get("items", []))
        ]

    @use_cache(3600 * 3, base_class=BrowseFolder)
    async def _get_featured_folders(self) -> list[BrowseFolder]:
        """Return a browse folder for each featured section."""
        return [
            BrowseFolder(
                item_id=section["id"]["value"],
                provider=self.instance_id,
                path=f"{self.instance_id}://{ROW_FEATURED}/{section['id']['value']}",
                name=section.get("title", ""),
                image=image_from_obj(self, section.get("artwork")),
            )
            for section in await self._get_featured_sections()
        ]

    @use_cache(3600 * 3)
    async def _get_featured_section_items(
        self, section_id: str
    ) -> list[Track | Album | Artist | Playlist]:
        """Return the media items of a featured section."""
        for section in await self._get_featured_sections():
            if section["id"]["value"] == section_id:
                return await self._resolve_contents(section.get("items", []))
        return []

    @use_cache(3600 * 3)
    async def _get_new_releases(self) -> list[Track | Album | Artist | Playlist]:
        """Return the recent releases on Anghami."""
        data = await self.client.get(
            "/v1/discovery/browse/new-releases",
            {
                "content_types": [CONTENT_TYPES[MediaType.ALBUM], CONTENT_TYPES[MediaType.TRACK]],
                "page_size": 25,
            },
        )
        return await self._resolve_contents(data.get("items", []))

    @use_cache(3600 * 3, base_class=BrowseFolder)
    async def _get_chart_folders(self) -> list[BrowseFolder]:
        """Return a browse folder for each music chart."""
        data = await self.client.get("/v1/discovery/browse/charts", {"page_size": MAX_PAGE_SIZE})
        return [
            BrowseFolder(
                item_id=chart["id"]["value"],
                provider=self.instance_id,
                path=f"{self.instance_id}://{ROW_CHARTS}/{chart['id']['value']}",
                name=chart.get("name", ""),
                image=image_from_obj(self, chart.get("artwork")),
            )
            for entry in data.get("charts", [])
            if (chart := entry["chart"]).get("contentType") in CONTENT_TYPES.values()
        ]

    @use_cache(3600 * 3)
    async def _get_chart_items(self, chart_id: str) -> list[Track | Album | Artist | Playlist]:
        """Return the media items of a chart, in chart order."""
        data = await self.client.get(
            f"/v1/discovery/browse/charts/{chart_id}/entries", {"page_size": MAX_PAGE_SIZE}
        )
        return await self._resolve_contents([entry["content"] for entry in data.get("entries", [])])


def _stream_audio_format(qualities: list[dict[str, Any]]) -> AudioFormat:
    """Return the audio format of the best quality a stream offers."""
    if not qualities:
        return AudioFormat(content_type=ContentType.UNKNOWN)
    best = max(qualities, key=lambda quality: quality.get("bitrateKbps", 0))
    return AudioFormat(
        content_type=ContentType.try_parse(best.get("codec", "")),
        bit_rate=best.get("bitrateKbps"),
    )
