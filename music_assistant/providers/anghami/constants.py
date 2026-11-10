"""Constants for the Anghami provider."""

from music_assistant_models.enums import AlbumType, MediaType, ProviderFeature

SUPPORTED_FEATURES = {
    ProviderFeature.LIBRARY_ARTISTS,
    ProviderFeature.LIBRARY_ALBUMS,
    ProviderFeature.LIBRARY_TRACKS,
    ProviderFeature.LIBRARY_PLAYLISTS,
    ProviderFeature.SEARCH,
    ProviderFeature.ARTIST_ALBUMS,
    ProviderFeature.ARTIST_TOPTRACKS,
    ProviderFeature.SIMILAR_ARTISTS,
    ProviderFeature.LYRICS,
    ProviderFeature.BROWSE,
    ProviderFeature.RECOMMENDATIONS,
}

API_BASE_URL = "https://sdk.anghami.com"
OAUTH_AUTHORIZE_URL = "https://landing.anghami.com/authorize"
# the SDK docs show /v1/auth/token with a JSON body, the live OAuth metadata publishes this one
OAUTH_TOKEN_URL = "https://sdk.anghami.com/token"  # noqa: S105 - URL
OAUTH_SCOPE = "read stream"
OAUTH_SCOPE_STREAM = "stream"

CONF_REFRESH_TOKEN = "refresh_token"  # noqa: S105 - config key
CONF_USER_ID = "user_id"

# largest page and batch size the API accepts
MAX_PAGE_SIZE = 100
RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"
# seconds before the access token expires at which it is refreshed
ACCESS_EXPIRY_MARGIN = 300
# seconds before a stream URL expires at which a new one is acquired
STREAM_EXPIRY_MARGIN = 60

# the server lowers this to the best quality the user's subscription includes
MAX_AUDIO_QUALITY = "AUDIO_QUALITY_LOSSLESS"
DRM_SCHEME_NONE = "DRM_SCHEME_UNSPECIFIED"

LIKED_SONGS = "LIKEABLE_CONTENT_TYPE_SONG"
LIKED_ALBUMS = "LIKEABLE_CONTENT_TYPE_ALBUM"

CONTENT_TYPES = {
    MediaType.TRACK: "CONTENT_TYPE_SONG",
    MediaType.ALBUM: "CONTENT_TYPE_ALBUM",
    MediaType.ARTIST: "CONTENT_TYPE_ARTIST",
    MediaType.PLAYLIST: "CONTENT_TYPE_PLAYLIST",
}

ALBUM_TYPES = {
    "ALBUM_TYPE_ALBUM": AlbumType.ALBUM,
    "ALBUM_TYPE_SINGLE": AlbumType.SINGLE,
    "ALBUM_TYPE_COMPILATION": AlbumType.COMPILATION,
    "ALBUM_TYPE_EP": AlbumType.EP,
}

ROW_FEATURED = "featured"
ROW_NEW_RELEASES = "new_releases"
ROW_CHARTS = "charts"

# the recommendation rows: name, material design icon
RECOMMENDATION_ROWS = {
    ROW_FEATURED: ("Featured", "mdi-star"),
    ROW_NEW_RELEASES: ("New releases", "mdi-new-box"),
    ROW_CHARTS: ("Charts", "mdi-chart-line"),
}
