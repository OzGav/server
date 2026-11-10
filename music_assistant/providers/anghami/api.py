"""Client for the Anghami SDK API."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError
from music_assistant_models.errors import (
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    ProviderPermissionDenied,
    RateLimited,
    ResourceTemporarilyUnavailable,
    SetupFailedError,
)

from music_assistant.helpers.app_vars import app_var
from music_assistant.helpers.throttle_retry import ThrottlerManager, throttle_with_retries

from .constants import (
    ACCESS_EXPIRY_MARGIN,
    API_BASE_URL,
    CONF_REFRESH_TOKEN,
    MAX_AUDIO_QUALITY,
    MAX_PAGE_SIZE,
    OAUTH_TOKEN_URL,
    RATE_LIMIT_RESET_HEADER,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from aiohttp import ClientResponse, ClientSession

    from .provider import AnghamiProvider


def client_credentials() -> tuple[str, str]:
    """
    Return the Anghami OAuth client id and client secret bundled with Music Assistant.

    :raises SetupFailedError: When this build carries no Anghami client credentials.
    """
    client_id = app_var("anghami_client_id")
    client_secret = app_var("anghami_client_secret")
    if not client_id or not client_secret:
        raise SetupFailedError(
            "This Music Assistant build has no Anghami client credentials, "
            "so Anghami can not be set up"
        )
    return client_id, client_secret


async def get_current_user(http_session: ClientSession, access_token: str) -> dict[str, Any]:
    """
    Return the profile of the Anghami user an access token belongs to.

    :param http_session: The HTTP session to send the request with.
    :param access_token: The access token of the user.
    :raises LoginFailed: When Anghami rejects the access token.
    :raises ResourceTemporarilyUnavailable: When Anghami can not be reached or fails.
    """
    try:
        async with http_session.get(
            f"{API_BASE_URL}/v1/user/me", headers={"Authorization": f"Bearer {access_token}"}
        ) as response:
            user: dict[str, Any] = (await _parse_response(response))["user"]
            return user
    except ClientError as err:
        raise ResourceTemporarilyUnavailable("Could not reach Anghami to sign in") from err


async def request_tokens(http_session: ClientSession, grant: dict[str, str]) -> dict[str, Any]:
    """
    Request OAuth tokens from Anghami and return the token response.

    :param http_session: The HTTP session to send the request with.
    :param grant: The grant fields of the request, such as the grant type and the code.
    :raises LoginFailed: When Anghami rejects the grant.
    :raises ResourceTemporarilyUnavailable: When Anghami can not be reached or fails.
    """
    client_id, client_secret = client_credentials()
    form = {**grant, "client_id": client_id, "client_secret": client_secret}
    try:
        async with http_session.post(OAUTH_TOKEN_URL, data=form) as response:
            return await _parse_token_response(response)
    except ClientError as err:
        raise ResourceTemporarilyUnavailable("Could not reach Anghami to sign in") from err


class AnghamiClient:
    """Client for the Anghami SDK API, signed in as the user of a provider instance."""

    def __init__(self, provider: AnghamiProvider) -> None:
        """
        Initialize the client.

        :param provider: The provider instance whose stored sign-in the client uses.
        """
        self.provider = provider
        self.logger = provider.logger
        # the API documents no fixed request rate, its limits apply per access token
        self.throttler = ThrottlerManager(rate_limit=4, period=1)
        self._access_token: str | None = None
        self._access_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Send a GET request and return the decoded response.

        :param path: The API path, e.g. /v1/music/songs/123.
        :param params: Optional query parameters.
        """
        return await self._request("GET", path, params=params)

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Send a POST request and return the decoded response.

        :param path: The API path, e.g. /v1/music/songs:batchGet.
        :param body: The JSON request body.
        """
        return await self._request("POST", path, json=body)

    async def paginate(
        self, path: str, key: str, params: dict[str, Any] | None = None
    ) -> AsyncGenerator[list[dict[str, Any]]]:
        """
        Yield every page of a paginated list.

        :param path: The API path of the list.
        :param key: The response field that holds the items of a page.
        :param params: Optional query parameters besides the paging ones.
        """
        query = {**(params or {}), "page_size": MAX_PAGE_SIZE}
        while True:
            data = await self.get(path, query)
            yield data.get(key, [])
            if not (next_page := data.get("nextPageToken")):
                return
            query = {**query, "page_token": next_page}

    async def batch_get(self, kind: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """
        Return the catalog entities for the given ids, keyed by id.

        Ids Anghami can not return, such as removed or region locked items, are left out.

        :param kind: The entity kind: song, album or artist.
        :param ids: The ids of the entities.
        """
        unique_ids = list(dict.fromkeys(ids))
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(unique_ids), MAX_PAGE_SIZE):
            chunk = unique_ids[start : start + MAX_PAGE_SIZE]
            data = await self.post(
                f"/v1/music/{kind}s:batchGet",
                {f"{kind}IdList": [{"value": item_id} for item_id in chunk]},
            )
            for item_id, entry in data.get("results", {}).items():
                if kind in entry:
                    result[item_id] = entry[kind]
                else:
                    self.logger.debug("Anghami returned no %s %s: %s", kind, item_id, entry)
        return result

    async def acquire_stream(self, song_id: str) -> dict[str, Any]:
        """
        Acquire a stream for a song and return its stream info.

        Every acquire is a billable stream for Anghami, so only call this to play the song.

        :param song_id: The id of the song.
        """
        data = await self.post(
            "/v1/streaming/music",
            {"songId": {"value": song_id}, "maxQuality": MAX_AUDIO_QUALITY},
        )
        stream: dict[str, Any] = data["stream"]
        return stream

    @throttle_with_retries
    async def _request(self, /, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send an authenticated request, refreshing the access token once when it is refused."""
        url = f"{API_BASE_URL}{path}"
        http_session = self.provider.mass.http_session
        access_token = await self._get_access_token()
        try:
            async with http_session.request(
                method, url, headers=self._headers(access_token), **kwargs
            ) as response:
                if response.status != 401:
                    return await _parse_response(response)
            # the access token was revoked or expired early
            access_token = await self._get_access_token(rejected=access_token)
            async with http_session.request(
                method, url, headers=self._headers(access_token), **kwargs
            ) as response:
                return await _parse_response(response)
        except ClientError as err:
            raise ResourceTemporarilyUnavailable(f"Could not reach Anghami for {path}") from err

    async def _get_access_token(self, rejected: str | None = None) -> str:
        """Return a valid access token, refreshing it when it expired or was rejected."""
        async with self._token_lock:
            if (
                self._access_token
                and self._access_token != rejected
                and time.time() < self._access_expires_at
            ):
                return self._access_token
            refresh_token = str(self.provider.get_setup_value(CONF_REFRESH_TOKEN) or "")
            if not refresh_token:
                raise LoginFailed("Sign in to Anghami again")
            try:
                tokens = await request_tokens(
                    self.provider.mass.http_session,
                    {"grant_type": "refresh_token", "refresh_token": refresh_token},
                )
            except LoginFailed as err:
                if self.provider.available:
                    self.provider.unload_with_error(err)
                raise
            self._access_token = str(tokens["access_token"])
            self._access_expires_at = (
                time.time() + int(tokens.get("expires_in", 3600)) - ACCESS_EXPIRY_MARGIN
            )
            # Anghami may rotate the refresh token, which retires the previous one
            if (
                new_refresh_token := tokens.get("refresh_token")
            ) and new_refresh_token != refresh_token:
                self.provider._update_setup_data(CONF_REFRESH_TOKEN, str(new_refresh_token))
            return self._access_token

    def _headers(self, access_token: str) -> dict[str, str]:
        """Return the headers of an authenticated request."""
        return {
            "Authorization": f"Bearer {access_token}",
            # selects the language of the localized names in the response
            "Accept-Language": self.provider.mass.metadata.preferred_language,
        }


async def _parse_response(response: ClientResponse) -> dict[str, Any]:
    """Return the decoded body of a successful response, or raise the matching error."""
    if response.status == 200:
        result: dict[str, Any] = await response.json()
        return result
    message = f"Anghami returned HTTP {response.status} for {response.url.path}: "
    message += await response.text()
    if response.status == 401:
        raise LoginFailed(message)
    if response.status == 403:
        raise ProviderPermissionDenied(message)
    if response.status == 404:
        raise MediaNotFoundError(message)
    if response.status == 429:
        raise RateLimited(message, backoff_time=_rate_limit_backoff(response))
    if response.status >= 500:
        raise ResourceTemporarilyUnavailable(message)
    raise InvalidDataError(message)


async def _parse_token_response(response: ClientResponse) -> dict[str, Any]:
    """Return the decoded body of a token response, or raise the matching error."""
    if response.status == 429:
        raise RateLimited(
            "Anghami rate limits the sign-in", backoff_time=_rate_limit_backoff(response)
        )
    if response.status >= 500:
        raise ResourceTemporarilyUnavailable(f"Anghami sign-in failed with HTTP {response.status}")
    if response.status != 200:
        raise LoginFailed(f"Anghami refused the sign-in: {await response.text()}")
    tokens: dict[str, Any] = await response.json()
    if not tokens.get("access_token"):
        raise LoginFailed("Anghami returned no access token")
    return tokens


def _rate_limit_backoff(response: ClientResponse) -> int:
    """Return the seconds until the rate limit window of a response resets."""
    try:
        reset_at = int(response.headers.get(RATE_LIMIT_RESET_HEADER, ""))
    except ValueError:
        return 0
    return max(0, reset_at - int(time.time()))
