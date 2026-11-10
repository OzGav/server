"""Tests for the Anghami API client."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest
from music_assistant_models.errors import (
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    ProviderPermissionDenied,
    RetriesExhausted,
    SetupFailedError,
)

from music_assistant.providers.anghami import api
from music_assistant.providers.anghami.constants import CONF_REFRESH_TOKEN, OAUTH_TOKEN_URL
from music_assistant.providers.anghami.provider import AnghamiProvider

from .conftest import FakeResponse, FakeSession

TOKENS = {"access_token": "access-1", "expires_in": 3600, "token_type": "Bearer"}


async def test_request_tokens_posts_client_secret(mass: Mock) -> None:
    """The token request is a form post that carries the client id and secret."""
    session = FakeSession([FakeResponse(200, TOKENS)])
    mass.http_session = session

    tokens = await api.request_tokens(
        mass.http_session,
        {"grant_type": "refresh_token", "refresh_token": "refresh-1"},
    )

    assert tokens == TOKENS
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", OAUTH_TOKEN_URL)
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
        "client_id": "client-id",
        "client_secret": "client-secret",
    }


async def test_request_tokens_refused(mass: Mock) -> None:
    """A refused grant is a login failure."""
    mass.http_session = FakeSession([FakeResponse(400, {"error": "invalid_grant"})])

    with pytest.raises(LoginFailed):
        await api.request_tokens(mass.http_session, {"grant_type": "refresh_token"})


async def test_get_current_user_sends_access_token() -> None:
    """The signed-in user is fetched with the given access token."""
    user = {"id": {"value": "u1"}, "displayName": "Listener"}
    session = FakeSession([FakeResponse(200, {"user": user})])

    assert await api.get_current_user(session, "access-1") == user  # type: ignore[arg-type]
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", "https://sdk.anghami.com/v1/user/me")
    assert kwargs["headers"] == {"Authorization": "Bearer access-1"}


def test_client_credentials_missing() -> None:
    """A build without the bundled credentials can not set up Anghami."""
    with (
        patch.object(api, "app_var", side_effect=lambda name: "" if "secret" in name else "x"),
        pytest.raises(SetupFailedError),
    ):
        api.client_credentials()


async def test_request_refreshes_rejected_token(mass: Mock, provider: AnghamiProvider) -> None:
    """A refused access token is refreshed once, and a rotated refresh token is stored."""
    session = FakeSession(
        [
            FakeResponse(200, TOKENS),
            FakeResponse(401, {"message": "token expired"}),
            FakeResponse(200, {**TOKENS, "access_token": "access-2", "refresh_token": "refresh-2"}),
            FakeResponse(200, {"user": {"displayName": "Listener"}}),
        ]
    )
    mass.http_session = session

    with patch.object(provider, "_update_setup_data") as update_setup_data:
        data = await provider.client.get("/v1/user/me")

    assert data == {"user": {"displayName": "Listener"}}
    assert session.calls[1][2]["headers"]["Authorization"] == "Bearer access-1"
    assert session.calls[3][2]["headers"]["Authorization"] == "Bearer access-2"
    update_setup_data.assert_called_once_with(CONF_REFRESH_TOKEN, "refresh-2")


async def test_paginate_follows_page_tokens(mass: Mock, provider: AnghamiProvider) -> None:
    """Pages are fetched at the largest size until no next page token is returned."""
    session = FakeSession(
        [
            FakeResponse(200, TOKENS),
            FakeResponse(200, {"items": [{"n": 1}, {"n": 2}], "nextPageToken": "next"}),
            FakeResponse(200, {"items": [{"n": 3}]}),
        ]
    )
    mass.http_session = session

    pages = [page async for page in provider.client.paginate("/v1/library/likes", "items")]

    assert pages == [[{"n": 1}, {"n": 2}], [{"n": 3}]]
    assert session.calls[1][2]["params"] == {"page_size": 100}
    assert session.calls[2][2]["params"] == {"page_size": 100, "page_token": "next"}


async def test_batch_get_chunks_and_skips_errors(mass: Mock, provider: AnghamiProvider) -> None:
    """Ids are fetched in batches of 100 and ids Anghami can not return are left out."""
    ids = [f"s{index}" for index in range(150)]
    first: dict[str, dict[str, Any]] = {
        item_id: {"song": {"id": {"value": item_id}}} for item_id in ids[:100]
    }
    first["s0"] = {"error": {"code": "ERROR_CODE_NOT_FOUND", "message": "song not found"}}
    second = {item_id: {"song": {"id": {"value": item_id}}} for item_id in ids[100:]}
    session = FakeSession(
        [
            FakeResponse(200, TOKENS),
            FakeResponse(200, {"results": first}),
            FakeResponse(200, {"results": second}),
        ]
    )
    mass.http_session = session

    result = await provider.client.batch_get("song", [*ids, "s1"])

    assert len(result) == 149
    assert "s0" not in result
    assert len(session.calls[1][2]["json"]["songIdList"]) == 100
    assert len(session.calls[2][2]["json"]["songIdList"]) == 50


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, InvalidDataError),
        (403, ProviderPermissionDenied),
        (404, MediaNotFoundError),
        (500, RetriesExhausted),
    ],
)
async def test_error_statuses(
    mass: Mock, provider: AnghamiProvider, status: int, error: type[Exception]
) -> None:
    """Error statuses map to the matching Music Assistant errors."""
    mass.http_session = FakeSession(
        [FakeResponse(200, TOKENS), FakeResponse(status, {"message": "nope"})]
    )

    with pytest.raises(error):
        await provider.client.get("/v1/music/songs/s1")


async def test_refused_twice_is_login_failure(mass: Mock, provider: AnghamiProvider) -> None:
    """A token refused again right after a refresh means the sign-in is no longer valid."""
    mass.http_session = FakeSession(
        [
            FakeResponse(200, TOKENS),
            FakeResponse(401),
            FakeResponse(200, TOKENS),
            FakeResponse(401),
        ]
    )

    with pytest.raises(LoginFailed):
        await provider.client.get("/v1/user/me")


async def test_rate_limit_holds_back_until_reset(mass: Mock, provider: AnghamiProvider) -> None:
    """A rate limit holds back further requests until the window in the reset header."""
    reset_at = str(int(time.time()) + 30)
    mass.http_session = FakeSession(
        [
            FakeResponse(200, TOKENS),
            FakeResponse(429, {"message": "slow down"}, {"X-RateLimit-Reset": reset_at}),
        ]
    )

    with pytest.raises(RetriesExhausted):
        await provider.client.get("/v1/music/songs/s1")

    assert 28 <= provider.client.throttler.cooldown_remaining <= 30
