"""Tests for the Anghami setup flow."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from music_assistant_models.enums import FlowStepType

from music_assistant.helpers.oauth import HOSTED_CALLBACK_URL
from music_assistant.helpers.pkce import pkce_challenge
from music_assistant.models.setup_flow import AbortFlow, SetupFlowContext, SetupSession
from music_assistant.providers.anghami import setup_flow
from music_assistant.providers.anghami.constants import CONF_REFRESH_TOKEN, CONF_USER_ID

USER = {"id": {"value": "u1"}, "displayName": "Listener"}


def _make_session(finish_handler: Any, other_user_id: str = "u2") -> SetupSession:
    """Build a SetupSession backed by a Mock mass, with one other instance of another account."""
    mass = Mock()
    mass.webserver.base_url = "http://ma.local:8095"
    mass.config.get_provider_configs = AsyncMock(return_value=[Mock(instance_id="anghami--other")])
    mass.config.get_provider_setup_value.return_value = other_user_id
    context = SetupFlowContext(kind="setup", reason="user", domain="anghami")
    return SetupSession(mass, "flow-test", context, finish_handler)


async def _sign_in(session: SetupSession) -> dict[str, list[str]]:
    """Wait for the browser step, return its authorize query and answer it with a code."""
    deadline = time.monotonic() + 5
    while not (session.current_step and session.current_step.type == FlowStepType.EXTERNAL):
        assert time.monotonic() < deadline, "no external step published"
        await asyncio.sleep(0.01)
    assert session.current_step.url is not None
    query = parse_qs(urlparse(session.current_step.url).query)
    await session.handle_callback(Mock(query={"code": "auth-code"}, method="GET"))
    return query


async def test_run_setup_stores_refresh_grant() -> None:
    """The flow signs in with PKCE and stores the refresh token as setup data."""
    collected: dict[str, Any] = {}

    async def finish_handler(_session: SetupSession, values: dict[str, Any]) -> dict[str, str]:
        collected.update(values)
        return {"instance_id": "anghami--test"}

    session = _make_session(finish_handler)
    tokens = {"access_token": "access", "refresh_token": "refresh", "scope": "read stream"}
    with (
        patch.object(setup_flow, "request_tokens", AsyncMock(return_value=tokens)) as request,
        patch.object(setup_flow, "get_current_user", AsyncMock(return_value=USER)),
    ):
        task = asyncio.create_task(setup_flow.run_setup(session))
        query = await _sign_in(session)
        await task

    assert query["client_id"] == ["client-id"]
    assert query["redirect_uri"] == [HOSTED_CALLBACK_URL]
    assert query["scope"] == ["read stream"]
    assert query["code_challenge_method"] == ["S256"]
    grant = request.call_args.args[1]
    assert grant["grant_type"] == "authorization_code"
    assert grant["code"] == "auth-code"
    assert pkce_challenge(grant["code_verifier"]) == query["code_challenge"][0]
    assert collected == {CONF_REFRESH_TOKEN: "refresh", CONF_USER_ID: "u1"}


async def test_run_setup_requires_playback_permission() -> None:
    """A sign-in that turned down playback does not finish the setup."""
    session = _make_session(AsyncMock())
    tokens = {"access_token": "access", "refresh_token": "refresh", "scope": "read"}
    with patch.object(setup_flow, "request_tokens", AsyncMock(return_value=tokens)):
        task = asyncio.create_task(setup_flow.run_setup(session))
        await _sign_in(session)
        with pytest.raises(AbortFlow, match="playback_not_allowed"):
            await task

    assert not session.finished


async def test_run_setup_without_credentials() -> None:
    """A build without the client credentials stops before opening the browser."""
    session = _make_session(AsyncMock())
    with (
        patch("music_assistant.providers.anghami.api.app_var", return_value=""),
        pytest.raises(AbortFlow, match="missing_credentials"),
    ):
        await setup_flow.run_setup(session)


async def test_run_setup_refuses_account_in_use() -> None:
    """An Anghami account that another instance already uses does not finish the setup."""
    session = _make_session(AsyncMock(), other_user_id="u1")
    tokens = {"access_token": "access", "refresh_token": "refresh", "scope": "read stream"}
    with (
        patch.object(setup_flow, "request_tokens", AsyncMock(return_value=tokens)),
        patch.object(setup_flow, "get_current_user", AsyncMock(return_value=USER)),
    ):
        task = asyncio.create_task(setup_flow.run_setup(session))
        await _sign_in(session)
        with pytest.raises(AbortFlow, match="account_already_configured"):
            await task

    assert not session.finished
