"""Setup flow for the Anghami provider."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

from music_assistant_models.errors import (
    LoginFailed,
    ResourceTemporarilyUnavailable,
    SetupFailedError,
)

from music_assistant.helpers.oauth import (
    OAUTH_STEP_TIMEOUT,
    authorization_code_from_params,
    hosted_bounce_redirect,
)
from music_assistant.helpers.pkce import generate_pkce_pair
from music_assistant.models.setup_flow import AbortFlow

from .api import client_credentials, get_current_user, request_tokens
from .constants import (
    CONF_REFRESH_TOKEN,
    CONF_USER_ID,
    OAUTH_AUTHORIZE_URL,
    OAUTH_SCOPE,
    OAUTH_SCOPE_STREAM,
)

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession


async def run_setup(session: SetupSession) -> None:
    """
    Run the Anghami setup flow.

    Signs the user in to Anghami in the browser and stores the resulting sign-in as setup data.

    :param session: The setup session driving the flow.
    """
    try:
        client_id, _ = client_credentials()
    except SetupFailedError as err:
        raise AbortFlow("missing_credentials") from err
    code_verifier, code_challenge = generate_pkce_pair()
    redirect_uri, state = hosted_bounce_redirect(session.callback_url)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": OAUTH_SCOPE,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    callback_params = await session.external(
        f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}",
        step_id="authenticate",
        expires_in=OAUTH_STEP_TIMEOUT,
    )
    code = authorization_code_from_params(callback_params)
    http_session = session.mass.http_session
    try:
        tokens = await request_tokens(
            http_session,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
        # the user can turn down playback on the consent page while still granting read access
        if OAUTH_SCOPE_STREAM not in str(tokens.get("scope", OAUTH_SCOPE)).split():
            raise AbortFlow("playback_not_allowed")
        if not (refresh_token := tokens.get("refresh_token")):
            raise AbortFlow("login_failed")
        user = await get_current_user(http_session, str(tokens["access_token"]))
    except LoginFailed as err:
        raise AbortFlow("login_failed") from err
    except ResourceTemporarilyUnavailable as err:
        raise AbortFlow("anghami_unreachable") from err
    user_id = str(user["id"]["value"])
    if await _account_in_use(session, user_id):
        raise AbortFlow("account_already_configured")
    setup_data = dict(session.context.setup_data)
    setup_data[CONF_REFRESH_TOKEN] = str(refresh_token)
    setup_data[CONF_USER_ID] = user_id
    await session.finish(setup_data)


async def _account_in_use(session: SetupSession, user_id: str) -> bool:
    """Return whether another Anghami provider instance is already set up for this account."""
    mass = session.mass
    for config in await mass.config.get_provider_configs(provider_domain=session.context.domain):
        if config.instance_id == session.context.instance_id:
            continue
        stored = mass.config.get_provider_setup_value(config.instance_id, CONF_USER_ID)
        if stored and str(stored) == user_id:
            return True
    return False
