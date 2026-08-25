"""Browser OAuth for cloud storage, completed on the NVR itself.

WHY THIS EXISTS. `rclone authorize` runs a one-shot web server on
127.0.0.1:53682 and asks the provider to redirect there. That address resolves
on the machine running the BROWSER, so when rclone is in a headless container
and the browser is on a laptop, the redirect lands nowhere. Hence the old
instruction to run rclone on your own desktop and paste a token back.

But an NVR on a home LAN is a reachable HTTP server — the operator is looking at
its web UI right now. So it can BE the redirect target: the provider sends the
browser to `http://<nvr>:8080/api/integrations/rclone/oauth/callback`, which is
this backend, and the whole handshake finishes without a terminal anywhere.

THE PRICE, and it is unavoidable rather than a design choice: a provider only
redirects to a URI registered on the OAuth app, and rclone's built-in app
registers only `localhost:53682`. Using our own redirect means using our own
app — so the operator creates a free app on the provider's developer site once
and pastes its client ID and secret. Five minutes in a browser, versus
installing rclone on a second machine. Both paths remain available.

SECURITY NOTES, since the callback is necessarily UNAUTHENTICATED (a provider
redirects a browser, carrying no Authorization header):
  * `state` is a 256-bit unguessable token and the ONLY thing that authorizes a
    callback. It is single-use and expires, so a replayed or guessed callback
    does nothing.
  * The pending flow holds the client secret in memory only, never on disk
    except as part of the finished rclone remote, and is never echoed back.
  * A callback for an unknown state is indistinguishable from an expired one on
    purpose — the response says "start again", not "that state was wrong".

PKCE (RFC 7636) IS NOT OPTIONAL HERE, and not merely for defence in depth.
Dropbox refuses outright:

    Invalid redirect_uri. When response_type=code without PKCE, only localhost
    URIs can start with "http://"; all others must start with "https://".

An NVR on a LAN is reached at `http://192.168.1.45:8080` — plain HTTP, not
localhost — so the whole in-browser flow depends on presenting a code
challenge. Which is the right rule: without PKCE, anyone who could observe the
redirect on an unencrypted hop could replay the code. With it, the code is
useless without the verifier, which never leaves this server.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

log = logging.getLogger(__name__)

# Where the provider sends the browser back. FIXED so the operator has one
# stable string to register, and so a mistyped path fails at registration time
# rather than silently mid-flow.
CALLBACK_PATH = "/api/integrations/rclone/oauth/callback"

# A pending flow is a few browser seconds plus however long the operator takes
# to log in and approve. Ten minutes is generous for that and short enough that
# an abandoned flow does not linger.
STATE_TTL_S = 600
# Cheap bound on concurrent flows; this is a single-admin appliance, so anything
# beyond a handful means something is wrong.
MAX_PENDING = 16


@dataclass(frozen=True)
class OAuthProvider:
    type: str
    auth_url: str
    token_url: str
    # Extra query params the provider needs to return a REFRESH token. Without
    # these each vendor hands back an access token that expires in hours and
    # cannot be renewed, so the archive silently stops working overnight.
    auth_extra: dict[str, str]
    scope: str = ""


OAUTH_PROVIDERS: dict[str, OAuthProvider] = {
    "dropbox": OAuthProvider(
        type="dropbox",
        auth_url="https://www.dropbox.com/oauth2/authorize",
        token_url="https://api.dropboxapi.com/oauth2/token",
        # token_access_type=offline is what makes Dropbox return a refresh
        # token; without it the remote works for ~4 hours and then stops.
        auth_extra={"token_access_type": "offline"},
    ),
    "drive": OAuthProvider(
        type="drive",
        auth_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scope="https://www.googleapis.com/auth/drive",
        # prompt=consent forces Google to re-issue a refresh token; it omits one
        # on re-authorization otherwise, which produces a remote that works now
        # and breaks after the first token expiry.
        auth_extra={"access_type": "offline", "prompt": "consent"},
    ),
    "onedrive": OAuthProvider(
        type="onedrive",
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        scope="Files.ReadWrite.All offline_access",
        auth_extra={},
    ),
}


def developer_console(type_: str) -> str:
    """Where the operator creates the app whose id/secret they will paste."""
    return {
        "dropbox": "https://www.dropbox.com/developers/apps",
        "drive": "https://console.cloud.google.com/apis/credentials",
        "onedrive": "https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps",
    }.get(type_, "")


class OAuthError(ValueError):
    """A rejected OAuth step; the message is written for the operator."""


@dataclass
class Pending:
    """One in-flight authorization, held only until its callback arrives."""

    state: str
    remote_name: str
    provider: OAuthProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    created_at: float
    # PKCE: the verifier stays HERE and is sent only on the back-channel token
    # exchange; only its SHA-256 challenge ever travels through the browser.
    code_verifier: str = ""


class PendingFlows:
    """In-memory store for authorizations between start and callback.

    Deliberately NOT persisted. These live for seconds, they hold a client
    secret, and a backend restart mid-flow is far better handled by asking the
    operator to click Connect again than by leaving credentials on disk waiting
    for a callback that may never come.
    """

    def __init__(self) -> None:
        self._flows: dict[str, Pending] = {}

    def _sweep(self, now: float) -> None:
        for state, flow in list(self._flows.items()):
            if now - flow.created_at > STATE_TTL_S:
                del self._flows[state]

    def add(self, flow: Pending) -> None:
        now = time.time()
        self._sweep(now)
        if len(self._flows) >= MAX_PENDING:
            # Drop the oldest rather than refusing: a stuck flow from an
            # abandoned tab must never block a real one.
            oldest = min(self._flows.values(), key=lambda f: f.created_at)
            del self._flows[oldest.state]
        self._flows[flow.state] = flow

    def take(self, state: str) -> Optional[Pending]:
        """Consume a pending flow. SINGLE USE — a replayed callback finds
        nothing, so an intercepted redirect cannot be used twice."""
        now = time.time()
        self._sweep(now)
        return self._flows.pop(state, None)

    def __len__(self) -> int:
        return len(self._flows)


def validate_origin(origin: str) -> str:
    """The browser origin the callback will come back to.

    Only the scheme and host matter — a path, query or fragment here would
    produce a redirect URI that never matches the one registered, and the
    failure would surface at the provider as an opaque error.
    """
    origin = (origin or "").strip().rstrip("/")
    if not origin:
        raise OAuthError("Could not work out this server's address.")
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OAuthError("The address must start with http:// or https://.")
    if parsed.path or parsed.query or parsed.fragment:
        raise OAuthError("Use just the scheme and host, e.g. http://192.168.1.45:8080")
    return f"{parsed.scheme}://{parsed.netloc}"


def redirect_uri_for(origin: str) -> str:
    return validate_origin(origin) + CALLBACK_PATH


def code_challenge_for(verifier: str) -> str:
    """S256 challenge: base64url(sha256(verifier)), unpadded per RFC 7636 §4.2.

    Unpadded matters — providers compare the string literally, and a trailing
    '=' makes an otherwise correct challenge mismatch at exchange time.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_auth_url(flow: Pending) -> str:
    params = {
        "client_id": flow.client_id,
        "response_type": "code",
        "redirect_uri": flow.redirect_uri,
        "state": flow.state,
        # See the module docstring: without these Dropbox rejects any http://
        # redirect that is not localhost, which is every LAN address.
        "code_challenge": code_challenge_for(flow.code_verifier),
        "code_challenge_method": "S256",
        **flow.provider.auth_extra,
    }
    if flow.provider.scope:
        params["scope"] = flow.provider.scope
    return f"{flow.provider.auth_url}?{urlencode(params)}"


def start_flow(
    *,
    remote_name: str,
    type_: str,
    client_id: str,
    client_secret: str,
    origin: str,
) -> Pending:
    prov = OAUTH_PROVIDERS.get(type_)
    if prov is None:
        raise OAuthError(f"{type_} does not use browser sign-in.")
    client_id = (client_id or "").strip()
    client_secret = (client_secret or "").strip()
    if not client_id or not client_secret:
        raise OAuthError("Paste the app key and app secret from the provider first.")
    return Pending(
        state=secrets.token_urlsafe(32),
        remote_name=remote_name,
        provider=prov,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri_for(origin),
        created_at=time.time(),
        # token_urlsafe(64) yields ~86 chars drawn from [A-Za-z0-9_-], inside
        # RFC 7636's 43-128 unreserved-character range with no escaping needed.
        code_verifier=secrets.token_urlsafe(64),
    )


def token_request_body(flow: Pending, code: str) -> dict[str, str]:
    """Form body for the code -> token exchange (RFC 6749 §4.1.3 + RFC 7636 §4.5).

    Both `client_secret` AND `code_verifier` are sent. The verifier is what the
    provider demanded to allow a plain-HTTP LAN redirect at all; the secret is
    still needed because this is a confidential client whose credentials rclone
    will reuse to refresh the token later.
    """
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": flow.client_id,
        "client_secret": flow.client_secret,
        "redirect_uri": flow.redirect_uri,
    }
    if flow.code_verifier:
        body["code_verifier"] = flow.code_verifier
    return body


def to_rclone_token(payload: dict[str, Any], *, now: Optional[datetime] = None) -> str:
    """A provider's token response -> the JSON blob rclone stores.

    rclone wants ``{access_token, token_type, refresh_token, expiry}`` with an
    RFC3339 expiry, and uses refresh_token + expiry to renew itself. A response
    WITHOUT a refresh token is rejected here rather than stored: it would work
    for a few hours and then leave the nightly archive failing at 03:00 with an
    auth error nobody is awake to read.
    """
    import json

    access = str(payload.get("access_token") or "")
    if not access:
        raise OAuthError("The provider did not return an access token.")
    refresh = str(payload.get("refresh_token") or "")
    if not refresh:
        raise OAuthError(
            "The provider returned a temporary token with no way to renew it. "
            "Remove Vigilume from your account's connected apps and try again "
            "so it issues a fresh, renewable token."
        )
    now = now or datetime.now(timezone.utc)
    try:
        lifetime = int(payload.get("expires_in") or 0)
    except (TypeError, ValueError):
        lifetime = 0
    # A conservative default when the provider omits expires_in: rclone refreshes
    # on expiry, so guessing SHORT costs an extra refresh and guessing long costs
    # a failed transfer.
    expiry = now + timedelta(seconds=lifetime if lifetime > 0 else 3600)
    return json.dumps({
        "access_token": access,
        "token_type": str(payload.get("token_type") or "bearer"),
        "refresh_token": refresh,
        "expiry": expiry.isoformat(),
    })


def remote_values(flow: Pending, token_blob: str) -> dict[str, str]:
    """The rclone config keys for a browser-authorized remote.

    client_id and client_secret are stored ALONGSIDE the token deliberately:
    the token came from the operator's own OAuth app, and rclone needs that
    app's credentials to refresh it. Store only the token and the remote works
    until the first expiry, then stops.
    """
    return {
        "client_id": flow.client_id,
        "client_secret": flow.client_secret,
        "token": token_blob,
    }
