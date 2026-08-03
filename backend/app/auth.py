"""Auth: single admin password -> JWT HS256 sessions.

- Secret + VAPID keypair auto-generated once and persisted in
  {DATA_DIR}/secrets.json (chmod 600).
- Session tokens: 30-day expiry, no scope claim.
- Media tokens: scope="media", short-lived; embedded as ?token= in
  push-notification image URLs (browsers fetch those without headers).
  Media tokens are accepted only on media (image/clip) routes.
- Light in-memory rate limit on login: only FAILED attempts count toward the
  window; a successful login clears the IP's failure history.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from base64 import urlsafe_b64encode
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException, Query, Request, status

_ALGO = "HS256"

# Built-in admin username: env-password-controlled, always role admin, never a
# DB row. Reserved — the users table must never store this name.
ADMIN_USERNAME = "admin"
ROLES = ("admin", "viewer")

# ---------- password hashing (stdlib pbkdf2-sha256, no new dependency) ----------

_PBKDF2_ITERS = 200_000


def hash_password(password: str) -> str:
    """pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64> with a random per-user salt."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password_hash(password: str, stored: str) -> bool:
    """Constant-time verify against a stored pbkdf2_sha256 string. Any malformed
    stored value (or wrong password) returns False — never raises."""
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters_s))
        return hmac.compare_digest(dk, expected)
    except Exception:  # noqa: BLE001 — a malformed hash must never crash login
        return False


def role_from_claims(claims: dict[str, Any]) -> str:
    """Resolve the role for a decoded token. New tokens carry an explicit
    ``role``; a legacy token (sub=="admin", no role claim) is treated as admin
    so existing sessions keep working. Anything else defaults to viewer."""
    role = claims.get("role")
    if role in ROLES:
        return role
    return "admin" if claims.get("sub") == ADMIN_USERNAME else "viewer"


def _b64url(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def load_or_create_secrets(path: Path) -> dict[str, str]:
    """Load secrets.json, generating JWT secret + VAPID keys on first boot."""
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if all(k in data for k in ("jwt_secret", "vapid_private_key", "vapid_public_key")):
                return data
        except (json.JSONDecodeError, OSError):
            pass  # regenerate below (corrupt file)

    private_key = ec.generate_private_key(ec.SECP256R1())
    d = private_key.private_numbers().private_value.to_bytes(32, "big")
    public_point = private_key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    data = {
        "jwt_secret": os.urandom(32).hex(),
        # Raw base64url forms: what pywebpush (py-vapid) and the browser's
        # applicationServerKey both consume directly.
        "vapid_private_key": _b64url(d),
        "vapid_public_key": _b64url(public_point),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return data


class AuthService:
    def __init__(self, secret: str, admin_password: str, token_days: int, media_token_days: int):
        self._secret = secret
        self._admin_password = admin_password
        self._token_days = token_days
        self._media_token_days = media_token_days
        # login rate limiting: ip -> deque of FAILED-attempt timestamps
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._max_failures = 10
        self._window_s = 300.0
        # WHOLE-BACKEND failure history, independent of the per-IP buckets.
        #
        # The per-IP limit is only as trustworthy as the IP, and the IP comes
        # from a proxy header. Anything that can vary that header gets a fresh
        # bucket per request, which makes the per-IP limit decorative against
        # exactly the attacker it exists to stop. This counter cannot be
        # partitioned by anything the client controls.
        #
        # Sized so a household never meets it — a real person fat-fingering a
        # password, on every device at once, does not produce 100 failures in
        # five minutes — while automated guessing hits it almost immediately.
        # A tighter number would itself become a lockout vector: anyone able to
        # fail logins could deny the owner access to their own cameras.
        self._global_failures: deque[float] = deque()
        self._max_global_failures = 100
        self._max_tracked_ips = 4096

    # ---------- login ----------

    def _expire(self, failures: deque[float], now: float) -> None:
        while failures and now - failures[0] > self._window_s:
            failures.popleft()

    def check_rate_limit(self, client_ip: str) -> bool:
        """True if this IP may attempt a login.

        Only failed attempts (see record_failed_login) count toward the
        window, so any number of successful logins never trips the limiter.
        """
        now = time.monotonic()
        # Global gate first: it holds regardless of what the client claims its
        # address is.
        self._expire(self._global_failures, now)
        if len(self._global_failures) >= self._max_global_failures:
            return False
        failures = self._failures.get(client_ip)
        if failures is None:
            return True
        self._expire(failures, now)
        if not failures:
            # Fully expired: drop the key so the map can't grow unbounded.
            del self._failures[client_ip]
            return True
        return len(failures) < self._max_failures

    def record_failed_login(self, client_ip: str) -> None:
        now = time.monotonic()
        self._global_failures.append(now)
        self._expire(self._global_failures, now)
        # BOUND THE MAP. Keys are only reclaimed when the SAME key is revisited
        # after expiry (check_rate_limit above), so a client varying its
        # apparent address leaks one dict entry plus a deque per attempt —
        # unauthenticated, unbounded memory growth on the box. Once the map is
        # implausibly large for a real deployment, sweep the expired entries.
        if len(self._failures) > self._max_tracked_ips:
            for key in [k for k, v in self._failures.items()
                        if not v or now - v[-1] > self._window_s]:
                del self._failures[key]
        self._failures[client_ip].append(now)

    def clear_failed_logins(self, client_ip: str) -> None:
        """Successful login: forget this IP's failure history.

        Deliberately does NOT clear the global counter: a correct login proves
        this client is legitimate, not that the hundred failures from elsewhere
        in the last five minutes were.
        """
        self._failures.pop(client_ip, None)

    def verify_password(self, password: str) -> bool:
        """Verify a password against the built-in admin's env password."""
        return hmac.compare_digest(password.encode(), self._admin_password.encode())

    # ---------- tokens ----------

    def create_session_token(self, username: str = ADMIN_USERNAME, role: str = "admin") -> str:
        now = int(time.time())
        return jwt.encode(
            {"sub": username, "role": role, "iat": now, "exp": now + self._token_days * 86400},
            self._secret,
            algorithm=_ALGO,
        )

    def create_media_token(
        self,
        username: str = ADMIN_USERNAME,
        role: str = "viewer",
        resource: str = "",
    ) -> str:
        """Mint a media-scope token for embedding in a URL.

        **Defaults to the LOWEST role, and callers must not raise it.** These
        tokens are built to leak: they are embedded in notification snapshot
        URLs (events_pipeline._media_url) and published RETAINED to the
        operator's MQTT broker (mqtt_ha._snapshot_url), where any subscriber
        reads them — and they are valid for `media_token_days` (7). Anything
        the role unlocks is therefore effectively public to anyone who sees a
        notification.

        The default used to be "admin", which silently made every notification
        URL an admin-role credential and let a leaked one through
        `require_media_admin` (suppression thumbnails). Nothing needs an admin
        media token: the media routes proper are any-auth, and the admin UI
        authenticates with the operator's SESSION token (web AuthImage sends an
        Authorization header; iOS mediaURL appends its session `?token=`) — not
        with one of these.

        The media scope is still the primary guard: `require_auth` rejects
        media-scope tokens outright, so one of these can never reach a normal
        API route regardless of role.

        `resource` binds the token to ONE object (e.g. ``"event:1234"``). Always
        pass it — an unbound token is only tolerated so that tokens already
        published to a retained MQTT topic keep working until they expire.
        """
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": username,
            "role": role,
            "scope": "media",
            "iat": now,
            "exp": now + self._media_token_days * 86400,
        }
        # BIND IT TO ONE OBJECT. Scope alone said "this is a media token"; it did
        # not say WHICH media. A token minted for one event's snapshot therefore
        # opened every other event's snapshot and every camera's live JPEG for
        # its full 7-day life. `res` narrows it to the single item it was created
        # for (see media_resource_of / require_media_auth).
        if resource:
            payload["res"] = resource
        return jwt.encode(payload, self._secret, algorithm=_ALGO)

    def decode(self, token: str) -> Optional[dict[str, Any]]:
        try:
            return jwt.decode(token, self._secret, algorithms=[_ALGO])
        except jwt.InvalidTokenError:
            return None


def _get_auth(request: Request) -> AuthService:
    return request.app.state.auth


def _extract_bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


async def require_auth(request: Request) -> dict[str, Any]:
    """Standard API auth: Authorization: Bearer <session token>."""
    token = _extract_bearer(request)
    claims = _get_auth(request).decode(token) if token else None
    if claims is None or claims.get("scope") == "media":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return claims


async def require_admin(request: Request) -> dict[str, Any]:
    """Admin-only API auth: a valid session token whose role resolves to admin.

    Reuses require_auth (which rejects missing/expired/media-scope tokens with
    401), then enforces the role — a viewer gets 403.
    """
    claims = await require_auth(request)
    if role_from_claims(claims) != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return claims


def media_resource_of(request: Request) -> str:
    """The single media object this request is asking for, as a `res` claim.

    Derived from the path params, so it cannot disagree with what the route
    will actually serve. Empty when a route has no single subject — and an
    empty subject REFUSES a bound token rather than waving it through.
    """
    params = request.path_params
    event_id = params.get("event_id")
    if event_id is not None:
        return f"event:{event_id}"
    name = params.get("name")
    if name is not None:
        return f"camera:{name}"
    return ""


async def require_media_auth(
    request: Request, token: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    """Media route auth: Bearer header OR ?token= (session or media scope).

    ?token= support exists because push-notification images are fetched by
    the browser/OS with no way to attach headers.

    NOTE: any-auth by design — a viewer may fetch event snapshots/clips. Use
    require_media_admin for imagery that belongs to an admin-only feature.

    A media token carrying `res` may only reach the ONE object it names. These
    tokens are designed to leave the system — they ride in notification bodies
    and in RETAINED MQTT messages every current and future subscriber can read
    — so "a token someone saw" must not become "every snapshot for 7 days".
    """
    raw = _extract_bearer(request) or token
    claims = _get_auth(request).decode(raw) if raw else None
    if claims is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    bound = claims.get("res") if claims.get("scope") == "media" else None
    if bound and bound != media_resource_of(request):
        # 404, not 403: a bound token probing for other events should not be
        # able to tell "exists but forbidden" from "does not exist".
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return claims


# WebSocket subprotocol carrying a bearer token: the client offers
# `Sec-WebSocket-Protocol: bearer, <jwt>` and the server echoes back "bearer".
_WS_BEARER = "bearer"


def ws_token(websocket: Any, query_token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Extract a WebSocket's session token, preferring the SUBPROTOCOL header.

    Returns ``(token, subprotocol_to_echo)``. The caller MUST pass that
    subprotocol to ``websocket.accept(subprotocol=...)`` — a browser aborts the
    connection if it offered subprotocols and the server selects none.

    WHY NOT ?token=. nginx writes the full request line, query string included,
    into its ERROR log, and `log_format` does not apply there — so any warn on a
    tokened URL prints a live 30-day admin credential in cleartext. That is not
    hypothetical: it happened, and the token had to be rotated. A browser cannot
    set an Authorization header on a WebSocket handshake, but it CAN offer
    subprotocols, which travel in a header and never appear in a URL.

    `?token=` is still accepted so an older client keeps working across the
    upgrade; it can be dropped a release later.
    """
    offered = ""
    try:
        offered = websocket.headers.get("sec-websocket-protocol") or ""
    except Exception:  # noqa: BLE001 — a fake/odd transport must not 500 the route
        offered = ""
    parts = [p.strip() for p in offered.split(",") if p.strip()]
    if len(parts) >= 2 and parts[0] == _WS_BEARER:
        return parts[1], _WS_BEARER
    return query_token, None


async def require_stream_auth(
    request: Request, token: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    """Auth for routes the BROWSER MEDIA STACK fetches — header OR ?token= —
    but NEVER a media-scope token.

    The distinction from require_media_auth: `?token=` here exists because an
    HLS playlist, a video segment and a download link cannot carry an
    Authorization header, NOT because the URL is meant to be shared. So a
    SESSION token in the query string is fine, and a media token — the kind
    that ships out in notifications and retained MQTT messages — is refused.

    Applied at the /api/recordings router, this is what stops one leaked
    notification token from unlocking 24/7 archive playback and export for
    every camera.
    """
    raw = _extract_bearer(request) or token
    claims = _get_auth(request).decode(raw) if raw else None
    if claims is None or claims.get("scope") == "media":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return claims


async def require_media_admin(
    request: Request, token: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    """Admin-only MEDIA route auth: require_media_auth (so an <img>/AsyncImage
    can pass ?token= with no headers) PLUS the admin role — a viewer gets 403.

    For imagery that only an admin screen ever shows, where require_media_auth
    alone would leave the bytes readable by any authenticated viewer who
    guesses a sequential id (the route's own listing being admin-gated does not
    protect it — the ids are enumerable)."""
    claims = await require_media_auth(request, token)
    if role_from_claims(claims) != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return claims
