"""Auth routes: login (username+password -> role-carrying JWT) + /me.

- POST /api/auth/login {username?, password} -> {token, role, username}
  username defaults to the built-in admin ("admin"); the built-in admin is
  verified against ADMIN_PASSWORD (env), every other username against its DB
  users row (pbkdf2 hash).
- GET /api/auth/me -> {username, role} for the current session token.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..auth import (
    ADMIN_USERNAME,
    require_auth,
    role_from_claims,
    verify_password_hash,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    # username defaults to the built-in admin so existing single-password
    # clients (which POST only {password}) keep logging in as admin.
    username: str = Field(default=ADMIN_USERNAME, max_length=64)
    password: str


@router.post("/login")
async def login(body: LoginRequest, request: Request) -> dict:
    state = request.app.state
    auth = state.auth
    client_ip = request.client.host if request.client else "unknown"
    if not auth.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts — try again later")

    username = (body.username or ADMIN_USERNAME).strip()
    role = "viewer"
    ok = False
    if username == ADMIN_USERNAME:
        # Built-in admin: env-password-controlled, always role admin.
        ok = auth.verify_password(body.password)
        role = "admin"
    else:
        user = await state.db.get_user_by_username(username)
        # OFF THE EVENT LOOP. verify_password_hash is 200k PBKDF2 iterations,
        # ~60-100 ms during which NOTHING else in the process runs — not the WS
        # broadcast, not an ingest handoff, not a recorder respawn. That is an
        # UNAUTHENTICATED stall: anyone who can reach /api/auth/login can pause
        # the whole NVR at will, no credentials needed. hashlib releases the GIL,
        # so the offload is real work moved off the loop, not just deferred.
        if user is not None and await run_in_threadpool(
            verify_password_hash, body.password, user["password_hash"]
        ):
            ok = True
            role = user["role"]

    if not ok:
        # Only failures count toward the rate-limit window.
        auth.record_failed_login(client_ip)
        # Small constant delay blunts timing/bruteforce probing.
        await asyncio.sleep(0.5)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    auth.clear_failed_logins(client_ip)
    return {
        "token": auth.create_session_token(username, role),
        "role": role,
        "username": username,
    }


@router.get("/me")
async def me(claims: dict = Depends(require_auth)) -> dict:
    """Identity + role of the current session token (backward-compat: a legacy
    no-role token with sub 'admin' resolves to admin)."""
    return {"username": claims.get("sub") or ADMIN_USERNAME, "role": role_from_claims(claims)}


@router.get("/verify-stream")
async def verify_stream(request: Request, token: str = Query(default="")) -> Response:
    """Gate for nginx `auth_request` in front of the go2rtc live-stream proxy.
    200 = allowed, 401 = refused. Body is never used.

    WHY THIS EXISTS: `location /go2rtc/` proxied the streaming endpoints with no
    authentication whatsoever. The site is published at nvr.example.com,
    and camera names are a short lowercase slug, so
    `GET /go2rtc/api/ws?src=<guess>` returned live video of that camera — indoor
    ones included — to anyone on the internet. Every other control in this
    system (JWT sessions, RBAC, media-token binding, Privacy Mode) sat behind
    that one open location block.

    A SESSION token only. Media-scope tokens are refused explicitly: they are
    minted into notification bodies and retained MQTT messages, so accepting one
    here would mean a single push notification granted permanent live-view
    access to every camera — trading one hole for a quieter one.

    Deliberately NOT `Depends(require_auth)`: this is called on an nginx
    subrequest with no Authorization header, only the `?token=` nginx copies
    from the original query string.
    """
    claims = request.app.state.auth.decode(token) if token else None
    if claims is None or claims.get("scope") == "media":
        # 401 (not 403) so nginx's auth_request maps it to a plain 401.
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return Response(status_code=204)
