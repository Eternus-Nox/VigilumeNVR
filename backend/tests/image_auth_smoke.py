#!/usr/bin/env python3
"""Recognition imagery must be fetchable WITHOUT an Authorization header.

THE BUG THIS EXISTS TO PREVENT
==============================
`routers/profiles.py` gates its whole router with `require_admin`, which takes
a token only from an `Authorization: Bearer` header. Its image routes carry
`require_media_admin`, which exists precisely so a header-less `<img>` or
SwiftUI `AsyncImage` can pass `?token=` instead.

A router-level dependency runs on EVERY route beneath it, BEFORE the route's
own. So while the images sat on that router, `require_media_admin` was
unreachable and every header-less fetch 401'd at the gate.

WHY IT SURVIVED SO LONG: the web client never hits it. `AuthImage` fetches the
bytes itself with `fetch()` and CAN set a header, so every crop loaded there.
On iOS `AsyncImage` cannot set one, so the Unknown Faces grid was a column of
broken thumbnails — and the one screen whose entire job is "look at this face
and decide who it is" had no face on it.

The fix is a second router with no router-level dependency. This suite pins
that, because the regression is a one-word edit (`dependencies=[...]` on
`media_router`) that nothing else would catch: the web would keep working, the
backend tests would keep passing, and only the phone would go blank.

Offline: no database, no models, no network. It builds routers from the REAL
auth dependencies, so it tracks what those functions actually do rather than a
paraphrase of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import AuthService  # noqa: E402
from app.routers import profiles as profiles_router  # noqa: E402

_failures: list[str] = []
_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


#: Every recognition image route, by the path a client actually requests.
#: A new one added to `media_router` and not listed here is caught below.
IMAGE_PATHS = (
    "/api/recognition/samples/1/image.jpg",
    "/api/recognition/candidates/1/image.jpg",
    "/api/recognition/candidates/1/frame.jpg",
)


class _EmptyCursor:
    async def fetchone(self):
        return None

    async def fetchall(self):
        return []


class _EmptyDB:
    """Every query finds nothing.

    Deliberately empty: this suite is about who may REACH the handler, not what
    the handler returns. With no rows each route answers 404, which is a clean
    "auth passed, the object is not here" — and unmistakable from the 401 that
    is the actual regression.
    """

    class _Conn:
        async def execute(self, *_a, **_k):
            return _EmptyCursor()

    conn = _Conn()


def build() -> tuple[TestClient, AuthService]:
    """The real routers, mounted the way main.py mounts them."""
    auth = AuthService(
        secret="s" * 32, admin_password="pw", token_days=30, media_token_days=7
    )
    app = FastAPI()
    app.include_router(profiles_router.router)
    app.include_router(profiles_router.media_router)
    app.state.auth = auth
    app.state.db = _EmptyDB()
    return TestClient(app), auth


def routing_checks() -> None:
    print("\nevery image route is on the header-free router")
    media_paths = {r.path for r in profiles_router.media_router.routes}  # type: ignore[attr-defined]
    for path in IMAGE_PATHS:
        template = (
            path.replace("/1/", "/{sample_id}/")
            if "samples" in path
            else path.replace("/1/", "/{candidate_id}/")
        )
        check(template in media_paths, f"{template} is on media_router")

    admin_paths = {r.path for r in profiles_router.router.routes}  # type: ignore[attr-defined]
    stragglers = sorted(p for p in admin_paths if p.endswith(".jpg"))
    check(not stragglers,
          f"...and NO image route is left on the admin-gated router "
          f"(stragglers: {stragglers})")

    check(not profiles_router.media_router.dependencies,
          "media_router carries NO router-level dependency. Adding one puts the "
          "image routes back behind a header-only gate and blanks the iOS grid, "
          "while the web keeps working and nothing else fails")


def auth_checks() -> None:
    client, auth = build()
    session = auth.create_session_token()

    print("\na header-less fetch reaches the route (this is what iOS sends)")
    for path in IMAGE_PATHS:
        status = client.get(path, params={"token": session}).status_code
        # 404 is the right answer here: auth PASSED and the handler looked for a
        # row that this bare app has no database for. What must never come back
        # is 401 — that is the router gate refusing a request it never should
        # have seen.
        check(status != 401,
              f"{path} with ?token= is not 401 (got {status}) — AsyncImage "
              "cannot set a header, so 401 here is a blank tile on the phone")

    print("\n...and the header still works, for the web client")
    for path in IMAGE_PATHS:
        status = client.get(
            path, headers={"Authorization": f"Bearer {session}"}
        ).status_code
        check(status != 401, f"{path} with a Bearer header is not 401 (got {status})")

    print("\nit is still ADMIN-ONLY, and still refuses a media-scope token")
    viewer = auth.create_session_token(username="watcher", role="viewer")
    for path in IMAGE_PATHS:
        check(client.get(path, params={"token": viewer}).status_code == 403,
              f"{path} refuses a VIEWER — ids are sequential, so any authenticated "
              "token would otherwise walk the whole biometric store")

    # These are the tokens that leave the system, in notification bodies and in
    # RETAINED MQTT messages every future subscriber can read. Two independent
    # guards have to hold, because either alone has a gap.
    leaked = auth.create_media_token(resource="event:1")
    for path in IMAGE_PATHS:
        status = client.get(path, params={"token": leaked}).status_code
        check(status in (401, 403, 404),
              f"{path} refuses a leaked media token (got {status}) — one "
              "notification URL must not unlock the face gallery")

    # GUARD 1: the role. `create_media_token` defaults to the lowest role and
    # the docstring tells callers not to raise it — that default is what stops
    # an UNBOUND media token (which the `res` check below cannot catch) from
    # reaching an admin-only image.
    import jwt as _jwt  # noqa: PLC0415 — test-only, to read the claim back

    claims = _jwt.decode(leaked, "s" * 32, algorithms=["HS256"])
    check(claims.get("role") == "viewer",
          "a media token is minted as a VIEWER by default. Raising that default "
          "would make every notification URL an admin credential, and an "
          "unbound one would then open the whole biometric store")

    # GUARD 2: the `res` binding, which holds even if guard 1 is subverted.
    # These paths carry no event_id/name, so media_resource_of() is "" and a
    # bound token can never match — an admin-role media token is still refused.
    forged = auth.create_media_token(role="admin", resource="event:1")
    for path in IMAGE_PATHS:
        status = client.get(path, params={"token": forged}).status_code
        check(status in (401, 403, 404),
              f"{path} refuses even an ADMIN-role media token (got {status}) — "
              "the resource binding catches what the role default would miss")

    for path in IMAGE_PATHS:
        check(client.get(path).status_code == 401, f"{path} refuses no token at all")

    print("\nthe rest of the router is unchanged — header only, no ?token=")
    listing = "/api/recognition/candidates"
    check(client.get(listing, params={"token": session}).status_code == 401,
          "the candidate LISTING still refuses ?token=. Only imagery needs the "
          "query form; a list URL with a session token in it is a credential in "
          "a browser history for no reason")
    check(client.get(listing, headers={"Authorization": f"Bearer {session}"})
          .status_code != 401,
          "...while the header still gets in")


def main() -> int:
    routing_checks()
    auth_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (recognition imagery auth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
