"""Smoke suite for multi-user RBAC (docs/CONTRACTS.md — RBAC addendum).

Security-critical: a viewer must not reach ANY admin capability. This suite
enumerates the FULL authorization matrix and asserts:

  - env-admin login works and carries role "admin"; GET /api/auth/me
  - pbkdf2 password hashing round-trips + rejects tampered/malformed hashes
  - user CRUD: create viewer (login works, wrong password 401), reserved
    "admin" -> 400, duplicate -> 409, short password/bad username rejected
  - GET /api/auth/me for both roles
  - EVERY any-auth route: a viewer token is NOT blocked (200/expected), and
    an admin token is NOT blocked
  - EVERY admin-only route (enumerated): a viewer token -> 403, an admin
    token -> not 401/403
  - guards: cannot demote the last admin; built-in admin never a deletable
    row; cannot create the reserved name
  - legacy no-role token (sub "admin") treated as admin
  - media token carries role; rejected on non-media routes, accepted on media

Runs the real FastAPI app via TestClient. Seeded cameras point at
127.0.0.1 (nothing listening -> instant connection refusal) so admin-side
camera-control routes fail fast at the network layer AFTER passing authz.

Usage: python backend/tests/rbac_smoke.py  (needs backend deps installed)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Environment must be clean before app config is instantiated (lifespan).
for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "admin-secret"
os.environ["PUBLIC_URL"] = ""
# Unroutable local port -> instant connection refusal, so go2rtc config
# syncs during camera CRUD never slow the suite down.
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-rbac-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import jwt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from app.auth import hash_password, role_from_claims, verify_password_hash  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import cameras as cameras_router  # noqa: E402

# TEST-NET / localhost cameras are unreachable; keep the add/edit probe fast.
cameras_router._PROBE_TIMEOUT_S = 0.3

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def login(client: TestClient, username: str, password: str):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def add_camera(client: TestClient, headers: dict, name: str,
               model: str = "IP8M-2779EW-AI") -> None:
    resp = client.post("/api/cameras", headers=headers, json={
        "name": name, "friendly_name": name.title(), "model": model,
        "ip": "127.0.0.1", "username": "u", "password": "p",
    })
    assert resp.status_code == 201, resp.text


# ---------------- password hashing ----------------


def rate_limit_checks() -> None:
    """The login limiter must not be defeatable by varying the apparent client.

    `check_rate_limit` is keyed on `request.client.host`, which on this
    deployment derives from a PROXY HEADER — uvicorn runs with
    `--forwarded-allow-ips *` and nginx uses $proxy_add_x_forwarded_for, which
    prepends whatever the client sent. So the per-IP bucket is only as
    trustworthy as a value the attacker types. Against an internet-reachable
    admin login with no lockout and no MFA, that made the limit decorative.

    The global counter is the part that cannot be partitioned by anything the
    client controls, so these are the checks that matter.
    """
    from app.auth import AuthService

    a = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=7)
    ip = "1.2.3.4"
    allowed_throughout = True
    for _ in range(a._max_failures):
        allowed_throughout = allowed_throughout and a.check_rate_limit(ip)
        a.record_failed_login(ip)
    check(allowed_throughout, "per-IP: every attempt up to the limit is allowed")
    check(not a.check_rate_limit(ip), "per-IP limit still trips after N failures from one address")
    check(a.check_rate_limit("9.9.9.9"), "a different address is unaffected by another's failures")

    # THE BYPASS: a fresh apparent address for every attempt.
    b = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=7)
    attempts = 0
    for i in range(1000):
        if not b.check_rate_limit(f"10.0.{i // 256}.{i % 256}"):
            break
        b.record_failed_login(f"10.0.{i // 256}.{i % 256}")
        attempts += 1
    check(attempts <= b._max_global_failures + 1,
          f"rotating the client address is capped by the GLOBAL counter "
          f"({attempts} attempts, cap {b._max_global_failures}) — not unlimited")

    # A correct login must not wipe the global evidence.
    b.clear_failed_logins("10.0.0.1")
    check(not b.check_rate_limit("10.0.0.1"),
          "a successful login clears only ITS OWN bucket, never the global counter")

    # The failure map is bounded: keys are otherwise reclaimed only when the
    # SAME key is revisited after expiry, so address rotation was also an
    # unauthenticated, unbounded memory leak on the box.
    c = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=7)
    c._max_global_failures = 10 ** 9   # isolate memory behaviour from the gate
    c._window_s = 0.0                  # every entry is instantly stale
    for i in range(9000):
        c.record_failed_login(f"172.16.{i // 256}.{i % 256}")
    check(len(c._failures) <= c._max_tracked_ips + 1,
          f"the per-IP map stays bounded under address rotation "
          f"({len(c._failures)} entries after 9000 distinct addresses)")



def hashing_checks() -> None:
    h = hash_password("correct horse battery")
    check(h.startswith("pbkdf2_sha256$"), "hash uses pbkdf2_sha256 format")
    check("$200000$" in h or h.split("$")[1] == "200000", "hash uses >=200000 iterations")
    check(verify_password_hash("correct horse battery", h), "hash round-trips for the right password")
    check(not verify_password_hash("wrong", h), "hash rejects the wrong password")
    # Two hashes of the same password differ (random salt).
    check(h != hash_password("correct horse battery"), "per-user random salt -> distinct hashes")
    # Tampered hash body -> reject, never raise.
    algo, iters, salt, digest = h.split("$")
    flipped = digest[:-2] + ("aa" if digest[-2:] != "aa" else "bb")
    tampered = "$".join([algo, iters, salt, flipped])
    check(not verify_password_hash("correct horse battery", tampered), "tampered hash rejected")
    check(not verify_password_hash("x", "not-a-valid-hash"), "malformed stored hash rejected (no crash)")


# ---------------- the matrix ----------------

# (method, path, json?) — every ADMIN-ONLY route. A viewer must get 403 on
# each; an admin must NOT get 401/403.
ADMIN_ROUTES = [
    ("POST", "/api/cameras", {"name": "z1", "friendly_name": "Z", "model": "IP8M-2779EW-AI",
                              "ip": "127.0.0.1", "username": "u", "password": "p"}),
    ("PUT", "/api/cameras/order", {"names": ["front"]}),
    ("PUT", "/api/cameras/front", {"name": "front", "friendly_name": "Front", "model": "IP8M-2779EW-AI",
                                   "ip": "127.0.0.1", "username": "", "password": ""}),
    ("DELETE", "/api/cameras/front", None),
    ("POST", "/api/cameras/front/probe", None),
    ("GET", "/api/cameras/front/settings", None),
    ("PUT", "/api/cameras/front/settings", {"ir_mode": "auto"}),
    ("POST", "/api/cameras/front/light", {"mode": "on"}),
    ("POST", "/api/cameras/front/siren", {}),
    ("POST", "/api/cameras/front/reboot", None),
    ("GET", "/api/settings", None),
    ("PUT", "/api/settings", None),  # body filled from GET at call time
    # PATCH is the WEB UI's primary settings writer (PUT is a full replace that
    # destroys omitted secrets), so it carries the same authority as PUT and
    # needs the same gate proven. An empty patch deep-merges to a no-op, so this
    # asserts the gate without mutating the document.
    ("PATCH", "/api/settings", {}),
    ("GET", "/api/detection/models", None),
    ("POST", "/api/detection/models/dfine_s/download", None),
    ("POST", "/api/detection/models/dfine_s/activate", None),
    ("DELETE", "/api/detection/models/dfine_n", None),
    ("GET", "/api/system/detector", None),
    ("POST", "/api/notifications/test", None),
    ("GET", "/api/users", None),
    ("POST", "/api/users", {"username": "someviewer2", "password": "password12", "role": "viewer"}),
    ("PUT", "/api/users/999999", {"role": "viewer"}),
    ("DELETE", "/api/users/999999", None),
]

# Every ANY-AUTH route (viewer must NOT be blocked). We assert authz passes
# (status not in {401, 403}); external-state routes may legitimately 404/502.
ANY_AUTH_ROUTES = [
    ("GET", "/api/cameras", None),
    ("GET", "/api/cameras/front/snapshot.jpg", None),
    # PTZ is VIEWER-accessible by product decision: aiming a camera is a
    # live-viewing action, not an admin config change (it stores nothing and is
    # bounded by the camera's own limits). Moved out of ADMIN_ONLY deliberately
    # — if this ever moves back, cameras.py's ptz route needs require_admin
    # again and the iOS/web capability-only gates need an isAdmin check.
    ("POST", "/api/cameras/front/ptz", {"action": "stop", "direction": "left"}),
    ("GET", "/api/events", None),
    ("GET", "/api/events?camera=front&limit=5", None),
    ("GET", "/api/events/1", None),
    ("GET", "/api/events/1/snapshot.jpg", None),
    ("GET", "/api/events/1/clip.mp4", None),
    ("GET", "/api/recordings/cameras", None),
    ("GET", "/api/recordings/front/index?date=2026-01-01", None),
    ("GET", "/api/groups", None),
    ("GET", "/api/auth/me", None),
    # APNs (iOS) push registration is any-auth by contract — a viewer's phone
    # gets pushes too (docs/push-architecture.md).
    ("POST", "/api/notifications/apns/register",
     {"device_token": "ab" * 32,
      "key_b64": "A" * 43 + "=",  # base64 of 32 zero bytes
      "environment": "production"}),
    ("GET", "/api/notifications/apns/devices", None),
    ("DELETE", "/api/notifications/apns/register", {"device_token": "ab" * 32}),
    # PushKit VoIP registration (CallKit doorbell ring) is any-auth too — a
    # viewer's phone rings. The contract field is `token`.
    ("POST", "/api/push/voip", {"token": "cd" * 32, "environment": "production"}),
    ("GET", "/api/push/voip/devices", None),
    ("DELETE", "/api/push/voip", {"token": "cd" * 32}),
]


def _do(client: TestClient, method: str, path: str, headers: dict, body):
    return client.request(method, path, headers=headers, json=body)


def viewer_forbidden_on_admin_routes(client: TestClient, viewer_h: dict) -> None:
    for method, path, body in ADMIN_ROUTES:
        resp = _do(client, method, path, viewer_h, body if isinstance(body, dict) else None)
        check(resp.status_code == 403,
              f"viewer -> 403 on {method} {path} (got {resp.status_code})")


def admin_allowed_on_admin_routes(client: TestClient, admin_h: dict) -> None:
    # Use a scratch camera for destructive ops so read checks keep working:
    # we never DELETE 'front' here (that admin route is exercised on 'scratch').
    add_camera(client, admin_h, "scratch")
    settings_body = client.get("/api/settings", headers=admin_h).json()
    for method, path, body in ADMIN_ROUTES:
        p = path
        b = body if isinstance(body, dict) else None
        if method == "DELETE" and path == "/api/cameras/front":
            p = "/api/cameras/scratch"  # don't nuke the read-check camera
        if method == "POST" and path == "/api/cameras" and isinstance(body, dict):
            b = {**body, "name": "z_admin"}  # unique name so it 201s
        if method == "PUT" and path == "/api/settings":
            b = settings_body
        resp = _do(client, method, p, admin_h, b)
        check(resp.status_code not in (401, 403),
              f"admin passes authz on {method} {p} (got {resp.status_code})")


def viewer_allowed_on_any_auth_routes(client: TestClient, viewer_h: dict, admin_h: dict) -> None:
    for method, path, body in ANY_AUTH_ROUTES:
        rv = _do(client, method, path, viewer_h, body)
        check(rv.status_code not in (401, 403),
              f"viewer passes authz on {method} {path} (got {rv.status_code})")
        ra = _do(client, method, path, admin_h, body)
        check(ra.status_code not in (401, 403),
              f"admin passes authz on {method} {path} (got {ra.status_code})")


def dynamic_admin_enumeration(client: TestClient, viewer_h: dict) -> None:
    """Adversarial: walk EVERY registered HTTP route straight off the app (not a
    hand-maintained list) and assert a viewer gets 403 on every route that isn't
    a known any-auth/no-auth path. Catches a newly-added admin route that someone
    forgets to gate — the exact regression this suite exists to prevent."""
    any_auth = {
        ("GET", "/api/system/health"), ("POST", "/api/auth/login"), ("GET", "/api/auth/me"),
        # nginx auth_request target for the go2rtc live-stream gate. It takes NO
        # dependency by design: nginx calls it on a subrequest with no
        # Authorization header, only the ?token= copied from the original query.
        # It is not an authorization hole — it validates that token itself and
        # returns 401 for anonymous, malformed, AND media-scope tokens (asserted
        # explicitly further down). It answers 401 rather than 403 to a viewer,
        # which is why it cannot ride the generic admin sweep.
        ("GET", "/api/auth/verify-stream"),
        # The rclone OAuth callback takes NO auth dependency BY DESIGN: the
        # provider redirects a bare browser here with no Authorization header.
        # It is authorized by `state` — 256 unguessable bits, single-use, short
        # TTL — which is minted ONLY by POST /api/integrations/rclone/oauth/start,
        # and that IS admin-gated. So a viewer can never hold a valid state, and
        # without one this answers 400 (unknown/expired), not 403, which is why
        # it cannot ride the generic admin sweep.
        ("GET", "/api/integrations/rclone/oauth/callback"),
        ("GET", "/api/cameras"), ("GET", "/api/cameras/{name}/snapshot.jpg"),
        # PTZ is viewer-accessible by product decision: aiming a camera is a
        # live-viewing action, not an admin config change. (The talk WS is the
        # same decision but is a WebSocket, so it isn't in this HTTP sweep — it
        # has its own close-code assertions further down.)
        ("POST", "/api/cameras/{name}/ptz"),
        # Software Privacy Mode is ADMIN-ONLY on BOTH verbs and is therefore
        # deliberately ABSENT from this allowlist — the sweep below asserts a
        # viewer gets 403 on GET and POST /api/privacy. A viewer still renders
        # the "Privacy Mode" overlay from the per-camera `private` flag on
        # GET /api/cameras, which needs no access to the configuration.
        ("GET", "/api/events"), ("GET", "/api/events/{event_id}"),
        ("GET", "/api/events/{event_id}/snapshot.jpg"), ("GET", "/api/events/{event_id}/clip.mp4"),
        # DELETE /api/events/{event_id} stays ADMIN-ONLY (viewers are view-only —
        # hard-deleting shared events/recordings is a management action).
        ("GET", "/api/recordings/cameras"), ("GET", "/api/recordings/{camera}/index"),
        ("GET", "/api/recordings/{camera}/playlist.m3u8"),
        ("GET", "/api/recordings/{camera}/seg/{start_ts}.ts"),
        ("GET", "/api/recordings/{camera}/export.mp4"),
        # Groups: READING is any-auth (a viewer needs the tabs to navigate the
        # cameras they may watch). The three MUTATING verbs are deliberately
        # absent — which cameras are grouped, and under what name, is shared
        # configuration, and a viewer is view-only. The sweep below asserts 403
        # on POST/PUT/DELETE.
        ("GET", "/api/groups"),
        ("GET", "/api/notifications/vapid-public-key"), ("POST", "/api/notifications/subscribe"),
        ("POST", "/api/notifications/unsubscribe"), ("POST", "/api/users/me/password"),
        # APNs push registration: any-auth (viewer phones get pushes too).
        ("POST", "/api/notifications/apns/register"),
        ("DELETE", "/api/notifications/apns/register"),
        ("GET", "/api/notifications/apns/devices"),
        # PushKit VoIP registration for the CallKit doorbell ring: any-auth
        # (a viewer's phone rings too).
        ("POST", "/api/push/voip"),
        ("DELETE", "/api/push/voip"),
        ("GET", "/api/push/voip/devices"),
    }

    def concrete(path: str) -> str:
        # Every path param MUST be listed. An unsubstituted "{param}" reaches
        # the route as a literal, fails type coercion, and returns 422 — which
        # is neither 403 nor a pass, so the route silently escapes this check.
        for token, val in (("{name}", "front"), ("{event_id}", "1"), ("{camera}", "front"),
                           ("{start_ts}", "1"), ("{group_id}", "1"), ("{user_id}", "999999"),
                           ("{key}", "dfine_s"), ("{suppression_id}", "1")):
            path = path.replace(token, val)
        return path

    def http_routes():
        # This Starlette wraps app.include_router() results in an _IncludedRouter
        # whose real routes live on `.original_router`; descend into it.
        for r in app.routes:
            orig = getattr(r, "original_router", None)
            for rt in (orig.routes if orig is not None else [r]):
                methods, path = getattr(rt, "methods", None), getattr(rt, "path", None)
                if methods and path:
                    yield methods, path

    seen: set[tuple[str, str]] = set()
    tested = 0
    for methods, path in http_routes():
        for m in sorted(set(methods) - {"HEAD", "OPTIONS"}):
            key = (m, path)
            if key in seen or key in any_auth:
                seen.add(key)
                continue
            seen.add(key)
            url = concrete(path)
            # A leftover "{param}" would 422 on type coercion before auth runs,
            # so the route would never actually be attacked and the miss would
            # look like an ordinary failure. Fail loudly on the real cause.
            check("{" not in url,
                  f"[dynamic] concrete() substitutes every path param in {path} "
                  f"(unhandled param -> 422 -> route escapes the RBAC sweep)")
            resp = client.request(m, url, headers=viewer_h,
                                  json={} if m in ("POST", "PUT") else None)
            tested += 1
            check(resp.status_code == 403,
                  f"[dynamic] viewer -> 403 on admin {m} {path} (got {resp.status_code})")
    check(tested >= 20, f"dynamic enumeration attacked {tested} admin routes (>=20)")


def token_integrity_attacks(client: TestClient, viewer_token: str, viewer_h: dict) -> None:
    """Role must come ONLY from the signed JWT — never a header/body — and forged
    signatures must be rejected."""
    now = int(time.time())
    # Forge role=admin with the WRONG secret -> signature check fails -> 401.
    forged = jwt.encode({"sub": "viewer1", "role": "admin", "iat": now, "exp": now + 3600},
                        "not-the-real-secret", algorithm="HS256")
    check(client.get("/api/settings", headers=bearer(forged)).status_code == 401,
          "forged admin token (wrong secret) rejected on an admin route")
    # alg=none downgrade -> rejected (decode pins HS256).
    none_tok = jwt.encode({"sub": "viewer1", "role": "admin"}, "", algorithm="none")
    check(client.get("/api/settings", headers=bearer(none_tok)).status_code == 401,
          "alg=none forged token rejected")
    # A genuine viewer token + spoofed role header must still be 403.
    check(client.get("/api/settings",
                     headers={**viewer_h, "X-Role": "admin", "Role": "admin"}).status_code == 403,
          "spoofed role header ignored (viewer still 403)")
    # role in the request body must not influence authorization.
    check(client.put("/api/settings", headers=viewer_h, json={"role": "admin"}).status_code == 403,
          "role in request body ignored (viewer still 403)")
    # A viewer cannot promote self through the users API.
    users = client.get("/api/users", headers=bearer(client.app.state.auth.create_session_token("admin", "admin"))).json()
    vid = next((u["id"] for u in users if u["username"] == "viewer1"), None)
    if vid is not None:
        check(client.put(f"/api/users/{vid}", headers=viewer_h, json={"role": "admin"}).status_code == 403,
              "viewer cannot promote self via PUT /api/users/{id}")


def viewer_grouping_and_push(client: TestClient, viewer_h: dict, admin_h: dict) -> None:
    """Groups are READ-ONLY for a viewer; push is theirs to toggle.

    Groups used to be viewer-writable on the reasoning that they are a
    navigation convenience. They are not: they are SHARED configuration, so one
    viewer renaming or deleting a group changes what every other account sees.
    A viewer is scoped to live view, talk, PTZ, events and the timeline.
    """
    check(client.post("/api/groups", headers=viewer_h,
                      json={"name": "ViewerGroup", "cameras": ["front"]}).status_code == 403,
          "viewer CANNOT create a camera group (shared config, not navigation)")
    # Make one as admin so the viewer's edit/delete attempts hit a REAL group —
    # a 403 against a nonexistent id would pass even if the guard were missing.
    resp = client.post("/api/groups", headers=admin_h,
                       json={"name": "AdminGroup", "cameras": ["front"]})
    check(resp.status_code == 201, "admin can create a camera group")
    gid = resp.json()["id"]
    check(client.put(f"/api/groups/{gid}", headers=viewer_h,
                     json={"name": "Renamed"}).status_code == 403,
          "viewer CANNOT rename an existing group")
    check(client.delete(f"/api/groups/{gid}", headers=viewer_h).status_code == 403,
          "viewer CANNOT delete an existing group")
    check(client.get("/api/groups", headers=viewer_h).status_code == 200,
          "but a viewer still READS groups — they need the tabs to navigate")
    check(client.delete(f"/api/groups/{gid}", headers=admin_h).status_code == 204,
          "admin can delete it again")
    sub = {"endpoint": "https://push.example/vw", "keys": {"p256dh": "k", "auth": "a"}}
    check(client.post("/api/notifications/subscribe", headers=viewer_h, json=sub).status_code == 204,
          "viewer can enable push (subscribe)")
    check(client.post("/api/notifications/unsubscribe", headers=viewer_h,
                      json={"endpoint": sub["endpoint"]}).status_code == 204,
          "viewer can disable push (unsubscribe)")


def privacy_mode_is_admin_only(client: TestClient, viewer_h: dict, admin_h: dict) -> None:
    """Software Privacy Mode is an ADMIN control end to end.

    Covers the two ways a viewer could otherwise reach it: the /api/privacy
    endpoint itself, and the INDIRECT path through camera groups (a privacy
    selection can name a group, and group membership feeds the resolved private
    set) — the escalation being that a viewer edits/deletes a privacy-selected
    group and cameras an admin switched off start recording again.
    """
    # ---- the endpoint: admin-only on BOTH verbs ----
    check(client.get("/api/privacy", headers=viewer_h).status_code == 403,
          "viewer is REJECTED from GET /api/privacy (cannot enumerate the config)")
    check(client.post("/api/privacy", headers=viewer_h, json={"cameras": []}).status_code == 403,
          "viewer is REJECTED from POST /api/privacy (cannot toggle capture)")
    check(client.get("/api/privacy", headers=admin_h).status_code == 200,
          "admin can read the privacy config")

    # ---- the viewer overlay still works WITHOUT that endpoint ----
    cams = client.get("/api/cameras", headers=viewer_h)
    check(cams.status_code == 200, "viewer can still list cameras")
    check(all("private" in c for c in cams.json()),
          "viewer still receives the per-camera `private` flag (renders the overlay)")

    # ---- camera RTSP overrides carry the CAMERA ADMIN PASSWORD ----
    # An override only works against an Amcrest/Dahua unit if it embeds
    # credentials (rtsp://admin:<password>@…). This router is require_auth, not
    # require_admin, so without redaction any viewer could read them from here
    # and log into the camera's own web UI — PTZ, firmware, factory reset —
    # entirely outside this system's RBAC and Privacy Mode.
    secret_url = "rtsp://admin:hunter2@192.168.1.87/cam/realmonitor?channel=1&subtype=0"
    # PUT /api/cameras/{name} is a FULL replace (CameraUpdate requires
    # name/friendly_name/model/ip/...), so merge onto the current row rather
    # than sending a partial body.
    current = next(c for c in client.get("/api/cameras", headers=admin_h).json()
                   if c["name"] == "front")
    body = {k: current[k] for k in
            ("name", "friendly_name", "model", "ip", "username", "password")
            if k in current}
    body.setdefault("username", "u")
    body.setdefault("password", "p")
    body["main_url"] = secret_url
    upd = client.put("/api/cameras/front", headers=admin_h, json=body)
    check(upd.status_code == 200, "admin set an RTSP override carrying a password")
    check(upd.json().get("main_url") == secret_url,
          "the ADMIN who set it reads it back in full (the settings form must round-trip)")

    viewer_cams = client.get("/api/cameras", headers=viewer_h).json()
    front = next(c for c in viewer_cams if c["name"] == "front")
    check(front.get("main_url") == "",
          "a VIEWER gets main_url redacted — the camera password never leaves the backend")
    check("hunter2" not in json.dumps(viewer_cams),
          "and the password appears NOWHERE in the viewer's camera payload")
    check(front.get("main_url_set") is True,
          "...but `main_url_set` still tells the UI an override is configured")

    admin_cams = client.get("/api/cameras", headers=admin_h).json()
    admin_front = next(c for c in admin_cams if c["name"] == "front")
    check(admin_front.get("main_url") == secret_url,
          "an ADMIN listing cameras still sees the override (settings page edits it here)")

    # ---- the INDIRECT path: a privacy-selected group is admin-locked ----
    g = client.post("/api/groups", headers=admin_h,
                    json={"name": "PrivacyLinked", "cameras": ["front"]})
    check(g.status_code == 201, "admin created a group to put into Privacy Mode")
    gid = g.json()["id"]
    check(client.post("/api/privacy", headers=admin_h, json={"groups": [gid]}).status_code == 200,
          "admin put the group into Privacy Mode")
    check("front" in client.get("/api/privacy", headers=admin_h).json()["private_cameras"],
          "the group's members resolved into the private set")

    check(client.put(f"/api/groups/{gid}", headers=viewer_h,
                     json={"cameras": []}).status_code == 403,
          "viewer CANNOT empty a privacy-selected group (would resume capture)")
    check(client.put(f"/api/groups/{gid}", headers=viewer_h,
                     json={"cameras": ["front", "back"]}).status_code == 403,
          "viewer CANNOT add cameras to a privacy-selected group (would blind them)")
    check(client.delete(f"/api/groups/{gid}", headers=viewer_h).status_code == 403,
          "viewer CANNOT delete a privacy-selected group (would resume capture)")
    check("front" in client.get("/api/privacy", headers=admin_h).json()["private_cameras"],
          "after all three viewer attempts the camera is STILL private")

    # ---- an UNRELATED group is now refused too, but for a DIFFERENT reason ----
    # This block used to prove the privacy guard was narrowly scoped: a group
    # Privacy Mode did not reference stayed viewer-writable. Groups are now
    # admin-only outright, so the outcome is the same 403 either way. The check
    # is kept because the two guards are still independent — if the blanket
    # admin gate were ever relaxed, the privacy-specific guard above must still
    # hold on its own, and this is what would show the difference.
    other = client.post("/api/groups", headers=viewer_h, json={"name": "NotPrivate"})
    check(other.status_code == 403,
          "viewer cannot create even a group Privacy Mode does not use "
          "(groups are admin-only; the privacy guard above is a second, "
          "independent layer)")
    oid = client.post("/api/groups", headers=admin_h,
                      json={"name": "NotPrivate"}).json()["id"]
    check(client.put(f"/api/groups/{oid}", headers=viewer_h,
                     json={"name": "NotPrivate2"}).status_code == 403,
          "nor edit one")
    check(client.delete(f"/api/groups/{oid}", headers=viewer_h).status_code == 403,
          "nor delete one")
    check(client.delete(f"/api/groups/{oid}", headers=admin_h).status_code == 204,
          "admin cleans it up")

    # ---- admin retains full control, and clean up ----
    check(client.delete(f"/api/groups/{gid}", headers=admin_h).status_code == 204,
          "admin CAN delete a privacy-selected group")
    check(client.post("/api/privacy", headers=admin_h,
                      json={"cameras": [], "groups": []}).status_code == 200,
          "admin cleared Privacy Mode")


def main() -> None:
    rate_limit_checks()
    hashing_checks()

    with TestClient(app) as client:
        auth = client.app.state.auth

        # ---- built-in admin login ----
        r = login(client, "admin", "admin-secret")
        check(r.status_code == 200, "env-admin login works")
        body = r.json()
        check(body.get("role") == "admin" and body.get("username") == "admin",
              "login response carries role=admin, username=admin")
        admin_token = body["token"]
        admin_h = bearer(admin_token)
        # legacy single-password clients (no username) still log in as admin.
        r2 = client.post("/api/auth/login", json={"password": "admin-secret"})
        check(r2.status_code == 200 and r2.json()["role"] == "admin",
              "username-less login defaults to the built-in admin")
        check(login(client, "admin", "wrong").status_code == 401, "wrong admin password -> 401")

        me = client.get("/api/auth/me", headers=admin_h)
        check(me.status_code == 200 and me.json() == {"username": "admin", "role": "admin"},
              "GET /api/auth/me for admin")

        # ---- user management (create viewer) ----
        r = client.post("/api/users", headers=admin_h,
                        json={"username": "viewer1", "password": "viewerpass1", "role": "viewer"})
        check(r.status_code == 201 and r.json()["role"] == "viewer", "admin creates a viewer user")
        check("password_hash" not in r.json() and "password" not in r.json(),
              "create-user response never leaks the hash")
        check(client.get("/api/users", headers=admin_h).status_code == 200, "admin lists users")
        check(all(u["username"] != "admin" for u in client.get("/api/users", headers=admin_h).json()),
              "built-in admin is never a users-table row")

        # reserved / duplicate / validation
        check(client.post("/api/users", headers=admin_h,
                          json={"username": "admin", "password": "password12"}).status_code == 400,
              "reserved username 'admin' -> 400")
        check(client.post("/api/users", headers=admin_h,
                          json={"username": "viewer1", "password": "password12"}).status_code == 409,
              "duplicate username -> 409")
        check(client.post("/api/users", headers=admin_h,
                          json={"username": "shortpw", "password": "x"}).status_code == 422,
              "too-short password rejected (422)")
        check(client.post("/api/users", headers=admin_h,
                          json={"username": "Bad Name!", "password": "password12"}).status_code
              in (400, 422),
              "invalid username charset rejected")

        # ---- viewer login ----
        rv = login(client, "viewer1", "viewerpass1")
        check(rv.status_code == 200 and rv.json()["role"] == "viewer", "viewer login works, role viewer")
        viewer_token = rv.json()["token"]
        viewer_h = bearer(viewer_token)
        check(login(client, "viewer1", "nope").status_code == 401, "viewer wrong password -> 401")
        vme = client.get("/api/auth/me", headers=viewer_h)
        check(vme.status_code == 200 and vme.json() == {"username": "viewer1", "role": "viewer"},
              "GET /api/auth/me for viewer")

        # ---- seed a read camera used across the matrix ----
        add_camera(client, admin_h, "front")

        # ---- THE MATRIX ----
        viewer_allowed_on_any_auth_routes(client, viewer_h, admin_h)
        viewer_grouping_and_push(client, viewer_h, admin_h)
        privacy_mode_is_admin_only(client, viewer_h, admin_h)
        viewer_forbidden_on_admin_routes(client, viewer_h)
        dynamic_admin_enumeration(client, viewer_h)
        token_integrity_attacks(client, viewer_token, viewer_h)
        admin_allowed_on_admin_routes(client, admin_h)

        # ---- no-auth routes ----
        check(client.get("/api/system/health").status_code == 200, "GET /api/system/health is no-auth")
        check(client.get("/api/notifications/vapid-public-key").status_code == 200,
              "GET /api/notifications/vapid-public-key is no-auth")

        # ---- WS gating ----
        with client.websocket_connect(f"/api/ws?token={viewer_token}") as ws:
            check(True, "viewer may open the /api/ws event socket")
            ws.close()
        # Viewers MAY talk (product decision): the role gate no longer rejects
        # them. ASSERT THE CLOSE CODE, not merely "did it disconnect" — the test
        # camera has no speaker, so a viewer who passes the RBAC gate is closed
        # with 4003 ("Camera has no speaker") while a REJECTED viewer would be
        # closed with 1008 (policy). Both raise WebSocketDisconnect, so only the
        # code distinguishes "allowed through" from "rejected"; a bare
        # disconnect check would pass even if viewers were still blocked.
        # Two shapes to handle: a PRE-accept reject (1008) makes
        # websocket_connect() itself raise WebSocketDisconnect, while an
        # accept-then-close (4003) connects fine and surfaces the close as a
        # `{"type": "websocket.close", "code": ...}` MESSAGE from receive().
        # Reading only the exception would report code=None for the accepted
        # case and silently prove nothing.
        def talk_close_code(token: str):
            try:
                with client.websocket_connect(f"/api/cameras/front/talk?token={token}") as ws:
                    msg = ws.receive()
                    if isinstance(msg, dict) and msg.get("type") == "websocket.close":
                        return msg.get("code")
                    return None
            except WebSocketDisconnect as exc:
                return exc.code

        viewer_talk_code = talk_close_code(viewer_token)
        check(viewer_talk_code != 1008,
              f"viewer NOT policy-rejected from the talk WS (close={viewer_talk_code})")
        check(viewer_talk_code == 4003,
              "viewer passes the RBAC gate and reaches the no-speaker capability gate (4003)")

        # MEDIA-scope tokens must STILL be refused: those are the long-lived,
        # widely shared image tokens in notifications/MQTT — they must never
        # open a live mic. This is the half of the old gate that is NOT relaxed.
        media_talk_code = talk_close_code(auth.create_media_token("admin", "admin"))
        check(media_talk_code == 1008,
              f"media-scope token STILL rejected from the talk WS (got {media_talk_code})")

        # ---- last-admin demotion guard ----
        r = client.post("/api/users", headers=admin_h,
                        json={"username": "dbadmin", "password": "password12", "role": "admin"})
        check(r.status_code == 201, "admin creates a second admin (DB)")
        db_admin_id = r.json()["id"]
        # dbadmin is the only DB admin -> demoting it is blocked.
        check(client.put(f"/api/users/{db_admin_id}", headers=admin_h,
                         json={"role": "viewer"}).status_code == 400,
              "cannot demote the last DB admin -> 400")
        # Add another DB admin, THEN the demotion is allowed.
        r2 = client.post("/api/users", headers=admin_h,
                         json={"username": "dbadmin2", "password": "password12", "role": "admin"})
        check(r2.status_code == 201, "admin creates a third admin (DB)")
        check(client.put(f"/api/users/{db_admin_id}", headers=admin_h,
                         json={"role": "viewer"}).status_code == 200,
              "demotion allowed once another admin exists")
        check(client.delete(f"/api/users/{r2.json()['id']}", headers=admin_h).status_code == 204,
              "admin deletes a user")
        check(client.put("/api/users/999999", headers=admin_h,
                         json={"role": "viewer"}).status_code == 404,
              "PUT unknown user id -> 404")

        # ---- self password change (DB user) ----
        check(client.post("/api/users/me/password", headers=viewer_h,
                          json={"current_password": "viewerpass1", "new_password": "newpass123"}).status_code == 204,
              "viewer changes own password")
        check(login(client, "viewer1", "newpass123").status_code == 200, "new password works after change")
        check(client.post("/api/users/me/password", headers=admin_h,
                          json={"current_password": "admin-secret", "new_password": "x-new-pass"}).status_code == 400,
              "built-in admin cannot change password via API (env-controlled)")

        # ---- legacy no-role token treated as admin ----
        secret = auth._secret
        now = int(time.time())
        legacy = jwt.encode({"sub": "admin", "iat": now, "exp": now + 3600}, secret, algorithm="HS256")
        lh = bearer(legacy)
        check(client.get("/api/auth/me", headers=lh).json() == {"username": "admin", "role": "admin"},
              "legacy no-role token resolves to admin at /me")
        check(client.get("/api/settings", headers=lh).status_code == 200,
              "legacy no-role token reaches an admin-only route")

        # ---- media token carries role + scope enforcement ----
        mt = auth.create_media_token("admin", "admin")
        mclaims = auth.decode(mt)
        check(mclaims.get("role") == "admin" and mclaims.get("scope") == "media",
              "media token carries role + media scope")
        check(client.get("/api/cameras", headers=bearer(mt)).status_code == 401,
              "media-scope token rejected on a non-media route")
        check(client.get(f"/api/events/1/snapshot.jpg?token={mt}").status_code in (200, 404),
              "media-scope token accepted on a media route (authz passes)")

        # ---- media tokens must DEFAULT to the lowest role ----
        # These are minted with NO ARGS by events_pipeline._media_url and
        # mqtt_ha._snapshot_url, then embedded in notification URLs and
        # published RETAINED to the operator's MQTT broker — i.e. they are
        # built to leak, and last media_token_days (7). The default was once
        # role="admin", which quietly turned every notification into an
        # admin-role credential and let a leaked one through
        # require_media_admin. Nothing needs an admin media token: the admin UI
        # authenticates with the operator's SESSION token instead.
        default_mt = auth.create_media_token()
        default_claims = auth.decode(default_mt)
        check(default_claims.get("role") != "admin",
              "media token DEFAULTS to a non-admin role (notification URLs leak by design)")
        check(role_from_claims(default_claims) != "admin",
              "a default media token does NOT resolve to admin (sub=admin must not promote it)")
        check(client.get(f"/api/detection/suppressions/1/thumb.jpg?token={default_mt}")
              .status_code == 403,
              "a leaked notification media token is REJECTED by an admin media route")
        # ...while the operator's own session token still reaches it, which is
        # how the web (AuthImage: Authorization header) and iOS (mediaURL:
        # session ?token=) actually load that thumbnail.
        check(client.get(f"/api/detection/suppressions/1/thumb.jpg?token={admin_token}")
              .status_code in (200, 404),
              "an admin SESSION token still reaches the admin media route")
        # A viewer session token also works as ?token= on media routes.
        check(client.get(f"/api/cameras/front/snapshot.jpg?token={viewer_token}").status_code
              not in (401, 403),
              "viewer session token accepted as ?token= on a media route")

        # ---- a media token is BOUND to the one object it was minted for ----
        # events_pipeline._media_url / mqtt_ha._snapshot_url mint these per
        # event and ship them OUT — into notification bodies and into RETAINED
        # MQTT messages every current and future subscriber can read. Scope
        # alone said "this is media"; it never said WHICH media, so one token
        # someone glanced at opened every other event's snapshot and every
        # camera's live JPEG for a full 7 days.
        bound = auth.create_media_token(resource="event:1")
        check(auth.decode(bound).get("res") == "event:1",
              "a minted media token carries the `res` claim naming its object")
        check(client.get(f"/api/events/1/snapshot.jpg?token={bound}").status_code in (200, 404),
              "a bound token reaches THE event it was minted for")
        check(client.get(f"/api/events/2/snapshot.jpg?token={bound}").status_code == 404,
              "a bound token CANNOT reach a different event")
        check(client.get(f"/api/cameras/front/snapshot.jpg?token={bound}").status_code == 404,
              "a bound token CANNOT reach a live camera snapshot")
        check(client.get(f"/api/events/1/clip.mp4?token={bound}").status_code in (200, 404),
              "a bound token still covers the same event's clip (same object)")

        # ---- the 24/7 ARCHIVE is off-limits to media tokens entirely ----
        # require_media_auth on the whole /api/recordings router meant a leaked
        # notification token could stream any camera's history and pull 30-minute
        # MP4 exports. It now takes require_stream_auth: `?token=` still works
        # (an HLS playlist cannot send headers) but ONLY as a session token.
        for path in (
            "/api/recordings/cameras",
            "/api/recordings/front/index?date=2026-01-01",
            "/api/recordings/front/playlist.m3u8?start=0&end=60",
            "/api/recordings/front/export.mp4?start=0&end=60",
        ):
            sep = "&" if "?" in path else "?"
            check(client.get(f"{path}{sep}token={bound}").status_code == 401,
                  f"media token REFUSED on {path.split('?')[0]}")
            check(client.get(f"{path}{sep}token={default_mt}").status_code == 401,
                  f"...including an UNBOUND legacy media token on {path.split('?')[0]}")
        # ...while the clients' own session token (what mediaUrl/mediaURL send)
        # still works, so the web timeline and iOS playback are unaffected.
        check(client.get(f"/api/recordings/cameras?token={admin_token}").status_code == 200,
              "a SESSION token as ?token= still reaches the recordings router")
        check(client.get(f"/api/recordings/cameras?token={viewer_token}").status_code == 200,
              "...for a viewer too (recordings are any-auth, just not media-token)")

        # ---- the nginx auth_request gate for LIVE STREAMS ----
        # /go2rtc/ proxied the streaming endpoints with NO authentication, and
        # the site is published at nvr.example.com — so
        # `GET /go2rtc/api/ws?src=<slug>` was live video of that camera, indoor
        # ones included, to anyone on the internet who guessed a name. nginx now
        # runs auth_request against this endpoint before proxying the handshake.
        check(client.get("/api/auth/verify-stream").status_code == 401,
              "verify-stream refuses a request with NO token (the anonymous case)")
        check(client.get("/api/auth/verify-stream?token=garbage").status_code == 401,
              "verify-stream refuses a malformed token")
        check(client.get(f"/api/auth/verify-stream?token={admin_token}").status_code == 204,
              "verify-stream accepts an admin SESSION token")
        check(client.get(f"/api/auth/verify-stream?token={viewer_token}").status_code == 204,
              "verify-stream accepts a viewer SESSION token (live view is any-auth)")
        # A media token must NOT open a live stream: those are minted into
        # notification bodies and retained MQTT messages, so accepting one would
        # mean a single push granted permanent live view of every camera.
        check(client.get(f"/api/auth/verify-stream?token={default_mt}").status_code == 401,
              "verify-stream REFUSES a media-scope token (a notification is not a key)")
        check(client.get(f"/api/auth/verify-stream?token={bound}").status_code == 401,
              "...including a resource-bound one")

        # ---- WebSocket auth rides the SUBPROTOCOL, not the query string ----
        # A ?token= WS URL lands verbatim in nginx's ERROR log (log_format does
        # not apply there), which printed a live 30-day admin JWT in cleartext
        # and forced a secret rotation. A browser cannot set an Authorization
        # header on a WS handshake, but subprotocols travel in a header.
        from app.auth import ws_token

        class _FakeWS:
            def __init__(self, offered: str = "") -> None:
                self.headers = {"sec-websocket-protocol": offered} if offered else {}

        tok, sub = ws_token(_FakeWS("bearer, abc.def.ghi"), None)
        check(tok == "abc.def.ghi" and sub == "bearer",
              "subprotocol bearer is extracted and 'bearer' is echoed back")
        tok, sub = ws_token(_FakeWS(), "query-token")
        check(tok == "query-token" and sub is None,
              "?token= still works for an older client, echoing NO subprotocol")
        tok, sub = ws_token(_FakeWS("bearer, hdr"), "query-token")
        check(tok == "hdr", "the subprotocol WINS over a query token when both are sent")
        tok, sub = ws_token(_FakeWS("chat, v2"), None)
        check(tok is None and sub is None,
              "an unrelated subprotocol offer is NOT treated as a token")
        tok, sub = ws_token(_FakeWS("bearer"), None)
        check(tok is None, "a malformed 'bearer' with no token yields nothing")

        # END TO END through the real app: the WS must accept a subprotocol
        # token and REJECT a media-scope one.
        with client.websocket_connect(
            "/api/ws", subprotocols=["bearer", admin_token]
        ) as sock:
            check(True, "GET /api/ws connects with the token in the subprotocol")
            sock.close()
        rejected = False
        try:
            with client.websocket_connect("/api/ws", subprotocols=["bearer", default_mt]):
                pass
        except Exception:  # noqa: BLE001 — starlette raises on a 1008 reject
            rejected = True
        check(rejected, "a media-scope token is REJECTED on the WS subprotocol path")




    print(f"\nALL {PASS} CHECKS PASSED (RBAC authorization matrix)")


if __name__ == "__main__":
    main()
