"""Live-server smoke test for the two NEW route surfaces: camera-health + faces.

Unlike the in-process fastapi.testclient suites in backend/tests/, this script
drives a REAL, RUNNING Vigilume backend over HTTP. It logs in as admin and
checks:

  CAMERA HEALTH  (GET /api/system/camera-health)
    - authenticated GET -> 200
    - the documented top-level shape {window:{since,until,hours}, cameras:[...]}
    - per-camera row field presence + types
      {camera, uptime_pct:number|null, online:bool|null, down_count:int,
       down_seconds:number, downs:[{start,end,seconds}]}  (downs capped at 20)
    - tolerant of an empty / zero-history / no-cameras box (never requires a
      camera to be down; accepts null uptime and empty downs[])
    - auth gate: the same GET with NO credential -> 401

  FACES  (/api/faces ...)
    - GET /api/faces -> 200 + array
    - create a throwaway person named '__smoke_test_person__' -> 200 + int id
      (these APIs give no random-suffix affordance, so BEFORE creating we list
       and delete any '__smoke_test_person__' left behind by a crashed prior run)
    - GET that person's thumb.jpg BEFORE any capture -> a clean 404 (not a 500)
    - live-capture route EXISTENCE probe with a deliberately bogus camera name
      -> a handled status, specifically NOT a 404 route-missing and NOT a 500.
         The capture handler (_go2rtc_frame) grabs a go2rtc frame BEFORE any
         enrol step, so a bogus camera fails there: go2rtc UP returns a fast
         non-200 for the unknown src -> 502 ("Could not grab a camera frame");
         go2rtc DOWN/slow (past its ~6s timeout) raises an httpx transport error
         that the handler now maps to a clean 503 ("Camera streaming service
         unreachable"). Either way it is a handled 502/503/4xx, NEVER an uncaught
         500 — so this probe asserts `!= 500` UNCONDITIONALLY. It still reads
         GET /api/system/health first, but only to print which handled status to
         expect (502 when go2rtc is up, 503 when it is down).
    - DELETE the throwaway person -> 204, then re-list and confirm it is gone
    - auth gate: GET /api/faces with NO credential -> 401

NON-DESTRUCTIVE. The ONLY mutation this script performs is creating, then
deleting, its own '__smoke_test_person__'. It never touches, mutates, or deletes
any real person; it never enrols a real face (the capture probe uses a bogus
camera and is expected to fail); and it never calls capture-event against a real
event. The create/delete pair runs inside try/finally so the throwaway person is
ALWAYS removed, even if an assertion fails midway.

Usage (on the box or anywhere on the LAN):
    export ADMIN_PASSWORD='...'          # required; read from env, never printed
    python3 faces_camera_health_smoke.py
    python3 faces_camera_health_smoke.py http://192.168.1.253:8080
    VIGILUME_BASE_URL=https://192.168.1.253:8443 python3 faces_camera_health_smoke.py

Environment variables:
    ADMIN_PASSWORD    (required) admin password; read from env, never printed
    ADMIN_USERNAME    (optional) admin username, default 'admin'
    VIGILUME_BASE_URL (optional) base URL, used when argv[1] is not given

Base-URL precedence: argv[1] > $VIGILUME_BASE_URL > http://192.168.1.253:8080
(the always-on nginx LAN endpoint). The 8443 base is Caddy 'tls internal'
(self-signed), so any https base is contacted with TLS verification disabled.

CAUTION — verify ADMIN_PASSWORD before rapid reruns. The backend's login
limiter keeps a WHOLE-BACKEND failure counter (100 failures / 5 min) that
CANNOT be partitioned by IP. A CORRECT password has zero effect (success
records no failure and clears the per-IP bucket), but each run with a WRONG
password records one failure against that shared counter — so hammering this
script with a bad password could, in the extreme, contribute toward a global
lockout that also blocks legitimate logins for up to five minutes. This script
already exits immediately on the first 401 from login; if that happens,
double-check ADMIN_PASSWORD rather than re-running blindly.

Requires: httpx (already a backend dependency) and a reachable live backend.
"""

from __future__ import annotations

import os
import sys
import warnings

import httpx

# --- constants -------------------------------------------------------------
DEFAULT_BASE_URL = "http://192.168.1.253:8080"  # always-on nginx LAN endpoint
SMOKE_NAME = "__smoke_test_person__"            # fixed (no random suffix) on purpose
BOGUS_CAMERA = "__smoke_test_bogus_camera__"    # a src go2rtc will not know

PASS = 0


def check(cond: bool, msg: str) -> None:
    """House-style fail-fast check. No local DB / async resources are held over
    the network, so a plain SystemExit(1) on the first failure is safe here —
    the try/finally around the faces create/delete still removes the throwaway
    person before the exception propagates out."""
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {msg}")
    else:
        print(f"FAIL: {msg}")
        raise SystemExit(1)


def is_num(x) -> bool:
    # JSON numbers arrive as int or float; bool is a subclass of int, exclude it.
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_num_or_none(x) -> bool:
    return x is None or is_num(x)


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _go2rtc_healthy(client: httpx.Client) -> bool:
    """Best-effort read of GET /api/system/health -> its `go2rtc` boolean.

    Used only to annotate the capture probe: _go2rtc_frame() now maps a go2rtc
    transport error to a clean 503, so the probe no longer needs this to stay
    lenient about a 500 — it just prints which handled status to expect (502
    when go2rtc is up, 503 when it is down). This endpoint needs no auth and
    returns 200 as soon as the app is up. Returns True ONLY when go2rtc is
    explicitly reported healthy; any error, non-200, or absent/false field
    yields False."""
    try:
        r = client.get("/api/system/health")
    except httpx.HTTPError:
        return False
    if r.status_code != 200:
        return False
    try:
        return (r.json() or {}).get("go2rtc") is True
    except ValueError:  # non-JSON body
        return False


# --- auth ------------------------------------------------------------------
def login(client: httpx.Client, username: str, password: str) -> dict:
    """POST /api/auth/login {username,password} -> {token,role,username}.
    Returns the Authorization header dict. Never prints the password or token."""
    try:
        r = client.post("/api/auth/login",
                        json={"username": username, "password": password})
    except httpx.HTTPError as exc:
        print(f"FAIL: could not reach {client.base_url} to log in: {exc!r}\n"
              f"      Is the backend up and the base URL correct?")
        raise SystemExit(1)
    if r.status_code != 200:
        print(f"FAIL: login returned {r.status_code} (expected 200). "
              f"Check the base URL, that the backend is running, and that "
              f"ADMIN_PASSWORD is correct. (Response body not printed — it may "
              f"echo input.)")
        raise SystemExit(1)
    token = (r.json() or {}).get("token")
    if not token:
        print("FAIL: login succeeded but the response had no 'token' field.")
        raise SystemExit(1)
    print("  ok: logged in as admin (token withheld from output)")
    return {"Authorization": f"Bearer {token}"}


# --- camera-health ---------------------------------------------------------
def camera_health_checks(client: httpx.Client, auth: dict) -> None:
    print("\ncamera-health: GET /api/system/camera-health")
    r = client.get("/api/system/camera-health", params={"hours": 24}, headers=auth)
    check(r.status_code == 200, f"authenticated GET returns 200 (got {r.status_code})")

    body = r.json()
    check(isinstance(body, dict), "response body is a JSON object")

    win = body.get("window")
    check(isinstance(win, dict), "body.window is an object")
    check(is_num(win.get("since")), "window.since is a number")
    check(is_num(win.get("until")), "window.until is a number")
    check(is_num(win.get("hours")), "window.hours is a number")

    cams = body.get("cameras")
    check(isinstance(cams, list), "body.cameras is a list")

    # Tolerant of a zero-history / freshly-booted box: a present-but-empty
    # history still yields one row per configured camera with uptime_pct=null,
    # down_count=0, down_seconds=0.0, downs=[]. No camera is required to be down,
    # and cameras[] may legitimately be empty if none are configured.
    for row in cams:
        check(isinstance(row, dict), "each cameras[] entry is an object")
        tag = row.get("camera")
        check(isinstance(tag, str), f"row.camera is a string ({tag!r})")
        check(is_num_or_none(row.get("uptime_pct")),
              f"row.uptime_pct is number|null ({tag})")
        online = row.get("online")
        check(online is None or isinstance(online, bool),
              f"row.online is bool|null ({tag})")
        check(is_int(row.get("down_count")), f"row.down_count is an int ({tag})")
        check(is_num(row.get("down_seconds")), f"row.down_seconds is a number ({tag})")
        downs = row.get("downs")
        check(isinstance(downs, list), f"row.downs is a list ({tag})")
        check(len(downs) <= 20, f"row.downs is capped at 20 entries ({tag})")
        for d in downs:
            check(isinstance(d, dict)
                  and is_num(d.get("start"))
                  and is_num(d.get("end"))
                  and is_num(d.get("seconds")),
                  f"each downs[] window has numeric start/end/seconds ({tag})")

    if not cams:
        print("  note: cameras[] is empty (no cameras configured) — shape still valid")

    # Auth gate: no Authorization header -> 401.
    r = client.get("/api/system/camera-health", params={"hours": 24})
    check(r.status_code == 401,
          f"unauthenticated camera-health GET is 401 (got {r.status_code})")


# --- faces -----------------------------------------------------------------
def faces_checks(client: httpx.Client, auth: dict) -> None:
    print("\nfaces: GET /api/faces (+ non-destructive create/probe/delete)")
    r = client.get("/api/faces", headers=auth)
    check(r.status_code == 200, f"authenticated list returns 200 (got {r.status_code})")
    persons = r.json()
    check(isinstance(persons, list), "GET /api/faces returns a JSON array")

    # The name is fixed (no random/time suffix, since these APIs offer no
    # affordance for one), so a crash mid-run could leave a stale copy behind.
    # Clean it up FIRST so the create below is deterministic.
    for p in persons:
        if isinstance(p, dict) and p.get("name") == SMOKE_NAME:
            pid = p.get("id")
            dr = client.delete(f"/api/faces/{pid}", headers=auth)
            check(dr.status_code == 204,
                  f"removed a leftover {SMOKE_NAME} (id={pid}) from a prior run")

    person_id = None
    try:
        # --- create the throwaway person ---
        r = client.post("/api/faces", json={"name": SMOKE_NAME}, headers=auth)
        check(r.status_code == 200,
              f"create {SMOKE_NAME} returns 200 (got {r.status_code})")
        created = r.json()
        person_id = created.get("id")
        check(is_int(person_id), f"create returns an integer id (got {person_id!r})")
        check(created.get("name") == SMOKE_NAME, "create echoes the submitted name")

        # --- thumb BEFORE any capture -> a clean 404 (a person with no
        #     embeddings has no thumbnail). It must NOT 500. ---
        r = client.get(f"/api/faces/{person_id}/thumb.jpg", headers=auth)
        check(r.status_code != 500,
              "thumb.jpg does not 500 for a person with no thumbnail")
        check(r.status_code == 404,
              f"thumb.jpg is a clean 404 before any capture (got {r.status_code})")

        # --- live-capture route EXISTENCE probe (NO real enrolment) ---
        # We pass a bogus ?camera=. How to tell "route missing" from "bad
        # camera": our person DOES exist, so the handler's only 404
        # ("Person not found") cannot fire — therefore ANY 404 here means the
        # ROUTE itself is unregistered (FastAPI's plain "Not Found"), i.e. the
        # surface is missing = FAIL. If the route matched and the handler ran,
        # it fails on the bogus camera at the frame-grab step (which runs BEFORE
        # the face-availability/enrol steps), yielding a HANDLED status:
        #   * go2rtc UP   -> fast non-200 for the unknown src -> 502
        #                    ("Could not grab a camera frame")
        #   * go2rtc DOWN -> httpx transport error, which _go2rtc_frame() maps
        #                    to 503 ("Camera streaming service unreachable")
        # Either way it is NEVER an uncaught 500, so the `!= 500` assertion is
        # UNCONDITIONAL. The health pre-read is now purely informational: it
        # tells the human which handled status to expect.
        go2rtc_up = _go2rtc_healthy(client)
        r = client.post(f"/api/faces/{person_id}/capture",
                        params={"camera": BOGUS_CAMERA}, headers=auth)
        sc = r.status_code
        print(f"  note: /api/system/health reports go2rtc "
              f"{'up (expect 502)' if go2rtc_up else 'down (expect 503)'}; "
              f"capture returned {sc}.")
        check(sc != 404,
              f"capture route is REGISTERED — not a 404 route-missing (got {sc})")
        check(sc != 500,
              f"capture never surfaces an uncaught 500 — a go2rtc outage now "
              f"maps to a clean 503 (got {sc})")
        check(sc == 502 or sc == 503 or (400 <= sc < 500),
              f"capture returns a handled 502/503/4xx for a bogus camera (got {sc})")

    finally:
        # ALWAYS remove the throwaway person, even if an assertion above raised.
        # Guard on is_int (not merely `is not None`): create_person always
        # returns an int id, but if a malformed body ever gave a non-int truthy
        # value we must NOT issue DELETE /api/faces/<non-int>. Also: if we are
        # already unwinding a primary failure (sys.exc_info() is set — check()
        # raises SystemExit, a BaseException that populates exc_info in finally),
        # do cleanup best-effort so a cleanup hiccup can't MASK that primary
        # failure. Only when we reached the finally cleanly do we assert on the
        # delete + re-list as first-class checks.
        if is_int(person_id):
            in_teardown = sys.exc_info()[0] is not None
            if in_teardown:
                try:
                    dr = client.delete(f"/api/faces/{person_id}", headers=auth)
                    if dr.status_code == 204:
                        print(f"  cleanup: removed {SMOKE_NAME} (id={person_id}) "
                              f"after an earlier failure")
                    else:
                        print(f"  cleanup WARN: DELETE {SMOKE_NAME} "
                              f"(id={person_id}) returned {dr.status_code}; the "
                              f"failure reported above still stands — remove it "
                              f"manually if it persists")
                except httpx.HTTPError as exc:
                    print(f"  cleanup WARN: could not DELETE {SMOKE_NAME} "
                          f"(id={person_id}): {exc!r}; the failure reported above "
                          f"still stands — remove it manually if it persists")
            else:
                r = client.delete(f"/api/faces/{person_id}", headers=auth)
                check(r.status_code == 204,
                      f"DELETE {SMOKE_NAME} (id={person_id}) returns 204")
                r = client.get("/api/faces", headers=auth)
                check(r.status_code == 200, "re-list after delete returns 200")
                names = [p.get("name") for p in r.json() if isinstance(p, dict)]
                check(SMOKE_NAME not in names, f"{SMOKE_NAME} is gone after delete")

    # Auth gate: no Authorization header -> 401.
    r = client.get("/api/faces")
    check(r.status_code == 401,
          f"unauthenticated GET /api/faces is 401 (got {r.status_code})")


# --- entrypoint ------------------------------------------------------------
def pick_base_url() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    env = os.environ.get("VIGILUME_BASE_URL", "").strip()
    if env:
        return env
    return DEFAULT_BASE_URL


def main() -> None:
    base = pick_base_url().rstrip("/")
    print(f"Base URL in use: {base}")

    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        print("ERROR: ADMIN_PASSWORD is not set.\n"
              "       Set it to the admin password and re-run, e.g.:\n"
              "           export ADMIN_PASSWORD='...'\n"
              "       (The value is read from the environment and never printed.)")
        raise SystemExit(2)
    username = os.environ.get("ADMIN_USERNAME", "admin")

    # https bases are Caddy 'tls internal' (self-signed) -> skip verification.
    # (httpx does not emit an unverified-HTTPS warning the way requests/urllib3
    # does, but we filter that message defensively so nothing is printed.)
    verify = not base.lower().startswith("https")
    if not verify:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    with httpx.Client(base_url=base, verify=verify, timeout=15.0) as client:
        auth = login(client, username, password)
        camera_health_checks(client, auth)
        faces_checks(client, auth)

    print(f"\nAll {PASS} checks passed.")


if __name__ == "__main__":
    main()
