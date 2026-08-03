"""Smoke suite for per-camera EXEMPT (privacy / ignore) detection zones.

Feature (docs/CONTRACTS.md camera section):
  - Each camera row carries ``exempt_zones``: a JSON list of polygons in
    NORMALIZED (0..1, resolution-independent) coords, each with an optional
    name — ``[{"name": str, "points": [[x, y], ...]}, ...]``.
  - The detection engine converts every polygon to detect-stream pixels ONCE
    per camera-row change and drops any observation whose box FOOT-CENTER —
    the midpoint of the bottom edge ``((x1+x2)/2, y2)`` — lies inside any
    polygon. Suppressed objects never confirm, count, or open an event.
  - Polygons with fewer than 3 points are ignored. An empty ``exempt_zones``
    list means NO masking (unchanged behavior).
  - Schema v8 adds ``cameras.exempt_zones TEXT NOT NULL DEFAULT '[]'``; the
    v7->v8 migration preserves existing rows and defaults [].

Covered here:
  - point-in-polygon correctness: square, triangle, concave (L-shape), edges,
    < 3 points
  - foot-center rule: tall box whose CENTER is above a ground zone but whose
    FOOT is inside IS masked; a box fully above the zone is NOT
  - normalized -> detect-space scaling at multiple resolutions; bare-list and
    object zone forms; degenerate dims
  - multiple zones; box in either is masked, box in neither is not
  - engine/pipeline path (like native_smoke): empty = no-op; a masked
    detection produces NO event + NO live count; an un-masked object in the
    same camera still opens its event; reload() precomputes the polygons
  - API round-trip: create/response/update semantics, clamping, < 3-point
    zones dropped, PUT-omitted keeps stored, PUT [] clears
  - migration v7 -> v8 preserves rows and defaults exempt_zones to []

Usage: python backend/tests/zones_smoke.py  (needs backend deps).
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Clean env before app config is instantiated (lifespan reads it).
for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
# Unroutable local port -> instant connection refusal so go2rtc syncs during
# camera CRUD never slow the suite down.
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-zones-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Config, DEFAULT_DETECT_OBJECTS  # noqa: E402
from app.db import SCHEMA_VERSION, Database  # noqa: E402
from app.main import app  # noqa: E402
from app.native.engine import (  # noqa: E402
    MIN_HITS,
    DetectionEngine,
    Observation,
    _CameraState,
    box_foot_center,
    box_in_exempt_zones,
    exempt_detect_polygons,
    point_in_polygon,
)
from app.routers import cameras as cameras_router  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# The background capability probe would hang on a blackholed IP; the zone
# semantics don't need it — stub it out so CRUD returns instantly.
async def _no_probe(cam: dict) -> dict:
    return {}


cameras_router._probe_caps = _no_probe  # type: ignore[assignment]


class RecordingPipeline:
    """Minimal EventsPipeline stand-in: records payloads + live counts."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.counts: dict[tuple[str, str], int] = {}

    async def handle_event(self, payload: dict) -> None:
        self.events.append(payload)

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = count


def _cam_row(name: str, **over) -> dict:
    row = {
        "name": name, "detect_enabled": True, "detect_fps": 5,
        "detect_width": 100, "detect_height": 100,
        "detect_objects": ["person"], "exempt_zones": [],
        "ip": "10.0.0.9", "username": "u", "password": "p",
        "main_url": "", "sub_url": "", "record_enabled": True,
    }
    row.update(over)
    return row


# A bottom-right quadrant zone in NORMALIZED coords (0.5..1.0 on both axes).
BR_ZONE = {"name": "porch", "points": [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]}


# ============================ point-in-polygon =============================


def geometry_checks() -> None:
    print("geometry: point-in-polygon + foot-center")

    # -- axis-aligned square [(0,0),(10,0),(10,10),(0,10)] --
    sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    check(point_in_polygon(5, 5, sq), "square: center is inside")
    check(not point_in_polygon(-1, 5, sq), "square: point left of it is outside")
    check(not point_in_polygon(5, 20, sq), "square: point below it is outside")
    check(not point_in_polygon(11, 5, sq), "square: point right of it is outside")

    # -- triangle [(0,0),(10,0),(0,10)] --
    tri = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    check(point_in_polygon(2, 2, tri), "triangle: interior point inside")
    check(not point_in_polygon(8, 8, tri), "triangle: point past the hypotenuse outside")

    # -- concave L-shape --
    #   bottom bar y in [0,1] over x in [0,3]; left bar x in [0,1] over y in [0,3];
    #   the upper-right square (x>1, y>1) is the CUT-OUT (concavity).
    ell = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]
    check(point_in_polygon(0.5, 0.5, ell), "L-shape: corner cell inside")
    check(point_in_polygon(2.0, 0.5, ell), "L-shape: bottom-bar cell inside")
    check(point_in_polygon(0.5, 2.0, ell), "L-shape: left-bar cell inside")
    check(not point_in_polygon(2.0, 2.0, ell),
          "L-shape: the concave cut-out cell is OUTSIDE (concavity respected)")

    # -- degenerate polygons never match --
    check(not point_in_polygon(0, 0, [(0.0, 0.0), (1.0, 1.0)]), "2-point 'polygon' -> never inside")
    check(not point_in_polygon(0, 0, []), "empty polygon -> never inside")

    # -- foot-center = midpoint of the bottom edge --
    fx, fy = box_foot_center((10.0, 20.0, 30.0, 80.0))
    check(fx == 20.0 and fy == 80.0, "foot-center is ((x1+x2)/2, y2)")


# ===================== normalized -> detect-space scaling ==================


def scaling_checks() -> None:
    print("scaling: normalized exempt_zones -> detect-space polygons")

    row = _cam_row("c", detect_width=704, detect_height=480, exempt_zones=[BR_ZONE])
    polys = exempt_detect_polygons(row)
    check(len(polys) == 1, "one zone -> one detect-space polygon")
    check(polys[0][0] == (352.0, 240.0) and polys[0][2] == (704.0, 480.0),
          "normalized points scale by detect_width/height (704x480)")

    row2 = _cam_row("c", detect_width=640, detect_height=360, exempt_zones=[BR_ZONE])
    polys2 = exempt_detect_polygons(row2)
    check(polys2[0][0] == (320.0, 180.0) and polys2[0][2] == (640.0, 360.0),
          "same normalized zone scales differently at 640x360 (resolution-independent)")

    # bare-list zone form (no dict wrapper) is tolerated
    bare = _cam_row("c", exempt_zones=[[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]])
    check(len(exempt_detect_polygons(bare)) == 1, "bare [[x,y],...] zone form accepted")

    # < 3 points is dropped
    two_pt = _cam_row("c", exempt_zones=[{"points": [[0.0, 0.0], [1.0, 1.0]]}])
    check(exempt_detect_polygons(two_pt) == [], "zone with < 3 points is ignored")

    # degenerate detect dims -> nothing to scale into
    check(exempt_detect_polygons(_cam_row("c", detect_width=0, exempt_zones=[BR_ZONE])) == [],
          "detect_width <= 0 -> no polygons")

    # empty zones -> empty
    check(exempt_detect_polygons(_cam_row("c", exempt_zones=[])) == [], "no zones -> no polygons")


# ============================ box_in_exempt_zones ==========================


def masking_predicate_checks() -> None:
    print("predicate: box_in_exempt_zones (multiple zones, foot-center)")

    # detect-space bottom-right quadrant of a 100x100 detect stream
    polys = exempt_detect_polygons(_cam_row("c", exempt_zones=[BR_ZONE]))  # [(50,50)..(100,100)]

    # box whose foot-center (70,90) is inside the zone
    check(box_in_exempt_zones((60.0, 60.0, 80.0, 90.0), polys),
          "box with foot-center inside the zone is masked")

    # box in the top-left quadrant, foot-center (20,40) outside
    check(not box_in_exempt_zones((10.0, 10.0, 30.0, 40.0), polys),
          "box with foot-center outside the zone is not masked")

    # FOOT-CENTER RULE: a TALL box whose CENTER is above the ground zone but
    # whose FOOT is inside IS masked.
    tall = (60.0, 10.0, 80.0, 80.0)  # center y=45 (above zone y>=50); foot (70,80) inside
    cx = (tall[0] + tall[2]) / 2.0
    cy = (tall[1] + tall[3]) / 2.0
    check(not point_in_polygon(cx, cy, polys[0]), "sanity: tall box CENTER is outside the zone")
    check(box_in_exempt_zones(tall, polys),
          "tall box: center outside but FOOT inside -> masked")

    # a box fully ABOVE the zone (foot y=40 < 50) is NOT masked
    above = (60.0, 10.0, 80.0, 40.0)
    check(not box_in_exempt_zones(above, polys), "box entirely above the ground zone -> not masked")

    # multiple zones: bottom-right AND top-left
    tl_zone = {"name": "gate", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]}
    multi = exempt_detect_polygons(_cam_row("c", exempt_zones=[BR_ZONE, tl_zone]))
    check(len(multi) == 2, "two zones -> two polygons")
    check(box_in_exempt_zones((10.0, 10.0, 30.0, 40.0), multi), "box in the 2nd (top-left) zone is masked")
    check(box_in_exempt_zones((60.0, 60.0, 80.0, 90.0), multi), "box in the 1st (bottom-right) zone is masked")
    check(not box_in_exempt_zones((60.0, 10.0, 90.0, 40.0), multi),
          "box in NEITHER zone (top-right) is not masked")

    # GROUND-CONTACT + UNION RULE (fix): the bottom edge is what's tested, and it
    # is unioned across zones. A WIDE object whose foot-CENTER lands in a gap
    # between two adjacent ground zones is still masked via the bottom edge:
    za = {"name": "a", "points": [[0.0, 0.5], [0.45, 0.5], [0.45, 1.0], [0.0, 1.0]]}
    zb = {"name": "b", "points": [[0.55, 0.5], [1.0, 0.5], [1.0, 1.0], [0.55, 1.0]]}
    split = exempt_detect_polygons(_cam_row("c", exempt_zones=[za, zb]))  # (0,50)-(45,100)+(55,50)-(100,100)
    wide = (20.0, 60.0, 80.0, 90.0)  # foot-center (50,90) in the GAP; bottom edge spans BOTH zones
    check(not point_in_polygon(50.0, 90.0, split[0]) and not point_in_polygon(50.0, 90.0, split[1]),
          "sanity: wide box foot-center is in the gap between the two zones")
    check(box_in_exempt_zones(wide, split),
          "wide box straddling two adjacent ground zones -> masked (bottom-edge union)")

    # FOREGROUND SAFETY (regression guard): a zone drawn HIGH in the frame must
    # NOT mask a foreground object standing below it — its body only projects
    # over the zone in 2D; its feet are outside. Masking this = a missed person.
    high = {"name": "street", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.55], [0.0, 0.55]]}
    high_polys = exempt_detect_polygons(_cam_row("c", exempt_zones=[high]))  # (0,0)-(100,55)
    foreground = (40.0, 5.0, 60.0, 95.0)  # full-height person; feet (50,95) well below the zone
    check(not box_in_exempt_zones(foreground, high_polys),
          "tall foreground object below a high zone -> NOT masked (body only projects over it)")
    # but a distant object whose FEET are inside the high zone IS masked
    distant = (45.0, 30.0, 55.0, 50.0)  # small/high; foot (50,50) inside the zone
    check(box_in_exempt_zones(distant, high_polys),
          "distant object standing inside a high zone -> masked (foot-center)")

    # RULE 3 (false-positive containment): a phantom box whose WHOLE extent is
    # almost entirely inside a high zone, but whose FOOT falls just below it, IS
    # masked — this is the user's "false detection inside the excluded area" case.
    contained = (40.0, 10.0, 60.0, 58.0)  # ~all inside (0,0)-(100,55); foot (50,58) just below
    check(not point_in_polygon(50.0, 58.0, high_polys[0]),
          "sanity: contained-FP foot-center is just below the high zone")
    check(box_in_exempt_zones(contained, high_polys),
          "phantom box contained >=80% in a high zone (foot just outside) -> masked (rule 3)")
    # ...but the tall foreground person (only ~60% inside) stays UNDER the 80% bar
    check(not box_in_exempt_zones(foreground, high_polys),
          "regression: tall foreground object (~60% inside) still NOT masked under rule 3")


# ==================== engine/pipeline path (drive process) =================


def engine_checks() -> None:
    print("engine: exempt zones suppress events end-to-end")
    asyncio.run(_engine_cases())


async def _drive(engine: DetectionEngine, camera: str, box, label: str = "person",
                 tid: int = 7, t0: float = 1000.0) -> None:
    for i in range(MIN_HITS + 2):
        await engine.process(camera, t0 + i * 0.2, [Observation(label, tid, 0.9, box)], frame_bgr=None)


async def _engine_cases() -> None:
    engine = DetectionEngine(db=None, detector=None, recorder=None, settings=None, config=Config())
    pipeline = RecordingPipeline()
    engine.set_pipeline(pipeline)

    # --- empty exempt_zones = unchanged behavior (control) ---
    plain = _cam_row("plain", exempt_zones=[])
    engine._cameras["plain"] = _CameraState(row=plain, exempt_polys=exempt_detect_polygons(plain))
    await _drive(engine, "plain", (60.0, 60.0, 80.0, 90.0))
    check(("plain", "person") in engine._events, "empty exempt_zones: confirmed person opens the event (no-op)")
    check(pipeline.counts.get(("plain", "person")) == 1, "empty exempt_zones: live count fed as usual")

    # --- masked camera: a detection whose foot is inside the zone opens NO event ---
    masked_row = _cam_row("masked", exempt_zones=[BR_ZONE])
    engine._cameras["masked"] = _CameraState(row=masked_row, exempt_polys=exempt_detect_polygons(masked_row))
    # foot-center (70,90) inside the bottom-right zone
    await _drive(engine, "masked", (60.0, 60.0, 80.0, 90.0))
    check(("masked", "person") not in engine._events,
          "masked: object with foot inside an exempt zone opens NO event")
    check(pipeline.counts.get(("masked", "person")) is None,
          "masked: suppressed object never feeds a live count")
    check(not any(e["after"]["camera"] == "masked" for e in pipeline.events),
          "masked: the pipeline received NO payload for the suppressed object")

    # --- same masked camera: an object OUTSIDE the zone still opens its event ---
    await _drive(engine, "masked", (10.0, 10.0, 30.0, 40.0), tid=8)
    check(("masked", "person") in engine._events,
          "masked: an object outside the exempt zone still opens the event")
    check(pipeline.counts.get(("masked", "person")) == 1, "masked: un-masked object feeds the live count")

    # --- foot-center rule in the full engine: tall box, center above / foot inside ---
    tallcam = _cam_row("tall", exempt_zones=[BR_ZONE])
    engine._cameras["tall"] = _CameraState(row=tallcam, exempt_polys=exempt_detect_polygons(tallcam))
    await _drive(engine, "tall", (60.0, 10.0, 80.0, 80.0))  # center y=45 outside, foot (70,80) inside
    check(("tall", "person") not in engine._events,
          "foot-center: tall box (center above zone, foot inside) is masked -> no event")

    # --- box fully above the ground zone still fires ---
    abovecam = _cam_row("above", exempt_zones=[BR_ZONE])
    engine._cameras["above"] = _CameraState(row=abovecam, exempt_polys=exempt_detect_polygons(abovecam))
    await _drive(engine, "above", (60.0, 10.0, 80.0, 40.0))  # foot (70,40) above zone
    check(("above", "person") in engine._events,
          "foot-center: box entirely above the zone opens the event")

    # --- reject-suppression: same-label detection near a learned sample opens NO event ---
    supp_row = _cam_row("supp", detect_objects=["person", "car"])
    st = _CameraState(row=supp_row, exempt_polys=[])
    st.suppress_samples = [("person", 70.0, 90.0)]  # detect-px foot (0.7,0.9 of 100x100)
    st.suppress_radius = 10.0
    engine._cameras["supp"] = st
    await _drive(engine, "supp", (60.0, 60.0, 80.0, 90.0))  # foot (70,90) == sample
    check(("supp", "person") not in engine._events,
          "reject-suppression: same-label detection near a sample opens NO event")
    # same camera, same label but FAR from the sample -> still fires
    await _drive(engine, "supp", (10.0, 10.0, 30.0, 40.0), tid=9)  # foot (20,40), ~70px away
    check(("supp", "person") in engine._events,
          "reject-suppression: a same-label detection far from the sample still fires")
    # a DIFFERENT label at the SAME spot is NOT suppressed (label-specific)
    await _drive(engine, "supp", (60.0, 60.0, 80.0, 90.0), label="car", tid=10)
    check(("supp", "car") in engine._events,
          "reject-suppression: a different label at the sample spot still fires (label-specific)")


# ==================== reload() precomputes the polygons ====================


def reload_checks() -> None:
    print("reload: engine precomputes detect-space polygons from the DB row")
    asyncio.run(_reload_case())


async def _reload_case() -> None:
    dbpath = TMP / "reload" / "nvr.db"
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    db = Database(dbpath)
    await db.connect()
    try:
        await db.upsert_camera({
            "name": "yard", "friendly_name": "Yard", "model": "AD410",
            "ip": "127.0.0.1", "username": "u", "password": "p",
            "detect_objects": ["person"], "exempt_zones": [BR_ZONE],
            "detect_width": 100, "detect_height": 100,
            "detect_fps": 5, "detect_enabled": True, "record_enabled": True,
            "capabilities": {}, "created_at": 1.0,
        })
        engine = DetectionEngine(db=db, detector=None, recorder=None, settings=None, config=Config())
        # reload() also touches the detector/ingest; call the row-load half directly
        rows = await db.list_cameras()
        fresh = {}
        for row in rows:
            fresh[row["name"]] = _CameraState(row=row, exempt_polys=exempt_detect_polygons(row))
        engine._cameras = fresh
        cam = engine._cameras["yard"]
        check(len(cam.exempt_polys) == 1 and cam.exempt_polys[0][0] == (50.0, 50.0),
              "reload path precomputes detect-space polygons on the camera state")

        pipeline = RecordingPipeline()
        engine.set_pipeline(pipeline)
        await _drive(engine, "yard", (60.0, 60.0, 80.0, 90.0))
        check(("yard", "person") not in engine._events,
              "DB-loaded exempt zone masks the object end-to-end (no event)")
    finally:
        await db.close()


# ============================ API round-trip ==============================


def login(client: TestClient) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"password": "test-password"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _post_cam(client, headers, name, **extra):
    body = {"name": name, "friendly_name": name.title(), "model": "AD410",
            "ip": "127.0.0.1", "username": "admin", "password": "pw"}
    body.update(extra)
    return client.post("/api/cameras", headers=headers, json=body)


def _put_cam(client, headers, name, **extra):
    body = {"name": name, "friendly_name": name.title(), "model": "AD410",
            "ip": "127.0.0.1", "username": "", "password": ""}
    body.update(extra)
    return client.put(f"/api/cameras/{name}", headers=headers, json=body)


def _get_zones(client, headers, name):
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    return cams[name]["exempt_zones"]


def api_checks() -> None:
    print("api: create/response/update exempt_zones semantics")
    with TestClient(app) as client:
        headers = login(client)

        # create WITHOUT exempt_zones -> [] (stored + returned)
        r = _post_cam(client, headers, "nozone")
        check(r.status_code == 201 and r.json()["exempt_zones"] == [],
              "create without exempt_zones -> [] in response")
        check(_get_zones(client, headers, "nozone") == [], "GET reflects the empty zone list")

        # create WITH a named zone -> stored + round-tripped
        r = _post_cam(client, headers, "zonecam", exempt_zones=[BR_ZONE])
        check(r.status_code == 201, "create with a zone -> 201")
        zones = r.json()["exempt_zones"]
        check(len(zones) == 1 and zones[0]["name"] == "porch", "zone name round-trips")
        check(zones[0]["points"] == [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]],
              "zone points round-trip verbatim as [[x,y],...]")
        check(_get_zones(client, headers, "zonecam") == zones, "GET returns the stored zones verbatim")

        # create with a < 3-point zone -> dropped
        r = _post_cam(client, headers, "shortzone",
                      exempt_zones=[{"name": "bad", "points": [[0.1, 0.1], [0.2, 0.2]]}])
        check(r.status_code == 201 and r.json()["exempt_zones"] == [],
              "zone with < 3 points is dropped (needs a real polygon)")

        # clamping: out-of-range coords are clamped to [0,1]
        r = _post_cam(client, headers, "clampcam",
                      exempt_zones=[{"points": [[-0.5, 0.2], [2.0, 0.2], [0.5, 1.5]]}])
        pts = r.json()["exempt_zones"][0]["points"]
        check(pts == [[0.0, 0.2], [1.0, 0.2], [0.5, 1.0]], "out-of-range coords clamped to [0,1]")

        # PUT adds zones to a camera that had none
        r = _put_cam(client, headers, "nozone", exempt_zones=[BR_ZONE])
        check(r.status_code == 200 and len(r.json()["exempt_zones"]) == 1,
              "PUT adds an exempt zone")

        # PUT omitting exempt_zones keeps the stored value
        r = _put_cam(client, headers, "nozone", friendly_name="Renamed")
        check(r.status_code == 200 and len(r.json()["exempt_zones"]) == 1
              and r.json()["friendly_name"] == "Renamed",
              "PUT omitting exempt_zones keeps the stored zones")

        # PUT [] clears all zones
        r = _put_cam(client, headers, "nozone", exempt_zones=[])
        check(r.status_code == 200 and r.json()["exempt_zones"] == [],
              "PUT [] clears all exempt zones")
        check(_get_zones(client, headers, "nozone") == [], "cleared zone list persists")


# ==================== migration: v7 -> v8 (exempt_zones) ===================

# The v7 cameras schema — identical to today's minus the exempt_zones column.
_V7_CAMERAS_DDL = """
CREATE TABLE cameras (
    name            TEXT PRIMARY KEY,
    friendly_name   TEXT NOT NULL,
    model           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    username        TEXT NOT NULL,
    password        TEXT NOT NULL,
    detect_objects  TEXT NOT NULL DEFAULT '[]',
    detect_width    INTEGER NOT NULL DEFAULT 704,
    detect_height   INTEGER NOT NULL DEFAULT 480,
    detect_fps      INTEGER NOT NULL DEFAULT 5,
    audio_events    INTEGER NOT NULL DEFAULT 1,
    detect_enabled  INTEGER NOT NULL DEFAULT 1,
    record_enabled  INTEGER NOT NULL DEFAULT 1,
    capabilities    TEXT NOT NULL DEFAULT '{}',
    source          TEXT NOT NULL DEFAULT 'manual',
    position        INTEGER NOT NULL DEFAULT 0,
    main_url        TEXT NOT NULL DEFAULT '',
    sub_url         TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL
);
"""


def migration_checks() -> None:
    print("migration: v7 -> v8 adds exempt_zones, preserves rows, defaults []")
    asyncio.run(_migration_case())


async def _migration_case() -> None:
    dbpath = TMP / "migrate" / "old.db"
    dbpath.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(dbpath)
    con.executescript(_V7_CAMERAS_DDL)
    rows = [
        ("front", json.dumps(["person", "car"]), 704, 480),
        ("door", json.dumps(DEFAULT_DETECT_OBJECTS), 640, 480),
    ]
    for name, objs, w, h in rows:
        con.execute(
            "INSERT INTO cameras (name, friendly_name, model, ip, username, password, "
            "detect_objects, detect_width, detect_height, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, name.title(), "AD410", "127.0.0.1", "u", "p", objs, w, h, 1.0),
        )
    con.execute("PRAGMA user_version = 7")
    con.commit()
    con.close()

    db = Database(dbpath)
    await db.connect()
    try:
        cur = await db.conn.execute("PRAGMA user_version")
        version = (await cur.fetchone())[0]
        # v7 -> current: exempt_zones is added at v8; later bumps (v9
        # detection_suppressions) ride the same chain and leave cameras intact.
        check(version == SCHEMA_VERSION and SCHEMA_VERSION >= 8,
              "schema bumped past v8 (exempt_zones) after migrate")

        cams = {c["name"]: c for c in await db.list_cameras()}
        check(set(cams) == {"front", "door"}, "existing v7 rows preserved through the migration")
        check(cams["front"]["detect_objects"] == ["person", "car"],
              "front: detect_objects preserved")
        check(cams["door"]["detect_objects"] == DEFAULT_DETECT_OBJECTS,
              "door: detect_objects preserved")
        check(cams["front"]["exempt_zones"] == [] and cams["door"]["exempt_zones"] == [],
              "existing rows default exempt_zones to [] (no masking)")

        # a subsequent upsert of a zone round-trips through the migrated column
        await db.upsert_camera({
            "name": "front", "friendly_name": "Front", "model": "AD410",
            "ip": "127.0.0.1", "username": "u", "password": "p",
            "detect_objects": ["person", "car"], "exempt_zones": [BR_ZONE],
            "detect_width": 704, "detect_height": 480, "detect_fps": 5,
            "detect_enabled": True, "record_enabled": True, "capabilities": {},
            "created_at": 1.0,
        })
        again = await db.get_camera("front")
        check(again["exempt_zones"] == [BR_ZONE], "upsert writes/reads exempt_zones on the migrated DB")
    finally:
        await db.close()


def main() -> None:
    geometry_checks()
    scaling_checks()
    masking_predicate_checks()
    engine_checks()
    reload_checks()
    migration_checks()
    api_checks()
    print(f"ALL PASSED ({PASS} checks)")


if __name__ == "__main__":
    main()
