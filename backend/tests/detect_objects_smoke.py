"""Smoke suite for the record-only detect_objects semantics (bug fix).

New contract (docs/CONTRACTS.md camera section):
  - ``detect_objects = []`` (an explicit empty list) means RECORD-ONLY: the
    camera records but detects nothing — no events, and no ingest ffmpeg /
    inference session at all.
  - ``detect_objects`` omitted on create means the DEFAULTS
    (``person/dog/cat/car``).
  - The API returns the STORED list verbatim (empty stays empty), so the
    object picker never sees the 4 defaults silently reappear.
  - One-time DB migration (schema v6): every existing camera whose
    detect_objects was NULL / '' / '[]' (historically MEANT "the defaults")
    is backfilled to the defaults, so deployed cameras keep detecting.

Covered here:
  - CREATE without detect_objects -> stored+returned defaults
  - CREATE with [] -> stored+returned [] (record-only)
  - CREATE with a custom list -> cleaned + round-tripped
  - PUT [] on a defaulted camera -> stored+returned []
  - PUT omitting detect_objects -> keeps the stored value
  - engine: a record-only ([]) camera tracks NO labels -> opens NO events,
    while a normal ([...]) camera still does
  - migration: an old (v5) DB with empty/custom rows -> empties become the
    defaults, customs untouched, user_version bumped to 6

The ingest-exclusion half (a record-only camera spawns no FrameSource) lives
in native_smoke.py's IngestManager cases.

Usage: python backend/tests/detect_objects_smoke.py  (needs backend deps).
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

TMP = Path(tempfile.mkdtemp(prefix="sentinel-detobj-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import DEFAULT_DETECT_OBJECTS  # noqa: E402
from app.db import SCHEMA_VERSION, Database  # noqa: E402
from app.main import app  # noqa: E402
from app.native.engine import (  # noqa: E402
    MIN_HITS,
    DetectionEngine,
    Observation,
    _CameraState,
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


# The background capability probe would hang on a blackholed IP; the object
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
        "detect_width": 8, "detect_height": 6,
        "ip": "10.0.0.9", "username": "u", "password": "p",
        "main_url": "", "sub_url": "", "record_enabled": True,
    }
    row.update(over)
    return row


def login(client: TestClient) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"password": "test-password"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _post_cam(client: TestClient, headers: dict, name: str, **extra):
    body = {
        "name": name, "friendly_name": name.title(), "model": "AD410",
        "ip": "127.0.0.1", "username": "admin", "password": "pw",
    }
    body.update(extra)
    return client.post("/api/cameras", headers=headers, json=body)


def _put_cam(client: TestClient, headers: dict, name: str, **extra):
    body = {
        "name": name, "friendly_name": name.title(), "model": "AD410",
        "ip": "127.0.0.1", "username": "", "password": "",
    }
    body.update(extra)
    return client.put(f"/api/cameras/{name}", headers=headers, json=body)


def _get_objects(client: TestClient, headers: dict, name: str) -> list[str]:
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    return cams[name]["detect_objects"]


# ---------------- API: create / response / update semantics ----------------


def api_checks() -> None:
    print("api: create/response/update detect_objects semantics")
    with TestClient(app) as client:
        headers = login(client)

        # create WITHOUT detect_objects -> defaults (stored + returned)
        r = _post_cam(client, headers, "defcam")
        check(r.status_code == 201, "POST without detect_objects -> 201")
        check(r.json()["detect_objects"] == DEFAULT_DETECT_OBJECTS,
              "create response defaults person/dog/cat/car when field omitted")
        check(_get_objects(client, headers, "defcam") == DEFAULT_DETECT_OBJECTS,
              "GET reflects the stored defaults (round-trip)")

        # create WITH [] -> record-only, stays empty
        r = _post_cam(client, headers, "reconly", detect_objects=[])
        check(r.status_code == 201 and r.json()["detect_objects"] == [],
              "create with [] returns [] (record-only, no default coercion)")
        check(_get_objects(client, headers, "reconly") == [],
              "GET shows the empty list verbatim — defaults do NOT reappear")

        # create WITH a custom (dirty) list -> cleaned + round-tripped
        r = _post_cam(client, headers, "customcam",
                      detect_objects=["Person", "", "car", "bad slug!"])
        check(r.status_code == 201 and r.json()["detect_objects"] == ["person", "car"],
              "create cleans/round-trips a custom list (junk slugs dropped)")

        # PUT [] on the defaulted camera -> stored/returned []
        r = _put_cam(client, headers, "defcam", detect_objects=[])
        check(r.status_code == 200 and r.json()["detect_objects"] == [],
              "PUT [] empties a previously-defaulted camera (record-only)")
        check(_get_objects(client, headers, "defcam") == [],
              "emptied list persists — the 4 defaults do not come back")

        # PUT WITHOUT detect_objects -> keeps the stored (now empty) value
        r = _put_cam(client, headers, "defcam", friendly_name="Renamed")
        check(r.status_code == 200 and r.json()["detect_objects"] == []
              and r.json()["friendly_name"] == "Renamed",
              "PUT omitting detect_objects keeps the stored empty list")

        # PUT a real list back on -> re-populates (re-enables detection)
        r = _put_cam(client, headers, "defcam", detect_objects=["dog"])
        check(r.status_code == 200 and r.json()["detect_objects"] == ["dog"],
              "PUT re-adds objects -> detection re-enabled")


# ---------------- engine: record-only camera opens no events ----------------


def engine_checks() -> None:
    print("engine: [] detect_objects tracks nothing (record-only)")
    asyncio.run(_engine_cases())


async def _engine_cases() -> None:
    from app.config import Config

    engine = DetectionEngine(db=None, detector=None, recorder=None,
                             settings=None, config=Config())
    pipeline = RecordingPipeline()
    engine.set_pipeline(pipeline)

    # record-only camera: empty detect_objects
    engine._cameras["rec"] = _CameraState(row=_cam_row("rec", detect_objects=[]))
    # normal camera: detects person (control)
    engine._cameras["det"] = _CameraState(row=_cam_row("det", detect_objects=["person"]))

    box = (10.0, 10.0, 40.0, 40.0)
    t0 = 1000.0
    for i in range(MIN_HITS + 2):  # plenty to confirm a track
        obs = [Observation("person", 7, 0.9, box)]
        await engine.process("rec", t0 + i * 0.2, obs, frame_bgr=None)
        await engine.process("det", t0 + i * 0.2, obs, frame_bgr=None)

    check(("rec", "person") not in engine._events,
          "record-only camera opens NO event despite a confirmed person track")
    check(pipeline.counts.get(("rec", "person")) is None,
          "record-only camera never feeds a live count")
    check(("det", "person") in engine._events,
          "control camera with ['person'] still opens the event")
    check(pipeline.counts.get(("det", "person")) == 1,
          "control camera feeds the live count as before")


# ---------------- migration: v5 -> v6 backfills empty rows ----------------


def migration_checks() -> None:
    print("migration: v5 empty detect_objects backfilled to defaults")
    asyncio.run(_migration_case())


_V5_CAMERAS_DDL = """
CREATE TABLE cameras (
    name            TEXT PRIMARY KEY,
    friendly_name   TEXT NOT NULL,
    model           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    username        TEXT NOT NULL,
    password        TEXT NOT NULL,
    detect_objects  TEXT,
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


async def _migration_case() -> None:
    dbpath = TMP / "migrate" / "old.db"
    dbpath.parent.mkdir(parents=True, exist_ok=True)

    # Seed a synthetic pre-v6 (user_version=5) DB by hand. detect_objects is
    # left nullable here so all three "empty" shapes (NULL / '' / '[]') that
    # the migration WHERE covers can be exercised.
    con = sqlite3.connect(dbpath)
    con.executescript(_V5_CAMERAS_DDL)
    rows = [
        ("nullobj", None),
        ("emptystr", ""),
        ("emptyjson", "[]"),
        ("custom1", json.dumps(["person"])),
        ("custom2", json.dumps(["dog", "cat"])),
    ]
    for name, objs in rows:
        con.execute(
            "INSERT INTO cameras (name, friendly_name, model, ip, username, "
            "password, detect_objects, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (name, name.title(), "AD410", "127.0.0.1", "u", "p", objs, 1.0),
        )
    con.execute("PRAGMA user_version = 5")
    con.commit()
    con.close()

    # Run the real migration.
    db = Database(dbpath)
    await db.connect()
    try:
        cur = await db.conn.execute("PRAGMA user_version")
        version = (await cur.fetchone())[0]
        check(version == SCHEMA_VERSION >= 6, "schema bumped to v6+ after migrate")

        cams = {c["name"]: c["detect_objects"] for c in await db.list_cameras()}
        for empty in ("nullobj", "emptystr", "emptyjson"):
            check(cams[empty] == DEFAULT_DETECT_OBJECTS,
                  f"'{empty}' backfilled to the defaults (keeps detecting)")
        check(cams["custom1"] == ["person"], "custom ['person'] left untouched")
        check(cams["custom2"] == ["dog", "cat"], "custom ['dog','cat'] left untouched")
    finally:
        await db.close()


def main() -> None:
    migration_checks()
    engine_checks()
    api_checks()
    print(f"ALL PASSED ({PASS} checks)")


if __name__ == "__main__":
    main()
