"""SQLite (aiosqlite) storage: cameras, camera groups, events, push
subscriptions, settings KV.

One shared connection; aiosqlite serializes access internally. Schema
migrations run on boot keyed off PRAGMA user_version.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .config import DEFAULT_DETECT_OBJECTS

SCHEMA_VERSION = 21

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    name            TEXT PRIMARY KEY,
    friendly_name   TEXT NOT NULL,
    model           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    username        TEXT NOT NULL,
    password        TEXT NOT NULL,
    detect_objects  TEXT NOT NULL DEFAULT '[]',
    exempt_zones    TEXT NOT NULL DEFAULT '[]',
    include_zones   TEXT NOT NULL DEFAULT '[]',
    cross_lines     TEXT NOT NULL DEFAULT '[]',
    notify_on_cross INTEGER NOT NULL DEFAULT 0,
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
    ir_state        TEXT NOT NULL DEFAULT '{}',
    detect_mode     TEXT,
    audio_codec     TEXT NOT NULL DEFAULT 'g711a',
    smart_spotlight INTEGER NOT NULL DEFAULT 0,
    spotlight_hold_seconds INTEGER NOT NULL DEFAULT 60,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS camera_groups (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT UNIQUE NOT NULL,
    cameras  TEXT NOT NULL DEFAULT '[]',
    position INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frigate_id   TEXT UNIQUE,
    camera       TEXT NOT NULL,
    label        TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 1,
    score        REAL NOT NULL DEFAULT 0,
    start_time   REAL NOT NULL,
    end_time     REAL,
    has_clip     INTEGER NOT NULL DEFAULT 0,
    has_snapshot INTEGER NOT NULL DEFAULT 0,
    zones        TEXT NOT NULL DEFAULT '[]',
    box          TEXT NOT NULL DEFAULT '[]',
    -- All distinct detected classes (on the camera detect list) seen across the
    -- event, accumulated by the pipeline. `label` stays the PRIMARY class for
    -- back-compat; `labels` is the multi-object superset. '[]' = legacy row
    -- (the serializer falls back to [label] then).
    labels       TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_events_start  ON events(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_label  ON events(label, start_time DESC);

CREATE TABLE IF NOT EXISTS detection_suppressions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    camera     TEXT NOT NULL,
    label      TEXT NOT NULL,
    foot_x     REAL NOT NULL,
    foot_y     REAL NOT NULL,
    has_thumb  INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_suppress_camera ON detection_suppressions(camera);

CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint     TEXT PRIMARY KEY,
    subscription TEXT NOT NULL,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS apns_devices (
    device_token TEXT PRIMARY KEY,
    device_name  TEXT NOT NULL DEFAULT '',
    key_b64      TEXT NOT NULL,
    environment  TEXT NOT NULL DEFAULT 'production'
                 CHECK(environment IN ('sandbox','production')),
    created_at   REAL NOT NULL
);

-- PushKit VoIP registrations for the CallKit doorbell ring. Distinct from
-- apns_devices: the VoIP push carries a MINIMAL, NON-E2E-encrypted payload
-- (the app must read it immediately to report the incoming call), so there is
-- no per-registration encryption key here — only the token + APNs environment.
CREATE TABLE IF NOT EXISTS voip_devices (
    device_token TEXT PRIMARY KEY,
    device_name  TEXT NOT NULL DEFAULT '',
    environment  TEXT NOT NULL DEFAULT 'production'
                 CHECK(environment IN ('sandbox','production')),
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('admin','viewer')),
    created_at    REAL NOT NULL
);

-- Camera reachability history, stored as TRANSITION intervals (one row per
-- state, not per poll — 11 cams x 45 s would otherwise be ~21k rows/day). A row
-- with end_ts NULL is the CURRENT state. Uptime over a window = sum of the
-- 'online' intervals clipped to the window / window length. See camera_health.py.
CREATE TABLE IF NOT EXISTS camera_health (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    camera   TEXT NOT NULL,
    online   INTEGER NOT NULL,
    start_ts REAL NOT NULL,
    end_ts   REAL
);
CREATE INDEX IF NOT EXISTS idx_camera_health_cam_start
    ON camera_health(camera, start_ts);
"""


class Database:
    def __init__(self, path: Path):
        self._path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        # NORMAL, not the SQLite default of FULL. Under WAL, FULL fsyncs the WAL
        # on EVERY commit — and every write helper here commits individually,
        # while the detection pipeline commits constantly (an update per tracked
        # object per heartbeat across 12 cameras). Every API read then queues
        # behind those fsyncs on the box's array.
        #
        # What NORMAL actually costs: under WAL it is durable against process
        # crash — including the restart watchdog's os._exit — and risks losing
        # only the last commits on an OS crash or power cut. Trading "the last
        # second of event rows survives a power cut" for making the whole API
        # responsive is the right trade for an NVR; the footage itself is
        # written by ffmpeg, not through this connection.
        await self._db.execute("PRAGMA synchronous=NORMAL")
        # Wait rather than raising SQLITE_BUSY the instant a writer holds the
        # lock — this DB is also read out-of-process (diagnostics on the box).
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._migrate()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("database not connected")
        return self._db

    async def _migrate(self) -> None:
        cur = await self.conn.execute("PRAGMA user_version")
        row = await cur.fetchone()
        version = row[0] if row else 0
        if version < 1:
            await self.conn.executescript(_SCHEMA)
        else:
            if version < 2:
                # v2: camera provenance (historical 'manual' | 'frigate';
                # kept harmlessly). Fresh DBs get it via _SCHEMA above.
                await self.conn.execute(
                    "ALTER TABLE cameras ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'"
                )
            if version < 3:
                # v3: dashboard camera ordering + camera groups. Existing
                # rows get positions in rowid order (their historical
                # insertion order) per the contract addendum.
                await self.conn.execute(
                    "ALTER TABLE cameras ADD COLUMN position INTEGER NOT NULL DEFAULT 0"
                )
                await self.conn.execute(
                    """
                    UPDATE cameras SET position = (
                        SELECT COUNT(*) FROM cameras AS c2 WHERE c2.rowid <= cameras.rowid
                    )
                    """
                )
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS camera_groups (
                        id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        name     TEXT UNIQUE NOT NULL,
                        cameras  TEXT NOT NULL DEFAULT '[]',
                        position INTEGER NOT NULL
                    )
                    """
                )
            if version < 4:
                # v4: native engine — optional RTSP URL overrides. Empty
                # string = "derive Amcrest default from ip+creds"
                # (native/streams.py). Fresh DBs get them via _SCHEMA above.
                await self.conn.execute(
                    "ALTER TABLE cameras ADD COLUMN main_url TEXT NOT NULL DEFAULT ''"
                )
                await self.conn.execute(
                    "ALTER TABLE cameras ADD COLUMN sub_url TEXT NOT NULL DEFAULT ''"
                )
            if version < 5:
                # v5: multi-user RBAC — DB users (admin/viewer). The built-in
                # admin stays env-controlled and is NOT a row here. Fresh DBs
                # get this table via _SCHEMA above.
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        username      TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role          TEXT NOT NULL CHECK(role IN ('admin','viewer')),
                        created_at    REAL NOT NULL
                    )
                    """
                )
            if version < 6:
                # v6: detect_objects semantics change — an EMPTY list now means
                # "record only, detect nothing" instead of "the defaults".
                # Historically every empty row (NULL / '' / '[]') MEANT the
                # defaults, so backfill those existing rows with
                # DEFAULT_DETECT_OBJECTS. This preserves current behavior
                # (deployed cameras keep detecting person/dog/cat/car); after
                # this, an empty list only ever results from an explicit user
                # action (emptying the object picker).
                await self.conn.execute(
                    """
                    UPDATE cameras SET detect_objects = ?
                    WHERE detect_objects IS NULL
                       OR detect_objects = ''
                       OR detect_objects = '[]'
                    """,
                    (json.dumps(DEFAULT_DETECT_OBJECTS),),
                )
            if version < 7:
                # v7: APNs (iOS) push registrations — docs/push-architecture.md.
                # device_token is the lowercased hex APNs token; key_b64 is the
                # per-registration 32-byte E2E key (base64). Fresh DBs get the
                # table via _SCHEMA above.
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS apns_devices (
                        device_token TEXT PRIMARY KEY,
                        device_name  TEXT NOT NULL DEFAULT '',
                        key_b64      TEXT NOT NULL,
                        environment  TEXT NOT NULL DEFAULT 'production'
                                     CHECK(environment IN ('sandbox','production')),
                        created_at   REAL NOT NULL
                    )
                    """
                )
            if version < 8:
                # v8: per-camera exempt (privacy/ignore) detection zones. A JSON
                # list of polygons in NORMALIZED (0..1) coords; any object whose
                # box foot-center falls inside a zone is suppressed by the engine
                # (no event / notification / annotation). Existing rows get the
                # empty default ([] = no masking, unchanged behavior). Fresh DBs
                # get the column via _SCHEMA above.
                await self.conn.execute(
                    "ALTER TABLE cameras ADD COLUMN exempt_zones TEXT NOT NULL DEFAULT '[]'"
                )
            if version < 9:
                # v9: reject-to-suppress ("wrong / not a <object>"). Two additive
                # changes, exactly mirroring the v8/exempt_zones pattern:
                #   * events.box — the best detection box (detect-stream px, a
                #     JSON [x1,y1,x2,y2]) captured with the event so a later
                #     reject can derive its normalized foot-point. Existing rows
                #     default '[]' (no box; a reject just deletes them).
                #   * detection_suppressions — label-specific auto soft-ignore
                #     samples learned from rejects: {camera, label, foot_x/foot_y
                #     (normalized 0..1 bottom-center), created_at}. The engine
                #     drops a matching detection near a sample BEFORE event
                #     creation. Fresh DBs get both via _SCHEMA above.
                #
                # The events ALTER is guarded on the table existing: a real DB
                # has had events since v1, but this is the first migration to
                # touch events, so a synthetic cameras-only upgrade fixture (the
                # pattern every prior version's migration test uses) would
                # otherwise fail here. The suppressions table is independent.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE events ADD COLUMN box TEXT NOT NULL DEFAULT '[]'"
                    )
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detection_suppressions (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera     TEXT NOT NULL,
                        label      TEXT NOT NULL,
                        foot_x     REAL NOT NULL,
                        foot_y     REAL NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                await self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_suppress_camera "
                    "ON detection_suppressions(camera)"
                )
            if version < 10:
                # v10: two additive changes for full IR control + reject thumbs.
                #   * cameras.ir_state — the user's DESIRED IR mode+brightness
                #     (JSON {"mode","brightness"[,"day_night"]}) so a doorbell
                #     that reverts IR on RTSP connect can be re-asserted. Existing
                #     rows default '{}' (nothing pinned). Guarded on the cameras
                #     table existing (a synthetic suppressions-only upgrade
                #     fixture would otherwise fail).
                #   * detection_suppressions.has_thumb — 1 when a cropped
                #     thumbnail of the rejected detection was saved beside the
                #     suppression (data_dir/suppression-thumbs/{id}.jpg). Existing
                #     rows default 0 (no thumb). Fresh DBs get both via _SCHEMA.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE cameras ADD COLUMN ir_state TEXT NOT NULL DEFAULT '{}'"
                    )
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='detection_suppressions'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE detection_suppressions "
                        "ADD COLUMN has_thumb INTEGER NOT NULL DEFAULT 0"
                    )
            if version < 11:
                # v11: per-camera server-detection mode (camera-AI-gated
                # detection). cameras.detect_mode is one of VALID_DETECT_MODES
                # ("always" | "camera_ai" | "camera_ai_only") or NULL. NULL means
                # "unset" -> inherit settings.detection.default_mode (defaulting
                # to "always"), so EXISTING rows keep the historical continuous-
                # inference behavior untouched. The column is deliberately
                # nullable (no SQL default) so "unset/inherit" is representable
                # and distinct from an explicit "always". Guarded on the cameras
                # table existing (a synthetic non-cameras upgrade fixture would
                # otherwise fail). Fresh DBs get the column via _SCHEMA above.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE cameras ADD COLUMN detect_mode TEXT"
                    )
            if version < 12:
                # v12: two additive changes.
                #   * events.labels — the full set of distinct detected classes
                #     (on the camera detect list) seen across a multi-object
                #     event. `label` stays the PRIMARY class for back-compat.
                #     Existing rows default '[]'; the serializer falls back to
                #     [label] for them. Guarded on the events table existing (a
                #     synthetic non-events upgrade fixture would otherwise fail).
                #   * voip_devices — PushKit VoIP tokens for the CallKit doorbell
                #     ring (no per-registration key: the VoIP payload is minimal
                #     + not E2E-encrypted). Fresh DBs get both via _SCHEMA above.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE events ADD COLUMN labels TEXT NOT NULL DEFAULT '[]'"
                    )
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS voip_devices (
                        device_token TEXT PRIMARY KEY,
                        device_name  TEXT NOT NULL DEFAULT '',
                        environment  TEXT NOT NULL DEFAULT 'production'
                                     CHECK(environment IN ('sandbox','production')),
                        created_at   REAL NOT NULL
                    )
                    """
                )
            if version < 13:
                # v13: per-camera live-view audio codec preference.
                # cameras.audio_codec is 'g711a' (DEFAULT) | 'aac'. 'g711a' forces
                # the device audio encoder to G.711A so its NATIVE RTSP audio is
                # WebRTC-legal and live-view (go2rtc/WebRTC) audio WORKS — the
                # historical behavior, hence the default. 'aac' forces AAC (higher
                # recording quality, but NO live-view audio: WebRTC can't carry
                # AAC). Existing rows default 'g711a' (unchanged behavior). Guarded
                # on the cameras table existing (a synthetic non-cameras upgrade
                # fixture would otherwise fail). Fresh DBs get the column via
                # _SCHEMA above.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE cameras ADD COLUMN audio_codec TEXT NOT NULL DEFAULT 'g711a'"
                    )
            if version < 14:
                # v14: per-camera "Smart spotlight". When on, a PERSON detected at
                # NIGHT (local sunset..sunrise) on a white_light-capable camera has
                # its on-demand spotlight turned ON (set_white_light) and held until
                # 60 s after the last person detection, then turned OFF — driven
                # live by native/spotlight.SpotlightController off the stored flag.
                # cameras.smart_spotlight is 0 (DEFAULT, off) | 1. Existing rows
                # default 0 (feature off — unchanged behavior). Guarded on the
                # cameras table existing (a synthetic non-cameras upgrade fixture
                # would otherwise fail). Fresh DBs get the column via _SCHEMA above.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE cameras ADD COLUMN smart_spotlight INTEGER NOT NULL DEFAULT 0"
                    )
            if version < 15:
                # v15: per-camera "Spotlight hold seconds". Extends smart_spotlight
                # (v14): how long the smart spotlight stays ON after the LAST person
                # detection at night before the controller turns it OFF. Replaces
                # the previously hardcoded 60 s trailing hold with a per-camera
                # value. cameras.spotlight_hold_seconds is an INTEGER (valid range
                # 5..600; validated on the API, clamped defensively in the
                # controller). Existing rows default 60 (the old hardcoded value —
                # unchanged behavior). Guarded on the cameras table existing (a
                # synthetic non-cameras upgrade fixture would otherwise fail). Fresh
                # DBs get the column via _SCHEMA above.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    await self.conn.execute(
                        "ALTER TABLE cameras ADD COLUMN spotlight_hold_seconds INTEGER NOT NULL DEFAULT 60"
                    )
            if version < 16:
                # v16: camera reachability history (up/down interval rows) that
                # drives the Camera Health screen + optional down-alerts. New
                # table only — touches no existing row, so it is always safe.
                await self.conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS camera_health (
                        id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera   TEXT NOT NULL,
                        online   INTEGER NOT NULL,
                        start_ts REAL NOT NULL,
                        end_ts   REAL
                    );
                    CREATE INDEX IF NOT EXISTS idx_camera_health_cam_start
                        ON camera_health(camera, start_ts);
                    """
                )
            if version < 19:
                # v19: face recognition REMOVED entirely. Idempotently drop the
                # enrolment tables and the hint columns / per-camera gate if a
                # prior build (v17/v18) created them — SQLite has no IF EXISTS for
                # a column, so pragma-check first. A box that never had face rec
                # runs these as pure no-ops.
                await self.conn.executescript(
                    """
                    DROP TABLE IF EXISTS face_unknowns;
                    DROP TABLE IF EXISTS face_embeddings;
                    DROP TABLE IF EXISTS face_persons;
                    """
                )
                for _tbl, _col in (
                    ("events", "person"), ("events", "person_score"),
                    ("cameras", "face_enabled"),
                ):
                    cur = await self.conn.execute(f"PRAGMA table_info({_tbl})")
                    if _col in [r[1] for r in await cur.fetchall()]:
                        await self.conn.execute(f"ALTER TABLE {_tbl} DROP COLUMN {_col}")
            if version < 20:
                # v20: the POSITIVE half of per-camera geometry, mirroring the
                # v8/exempt_zones pattern exactly.
                #   * cameras.include_zones — allow-list polygons (normalized
                #     0..1). [] keeps the existing behavior (watch everything);
                #     any entry makes the camera drop detections whose feet land
                #     outside every zone, before an event can open.
                #   * cameras.cross_lines — {name, start:[x,y], end:[x,y]}
                #     boundaries counted in/out by sv.LineZone. [] = none.
                # Both default to '[]' so an upgraded box behaves identically
                # until the operator draws something. Guarded on the cameras
                # table existing so a synthetic single-table upgrade fixture (the
                # pattern the migration tests use) still runs.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    existing = [
                        r[1] for r in await (
                            await self.conn.execute("PRAGMA table_info(cameras)")
                        ).fetchall()
                    ]
                    for _col in ("include_zones", "cross_lines"):
                        if _col not in existing:
                            await self.conn.execute(
                                f"ALTER TABLE cameras ADD COLUMN {_col} "
                                "TEXT NOT NULL DEFAULT '[]'"
                            )
            if version < 21:
                # v21: cameras.notify_on_cross — "only alert me when something
                # crosses a line on this camera". Additive, defaults 0 (off), so
                # an upgraded box notifies exactly as it did before.
                #
                # It gates the NOTIFICATION only: the event, clip and snapshot
                # are recorded either way. And it is inert unless the camera has
                # crossing lines — a flag that silently kills every alert because
                # the last line was deleted is not a setting, it is a trap.
                cur = await self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cameras'"
                )
                if await cur.fetchone() is not None:
                    existing = [
                        r[1] for r in await (
                            await self.conn.execute("PRAGMA table_info(cameras)")
                        ).fetchall()
                    ]
                    if "notify_on_cross" not in existing:
                        await self.conn.execute(
                            "ALTER TABLE cameras ADD COLUMN notify_on_cross "
                            "INTEGER NOT NULL DEFAULT 0"
                        )
        if version < SCHEMA_VERSION:
            await self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await self.conn.commit()

    # ---------- cameras ----------

    @staticmethod
    def camera_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "friendly_name": row["friendly_name"],
            "model": row["model"],
            "ip": row["ip"],
            "username": row["username"],
            "password": row["password"],
            "detect_objects": json.loads(row["detect_objects"]),
            "exempt_zones": json.loads(row["exempt_zones"]),
            # INCLUDE zones (native/zones.py): the allow-list counterpart of
            # exempt_zones. [] (the default) watches the whole frame; any
            # entry drops every detection whose feet land outside them all.
            "include_zones": json.loads(row["include_zones"]),
            # Crossing lines: normalized {name, start:[x,y], end:[x,y]} pairs
            # counted in/out by sv.LineZone. [] = no crossing detection.
            "cross_lines": json.loads(row["cross_lines"]),
            # When true AND the camera has crossing lines, an object event on
            # this camera only NOTIFIES once something crosses one. The event,
            # its clip and its snapshot are recorded either way — this gates the
            # alert, never the footage.
            "notify_on_cross": bool(row["notify_on_cross"]),
            "detect_width": row["detect_width"],
            "detect_height": row["detect_height"],
            "detect_fps": row["detect_fps"],
            "audio_events": bool(row["audio_events"]),
            "detect_enabled": bool(row["detect_enabled"]),
            "record_enabled": bool(row["record_enabled"]),
            "capabilities": json.loads(row["capabilities"]),
            "source": row["source"],
            "position": row["position"],
            "main_url": row["main_url"],
            "sub_url": row["sub_url"],
            # Desired IR mode+brightness the operator pinned (re-asserted on
            # doorbell stream reconnect). {} = nothing pinned.
            "ir_state": json.loads(row["ir_state"]),
            # Per-camera server-detection mode (VALID_DETECT_MODES) or None when
            # unset (-> inherit settings.detection.default_mode). See config.
            "detect_mode": row["detect_mode"],
            # Per-camera live-view audio codec preference: 'g711a' (default;
            # WebRTC-legal, so live-view audio works) | 'aac' (higher recording
            # quality, no live-view audio). Drives provision_audio on the camera.
            "audio_codec": row["audio_codec"],
            # Per-camera "Smart spotlight": when true, a person detected at night
            # on a white_light camera auto-turns the spotlight on (held 60 s past
            # the last person). Persist-only — SpotlightController reads it live.
            "smart_spotlight": bool(row["smart_spotlight"]),
            # Per-camera smart-spotlight trailing hold (seconds): how long the
            # spotlight stays on after the LAST person detection before it turns
            # off. Integer, default 60, valid 5..600. Persist-only — the
            # SpotlightController reads the stored value live (and clamps it).
            "spotlight_hold_seconds": int(row["spotlight_hold_seconds"]),
        }

    async def list_cameras(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM cameras ORDER BY position, name")
        rows = await cur.fetchall()
        return [self.camera_row_to_dict(r) for r in rows]

    async def get_camera(self, name: str) -> Optional[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM cameras WHERE name = ?", (name,))
        row = await cur.fetchone()
        return self.camera_row_to_dict(row) if row else None

    # New rows append at the end of the display order (position = max+1)
    # unless the caller supplies an explicit position; ON CONFLICT updates
    # never touch position, so edits keep a camera's slot.
    _CAMERA_INSERT_SQL = """
        INSERT INTO cameras (name, friendly_name, model, ip, username, password,
                             detect_objects, exempt_zones, include_zones, cross_lines,
                             notify_on_cross,
                             detect_width, detect_height,
                             detect_fps, audio_events, detect_enabled, record_enabled,
                             capabilities, source, position, main_url, sub_url,
                             ir_state, detect_mode, audio_codec, smart_spotlight,
                             spotlight_hold_seconds, created_at)
        VALUES (:name, :friendly_name, :model, :ip, :username, :password,
                :detect_objects, :exempt_zones, :include_zones, :cross_lines,
                :notify_on_cross,
                :detect_width, :detect_height,
                :detect_fps, :audio_events, :detect_enabled, :record_enabled,
                :capabilities, :source,
                COALESCE(:position, (SELECT COALESCE(MAX(position), 0) + 1 FROM cameras)),
                :main_url, :sub_url, :ir_state, :detect_mode, :audio_codec,
                :smart_spotlight, :spotlight_hold_seconds, :created_at)
    """

    @staticmethod
    def _camera_params(cam: dict[str, Any]) -> dict[str, Any]:
        return {
            **cam,
            "detect_objects": json.dumps(cam.get("detect_objects") or []),
            "exempt_zones": json.dumps(cam.get("exempt_zones") or []),
            "include_zones": json.dumps(cam.get("include_zones") or []),
            "cross_lines": json.dumps(cam.get("cross_lines") or []),
            "notify_on_cross": int(cam.get("notify_on_cross") or False),
            "capabilities": json.dumps(cam.get("capabilities") or {}),
            "audio_events": int(cam.get("audio_events", True)),
            "detect_enabled": int(cam.get("detect_enabled", True)),
            "record_enabled": int(cam.get("record_enabled", True)),
            "source": cam.get("source") or "manual",
            "position": cam.get("position"),
            "main_url": cam.get("main_url") or "",
            "sub_url": cam.get("sub_url") or "",
            "ir_state": json.dumps(cam.get("ir_state") or {}),
            # None (unset) is stored as SQL NULL and inherits the settings
            # default; a valid mode string is stored verbatim. An empty string
            # is normalized to NULL so "unset" round-trips cleanly.
            "detect_mode": (cam.get("detect_mode") or None),
            # Live-view audio codec preference; defaults to 'g711a' (WebRTC-legal,
            # live-view audio works) when unset.
            "audio_codec": (cam.get("audio_codec") or "g711a"),
            # Per-camera Smart-spotlight flag; stored as 0/1, default 0 (off).
            "smart_spotlight": int(cam.get("smart_spotlight") or False),
            # Per-camera smart-spotlight trailing hold in seconds; default 60 when
            # unset/None (the historical hardcoded value). Stored verbatim; the
            # API validates 5..600 and the controller clamps defensively.
            "spotlight_hold_seconds": int(cam.get("spotlight_hold_seconds") or 60),
            "created_at": cam.get("created_at") or time.time(),
        }

    async def upsert_camera(self, cam: dict[str, Any]) -> None:
        await self.conn.execute(
            self._CAMERA_INSERT_SQL
            + """
            ON CONFLICT(name) DO UPDATE SET
                friendly_name = excluded.friendly_name,
                model         = excluded.model,
                ip            = excluded.ip,
                username      = excluded.username,
                password      = excluded.password,
                detect_objects= excluded.detect_objects,
                exempt_zones  = excluded.exempt_zones,
                include_zones = excluded.include_zones,
                cross_lines   = excluded.cross_lines,
                notify_on_cross = excluded.notify_on_cross,
                detect_width  = excluded.detect_width,
                detect_height = excluded.detect_height,
                detect_fps    = excluded.detect_fps,
                audio_events  = excluded.audio_events,
                detect_enabled= excluded.detect_enabled,
                record_enabled= excluded.record_enabled,
                capabilities  = excluded.capabilities,
                source        = excluded.source,
                main_url      = excluded.main_url,
                sub_url       = excluded.sub_url,
                ir_state      = excluded.ir_state,
                detect_mode   = excluded.detect_mode,
                audio_codec   = excluded.audio_codec,
                smart_spotlight = excluded.smart_spotlight,
                spotlight_hold_seconds = excluded.spotlight_hold_seconds
            """,
            self._camera_params(cam),
        )
        await self.conn.commit()

    async def insert_camera_if_absent(self, cam: dict[str, Any]) -> bool:
        """Atomic INSERT .. ON CONFLICT(name) DO NOTHING; True when the row
        was created (never overwrites concurrent adds — the check-then-insert
        alternative has a window between the awaits)."""
        cur = await self.conn.execute(
            self._CAMERA_INSERT_SQL + " ON CONFLICT(name) DO NOTHING",
            self._camera_params(cam),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def set_camera_ir_state(self, name: str, ir_state: dict[str, Any]) -> None:
        """Persist a camera's desired IR mode+brightness (targeted update — does
        not touch any other column). Used by PUT settings so the doorbell IR
        re-assert has a stored target."""
        await self.conn.execute(
            "UPDATE cameras SET ir_state = ? WHERE name = ?",
            (json.dumps(ir_state or {}), name),
        )
        await self.conn.commit()

    async def set_camera_capability(self, name: str, key: str, value: bool) -> bool:
        """Set a single capability flag on a camera's stored capabilities JSON,
        leaving every other capability untouched (targeted json_set — avoids the
        read-modify-write race a full upsert would have against a concurrent
        edit). Used by the on-connect speaker probe. Returns True when a row was
        updated. ``key`` is a fixed capability name (never user input)."""
        cur = await self.conn.execute(
            "UPDATE cameras SET capabilities = json_set(capabilities, '$.' || ?, json(?)) "
            "WHERE name = ?",
            (key, "true" if value else "false", name),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def delete_camera(self, name: str) -> bool:
        cur = await self.conn.execute("DELETE FROM cameras WHERE name = ?", (name,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def camera_count(self) -> int:
        cur = await self.conn.execute("SELECT COUNT(*) FROM cameras")
        row = await cur.fetchone()
        return int(row[0])

    # ---------- camera health / reachability history ----------

    async def camera_health_open_states(self) -> dict[str, bool]:
        """The CURRENT (open, end_ts IS NULL) reachability state per camera.

        Loaded on boot so the tracker resumes from where it left off and never
        writes a duplicate interval for a state that has not actually changed.
        Defensive against a corrupt double-open: takes the latest per camera."""
        cur = await self.conn.execute(
            "SELECT camera, online FROM camera_health "
            "WHERE end_ts IS NULL ORDER BY start_ts"
        )
        rows = await cur.fetchall()
        return {r[0]: bool(r[1]) for r in rows}

    async def record_camera_health(self, camera: str, online: bool, ts: float) -> None:
        """Record a reachability TRANSITION: close any open interval for this
        camera, then open a new one in the new state. Idempotent-ish — callers
        only invoke this on an actual state change (see CameraHealthTracker)."""
        await self.conn.execute(
            "UPDATE camera_health SET end_ts = ? "
            "WHERE camera = ? AND end_ts IS NULL",
            (ts, camera),
        )
        await self.conn.execute(
            "INSERT INTO camera_health(camera, online, start_ts, end_ts) "
            "VALUES(?, ?, ?, NULL)",
            (camera, 1 if online else 0, ts),
        )
        await self.conn.commit()

    async def camera_health_intervals(
        self, since: float, until: float
    ) -> list[dict[str, Any]]:
        """Every interval overlapping [since, until], clipped to the window.
        An open interval (end_ts NULL) is treated as ending at ``until``."""
        cur = await self.conn.execute(
            "SELECT camera, online, start_ts, end_ts FROM camera_health "
            "WHERE start_ts < ? AND (end_ts IS NULL OR end_ts > ?) "
            "ORDER BY camera, start_ts",
            (until, since),
        )
        rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for camera, online, start_ts, end_ts in rows:
            s = max(float(start_ts), since)
            e = min(float(end_ts) if end_ts is not None else until, until)
            if e > s:
                out.append({"camera": camera, "online": bool(online),
                            "start": s, "end": e})
        return out

    async def prune_camera_health(self, before: float) -> int:
        """Drop CLOSED intervals that ended before ``before`` (retention).
        Never touches an open interval."""
        cur = await self.conn.execute(
            "DELETE FROM camera_health WHERE end_ts IS NOT NULL AND end_ts < ?",
            (before,),
        )
        await self.conn.commit()
        return cur.rowcount

    async def delete_camera_health(self, camera: str) -> None:
        """Purge a removed camera's history (called from delete_camera flows)."""
        await self.conn.execute("DELETE FROM camera_health WHERE camera = ?", (camera,))
        await self.conn.commit()

    async def set_camera_order(self, names: list[str]) -> None:
        """Assign display positions in the given order. Names not listed keep
        their relative order after the listed ones; unknown names are ignored
        (PUT /api/cameras/order contract)."""
        cur = await self.conn.execute("SELECT name FROM cameras ORDER BY position, name")
        current = [r["name"] for r in await cur.fetchall()]
        known = set(current)
        listed: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in known and name not in seen:
                listed.append(name)
                seen.add(name)
        ordered = listed + [n for n in current if n not in seen]
        for pos, name in enumerate(ordered, start=1):
            await self.conn.execute(
                "UPDATE cameras SET position = ? WHERE name = ?", (pos, name)
            )
        await self.conn.commit()

    # ---------- camera groups ----------

    @staticmethod
    def group_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "cameras": json.loads(row["cameras"]),
            "position": row["position"],
        }

    async def list_groups(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM camera_groups ORDER BY position, id")
        rows = await cur.fetchall()
        return [self.group_row_to_dict(r) for r in rows]

    async def get_group(self, group_id: int) -> Optional[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM camera_groups WHERE id = ?", (group_id,))
        row = await cur.fetchone()
        return self.group_row_to_dict(row) if row else None

    async def create_group(self, name: str, cameras: list[str]) -> Optional[dict[str, Any]]:
        """Insert a group at the end of the group order (position = max+1).
        Returns the created row, or None on a duplicate name."""
        try:
            cur = await self.conn.execute(
                """
                INSERT INTO camera_groups (name, cameras, position)
                VALUES (?, ?, (SELECT COALESCE(MAX(position), 0) + 1 FROM camera_groups))
                """,
                (name, json.dumps(cameras)),
            )
        except aiosqlite.IntegrityError:
            await self.conn.rollback()  # close the implicit transaction
            return None
        await self.conn.commit()
        return await self.get_group(int(cur.lastrowid))

    async def update_group(
        self,
        group_id: int,
        name: Optional[str] = None,
        cameras: Optional[list[str]] = None,
        position: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Partial update; `cameras` REPLACES the full ordered list. Returns
        the updated row, None when the id does not exist. Raises
        aiosqlite.IntegrityError on a rename to an existing name."""
        cols: list[str] = []
        vals: list[Any] = []
        if name is not None:
            cols.append("name = ?")
            vals.append(name)
        if cameras is not None:
            cols.append("cameras = ?")
            vals.append(json.dumps(cameras))
        if position is not None:
            cols.append("position = ?")
            vals.append(position)
        if cols:
            vals.append(group_id)
            try:
                await self.conn.execute(
                    f"UPDATE camera_groups SET {', '.join(cols)} WHERE id = ?", vals
                )
            except aiosqlite.IntegrityError:
                await self.conn.rollback()  # close the implicit transaction
                raise
            await self.conn.commit()
        return await self.get_group(group_id)

    async def delete_group(self, group_id: int) -> bool:
        cur = await self.conn.execute("DELETE FROM camera_groups WHERE id = ?", (group_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    # ---------- events ----------

    @staticmethod
    def event_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "frigate_id": row["frigate_id"],
            "camera": row["camera"],
            "label": row["label"],
            "count": row["count"],
            "score": row["score"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "has_clip": bool(row["has_clip"]),
            "has_snapshot": bool(row["has_snapshot"]),
            "zones": json.loads(row["zones"]),
            # Best detection box in detect-stream px ([x1,y1,x2,y2]) or [] when
            # unknown (doorbell/audio/legacy rows). Backs the reject foot-point.
            "box": json.loads(row["box"]),
            # All distinct detected classes on the camera detect list seen during
            # the event (multi-object). Legacy/empty rows fall back to the single
            # primary `label` so every event always exposes at least one label.
            "labels": json.loads(row["labels"]) or [row["label"]],
        }

    async def insert_event(
        self,
        frigate_id: Optional[str],
        camera: str,
        label: str,
        count: int,
        score: float,
        start_time: float,
        zones: list[str] | None = None,
        end_time: Optional[float] = None,
        has_clip: bool = False,
        has_snapshot: bool = False,
        box: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> int:
        # `labels` defaults to the single primary label so every row (doorbell /
        # camera-AI / engine) always carries at least its primary class.
        labels_list = labels if labels is not None else [label]
        cur = await self.conn.execute(
            """
            INSERT INTO events (frigate_id, camera, label, count, score, start_time,
                                end_time, has_clip, has_snapshot, zones, box, labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frigate_id,
                camera,
                label,
                count,
                score,
                start_time,
                end_time,
                int(has_clip),
                int(has_snapshot),
                json.dumps(zones or []),
                json.dumps(box or []),
                json.dumps(labels_list),
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def update_event(self, event_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        vals = []
        for key, value in fields.items():
            if key in ("zones", "box", "labels"):
                value = json.dumps(value)
            elif key in ("has_clip", "has_snapshot"):
                value = int(bool(value))
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(event_id)
        await self.conn.execute(f"UPDATE events SET {', '.join(cols)} WHERE id = ?", vals)
        await self.conn.commit()

    async def get_event(self, event_id: int) -> Optional[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = await cur.fetchone()
        return self.event_row_to_dict(row) if row else None

    async def get_event_by_frigate_id(self, frigate_id: str) -> Optional[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM events WHERE frigate_id = ?", (frigate_id,))
        row = await cur.fetchone()
        return self.event_row_to_dict(row) if row else None

    async def close_open_doorbell_events(self, max_duration_s: float) -> int:
        """Close doorbell rows left open by a restart, returning how many.

        A press holds its event open (end_time NULL) while the visitor is still
        there. If the process dies mid-visit the supervisor never closes it, and
        the restart watchdog's force-exit path skips `finally` handlers by
        design — so the sweep on the way back up is the only thing that reclaims
        these. An eternally-open row reads as "processing" forever in the UI.

        Closed at the hard cap the supervisor would have applied, but CLAMPED TO
        NOW: a press 3 s before the process died would otherwise be stamped
        ~117 s into the future, which renders as a fabricated 2-minute event, a
        timeline bar extending past the present, and a negative "ended_ago" that
        makes the API report a clip as still processing for a clip that is
        definitively never coming.
        """
        cur = await self.conn.execute(
            "UPDATE events SET end_time = MIN(start_time + ?, ?) "
            "WHERE end_time IS NULL AND frigate_id LIKE 'doorbell.%'",
            (max_duration_s, time.time()),
        )
        await self.conn.commit()
        return cur.rowcount or 0

    async def list_events(
        self,
        camera: Optional[str] = None,
        label: Optional[str] = None,
        after: Optional[float] = None,
        before: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where = []
        params: list[Any] = []
        if camera:
            where.append("camera = ?")
            params.append(camera)
        if label:
            where.append("label = ?")
            params.append(label)
        if after is not None:
            where.append("start_time >= ?")
            params.append(after)
        if before is not None:
            where.append("start_time <= ?")
            params.append(before)
        clause = f"WHERE {' AND '.join(where)}" if where else ""

        cur = await self.conn.execute(f"SELECT COUNT(*) FROM events {clause}", params)
        total = int((await cur.fetchone())[0])

        cur = await self.conn.execute(
            f"SELECT * FROM events {clause} ORDER BY start_time DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        rows = await cur.fetchall()
        return [self.event_row_to_dict(r) for r in rows], total

    async def delete_event(self, event_id: int) -> bool:
        cur = await self.conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def delete_all_events(self) -> int:
        """ADMIN: delete EVERY event row; return the number deleted. The caller
        is responsible for purging the associated snapshot + clip files (mirrors
        delete_event -> _purge_event_media, but for the whole table)."""
        cur = await self.conn.execute("DELETE FROM events")
        await self.conn.commit()
        return max(cur.rowcount, 0)

    async def prune_events_older_than(self, cutoff_epoch: float) -> list[int]:
        cur = await self.conn.execute("SELECT id FROM events WHERE start_time < ?", (cutoff_epoch,))
        ids = [int(r[0]) for r in await cur.fetchall()]
        if ids:
            await self.conn.execute("DELETE FROM events WHERE start_time < ?", (cutoff_epoch,))
            await self.conn.commit()
        return ids

    # ---------- detection suppressions (reject-to-suppress) ----------
    # A learned false positive: a per-camera/label foot-point (normalized 0..1
    # bottom-center) that mutes future matching detections BEFORE they open an
    # event. Written by rejecting an event ("not a real <object>"); read by the
    # engine in reload() and applied in process(). Complements exempt zones
    # (broad drawn areas) with pinpoint, type-specific suppression.

    @staticmethod
    def suppression_row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "camera": row["camera"],
            "label": row["label"],
            "foot_x": row["foot_x"],
            "foot_y": row["foot_y"],
            "has_thumb": bool(row["has_thumb"]),
            "created_at": row["created_at"],
        }

    async def insert_suppression(
        self,
        camera: str,
        label: str,
        foot_x: float,
        foot_y: float,
        has_thumb: bool = False,
    ) -> int:
        cur = await self.conn.execute(
            """INSERT INTO detection_suppressions
                   (camera, label, foot_x, foot_y, has_thumb, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (camera, label, float(foot_x), float(foot_y), int(has_thumb), time.time()),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def list_suppressions(self, camera: Optional[str] = None) -> list[dict[str, Any]]:
        if camera is None:
            cur = await self.conn.execute(
                "SELECT * FROM detection_suppressions ORDER BY created_at DESC"
            )
        else:
            cur = await self.conn.execute(
                "SELECT * FROM detection_suppressions WHERE camera = ? ORDER BY created_at DESC",
                (camera,),
            )
        return [self.suppression_row_to_dict(r) for r in await cur.fetchall()]

    async def delete_suppression(self, suppression_id: int) -> bool:
        cur = await self.conn.execute(
            "DELETE FROM detection_suppressions WHERE id = ?", (suppression_id,)
        )
        await self.conn.commit()
        return cur.rowcount > 0

    # ---------- push subscriptions ----------

    async def upsert_subscription(self, endpoint: str, subscription: dict[str, Any]) -> None:
        await self.conn.execute(
            """
            INSERT INTO push_subscriptions (endpoint, subscription, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET subscription = excluded.subscription
            """,
            (endpoint, json.dumps(subscription), time.time()),
        )
        await self.conn.commit()

    async def delete_subscription(self, endpoint: str) -> bool:
        cur = await self.conn.execute(
            "DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,)
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def list_subscriptions(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute("SELECT subscription FROM push_subscriptions")
        rows = await cur.fetchall()
        return [json.loads(r["subscription"]) for r in rows]

    # ---------- APNs device registrations (docs/push-architecture.md) ----------

    async def upsert_apns_device(
        self, device_token: str, device_name: str, key_b64: str, environment: str
    ) -> None:
        """Register (or re-register) an APNs device. `device_token` MUST already
        be lowercased hex (the router enforces it); latest key/name/env win."""
        await self.conn.execute(
            """
            INSERT INTO apns_devices (device_token, device_name, key_b64, environment, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(device_token) DO UPDATE SET
                device_name = excluded.device_name,
                key_b64     = excluded.key_b64,
                environment = excluded.environment
            """,
            (device_token, device_name, key_b64, environment, time.time()),
        )
        await self.conn.commit()

    async def delete_apns_device(self, device_token: str) -> bool:
        """Prune/unregister one APNs registration. Idempotent — False when the
        token was not registered. Used on 410/unregistered + bad_device_token."""
        cur = await self.conn.execute(
            "DELETE FROM apns_devices WHERE device_token = ?", (device_token,)
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def list_apns_devices(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT device_token, device_name, key_b64, environment, created_at "
            "FROM apns_devices ORDER BY created_at"
        )
        rows = await cur.fetchall()
        return [
            {
                "device_token": r["device_token"],
                "device_name": r["device_name"],
                "key_b64": r["key_b64"],
                "environment": r["environment"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ---------- VoIP (PushKit) device registrations (CallKit doorbell ring) ----

    async def upsert_voip_device(
        self, device_token: str, device_name: str, environment: str
    ) -> None:
        """Register (or re-register) a PushKit VoIP token. `device_token` MUST
        already be lowercased hex (the router enforces it); latest name/env win.
        No encryption key: the VoIP payload is minimal + not E2E-encrypted."""
        await self.conn.execute(
            """
            INSERT INTO voip_devices (device_token, device_name, environment, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(device_token) DO UPDATE SET
                device_name = excluded.device_name,
                environment = excluded.environment
            """,
            (device_token, device_name, environment, time.time()),
        )
        await self.conn.commit()

    async def delete_voip_device(self, device_token: str) -> bool:
        """Prune/unregister one VoIP registration. Idempotent — False when the
        token was not registered. Used on 410/unregistered + bad_device_token."""
        cur = await self.conn.execute(
            "DELETE FROM voip_devices WHERE device_token = ?", (device_token,)
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def list_voip_devices(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT device_token, device_name, environment, created_at "
            "FROM voip_devices ORDER BY created_at"
        )
        rows = await cur.fetchall()
        return [
            {
                "device_token": r["device_token"],
                "device_name": r["device_name"],
                "environment": r["environment"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ---------- users (RBAC) ----------

    @staticmethod
    def _user_public(row: aiosqlite.Row) -> dict[str, Any]:
        """Public user shape — NEVER includes password_hash."""
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "created_at": row["created_at"],
        }

    async def list_users(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY username"
        )
        return [self._user_public(r) for r in await cur.fetchall()]

    async def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        """Full row INCLUDING password_hash — for login verification only."""
        cur = await self.conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "role": row["role"],
            "created_at": row["created_at"],
        }

    async def get_user(self, user_id: int) -> Optional[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT id, username, role, created_at FROM users WHERE id = ?", (user_id,)
        )
        row = await cur.fetchone()
        return self._user_public(row) if row else None

    async def create_user(
        self, username: str, password_hash: str, role: str
    ) -> Optional[dict[str, Any]]:
        """Insert a user. Returns the public row, or None on a duplicate
        username (UNIQUE violation)."""
        try:
            cur = await self.conn.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, role, time.time()),
            )
        except aiosqlite.IntegrityError:
            await self.conn.rollback()
            return None
        await self.conn.commit()
        return await self.get_user(int(cur.lastrowid))

    async def update_user(
        self,
        user_id: int,
        password_hash: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        cols: list[str] = []
        vals: list[Any] = []
        if password_hash is not None:
            cols.append("password_hash = ?")
            vals.append(password_hash)
        if role is not None:
            cols.append("role = ?")
            vals.append(role)
        if cols:
            vals.append(user_id)
            await self.conn.execute(
                f"UPDATE users SET {', '.join(cols)} WHERE id = ?", vals
            )
            await self.conn.commit()
        return await self.get_user(user_id)

    async def delete_user(self, user_id: int) -> bool:
        cur = await self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def count_admin_users(self) -> int:
        """Number of DB users with role 'admin' (excludes the built-in env
        admin, which is not a row). Backs the last-admin demotion guard."""
        cur = await self.conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        row = await cur.fetchone()
        return int(row[0])

    # ---------- settings KV ----------

    async def get_setting(self, key: str) -> Optional[Any]:
        cur = await self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cur.fetchone()
        return json.loads(row["value"]) if row else None

    async def set_setting(self, key: str, value: Any) -> None:
        await self.conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, json.dumps(value)),
        )
        await self.conn.commit()
