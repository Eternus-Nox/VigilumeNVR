"""Smoke suite for Groups / camera ordering / notification-settings round-trip
(docs/CONTRACTS.md — "Camera groups", camera `position`, notifications test
route semantics).

NB this was once the "ntfy-removal addendum" and asserted that a persisted
notifications.ntfy block was silently stripped. ntfy is a SUPPORTED channel
again (it is how a self-hoster gets push without an Apple developer account),
so those assertions are now inverted: the block must round-trip and survive on
a legacy volume. It did NOT retire the APNs relay — that reversed the same day,
because ntfy cannot ring a doorbell.

Covers:
  - groups CRUD: create (201, position=max+1), duplicate name 409 (create and
    rename), partial PUT (name/cameras/position), cameras REPLACES the full
    ordered list, delete + 404s
  - unknown/deleted camera names in a group tolerated: stored raw, filtered
    out of API responses at read time, re-appear when the camera exists again
  - camera display order: GET /api/cameras ordered by position then name,
    PUT /api/cameras/order (partial list, unknown names ignored), new cameras
    append at the end
  - migration from schema v2 with existing data: position backfilled in rowid
    order, camera_groups created, user_version bumped
  - notifications.ntfy round-trips through GET/PUT and survives on a legacy
    /data volume (it is no longer stripped); ships disabled with no topic
  - POST /api/notifications/test: 400 exact no-subscriptions detail, 502 when
    every push send fails (fake failing webpush; detail names the error),
    200 {push_sent} on success (the test route is web-push-only)

Usage: python backend/tests/groups_smoke.py  (needs backend deps installed)
"""
from __future__ import annotations

import json
import os
import sqlite3
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
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
# Unroutable local port -> instant connection refusal, so go2rtc config
# syncs during camera CRUD never slow the suite down.
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-groups-smoke-"))

from fastapi.testclient import TestClient  # noqa: E402
from pywebpush import WebPushException  # noqa: E402

from app.main import app  # noqa: E402
from app.notify import push as push_module  # noqa: E402
from app.routers import cameras as cameras_router  # noqa: E402

# TEST-NET IPs are unroutable; keep the add/edit capability probe fast.
cameras_router._PROBE_TIMEOUT_S = 0.5

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


def login(client: TestClient) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"password": "test-password"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def add_camera(client: TestClient, headers: dict, name: str, ip: str,
               model: str = "IP8M-2779EW-AI") -> dict:
    resp = client.post("/api/cameras", headers=headers, json={
        "name": name, "friendly_name": name.replace("_", " ").title(),
        "model": model, "ip": ip, "username": "admin", "password": "pw",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def camera_names(client: TestClient, headers: dict) -> list[str]:
    return [c["name"] for c in client.get("/api/cameras", headers=headers).json()]


# ---------------- camera order ----------------


def order_checks(client: TestClient, headers: dict) -> None:
    print("camera order: position column + PUT /api/cameras/order")
    add_camera(client, headers, "cam_a", "192.0.2.101")
    add_camera(client, headers, "cam_b", "192.0.2.102")
    add_camera(client, headers, "cam_c", "192.0.2.103")
    check(camera_names(client, headers) == ["cam_a", "cam_b", "cam_c"],
          "fresh cameras listed in insertion order (position ascending)")

    resp = client.put("/api/cameras/order", headers=headers,
                      json={"names": ["cam_c", "cam_a"]})
    check(resp.status_code == 200, "PUT /api/cameras/order -> 200")
    check([c["name"] for c in resp.json()] == ["cam_c", "cam_a", "cam_b"],
          "partial order list: unlisted cameras keep relative order after listed")
    check(camera_names(client, headers) == ["cam_c", "cam_a", "cam_b"],
          "new order persisted in GET /api/cameras")

    resp = client.put("/api/cameras/order", headers=headers,
                      json={"names": ["nope", "cam_b"]})
    check(resp.status_code == 200
          and [c["name"] for c in resp.json()] == ["cam_b", "cam_c", "cam_a"],
          "unknown names in the order list ignored")

    add_camera(client, headers, "cam_d", "192.0.2.104")
    check(camera_names(client, headers) == ["cam_b", "cam_c", "cam_a", "cam_d"],
          "newly added camera appends at the end of the order")

    # Edits must not move a camera (upsert never touches position).
    resp = client.put("/api/cameras/cam_c", headers=headers, json={
        "name": "cam_c", "friendly_name": "Cam C Renamed",
        "model": "IP8M-2779EW-AI", "ip": "192.0.2.103",
        "username": "", "password": "",
    })
    check(resp.status_code == 200, "PUT camera edit still works")
    check(camera_names(client, headers) == ["cam_b", "cam_c", "cam_a", "cam_d"],
          "editing a camera keeps its position")


# ---------------- groups CRUD ----------------


def groups_checks(client: TestClient, headers: dict) -> None:
    print("groups: CRUD, 409s, replacement, unknown names tolerated")
    check(client.get("/api/groups", headers=headers).json() == [],
          "GET /api/groups starts empty")

    resp = client.post("/api/groups", headers=headers,
                       json={"name": "Yard", "cameras": ["cam_a", "cam_c"]})
    check(resp.status_code == 201, "POST group -> 201")
    yard = resp.json()
    check(yard["name"] == "Yard" and yard["position"] == 1
          and yard["cameras"] == ["cam_a", "cam_c"],
          "created group keeps the given camera display order, position=1")

    resp = client.post("/api/groups", headers=headers, json={"name": "Yard"})
    check(resp.status_code == 409, "duplicate group name -> 409")

    resp = client.post("/api/groups", headers=headers, json={"name": "Indoors"})
    check(resp.status_code == 201 and resp.json()["position"] == 2
          and resp.json()["cameras"] == [],
          "second group: cameras optional (defaults []), position=max+1")
    indoors = resp.json()

    groups = client.get("/api/groups", headers=headers).json()
    check([g["name"] for g in groups] == ["Yard", "Indoors"],
          "GET /api/groups ordered by position")

    # Reorder groups via PUT {position}.
    resp = client.put(f"/api/groups/{indoors['id']}", headers=headers,
                      json={"position": 0})
    check(resp.status_code == 200 and resp.json()["position"] == 0,
          "PUT group position -> 200")
    groups = client.get("/api/groups", headers=headers).json()
    check([g["name"] for g in groups] == ["Indoors", "Yard"],
          "group order follows updated positions")

    # cameras REPLACES the full ordered list; unknown name tolerated.
    resp = client.put(f"/api/groups/{yard['id']}", headers=headers,
                      json={"cameras": ["cam_c", "cam_a", "ghost"]})
    check(resp.status_code == 200, "PUT group cameras replacement -> 200")
    check(resp.json()["cameras"] == ["cam_c", "cam_a"],
          "unknown camera name filtered from the response (not an error)")
    raw = json.loads(sqlite3.connect(TMP / "fresh" / "nvr.db").execute(
        "SELECT cameras FROM camera_groups WHERE id = ?", (yard["id"],)
    ).fetchone()[0])
    check(raw == ["cam_c", "cam_a", "ghost"],
          "unknown name stored raw (filtering happens at read time)")

    # When a camera by that name appears, it re-materializes in the group.
    add_camera(client, headers, "ghost", "192.0.2.105")
    groups = {g["name"]: g for g in client.get("/api/groups", headers=headers).json()}
    check(groups["Yard"]["cameras"] == ["cam_c", "cam_a", "ghost"],
          "stored unknown name re-appears once the camera exists")

    # Reorder within a group = PUT with the new order.
    resp = client.put(f"/api/groups/{yard['id']}", headers=headers,
                      json={"cameras": ["ghost", "cam_a", "cam_c"]})
    check(resp.json()["cameras"] == ["ghost", "cam_a", "cam_c"],
          "group camera reorder via full-list replacement")

    # Deleted camera tolerated at read time.
    resp = client.delete("/api/cameras/ghost", headers=headers)
    check(resp.status_code == 204, "DELETE camera ghost -> 204")
    groups = {g["name"]: g for g in client.get("/api/groups", headers=headers).json()}
    check(groups["Yard"]["cameras"] == ["cam_a", "cam_c"],
          "deleted camera filtered from group responses, order intact")

    # Rename: conflict + success; 404s.
    resp = client.put(f"/api/groups/{indoors['id']}", headers=headers,
                      json={"name": "Yard"})
    check(resp.status_code == 409, "rename to an existing group name -> 409")
    resp = client.put(f"/api/groups/{indoors['id']}", headers=headers,
                      json={"name": "Inside"})
    check(resp.status_code == 200 and resp.json()["name"] == "Inside",
          "rename -> 200 with updated name")
    check(client.put("/api/groups/9999", headers=headers,
                     json={"name": "X"}).status_code == 404,
          "PUT unknown group id -> 404")

    resp = client.delete(f"/api/groups/{indoors['id']}", headers=headers)
    check(resp.status_code == 204, "DELETE group -> 204")
    check(client.delete(f"/api/groups/{indoors['id']}", headers=headers).status_code == 404,
          "second DELETE -> 404")
    check([g["name"] for g in client.get("/api/groups", headers=headers).json()] == ["Yard"],
          "deleted group gone from the list")


# ---------------- notifications: test route + legacy ntfy ----------------


def notification_checks(client: TestClient, headers: dict) -> None:
    print("notifications: test-route semantics + legacy ntfy dropped on PUT")
    resp = client.post("/api/notifications/test", headers=headers)
    check(resp.status_code == 400
          and resp.json()["detail"]
          == "No push subscriptions registered — enable notifications on a device first",
          "no subscriptions -> 400 with contract detail")

    resp = client.post("/api/notifications/subscribe", headers=headers, json={
        "endpoint": "https://push.example.net/sub-1",
        "keys": {"p256dh": "fake-p256dh", "auth": "fake-auth"},
    })
    check(resp.status_code == 204, "push subscribe -> 204")

    original_webpush = push_module.webpush

    def failing_webpush(**kwargs):
        raise WebPushException("boom: fake push service outage")

    push_module.webpush = failing_webpush
    try:
        resp = client.post("/api/notifications/test", headers=headers)
        check(resp.status_code == 502, "all sends failed -> 502")
        detail = resp.json()["detail"]
        check(detail.startswith("All push sends failed:") and "boom" in detail,
              "502 detail names the first push error")
    finally:
        push_module.webpush = original_webpush

    push_module.webpush = lambda **kwargs: None
    try:
        resp = client.post("/api/notifications/test", headers=headers)
        check(resp.status_code == 200, "successful send -> 200")
        body = resp.json()
        check(body == {"push_sent": 1},
              "200 body is exactly {push_sent: 1} (no ntfy_sent field)")
    finally:
        push_module.webpush = original_webpush

    # ntfy is a SUPPORTED channel again — it is how a self-hoster gets push
    # with no Apple developer account, and it is what let the APNs push relay
    # be retired. These assertions used to prove the opposite (that a persisted
    # ntfy block was silently stripped on load and on PUT); they now prove the
    # block round-trips, which is what the UI depends on.
    settings = client.get("/api/settings", headers=headers).json()
    check("ntfy" in settings["notifications"], "GET /api/settings HAS an ntfy block")
    check(settings["notifications"]["ntfy"]["enabled"] is False
          and settings["notifications"]["ntfy"]["topic"] == "",
          "ntfy ships disabled with NO default topic (the topic is a shared secret)")
    settings["notifications"]["ntfy"] = {
        "enabled": True, "server": "https://ntfy.sh", "topic": "t9x_QZ-4", "auth_token": "tk_x",
    }
    settings["notifications"]["cooldown_seconds"] = 45
    resp = client.put("/api/settings", headers=headers, json=settings)
    check(resp.status_code == 200, "PUT settings with an ntfy block -> 200")
    body = resp.json()["notifications"]
    check(body["ntfy"]["enabled"] is True and body["ntfy"]["topic"] == "t9x_QZ-4"
          and body["cooldown_seconds"] == 45,
          "ntfy block is PERSISTED on PUT (no longer stripped as legacy)")
    # And it must survive into the stored blob, not just the API response —
    # _strip_legacy used to pop it on every save.
    stored = json.loads(sqlite3.connect(TMP / "fresh" / "nvr.db").execute(
        "SELECT value FROM settings WHERE key = 'app_settings'"
    ).fetchone()[0])
    check(stored["notifications"]["ntfy"]["topic"] == "t9x_QZ-4",
          "persisted settings blob KEEPS the ntfy key")


# ---------------- migration from schema v2 ----------------

_V2_SCHEMA = """
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
    created_at      REAL NOT NULL
);
CREATE TABLE events (
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
    zones        TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE push_subscriptions (
    endpoint     TEXT PRIMARY KEY,
    subscription TEXT NOT NULL,
    created_at   REAL NOT NULL
);
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_LEGACY_SETTINGS = {
    "notifications": {
        "enabled": True,
        "labels": ["person"],
        "cooldown_seconds": 30,
        "min_score": 0.5,
        "ntfy": {"enabled": True, "server": "https://ntfy.sh",
                 "topic": "legacy-topic", "auth_token": "tok"},
    },
    "recording": {"continuous_days": 3, "event_days": 5, "snapshot_days": 5},
    "detection": {"audio_events": False, "audio_labels": ["bark"]},
    "system": {"public_url": ""},
}


def _build_v2_db(db_path: Path) -> None:
    """A realistic pre-addendum /data: schema v2, two cameras whose rowid
    order deliberately disagrees with both name and created_at order, and a
    persisted settings blob containing the removed ntfy block."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_V2_SCHEMA)
        rows = [
            # (rowid order) zebra first despite later created_at + later name
            ("zebra_cam", "Zebra Cam", "IP8M-2779EW-AI", "192.0.2.201", 2000.0),
            ("alpha_cam", "Alpha Cam", "AD410", "192.0.2.202", 1000.0),
        ]
        for name, friendly, model, ip, created in rows:
            conn.execute(
                """
                INSERT INTO cameras (name, friendly_name, model, ip, username,
                                     password, capabilities, created_at)
                VALUES (?, ?, ?, ?, 'admin', 'pw', '{}', ?)
                """,
                (name, friendly, model, ip, created),
            )
        conn.execute(
            "INSERT INTO settings (key, value) VALUES ('app_settings', ?)",
            (json.dumps(_LEGACY_SETTINGS),),
        )
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    finally:
        conn.close()


def migration_checks() -> None:
    print("migration: schema v2 + existing data -> v4 (position, groups, url overrides, ntfy drop)")
    data_dir = TMP / "migrated"
    db_path = data_dir / "nvr.db"
    _build_v2_db(db_path)
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["MEDIA_DIR"] = str(TMP / "migrated-media")
    os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "migrated-go2rtc-config")

    with TestClient(app) as client:
        headers = login(client)

        # An old volume that still has an ntfy block loads clean AND KEEPS it:
        # ntfy is a supported channel again, so a hoster who configured it
        # before it was removed gets their config back rather than wiped.
        resp = client.get("/api/settings", headers=headers)
        check(resp.status_code == 200, "GET /api/settings on legacy data -> 200 (no 500)")
        notifications = resp.json()["notifications"]
        check(notifications["ntfy"]["topic"] == "legacy-topic"
              and notifications["ntfy"]["enabled"] is True,
              "a persisted ntfy block SURVIVES the load (no longer stripped as legacy)")
        check(notifications["cooldown_seconds"] == 30
              and notifications["labels"] == ["person"],
              "other persisted notification settings survive")

        # The test route still only counts WEB PUSH subscribers — it does not
        # yet fan out to ntfy (see the ntfy Test button, not built here).
        resp = client.post("/api/notifications/test", headers=headers)
        check(resp.status_code == 400,
              "test route is web-push-only: no subscriptions -> 400")

        # A settings round-trip on the legacy volume KEEPS the ntfy key (raw
        # blob checked after the client closes, below).
        resp = client.put("/api/settings", headers=headers,
                          json=client.get("/api/settings", headers=headers).json())
        check(resp.status_code == 200
              and resp.json()["notifications"]["ntfy"]["topic"] == "legacy-topic",
              "settings PUT round-trip on legacy /data -> 200, ntfy preserved")

        names = camera_names(client, headers)
        check(names == ["zebra_cam", "alpha_cam"],
              "migrated positions follow rowid order (not created_at, not name)")

        # Groups work on a migrated DB.
        resp = client.post("/api/groups", headers=headers,
                           json={"name": "Migrated", "cameras": ["alpha_cam", "gone"]})
        check(resp.status_code == 201 and resp.json()["cameras"] == ["alpha_cam"],
              "group create on migrated DB; unknown name filtered")

        # Reorder + append-on-add work on migrated rows.
        resp = client.put("/api/cameras/order", headers=headers,
                          json={"names": ["alpha_cam", "zebra_cam"]})
        check([c["name"] for c in resp.json()] == ["alpha_cam", "zebra_cam"],
              "PUT /api/cameras/order works on migrated rows")
        add_camera(client, headers, "new_cam", "192.0.2.203")
        check(camera_names(client, headers) == ["alpha_cam", "zebra_cam", "new_cam"],
              "camera added post-migration appends at the end")

    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        check(version >= 6,
              "user_version bumped to 6+ (position/groups + native urls + RBAC + "
              "detect_objects record-only backfill; v7 adds apns_devices)")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(cameras)").fetchall()}
        check({"main_url", "sub_url"} <= cols,
              "v4 migration added main_url/sub_url columns")
        user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        check({"id", "username", "password_hash", "role", "created_at"} <= user_cols,
              "v5 migration created the users table")
        positions = dict(conn.execute("SELECT name, position FROM cameras").fetchall())
        check(positions == {"alpha_cam": 1, "zebra_cam": 2, "new_cam": 3},
              "positions persisted contiguously after reorder + add")
        group_count = conn.execute("SELECT COUNT(*) FROM camera_groups").fetchone()[0]
        check(group_count == 1, "camera_groups table created and populated")
        stored = json.loads(conn.execute(
            "SELECT value FROM settings WHERE key = 'app_settings'"
        ).fetchone()[0])
        check(stored["notifications"]["ntfy"]["topic"] == "legacy-topic"
              and stored["notifications"]["cooldown_seconds"] == 30,
              "legacy volume re-persisted WITHOUT the ntfy key, values intact")
    finally:
        conn.close()


# ---------------- main ----------------


def fresh_checks() -> None:
    data_dir = TMP / "fresh"
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["MEDIA_DIR"] = str(TMP / "fresh-media")
    os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "fresh-go2rtc-config")
    with TestClient(app) as client:
        headers = login(client)
        order_checks(client, headers)
        groups_checks(client, headers)
        notification_checks(client, headers)


def main() -> None:
    fresh_checks()
    migration_checks()
    print(f"ALL PASSED ({PASS} checks)")


if __name__ == "__main__":
    main()
