"""Smoke suite for opening an event's clip (docs/CONTRACTS.md, "Recording + clips").

  - a clip the retention sweep deleted is NOT reported "ready" (has_clip is a
    DB flag the sweep never clears; the apps used to mount a player against a
    404), and says why
  - a ready clip still reads "ready"
  - POST /api/events/{id}/clip/retry queues a new cut ("processing"), refuses
    events that can never have one, still-open events and recording-off
    cameras, and reports "ready" when the file is already there

Usage: python backend/tests/clip_access_smoke.py  (needs backend deps)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
TMP = Path(tempfile.mkdtemp(prefix="vigilume-clipaccess-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


async def seed() -> dict[str, int]:
    cfg = Config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(cfg.db_path)
    await db.connect()
    now = time.time()
    for name, record in (("front", 1), ("norec", 0)):
        await db.conn.execute(
            "INSERT INTO cameras (name, friendly_name, model, ip, username, password,"
            " record_enabled, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (name, name.title(), "IP8M", "192.0.2.9", "u", "p", record, now),
        )
    await db.conn.commit()
    ids = {
        "pruned": await db.insert_event("native.pruned", "front", "car", 1, 0.9, now - 900,
                                        end_time=now - 880, has_clip=True),
        "ready": await db.insert_event("native.ready", "front", "car", 1, 0.9, now - 900,
                                       end_time=now - 880, has_clip=True),
        "failed": await db.insert_event("native.failed", "front", "car", 1, 0.9, now - 900,
                                        end_time=now - 880),
        "open": await db.insert_event("native.open", "front", "car", 1, 0.9, now - 60),
        "audio": await db.insert_event("audio.front.1", "front", "bark", 1, 0.9, now - 900,
                                       end_time=now - 880),
        "norec": await db.insert_event("native.norec", "norec", "car", 1, 0.9, now - 900,
                                       end_time=now - 880),
    }
    await db.update_event(ids["failed"], clip_error="ffmpeg exited 1")
    await db.close()
    return ids


class StubRecorder:
    """Stands in for the running recorder's clip queue."""

    def __init__(self) -> None:
        self.queued: list[str] = []

    def retry_clip(self, camera, fid, start, end) -> bool:
        self.queued.append(fid)
        return True

    def clip_pending(self, fid: str) -> bool:
        return fid in self.queued


def main() -> None:
    ids = asyncio.run(seed())
    from app.main import app

    with TestClient(app) as client:
        token = client.post("/api/auth/login", json={"password": "test-password"}).json()["token"]
        h = {"Authorization": f"Bearer {token}"}
        clip = app.state.media.clip_path(ids["ready"])
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"MP4" * 64)

        print("stale has_clip")
        e = client.get(f"/api/events/{ids['pruned']}", headers=h).json()
        check(e["clip_state"] != "ready" and e["has_clip"] is False,
              f"a clip deleted by retention is not 'ready' ({e['clip_state']})")
        check("retention" in (e.get("clip_error") or ""), "...and says why")
        e = client.get(f"/api/events/{ids['ready']}", headers=h).json()
        check(e["clip_state"] == "ready" and e["has_clip"] is True, "a present clip is 'ready'")
        e = client.get(f"/api/events/{ids['failed']}", headers=h).json()
        check(e["clip_state"] == "unavailable" and e["clip_error"] == "ffmpeg exited 1",
              "a failed clip carries its reason")

        print("retry")
        stub = StubRecorder()
        real = app.state.recorder
        app.state.recorder = stub
        try:
            r = client.post(f"/api/events/{ids['failed']}/clip/retry", headers=h)
            check(r.status_code == 200 and r.json()["clip_state"] == "processing"
                  and stub.queued == ["native.failed"], "retry queues a new cut")
            e = client.get(f"/api/events/{ids['failed']}", headers=h).json()
            check(e["clip_state"] == "processing", "the event now reads 'processing'")
            r = client.post(f"/api/events/{ids['failed']}/clip/retry", headers=h)
            check(r.status_code == 200 and stub.queued == ["native.failed"],
                  "a second tap while queued does not queue twice")
            r = client.post(f"/api/events/{ids['pruned']}/clip/retry", headers=h)
            check(r.status_code == 200 and "native.pruned" in stub.queued,
                  "a pruned clip can be cut again")
            r = client.post(f"/api/events/{ids['ready']}/clip/retry", headers=h)
            check(r.status_code == 200 and r.json()["clip_state"] == "ready"
                  and "native.ready" not in stub.queued, "a present clip is not re-cut")
            # Boot closes events a previous run left open, so reopen it now.
            import sqlite3
            with sqlite3.connect(Config().db_path) as raw:
                raw.execute("UPDATE events SET end_time = NULL WHERE id = ?", (ids["open"],))
            r = client.post(f"/api/events/{ids['open']}/clip/retry", headers=h)
            check(r.status_code == 409, "a still-open event is refused (cut when it ends)")
            r = client.post(f"/api/events/{ids['audio']}/clip/retry", headers=h)
            check(r.status_code == 400, "an event that never has a clip is refused")
            r = client.post(f"/api/events/{ids['norec']}/clip/retry", headers=h)
            check(r.status_code == 409, "a recording-off camera is refused")
            r = client.post(f"/api/events/{ids['failed']}/clip/retry")
            check(r.status_code == 401, "signed-in users only")
        finally:
            app.state.recorder = real
    print(f"\nALL {PASS} CHECKS PASSED (clip access)")


if __name__ == "__main__":
    main()
