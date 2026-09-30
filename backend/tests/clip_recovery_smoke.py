"""Event clips that vanished or never linked up — the recovery paths.

A clip job waits ~20 s after its event ends and lives only in memory. Three
ways an event lost its clip for good, each pinned here:

  1. A restart (an update, a crash, the restart watchdog) cancelled every
     queued job. Nothing cut those clips again, so the event read "no
     recording" forever. -> recover_clips() re-queues them after start-up.
  2. A clip still queued or being cut past the 45 s processing window read
     "unavailable" and then quietly appeared later. -> an event whose job is
     queued or running reads "processing".
  3. An event still open when the process died kept end_time NULL: it read
     "processing" forever and never got a clip. -> closed at start-up, so the
     recovery can cut it.

CPU-only; ffmpeg is stubbed (the real cut is recording_smoke.py's job).

    python backend/tests/clip_recovery_smoke.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

os.environ["ADMIN_PASSWORD"] = "test-password"
TMP = Path(tempfile.mkdtemp(prefix="vigilume-clip-recovery-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")

from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402
from app.native.recorder import CLIP_RECOVER_S, Recorder  # noqa: E402
from app.routers.events import CLIP_PROCESSING_WINDOW_S, _clip_state  # noqa: E402

PASS = 0
_failures: list[str] = []


def check(cond: bool, label: str) -> None:
    global PASS
    PASS += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


class FakeSettings:
    private_cameras: frozenset = frozenset()
    recording = {"continuous_days": 7, "event_days": 14, "snapshot_days": 14}

    def __init__(self, private: frozenset = frozenset()) -> None:
        self.private = private

    def is_private(self, camera: str) -> bool:
        return camera in self.private


def make_config(tag: str) -> Config:
    cfg = Config()
    cfg.data_dir = TMP / tag / "data"
    cfg.media_dir = TMP / tag / "media"
    return cfg


def make_segments(cfg: Config, camera: str, start: float, seconds: int) -> None:
    cam_dir = cfg.recordings_dir / camera
    t = start - start % 10
    while t < start + seconds:
        dt = datetime.fromtimestamp(t)
        p = cam_dir / dt.strftime("%Y-%m-%d") / dt.strftime("%H") / dt.strftime("%M.%S.ts")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x47" * 188)
        t += 10


async def add_camera(db: Database, name: str, *, record: bool = True) -> None:
    await db.conn.execute(
        "INSERT INTO cameras (name, friendly_name, model, ip, username, password,"
        " record_enabled, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (name, name.title(), "IP8M", "192.0.2.9", "u", "p", int(record), time.time()),
    )
    await db.conn.commit()


def new_recorder(cfg: Config, db: Database, settings=None) -> tuple[Recorder, list[str]]:
    """A recorder that is 'running' with a stub ffmpeg that writes the clip
    and records which events it cut."""
    rec = Recorder(cfg, db, settings or FakeSettings())
    rec._ffmpeg_path = "/fake/ffmpeg"
    rec._running = True
    rec.clip_delay_s = 0.05
    cut: list[str] = []

    async def run(args: list[str]) -> int:
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP4" * 16)
        cut.append(out.name)  # the part file: ".{event id}.part.mp4"
        return 0

    rec._run_ffmpeg = run
    return rec, cut


async def settle(rec: Recorder) -> None:
    for _ in range(100):
        if not rec._clip_tasks:
            return
        await asyncio.sleep(0.02)


async def pending_cases() -> None:
    print("\n1. a queued clip reads 'processing', however long it takes")
    cfg = make_config("pending")
    db = Database(cfg.data_dir / "a.db")
    await db.connect()
    now = time.time()
    make_segments(cfg, "front", now - 400, 400)
    await add_camera(db, "front")
    end = now - CLIP_PROCESSING_WINDOW_S - 60  # well past the processing window
    eid = await db.insert_event("native.slow", "front", "person", 1, 0.9, end - 30, end_time=end)
    rec, cut = new_recorder(cfg, db)
    rec.clip_delay_s = 30.0  # the job is stuck in its wait
    await rec.schedule_clip("front", "native.slow", end - 30, end)
    event = await db.get_event(eid)
    check(rec.clip_pending("native.slow"), "schedule_clip marks the event's clip pending")
    check(_clip_state(event, True, pending=True) == "processing",
          "an event past the processing window with its job queued reads 'processing'")
    check(_clip_state(event, True) == "unavailable",
          "(without the job it would read 'unavailable' — the old 'no recording')")
    check(_clip_state(event, False, pending=True) == "recording_disabled",
          "recording off still wins over a pending job")

    print("\n2. a restart cancels the job — and the next process cuts the clip")
    await rec.stop()
    check(not rec.clip_pending("native.slow") and cut == [],
          "stop() cancels the queued job: no clip, nothing pending")
    check((await db.get_event(eid))["has_clip"] is False, "the event has no clip after the restart")

    rec2, cut2 = new_recorder(cfg, db)
    landed: list[int] = []

    async def on_clip(event_id: int) -> None:
        landed.append(event_id)

    rec2.set_on_clip(on_clip)
    queued = await rec2.recover_clips(now)
    check(queued == 1, f"the new process's recovery queues it (queued {queued})")
    check(rec2.clip_pending("native.slow"), "and it reads 'processing' while it is cut")
    await settle(rec2)
    check((await db.get_event(eid))["has_clip"] is True and cut2 == [f".{eid}.part.mp4"],
          "the clip is cut and linked to the event")
    check(landed == [eid], "open clients are told the clip landed (event_update hook)")
    check(not rec2.clip_pending("native.slow"), "and nothing is left pending")
    check(await rec2.recover_clips(now) == 0, "a second pass finds nothing left to recover")
    await db.close()


async def skip_cases() -> None:
    print("\n3. the recovery leaves alone what it must")
    cfg = make_config("skips")
    db = Database(cfg.data_dir / "b.db")
    await db.connect()
    now = time.time()
    await add_camera(db, "front")
    await add_camera(db, "norec", record=False)
    await add_camera(db, "private")
    make_segments(cfg, "front", now - CLIP_RECOVER_S - 900, CLIP_RECOVER_S + 900)
    end = now - 600

    ids = {}
    ids["failed"] = await db.insert_event("native.failed", "front", "car", 1, 0.9, end - 20, end_time=end)
    await db.update_event(ids["failed"], clip_error="There was no 24/7 footage covering this moment.")
    ids["done"] = await db.insert_event("native.done", "front", "car", 1, 0.9, end - 20,
                                        end_time=end, has_clip=True)
    ids["doorbell"] = await db.insert_event("doorbell.front.1", "front", "person", 1, 0.9,
                                            end - 20, end_time=end)
    ids["audio"] = await db.insert_event("audio.front.1", "front", "bark", 1, 0.9, end - 20, end_time=end)
    ids["norec"] = await db.insert_event("native.norec", "norec", "car", 1, 0.9, end - 20, end_time=end)
    ids["private"] = await db.insert_event("native.private", "private", "car", 1, 0.9, end - 20, end_time=end)
    ids["gone"] = await db.insert_event("native.gone", "deleted_cam", "car", 1, 0.9, end - 20, end_time=end)
    ids["ancient"] = await db.insert_event("native.ancient", "front", "car", 1, 0.9,
                                           now - CLIP_RECOVER_S - 600, end_time=now - CLIP_RECOVER_S - 580)
    ids["fresh"] = await db.insert_event("native.fresh", "front", "car", 1, 0.9, now - 25, end_time=now - 5)
    ids["open"] = await db.insert_event("native.open", "front", "car", 1, 0.9, now - 25)
    ids["lost"] = await db.insert_event("native.lost", "front", "person", 1, 0.9, end - 20, end_time=end)

    rec, cut = new_recorder(cfg, db, FakeSettings(private=frozenset({"private"})))
    queued = await rec.recover_clips(now)
    await settle(rec)
    check(queued == 1 and cut == [f".{ids['lost']}.part.mp4"],
          f"exactly the one lost clip is recovered (cut {cut})")
    check((await db.get_event(ids["done"]))["has_clip"] is True, "  an event with a clip is untouched")
    for key, why in (
        ("failed", "a recorded failure (clip_error) is not retried"),
        ("doorbell", "doorbell events are the doorbell pipeline's"),
        ("audio", "audio events never have a clip"),
        ("norec", "a camera with recording off is skipped"),
        ("private", "a camera in privacy mode is skipped"),
        ("gone", "a deleted camera's event is skipped"),
        ("ancient", f"an event older than {CLIP_RECOVER_S // 3600} h is left alone"),
        ("fresh", "an event that just ended is left to its own job"),
        ("open", "an open event is left alone"),
    ):
        check((await db.get_event(ids[key]))["has_clip"] is False, f"  {why}")

    print("\n4. a job that crashes is not retried every pass")

    async def crash(args: list[str]) -> int:
        raise RuntimeError("encoder fell over")

    ids["crash"] = await db.insert_event("native.crash", "front", "car", 1, 0.9, end - 20, end_time=end)
    rec._run_ffmpeg = crash
    rec_log = logging.getLogger("app.native.recorder")
    rec_log.disabled = True  # the crash's traceback is expected here
    try:
        first = await rec.recover_clips(now)
        await settle(rec)
    finally:
        rec_log.disabled = False
    again = await rec.recover_clips(now)
    check(first == 1 and again == 0,
          f"tried once in this process, not every half hour (queued {first}, then {again})")
    await rec.stop()
    await db.close()


async def orphan_cases() -> None:
    print("\n5. events a dead process left open are closed at start-up")
    cfg = make_config("orphans")
    db = Database(cfg.data_dir / "c.db")
    await db.connect()
    now = time.time()
    await add_camera(db, "front")
    make_segments(cfg, "front", now - 900, 900)
    open_id = await db.insert_event("native.open", "front", "person", 1, 0.9, now - 600)
    closed_id = await db.insert_event("native.closed", "front", "person", 1, 0.9,
                                      now - 500, end_time=now - 480)
    door_id = await db.insert_event("doorbell.front.9", "front", "person", 1, 0.9, now - 600)

    n = await db.close_orphaned_native_events(estimate_s=60.0)
    check(n == 1, f"one open engine event is closed (closed {n})")
    row = await db.get_event(open_id)
    check(row["end_time"] == row["start_time"] + 60.0,
          "at start + 60 s — enough footage to cut a clip from")
    check((await db.get_event(closed_id))["end_time"] == now - 480,
          "an event that had ended keeps its real end")
    check((await db.get_event(door_id))["end_time"] is None,
          "a doorbell visit is left to the doorbell's own start-up close")
    check(_clip_state(await db.get_event(open_id), True) == "unavailable",
          "closed, it no longer reads 'processing' forever")

    rec, cut = new_recorder(cfg, db)
    await rec.recover_clips(now)
    await settle(rec)
    check((await db.get_event(open_id))["has_clip"] is True,
          "and the recovery then gives it a clip")
    await db.close()


async def main() -> int:
    await pending_cases()
    await skip_cases()
    await orphan_cases()
    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (event clip recovery)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
