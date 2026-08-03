"""Doorbell hold-open recording smoke.

A button press used to write a zero-length, clipless marker row
(start_time == end_time, has_clip=False). It now OPENS an event, holds it open
while a person is still in frame, then closes it and schedules a clip cut from
the 24/7 segments. Clip length is derived purely from the event's start/end
times, so holding the row open IS the whole feature — the recorder is untouched.

Everything risky here is loop-termination logic that fails in the direction of
recording FOREVER, which on a security system is worse than not recording at
all. Specifically:

  - `current_count()` FLOORS ITS RESULT AT 1 as a defence for engine-produced
    events, so it can never report zero. A presence loop built on it never
    terminates. Section 2 pins that the raw `counts` dict is what is read, and
    section 1 pins the floor itself so the trap stays documented in a test.
  - the hard cap is the only thing standing between a stuck track and an
    unbounded clip + a permanently open row.
  - Privacy Mode is a capture kill switch that can be flipped MID-visit.
  - the restart watchdog force-exits (os._exit), which skips `finally`, so the
    boot sweep is the only thing that reclaims rows orphaned that way.

Time is compressed by monkeypatching the module's window constants; the suite
runs in a couple of seconds and touches no camera, GPU or network.

Usage: python backend/tests/doorbell_recording_smoke.py
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

TMP = Path(tempfile.mkdtemp(prefix="sentinel-doorbell-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import app.events_pipeline as ep  # noqa: E402
from app.auth import AuthService  # noqa: E402
from app.db import Database  # noqa: E402
from app.events_pipeline import EventsPipeline  # noqa: E402
from app.settings_store import SettingsStore  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"  FAIL: {msg}", flush=True)
        # os._exit, not SystemExit: aiosqlite's connection worker is a
        # NON-DAEMON thread, so unwinding out of a failed check without closing
        # the Database wedges the interpreter on the atexit join for minutes.
        # That is the same failure mode this suite's subject (the restart
        # watchdog) exists to bound — no reason to also suffer it in CI.
        os._exit(1)
    PASS += 1
    print(f"  ok: {msg}")


class FakeMedia:
    async def latest_jpg(self, camera, height=None):
        return None          # no snapshot — irrelevant to the recording window

    async def event_snapshot(self, fid, retries=3):
        return None

    async def detect_dims(self, camera):
        return None


class FakeWS:
    def __init__(self) -> None:
        self.msgs: list[dict] = []

    async def broadcast(self, msg):
        self.msgs.append(msg)


class FakePush:
    class _R:
        sent = 0
        attempted = 0

    async def send_to_all(self, payload):
        return FakePush._R()


class FakeRecorder:
    """Captures schedule_clip calls — the only thing the supervisor asks of the
    recorder, and the thing that must NOT happen under Privacy Mode."""

    def __init__(self) -> None:
        self.clips: list[tuple[str, str, float, float]] = []

    async def schedule_clip(self, camera, frigate_id, start_time, end_time):
        self.clips.append((camera, frigate_id, start_time, end_time))


# Compressed windows. The real values (5s absence / 120s cap / 10s floor) would
# make this suite take minutes; the LOGIC under test is the comparisons, not the
# magnitudes.
FAST = {
    "_DOORBELL_ABSENCE_S": 0.30,
    "DOORBELL_MAX_S": 1.20,
    "_DOORBELL_MIN_S": 0.20,
    "_DOORBELL_POLL_S": 0.05,
}


def _apply_fast_windows() -> None:
    for name, value in FAST.items():
        setattr(ep, name, value)


async def _make_pipeline(record_enabled: bool = True):
    dbpath = TMP / f"db-{record_enabled}-{time.time_ns()}" / "nvr.db"
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    db = Database(dbpath)
    await db.connect()
    await db.upsert_camera({
        "name": "front_door", "friendly_name": "Front Door", "model": "AD410",
        "ip": "127.0.0.1", "username": "u", "password": "p",
        "detect_objects": ["person"], "detect_width": 704, "detect_height": 480,
        "detect_fps": 5, "detect_enabled": True, "record_enabled": record_enabled,
        "detect_mode": "always", "created_at": time.time(),
    })
    settings = SettingsStore(db)
    await settings.load()
    ws = FakeWS()
    auth = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=1)
    pipeline = EventsPipeline(
        db, FakeMedia(), ws, FakePush(), settings, auth, TMP / "snaps",
    )
    rec = FakeRecorder()
    pipeline.set_recorder(rec)
    return db, settings, ws, pipeline, rec


def _engage_privacy(settings, camera: str) -> None:
    """Flip the capture gate the way privacy.refresh() does — an atomic
    whole-set rebind, which is exactly how it happens live mid-visit."""
    settings.set_private_cameras(frozenset({camera}))


async def _open_row(db, fid: str, start: float) -> int:
    """An OPEN doorbell row, exactly as handle_doorbell writes one. The
    supervisor updates and re-reads it, so it has to actually exist."""
    return await db.insert_event(
        frigate_id=fid, camera="front_door", label="doorbell",
        count=1, score=1.0, start_time=start, end_time=None,
    )


# ---------------------------------------------------------------------------
# 1. the count trap
# ---------------------------------------------------------------------------
async def count_trap() -> None:
    print("\n1. the current_count() floor — why the loop must read `counts` directly")
    db, _, _, pipeline, _ = await _make_pipeline()
    # No count has ever been recorded for this pair.
    check(pipeline.current_count("front_door", "person") == 1,
          "current_count() returns 1 for an UNKNOWN pair (documented cold-cache floor)")
    pipeline.update_count("front_door", "person", 0)
    check(pipeline.current_count("front_door", "person") == 1,
          "current_count() STILL returns 1 after an explicit zero — it can never report absence")
    check(pipeline.counts.get(("front_door", "person")) == 0,
          "the raw counts dict DOES record zero — this is what the supervisor must read")
    await db.close()


# ---------------------------------------------------------------------------
# 2. the recording window
# ---------------------------------------------------------------------------
async def window_cases() -> None:
    print("\n2. the hold-open window")
    _apply_fast_windows()

    # --- a visitor who arrives and then leaves ---
    db, _, ws, pipeline, rec = await _make_pipeline()
    start = time.time()
    eid = await _open_row(db, "doorbell.1", start)
    pipeline.update_count("front_door", "person", 1)
    task = asyncio.create_task(
        pipeline._doorbell_recording("front_door", "doorbell.1", eid, start))
    await asyncio.sleep(0.45)
    check(not task.done(), "the event stays OPEN while a person is in frame")
    pipeline.update_count("front_door", "person", 0)
    await asyncio.wait_for(task, timeout=3.0)
    check(len(rec.clips) == 1, "one clip is scheduled once the visitor leaves")
    cam, fid, s, e = rec.clips[0]
    check(cam == "front_door" and fid == "doorbell.1", "clip is scheduled for the pressing camera + fid")
    check(e > s, "the clip window has real duration (end > start)")
    check(e - s < ep.DOORBELL_MAX_S, "a visitor who leaves does NOT run to the cap")
    check(any(m["type"] == "event_end" for m in ws.msgs),
          "closing the event broadcasts event_end so clients pick up the duration")
    await db.close()

    # --- a visitor who never leaves: the cap is the only backstop ---
    db, _, _, pipeline, rec = await _make_pipeline()
    start = time.time()
    eid = await _open_row(db, "doorbell.2", start)
    pipeline.update_count("front_door", "person", 1)   # never cleared
    await asyncio.wait_for(
        pipeline._doorbell_recording("front_door", "doorbell.2", eid, start), timeout=5.0,
    )
    check(len(rec.clips) == 1, "a permanently-present person still terminates (hard cap)")
    _, _, s2, e2 = rec.clips[0]
    check(e2 - s2 >= ep.DOORBELL_MAX_S * 0.9,
          f"the capped clip runs to ~the cap ({e2 - s2:.2f}s vs cap {ep.DOORBELL_MAX_S}s)")
    check(e2 - s2 < ep.DOORBELL_MAX_S * 2,
          "the cap actually BOUNDS the window rather than merely delaying it")
    await db.close()

    # --- nobody ever detected: the floor still yields a usable clip ---
    db, _, _, pipeline, rec = await _make_pipeline()
    start = time.time()
    eid = await _open_row(db, "doorbell.3", start)
    pipeline.update_count("front_door", "person", 0)
    await asyncio.wait_for(
        pipeline._doorbell_recording("front_door", "doorbell.3", eid, start), timeout=5.0,
    )
    check(len(rec.clips) == 1, "a press with NO person detected still schedules a clip")
    _, _, s3, e3 = rec.clips[0]
    check(e3 - s3 >= ep._DOORBELL_MIN_S,
          f"the clip respects the minimum floor ({e3 - s3:.2f}s >= {ep._DOORBELL_MIN_S}s)")
    await db.close()


# ---------------------------------------------------------------------------
# 3. privacy mode mid-visit
# ---------------------------------------------------------------------------
async def privacy_case() -> None:
    print("\n3. Privacy Mode engaged MID-visit")
    _apply_fast_windows()
    db, settings, _, pipeline, rec = await _make_pipeline()
    start = time.time()
    eid = await _open_row(db, "doorbell.4", start)
    pipeline.update_count("front_door", "person", 1)
    task = asyncio.create_task(
        pipeline._doorbell_recording("front_door", "doorbell.4", eid, start))
    await asyncio.sleep(0.20)
    _engage_privacy(settings, "front_door")
    await asyncio.wait_for(task, timeout=3.0)
    check(rec.clips == [],
          "NO clip is scheduled when privacy engages mid-visit (capture kill switch)")
    row = await db.get_event(eid)
    check(row["end_time"] is not None,
          "the event is still CLOSED — privacy stops the recording, it does not orphan the row")
    check(row["has_clip"] in (0, False), "and the row never claims a clip")
    await db.close()


# ---------------------------------------------------------------------------
# 4. handle_doorbell opens the row (and when it must not)
# ---------------------------------------------------------------------------
async def press_cases() -> None:
    print("\n4. handle_doorbell row shape")
    _apply_fast_windows()

    db, _, _, pipeline, _ = await _make_pipeline(record_enabled=True)
    await pipeline.handle_doorbell("front_door")
    events, total = await db.list_events(camera="front_door")
    check(total == 1, "a press creates exactly one event row")
    row = events[0]
    check(row["label"] == "doorbell", "the row is labelled 'doorbell'")
    check(str(row["frigate_id"]).startswith("doorbell."),
          "the row keeps the doorbell. prefix (snapshot still served from disk)")
    check(row["end_time"] is None,
          "with recording ON the row is left OPEN (end_time NULL) for the supervisor")
    await pipeline.shutdown()          # cancels the supervisor
    await asyncio.sleep(0.05)
    closed = await db.get_event(int(row["id"]))
    check(closed["end_time"] is not None,
          "cancelling the supervisor (shutdown) still CLOSES the row — never left open")
    await db.close()

    # recording off: nothing can be cut, so do not leave it open
    db, _, _, pipeline, rec = await _make_pipeline(record_enabled=False)
    await pipeline.handle_doorbell("front_door")
    events, _ = await db.list_events(camera="front_door")
    row = events[0]
    check(row["end_time"] is not None,
          "with recording OFF the row is closed immediately (no clip is possible)")
    check(rec.clips == [], "with recording OFF no clip is scheduled")
    await db.close()


# ---------------------------------------------------------------------------
# 5. the boot sweep
# ---------------------------------------------------------------------------
async def sweep_case() -> None:
    print("\n5. boot sweep for rows orphaned by a force-exit")
    db, _, _, _, _ = await _make_pipeline()
    now = time.time()
    # Started 5 minutes ago, so the 120s cap binds rather than the now-clamp
    # (the clamp is exercised separately in section 8).
    old_start = now - 300
    open_id = await db.insert_event(
        frigate_id=f"doorbell.{int(now * 1000)}", camera="front_door", label="doorbell",
        count=1, score=1.0, start_time=old_start, end_time=None,
    )
    det_id = await db.insert_event(
        frigate_id="native.99", camera="front_door", label="person",
        count=1, score=0.9, start_time=now, end_time=None,
    )
    swept = await db.close_open_doorbell_events(120.0)
    check(swept == 1, "the sweep closes exactly the one orphaned doorbell row")
    doorbell_row = await db.get_event(open_id)
    check(doorbell_row["end_time"] is not None, "the orphaned doorbell row now has an end_time")
    check(abs(doorbell_row["end_time"] - (old_start + 120.0)) < 0.01,
          "a long-running orphan closes at the hard cap — the longest it could have been")
    det_row = await db.get_event(det_id)
    check(det_row["end_time"] is None,
          "an OPEN DETECTION event is left alone (the engine owns those, not the sweep)")
    check(await db.close_open_doorbell_events(120.0) == 0, "the sweep is idempotent")
    await db.close()


# ---------------------------------------------------------------------------
# 6. the clip/snapshot predicate split
# ---------------------------------------------------------------------------
def predicate_split() -> None:
    print("\n6. _is_synthetic (snapshot) vs _never_has_clip (clip) are now DISTINCT")
    from app.routers.events import _clip_state, _is_synthetic, _never_has_clip

    doorbell = {"frigate_id": "doorbell.1"}
    check(_is_synthetic(doorbell) is True,
          "a doorbell event is STILL synthetic — its snapshot comes from disk, unchanged")
    check(_never_has_clip(doorbell) is False,
          "...but it is no longer clipless: the clip route must serve it")
    for fid in ("audio.1", "cameraai.1", ""):
        row = {"frigate_id": fid}
        check(_is_synthetic(row) and _never_has_clip(row),
              f"'{fid or '(empty)'}' remains BOTH synthetic and clipless")
    check(not _is_synthetic({"frigate_id": "native.1"})
          and not _never_has_clip({"frigate_id": "native.1"}),
          "a real engine event is neither")
    now = time.time()
    check(_clip_state({"frigate_id": "doorbell.1", "has_clip": False, "end_time": None}, True)
          == "processing",
          "a doorbell visit in progress reads as 'processing', not 'recording_disabled'")


# ---------------------------------------------------------------------------
# 7. the exit paths that must NOT leave a row open or claim a phantom clip
# ---------------------------------------------------------------------------
async def robustness_cases() -> None:
    print("\n7. no exit path leaves the row open or advertises a clip that isn't coming")
    _apply_fast_windows()
    from app.routers.events import _clip_state, _never_has_clip

    # --- a raising DB must still close the row ---
    db, _, _, pipeline, rec = await _make_pipeline()
    start = time.time()
    eid = await _open_row(db, "doorbell.7", start)
    boom = {"n": 0}
    real_is_private = pipeline._settings.is_private

    def explode(camera):
        boom["n"] += 1
        if boom["n"] > 2:
            raise RuntimeError("simulated storage fault")
        return real_is_private(camera)

    pipeline._settings.is_private = explode  # type: ignore[method-assign]
    await asyncio.wait_for(
        pipeline._doorbell_recording("front_door", "doorbell.7", eid, start), timeout=5.0,
    )
    pipeline._settings.is_private = real_is_private  # type: ignore[method-assign]
    row = await db.get_event(eid)
    check(row["end_time"] is not None,
          "an unexpected exception still CLOSES the row (never 'processing' forever)")
    check(rec.clips == [], "and schedules no clip")
    check(_never_has_clip(row), "the errored row reports 'no clip', not a recorder failure")
    check("front_door" not in pipeline._doorbell_recording_cams,
          "the in-flight marker is released even on the error path")
    await db.close()

    # --- repeat presses during one visit ---
    db, _, _, pipeline, rec = await _make_pipeline()
    pipeline.update_count("front_door", "person", 1)
    await pipeline.handle_doorbell("front_door")
    pipeline._cooldowns.clear()          # simulate the 15s press cooldown expiring
    await pipeline.handle_doorbell("front_door")
    pipeline._cooldowns.clear()
    await pipeline.handle_doorbell("front_door")
    events, total = await db.list_events(camera="front_door")
    check(total == 3, "three presses create three event rows (every ring is recorded)")
    open_rows = [e for e in events if e["end_time"] is None]
    check(len(open_rows) == 1,
          "...but only ONE is held open — one visit gets one recording, not three")
    markers = [e for e in events if e["end_time"] is not None]
    check(all(_never_has_clip(m) for m in markers),
          "the repeat-press rows report 'no clip' rather than a phantom recorder fault")
    pipeline.update_count("front_door", "person", 0)
    await asyncio.sleep(0.8)
    check(len(rec.clips) == 1, "one visit produces exactly ONE clip extraction")
    await pipeline.shutdown()
    await db.close()

    # --- privacy abort must not read as "processing" ---
    db, settings, _, pipeline, rec = await _make_pipeline()
    start = time.time()
    eid = await _open_row(db, "doorbell.8", start)
    pipeline.update_count("front_door", "person", 1)
    task = asyncio.create_task(
        pipeline._doorbell_recording("front_door", "doorbell.8", eid, start))
    await asyncio.sleep(0.15)
    _engage_privacy(settings, "front_door")
    await asyncio.wait_for(task, timeout=3.0)
    row = await db.get_event(eid)
    check(_never_has_clip(row),
          "a privacy-aborted visit reports 'no clip' immediately")
    check(_clip_state(row, True) == "recording_disabled",
          "...NOT 'processing' — the UI must not claim a clip is being cut, and iOS "
          "must not mount a live player on the camera privacy just shut off")
    await db.close()

    # --- historical rows keep their meaning ---
    print("\n   historical doorbell rows (written before hold-open existed)")
    old = time.time() - 86_400
    hist = {"frigate_id": "doorbell.old", "has_clip": 0,
            "start_time": old, "end_time": old}
    check(_never_has_clip(hist),
          "a pre-existing doorbell row (end_time == start_time) is still clipless")
    check(_clip_state(hist, True) == "recording_disabled",
          "...so it does NOT retroactively read as 'unavailable' (= 'the recorder failed')")
    held = {"frigate_id": "doorbell.new", "has_clip": 0,
            "start_time": old, "end_time": old + 30}
    check(not _never_has_clip(held),
          "a genuinely held-open row (end_time > start_time) IS clip-eligible")

    # --- a stale has_clip must not promise a player ---
    pruned = {"frigate_id": "native.5", "has_clip": 1, "end_time": old}
    check(_clip_state(pruned, True) == "ready", "has_clip + file present -> 'ready'")
    check(_clip_state(pruned, True, file_present=False) != "ready",
          "has_clip but the file was pruned -> NOT 'ready' (no player against a 404)")


# ---------------------------------------------------------------------------
# 8. boot sweep must not stamp the future
# ---------------------------------------------------------------------------
async def sweep_clamp() -> None:
    print("\n8. boot sweep clamps to now")
    from app.routers.events import _clip_state
    db, _, _, _, _ = await _make_pipeline()
    now = time.time()
    recent = await db.insert_event(
        frigate_id="doorbell.recent", camera="front_door", label="doorbell",
        count=1, score=1.0, start_time=now - 3, end_time=None,
    )
    await db.close_open_doorbell_events(120.0)
    row = await db.get_event(recent)
    check(row["end_time"] <= time.time() + 0.5,
          "a press 3s before the crash is NOT stamped ~117s into the future")
    check(_clip_state(row, True) != "processing" or row["end_time"] <= time.time(),
          "so it cannot report 'processing' off a negative elapsed time")
    await db.close()


def main() -> None:
    asyncio.run(count_trap())
    asyncio.run(window_cases())
    asyncio.run(privacy_case())
    asyncio.run(press_cases())
    asyncio.run(sweep_case())
    predicate_split()
    asyncio.run(robustness_cases())
    asyncio.run(sweep_clamp())
    print(f"\nALL {PASS} CHECKS PASSED (doorbell hold-open recording)")


if __name__ == "__main__":
    main()
