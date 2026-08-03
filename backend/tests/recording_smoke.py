"""Recording correctness + recordings API smoke suite.

Covers the clip-availability fix and the new recordings index/playback API
(docs/CONTRACTS.md recordings routes). CPU-only, no network, no GPU; ffmpeg
is feature-detected (the REAL end-to-end clip section runs only when ffmpeg
is on PATH, per project rules — never installs it).

Sections:

  1. clip semantics — has_clip stays FALSE until a non-empty clip file lands;
     extraction failure (no segments / ffmpeg rc!=0 / empty output) leaves
     has_clip false and logs a greppable WARNING naming the reason.
  2. clip_state — the GET /api/events/{id} derivation for all four states
     (ready / processing / recording_disabled / unavailable) + the clip route
     404 detail matching the state, through the real app.
  3. index/ranges — GET /api/recordings/{camera}/index on a synthetic segment
     tree, incl. a gap that splits coverage into two ranges; missing day empty.
  4. cameras — GET /api/recordings/cameras bounds.
  5. playlist — GET /api/recordings/{camera}/playlist.m3u8 well-formed HLS VOD,
     only in-window segments, ?token= carried, 6 h window cap.
  6. seg — GET /api/recordings/{camera}/seg/{ts}.ts serves (Range ok), 404s an
     unknown ts, and _camera_dir refuses path traversal.
  7. REAL ffmpeg — synthesize TS segments, run build_clip_args extraction for
     real, assert a valid non-empty mp4 that ffprobe reports duration > 0
     (the real catch for a clip-extraction bug).

    python backend/tests/recording_smoke.py
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["SENTINEL_REQUIRE_GPU"] = "1"          # GPU-less host: detector hard-fails fast/cheap
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"   # unroutable -> instant refusal
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-recording-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import asyncio  # noqa: E402
import logging  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402
from app.routers.events import CLIP_PROCESSING_WINDOW_S, _clip_state  # noqa: E402
from app.routers.recordings import (  # noqa: E402
    MAX_PLAYLIST_WINDOW_S,
    _camera_dir,
    _merge_ranges,
    _scan_bounds,
)
from app.native.recorder import (  # noqa: E402
    SEGMENT_SECONDS,
    Recorder,
    build_clip_args,
    build_concat_list,
)

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


DEFAULT_RECORDING = {"continuous_days": 7, "event_days": 14, "snapshot_days": 14}


class FakeSettings:

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False
    def __init__(self, recording: dict | None = None):
        self.recording = dict(recording or DEFAULT_RECORDING)


def make_config(tag: str) -> Config:
    cfg = Config()
    cfg.data_dir = TMP / tag / "data"
    cfg.media_dir = TMP / tag / "media"
    return cfg


def seg_path(cam_dir: Path, dt: datetime) -> Path:
    return cam_dir / dt.strftime("%Y-%m-%d") / dt.strftime("%H") / dt.strftime("%M.%S.ts")


def make_seg(cam_dir: Path, dt: datetime, body: bytes = b"\x47" * 188) -> Path:
    p = seg_path(cam_dir, dt)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def warnings(self) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]


# =====================================================================
# 1. clip semantics — has_clip flips only when a non-empty file lands
# =====================================================================


async def _clip_semantics_cases() -> None:
    cfg = make_config("clipsem")
    db = Database(cfg.data_dir / "clip.db")
    await db.connect()

    t0 = datetime(2026, 7, 4, 10, 0, 0).timestamp()
    cam_dir = cfg.recordings_dir / "front"
    for off in (0, 10, 20, 30, 40):
        make_seg(cam_dir, datetime.fromtimestamp(t0 + off))
    start_time, end_time = t0 + 20, t0 + 35

    rec = Recorder(cfg, db, FakeSettings())
    rec._ffmpeg_path = "/fake/ffmpeg"

    cap = LogCapture()
    rec_logger = logging.getLogger("app.native.recorder")
    prev_level = rec_logger.level
    rec_logger.setLevel(logging.INFO)  # capture INFO (basicConfig runs later, at app import)
    rec_logger.addHandler(cap)
    try:
        # -- success: has_clip false BEFORE, true only AFTER a non-empty file --
        eid = await db.insert_event("native.cs-ok", "front", "person", 1, 0.9,
                                    start_time, end_time=end_time)
        check((await db.get_event(eid))["has_clip"] is False,
              "event row starts has_clip=false (engine never asserts it optimistically)")

        async def run_ok(args: list[str]) -> int:
            out = Path(args[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"MP4-BYTES" * 8)  # non-empty
            return 0

        rec._run_ffmpeg = run_ok
        out = await rec.extract_clip("front", "native.cs-ok", start_time, end_time)
        check(out == rec.clip_path(eid) and out.is_file() and out.stat().st_size > 0,
              "successful extraction writes a non-empty clip file")
        check((await db.get_event(eid))["has_clip"] is True,
              "has_clip flips true ONLY after the clip file exists")
        check(any("clip ready event=native.cs-ok" in m for m in cap.warnings()) is False,
              "success path logs no WARNING")
        check(any(r.getMessage().startswith("recorder: clip ready event=native.cs-ok")
                  for r in cap.records),
              "success logs a greppable 'recorder: clip ready' line with bytes")

        # -- empty output: ffmpeg rc=0 but 0-byte file -> has_clip stays false --
        cap.records.clear()
        eid2 = await db.insert_event("native.cs-empty", "front", "dog", 1, 0.8,
                                     start_time, end_time=end_time)

        async def run_empty(args: list[str]) -> int:
            out = Path(args[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")  # empty
            return 0

        rec._run_ffmpeg = run_empty
        out2 = await rec.extract_clip("front", "native.cs-empty", start_time, end_time)
        check(out2 is None and not rec.clip_path(eid2).exists(),
              "empty output -> no clip served")
        check((await db.get_event(eid2))["has_clip"] is False,
              "empty output leaves has_clip false")
        check(any("clip FAILED event=native.cs-empty" in w and "empty output" in w
                  for w in cap.warnings()),
              "empty output logs a WARNING naming the reason ('empty output')")

        # -- ffmpeg rc!=0 -> has_clip false + WARNING 'ffmpeg exited' --
        cap.records.clear()
        eid3 = await db.insert_event("native.cs-rc", "front", "cat", 1, 0.7,
                                     start_time, end_time=end_time)

        async def run_fail(args: list[str]) -> int:
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(b"HALF")
            return 1

        rec._run_ffmpeg = run_fail
        out3 = await rec.extract_clip("front", "native.cs-rc", start_time, end_time)
        check(out3 is None and (await db.get_event(eid3))["has_clip"] is False,
              "ffmpeg rc!=0 -> None, has_clip false")
        check(any("clip FAILED event=native.cs-rc" in w and "ffmpeg exited 1" in w
                  for w in cap.warnings()),
              "ffmpeg rc!=0 logs a WARNING naming the reason ('ffmpeg exited N')")

        # -- no segments in window -> has_clip false + WARNING 'no segments' --
        cap.records.clear()
        eid4 = await db.insert_event("native.cs-gap", "front", "car", 1, 0.6,
                                     t0 + 9000, end_time=t0 + 9010)
        rec._run_ffmpeg = run_ok
        out4 = await rec.extract_clip("front", "native.cs-gap", t0 + 9000, t0 + 9010)
        check(out4 is None and (await db.get_event(eid4))["has_clip"] is False,
              "no covering segments -> None, has_clip false")
        check(any("clip FAILED event=native.cs-gap" in w and "no segments in window" in w
                  for w in cap.warnings()),
              "no segments logs a WARNING naming the reason ('no segments in window')")

        # -- clip START log names the event, window and candidate count --
        check(any(r.getMessage().startswith("recorder: clip extract start event=native.cs-gap")
                  and "candidates=0" in r.getMessage() for r in cap.records),
              "clip extraction logs a START line (event id + window + N candidates)")
    finally:
        rec_logger.removeHandler(cap)
        rec_logger.setLevel(prev_level)
        await db.close()


def clip_semantics_checks() -> None:
    print("1. clip semantics — has_clip false until a non-empty clip file lands")
    asyncio.run(_clip_semantics_cases())


# =====================================================================
# 2. clip_state derivation (unit) + through the real app
# =====================================================================


def clip_state_unit_checks() -> None:
    print("2a. clip_state derivation — all four states")
    now = time.time()
    ready = {"frigate_id": "native.1", "has_clip": True, "end_time": now - 100}
    check(_clip_state(ready, True) == "ready", "has_clip -> 'ready'")
    check(_clip_state(ready, False) == "ready", "has_clip wins even if record disabled")

    proc = {"frigate_id": "native.2", "has_clip": False, "end_time": now - 5}
    check(_clip_state(proc, True) == "processing",
          "record on + ended recently + no clip -> 'processing'")
    open_evt = {"frigate_id": "native.2b", "has_clip": False, "end_time": None}
    check(_clip_state(open_evt, True) == "processing",
          "record on + not yet ended -> 'processing'")

    disabled = {"frigate_id": "native.3", "has_clip": False, "end_time": now - 5}
    check(_clip_state(disabled, False) == "recording_disabled",
          "record disabled -> 'recording_disabled'")
    # A doorbell press now HOLDS ITS EVENT OPEN until the visitor leaves and
    # schedules a real clip, so it is no longer clipless. It stays "synthetic"
    # for snapshot purposes (no engine frame) — the two properties were split
    # precisely here. These three pin that split.
    db_open = {"frigate_id": "doorbell.front.1", "has_clip": False, "end_time": None}
    check(_clip_state(db_open, True) == "processing",
          "doorbell visit in progress (end_time NULL) -> 'processing', not 'recording_disabled'")
    db_done = {"frigate_id": "doorbell.front.2", "has_clip": True, "end_time": now - 5}
    check(_clip_state(db_done, True) == "ready",
          "doorbell event with an assembled clip -> 'ready'")
    db_norec = {"frigate_id": "doorbell.front.3", "has_clip": False, "end_time": now - 5}
    check(_clip_state(db_norec, False) == "recording_disabled",
          "doorbell on a camera with recording off -> 'recording_disabled'")
    # ...while the OTHER synthetic kinds must still never offer a clip. If this
    # regresses, the _NO_CLIP_PREFIXES split has been widened by accident.
    for fid in ("audio.1", "cameraai.1"):
        row = {"frigate_id": fid, "has_clip": False, "end_time": now - 5}
        check(_clip_state(row, True) == "recording_disabled",
              f"{fid} still has no clip ever -> 'recording_disabled'")
    check(_clip_state({"frigate_id": "", "has_clip": False, "end_time": now - 5}, True)
          == "recording_disabled",
          "event with no frigate_id -> 'recording_disabled'")

    old = {"frigate_id": "native.4", "has_clip": False,
           "end_time": now - CLIP_PROCESSING_WINDOW_S - 10}
    check(_clip_state(old, True) == "unavailable",
          "record on + old + no clip -> 'unavailable'")


# ---------- app-level fixtures ----------


def _insert_camera(db_file: Path, name: str, friendly: str, record_enabled: bool = True) -> None:
    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT OR REPLACE INTO cameras (name, friendly_name, model, ip, username, password,"
        " record_enabled, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (name, friendly, "AD410", "192.0.2.9", "u", "p", int(record_enabled), time.time()),
    )
    conn.commit()
    conn.close()


def _insert_event(db_file: Path, fid: str, camera: str, has_clip: int, end_time: float) -> int:
    conn = sqlite3.connect(db_file)
    cur = conn.execute(
        "INSERT INTO events (frigate_id, camera, label, count, score, start_time,"
        " end_time, has_clip, has_snapshot, zones) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (fid, camera, "person", 1, 0.9, end_time - 20, end_time, has_clip, 0, "[]"),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return int(eid)


def clip_state_route_checks(client: TestClient, db_file: Path, headers: dict) -> None:
    print("2b. clip_state + clip 404 detail through the real app")
    now = time.time()
    _insert_camera(db_file, "on_cam", "On Cam", record_enabled=True)
    _insert_camera(db_file, "off_cam", "Off Cam", record_enabled=False)

    # processing: recording on, just ended, no clip
    proc_id = _insert_event(db_file, "native.rt-proc", "on_cam", 0, now - 5)
    detail = client.get(f"/api/events/{proc_id}", headers=headers).json()
    check(detail["record_enabled"] is True and detail["clip_state"] == "processing",
          "GET /events/{id}: record_enabled + clip_state='processing'")
    resp = client.get(f"/api/events/{proc_id}/clip.mp4", headers=headers)
    check(resp.status_code == 404 and resp.json()["detail"] == "Clip is still being prepared",
          "clip 404 detail matches 'processing'")

    # unavailable: recording on, old, no clip
    unav_id = _insert_event(db_file, "native.rt-unav", "on_cam", 0,
                            now - CLIP_PROCESSING_WINDOW_S - 30)
    detail = client.get(f"/api/events/{unav_id}", headers=headers).json()
    check(detail["clip_state"] == "unavailable", "old event with no clip -> 'unavailable'")
    resp = client.get(f"/api/events/{unav_id}/clip.mp4", headers=headers)
    check(resp.status_code == 404 and resp.json()["detail"] == "Clip not available",
          "clip 404 detail matches 'unavailable'")

    # recording_disabled: camera not recording
    off_id = _insert_event(db_file, "native.rt-off", "off_cam", 0, now - 5)
    detail = client.get(f"/api/events/{off_id}", headers=headers).json()
    check(detail["record_enabled"] is False and detail["clip_state"] == "recording_disabled",
          "camera record disabled -> clip_state='recording_disabled'")
    resp = client.get(f"/api/events/{off_id}/clip.mp4", headers=headers)
    check(resp.status_code == 404
          and resp.json()["detail"] == "Recording is disabled for this camera",
          "clip 404 detail matches 'recording_disabled'")

    # ready: has_clip true + file present
    ready_id = _insert_event(db_file, "native.rt-ready", "on_cam", 1, now - 5)
    clips_dir = Path(os.environ["MEDIA_DIR"]) / "native" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / f"{ready_id}.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
    detail = client.get(f"/api/events/{ready_id}", headers=headers).json()
    check(detail["clip_state"] == "ready", "has_clip + file present -> clip_state='ready'")
    resp = client.get(f"/api/events/{ready_id}/clip.mp4", headers=headers)
    check(resp.status_code == 200, "clip serves when has_clip + file present")


# =====================================================================
# 3-6. recordings API through the real app
# =====================================================================

INDEX_DAY = datetime(2026, 7, 4)


# A segment on the NEXT day, well past the 6 h playlist cap from 12:00 — used
# to prove the cap excludes it without polluting the 2026-07-04 index.
FAR_SEG = datetime(2026, 7, 5, 6, 0, 0)


def _build_segment_tree(cam_dir: Path) -> None:
    day = INDEX_DAY
    # range 1: 12:00:00, :10, :20
    for s in (0, 10, 20):
        make_seg(cam_dir, day.replace(hour=12, minute=0, second=s))
    # gap, range 2: 12:05:00, :10
    make_seg(cam_dir, day.replace(hour=12, minute=5, second=0))
    make_seg(cam_dir, day.replace(hour=12, minute=5, second=10))
    make_seg(cam_dir, FAR_SEG)


def index_checks(client: TestClient, db_file: Path, headers: dict) -> None:
    print("3. GET /api/recordings/{camera}/index — segments + merged ranges")
    _insert_camera(db_file, "yard", "Back Yard", record_enabled=True)
    cam_dir = Path(os.environ["MEDIA_DIR"]) / "native" / "recordings" / "yard"
    _build_segment_tree(cam_dir)

    resp = client.get("/api/recordings/yard/index?date=2026-07-04", headers=headers)
    check(resp.status_code == 200, "index route is 200")
    body = resp.json()
    check(body["date"] == "2026-07-04" and isinstance(body["tz_offset"], int),
          "index carries date + integer tz_offset")
    starts = [s["start"] for s in body["segments"]]
    day12 = INDEX_DAY.replace(hour=12).timestamp()
    check(len(body["segments"]) == 5 and all(s["duration"] == SEGMENT_SECONDS
          for s in body["segments"]),
          "day has exactly its 5 same-day segments (next-day segment not counted)")
    check(starts == sorted(starts), "segments sorted ascending by start")
    check(int(day12) in starts and int(FAR_SEG.timestamp()) not in starts,
          "segments are the same-day ones only")
    ranges = body["ranges"]
    check(len(ranges) == 2, "a >SEGMENT_SECONDS gap splits coverage into two ranges")
    check(ranges[0]["start"] == int(day12) and ranges[0]["end"] == int(day12) + 30,
          "range 1 covers the three contiguous segments (+ one segment length)")
    check(ranges[1]["start"] == int(day12 + 300)
          and ranges[1]["end"] == int(day12 + 300) + 20,
          "range 2 covers the post-gap segments")

    resp = client.get("/api/recordings/yard/index?date=2026-07-06", headers=headers)
    check(resp.status_code == 200 and resp.json()["segments"] == []
          and resp.json()["ranges"] == [],
          "a day with no footage returns empty segments/ranges (tolerated)")

    resp = client.get("/api/recordings/yard/index?date=not-a-date", headers=headers)
    check(resp.status_code == 400, "malformed date -> 400")


def cameras_checks(client: TestClient, db_file: Path, headers: dict) -> None:
    print("4. GET /api/recordings/cameras — bounds")
    resp = client.get("/api/recordings/cameras", headers=headers)
    check(resp.status_code == 200, "cameras route is 200")
    by_name = {c["camera"]: c for c in resp.json()}
    check("yard" in by_name and by_name["yard"]["friendly_name"] == "Back Yard",
          "entry per camera with friendly_name")
    yard = by_name["yard"]
    day12 = int(INDEX_DAY.replace(hour=12).timestamp())
    far_end = int(FAR_SEG.timestamp()) + SEGMENT_SECONDS
    check(yard["has_recordings"] is True and yard["earliest"] == day12
          and yard["latest"] == far_end,
          "bounds: earliest = first segment start, latest = last start + one segment")
    # a camera with no footage
    _insert_camera(db_file, "empty_cam", "Empty", record_enabled=True)
    resp = client.get("/api/recordings/cameras", headers=headers)
    empty = {c["camera"]: c for c in resp.json()}["empty_cam"]
    check(empty["has_recordings"] is False and empty["earliest"] is None
          and empty["latest"] is None,
          "camera with no segments -> has_recordings false, null bounds")


def playlist_checks(client: TestClient, headers: dict, token: str) -> None:
    print("5. GET /api/recordings/{camera}/playlist.m3u8 — HLS VOD")
    day12 = int(INDEX_DAY.replace(hour=12).timestamp())
    resp = client.get(f"/api/recordings/yard/playlist.m3u8?start={day12}&end={day12 + 25}",
                      headers=headers)
    check(resp.status_code == 200, "playlist route is 200")
    check(resp.headers["content-type"].startswith("application/vnd.apple.mpegurl"),
          "playlist content-type is HLS")
    text = resp.text
    lines = text.strip().splitlines()
    check(lines[0] == "#EXTM3U" and "#EXT-X-VERSION:3" in lines
          and "#EXT-X-PLAYLIST-TYPE:VOD" in lines
          and f"#EXT-X-TARGETDURATION:{SEGMENT_SECONDS}" in lines
          and lines[-1] == "#EXT-X-ENDLIST",
          "playlist is a well-formed HLS VOD (header, version, VOD type, targetdur, endlist)")
    seg_lines = [ln for ln in lines if ln.startswith("seg/")]
    check([ln.split("?")[0] for ln in seg_lines]
          == [f"seg/{day12}.ts", f"seg/{day12 + 10}.ts", f"seg/{day12 + 20}.ts"],
          "playlist lists ONLY the in-window segments (relative seg/ URLs)")
    check(text.count("#EXTINF:") == 3, "one #EXTINF per segment")

    # ?token= is carried through onto the segment URLs
    resp = client.get(f"/api/recordings/yard/playlist.m3u8?start={day12}&end={day12 + 25}"
                      f"&token={token}")
    check(all(f"?token={token}" in ln for ln in resp.text.splitlines()
              if ln.startswith("seg/")),
          "?token= carried through onto each relative seg URL")

    # window cap: a huge end is clamped to +6 h, excluding the next-day segment
    far_ts = int(FAR_SEG.timestamp())
    resp = client.get(f"/api/recordings/yard/playlist.m3u8?start={day12}&end={day12 + 100000}",
                      headers=headers)
    seg_ts = [ln.split("/")[1].split(".")[0] for ln in resp.text.splitlines()
              if ln.startswith("seg/")]
    check(str(far_ts) not in seg_ts and len(seg_ts) == 5,
          f"window capped at {MAX_PLAYLIST_WINDOW_S}s -> the out-of-window segment is excluded")

    resp = client.get(f"/api/recordings/yard/playlist.m3u8?start={day12}&end={day12}",
                      headers=headers)
    check(resp.status_code == 400, "empty/inverted window -> 400")


def seg_checks(client: TestClient, headers: dict) -> None:
    print("6. GET /api/recordings/{camera}/seg/{ts}.ts — serve + guard")
    day12 = int(INDEX_DAY.replace(hour=12).timestamp())
    seg_file = seg_path(Path(os.environ["MEDIA_DIR"]) / "native" / "recordings" / "yard",
                        datetime.fromtimestamp(day12))
    expected = seg_file.read_bytes()

    resp = client.get(f"/api/recordings/yard/seg/{day12}.ts", headers=headers)
    check(resp.status_code == 200 and resp.content == expected,
          "seg route serves the exact segment file")
    check(resp.headers["content-type"] == "video/mp2t", "seg content-type video/mp2t")

    resp = client.get(f"/api/recordings/yard/seg/{day12}.ts",
                      headers={**headers, "Range": "bytes=0-9"})
    check(resp.status_code == 206 and resp.content == expected[:10],
          "seg route serves Range requests (206)")

    resp = client.get(f"/api/recordings/yard/seg/{day12 + 5}.ts", headers=headers)
    check(resp.status_code == 404, "unknown segment ts -> 404")

    resp = client.get(f"/api/recordings/yard/seg/{day12}.ts")
    check(resp.status_code in (401, 403), "seg route requires auth")

    # traversal: _camera_dir refuses anything that isn't a direct child dir
    class _FakeReq:
        def __init__(self, root: Path):
            self.app = type("A", (), {"state": type("S", (), {"config":
                type("C", (), {"recordings_dir": root})()})()})()

    from fastapi import HTTPException  # noqa: PLC0415
    root = Path(os.environ["MEDIA_DIR"]) / "native" / "recordings"
    for bad in ("../secret", "a/b", "..", "/etc"):
        raised = False
        try:
            _camera_dir(_FakeReq(root), bad)
        except HTTPException as exc:
            raised = exc.status_code == 404
        check(raised, f"_camera_dir refuses traversal segment {bad!r} (404)")
    ok_dir = _camera_dir(_FakeReq(root), "yard")
    check(ok_dir == (root / "yard").resolve(), "_camera_dir resolves a legit camera dir")


# =====================================================================
# 7. REAL ffmpeg end-to-end clip extraction
# =====================================================================


def real_ffmpeg_checks() -> None:
    print("7. REAL ffmpeg — build_clip_args extraction produces a playable mp4")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None:
        print("  -- skipped (no ffmpeg on PATH; mocked extraction path covered in section 1)")
        return
    asyncio.run(_real_ffmpeg_cases(ffmpeg, ffprobe))


async def _real_ffmpeg_cases(ffmpeg: str, ffprobe: str | None) -> None:
    cfg = make_config("realclip")
    db = Database(cfg.data_dir / "real.db")
    await db.connect()

    # Synthesize two real TS segments laid out exactly like the recorder writes
    # them (build_segment_args is RTSP-only and golden-tested for argv in
    # native_smoke; here we exercise the REAL clip-extraction path).
    t0 = datetime(2026, 7, 4, 9, 0, 10).timestamp()
    cam_dir = cfg.recordings_dir / "front"
    for off in (0, 10):
        dest = seg_path(cam_dir, datetime.fromtimestamp(t0 + off))
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
             "-c:v", "mpeg4", "-f", "mpegts", str(dest)],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0 or not dest.is_file():
            print("  -- skipped (test segment generation failed)")
            await db.close()
            return

    start_time = t0 + 5.5
    end_time = start_time
    event_id = await db.insert_event("native.real", "front", "person", 1, 0.9,
                                     start_time, end_time=end_time)
    rec = Recorder(cfg, db, FakeSettings())
    rec._ffmpeg_path = ffmpeg  # use the REAL _run_ffmpeg + build_clip_args + concat

    out = await rec.extract_clip("front", "native.real", start_time, end_time)
    check(out is not None and out.is_file() and out.stat().st_size > 0,
          "REAL ffmpeg: concat + stream-copy cut produced a non-empty mp4")
    check(b"ftyp" in out.read_bytes()[:64], "REAL ffmpeg: output has an mp4 ftyp box")
    check((await db.get_event(event_id))["has_clip"] is True,
          "REAL ffmpeg: has_clip flipped true only after the file landed")

    # concat-list content is well-formed for the two covering segments
    check("ffconcat version 1.0" in build_concat_list([out]),
          "concat list builder still emits the ffconcat header")

    if ffprobe is not None:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip() or 0.0)
        check(probe.returncode == 0 and duration > 0.0,
              f"REAL ffprobe: clip is playable with duration > 0 ({duration:.2f}s)")
    else:
        print("  -- ffprobe absent: skipped duration assertion (clip bytes verified)")

    # build_clip_args argv is exactly the design-doc command for a real window
    argv = build_clip_args(Path("/tmp/c.txt"), 5.0, 25.0, Path("/o.mp4"))
    check("-movflags" in argv and argv[argv.index("-movflags") + 1] == "+faststart"
          and "-c" in argv and argv[argv.index("-c") + 1] == "copy",
          "build_clip_args cuts with stream-copy + faststart")
    await db.close()


def main() -> None:
    clip_semantics_checks()
    clip_state_unit_checks()

    from app.main import app  # noqa: PLC0415 — after env setup at module top

    with TestClient(app) as client:
        token = client.post("/api/auth/login",
                            json={"password": "test-password"}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        db_file = Path(os.environ["DATA_DIR"]) / "nvr.db"

        clip_state_route_checks(client, db_file, headers)
        index_checks(client, db_file, headers)
        cameras_checks(client, db_file, headers)
        playlist_checks(client, headers, token)
        seg_checks(client, headers)

    real_ffmpeg_checks()
    print(f"\nALL {PASS} CHECKS PASSED (recording correctness + recordings API)")


if __name__ == "__main__":
    main()
