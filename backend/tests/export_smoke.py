"""Download + timeline-range-export smoke suite.

Covers the two user-facing "get the footage out" features layered on the
existing media machinery (docs/CONTRACTS.md Events + Recordings sections):

  1. Event download — GET /api/events/{id}/clip.mp4 and .../snapshot.jpg accept
     ?download=1 and then carry ``Content-Disposition: attachment`` with a
     sanitized ``<camera>_<label>_<YYYY-MM-DD_HH-MM-SS>.<ext>`` filename; the
     default (no ?download) stays inline. Range/media-auth are unchanged.
  2. Timeline range export — GET /api/recordings/{camera}/export.mp4?start=&end=
     builds a downloadable H.264 MP4 over the window by REUSING the recorder's
     segment selection + clip/transcode machinery (Recorder.export_range):
       - filename/sanitize helpers;
       - export_range: no-ffmpeg / no-segments -> None; success writes a cached
         mp4 + cleans temp files; a cache HIT reuses the file; concurrent
         identical windows DE-DUP to one build; ffmpeg rc!=0 -> None + cleanup;
         the export cache LRU evicts oldest over the byte cap;
       - the route: EXPORT_MAX_SECONDS cap -> 400, inverted window -> 400, empty
         window -> 404, media-auth required, attachment header + filename, and a
         REAL ffmpeg end-to-end export that ffprobe reports duration>0 (arg-assert
         fallback when ffmpeg is absent).

CPU-only, no network, no GPU; ffmpeg is feature-detected (never installed).

    python backend/tests/export_smoke.py
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
os.environ["SENTINEL_REQUIRE_GPU"] = "1"          # GPU-less host: detector hard-fails fast
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"   # unroutable -> instant refusal
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-export-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import asyncio  # noqa: E402
import logging  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402
from app.routers.events import (  # noqa: E402
    _download_filename,
    _sanitize_filename_part as _events_sanitize,
)
from app.routers.recordings import (  # noqa: E402
    EXPORT_MAX_SECONDS,
    _export_filename,
    _sanitize_filename_part as _rec_sanitize,
)
from app.native.recorder import (  # noqa: E402
    Recorder,
    _evict_export_cache,
)
from app.native.transcode import Transcoder  # noqa: E402

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


def disabled_transcoder(cfg: Config) -> Transcoder:
    """A passthrough transcoder (no ffmpeg/ffprobe) so clip_plan always chooses
    the stream-copy branch — keeps the mocked export_range unit tests hermetic
    and fast (no real probe subprocess on fake segments)."""
    return Transcoder(
        cache_dir=cfg.media_dir / "native" / "tmp" / "transcode-cache",
        ffmpeg=None,
        ffprobe=None,
    )


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def warnings(self) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]


# =====================================================================
# 1. filename + sanitize helpers
# =====================================================================


def filename_checks() -> None:
    print("1. filename + sanitize helpers")
    # sanitize: unsafe chars collapse to '_', separators trimmed, empty->fallback
    check(_rec_sanitize("front door", "camera") == "front_door",
          "sanitize collapses spaces to underscore")
    check(_rec_sanitize('a/b\\c"d;e', "camera") == "a_b_c_d_e",
          "sanitize collapses slashes/quotes/semicolons (no header/quote injection)")
    check(_rec_sanitize("...", "camera") == "camera",
          "sanitize falls back when nothing safe remains")
    check(_rec_sanitize("café#1", "camera") == "caf_1",
          "sanitize drops non-ASCII + specials")
    # events + recordings share the same sanitizer semantics
    check(_events_sanitize("x y", "event") == "x_y" and _rec_sanitize("x y", "event") == "x_y",
          "events + recordings sanitizers agree")

    ts = datetime(2026, 7, 4, 13, 5, 9).timestamp()
    ev = {"camera": "front door", "label": "person", "start_time": ts}
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(ts))
    check(_download_filename(ev, "mp4") == f"front_door_person_{stamp}.mp4",
          "event clip download filename is <camera>_<label>_<stamp>.mp4 (sanitized)")
    check(_download_filename(ev, "jpg") == f"front_door_person_{stamp}.jpg",
          "event snapshot download filename uses the .jpg extension")
    ev_missing = {"camera": None, "label": None, "start_time": None}
    fn = _download_filename(ev_missing, "mp4")
    check(fn.startswith("camera_event_") and fn.endswith(".mp4"),
          "event download filename tolerates missing camera/label/start_time (fallbacks)")

    start = datetime(2026, 7, 4, 13, 0, 0).timestamp()
    end = datetime(2026, 7, 4, 13, 5, 30).timestamp()
    day = time.strftime("%Y-%m-%d", time.localtime(start))
    shms = time.strftime("%H-%M-%S", time.localtime(start))
    ehms = time.strftime("%H-%M-%S", time.localtime(end))
    check(_export_filename("back yard", start, end) == f"back_yard_{day}_{shms}-{ehms}.mp4",
          "export filename is <camera>_<start-date>_<HH-MM-SS>-<HH-MM-SS>.mp4 (sanitized)")


# =====================================================================
# 2. Recorder.export_range — reuse of the clip/transcode machinery
# =====================================================================


async def _export_range_cases() -> None:
    cfg = make_config("exprange")
    db = Database(cfg.data_dir / "exp.db")
    await db.connect()

    t0 = datetime(2026, 7, 4, 10, 0, 0).timestamp()
    cam_dir = cfg.recordings_dir / "front"
    for off in (0, 10, 20, 30, 40):
        make_seg(cam_dir, datetime.fromtimestamp(t0 + off))
    start, end = t0 + 12, t0 + 33

    rec = Recorder(cfg, db, FakeSettings())
    rec._transcode = disabled_transcoder(cfg)  # force the stream-copy branch

    cap = LogCapture()
    rec_logger = logging.getLogger("app.native.recorder")
    prev_level = rec_logger.level
    rec_logger.setLevel(logging.INFO)
    rec_logger.addHandler(cap)
    try:
        # -- ffmpeg unavailable -> None (no build attempted) --
        rec._ffmpeg_path = None
        out = await rec.export_range("front", start, end)
        check(out is None, "export_range returns None when ffmpeg is unavailable")
        check(any("export FAILED" in w and "ffmpeg unavailable" in w for w in cap.warnings()),
              "no-ffmpeg export logs a WARNING naming the reason")

        rec._ffmpeg_path = "/fake/ffmpeg"

        # -- no segments in the window -> None + WARNING --
        cap.records.clear()
        out = await rec.export_range("front", t0 + 9000, t0 + 9010)
        check(out is None, "export_range returns None when no segments cover the window")
        check(any("export FAILED" in w and "no segments in window" in w for w in cap.warnings()),
              "empty-window export logs a WARNING naming the reason")

        # -- success: builds a cached mp4, selecting the covering segments --
        cap.records.clear()
        calls = {"n": 0, "seek": None, "dur": None, "concat_lines": None}

        async def run_ok(args: list[str]) -> int:
            calls["n"] += 1
            # capture the cut math from build_clip_args argv
            if "-ss" in args:
                calls["seek"] = float(args[args.index("-ss") + 1])
            if "-t" in args:
                calls["dur"] = float(args[args.index("-t") + 1])
            # capture the concat list content (segments selected)
            concat = Path(args[args.index("-i") + 1])
            calls["concat_lines"] = concat.read_text().count("file '")
            out_p = Path(args[-1])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(b"MP4-EXPORT" * 16)  # non-empty
            return 0

        rec._run_ffmpeg = run_ok
        out = await rec.export_range("front", start, end)
        check(out is not None and out.is_file() and out.stat().st_size > 0,
              "export_range builds a non-empty mp4 for a covered window")
        check(out.parent == rec._export_cache_dir and out.suffix == ".mp4",
              "export lands in the bounded export-cache dir")
        # window [t0+12, t0+33] pulls segments starting at t0+10,20,30 (three)
        check(calls["concat_lines"] == 3,
              "export selected exactly the covering segments (concat has 3 files)")
        # seek = window_start - first_segment_start = (t0+12) - (t0+10) = 2.0; dur = 21.0
        check(abs(calls["seek"] - 2.0) < 0.01 and abs(calls["dur"] - 21.0) < 0.01,
              "export cut math: seek = start - first-segment-start, duration = end - start")
        leftovers = [p for p in rec._export_cache_dir.iterdir()
                     if p.name.startswith(".") or p.suffix in (".part", ".txt")
                     or ".part." in p.name or ".concat." in p.name]
        check(not leftovers, "successful export leaves no temp .part/.concat files behind")
        check(any(r.getMessage().startswith("recorder: export ready cam=front")
                  for r in cap.records),
              "success logs a greppable 'recorder: export ready' line")

        # -- cache HIT: an identical window reuses the file (no rebuild) --
        before = calls["n"]
        out2 = await rec.export_range("front", start, end)
        check(out2 == out and calls["n"] == before,
              "identical export window is a cache hit (no second ffmpeg build)")

        # -- ffmpeg rc!=0 -> None + no leftovers --
        cap.records.clear()

        async def run_fail(args: list[str]) -> int:
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(b"HALF")  # partial
            return 1

        rec._run_ffmpeg = run_fail
        # a fresh, uncached window so it actually attempts a build
        out3 = await rec.export_range("front", t0 + 2, t0 + 24)
        check(out3 is None, "export_range returns None when ffmpeg exits non-zero")
        check(any("export FAILED" in w and "ffmpeg exited" in w for w in cap.warnings()),
              "ffmpeg-failure export logs a WARNING naming the reason")
        bad_leftovers = [p for p in rec._export_cache_dir.iterdir()
                         if p.name.startswith(".")]
        check(not bad_leftovers,
              "a failed export cleans up its temp .part/.concat files")
    finally:
        rec_logger.removeHandler(cap)
        rec_logger.setLevel(prev_level)
        await db.close()


async def _export_dedupe_case() -> None:
    """Two concurrent identical-window exports share ONE build (in-flight
    de-dup), mirroring the transcoder's segment de-dup."""
    cfg = make_config("expdedupe")
    db = Database(cfg.data_dir / "d.db")
    await db.connect()
    t0 = datetime(2026, 7, 4, 11, 0, 0).timestamp()
    cam_dir = cfg.recordings_dir / "yard"
    for off in (0, 10, 20):
        make_seg(cam_dir, datetime.fromtimestamp(t0 + off))
    rec = Recorder(cfg, db, FakeSettings())
    rec._transcode = disabled_transcoder(cfg)
    rec._ffmpeg_path = "/fake/ffmpeg"

    builds = {"n": 0}

    async def run_slow(args: list[str]) -> int:
        builds["n"] += 1
        await asyncio.sleep(0.05)  # keep the build in-flight for the 2nd caller
        out_p = Path(args[-1])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"DEDUPE" * 8)
        return 0

    rec._run_ffmpeg = run_slow
    a, b = await asyncio.gather(
        rec.export_range("yard", t0 + 1, t0 + 21),
        rec.export_range("yard", t0 + 1, t0 + 21),
    )
    check(a is not None and a == b, "concurrent identical exports return the same file")
    check(builds["n"] == 1, "concurrent identical exports share ONE ffmpeg build (deduped)")
    await db.close()


def export_range_checks() -> None:
    print("2. Recorder.export_range — segment reuse, cache, cleanup, dedupe")
    asyncio.run(_export_range_cases())
    asyncio.run(_export_dedupe_case())


def evict_checks() -> None:
    print("2b. export cache LRU eviction")
    cache = TMP / "evict-cache"
    cache.mkdir(parents=True, exist_ok=True)
    now = time.time()
    # three 100-byte exports, distinct mtimes (oldest first)
    for i, age in enumerate((300, 200, 100)):
        p = cache / f"cam__{i}__{i}.mp4"
        p.write_bytes(b"x" * 100)
        os.utime(p, (now - age, now - age))
    # a temp part must never be evicted by the *.mp4 sweep
    part = cache / ".cam__9__9.abc.part.mp4"
    part.write_bytes(b"y" * 100)
    removed = _evict_export_cache(cache, max_bytes=150)  # keep < 2 files
    check(len(removed) >= 1 and all(p.suffix == ".mp4" for p in removed),
          "eviction removes oldest *.mp4 exports until under the byte cap")
    remaining = sorted(p.name for p in cache.glob("*.mp4"))
    check("cam__2__2.mp4" in remaining and "cam__0__0.mp4" not in remaining,
          "eviction keeps the newest export, drops the oldest")
    check(part.exists(), "eviction never deletes an in-flight temp .part file")


# =====================================================================
# 3. event download headers through the real app
# =====================================================================


def _insert_camera(db_file: Path, name: str, friendly: str, record_enabled: bool = True) -> None:
    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT OR REPLACE INTO cameras (name, friendly_name, model, ip, username, password,"
        " record_enabled, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (name, friendly, "AD410", "192.0.2.9", "u", "p", int(record_enabled), time.time()),
    )
    conn.commit()
    conn.close()


def _insert_event(db_file: Path, fid: str, camera: str, label: str, has_clip: int,
                  start_time: float) -> int:
    conn = sqlite3.connect(db_file)
    cur = conn.execute(
        "INSERT INTO events (frigate_id, camera, label, count, score, start_time,"
        " end_time, has_clip, has_snapshot, zones) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (fid, camera, label, 1, 0.9, start_time, start_time + 15, has_clip, 1, "[]"),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return int(eid)


def event_download_checks(client: TestClient, db_file: Path, headers: dict, token: str) -> None:
    print("3. event clip/snapshot ?download=1 attachment headers (through the app)")
    _insert_camera(db_file, "porch", "Front Porch", record_enabled=True)
    start_time = datetime(2026, 7, 4, 8, 30, 15).timestamp()
    eid = _insert_event(db_file, "native.dl-1", "porch", "person", 1, start_time)

    clips_dir = Path(os.environ["MEDIA_DIR"]) / "native" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / f"{eid}.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    snaps_dir = Path(os.environ["DATA_DIR"]) / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    (snaps_dir / f"{eid}.jpg").write_bytes(b"\xff\xd8\xff\xe0JFIF" + b"\x00" * 32)

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
    want_mp4 = f"porch_person_{stamp}.mp4"
    want_jpg = f"porch_person_{stamp}.jpg"

    # -- clip: inline by default (no attachment disposition) --
    resp = client.get(f"/api/events/{eid}/clip.mp4", headers=headers)
    check(resp.status_code == 200, "clip serves 200 (inline default)")
    check("attachment" not in resp.headers.get("content-disposition", ""),
          "clip default response is NOT an attachment (inline)")

    # -- clip: ?download=1 -> attachment + sanitized filename --
    resp = client.get(f"/api/events/{eid}/clip.mp4?download=1", headers=headers)
    cd = resp.headers.get("content-disposition", "")
    check(resp.status_code == 200 and cd == f'attachment; filename="{want_mp4}"',
          f"clip ?download=1 sets Content-Disposition attachment filename={want_mp4}")
    check(resp.content[:12].find(b"ftyp") != -1, "clip download still serves the mp4 bytes")

    # -- clip download: Range still honoured with the attachment header --
    resp = client.get(f"/api/events/{eid}/clip.mp4?download=1",
                      headers={**headers, "Range": "bytes=0-9"})
    check(resp.status_code == 206 and len(resp.content) == 10,
          "clip download still supports Range (206) alongside the attachment header")

    # -- snapshot: inline default vs ?download=1 attachment --
    resp = client.get(f"/api/events/{eid}/snapshot.jpg", headers=headers)
    check(resp.status_code == 200
          and "attachment" not in resp.headers.get("content-disposition", ""),
          "snapshot default response is inline (no attachment)")
    resp = client.get(f"/api/events/{eid}/snapshot.jpg?download=1", headers=headers)
    check(resp.status_code == 200
          and resp.headers.get("content-disposition", "") == f'attachment; filename="{want_jpg}"',
          f"snapshot ?download=1 sets attachment filename={want_jpg}")

    # -- media-auth still required + ?token= still works with ?download= --
    resp = client.get(f"/api/events/{eid}/clip.mp4?download=1")
    check(resp.status_code in (401, 403), "clip download still requires media-auth")
    resp = client.get(f"/api/events/{eid}/clip.mp4?download=1&token={token}")
    check(resp.status_code == 200
          and resp.headers.get("content-disposition", "") == f'attachment; filename="{want_mp4}"',
          "clip download honours ?token= media-auth (attachment served)")


# =====================================================================
# 4. export.mp4 route through the real app
# =====================================================================


def export_route_checks(client: TestClient, db_file: Path, headers: dict, token: str) -> None:
    print("4. GET /api/recordings/{camera}/export.mp4 — route contract")
    rec_root = Path(os.environ["MEDIA_DIR"]) / "native" / "recordings"
    cam_dir = rec_root / "expcam"
    # three contiguous real-or-fake segments on a fixed historical day
    base = datetime(2026, 7, 4, 9, 0, 0)
    starts = [base.replace(second=0), base.replace(second=10), base.replace(second=20)]
    win_start = starts[0].timestamp() + 2
    win_end = starts[2].timestamp() + 8

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    real = ffmpeg is not None
    if real:
        ok = True
        for dt in starts:
            dest = seg_path(cam_dir, dt)
            dest.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
                 "-c:v", "mpeg4", "-f", "mpegts", str(dest)],
                capture_output=True, timeout=60,
            )
            ok = ok and proc.returncode == 0 and dest.is_file()
        if not ok:
            print("  -- real segment generation failed; falling back to fake segments")
            real = False
    if not real:
        for dt in starts:
            make_seg(cam_dir, dt)

    # -- inverted window -> 400 (before any build) --
    resp = client.get(f"/api/recordings/expcam/export.mp4?start={win_start}&end={win_start}",
                      headers=headers)
    check(resp.status_code == 400, "export inverted/empty window (end<=start) -> 400")

    # -- over-cap window -> 400 with a clear message --
    over = win_start + EXPORT_MAX_SECONDS + 60
    resp = client.get(f"/api/recordings/expcam/export.mp4?start={win_start}&end={over}",
                      headers=headers)
    check(resp.status_code == 400 and "too long" in resp.json()["detail"].lower(),
          f"export window over EXPORT_MAX_SECONDS ({EXPORT_MAX_SECONDS}s) -> 400 with a clear detail")

    # -- empty window (no footage) -> 404 --
    far = starts[0].timestamp() + 100000
    resp = client.get(f"/api/recordings/expcam/export.mp4?start={far}&end={far + 20}",
                      headers=headers)
    check(resp.status_code == 404, "export window with no footage -> 404")

    # -- media-auth required --
    resp = client.get(f"/api/recordings/expcam/export.mp4?start={win_start}&end={win_end}")
    check(resp.status_code in (401, 403), "export route requires media-auth")

    # -- unknown/traversal camera -> 404 --
    resp = client.get(f"/api/recordings/..%2fsecret/export.mp4?start={win_start}&end={win_end}",
                      headers=headers)
    check(resp.status_code == 404, "export refuses a traversal camera segment (404)")

    if real:
        want = _export_filename("expcam", win_start, win_end)
        resp = client.get(
            f"/api/recordings/expcam/export.mp4?start={win_start}&end={win_end}&token={token}"
        )
        check(resp.status_code == 200, "REAL ffmpeg: export.mp4 returns 200")
        check(resp.headers.get("content-type", "").startswith("video/mp4"),
              "export Content-Type is video/mp4")
        check(resp.headers.get("content-disposition", "") == f'attachment; filename="{want}"',
              f"export sets Content-Disposition attachment filename={want}")
        body = resp.content
        check(b"ftyp" in body[:64], "REAL ffmpeg: export body is an mp4 (ftyp box)")

        # written to the bounded export cache with no temp leftovers
        exp_cache = Path(os.environ["MEDIA_DIR"]) / "native" / "tmp" / "export-cache"
        mp4s = list(exp_cache.glob("*.mp4"))
        temps = [p for p in exp_cache.iterdir() if p.name.startswith(".")]
        check(len(mp4s) == 1 and not temps,
              "export cached exactly one mp4, no temp .part/.concat leftovers")

        if ffprobe is not None:
            tmp_out = TMP / "export-probe.mp4"
            tmp_out.write_bytes(body)
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(tmp_out)],
                capture_output=True, text=True, timeout=30,
            )
            duration = float(probe.stdout.strip() or 0.0)
            check(probe.returncode == 0 and duration > 0.0,
                  f"REAL ffprobe: exported mp4 is playable with duration>0 ({duration:.2f}s)")
        else:
            print("  -- ffprobe absent: skipped duration assertion (mp4 bytes verified)")

        # a second identical request is a cache hit (still 200, same bytes)
        resp2 = client.get(
            f"/api/recordings/expcam/export.mp4?start={win_start}&end={win_end}&token={token}"
        )
        check(resp2.status_code == 200 and resp2.content == body,
              "identical export request is served from cache (same bytes)")
    else:
        # arg-assert fallback (no ffmpeg): the recorder's clip argv is the cut we rely on
        from app.native.recorder import build_clip_args  # noqa: PLC0415
        argv = build_clip_args(Path("/tmp/c.txt"), 2.0, 20.0, Path("/o.mp4"))
        check("-movflags" in argv and argv[argv.index("-movflags") + 1] == "+faststart",
              "arg-assert (no ffmpeg): export reuses the faststart stream-copy cut")


def main() -> None:
    filename_checks()
    export_range_checks()
    evict_checks()

    from app.main import app  # noqa: PLC0415 — after env setup at module top

    with TestClient(app) as client:
        token = client.post("/api/auth/login",
                            json={"password": "test-password"}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        db_file = Path(os.environ["DATA_DIR"]) / "nvr.db"

        event_download_checks(client, db_file, headers, token)
        export_route_checks(client, db_file, headers, token)

    print(f"\nALL {PASS} CHECKS PASSED (event download + timeline range export)")


if __name__ == "__main__":
    main()
