"""Recorder — 24/7 segment recording + event clip extraction.

Implements docs/native-mode-design.md §5 (contract addendum applies).

Segments
--------
One ffmpeg child per ``record_enabled`` camera, stream-copying the go2rtc
MAIN restream (``{go2rtc_rtsp}/{name}`` — AAC audio via the #audio=aac
source) into 10 s MPEG-TS segments:

    <recordings_dir>/{name}/%Y-%m-%d/%H/%M.%S.ts

argv is ``build_segment_args()`` (final — golden-tested). ffmpeg's strftime
does NOT mkdir: the current+next day/hour dirs are pre-created before each
spawn and on a small clock task. Watchdog: process exit OR no new segment
file for 30 s => kill + respawn with backoff (2 s doubling to 60 s cap; a
child that survived 60 s resets the backoff).

Index: filesystem-scan based (design doc's choice) — segment start times are
parsed straight from the ``{YYYY-MM-DD}/{HH}/{MM.SS}.ts`` path, no sqlite.

Retention (hourly task, blocking work in a thread)
--------------------------------------------------
- delete recording hour-dirs whose newest content mtime is older than
  ``settings.recording.continuous_days`` (then remove empty day dirs);
- delete ``clips/*.mp4`` older than ``settings.recording.event_days``
  (the event-row pruner in main.py also unlinks a clip when it drops rows;
  stale ``.{id}.part.mp4`` leftovers are swept by the same pass);
Space-based rotation (every 60 s, blocking work in a thread)
------------------------------------------------------------
"Overwrite the oldest with the newest", applied ON TOP of the day-based
cutoffs above — whichever frees a recording first wins. ``space_pass()``
deletes the oldest recording hour dirs until BOTH limits hold:

- ``settings.recording.max_storage_gb`` — a cap on the recordings tree
  (0 = uncapped). What you want when the disk is shared with other data.
- ``settings.recording.min_free_gb`` — a free-space floor on the media
  filesystem (default 5 GB), the backstop if something else fills the disk.

It runs on its OWN minute timer, not the hourly retention one: three cameras
write ~5.6 GB/hour, more than the whole default floor, so an hourly guard can
let the disk go genuinely full — and a full disk does not rotate, it makes
ffmpeg fail its writes. Deletion overshoots each limit slightly (hysteresis)
so a disk sitting at the threshold does not re-prune every tick. Hour dirs
written to in the last two minutes are never deleted (ffmpeg is inside them),
and EVENT CLIPS ARE NEVER DELETED FOR SPACE — they expire only by
``event_days``. Still short after rotating everything prunable => a loud
ERROR, not an escalation.

Clips
-----
``schedule_clip()`` is called by the engine right after it emits an ``end``
payload; it returns immediately. Internally: wait ~20 s (guarantees the
segment covering end+5 s has closed), select segments intersecting
``[start-5, end+5]`` by filename timestamp (including segments starting up
to 10 s before the window), write a concat list, run ``build_clip_args()``
(stream copy + faststart) into a hidden ``.part.mp4``, atomically rename to
``clip_path(event_id)`` (the id resolved via
``db.get_event_by_frigate_id``), then ``db.update_event(has_clip=True)``.
Missing segments => log once, leave has_clip false, never retry-loop.

Async model
-----------
All coroutines run on the app event loop; ffmpeg children via
``asyncio.create_subprocess_exec``. ``start()``/``stop()``/``reload()`` are
idempotent; ``stop()`` terminates children (SIGTERM, 5 s, then SIGKILL) and
cancels internal tasks. ffmpeg absence (dev Macs) => one log line per boot
and the recorder behaves as if zero cameras are record-enabled; ``status()``
still answers.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Sequence

from .transcode import HW_ENCODERS, LIBX264, Transcoder, build_transcode_args

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..db import Database
    from ..settings_store import SettingsStore

log = logging.getLogger(__name__)

SEGMENT_SECONDS = 10
CLIP_PAD_S = 5.0          # DEFAULT clip padding; overridden per-install by
                          # settings.recording.clip_pre_s / clip_post_s
CLIP_DELAY_S = 20.0       # DEFAULT post-event wait before extraction, overridden
                          # per-install by settings.recording.clip_delay_s


def max_clip_post_s(clip_delay_s: float) -> int:
    """Ceiling on post-roll for a given extraction delay.

    Forced by the recorder rather than chosen: extraction starts ``clip_delay_s``
    after the event ends, and the segment covering any instant is only closed and
    on disk ``SEGMENT_SECONDS`` after it opens. Footage past that horizon has not
    been written when the clip is cut — ffmpeg would stop at the end of what
    exists and return a clip quietly shorter than configured.

    A function, not a constant, because the delay is now a setting: raising it
    is exactly how an operator buys more post-roll than the default allows.
    """
    return max(0, int(clip_delay_s - SEGMENT_SECONDS))


MAX_CLIP_POST_S = max_clip_post_s(CLIP_DELAY_S)  # the ceiling at the default delay
SEGMENT_STALL_S = 30.0    # watchdog: no new segment file => respawn
LOW_DISK_BYTES = 5 * 1024**3   # default free-space floor (settings.min_free_gb)

# Space-based rotation ("overwrite oldest with newest") runs on THIS cadence,
# not the hourly retention one.
#
# The hourly sweep is far too slow to be the only guard: three cameras write
# ~135 GB/day, i.e. ~5.6 GB/hour — MORE than the whole 5 GB default floor. The
# disk can therefore cross the floor and go genuinely full between two hourly
# passes, and a full disk does not rotate, it makes ffmpeg fail its writes. One
# minute bounds the overshoot to ~100 MB at that rate.
SPACE_CHECK_INTERVAL_S = 60.0

# Rotation deletes PAST the limit by this much, so the next minute's writes do
# not immediately trip it again — without hysteresis a disk sitting exactly at
# the threshold re-prunes every tick, one hour dir at a time, forever.
SPACE_HEADROOM_FRACTION = 0.02   # free 2% beyond the cap
SPACE_HEADROOM_MIN_BYTES = 1024**3   # ...but at least 1 GB

_BACKOFF_MIN_S = 2.0
_BACKOFF_MAX_S = 60.0
_HEALTHY_RUN_S = 60.0        # child alive this long => backoff resets
_WATCH_POLL_S = 5.0
_ROLLOVER_INTERVAL_S = 20.0
_RETENTION_INTERVAL_S = 3600.0
_BOOT_SEGMENT_CHECK_S = 30.0  # warn if a record_enabled cam has no segment by now
_ACTIVE_GRACE_S = 120.0      # never delete files/dirs written this recently
_TERMINATE_WAIT_S = 5.0
_CLIP_FFMPEG_TIMEOUT_S = 120.0

# Cap on CONCURRENT clip extractions. Events on different cameras routinely end
# together (one person walks past three of them), and each clip on an HEVC
# camera is a full re-encode — on a box that falls back to libx264 those pile up,
# starve the detect-ingest ffmpeg children of CPU, and can push each other past
# _CLIP_FFMPEG_TIMEOUT_S. A timed-out transcode falls back to a stream-copy,
# which lands an HEVC clip the browser cannot play, so the pile-up degrades the
# ONE artifact the event exists to produce. Queueing costs a few seconds of clip
# latency and is invisible (clips are already delayed CLIP_DELAY_S). Mirrors the
# export path's semaphore; 2 keeps a hardware encoder busy without thrashing CPU.
CLIP_CONCURRENCY = 2

# Timeline range export (the recordings router's export.mp4). The window is
# capped by the router (EXPORT_MAX_SECONDS); finished exports live in a bounded
# on-disk LRU (identical windows share the cached file + one in-flight build).
EXPORT_CACHE_MAX_BYTES = 2 * 1024**3  # 2 GB of cached range exports


def _ffmpeg_available() -> Optional[str]:
    """Feature detection hook (tests monkeypatch this)."""
    return shutil.which("ffmpeg")


# ---------- pure argv builders (final; golden-tested in native_smoke) ----------


def build_segment_args(input_url: str, out_pattern: str) -> list[str]:
    """ffmpeg argv for the 24/7 segment recorder (§5.1): video stream-copy,
    audio RE-ENCODED to AAC.

    ``out_pattern`` is the strftime pattern, e.g.
    ``/media/native/recordings/front_door/%Y-%m-%d/%H/%M.%S.ts``.

    *** WHY AUDIO IS NOT STREAM-COPIED — this cost every recording its audio. ***
    The camera's NATIVE audio is G.711A (``pcm_alaw``): the backend deliberately
    provisions it that way so WebRTC live audio works (see
    native/streams.py:stream_sources + amcrest.audio_provision). But
    **``pcm_alaw`` has no MPEG-TS mapping in ffmpeg**, so a blanket ``-c copy``
    into the ``.ts`` segments cannot carry it — recordings, and therefore every
    event clip cut from them, came out silent on EVERY camera.

    The go2rtc ``ffmpeg:{name}#audio=aac`` source was supposed to hand us
    TS-legal AAC, but it never produces anything: the go2rtc image ships without
    a working ffmpeg (streams.py says so explicitly). So the transcode has to
    happen HERE, in the backend image, which does have ffmpeg.

    ``-c:v copy`` keeps video a pure stream-copy (no transcode cost — the whole
    point of the recorder). Only the audio is encoded, and G.711A is 8 kHz mono,
    so that is a rounding error of CPU per camera. A camera with no audio track
    at all is unaffected (nothing to map). Do NOT "simplify" this back to
    ``-c copy``.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",
        "-i", input_url,
        "-map", "0", "-c:v", "copy", "-c:a", "aac",
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-segment_atclocktime", "1",
        "-reset_timestamps", "1",
        "-strftime", "1",
        out_pattern,
    ]


def build_clip_args(concat_list: Path, seek_s: float, duration_s: float, out_path: Path) -> list[str]:
    """ffmpeg argv for event clip extraction (concat + stream-copy cut, §5.3).

    ``seek_s`` = window_start - first_segment_start (cuts at the previous
    keyframe — sub-second slop accepted).
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-ss", f"{max(0.0, seek_s):.3f}",
        "-t", f"{max(0.0, duration_s):.3f}",
        "-c", "copy", "-movflags", "+faststart",
        str(out_path),
    ]


def build_concat_list(segments: Sequence[Path]) -> str:
    """ffmpeg concat-demuxer list file content for the given segments
    (single quotes escaped the ffmpeg way: ' -> '\\'')."""
    lines = ["ffconcat version 1.0"]
    for path in segments:
        escaped = str(path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


# ---------- segment index: filesystem scan by filename timestamp ----------


def hour_dir(camera_dir: Path, ts: float) -> Path:
    """The hour directory a segment started at local epoch ``ts`` lives in."""
    lt = time.localtime(ts)
    return camera_dir / time.strftime("%Y-%m-%d", lt) / time.strftime("%H", lt)


def parse_segment_start(path: Path) -> Optional[float]:
    """Local-epoch start time parsed from ``.../{YYYY-MM-DD}/{HH}/{MM.SS}.ts``
    (ffmpeg -strftime writes local time). None when the path doesn't match."""
    try:
        year_s, month_s, day_s = path.parents[1].name.split("-")
        minute_s, second_s = path.stem.split(".")
        dt = datetime(
            int(year_s), int(month_s), int(day_s),
            int(path.parents[0].name), int(minute_s), int(second_s),
        )
    except (ValueError, IndexError, OverflowError):
        return None
    return dt.timestamp()


def _segment_start(day: date, hour_name: str, seg_name: str) -> Optional[float]:
    """Local-epoch start time from an already-parsed day + hour dir name and the
    ``{MM.SS}.ts`` filename — the fast path for the timeline index scan (avoids
    re-deriving day/hour from each Path). None when the names don't parse."""
    try:
        minute_s, second_s = seg_name[:-3].split(".")
        return datetime(
            day.year, day.month, day.day, int(hour_name), int(minute_s), int(second_s)
        ).timestamp()
    except (ValueError, OverflowError):
        return None


def _hour_start(day: date, hour_name: str) -> Optional[float]:
    """Local-epoch start of an ``{HH}`` hour dir, or None when it isn't one."""
    try:
        return datetime(day.year, day.month, day.day, int(hour_name)).timestamp()
    except (ValueError, OverflowError):
        return None


def select_segments(
    camera_dir: Path, window_start: float, window_end: float
) -> list[tuple[float, Path]]:
    """Segments that can intersect [window_start, window_end), sorted by
    start time: filename timestamp in [window_start - SEGMENT_SECONDS,
    window_end) — a segment starting up to 10 s before the window can still
    contain its head (§5.3)."""
    if not camera_dir.is_dir() or window_end <= window_start:
        return []
    lo = window_start - SEGMENT_SECONDS
    first_day = date.fromtimestamp(lo)
    last_day = date.fromtimestamp(window_end)
    out: list[tuple[float, Path]] = []
    # os.scandir (not iterdir/glob): dir-type checks reuse the readdir dtype
    # instead of a stat() per entry, and each segment's start time is computed
    # from the already-parsed day + hour dir names — this is the hot path behind
    # the timeline day index (thousands of .ts files per day per camera).
    try:
        day_scan = os.scandir(camera_dir)
    except OSError:
        return []
    with day_scan:
        for day_e in day_scan:
            try:
                day = date.fromisoformat(day_e.name)
            except ValueError:
                continue
            if day < first_day or day > last_day or not day_e.is_dir():
                continue
            try:
                hour_scan = os.scandir(day_e.path)
            except OSError:
                continue
            with hour_scan:
                for hour_e in hour_scan:
                    if not hour_e.is_dir():
                        continue
                    # PRUNE BY HOUR, not just by day. A clip window is tens of
                    # seconds, but a day holds 24 hour dirs of ~360 segments, so
                    # scanning the whole day cost ~8.6k directory entries to find
                    # the three or four that intersect — per event, forever.
                    #
                    # hour_start is DST-correct (datetime().timestamp() resolves
                    # the real offset). The two tests are deliberately asymmetric:
                    # forward is exact, since an hour starting at/after the window
                    # end cannot hold anything inside it; backward allows TWO
                    # hours, because a segment may start up to an hour after its
                    # dir does and a DST fall-back can stretch that hour to two in
                    # epoch terms. Over-scanning one extra dir is free; pruning a
                    # real one loses footage.
                    hour_start = _hour_start(day, hour_e.name)
                    if hour_start is None:
                        continue  # not an hour dir; its segments could not parse
                    if hour_start >= window_end or hour_start + 2 * 3600 <= lo:
                        continue
                    try:
                        seg_scan = os.scandir(hour_e.path)
                    except OSError:
                        continue
                    with seg_scan:
                        for seg_e in seg_scan:
                            if not seg_e.name.endswith(".ts"):
                                continue
                            ts = _segment_start(day, hour_e.name, seg_e.name)
                            if ts is not None and lo <= ts < window_end:
                                out.append((ts, Path(seg_e.path)))
    out.sort(key=lambda item: (item[0], str(item[1])))
    return out


# ---------- retention (pure-ish, blocking; run via asyncio.to_thread) ----------


def iter_hour_dirs(recordings_dir: Path) -> list[tuple[float, Path]]:
    """All ``{camera}/{day}/{hour}`` dirs as (newest content mtime, path),
    oldest first. Empty hour dirs use their own mtime."""
    out: list[tuple[float, Path]] = []
    if not recordings_dir.is_dir():
        return out
    for cam_dir in recordings_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        for day_dir in cam_dir.iterdir():
            if not day_dir.is_dir():
                continue
            for hd in day_dir.iterdir():
                if not hd.is_dir():
                    continue
                newest: Optional[float] = None
                try:
                    for entry in hd.iterdir():
                        with contextlib.suppress(OSError):
                            mtime = entry.stat().st_mtime
                            newest = mtime if newest is None else max(newest, mtime)
                    if newest is None:
                        newest = hd.stat().st_mtime
                except OSError:
                    continue
                out.append((newest, hd))
    out.sort(key=lambda item: (item[0], str(item[1])))
    return out


def hour_dir_size(hour_dir: Path) -> int:
    """Total bytes of the segment files directly inside one hour dir."""
    total = 0
    try:
        with os.scandir(hour_dir) as it:
            for entry in it:
                with contextlib.suppress(OSError):
                    if entry.is_file():
                        total += entry.stat().st_size
    except OSError:
        return 0
    return total


class HourDirSizes:
    """Cached ``hour dir -> bytes``, keyed on the dir's newest-content mtime.

    Space-based rotation has to know how much the recordings tree occupies, and
    it has to know it EVERY MINUTE (see SPACE_CHECK_INTERVAL_S). A naive
    recursive scan re-stats every segment file each time — for a week of three
    cameras that is ~180k files a minute, all of it to re-measure hour dirs that
    were sealed days ago and cannot have changed.

    A finished hour dir is immutable, so its size is cached against the mtime
    ``iter_hour_dirs`` already computes; only the two or three hour dirs ffmpeg
    is actively writing get re-measured. Entries for deleted dirs are dropped on
    each pass, so the cache cannot grow without bound.
    """

    def __init__(self) -> None:
        self._sizes: dict[Path, tuple[float, int]] = {}

    def total(self, entries: list[tuple[float, Path]]) -> int:
        """Total bytes across ``entries`` (as returned by ``iter_hour_dirs``)."""
        fresh: dict[Path, tuple[float, int]] = {}
        total = 0
        for mtime, hd in entries:
            cached = self._sizes.get(hd)
            size = cached[1] if cached is not None and cached[0] == mtime else hour_dir_size(hd)
            fresh[hd] = (mtime, size)
            total += size
        self._sizes = fresh          # drops entries for pruned dirs
        return total

    def forget(self, hour_dir: Path) -> None:
        self._sizes.pop(hour_dir, None)


def _remove_empty_day_dirs(recordings_dir: Path) -> None:
    if not recordings_dir.is_dir():
        return
    for cam_dir in recordings_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        for day_dir in cam_dir.iterdir():
            if day_dir.is_dir():
                with contextlib.suppress(OSError):
                    day_dir.rmdir()  # only succeeds when empty


def prune_recordings(recordings_dir: Path, cutoff: float) -> list[Path]:
    """Delete hour dirs whose newest content mtime < cutoff; then remove
    empty day dirs. Returns the deleted hour dirs (oldest first)."""
    removed: list[Path] = []
    for newest, hd in iter_hour_dirs(recordings_dir):
        if newest < cutoff:
            shutil.rmtree(hd, ignore_errors=True)
            removed.append(hd)
    if removed:
        _remove_empty_day_dirs(recordings_dir)
    return removed


def prune_clips(clips_dir: Path, cutoff: float) -> list[Path]:
    """Delete ``*.mp4`` clips older than cutoff (also sweeps stale hidden
    ``.{id}.part.mp4`` leftovers from aborted extractions)."""
    removed: list[Path] = []
    if not clips_dir.is_dir():
        return removed
    for clip in clips_dir.glob("*.mp4"):
        try:
            if clip.is_file() and clip.stat().st_mtime < cutoff:
                clip.unlink()
                removed.append(clip)
        except OSError:
            continue
    return removed


@dataclass
class _CameraRecorder:
    task: Optional[asyncio.Task] = None
    proc: Optional["asyncio.subprocess.Process"] = None
    last_segment_mtime: Optional[float] = None
    started_monotonic: Optional[float] = None  # when this camera's recorder began
    first_segment_seen: bool = False           # first segment observed (logged once)
    boot_warned: bool = False                  # "no segment within 30s" warned once


class Recorder:
    """24/7 segment recorder + event clip extractor (module docstring)."""

    def __init__(self, config: "Config", db: "Database", settings: "SettingsStore"):
        self._config = config
        self._db = db
        self._settings = settings
        self._running = False
        self._ffmpeg_path: Optional[str] = None
        self._cams: dict[str, _CameraRecorder] = {}
        self._tasks: list[asyncio.Task] = []
        self._clip_tasks: set[asyncio.Task] = set()
        # Serializes an admin purge_all_recordings against camera-CRUD reload()
        # (both contend on this lock) so nothing respawns ffmpeg mid-wipe.
        self._purge_lock = asyncio.Lock()
        # True only while a purge holds/awaits _purge_lock — lets a concurrent
        # purge 409 without false-positiving on a normal reload().
        self._purging = False
        # OVERRIDE, not the value: None means "use settings.recording.
        # clip_delay_s" (see _clip_delay). Tests set a number here to shrink the
        # post-end wait, and that still wins over whatever is configured.
        self.clip_delay_s: Optional[float] = None
        # Bounds concurrent clip extractions (see CLIP_CONCURRENCY). Held only
        # around the ffmpeg run, never across the post-event delay, so queued
        # clips keep their own timing.
        self._clip_sem = asyncio.Semaphore(CLIP_CONCURRENCY)
        # Cached hour-dir sizes for the per-minute space pass (see HourDirSizes).
        self._hour_sizes = HourDirSizes()
        # HEVC->H.264 transcoding for browser playback (segments served by the
        # recordings router via .transcoder; clips transcoded in extract_clip).
        # Recordings on disk stay HEVC; this only affects what the browser gets.
        self._transcode = Transcoder(
            cache_dir=config.media_dir / "native" / "tmp" / "transcode-cache"
        )
        # Bounded on-disk cache of timeline range exports (export.mp4) keyed by
        # camera+window, with in-flight de-duplication so two identical exports
        # share one ffmpeg build (mirrors the transcoder's segment LRU).
        self._export_cache_dir = config.media_dir / "native" / "tmp" / "export-cache"
        self._export_inflight: dict[str, asyncio.Future] = {}
        # Cap on CONCURRENT export builds. A 30-minute export of an HEVC camera
        # is a full NVENC/libx264 re-encode; _get_or_build_export only collapses
        # IDENTICAL windows, so N distinct requests fan out to N encoders. On a
        # box whose GPU is also running detection for 12 cameras, that is a
        # self-inflicted outage of the thing the system exists to do — and it is
        # reachable by anyone holding a media token. Exports queue instead.
        self._export_sem = asyncio.Semaphore(2)
        # Optional hook fired once per (re)connect cycle when a camera's stream
        # is confirmed live (first segment of the cycle). Used to re-assert
        # doorbell IR, which the AD410 resets when RTSP streaming starts.
        self._on_connect: Optional[Callable[[str], None]] = None

    def set_on_connect(self, callback: Optional[Callable[[str], None]]) -> None:
        """Register a sync callback ``cb(camera_name)`` fired each time a
        camera's ffmpeg confirms a fresh connection (first segment per spawn
        cycle). Best-effort — exceptions are logged, never propagated."""
        self._on_connect = callback

    def _notify_connect(self, name: str) -> None:
        cb = self._on_connect
        if cb is None:
            return
        try:
            cb(name)
        except Exception:  # noqa: BLE001 — a bad hook must not kill the recorder
            log.exception("recorder[%s]: on_connect hook failed", name)

    @property
    def transcoder(self) -> Transcoder:
        """Shared HEVC->H.264 transcoder (the recordings router uses it to serve
        browser-playable timeline segments)."""
        return self._transcode

    # ---------- paths (final) ----------

    def camera_dir(self, camera: str) -> Path:
        return self._config.recordings_dir / camera

    def segment_pattern(self, camera: str) -> str:
        return str(self.camera_dir(camera) / "%Y-%m-%d" / "%H" / "%M.%S.ts")

    def segment_input_url(self, camera: str) -> str:
        """go2rtc MAIN restream (AAC audio — TS-legal) for this camera."""
        return f"{self._config.go2rtc_rtsp_url}/{camera}"

    def clip_path(self, event_id: int) -> Path:
        """Clip file for a DB event row id (served by /api/events/{id}/clip.mp4)."""
        return self._config.clips_dir / f"{event_id}.mp4"

    # ---------- lifecycle ----------

    async def start(self) -> None:
        """Spawn per-camera segment recorders (record_enabled rows) plus the
        dir-rollover and retention tasks. Idempotent."""
        if self._running:
            return
        try:
            self._config.recordings_dir.mkdir(parents=True, exist_ok=True)
            self._config.clips_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.exception("could not create recorder directories under %s", self._config.media_dir)
        self._ffmpeg_path = _ffmpeg_available()
        if self._ffmpeg_path is None:
            log.warning(
                "ffmpeg not found on PATH — 24/7 recording and clip extraction are disabled"
            )
        self._running = True
        await self.reload()
        self._tasks.append(asyncio.create_task(self._rollover_loop(), name="recorder-rollover"))
        self._tasks.append(asyncio.create_task(self._retention_loop(), name="recorder-retention"))
        self._tasks.append(asyncio.create_task(self._space_loop(), name="recorder-space"))
        self._tasks.append(asyncio.create_task(self._boot_check_loop(), name="recorder-boot-check"))

    async def stop(self) -> None:
        """Terminate ffmpeg children and cancel internal tasks. Idempotent."""
        self._running = False
        pending = [*self._tasks, *self._clip_tasks]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._clip_tasks.clear()
        for name in list(self._cams):
            await self._stop_camera(name)

    async def reload(self) -> None:
        """Re-read camera rows; start/stop per-camera recorders so the set of
        running ffmpeg children matches record_enabled cameras. No-op when
        nothing changed (and before start()). Serialized against an in-progress
        purge via _purge_lock so a camera edit can't respawn ffmpeg into a tree
        that purge_all_recordings is mid-wipe on."""
        async with self._purge_lock:
            await self._reload_locked()

    async def _reload_locked(self) -> None:
        """reload() body. Caller MUST hold _purge_lock — the public reload()
        acquires it; purge_all_recordings already holds it when it resumes."""
        if not self._running:
            return
        wanted: set[str] = set()
        if self._ffmpeg_path is not None:
            rows = await self._db.list_cameras()
            wanted = {row["name"] for row in rows if row.get("record_enabled", True)}
        # PRIVACY MODE GATE (app/privacy.py). Drop private cameras from `wanted`
        # so the stop loop below tears their ffmpeg down and the start loop never
        # respawns it. This MUST be here, inside _reload_locked under
        # _purge_lock, not at a call site: the per-camera supervisor respawns a
        # killed ffmpeg within seconds, so the only durable stop is removing the
        # camera from `self._cams` — and doing it under the lock avoids a TOCTOU
        # race with a concurrent purge or camera CRUD reload.
        private = self._settings.private_cameras
        if private:
            blocked = wanted & private
            if blocked:
                log.info("recorder: privacy mode — not recording %s", ", ".join(sorted(blocked)))
            wanted -= private
        for name in [n for n in self._cams if n not in wanted]:
            log.info("recorder[%s]: stopping (record disabled, privacy mode, or camera removed)", name)
            await self._stop_camera(name)
        for name in sorted(wanted - self._cams.keys()):
            state = _CameraRecorder(started_monotonic=time.monotonic())
            self._cams[name] = state
            state.task = asyncio.create_task(self._run_camera(name), name=f"recorder-{name}")
            log.info("recorder: recording %s from %s", name, self.segment_input_url(name))

    async def _stop_camera(self, name: str) -> None:
        state = self._cams.pop(name, None)
        if state is None:
            return
        if state.task is not None:
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — a crashed supervisor must not break CRUD
                log.exception("recorder[%s]: supervisor task crashed", name)

    # ---------- admin: purge all continuous recordings ----------

    async def purge_all_recordings(self) -> dict[str, int]:
        """ADMIN: delete ALL continuous recording segments for ALL cameras.

        Stops each camera's ffmpeg child FIRST (via _stop_camera → SIGTERM/kill)
        so no in-flight segment file is unlinked mid-write, wipes the recordings
        tree + derived footage caches in a thread, then reload()s so
        record-enabled cameras resume immediately. Event clips (clips_dir),
        snapshots and the events DB are left untouched. Irreversible."""
        if self._purging:
            raise RuntimeError("a recordings purge is already in progress")
        self._purging = True
        stopped: list[str] = []
        dirs_removed = 0
        try:
            async with self._purge_lock:
                stopped = list(self._cams)
                for name in stopped:
                    await self._stop_camera(name)
                try:
                    dirs_removed = await asyncio.to_thread(self._wipe_recordings_tree)
                finally:
                    # ALWAYS bring recorders back — even if the wipe raised — so a
                    # failed purge never leaves 24/7 recording silently off. We
                    # already hold _purge_lock, so call the unlocked body.
                    await self._reload_locked()
        finally:
            self._purging = False
        log.warning(
            "ADMIN purge: deleted all continuous recordings (%d camera dir(s)); "
            "%d recorder(s) restarted",
            dirs_removed, len(stopped),
        )
        return {"camera_dirs_removed": dirs_removed, "recorders_restarted": len(stopped)}

    def _wipe_recordings_tree(self) -> int:
        """Blocking: rmtree every per-camera subtree under recordings_dir, then
        recreate the empty base dir. Also drops the derived caches of continuous
        footage (export-cache + transcode-cache) so a purge leaves no cached
        copy of deleted footage. Best-effort (ignore_errors tolerates a racing
        retention/serve). Event clips are NEVER touched. Returns dirs removed."""
        root = self._config.recordings_dir
        removed = 0
        if root.is_dir():
            for child in list(root.iterdir()):
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        root.mkdir(parents=True, exist_ok=True)
        for cache in (
            self._export_cache_dir,
            self._config.media_dir / "native" / "tmp" / "transcode-cache",
        ):
            shutil.rmtree(cache, ignore_errors=True)
        return removed

    # ---------- per-camera supervision ----------

    async def _run_camera(self, name: str) -> None:
        state = self._cams.get(name)
        if state is None:
            return
        backoff = _BACKOFF_MIN_S
        try:
            while self._running:
                try:
                    backoff = await self._record_once(name, state, backoff)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — the supervisor must outlive any cycle
                    log.exception("recorder[%s]: supervisor cycle failed", name)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
        finally:
            proc = state.proc
            state.proc = None
            if proc is not None:
                await self._terminate(proc)

    async def _record_once(self, name: str, state: _CameraRecorder, backoff: float) -> float:
        """One spawn/supervise cycle; returns the backoff for the next one."""
        self._ensure_hour_dirs(name)
        args = build_segment_args(self.segment_input_url(name), self.segment_pattern(name))
        started = time.monotonic()
        try:
            state.proc = await self._spawn(args)
        except (OSError, ValueError):
            log.exception("recorder[%s]: could not spawn ffmpeg", name)
            state.proc = None
            return backoff
        proc = state.proc
        drain: Optional[asyncio.Task] = None
        if proc.stderr is not None:
            drain = asyncio.create_task(self._drain_stderr(name, proc))
        stalled = False
        try:
            stalled = await self._watch(name, state, proc, started)
        finally:
            state.proc = None
            await self._terminate(proc)
            if drain is not None:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain
        ran = time.monotonic() - started
        if ran >= _HEALTHY_RUN_S:
            backoff = _BACKOFF_MIN_S
        reason = "stalled (no new segments)" if stalled else f"exited rc={proc.returncode}"
        log.warning(
            "recorder: %s ffmpeg %s after %.0f s — respawn in %.0f s (backoff)",
            name, reason, ran, backoff,
        )
        return backoff

    async def _spawn(self, args: list[str]) -> "asyncio.subprocess.Process":
        return await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _watch(
        self,
        name: str,
        state: _CameraRecorder,
        proc: "asyncio.subprocess.Process",
        started_monotonic: float,
    ) -> bool:
        """Wait for the child to exit; kill it when no new segment lands for
        SEGMENT_STALL_S. Returns True when the exit was a stall-kill."""
        camera_dir = self.camera_dir(name)
        connect_notified = False  # per-cycle: fire the connect hook once per spawn
        while True:
            try:
                await asyncio.wait_for(proc.wait(), timeout=_WATCH_POLL_S)
                return False  # exited on its own
            except asyncio.TimeoutError:
                pass
            mtime = await asyncio.to_thread(self._newest_segment_mtime, camera_dir)
            if mtime is not None:
                state.last_segment_mtime = mtime
                if not state.first_segment_seen:
                    state.first_segment_seen = True
                    log.info("recorder: first segment written for %s", name)
                if not connect_notified:
                    # This spawn's stream is confirmed live — the AD410 has just
                    # (re)started RTSP and reset its IR, so re-assert desired IR.
                    connect_notified = True
                    self._notify_connect(name)
            if time.monotonic() - started_monotonic < SEGMENT_STALL_S:
                continue  # grace for connect + first segment
            if mtime is None or time.time() - mtime > SEGMENT_STALL_S:
                log.warning(
                    "recorder: %s no new segment for %.0f s — killing ffmpeg (respawn)",
                    name, SEGMENT_STALL_S,
                )
                return True

    def _newest_segment_mtime(self, camera_dir: Path) -> Optional[float]:
        """Mtime of the newest .ts segment (cheap — reads at most two hour dirs
        and stat()s a single file). Segment names are '%M.%S.ts' with zero-padded
        fields, so the lexicographically-greatest name in the most-recent
        non-empty hour dir is the newest segment; only that one file is stat()ed.
        Equivalent to the old full-scan max because ffmpeg only ever appends to
        the highest-named (current) segment."""
        now = time.time()
        for ts in (now, now - 3600.0):
            hd = hour_dir(camera_dir, ts)
            if not hd.is_dir():
                continue
            newest_name = max((p.name for p in hd.glob("*.ts")), default=None)
            if newest_name is None:
                continue
            with contextlib.suppress(OSError):
                return (hd / newest_name).stat().st_mtime
        return None

    async def _drain_stderr(self, name: str, proc: "asyncio.subprocess.Process") -> None:
        assert proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace").strip()
                if text:
                    log.warning("recorder[%s] ffmpeg: %s", name, text[:300])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a logging drain must never crash the loop
            return

    async def _terminate(self, proc: "asyncio.subprocess.Process") -> None:
        """SIGTERM, wait 5 s, then SIGKILL."""
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_WAIT_S)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_WAIT_S)

    # ---------- directory rollover (ffmpeg strftime does not mkdir) ----------

    def _ensure_hour_dirs(self, name: str) -> None:
        camera_dir = self.camera_dir(name)
        now = time.time()
        for ts in (now, now + 3600.0):
            with contextlib.suppress(OSError):
                hour_dir(camera_dir, ts).mkdir(parents=True, exist_ok=True)

    async def _rollover_loop(self) -> None:
        while True:
            await asyncio.sleep(_ROLLOVER_INTERVAL_S)
            try:
                for name in list(self._cams):
                    self._ensure_hour_dirs(name)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("recorder dir rollover failed")

    # ---------- boot observability ----------

    async def _boot_check_loop(self) -> None:
        """Warn (once per camera) when a record_enabled camera still has no
        segment on disk ~30 s after its recorder started — the single loudest
        signal that recording is broken (bad go2rtc restream URL, camera down,
        ffmpeg rejecting the stream) on the operator's box."""
        while True:
            await asyncio.sleep(_WATCH_POLL_S)
            try:
                now = time.monotonic()
                for name, state in list(self._cams.items()):
                    if state.first_segment_seen or state.boot_warned:
                        continue
                    started = state.started_monotonic
                    if started is None or now - started < _BOOT_SEGMENT_CHECK_S:
                        continue
                    state.boot_warned = True
                    log.warning(
                        "recorder: %s produced NO segment within %.0f s of start — "
                        "recording is not working (check the go2rtc main restream %s)",
                        name, _BOOT_SEGMENT_CHECK_S, self.segment_input_url(name),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("recorder boot-check cycle failed")

    # ---------- retention ----------

    async def _retention_loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.retention_pass, time.time())
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("recorder retention pass failed")
            await asyncio.sleep(_RETENTION_INTERVAL_S)

    async def _space_loop(self) -> None:
        """Space-based rotation, every SPACE_CHECK_INTERVAL_S.

        Separate from the hourly retention loop on purpose: running out of disk
        is the failure that stops recording outright, and at real camera
        bitrates the hourly sweep cannot react before the disk is genuinely
        full. Blocking work (scandir + rmtree) runs in a thread so the event
        loop keeps serving.
        """
        while True:
            await asyncio.sleep(SPACE_CHECK_INTERVAL_S)
            try:
                await asyncio.to_thread(self.space_pass, time.time())
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("recorder space pass failed")

    def retention_pass(self, now: Optional[float] = None) -> dict[str, list[Path]]:
        """One blocking retention sweep (called hourly via to_thread; tests
        call it directly). The cutoffs never reach into the active-write
        grace window, so ``*_days: 0`` cannot delete in-flight files."""
        now = time.time() if now is None else now
        recording = self._settings.recording
        continuous_days = max(0, int(recording.get("continuous_days", 7)))
        event_days = max(0, int(recording.get("event_days", 14)))
        rec_cutoff = min(now - continuous_days * 86400.0, now - _ACTIVE_GRACE_S)
        clip_cutoff = min(now - event_days * 86400.0, now - _ACTIVE_GRACE_S)
        removed_hours = prune_recordings(self._config.recordings_dir, rec_cutoff)
        removed_clips = prune_clips(self._config.clips_dir, clip_cutoff)
        forced = self._low_disk_prune(now)
        if removed_hours or removed_clips:
            log.info(
                "retention: removed %d recording hour dir(s), %d clip file(s)",
                len(removed_hours), len(removed_clips),
            )
        return {"hours": removed_hours, "clips": removed_clips, "low_disk": forced}

    def _disk_free(self) -> Optional[int]:
        """Free bytes on the media filesystem (tests monkeypatch this)."""
        for path in (self._config.recordings_dir, self._config.media_dir):
            try:
                return shutil.disk_usage(path).free
            except OSError:
                continue
        return None

    def _space_limits(self) -> tuple[int, int]:
        """(max_storage_bytes, min_free_bytes) from settings. max 0 = no cap."""
        recording = self._settings.recording
        cap_gb = max(0, int(recording.get("max_storage_gb", 0) or 0))
        free_gb = max(1, int(recording.get("min_free_gb", 0) or 0)
                      or (LOW_DISK_BYTES // 1024**3))
        return cap_gb * 1024**3, free_gb * 1024**3

    def _clip_delay(self) -> float:
        """Seconds to wait after an event ends before cutting its clip.

        ``self.clip_delay_s`` is an override (tests); otherwise settings decide.
        """
        if self.clip_delay_s is not None:
            return max(0.0, float(self.clip_delay_s))
        configured = self._settings.recording.get("clip_delay_s")
        return CLIP_DELAY_S if configured is None else max(0.0, float(configured))

    def _clip_pads(self) -> tuple[float, float]:
        """(pre_s, post_s) padding for the event clip window, from settings.

        Post-roll is clamped against the EFFECTIVE delay here as well as in the
        settings schema: the schema guards the UI and the API, this guards a
        settings document written before the bound existed (or by hand), and a
        test that shrinks the delay without touching the pads.
        """
        recording = self._settings.recording
        pre = recording.get("clip_pre_s")
        post = recording.get("clip_post_s")
        # `or CLIP_PAD_S` would turn a deliberate 0 into 5 — a real choice for
        # someone who wants clips to start exactly on detection, so None (the
        # key is absent) is the only case that falls back to the default.
        pre_s = CLIP_PAD_S if pre is None else max(0.0, float(pre))
        post_s = CLIP_PAD_S if post is None else max(0.0, float(post))
        return pre_s, min(post_s, float(max_clip_post_s(self._clip_delay())))

    @staticmethod
    def _headroom(limit_bytes: int) -> int:
        """How far PAST a limit to prune, so the next tick does not re-trip it."""
        return max(SPACE_HEADROOM_MIN_BYTES, int(limit_bytes * SPACE_HEADROOM_FRACTION))

    def space_pass(self, now: Optional[float] = None) -> list[Path]:
        """Space-based rotation: delete the OLDEST continuous footage until the
        recordings tree is under ``max_storage_gb`` and the filesystem has at
        least ``min_free_gb`` free. Returns the hour dirs removed.

        This is the "overwrite oldest with newest" behavior. It is deliberately
        SEPARATE from (and much more frequent than) the day-based retention
        sweep, and applies on top of it: whichever frees a recording first wins.

        Event clips are never deleted here. They are the evidence the system
        exists to produce and are tiny next to continuous footage, so they
        expire only by ``event_days``. If every prunable hour dir is gone and
        space is STILL short, that is logged loudly rather than escalating.
        """
        now = time.time() if now is None else now
        cap, min_free = self._space_limits()
        entries = iter_hour_dirs(self._config.recordings_dir)
        used = self._hour_sizes.total(entries)
        free = self._disk_free()

        over_cap = cap > 0 and used > cap
        under_free = free is not None and free < min_free
        if not over_cap and not under_free:
            return []

        # Targets include headroom so rotation is not re-triggered every tick.
        want_used = (cap - self._headroom(cap)) if over_cap else None
        want_free = (min_free + self._headroom(min_free)) if under_free else None
        if over_cap:
            log.warning(
                "storage: recordings use %.1f GB > cap %.1f GB — rotating oldest footage",
                used / 1024**3, cap / 1024**3,
            )
        if under_free:
            log.warning(
                "storage: %.2f GB free < floor %.2f GB — rotating oldest footage",
                (free or 0) / 1024**3, min_free / 1024**3,
            )

        removed: list[Path] = []
        for newest, hd in entries:            # iter_hour_dirs is oldest-first
            if now - newest < _ACTIVE_GRACE_S:
                continue                      # ffmpeg is writing here — never delete
            size = hour_dir_size(hd)
            shutil.rmtree(hd, ignore_errors=True)
            self._hour_sizes.forget(hd)
            removed.append(hd)
            used -= size
            if free is not None:
                free += size
            if (want_used is None or used <= want_used) and \
               (want_free is None or free is None or free >= want_free):
                break

        if removed:
            _remove_empty_day_dirs(self._config.recordings_dir)
            log.warning(
                "storage: rotated %d oldest recording hour dir(s) — now %.1f GB used, "
                "%.2f GB free", len(removed), used / 1024**3, (free or 0) / 1024**3,
            )
        # Nothing left that MAY be deleted: every remaining hour dir is inside
        # the active-write grace window, or the space is held by something that
        # is not continuous footage (clips, or other data sharing the disk).
        # Report only the limit actually still breached — naming a disabled cap
        # as "cap 0.0 GB" sends whoever reads this log after the wrong thing.
        unmet: list[str] = []
        if cap > 0 and used > cap:
            unmet.append(f"{used / 1024**3:.1f} GB used vs cap {cap / 1024**3:.1f} GB")
        if free is not None and free < min_free:
            unmet.append(
                f"{free / 1024**3:.2f} GB free vs floor {min_free / 1024**3:.2f} GB"
            )
        if unmet:
            log.error(
                "storage: STILL over limits after rotating every prunable recording "
                "hour (%s). Event clips are never auto-deleted for space — raise the "
                "limit, lower event_days, or add disk.",
                "; ".join(unmet),
            )
        return removed

    def _low_disk_prune(self, now: float) -> list[Path]:
        """Back-compat wrapper: the hourly retention sweep still triggers a
        space pass, so a box that never hits the minute timer (tests, a short
        run) behaves as before."""
        return self.space_pass(now)

    # ---------- clips ----------

    async def schedule_clip(
        self, camera: str, frigate_id: str, start_time: float, end_time: float
    ) -> None:
        """Called by the engine right after an ``end`` payload. Returns
        immediately; extraction runs ~20 s later (fire-and-forget)."""
        if not self._running:
            log.debug("clip for %s dropped — recorder not running", frigate_id)
            return
        if self._settings.is_private(camera):
            log.info("clip for %s dropped — %s is in privacy mode", frigate_id, camera)
            return
        task = asyncio.create_task(
            self._clip_worker(camera, frigate_id, start_time, end_time),
            name=f"recorder-clip-{frigate_id}",
        )
        self._clip_tasks.add(task)
        task.add_done_callback(self._clip_tasks.discard)

    async def _clip_worker(
        self, camera: str, frigate_id: str, start_time: float, end_time: float
    ) -> None:
        try:
            await asyncio.sleep(self._clip_delay())
            # Acquire AFTER the delay: the wait is per-event pacing, not work,
            # and holding a slot through it would serialize unrelated events.
            async with self._clip_sem:
                await self.extract_clip(camera, frigate_id, start_time, end_time)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — clip failure must never propagate
            log.exception("clip extraction crashed for %s (%s)", frigate_id, camera)

    async def extract_clip(
        self, camera: str, frigate_id: str, start_time: float, end_time: float
    ) -> Optional[Path]:
        """Concat the covering segments into a faststart mp4 and mark the
        event row has_clip. Returns the clip path, or None (logged, no
        retries — §5.3 failure semantics)."""
        if self._ffmpeg_path is None:
            log.warning("recorder: clip FAILED event=%s — ffmpeg unavailable", frigate_id)
            return None
        # Re-check privacy HERE, not only in schedule_clip: a worker queued just
        # before the toggle sleeps ~20 s (clip_delay_s) before reaching this
        # point, and would otherwise assemble pre-privacy footage into a new
        # clip AFTER the operator turned Privacy Mode on.
        if self._settings.is_private(camera):
            log.info("recorder: clip event=%s dropped — %s is in privacy mode", frigate_id, camera)
            return None
        row = await self._db.get_event_by_frigate_id(frigate_id)
        if row is None:
            log.warning("recorder: clip FAILED event=%s — no matching event row", frigate_id)
            return None
        event_id = int(row["id"])
        pre_s, post_s = self._clip_pads()
        window_start = start_time - pre_s
        window_end = end_time + post_s
        segments = await asyncio.to_thread(
            select_segments, self.camera_dir(camera), window_start, window_end
        )
        log.info(
            "recorder: clip extract start event=%s id=%d cam=%s window=%.1f..%.1f candidates=%d",
            frigate_id, event_id, camera, window_start, window_end, len(segments),
        )
        if not segments:
            # Recorder was down for the window (or retention swept it): the
            # clip is never coming — leave has_clip false, do NOT retry-loop.
            log.warning(
                "recorder: clip FAILED event=%s cam=%s — no segments in window [%.1f, %.1f]",
                frigate_id, camera, window_start, window_end,
            )
            return None

        first_segment_start = segments[0][0]
        out_path = self.clip_path(event_id)
        part_path = out_path.with_name(f".{event_id}.part.mp4")
        concat_path = out_path.with_name(f".{event_id}.concat.txt")
        ok = False
        try:
            await asyncio.to_thread(
                self._write_concat, concat_path, [seg for _, seg in segments]
            )
            seek_s = window_start - first_segment_start
            duration_s = window_end - window_start
            returncode = await self._extract_to_part(
                camera, frigate_id, segments[0][1],
                concat_path, seek_s, duration_s, part_path,
            )
            if returncode != 0:
                log.warning(
                    "recorder: clip FAILED event=%s cam=%s — ffmpeg exited %s",
                    frigate_id, camera, returncode,
                )
                return None
            size = await asyncio.to_thread(_file_size, part_path)
            if size <= 0:
                # ffmpeg returned 0 but produced nothing usable (e.g. all
                # segments unreadable): treat as a hard failure, not a clip.
                log.warning(
                    "recorder: clip FAILED event=%s cam=%s — empty output (%d bytes)",
                    frigate_id, camera, size,
                )
                return None
            await asyncio.to_thread(part_path.replace, out_path)
            # has_clip flips to true ONLY here — after a non-empty file exists.
            if not row.get("has_clip"):
                await self._db.update_event(event_id, has_clip=True)
            ok = True
            log.info(
                "recorder: clip ready event=%s -> %s (bytes=%d, %d segments)",
                frigate_id, out_path.name, size, len(segments),
            )
            return out_path
        finally:
            leftovers = (concat_path,) if ok else (concat_path, part_path)
            for leftover in leftovers:
                await asyncio.to_thread(_unlink_quiet, leftover)

    # ---------- timeline range export (recordings router export.mp4) ----------

    async def export_range(
        self, camera: str, start_time: float, end_time: float
    ) -> Optional[Path]:
        """Build (or reuse a cached) faststart H.264 MP4 covering
        ``[start_time, end_time]`` for ``camera`` and return its path.

        Reuses the event-clip machinery end-to-end: the same segment selection
        (``select_segments``), concat list, cut math and ``_extract_to_part``
        (stream-copy for H.264 sources, else hardware→libx264→copy transcode). The
        window is expected to be pre-validated/-capped by the caller (the router
        enforces ``EXPORT_MAX_SECONDS``).

        Returns:
        - the cached/built MP4 path on success;
        - ``None`` when there is **no footage** in the window (caller → 404) or
          when ffmpeg failed / is unavailable (caller → 5xx). The distinction is
          logged; both leave no partial file behind.

        Identical concurrent windows share one build (in-flight de-dup) and the
        finished file, which lives in a bounded on-disk LRU."""
        if self._ffmpeg_path is None:
            log.warning("recorder: export FAILED cam=%s — ffmpeg unavailable", camera)
            return None
        segments = await asyncio.to_thread(
            select_segments, self.camera_dir(camera), start_time, end_time
        )
        if not segments:
            log.warning(
                "recorder: export FAILED cam=%s — no segments in window [%.1f, %.1f]",
                camera, start_time, end_time,
            )
            return None
        key = self._export_key(camera, start_time, end_time)
        out_path = self._export_cache_dir / f"{key}.mp4"
        if await asyncio.to_thread(_file_size, out_path) > 0:
            await asyncio.to_thread(_touch, out_path)
            log.info("recorder: export cache hit cam=%s -> %s", camera, out_path.name)
            return out_path
        return await self._get_or_build_export(
            key,
            lambda: self._build_export(camera, segments, start_time, end_time, out_path),
        )

    @staticmethod
    def _export_key(camera: str, start_time: float, end_time: float) -> str:
        safe_cam = camera.replace("/", "_").replace("\\", "_")
        return f"{safe_cam}__{int(round(start_time))}__{int(round(end_time))}"

    async def _get_or_build_export(
        self, key: str, factory: Callable[[], Awaitable[Optional[Path]]]
    ) -> Optional[Path]:
        """Share a single in-flight export build across concurrent same-window
        requests (mirrors Transcoder._get_or_transcode_segment)."""
        fut = self._export_inflight.get(key)
        if fut is None:
            fut = asyncio.ensure_future(factory())
            self._export_inflight[key] = fut
            fut.add_done_callback(lambda _f, k=key: self._export_inflight.pop(k, None))
        # shield so a client that disconnects mid-download can't cancel the
        # build another waiter is still relying on.
        return await asyncio.shield(fut)

    async def _build_export(
        self,
        camera: str,
        segments: list[tuple[float, Path]],
        start_time: float,
        end_time: float,
        out_path: Path,
    ) -> Optional[Path]:
        """Concat + cut the covering segments into ``out_path`` (transcoding to
        H.264 when the source is HEVC), atomically. Returns the path on success,
        ``None`` on ffmpeg failure. Temp files are always cleaned up."""
        # Acquired around the WHOLE build, so the queue forms before any temp
        # file or ffmpeg process exists.
        async with self._export_sem:
            return await self._build_export_locked(
                camera, segments, start_time, end_time, out_path
            )

    async def _build_export_locked(
        self,
        camera: str,
        segments: list[tuple[float, Path]],
        start_time: float,
        end_time: float,
        out_path: Path,
    ) -> Optional[Path]:
        first_segment_start = segments[0][0]
        stem = out_path.stem
        part_path = out_path.with_name(f".{stem}.{uuid.uuid4().hex}.part.mp4")
        concat_path = out_path.with_name(f".{stem}.{uuid.uuid4().hex}.concat.txt")
        ok = False
        try:
            await asyncio.to_thread(
                self._write_concat, concat_path, [seg for _, seg in segments]
            )
            seek_s = start_time - first_segment_start
            duration_s = end_time - start_time
            log.info(
                "recorder: export start cam=%s window=%.1f..%.1f segments=%d -> %s",
                camera, start_time, end_time, len(segments), out_path.name,
            )
            returncode = await self._extract_to_part(
                camera, f"export:{camera}", segments[0][1],
                concat_path, seek_s, duration_s, part_path,
            )
            if returncode != 0:
                log.warning(
                    "recorder: export FAILED cam=%s — ffmpeg exited %s", camera, returncode
                )
                return None
            size = await asyncio.to_thread(_file_size, part_path)
            if size <= 0:
                log.warning(
                    "recorder: export FAILED cam=%s — empty output (%d bytes)", camera, size
                )
                return None
            await asyncio.to_thread(part_path.replace, out_path)
            ok = True
            await asyncio.to_thread(_evict_export_cache, self._export_cache_dir)
            log.info(
                "recorder: export ready cam=%s -> %s (bytes=%d, %d segments)",
                camera, out_path.name, size, len(segments),
            )
            return out_path
        finally:
            leftovers = (concat_path,) if ok else (concat_path, part_path)
            for leftover in leftovers:
                await asyncio.to_thread(_unlink_quiet, leftover)

    @staticmethod
    def _write_concat(path: Path, segments: list[Path]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_concat_list(segments))

    async def _extract_to_part(
        self,
        camera: str,
        frigate_id: str,
        sample_segment: Path,
        concat_path: Path,
        seek_s: float,
        duration_s: float,
        part_path: Path,
    ) -> Optional[int]:
        """Cut the clip into ``part_path`` and return the ffmpeg return code.

        H.264 source → the fast stream-copy path (``build_clip_args``, unchanged).
        HEVC (or any non-browser codec) → transcode to a faststart H.264 mp4 so
        ``clip.mp4`` is browser-playable, with a hardware→libx264 runtime
        fallback and, as a last resort, a stream-copy so a clip still lands
        (logged; that clip stays HEVC and may not play). Recordings on disk are
        untouched."""
        plan = await self._transcode.clip_plan(camera, sample_segment)
        if not plan.transcode:
            args = build_clip_args(concat_path, seek_s, duration_s, part_path)
            return await self._run_ffmpeg(args)

        def transcode_args(enc: str) -> list[str]:
            return build_transcode_args(
                "ffmpeg", enc, container="mp4", output=part_path,
                concat_list=concat_path, seek_s=seek_s, duration_s=duration_s,
                audio_codec=plan.audio_codec,
                vaapi_device=self._transcode.vaapi_device,
            )

        encoder = plan.encoder or LIBX264
        log.info(
            "transcode: %s clip event=%s cam=%s (%s->h264)",
            encoder, frigate_id, camera, plan.video_codec or "?",
        )
        returncode = await self._run_ffmpeg(transcode_args(encoder))
        if returncode != 0 and encoder in HW_ENCODERS:
            # Runtime hardware-encoder failure → exclude it globally and retry on
            # whatever the transcoder re-selects (the next GPU encoder if there
            # is one, otherwise libx264).
            retry = self._transcode.mark_hw_failed(encoder)
            log.info(
                "transcode: %s clip event=%s cam=%s (%s retry)",
                retry, frigate_id, camera, encoder,
            )
            returncode = await self._run_ffmpeg(transcode_args(retry))
        if returncode != 0:
            # Both encoders failed → fall back to a stream-copy so a clip still
            # exists (today's behavior). It stays HEVC and may not play, but the
            # pipeline never regresses to producing nothing.
            log.warning(
                "transcode: clip transcode FAILED event=%s cam=%s — falling back "
                "to stream-copy (clip stays HEVC; may not play in the browser)",
                frigate_id, camera,
            )
            _unlink_quiet(part_path)
            returncode = await self._run_ffmpeg(
                build_clip_args(concat_path, seek_s, duration_s, part_path)
            )
        return returncode

    async def _run_ffmpeg(self, args: list[str]) -> Optional[int]:
        """Run a one-shot ffmpeg (clip extraction); returns its exit code or
        None when it could not run / timed out."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError):
            log.exception("could not spawn ffmpeg for clip extraction")
            return None
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_CLIP_FFMPEG_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            log.error("clip ffmpeg timed out after %.0f s — killing", _CLIP_FFMPEG_TIMEOUT_S)
            await self._terminate(proc)
            return None
        except asyncio.CancelledError:
            await self._terminate(proc)
            raise
        if proc.returncode != 0 and stderr:
            log.warning("clip ffmpeg stderr: %s", stderr.decode("utf-8", "replace").strip()[-500:])
        return proc.returncode

    # ---------- stats ----------

    def status(self) -> dict[str, dict[str, Any]]:
        """Per-camera recorder state for /api/system/detector:
        {camera: {"recording": bool, "last_segment_age_s": float|None}}."""
        now = time.time()
        out: dict[str, dict[str, Any]] = {}
        for name, state in self._cams.items():
            age = (now - state.last_segment_mtime) if state.last_segment_mtime else None
            proc = state.proc
            out[name] = {
                "recording": proc is not None and proc.returncode is None,
                "last_segment_age_s": round(age, 1) if age is not None else None,
            }
        return out


def _unlink_quiet(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _file_size(path: Path) -> int:
    """Byte size of ``path``, or -1 when it does not exist / can't be stat'd."""
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _touch(path: Path) -> None:
    """Bump mtime so an LRU sweep treats a cache hit as freshly used."""
    now = time.time()
    with contextlib.suppress(OSError):
        os.utime(path, (now, now))


def _evict_export_cache(cache_dir: Path, max_bytes: int = EXPORT_CACHE_MAX_BYTES) -> list[Path]:
    """Evict oldest finished range-export MP4s until the cache is under the byte
    cap. Only touches ``*.mp4`` files (temp parts are dot-prefixed). Returns the
    files removed (oldest first)."""
    removed: list[Path] = []
    try:
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in cache_dir.glob("*.mp4"):
            # Skip in-flight temp parts (``.{stem}.{uuid}.part.mp4``) — pathlib's
            # glob matches dot-prefixed names, and a build in progress must never
            # be evicted or counted toward the cap.
            if p.name.startswith("."):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        if total <= max_bytes:
            return removed
        entries.sort(key=lambda e: e[0])  # oldest first
        for _mtime, size, p in entries:
            if total <= max_bytes:
                break
            with contextlib.suppress(OSError):
                p.unlink()
                total -= size
                removed.append(p)
    except OSError:
        pass
    return removed
