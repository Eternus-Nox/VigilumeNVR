"""Frame ingest — ffmpeg rawvideo pipes -> latest-frame slots -> inference.

Design doc §3: per detect-enabled camera one ffmpeg child decodes the
substream via Vigilume's go2rtc restream (``rtsp://go2rtc:8554/{name}_sub``)
at ``detect_fps``, writing raw BGR24 frames to stdout. The reader keeps a
size-1 latest-frame slot per camera (**drop, never queue** — inference
always sees the freshest frame), applies a 15 s staleness watchdog
(kill + respawn on silent stalls), and respawns dead children with
2 s -> 60 s backoff. Process death IS the error signal.

A single ``IngestManager`` worker services all cameras: for each fresh
frame it runs ``detector.detect`` in a thread, updates the camera's
``ByteTrackTracker`` (trackers==2.4.0 — one instance PER CAMERA), and
awaits ``engine.process`` on the event loop. Frames are processed even
while the detector is not ready (empty observations) so the live-snapshot
frame cache and ingest stats keep working.

ffmpeg absence (dev Macs) is feature-detected once per boot: a warning is
logged and no sources spawn; everything else keeps running.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from ..config import effective_detect_mode
from .engine import observations_from_supervision
from .streams import resolve_urls, sub_stream_name

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from .detector import OnnxDetector
    from .engine import DetectionEngine

log = logging.getLogger(__name__)

STALL_TIMEOUT_S = 15.0     # no frame this long => kill + respawn (watchdog)
BACKOFF_INITIAL_S = 2.0
BACKOFF_CAP_S = 60.0
HEALTHY_RESET_S = 30.0     # a connection that lived this long resets backoff
_KILL_WAIT_S = 5.0
STARVATION_WARN_S = 90.0   # no frame for this long across respawns => loud WARNING

# --- self-heal tunables (module-level so tests can monkeypatch) ---
# The worker wakes at least this often even with no frames so the engine keeps
# getting ticked (absence-based event ending) during an ingest stall. A source
# with a frame older than this also gets an empty-observation heartbeat tick.
HEARTBEAT_TICK_S = 1.0
# One detect() call may block the single worker at most this long; on timeout
# the frame yields empty observations and the detector is flagged for reinit.
DETECT_TIMEOUT_S = 8.0
# How often the reinit supervisor checks the detector's needs-reinit flag.
REINIT_SUPERVISOR_S = 5.0


def build_ingest_args(input_url: str, detect_fps: int, width: int, height: int) -> list[str]:
    """ffmpeg argv for the rawvideo detect pipe (design doc §3.1).

    The explicit ``scale`` keeps the pipe byte-exact at
    ``width*height*3`` per frame even when a camera's substream resolution
    differs from the stored detect dims (boxes are decoded into the same
    detect-pixel space, so geometry stays consistent).
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", input_url,
        "-vf", f"fps={detect_fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


async def _kill_process(proc: Optional[asyncio.subprocess.Process]) -> None:
    if proc is None or proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=_KILL_WAIT_S)
    except asyncio.TimeoutError:  # pragma: no cover — kill() is not ignorable
        log.warning("ffmpeg ingest child did not exit after kill")


class FrameSource:
    """One camera's ffmpeg ingest child + latest-frame slot + watchdog."""

    def __init__(
        self,
        name: str,
        url: str,
        width: int,
        height: int,
        detect_fps: int,
        on_frame: Callable[[], None],
        ffmpeg: str = "ffmpeg",
        stall_timeout_s: float = STALL_TIMEOUT_S,
        backoff_initial_s: float = BACKOFF_INITIAL_S,
        backoff_cap_s: float = BACKOFF_CAP_S,
    ):
        self.name = name
        self.url = url
        self.width = width
        self.height = height
        self.detect_fps = detect_fps
        self._frame_bytes = width * height * 3
        self._on_frame = on_frame
        self._ffmpeg = ffmpeg
        self._stall_timeout_s = stall_timeout_s
        self._backoff_initial_s = backoff_initial_s
        self._backoff_cap_s = backoff_cap_s
        self._latest: Optional[tuple[np.ndarray, float]] = None
        self.spawn_count = 0  # observability + tests
        # Monotonic clock of the last frame actually read off the pipe (survives
        # take_latest, unlike _latest). None until the source ever produces a
        # frame. Drives the frame-starvation WARNING + the /detector status.
        self.last_frame_monotonic: Optional[float] = None
        self._run_started_monotonic: Optional[float] = None
        self._starvation_warned_at: float = 0.0

    def last_frame_age_s(self) -> Optional[float]:
        """Seconds since the last frame was read off the pipe (None = never)."""
        if self.last_frame_monotonic is None:
            return None
        return time.monotonic() - self.last_frame_monotonic

    def stalled(self) -> bool:
        """True when no frame has arrived for longer than the watchdog window
        (or the source has never produced a frame)."""
        age = self.last_frame_age_s()
        return age is None or age >= self._stall_timeout_s

    def take_latest(self) -> Optional[tuple[np.ndarray, float]]:
        """Pop the latest (frame_bgr, epoch_time); None when nothing new.
        Ownership of the frame passes to the caller (drop-not-queue slot)."""
        item = self._latest
        self._latest = None
        return item

    async def _spawn(self) -> asyncio.subprocess.Process:
        """Start the ffmpeg child (separate method so tests can fake it)."""
        return await asyncio.create_subprocess_exec(
            *build_ingest_args(self.url, self.detect_fps, self.width, self.height),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def run(self) -> None:
        """Ingest loop: spawn -> read frames -> (stall|exit) -> backoff -> respawn.
        Runs until cancelled; never raises anything else."""
        backoff = self._backoff_initial_s
        self._run_started_monotonic = time.monotonic()
        while True:
            proc: Optional[asyncio.subprocess.Process] = None
            started = time.monotonic()
            try:
                argv0 = self._ffmpeg
                proc = await self._spawn()
                self.spawn_count += 1
                log.info(
                    "ingest %s: ffmpeg started (pid=%s, %dx%d @ %d fps)",
                    self.name, getattr(proc, "pid", "?"), self.width, self.height,
                    self.detect_fps,
                )
                assert proc.stdout is not None
                while True:
                    chunk = await asyncio.wait_for(
                        proc.stdout.readexactly(self._frame_bytes),
                        timeout=self._stall_timeout_s,
                    )
                    frame = (
                        np.frombuffer(chunk, dtype=np.uint8)
                        .reshape(self.height, self.width, 3)
                        .copy()
                    )
                    self._latest = (frame, time.time())
                    self.last_frame_monotonic = time.monotonic()
                    self._on_frame()
            except asyncio.CancelledError:
                await _kill_process(proc)
                raise
            except asyncio.TimeoutError:
                log.warning(
                    "ingest %s: no frame for %.0f s — restarting ffmpeg (watchdog)",
                    self.name, self._stall_timeout_s,
                )
            except asyncio.IncompleteReadError:
                log.warning("ingest %s: ffmpeg exited (stream ended/unreachable)", self.name)
            except FileNotFoundError:
                log.error("ingest %s: %r not found — ingest disabled for this source", self.name, argv0)
                return
            except Exception:  # noqa: BLE001 — ingest must survive anything
                log.exception("ingest %s: reader crashed — restarting ffmpeg", self.name)
            await _kill_process(proc)
            self._warn_if_starved()
            if time.monotonic() - started >= HEALTHY_RESET_S:
                backoff = self._backoff_initial_s
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._backoff_cap_s)

    def _warn_if_starved(self) -> None:
        """Emit a throttled WARNING when this source has produced no frame for
        a long stretch despite the watchdog's respawns — makes a dead camera /
        go2rtc sub-stream visible instead of silently starving detection."""
        now = time.monotonic()
        ref = self.last_frame_monotonic
        if ref is None:
            ref = self._run_started_monotonic or now
        age = now - ref
        if age >= STARVATION_WARN_S and now - self._starvation_warned_at >= STARVATION_WARN_S:
            self._starvation_warned_at = now
            log.warning(
                "ingest %s: no frames for %.0fs despite %d respawns — "
                "check the camera/go2rtc sub-stream",
                self.name, age, self.spawn_count,
            )


def _make_tracker(detect_fps: int) -> Any:
    """One ByteTrackTracker per camera (pinned trackers==2.4.0 signature,
    design doc §1.4). Imported lazily: the trackers import is heavyweight
    and not needed by web-only code paths."""
    from trackers import ByteTrackTracker

    return ByteTrackTracker(
        lost_track_buffer=25,
        frame_rate=float(detect_fps),
        track_activation_threshold=0.5,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
    )


class IngestManager:
    """Per-camera FrameSources + the single inference worker.

    Owned by the DetectionEngine (created in ``engine.start()``); ``reload``
    reconciles running sources against the detect-enabled camera rows and is
    called at start, after camera CRUD, and after settings changes.
    """

    def __init__(
        self,
        engine: "DetectionEngine",
        detector: "OnnxDetector",
        config: "Config",
        ffmpeg_path: Optional[str] = "auto",
        settings: Optional[Any] = None,
    ):
        self._engine = engine
        self._detector = detector
        self._config = config
        # The SettingsStore, purely so this manager can read the live Privacy
        # Mode set (settings.is_private) in BOTH the reload gate and the
        # per-frame gate. Optional so existing tests that construct an
        # IngestManager directly keep working; None reads as "nothing private",
        # which is fail-open by design — a test harness must never be silently
        # blacked out, and production always passes the real store (engine.py).
        self._settings = settings
        # "auto" (default) feature-detects ffmpeg on PATH; tests pass an
        # explicit path or None (= simulate an ffmpeg-less host).
        self._ffmpeg = shutil.which("ffmpeg") if ffmpeg_path == "auto" else ffmpeg_path
        self._ffmpeg_warned = False
        # A fresh frame from any source sets this Event to wake the single
        # inference worker.
        self._wake = asyncio.Event()
        self._sources: dict[str, FrameSource] = {}
        self._source_tasks: dict[str, asyncio.Task] = {}
        self._trackers: dict[str, Any] = {}
        self._cams: dict[str, dict[str, Any]] = {}
        self._keys: dict[str, tuple] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._detect_error_logged: dict[str, float] = {}
        # Monotonic time each source last delivered a FRESH frame to the worker.
        # Drives the empty-observation heartbeat: a source stale for longer than
        # HEARTBEAT_TICK_S gets ticked so its open events end by absence.
        self._last_frame_mono: dict[str, float] = {}
        # Camera-AI event listener (set via set_ai_events); consulted per frame
        # to gate GPU inference for cameras in "camera_ai" mode. None-safe.
        self._ai_events: Optional[Any] = None
        # settings.detection.default_mode applied to cameras with detect_mode
        # unset/NULL. Refreshed on every reload().
        self._default_mode: str = "always"
        # name -> monotonic time the watcher-down failsafe last logged, so the
        # per-frame gate logs the failsafe engaging at most once/minute/camera.
        self._failsafe_logged: dict[str, float] = {}

    def set_ai_events(self, ai_events: Optional[Any]) -> None:
        """Wire the camera-AI event listener used to gate ``camera_ai`` cameras.
        Set once at boot (main.py) after the listener is constructed."""
        self._ai_events = ai_events

    def _is_private(self, camera: str) -> bool:
        """Live Privacy Mode read (app/privacy.py). Deliberately re-read on every
        call rather than cached per source: a toggle must take effect on the very
        next frame, not at the next reload."""
        return self._settings is not None and self._settings.is_private(camera)

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="ingest-worker")
        if self._supervisor_task is None:
            self._supervisor_task = asyncio.create_task(
                self._supervisor(), name="ingest-supervisor"
            )

    async def stop(self) -> None:
        for name in list(self._sources):
            await self._stop_source(name)
        tasks = []
        if self._worker_task is not None:
            tasks.append(self._worker_task)
        if self._supervisor_task is not None:
            tasks.append(self._supervisor_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._worker_task = None
        self._supervisor_task = None

    async def reload(
        self, cameras: list[dict[str, Any]], default_mode: str = "always"
    ) -> None:
        """Reconcile sources/trackers with the detect-enabled camera rows.
        A change in a camera's stream URL, fps or detect dims restarts just
        that camera's ffmpeg child."""
        self._default_mode = default_mode or "always"
        # Detect-enabled AND has at least one wanted object. A detect-enabled
        # camera with an EMPTY detect_objects records only (detects nothing) —
        # it must run NO ingest ffmpeg/inference at all (no GPU, no detect
        # sub-stream session), so exclude it here. Emptying the object list via
        # PUT flows through _apply_camera_change -> engine.reload -> this
        # reload, which stops the source; re-adding an object restarts it.
        #
        # A "camera_ai_only" camera also runs NO server inference at all (its
        # events come straight from the camera's AI stream), so it too spawns no
        # ffmpeg ingest source. "camera_ai" DOES spawn a source (frames flow so
        # live view / frame cache work) — only the per-frame detect() call is
        # gated on the camera's AI being active (see _process_frame).
        wanted: dict[str, dict[str, Any]] = {
            cam["name"]: cam
            for cam in cameras
            if bool(cam.get("detect_enabled", True))
            and (cam.get("detect_objects") or [])
            and effective_detect_mode(
                cam.get("detect_mode"),
                self._default_mode,
                ai_on_camera=bool((cam.get("capabilities") or {}).get("ai_on_camera")),
            )
            != "camera_ai_only"
        }
        # PRIVACY MODE GATE (app/privacy.py). Drop private cameras before the
        # reconcile loop below, so their decode ffmpeg is killed and none is
        # respawned: no frames pulled, no GPU inference, no go2rtc detect-substream
        # session. NOT folded into effective_detect_mode() — that helper's contract
        # is "an unknown value never disables detection", the opposite of what a
        # privacy gate must do.
        private_now = {n for n in wanted if self._is_private(n)}
        if private_now:
            log.info("ingest: privacy mode — no capture for %s", ", ".join(sorted(private_now)))
            wanted = {n: c for n, c in wanted.items() if n not in private_now}
        if wanted and self._ffmpeg is None:
            if not self._ffmpeg_warned:
                log.error(
                    "ffmpeg not found on PATH — frame ingest/detection disabled "
                    "(%d detect-enabled cameras)", len(wanted),
                )
                self._ffmpeg_warned = True
            wanted = {}

        for name in list(self._sources):
            cam = wanted.get(name)
            if cam is None or self._source_key(cam) != self._keys.get(name):
                await self._stop_source(name)
        for name, cam in wanted.items():
            self._cams[name] = cam
            if name not in self._sources:
                self._start_source(cam)
        for name in list(self._cams):
            if name not in wanted:
                del self._cams[name]

    # ---------- source management ----------

    @staticmethod
    def _detect_dims(cam: dict[str, Any]) -> tuple[int, int]:
        width = int(cam.get("detect_width") or 704)
        height = int(cam.get("detect_height") or 480)
        return max(width, 2), max(height, 2)

    def _sub_url(self, cam: dict[str, Any]) -> str:
        return f"{self._config.go2rtc_rtsp_url}/{sub_stream_name(cam['name'])}"

    def _source_key(self, cam: dict[str, Any]) -> tuple:
        width, height = self._detect_dims(cam)
        _, sub = resolve_urls(cam)  # override changes must respawn ffmpeg
        return (self._sub_url(cam), sub, int(cam.get("detect_fps") or 5), width, height)

    def _start_source(self, cam: dict[str, Any]) -> None:
        name = cam["name"]
        width, height = self._detect_dims(cam)
        fps = int(cam.get("detect_fps") or 5)
        source = FrameSource(
            name=name,
            url=self._sub_url(cam),
            width=width,
            height=height,
            detect_fps=fps,
            on_frame=self._wake.set,
            ffmpeg=self._ffmpeg or "ffmpeg",
        )
        self._sources[name] = source
        self._keys[name] = self._source_key(cam)
        self._trackers[name] = _make_tracker(fps)
        # Seed to "now" so a freshly-started source isn't treated as stale (and
        # doesn't emit a spurious heartbeat tick) during its first second.
        self._last_frame_mono[name] = time.monotonic()
        self._source_tasks[name] = asyncio.create_task(source.run(), name=f"ingest-{name}")

    async def _stop_source(self, name: str) -> None:
        task = self._source_tasks.pop(name, None)
        self._sources.pop(name, None)
        self._keys.pop(name, None)
        self._trackers.pop(name, None)
        self._last_frame_mono.pop(name, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ---------- the single inference worker ----------

    async def _worker(self) -> None:
        """The single inference worker with a heartbeat.

        Blocks on the wake Event with a HEARTBEAT_TICK_S timeout so it can never
        wedge when frames stop flowing: a source with a FRESH frame is processed
        (detect -> track -> engine), while a source that has gone stale is ticked
        with EMPTY observations so the engine still ends its open events by
        absence. Stale ticks re-run NO inference (no wasted/duplicate detect, no
        busy-spin — the 1 s timeout bounds CPU)."""
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=HEARTBEAT_TICK_S)
            except asyncio.TimeoutError:
                pass  # heartbeat: fall through and tick the engine anyway
            except asyncio.CancelledError:
                raise
            self._wake.clear()
            now_mono = time.monotonic()
            for name in list(self._sources):
                source = self._sources.get(name)
                cam = self._cams.get(name)
                if source is None or cam is None:
                    continue
                item = source.take_latest()
                try:
                    if item is not None:
                        self._last_frame_mono[name] = now_mono
                        frame, frame_time = item
                        await self._process_frame(name, cam, frame, frame_time)
                    else:
                        await self._heartbeat_tick(name, now_mono)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad frame never stops ingest
                    log.exception("ingest worker: processing frame for %s failed", name)

    async def _heartbeat_tick(self, name: str, now_mono: float) -> None:
        """Feed the engine an empty observation set for a source that has no
        fresh frame, once it has been stale for at least HEARTBEAT_TICK_S. This
        keeps absence-based event ending alive during an ingest stall without
        re-running inference on the already-processed frame or touching the
        camera's latest-frame cache (frame_bgr=None)."""
        last = self._last_frame_mono.get(name)
        if last is not None and now_mono - last < HEARTBEAT_TICK_S:
            return  # frames are still flowing (sub-second gap) — nothing to do
        await self._engine.process(name, time.time(), [], frame_bgr=None)

    def _should_infer(self, name: str, cam: dict[str, Any]) -> bool:
        """Camera-AI gate: whether the GPU detector should run on this camera's
        current frame. Cheap (a mode string check + one dict lookup) so it runs
        per frame, not per detection:

        - ``always``          -> always infer (historical behavior).
        - ``camera_ai``       -> infer ONLY while the camera's on-board AI is
          active (an AI Start..Stop + cooldown window); otherwise idle the GPU
          for this camera. This is the load win. FAILSAFE: we trust the idle gate
          (GPU off) whenever the AI trigger is RELIABLE — a connected watcher, or
          one that dropped only briefly (a normal reconnect within the grace
          window). We only run detection despite idle when the trigger is
          genuinely unreliable: no listener wired, or the subscription never
          connected / has been down past the grace window (``failsafe_needed``).
          So a healthy AI camera keeps its full GPU savings, but a genuinely
          broken AI trigger can't leave the camera silently blind.
        - ``camera_ai_only``  -> never spawns a source (excluded in reload), so
          this branch is defensive: never infer.
        """
        mode = effective_detect_mode(
            cam.get("detect_mode"),
            self._default_mode,
            ai_on_camera=bool((cam.get("capabilities") or {}).get("ai_on_camera")),
        )
        if mode == "always":
            return True
        if mode == "camera_ai":
            ai = self._ai_events
            # No listener wired at all -> fail safe: detect.
            if ai is None:
                self._log_failsafe(name)
                return True
            # AI currently firing -> detect (the normal camera-AI trigger).
            if ai.is_active(name):
                return True
            # Idle. Trust the AI gate (idle the GPU — the load win) as long as
            # the trigger is RELIABLE: a connected watcher, or one that dropped
            # only briefly (a normal reconnect). Only run detection when the
            # trigger is genuinely unreliable — never connected, or down past the
            # grace window — so a healthy AI camera gates off but a broken one
            # can't go silently blind.
            if ai.failsafe_needed(name):
                self._log_failsafe(name)
                return True
            return False  # AI trigger reliable + genuinely idle -> idle the GPU
        return False  # camera_ai_only (defensive)

    def _log_failsafe(self, name: str) -> None:
        """Throttled (<=1/min/camera) log when the camera_ai watcher-down
        failsafe runs detection because the AI event watcher is not connected."""
        now = time.monotonic()
        if now - self._failsafe_logged.get(name, 0.0) > 60.0:
            self._failsafe_logged[name] = now
            log.warning(
                "camera_ai failsafe: %s AI event watcher not connected — "
                "running detection instead of gating the GPU off (dead trigger)",
                name,
            )

    async def _process_frame(
        self,
        name: str,
        cam: dict[str, Any],
        frame: np.ndarray,
        frame_time: float,
    ) -> None:
        # PRIVACY MODE GATE — return BEFORE inference AND before handing the
        # frame to the engine. The reload gate above kills the ffmpeg sources,
        # but a frame already sitting in the size-1 latest-slot at the instant of
        # the toggle would otherwise still run one more detect() and, worse,
        # refresh engine.latest_frame — which the snapshot/JPEG surface serves.
        # That is real footage captured after the operator turned Privacy on, so
        # this check is required, not defence in depth.
        if self._is_private(name):
            return
        width, height = self._detect_dims(cam)
        observations: list = []
        detector = self._detector
        # Camera-AI gate: skip inference entirely when the camera's mode says so
        # (camera_ai + AI idle). The frame is still fed to the engine below so the
        # live-snapshot cache / fps stats / absence-based event ending keep working
        # exactly as an empty-observation heartbeat would.
        if detector.ready and self._should_infer(name, cam):
            try:
                detections = await asyncio.wait_for(
                    asyncio.to_thread(detector.detect, frame, width, height),
                    timeout=DETECT_TIMEOUT_S,
                )
                tracker = self._trackers.get(name)
                if tracker is not None:
                    detections = tracker.update(detections)
                observations = observations_from_supervision(detections)
                detector.note_detect_ok()
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                # The to_thread detect() is abandoned; the underlying thread may
                # linger (threads can't be cancelled) — that is bounded by the
                # detector reinit the failure triggers. Do NOT block the worker.
                detector.note_detect_failure()
                now = time.monotonic()
                if now - self._detect_error_logged.get(name, 0.0) > 60.0:
                    self._detect_error_logged[name] = now
                    log.warning(
                        "detection timed out for %s after %.0fs — flagged detector reinit",
                        name, DETECT_TIMEOUT_S,
                    )
            except Exception as exc:  # noqa: BLE001 — keep the frame cache alive
                detector.note_detect_failure()
                now = time.monotonic()
                if now - self._detect_error_logged.get(name, 0.0) > 60.0:
                    self._detect_error_logged[name] = now
                    log.exception("detection failed for %s: %s", name, exc)
        # Always feed the engine — the latest-frame cache, fps stats and
        # absence-based event ending depend on empty frames too.
        await self._engine.process(name, frame_time, observations, frame_bgr=frame)

    # ---------- detector reinit supervisor ----------

    async def _supervisor(self) -> None:
        """Watch the detector's needs-reinit flag and rebuild it in the
        background (off the worker's hot path) when a detect timeout / failure
        storm or a device dropout flips it. reinit() is cooldown-guarded so this
        never thrashes; a failed reinit re-sets the flag and is retried on the
        next pass. Never crashes the app."""
        while True:
            try:
                await asyncio.sleep(REINIT_SUPERVISOR_S)
                if self._detector.needs_reinit:
                    await self._detector.reinit()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — supervisor must survive anything
                log.exception("ingest supervisor: detector reinit check failed")

    # ---------- status (routers/system.py per_camera health) ----------

    def source_stats(self) -> dict[str, dict[str, Any]]:
        """Per-camera ingest health for GET /api/system/detector:
        ``{name: {stalled, respawns, last_frame_age_s}}``. Covers running
        sources only; system.py merges it onto engine.camera_stats()."""
        out: dict[str, dict[str, Any]] = {}
        for name, source in self._sources.items():
            age = source.last_frame_age_s()
            out[name] = {
                "stalled": source.stalled(),
                "respawns": max(source.spawn_count - 1, 0),
                "last_frame_age_s": round(age, 2) if age is not None else None,
            }
        return out
