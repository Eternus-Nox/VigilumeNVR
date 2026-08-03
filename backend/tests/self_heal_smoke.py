"""Self-heal smoke suite — the detection path can never silently wedge.

The detection path is purely frame-driven with a SINGLE inference worker, so
before this work it went silent forever if frames stopped, the detector went
un-ready, or a detect() call hung. This suite exercises the four safety nets:

  (a) WORKER HEARTBEAT — the worker wakes on a bounded timeout, so a source
      that yields no fresh frame still gets ticked with EMPTY observations and
      the engine ends its open events by absence during an ingest stall. The
      idle loop is bounded (no busy-spin: ~one tick per HEARTBEAT_TICK_S).
  (b) PER-INFERENCE TIMEOUT — a hung detect() is abandoned by asyncio.wait_for
      without wedging the worker (a second camera keeps processing), and the
      run of failures flags the detector for reinit.
  (c) DETECTOR AUTO-REINIT — crossing the consecutive-failure threshold flags
      reinit; detector.reinit() rebuilds the ORT session (stubbed here) and
      flips ready true again, cooldown-guarded so it never thrashes; the ingest
      supervisor triggers reinit off the needs-reinit flag.
  (d) STATUS — detector status exposes the self-heal counters + reinit age, and
      per_camera carries per-source stalled + last_frame_age + respawns.

CPU-only, no real model needed. Usage: python backend/tests/self_heal_smoke.py
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Clean env before app config is instantiated (same guard as the sibling suites).
for _i in (1, 2, 3):
    for _suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{_i}_{_suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-selfheal-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402

import app.native.ingest as ingest_module  # noqa: E402
from app.config import Config  # noqa: E402
from app.native.detector import (  # noqa: E402
    DETECT_FAILURE_THRESHOLD,
    REINIT_COOLDOWN_S,
    OnnxDetector,
)
from app.native.engine import (  # noqa: E402
    ABSENCE_TIMEOUT_S,
    DetectionEngine,
    Observation,
    _CameraState,
)
from app.native.ingest import FrameSource, IngestManager  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------------------------------------------------------------------
# shared fakes
# ---------------------------------------------------------------------------


class RecordingPipeline:
    """Minimal EventsPipeline stand-in: records payloads + live counts."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.counts: dict[tuple[str, str], int] = {}

    async def handle_event(self, payload: dict) -> None:
        self.events.append(payload)

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = count


class QuietDetector:
    """Detector that is never asked to detect (source yields no frames)."""

    def __init__(self) -> None:
        self.ready = False
        self.needs_reinit = False

    def note_detect_ok(self) -> None:  # pragma: no cover - unused here
        pass

    def note_detect_failure(self) -> None:  # pragma: no cover - unused here
        pass

    async def reinit(self, *, force: bool = False) -> bool:  # pragma: no cover
        return True


class HangDetector:
    """Ready detector whose detect() wedges (``hang_fill=None`` -> every frame,
    modelling a detector-wide CUDA hang; else only frames of that fill). Uses
    the REAL consecutive-failure/needs-reinit contract so the worker's timeout
    path flips the reinit flag for real."""

    def __init__(self, hang_s: float, hang_fill=None) -> None:
        self.ready = True
        self.hang_fill = hang_fill
        self.hang_s = hang_s
        self._consecutive_failures = 0
        self._needs_reinit = False
        self.reinit_calls = 0

    def detect(self, frame: np.ndarray, dw: int, dh: int) -> sv.Detections:
        if self.hang_fill is None or int(frame[0, 0, 0]) == self.hang_fill:
            time.sleep(self.hang_s)  # simulate a wedged inference call
        return sv.Detections.empty()

    def note_detect_ok(self) -> None:
        self._consecutive_failures = 0

    def note_detect_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= DETECT_FAILURE_THRESHOLD:
            self._needs_reinit = True

    @property
    def needs_reinit(self) -> bool:
        return self._needs_reinit

    async def reinit(self, *, force: bool = False) -> bool:
        self.reinit_calls += 1
        self._needs_reinit = False
        return True


class StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def process(self, camera, frame_time, observations, frame_bgr=None):
        self.calls.append((camera, frame_time, list(observations), frame_bgr))


class NoneSource:
    """FrameSource replacement whose slot is always empty (frames stopped)."""

    def __init__(self, name, url, width, height, detect_fps, on_frame, ffmpeg, **kw):
        self.name, self.url = name, url
        self.width, self.height, self.detect_fps = width, height, detect_fps
        self.last_frame_monotonic = None
        self.spawn_count = 1

    def take_latest(self):
        return None

    def last_frame_age_s(self):
        return None

    def stalled(self) -> bool:
        return True

    async def run(self):
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)


class LiveSource:
    """FrameSource replacement that always hands out a fresh fixed-fill frame."""

    def __init__(self, name, url, width, height, detect_fps, on_frame, ffmpeg, **kw):
        self.name, self.url = name, url
        self.width, self.height, self.detect_fps = width, height, detect_fps
        self.fill = kw.get("fill", 0)
        self.last_frame_monotonic = time.monotonic()
        self.spawn_count = 1

    def take_latest(self):
        frame = np.full((self.height, self.width, 3), self.fill, dtype=np.uint8)
        return (frame, time.time())

    def last_frame_age_s(self):
        return 0.0

    def stalled(self) -> bool:
        return False

    async def run(self):
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)


def _cam_row(name: str, **over) -> dict:
    row = {
        "name": name, "detect_enabled": True, "detect_fps": 5,
        "detect_width": 8, "detect_height": 6,
        # Non-empty by default: a normal detecting camera. An empty
        # detect_objects now means record-only (no ingest source).
        "detect_objects": ["person"],
        "ip": "10.0.0.9", "username": "u", "password": "p",
        "main_url": "", "sub_url": "",
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# (a) worker heartbeat ends open events by absence during an ingest stall
# ---------------------------------------------------------------------------


async def _heartbeat_case() -> None:
    print("self-heal (a): worker heartbeat ticks the engine during an ingest stall")

    engine = DetectionEngine(db=None, detector=QuietDetector(), recorder=None,
                             settings=None, config=Config())
    pipeline = RecordingPipeline()
    engine.set_pipeline(pipeline)
    # Inject a detect-enabled camera + confirm a 'person' track by hand (no db).
    engine._cameras["cam"] = _CameraState(row=_cam_row("cam", detect_objects=["person"],
                                                        record_enabled=False))
    t0 = time.time()
    box = (10.0, 10.0, 40.0, 40.0)
    for i in range(4):  # >= MIN_HITS frames carrying tracker_id 5 -> confirmed
        obs = [Observation("person", 5, 0.9, box)]
        await engine.process("cam", t0 + i * 0.2, obs, frame_bgr=None)
    st = engine._events.get(("cam", "person"))
    check(st is not None, "a confirmed person track opened an event")
    check(pipeline.counts.get(("cam", "person")) == 1, "live count is 1 while the event is open")

    # Age the event so the NEXT empty heartbeat tick ends it by absence.
    st.last_seen = time.time() - (ABSENCE_TIMEOUT_S + 5.0)

    # Spy the engine's process so we can count heartbeat (empty, frame=None) ticks.
    orig_process = engine.process
    hb_calls: list[float] = []

    async def spy(camera, frame_time, observations, frame_bgr=None):
        if not observations and frame_bgr is None:
            hb_calls.append(time.monotonic())
        return await orig_process(camera, frame_time, observations, frame_bgr=frame_bgr)

    engine.process = spy  # type: ignore[method-assign]

    real_source = ingest_module.FrameSource
    saved_tick = ingest_module.HEARTBEAT_TICK_S
    ingest_module.FrameSource = NoneSource  # type: ignore[misc]
    ingest_module.HEARTBEAT_TICK_S = 0.2  # tick fast so the test stays quick
    try:
        mgr = IngestManager(engine, QuietDetector(), Config(), ffmpeg_path="/fake/ffmpeg")
        await mgr.start()
        await mgr.reload([_cam_row("cam", detect_objects=["person"], record_enabled=False)])
        check(set(mgr._sources) == {"cam"}, "a source spawned for the stalled camera")

        # Wait for the heartbeat to end the event by absence.
        deadline = time.monotonic() + 3.0
        while ("cam", "person") in engine._events and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        check(("cam", "person") not in engine._events,
              "heartbeat empty-observation ticks ended the open event by absence")
        check(pipeline.counts.get(("cam", "person")) == 0, "live count zeroed on the absence end")
        ended = [e for e in pipeline.events if e["type"] == "end"]
        check(len(ended) == 1 and ended[0]["after"]["camera"] == "cam",
              "an 'end' payload was emitted for the stalled camera")
        check(len(hb_calls) >= 1, "the worker fed the engine EMPTY observations (frame_bgr=None)")

        # No busy-spin: over a ~1 s idle window the heartbeat fires ~1/tick, not
        # thousands of times (the wait_for timeout bounds the loop).
        hb_calls.clear()
        window = 1.0
        await asyncio.sleep(window)
        rate = len(hb_calls)
        check(1 <= rate <= 12,
              f"idle heartbeat is bounded (~1 tick / {ingest_module.HEARTBEAT_TICK_S}s), "
              f"not a busy-spin (saw {rate} ticks in {window}s)")

        await mgr.stop()
    finally:
        ingest_module.FrameSource = real_source  # type: ignore[misc]
        ingest_module.HEARTBEAT_TICK_S = saved_tick


# ---------------------------------------------------------------------------
# (b) a hung detect is abandoned by the timeout; other cameras keep processing
# ---------------------------------------------------------------------------


async def _hang_timeout_case() -> None:
    print("self-heal (b): a hung detect() is abandoned; the other camera keeps processing")

    engine = StubEngine()
    # Detector-wide hang (every detect() wedges) — the CUDA-hang failure mode
    # that must trigger reinit; a per-camera-only hang would (correctly) reset
    # on the healthy camera's success and never flag reinit.
    detector = HangDetector(hang_s=0.4)

    real_source = ingest_module.FrameSource
    saved_timeout = ingest_module.DETECT_TIMEOUT_S
    saved_sup = ingest_module.REINIT_SUPERVISOR_S

    def make_source(name, url, width, height, detect_fps, on_frame, ffmpeg, **kw):
        return LiveSource(name, url, width, height, detect_fps, on_frame, ffmpeg,
                          fill=1 if name == "A" else 2)

    ingest_module.FrameSource = make_source  # type: ignore[misc]
    ingest_module.DETECT_TIMEOUT_S = 0.15   # abandon the hung detect fast
    ingest_module.REINIT_SUPERVISOR_S = 100.0  # keep the reinit flag observable
    try:
        mgr = IngestManager(engine, detector, Config(), ffmpeg_path="/fake/ffmpeg")
        await mgr.start()
        await mgr.reload([_cam_row("A"), _cam_row("B")])
        check(set(mgr._sources) == {"A", "B"}, "both camera sources are running")

        # Drive several ticks: A wedges every tick, B returns immediately.
        # The single worker services both cameras.
        for _ in range(5):
            mgr._wake.set()
            await asyncio.sleep(ingest_module.DETECT_TIMEOUT_S + 0.1)

        check(not mgr._worker_task.done(), "the worker is still alive (a hung detect never wedged it)")
        cams_processed = {c[0] for c in engine.calls}
        check("B" in cams_processed and "A" in cams_processed,
              "both cameras keep reaching the engine (worker not wedged behind the hung detect)")
        check(all(c[2] == [] for c in engine.calls),
              "an abandoned detect yields EMPTY observations (frame cache/ingest stays alive)")
        check(detector._consecutive_failures >= DETECT_FAILURE_THRESHOLD,
              "repeated detect timeouts accumulated as consecutive failures")
        check(detector.needs_reinit is True,
              "crossing the failure threshold flagged the detector for reinit")

        await mgr.stop()
    finally:
        ingest_module.FrameSource = real_source  # type: ignore[misc]
        ingest_module.DETECT_TIMEOUT_S = saved_timeout
        ingest_module.REINIT_SUPERVISOR_S = saved_sup


# ---------------------------------------------------------------------------
# (c) reinit rebuilds the session + flips ready; cooldown prevents thrash;
#     the supervisor triggers reinit off the needs-reinit flag
# ---------------------------------------------------------------------------


def _stub_bootstrap(det: OnnxDetector, counter: dict, *, succeed: bool):
    """Replace the heavy ORT bootstrap with a stub so no real model is needed."""

    async def fake_bootstrap() -> bool:
        counter["n"] += 1
        det._ready = succeed
        det._device = "cpu" if succeed else None
        if succeed:
            det._consecutive_failures = 0
            det._needs_reinit = False
        return True

    det._bootstrap_once = fake_bootstrap  # type: ignore[method-assign]


async def _reinit_case() -> None:
    print("self-heal (c): detector reinit rebuilds the session, cooldown-guarded")

    det = OnnxDetector(models_dir=TMP / "models", model_key="dfine_s",
                       confidence=0.5, require_gpu=False)
    det._started = True
    calls: dict[str, int] = {"n": 0}
    _stub_bootstrap(det, calls, succeed=True)

    # A run of detect failures flags reinit.
    for _ in range(DETECT_FAILURE_THRESHOLD):
        det.note_detect_failure()
    check(det.needs_reinit is True and det.consecutive_failures == DETECT_FAILURE_THRESHOLD,
          "consecutive detect failures crossing the threshold flag reinit")
    check(det.ready is False, "detector is not-ready before reinit")

    ok = await det.reinit()
    check(ok is True and det.ready is True, "reinit rebuilt the session and flipped ready true")
    check(det.needs_reinit is False and det.consecutive_failures == 0,
          "a successful reinit clears the failure run + reinit flag")
    check(calls["n"] == 1, "reinit rebuilt the ORT session exactly once")
    age = det.last_reinit_age_s()
    check(age is not None and age >= 0.0, "last_reinit_age_s is populated after a reinit")

    # Cooldown: an immediate second reinit is a no-op (anti-thrash).
    await det.reinit()
    check(calls["n"] == 1, "a second reinit within the cooldown does NOT rebuild (no thrash)")
    # force bypasses the cooldown.
    await det.reinit(force=True)
    check(calls["n"] == 2, "reinit(force=True) bypasses the cooldown")
    check(REINIT_COOLDOWN_S >= 20.0, "reinit cooldown is at least 20 s")

    # A reinit that cannot restore readiness stays not-ready + re-arms the flag.
    det_fail = OnnxDetector(models_dir=TMP / "models", model_key="dfine_s",
                            confidence=0.5, require_gpu=False)
    det_fail._started = True
    fcalls: dict[str, int] = {"n": 0}
    _stub_bootstrap(det_fail, fcalls, succeed=False)
    got = await det_fail.reinit(force=True)
    check(got is False and det_fail.ready is False,
          "a reinit that can't restore the device stays ready:false")
    check(det_fail.needs_reinit is True,
          "a failed reinit re-arms needs_reinit so the supervisor retries after cooldown")

    # The ingest supervisor triggers reinit off the flag (no worker involvement).
    saved_sup = ingest_module.REINIT_SUPERVISOR_S
    ingest_module.REINIT_SUPERVISOR_S = 0.1
    det_sup = OnnxDetector(models_dir=TMP / "models", model_key="dfine_s",
                           confidence=0.5, require_gpu=False)
    det_sup._started = True
    scalls: dict[str, int] = {"n": 0}
    _stub_bootstrap(det_sup, scalls, succeed=True)
    for _ in range(DETECT_FAILURE_THRESHOLD):
        det_sup.note_detect_failure()
    check(det_sup.needs_reinit is True and det_sup.ready is False,
          "detector flagged + not-ready before the supervisor runs")
    try:
        mgr = IngestManager(StubEngine(), det_sup, Config(), ffmpeg_path="/fake/ffmpeg")
        await mgr.start()
        deadline = time.monotonic() + 3.0
        while not det_sup.ready and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        check(det_sup.ready is True and det_sup.needs_reinit is False,
              "the ingest supervisor picked up the flag and reinited the detector")
        await mgr.stop()
    finally:
        ingest_module.REINIT_SUPERVISOR_S = saved_sup


# ---------------------------------------------------------------------------
# (d) status exposes per-camera stalled/last_frame_age + detector reinit age
# ---------------------------------------------------------------------------


async def _status_case() -> None:
    print("self-heal (d): status exposes ingest + reinit health")
    from app.routers.system import _per_camera_health

    # detector.status() carries the self-heal block.
    fresh = OnnxDetector(models_dir=TMP / "models", model_key="dfine_s",
                         confidence=0.5, require_gpu=False)
    s = fresh.status()
    check({"consecutive_failures", "needs_reinit", "last_reinit_age_s"} <= set(s),
          "detector.status() carries consecutive_failures/needs_reinit/last_reinit_age_s")
    check(s["last_reinit_age_s"] is None and s["consecutive_failures"] == 0,
          "a never-reinited detector reports last_reinit_age_s=None, 0 failures")
    fresh._started = True
    _stub_bootstrap(fresh, {"n": 0}, succeed=True)
    await fresh.reinit(force=True)
    check(isinstance(fresh.status()["last_reinit_age_s"], float),
          "after a reinit, status().last_reinit_age_s is a float age")

    # FrameSource frame-age / stalled semantics.
    src = FrameSource(name="c", url="rtsp://x/c_sub", width=8, height=6,
                      detect_fps=5, on_frame=lambda: None, ffmpeg="/fake/ffmpeg")
    check(src.last_frame_age_s() is None and src.stalled() is True,
          "a source that never produced a frame is stalled with age None")
    src.last_frame_monotonic = time.monotonic()
    src.spawn_count = 3
    check(src.stalled() is False and (src.last_frame_age_s() or 0.0) < 1.0,
          "a source with a recent frame is not stalled and reports a small age")

    mgr = IngestManager(StubEngine(), QuietDetector(), Config(), ffmpeg_path="/fake/ffmpeg")
    mgr._sources["c"] = src  # type: ignore[assignment]
    stats = mgr.source_stats()
    check(stats["c"]["stalled"] is False and stats["c"]["respawns"] == 2
          and isinstance(stats["c"]["last_frame_age_s"], float),
          "source_stats() reports {stalled, respawns, last_frame_age_s} (respawns = spawns-1)")

    # system.py merges engine.camera_stats() with the ingest source stats.
    class _FakeIngest:
        def source_stats(self):
            return {"cam1": {"stalled": True, "respawns": 2, "last_frame_age_s": 30.0}}

    class _FakeEngine:
        _ingest = _FakeIngest()

        def camera_stats(self):
            return [
                {"name": "cam1", "ingest_ok": False, "fps": 0.0, "last_frame_age_s": 30.0},
                {"name": "cam2", "ingest_ok": True, "fps": 5.0, "last_frame_age_s": 0.2},
            ]

    class _FakeState:
        engine = _FakeEngine()

    merged = _per_camera_health(_FakeState())
    by_name = {c["name"]: c for c in merged}
    check(by_name["cam1"]["stalled"] is True and by_name["cam1"]["respawns"] == 2
          and by_name["cam1"]["fps"] == 0.0 and by_name["cam1"]["last_frame_age_s"] == 30.0,
          "per_camera merges stalled+respawns while preserving fps/last_frame_age_s")
    check(by_name["cam2"]["stalled"] is True and by_name["cam2"]["respawns"] == 0,
          "a camera with no running source degrades to stalled with 0 respawns")


def _endpoint_case() -> None:
    """The live GET /api/system/detector wires the new self-heal fields."""
    print("self-heal (d): GET /api/system/detector exposes the self-heal block")
    from fastapi.testclient import TestClient

    import app.main

    with TestClient(app.main.app) as client:
        token = client.post("/api/auth/login", json={"password": "test-password"}).json()["token"]
        r = client.get("/api/system/detector", headers={"Authorization": f"Bearer {token}"})
        check(r.status_code == 200, "GET /api/system/detector -> 200")
        body = r.json()
        check({"consecutive_failures", "needs_reinit", "last_reinit_age_s"} <= set(body),
              "endpoint surfaces the top-level self-heal detector fields")
        check(isinstance(body["per_camera"], list), "per_camera is a list (empty with no cameras)")


def main() -> None:
    asyncio.run(_heartbeat_case())
    asyncio.run(_hang_timeout_case())
    asyncio.run(_reinit_case())
    asyncio.run(_status_case())
    _endpoint_case()
    print(f"\nALL {PASS} CHECKS PASSED (self-heal: heartbeat + timeout + reinit + status)")


if __name__ == "__main__":
    main()
