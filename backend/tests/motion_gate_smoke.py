"""Smoke suite for the ingest motion gate (docs/CONTRACTS.md, "Motion gate").

A frame that looks like the one inference last ran on is not run through the
detector again; the last observations are re-used instead. Checked here:
  - a repeated still frame skips inference and re-uses the last observations;
  - a frame with movement runs inference at once;
  - inference is forced again once MOTION_FORCE_S has passed;
  - sensor-noise-sized differences do not count as movement;
  - motion_gate=False, or no settings store at all, runs every frame;
  - source_stats counts inferred / skipped_still.

Usage: python backend/tests/motion_gate_smoke.py  (needs backend deps).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
TMP = Path(tempfile.mkdtemp(prefix="vigilume-motiongate-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402

from app.config import Config  # noqa: E402
from app.native import ingest as ingest_mod  # noqa: E402
from app.native.ingest import IngestManager, motion_between, motion_thumb  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


class StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def process(self, camera, frame_time, observations, frame_bgr=None):
        self.calls.append((camera, frame_time, list(observations), frame_bgr))


class StubDetector:
    def __init__(self) -> None:
        self.ready = True
        self.detect_calls = 0
        self.result = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 200.0, 300.0]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={"class_name": np.array(["person"])},
        )

    def detect(self, frame, dw, dh):
        self.detect_calls += 1
        return self.result[np.arange(len(self.result))]  # a fresh copy each call

    def note_detect_ok(self):
        pass

    def note_detect_failure(self):
        pass


class StubTracker:
    """Confirms every detection at once (ByteTrack needs a few frames)."""

    def __init__(self, confirm: bool = True) -> None:
        self.confirm = confirm

    def update(self, detections):
        n = len(detections)
        detections.tracker_id = np.arange(1, n + 1) if self.confirm else np.full(n, -1)
        return detections


class StubSettings:
    def __init__(self, gate: bool | None) -> None:
        self.detection = {} if gate is None else {"motion_gate": gate}

    def is_private(self, camera: str) -> bool:
        return False


def _cam(name: str) -> dict:
    return {
        "name": name, "ip": "10.0.0.9", "username": "u", "password": "p",
        "detect_objects": ["person", "car"], "detect_enabled": True,
        "detect_fps": 5, "detect_width": 704, "detect_height": 480,
        "main_url": "", "sub_url": "", "record_enabled": True,
        "capabilities": {}, "detect_mode": "always",
    }


def _scene(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 220, size=(48, 70, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.kron(base, np.ones((10, 10, 1), dtype=np.uint8)))


def _manager(settings) -> tuple[IngestManager, StubEngine, StubDetector]:
    engine, detector = StubEngine(), StubDetector()
    mgr = IngestManager(engine, detector, Config(), ffmpeg_path="/fake/ffmpeg",
                        settings=settings)
    mgr._default_mode = "always"
    mgr._trackers["c"] = StubTracker()
    return mgr, engine, detector


def unit_checks() -> None:
    print("comparison")
    a = _scene()
    check(not motion_between(motion_thumb(a), motion_thumb(a.copy())),
          "an identical frame is not movement")
    rng = np.random.default_rng(7)
    noisy = np.clip(a.astype(np.int16) + rng.integers(-4, 5, a.shape), 0, 255).astype(np.uint8)
    check(not motion_between(motion_thumb(a), motion_thumb(noisy)),
          "sensor-noise-sized jitter is not movement")
    moved = a.copy()
    moved[200:300, 300:340] = 255 - moved[200:300, 300:340]
    check(motion_between(motion_thumb(a), motion_thumb(moved)),
          "a person-sized change is movement")
    small = np.zeros((240, 352, 3), np.uint8)
    check(motion_between(motion_thumb(a), motion_thumb(small)),
          "thumbnails of different sizes count as movement")


async def _gate_cases() -> None:
    print("gate on")
    mgr, engine, detector = _manager(StubSettings(True))
    cam = _cam("c")
    frame = _scene()
    await mgr._process_frame("c", cam, frame, 1.0)
    check(detector.detect_calls == 1, "the first frame runs inference")
    first_obs = engine.calls[-1][2]
    check(len(first_obs) == 1, "the inference produced one observation")

    await mgr._process_frame("c", cam, frame.copy(), 1.2)
    await mgr._process_frame("c", cam, frame.copy(), 1.4)
    check(detector.detect_calls == 1, "still frames skip inference")
    check(len(engine.calls) == 3, "still frames still reach the engine")
    check(engine.calls[-1][2] == first_obs and engine.calls[-1][1] == 1.4,
          "a still frame re-uses the last observations at its own time")
    check(engine.calls[-1][2] is not first_obs, "re-used observations are a copy")

    moved = frame.copy()
    moved[200:300, 300:340] = 255 - moved[200:300, 300:340]
    await mgr._process_frame("c", cam, moved, 1.6)
    check(detector.detect_calls == 2, "a frame with movement runs inference at once")

    stats = mgr._inferred.get("c"), mgr._skipped.get("c")
    check(stats == (2, 2), f"counters track inferred/skipped (got {stats})")

    # Forced re-run: age the last inference past MOTION_FORCE_S.
    thumb, mono, obs = mgr._motion["c"]
    mgr._motion["c"] = (thumb, mono - ingest_mod.MOTION_FORCE_S - 0.01, obs)
    await mgr._process_frame("c", cam, moved.copy(), 2.8)
    check(detector.detect_calls == 3, "inference is forced after MOTION_FORCE_S")

    await mgr._stop_source("c")
    check("c" not in mgr._motion, "stopping a camera drops its motion state")

    print("unconfirmed objects")
    mgr, engine, detector = _manager(StubSettings(True))
    mgr._trackers["c"].confirm = False
    for t in (1.0, 1.2, 1.4):
        await mgr._process_frame("c", cam, frame.copy(), t)
    check(detector.detect_calls == 3,
          "still frames run inference while the tracker has not confirmed an object")
    mgr._trackers["c"].confirm = True
    for t in (1.6, 1.8, 2.0):
        await mgr._process_frame("c", cam, frame.copy(), t)
    check(detector.detect_calls == 4, "once confirmed, still frames skip again")

    print("gate off")
    for label, settings in (("motion_gate=False", StubSettings(False)), ("no settings store", None)):
        mgr, engine, detector = _manager(settings)
        for t in (1.0, 1.2, 1.4):
            await mgr._process_frame("c", cam, frame.copy(), t)
        check(detector.detect_calls == 3, f"{label}: every frame runs inference")

    print("default")
    mgr, engine, detector = _manager(StubSettings(None))
    for t in (1.0, 1.2):
        await mgr._process_frame("c", cam, frame.copy(), t)
    check(detector.detect_calls == 1, "a settings block without the key gates (default on)")


def main() -> None:
    unit_checks()
    asyncio.run(_gate_cases())
    print(f"\nmotion_gate_smoke: {PASS} checks passed")


if __name__ == "__main__":
    main()
