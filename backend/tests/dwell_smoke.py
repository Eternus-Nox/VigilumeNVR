#!/usr/bin/env python3
"""Loitering alerts: "someone is STILL at the front door".

WHAT IT IS, AND WHAT IT IS NOT
==============================
An ordinary alert says a person ARRIVED. It fires once and is then suppressed
by `state["notified"]` and by the per-(camera, label) cooldown, both of which
exist so one subject cannot produce a stream of notifications. That is right
for an arrival and wrong for the thing people actually worry about, which is
somebody who arrived and did not leave.

So a dwell alert is a SECOND, different statement about the same subject, and
it deliberately bypasses both of those gates. Everything below is about making
that bypass safe:

  * ONE per event. "Still there" repeated every minute is precisely the noise
    this replaces, and the event is already on screen.
  * measured from the EVENT's start, not a track's — a subject whose track is
    lost behind a pillar and re-acquired has not just arrived, and restarting
    the clock there would let a loiterer avoid the alert by standing where
    tracking is poor.
  * OFF by default. Unlike the stationary filter (which only ever removes
    alerts), this ADDS a kind of notification, and a security system that
    starts pushing new alerts after an update is one people mute entirely.
  * a MUTED profile silences it too, or the loitering path would be a way for
    a muted subject to generate notifications anyway.

Offline: no models, no network. The engine half runs against a real
DetectionEngine; the notification half drives the pipeline hook directly.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config, DEFAULT_SETTINGS  # noqa: E402
from app.native.engine import (  # noqa: E402
    MIN_HITS, DetectionEngine, Observation, _CameraState,
)

_failures: list[str] = []
_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


class Settings:
    def __init__(self, **over):
        self.detection = {**DEFAULT_SETTINGS["detection"], **over}
        self.notifications = {**DEFAULT_SETTINGS["notifications"]}
        self.recognition = {**DEFAULT_SETTINGS["recognition"]}

    def is_private(self, _name: str) -> bool:
        return False


class Pipe:
    """Records the dwell hook rather than sending anything."""

    def __init__(self) -> None:
        self.dwells: list[tuple[str, str, int]] = []
        self.counts: dict[tuple[str, str], int] = {}

    async def handle_event(self, payload: dict) -> None:
        return None

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = count

    def note_dwell(self, fid: str, label: str, seconds: int) -> None:
        self.dwells.append((fid, label, seconds))


def box_at(cx: float, cy: float, w: float = 100.0, h: float = 250.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


async def run(engine, pipe, *, camera="porch", seconds=400, step=1.0, t0=None):
    """Walk a person in, then hold them there for `seconds` of frame time."""
    t0 = t0 if t0 is not None else time.time()
    frame = 0
    for i in range(MIN_HITS + 3):  # arriving: must move, or it is furniture
        await engine.process(camera, t0 + frame * 0.2,
                             [Observation("person", 7, 0.9, box_at(100 + i * 60, 400))],
                             frame_bgr=None)
        frame += 1
    settled = t0 + frame * 0.2
    held = box_at(100 + (MIN_HITS + 2) * 60, 400)
    n = int(seconds / step)
    for i in range(n):
        await engine.process(camera, settled + i * step,
                             [Observation("person", 7, 0.9, held)], frame_bgr=None)
    return settled


class QuietRecorder:
    """An event ending schedules a clip; this suite is about alerts, not clips."""

    async def schedule_clip(self, *_a, **_k) -> None:
        return None


def engine_with(pipe, **detection):
    engine = DetectionEngine(db=None, detector=None, recorder=QuietRecorder(),
                             settings=Settings(**detection), config=Config())
    engine.set_pipeline(pipe)
    return engine


def add_camera(engine, name="porch", **kw):
    st = _CameraState(
        row={"name": name, "detect_objects": ["person"], "record_enabled": True}
    )
    for k, v in kw.items():
        setattr(st, k, v)
    engine._cameras[name] = st
    return st


async def engine_checks() -> None:
    print("\noff by default")
    pipe = Pipe()
    e = engine_with(pipe)
    add_camera(e)
    await run(e, pipe, seconds=600)
    check(pipe.dwells == [],
          "with dwell_alert_seconds unset, ten minutes of standing produces NO "
          "dwell alert — this feature must be opted into, not inherited")

    print("\nthe global setting turns it on")
    pipe = Pipe()
    e = engine_with(pipe, dwell_alert_seconds=60)
    add_camera(e)
    await run(e, pipe, seconds=300)
    check(len(pipe.dwells) == 1,
          f"a subject present past the threshold fires EXACTLY ONE dwell alert "
          f"(got {len(pipe.dwells)}) — 'still there' every minute is the noise "
          "this replaces")
    if pipe.dwells:
        _, label, secs = pipe.dwells[0]
        check(label == "person", "it names the label")
        check(secs >= 60,
              f"and reports how long they had been there ({secs}s), measured "
              "from the event's start")

    print("\na short visit is not loitering")
    pipe = Pipe()
    e = engine_with(pipe, dwell_alert_seconds=120)
    add_camera(e)
    await run(e, pipe, seconds=30)
    check(pipe.dwells == [],
          "someone who leaves before the threshold never triggers it")

    print("\nper camera: the override wins in BOTH directions")
    pipe = Pipe()
    e = engine_with(pipe, dwell_alert_seconds=0)  # globally off
    add_camera(e, dwell_seconds=60)               # but on HERE
    await run(e, pipe, seconds=200)
    check(len(pipe.dwells) == 1,
          "a camera pinned ON overrides a global OFF — one front door can want "
          "loitering alerts while the rest of the system does not")

    pipe = Pipe()
    e = engine_with(pipe, dwell_alert_seconds=60)  # globally on
    add_camera(e, dwell_seconds=0)                 # off HERE
    await run(e, pipe, seconds=300)
    check(pipe.dwells == [],
          "a camera pinned to 0 overrides a global ON — which is how a "
          "pavement-facing camera opts out. 0 and 'unset' must not collapse "
          "into the same thing")

    pipe = Pipe()
    e = engine_with(pipe, dwell_alert_seconds=60)
    add_camera(e, dwell_seconds=None)  # explicit inherit
    await run(e, pipe, seconds=200)
    check(len(pipe.dwells) == 1, "and None inherits the global setting")

    print("\ndwell and the stationary filter must COMPOSE, not cancel out")
    pipe = Pipe()
    # Dwell well ABOVE the dormancy window: without the camera-aware dormancy
    # the subject would be dropped at 60 s, its label would go absent, the
    # event would end, and the alert could never fire.
    e = engine_with(pipe, dwell_alert_seconds=150, stationary_after_s=60)
    add_camera(e)
    await run(e, pipe, seconds=400)
    check(len(pipe.dwells) == 1,
          f"a dwell threshold ABOVE stationary_after_s still fires (got "
          f"{len(pipe.dwells)}) — a settled subject is held until past its "
          "dwell time instead of being dropped as motionless")

    pipe = Pipe()
    e = engine_with(pipe, dwell_alert_seconds=0, stationary_after_s=60)
    add_camera(e)
    await run(e, pipe, seconds=400)
    check(("porch", "person") not in e._events,
          "...and with dwell OFF the stationary filter still releases a settled "
          "subject as before — the hold is bought by dwell, not granted always")

    print("\na pipeline without the hook must not break detection")
    class OldPipe(Pipe):
        note_dwell = None  # type: ignore[assignment]

    pipe = OldPipe()
    e = engine_with(pipe, dwell_alert_seconds=30)
    add_camera(e)
    await run(e, pipe, seconds=120)
    check(True,
          "an events pipeline with no note_dwell is tolerated rather than "
          "raising five times a second — detection outranks this feature")


def pipeline_checks() -> None:
    print("\nthe notification is a SECOND statement, not a repeat")
    from app.events_pipeline import EventsPipeline

    pipe = EventsPipeline.__new__(EventsPipeline)
    pipe._settings = Settings(dwell_alert_seconds=60)  # type: ignore[attr-defined]
    sent: list = []
    pipe._spawn = lambda coro: (sent.append(coro), coro.close())  # type: ignore[attr-defined]
    pipe._active = {  # type: ignore[attr-defined]
        "fid-1": {"notified": True, "dwell_notified": False, "recognitions": [],
                  "event_id": 5, "camera": "porch", "last_after": {"camera": "porch"}},
    }
    pipe.note_dwell("fid-1", "person", 90)
    check(len(sent) == 1,
          "note_dwell sends even though the event is ALREADY notified — an "
          "arrival alert must not swallow the loitering one")
    check(pipe._active["fid-1"]["dwell_notified"] is True,  # type: ignore[attr-defined]
          "...and marks the event, so a second call is a no-op")

    sent.clear()
    pipe.note_dwell("fid-1", "person", 150)
    check(sent == [], "a repeat call sends nothing")

    sent.clear()
    pipe.note_dwell("missing-fid", "person", 90)
    check(sent == [], "an unknown event id is ignored rather than raising")


def settings_checks() -> None:
    print("\nthe setting is modelled and validated")
    from app.routers.settings import AppSettings, DetectionSettings

    check(DetectionSettings().dwell_alert_seconds == 0, "0 (off) is the default")
    check("dwell_alert_seconds" in DEFAULT_SETTINGS["detection"],
          "DEFAULT_SETTINGS carries it — AppSettings drops what it does not "
          "model, so a key in one and not the other is wiped by every save")
    doc = AppSettings(**{"detection": {"dwell_alert_seconds": 120}})
    check(doc.detection.dwell_alert_seconds == 120, "it round-trips")

    bad = False
    try:
        DetectionSettings(dwell_alert_seconds=3)
    except Exception:
        bad = True
    check(bad,
          "a non-zero value under 10 s is refused — below that it is not "
          "loitering, it is the same alert twice")
    ok = True
    try:
        DetectionSettings(dwell_alert_seconds=0)
    except Exception:
        ok = False
    check(ok, "...while 0 stays legal, because 0 means off")


def main() -> int:
    asyncio.run(engine_checks())
    pipeline_checks()
    settings_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (loitering alerts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
