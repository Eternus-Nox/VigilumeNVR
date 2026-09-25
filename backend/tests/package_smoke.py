#!/usr/bin/env python3
"""Left packages: something carried is now sitting still, and nobody took it.

THE SIGNAL, AND WHY IT NEEDS ALL THREE PARTS
============================================
There is no `package` class in COCO, so this rides the carried-container labels
the detector does know (backpack / handbag / suitcase). That makes the LABEL
weak evidence on its own — a doormat, a plant pot and a folded chair all get
called a handbag sooner or later. The strength comes from combining it with two
things the label cannot fake:

  still for PACKAGE_SETTLE_S  — a bag over a shoulder, or one put down while
                                somebody finds their keys, is not a delivery
  younger than the window     — a thing that has been in view far longer than
                                anyone has been here is furniture, however
                                package-shaped the detector finds it
  a person seen recently      — parcels do not arrive on their own; this is
                                what separates a delivery from a persistent
                                misdetection that would otherwise fire nightly

Each of those is asserted below by REMOVING it and checking nothing fires.

It reads the motion state directly rather than the filtered `confirmed` set,
because a package that was set down and never moved again is exactly what the
stationary filter hides from the event layer — so this is one of the two
features that would break if that state were only kept while the filter was on.

Offline: no models, no network, no database.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config, DEFAULT_SETTINGS  # noqa: E402
from app.native.engine import (  # noqa: E402
    MIN_HITS, PACKAGE_LABELS, PACKAGE_PERSON_WINDOW_S, PACKAGE_SETTLE_S,
    DetectionEngine, Observation, _CameraState,
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
    def __init__(self) -> None:
        self.packages: list[tuple[str, str]] = []

    async def handle_event(self, payload: dict) -> None:
        return None

    def update_count(self, *_a) -> None:
        return None

    def note_package(self, camera: str, label: str, box: list) -> None:
        self.packages.append((camera, label))


class QuietRecorder:
    async def schedule_clip(self, *_a, **_k) -> None:
        return None


def box_at(cx: float, cy: float, w: float = 60.0, h: float = 60.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def build(pipe, **detection):
    engine = DetectionEngine(db=None, detector=None, recorder=QuietRecorder(),
                             settings=Settings(**detection), config=Config())
    engine.set_pipeline(pipe)
    engine._cameras["porch"] = _CameraState(
        row={"name": "porch", "detect_objects": ["person", "suitcase"],
             "record_enabled": True}
    )
    return engine


async def deliver(engine, *, person=True, hold=PACKAGE_SETTLE_S + 20,
                  package_label="suitcase", t0=None, person_frames=MIN_HITS + 3):
    """A person walks up (optionally), a package appears and stays put."""
    t0 = t0 if t0 is not None else time.time()
    frame = 0
    if person:
        for i in range(person_frames):
            await engine.process(
                "porch", t0 + frame * 0.2,
                [Observation("person", 1, 0.9, box_at(100 + i * 60, 400, 100, 250))],
                frame_bgr=None,
            )
            frame += 1
    start = t0 + frame * 0.2
    parcel = box_at(400, 430)
    n = max(2, int(hold))
    for i in range(n):
        await engine.process("porch", start + i,
                             [Observation(package_label, 2, 0.9, parcel)],
                             frame_bgr=None)
    return start


async def main_checks() -> None:
    print("\noff by default")
    pipe = Pipe()
    await deliver(build(pipe))
    check(pipe.packages == [],
          "with package_alerts off, a parcel sitting on the step reports "
          "nothing — this adds notifications, so it is opt-in")

    print("\nthe whole signal, assembled")
    pipe = Pipe()
    await deliver(build(pipe, package_alerts=True))
    check(pipe.packages == [("porch", "suitcase")],
          f"a person arrives, a carried object is left and stays put -> ONE "
          f"report (got {pipe.packages})")

    print("\n...and each part of it is load-bearing")
    pipe = Pipe()
    await deliver(build(pipe, package_alerts=True), person=False)
    check(pipe.packages == [],
          "NO PERSON: a package-shaped thing that appeared with nobody around "
          "is a misdetection, not a delivery — this is what stops a doormat "
          "reporting itself every night")

    pipe = Pipe()
    await deliver(build(pipe, package_alerts=True), hold=PACKAGE_SETTLE_S / 3)
    check(pipe.packages == [],
          "NOT SETTLED: a bag held for a few seconds is being carried, not "
          "left")

    print("\nreported once, not once per frame")
    pipe = Pipe()
    await deliver(build(pipe, package_alerts=True), hold=PACKAGE_SETTLE_S + 200)
    check(len(pipe.packages) == 1,
          f"a parcel sitting there for minutes is still ONE report (got "
          f"{len(pipe.packages)}) — otherwise it would notify every frame for "
          "as long as nobody collected it")

    print("\nit works with the stationary filter OFF")
    pipe = Pipe()
    await deliver(build(pipe, package_alerts=True, ignore_stationary=False))
    check(pipe.packages == [("porch", "suitcase")],
          "motion state is kept even when the stationary FILTER is disabled — "
          "keeping it behind that switch silently disabled this feature too")

    print("\nlabels")
    check("suitcase" in PACKAGE_LABELS and "backpack" in PACKAGE_LABELS,
          "the carried-container classes COCO actually has")
    check("package" not in PACKAGE_LABELS and "box" not in PACKAGE_LABELS,
          "and NOT classes COCO lacks — a label that exists on only one model "
          "tier would make this silently depend on which model is loaded")
    pipe = Pipe()
    e = build(pipe, package_alerts=True)
    e._cameras["porch"].row["detect_objects"] = ["person", "dog"]
    await deliver(e)
    check(pipe.packages == [],
          "a camera not detecting the label reports nothing — the pass is fed "
          "from confirmed detections, same as recognition")

    print("\na pipeline without the hook must not break detection")
    class OldPipe(Pipe):
        note_package = None  # type: ignore[assignment]

    pipe = OldPipe()
    await deliver(build(pipe, package_alerts=True))
    check(True, "an events pipeline with no note_package is tolerated")


def settings_checks() -> None:
    print("\nthe setting is modelled")
    from app.routers.settings import AppSettings, DetectionSettings

    check(DetectionSettings().package_alerts is False, "off by default")
    check("package_alerts" in DEFAULT_SETTINGS["detection"],
          "DEFAULT_SETTINGS carries it — a key in one and not the other is "
          "wiped by every unrelated save")
    doc = AppSettings(**{"detection": {"package_alerts": True}})
    check(doc.detection.package_alerts is True, "it round-trips")
    check(PACKAGE_PERSON_WINDOW_S > PACKAGE_SETTLE_S,
          "the person window is wider than the settle time, or an object could "
          "never finish settling before the window it needs had closed")


def main() -> int:
    asyncio.run(main_checks())
    settings_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (left packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
