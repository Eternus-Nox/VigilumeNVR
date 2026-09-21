#!/usr/bin/env python3
"""Stationary objects, and the events they must and must not produce.

WHAT THIS IS ACTUALLY GUARDING
==============================
A detector has no notion of news: it answers "is there a car here?" on every
frame, so a parked car is detected five times a second for as long as it is
parked. Events are keyed (camera, label) and end only when the label goes
ABSENT, so before this existed:

1. the parked car's event never ended;
2. it heartbeat an "update" every 10 s, forever;
3. and WHILE IT WAS OPEN NO NEW CAR EVENT COULD BE. A car pulling into the
   drive arrived as a count change on a stale event instead of a new event.

(3) is the one that matters. It is a missed detection, not noise, and it is the
first thing asserted below.

THE OPPOSITE MISTAKE IS WORSE
-----------------------------
This is a security system, so the failure to fear is suppressing something
real. Two properties are therefore load-bearing and are asserted from several
directions:

- a subject that ARRIVES always gets its event, even if it then stands
  perfectly still for an hour;
- an unknown track — one the motion tracker has no opinion about — is treated
  as ACTIVE. No opinion must mean detect, never suppress.

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
    MIN_HITS, UPDATE_HEARTBEAT_S, DetectionEngine, Observation, _CameraState,
)
from app.native.stillness import (  # noqa: E402
    MIN_MOVE_PX, STATIONARY_AFTER_S, Stillness, clamp_stationary_after,
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


def box_at(cx: float, cy: float, w: float = 100.0, h: float = 250.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------------------
# The motion primitive
# ---------------------------------------------------------------------------


def motion_checks() -> None:
    print("\nwhat counts as motion")
    s = Stillness()

    # Wobble must NEVER accumulate into movement, however long you watch. This
    # is the property that makes the anchor design work: a threshold compared
    # against consecutive frames would have to sit below one frame of walking
    # and above one frame of jitter, and on a distant subject there is no such
    # number.
    t = 1000.0
    for i in range(400):
        jitter = 4.0 if i % 2 else -4.0
        s.update(1, box_at(300.0 + jitter, 400.0), t + i * 0.2)
    check(not s.tracks[1].ever_moved,
          "400 frames of ±4 px box wobble is still NOT movement — jitter is "
          "bounded and measured from an anchor, so it cannot add up")
    check(not s.is_active(1, t + 80.0), "...so it never becomes active")

    # A slow walk, well under the threshold per frame, must accumulate.
    t2 = 2000.0
    for i in range(12):
        s.update(2, box_at(300.0 + i * 4.0, 400.0), t2 + i * 0.2)
    check(s.tracks[2].ever_moved,
          "12 frames of 4 px/frame — under the per-frame threshold every time — "
          "DOES accumulate into movement, which is how a distant subject is seen")

    # Approaching the lens: the centre barely moves, the box doubles.
    t3 = 3000.0
    s.update(3, box_at(300.0, 400.0, w=40.0, h=100.0), t3)
    for i in range(1, 10):
        s.update(3, box_at(302.0, 400.0, w=40.0 + i * 8, h=100.0 + i * 20), t3 + i * 0.2)
    check(s.tracks[3].ever_moved,
          "walking straight AT the camera counts — the centre is still but the "
          "box grows, and centre-only would call that parked")

    print("\nthe threshold scales with the box")
    small = Stillness()
    small.update(9, box_at(100.0, 100.0, w=16.0, h=36.0), 1.0)
    small.update(9, box_at(100.0 + MIN_MOVE_PX - 1.0, 100.0, w=16.0, h=36.0), 1.2)
    check(not small.tracks[9].ever_moved,
          f"a tiny distant box is held to an absolute floor ({MIN_MOVE_PX} px), so "
          "its proportionally larger jitter does not read as a walk")


def state_checks() -> None:
    print("\nthe two stationary cases are NOT the same thing")
    s = Stillness(stationary_after_s=100.0)
    t = 1000.0

    # Furniture: present from the first frame, never moves.
    for i in range(30):
        s.update(1, box_at(300.0, 400.0), t + i)
    check(not s.is_active(1, t + 29),
          "something that has NEVER moved is furniture — a parked car, a bin the "
          "detector reads as a person — and is never active")

    # A real subject: arrives, then stops.
    for i in range(10):
        s.update(2, box_at(100.0 + i * 40.0, 400.0), t + i)
    arrived = t + 9
    check(s.is_active(2, arrived), "a subject that ARRIVED is active")
    for i in range(10, 40):
        s.update(2, box_at(460.0, 400.0), t + i)
    check(s.is_active(2, t + 39),
          "...and STAYS active while standing still — someone at a door is "
          "exactly the sighting worth keeping, so it is not cut short")
    check(s.is_active(2, arrived + 99) and not s.is_active(2, arrived + 101),
          "...until stationary_after_s, after which it stops sustaining its event")

    # Dormancy is not terminal.
    for i in range(6):
        s.update(2, box_at(460.0 + i * 40.0, 400.0), t + 200 + i)
    check(s.is_active(2, t + 205),
          "moving again clears dormancy — which is how a parked car pulling out "
          "becomes news again")

    print("\nno opinion means DETECT")
    check(s.is_active(999, t),
          "a track the motion state has never seen reads as ACTIVE. This is a "
          "security system: an unknown must never be silently suppressed")
    check(not s.all_still([999]),
          "...and an unknown track is never counted as 'everyone is still', so "
          "it cannot quietly hold back an event's heartbeat")
    check(not s.all_still([]),
          "an EMPTY group is not 'all still' either — vacuous truth here would "
          "suppress the heartbeat of an event with nothing in it")

    print("\nhousekeeping")
    s.forget([1])
    check(1 not in s.tracks, "a retired track's motion state retires with it")
    check(s.is_active(1, t + 29),
          "...and a REUSED tracker id starts clean rather than inheriting the "
          "previous occupant's history")
    s.retain([2])
    check(list(s.tracks) == [2], "retain() drops everything not currently live")


def clamp_checks() -> None:
    print("\nthe setting reaches the frame loop — a bad value must never raise")
    for bad in (None, "later", "", [], {}, float("nan")):
        check(clamp_stationary_after(bad) == STATIONARY_AFTER_S,
              f"{bad!r} falls back to the shipped default")
    check(clamp_stationary_after(5) == 10.0,
          "a floor of 10 s — a few seconds would make a subject who pauses "
          "mid-driveway dormant and chop them into two events")
    check(clamp_stationary_after(10**9) == 3600.0, "and a ceiling of an hour")
    check(clamp_stationary_after(240) == 240.0, "a sensible value is used as given")


# ---------------------------------------------------------------------------
# End to end through engine.process — the behaviour people actually see
# ---------------------------------------------------------------------------


class Pipe:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.counts: dict[tuple[str, str], int] = {}

    async def handle_event(self, payload: dict) -> None:
        self.payloads.append(payload)

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = count


class Settings:
    """Just the surface the engine reads."""

    def __init__(self, **over):
        self.detection = {**DEFAULT_SETTINGS["detection"], **over}

    def is_private(self, _name: str) -> bool:
        return False


def kinds(pipe: Pipe, label: str) -> list[str]:
    return [p["type"] for p in pipe.payloads if p["after"]["label"] == label]


async def engine_cases() -> None:
    print("\nend to end: the parked car")
    engine = DetectionEngine(db=None, detector=None, recorder=None,
                             settings=Settings(), config=Config())
    pipe = Pipe()
    engine.set_pipeline(pipe)
    engine._cameras["drive"] = _CameraState(
        row={"name": "drive", "detect_objects": ["car"], "record_enabled": True}
    )

    t = time.time()
    parked = box_at(200.0, 300.0, w=180.0, h=120.0)
    for i in range(60):
        await engine.process("drive", t + i, [Observation("car", 1, 0.9, parked)],
                             frame_bgr=None)
    check(kinds(pipe, "car") == [],
          "a car parked in the drive produces NO event at all — not one that "
          "closes, not a heartbeat, nothing")
    check(("drive", "car") not in engine._events,
          "...and holds no open event, which is what was blocking every "
          "later arrival")

    # THE POINT OF THE WHOLE FEATURE: a second car arrives while the first is
    # still parked, and gets its own event.
    t2 = t + 100
    for i in range(MIN_HITS + 3):
        arriving = box_at(500.0 - i * 60.0, 300.0, w=180.0, h=120.0)
        await engine.process(
            "drive", t2 + i * 0.2,
            [Observation("car", 1, 0.9, parked), Observation("car", 2, 0.9, arriving)],
            frame_bgr=None,
        )
    check("new" in kinds(pipe, "car"),
          "a car PULLING IN past the parked one opens its own 'new' event — "
          "the whole point: before this it was a count change on a stale event")
    live = pipe.counts.get(("drive", "car"))
    check(live == 1,
          f"...and the live count is ONE car, not two — the parked one is not a "
          f"participant in what is happening (got {live})")

    # But the PICTURE must still be honest. `scene` is what gets boxed on the
    # saved snapshot, and an image that quietly omits the parked car is an
    # image that misrepresents the frame it claims to be.
    new_payloads = [p for p in pipe.payloads if p["type"] == "new"]
    scene = new_payloads[0]["after"]["scene"] if new_payloads else []
    check(len(scene) == 2,
          f"the saved SCENE still holds both cars — suppression decides what is "
          f"an event, not what the camera saw (got {len(scene)})")

    print("\nend to end: the person who arrives and stands still")
    engine2 = DetectionEngine(db=None, detector=None, recorder=None,
                              settings=Settings(), config=Config())
    pipe2 = Pipe()
    engine2.set_pipeline(pipe2)
    engine2._cameras["porch"] = _CameraState(
        row={"name": "porch", "detect_objects": ["person"], "record_enabled": True}
    )

    t3 = time.time()
    step = 0.2
    frame = 0
    for i in range(MIN_HITS + 3):  # walking up
        await engine2.process(
            "porch", t3 + frame * step,
            [Observation("person", 7, 0.9, box_at(100.0 + i * 50.0, 400.0))],
            frame_bgr=None,
        )
        frame += 1
    check("new" in kinds(pipe2, "person"),
          "walking up to the door opens the event — the arrival is reported")

    before = len(pipe2.payloads)
    still = box_at(400.0, 400.0)
    held = t3 + frame * step
    for i in range(int(UPDATE_HEARTBEAT_S * 4 / step)):  # ~4 heartbeats' worth
        await engine2.process("porch", held + i * step,
                              [Observation("person", 7, 0.9, still)], frame_bgr=None)
    check(("porch", "person") in engine2._events,
          "standing perfectly still KEEPS the event open — this is the case the "
          "whole feature must not break")
    check(len(pipe2.payloads) == before,
          f"...and emits nothing further: the 10 s heartbeat is held back while "
          f"every subject is motionless (got {len(pipe2.payloads) - before} extra)")

    print("\nthe setting turns it off")
    engine3 = DetectionEngine(db=None, detector=None, recorder=None,
                              settings=Settings(ignore_stationary=False),
                              config=Config())
    pipe3 = Pipe()
    engine3.set_pipeline(pipe3)
    engine3._cameras["drive"] = _CameraState(
        row={"name": "drive", "detect_objects": ["car"], "record_enabled": True}
    )
    t4 = time.time()
    for i in range(MIN_HITS + 3):
        await engine3.process("drive", t4 + i * 0.2,
                              [Observation("car", 1, 0.9, parked)], frame_bgr=None)
    check("new" in kinds(pipe3, "car"),
          "with ignore_stationary off the old behaviour is intact, so an "
          "operator who wants every sighting can have it")


def settings_checks() -> None:
    print("\nthe settings model carries the knobs")
    from app.routers.settings import AppSettings, DetectionSettings

    model = DetectionSettings()
    check(model.ignore_stationary is True,
          "ON by default — what it replaces is not 'a bit noisy', it is a "
          "parked car blocking every later arrival on that camera")
    check(model.stationary_after_s == int(STATIONARY_AFTER_S),
          "and the model default matches the engine's")
    for key in ("ignore_stationary", "stationary_after_s"):
        check(key in DEFAULT_SETTINGS["detection"],
              f"DEFAULT_SETTINGS carries {key} — AppSettings drops what it does "
              "not model, so a key in one and not the other is wiped by every "
              "unrelated save")

    doc = AppSettings(**{"detection": {"ignore_stationary": False,
                                       "stationary_after_s": 600}})
    check(doc.detection.ignore_stationary is False
          and doc.detection.stationary_after_s == 600,
          "AppSettings round-trips them rather than dropping them")

    bad = False
    try:
        DetectionSettings(stationary_after_s=2)
    except Exception:
        bad = True
    check(bad, "an out-of-range stationary_after_s is a 422, not a silent clamp")


def main() -> int:
    motion_checks()
    state_checks()
    clamp_checks()
    asyncio.run(engine_cases())
    settings_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (stationary-object suppression)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
