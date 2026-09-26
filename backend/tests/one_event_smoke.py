#!/usr/bin/env python3
"""One event per camera, open until everything has left.

Events used to be one per (camera, label): a person and the car they arrived
in were two events a second apart, and the list showed two rows for one thing
that happened. Now a camera has ONE open event. It opens on the first
confirmed object of any type, gathers every type that appears while it is
open, and ends only once ALL of them have been gone for the absence timeout.

What has to stay true for that not to cost anything:

  * the event is NAMED after its most important type (a car that a person then
    gets out of reads "person"), and its snapshot follows that person;
  * a type leaving while others stay is reported, and its Home Assistant
    sensor turns off — HA still has one sensor per type;
  * a type worth an alert that joins an open event still ALERTS — "a person
    got out of the car" must reach the phone even though it is no longer a
    new event;
  * filtering the event list by a type finds events that type was part of,
    not only ones named after it.

Offline: no models, no network.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_SETTINGS, Config  # noqa: E402
from app.event_labels import label_rank, primary_label  # noqa: E402
from app.events_pipeline import EventsPipeline  # noqa: E402
from app.native.engine import MIN_HITS, DetectionEngine, Observation, _CameraState  # noqa: E402

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
    def __init__(self, **detection):
        self.detection = {**DEFAULT_SETTINGS["detection"], "ignore_stationary": False,
                          **detection}
        self.notifications = {**DEFAULT_SETTINGS["notifications"],
                              "labels": ["person", "car"]}
        self.recognition = {**DEFAULT_SETTINGS["recognition"]}

    def is_private(self, _name: str) -> bool:
        return False


class Capture:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.counts: dict[tuple[str, str], int] = {}
        self.dwells: list[tuple[str, str, int]] = []

    async def handle_event(self, payload: dict) -> None:
        self.payloads.append(payload)

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = count

    def note_dwell(self, fid: str, label: str, seconds: int) -> None:
        self.dwells.append((fid, label, seconds))

    def of(self, etype: str) -> list[dict]:
        return [p["after"] for p in self.payloads if p["type"] == etype]


class QuietRecorder:
    async def schedule_clip(self, *_a, **_k) -> None:
        return None


def make_engine(**detection):
    cap = Capture()
    engine = DetectionEngine(db=None, detector=None, recorder=QuietRecorder(),
                             settings=Settings(**detection), config=Config())
    engine.set_pipeline(cap)
    engine._cameras["drive"] = _CameraState(
        row={"name": "drive", "detect_objects": ["person", "car", "dog"],
             "record_enabled": True}
    )
    return engine, cap


def person(i: int, tid: int = 1, score: float = 0.8) -> Observation:
    dx = 30.0 if i % 2 else 0.0
    return Observation("person", tid, score, (100 + dx, 100, 160 + dx, 300))


def car(i: int, tid: int = 2, score: float = 0.95) -> Observation:
    dx = 40.0 if i % 2 else 0.0
    return Observation("car", tid, score, (300 + dx, 200, 600 + dx, 400))


async def feed(engine, t: float, frames: int, make, step: float = 0.5) -> float:
    for i in range(frames):
        await engine.process("drive", t, make(i), frame_bgr=None)
        t += step
    return t


async def engine_checks() -> None:
    print("\na person and a car arriving together are ONE event")
    engine, cap = make_engine()
    t = await feed(engine, 1000.0, MIN_HITS + 2, lambda i: [person(i), car(i)])
    news = cap.of("new")
    check(len(news) == 1, f"one 'new', not one per type (got {len(news)})")
    check(news and news[0]["label"] == "person",
          "named after the person, the more important of the two")
    check(news and sorted(news[0]["labels"]) == ["car", "person"],
          "and carrying both types")

    print("\na car first, then the driver gets out")
    engine, cap = make_engine()
    t = await feed(engine, 1000.0, MIN_HITS + 4, lambda i: [car(i)])
    check([a["label"] for a in cap.of("new")] == ["car"], "the car opens the event")
    fid = cap.of("new")[0]["id"]
    t = await feed(engine, t, MIN_HITS + 2, lambda i: [car(i), person(i)])
    check(len(cap.of("new")) == 1, "the person does NOT open a second event")
    ups = cap.of("update")
    check(ups and ups[-1]["id"] == fid and ups[-1]["label"] == "person",
          "the same event is renamed to the person")
    check(ups and sorted(ups[-1]["present_labels"]) == ["car", "person"],
          "and reports both as in view")
    st = engine.open_event("drive")
    check(st is not None and st.best_label == "person",
          "its snapshot switches to the person even though the car scored higher")

    print("\nthe person leaves, the car stays: the event stays open")
    t = await feed(engine, t, 16, lambda i: [car(i)])  # 8 s of car only
    st = engine.open_event("drive")
    check(st is not None and st.fid == fid, "still the same open event")
    check(engine.open_event("drive", "person") is None
          and engine.open_event("drive", "car") is not None,
          "person no longer in view, car still is")
    check(cap.of("update")[-1]["present_labels"] == ["car"],
          "and an update said so, so Home Assistant can turn the person sensor off")
    check(cap.counts.get(("drive", "person")) == 0, "the person count went to 0")
    check(not cap.of("end"), "nothing has ended")

    print("\neverything leaves: then, and only then, it ends")
    before_end = t
    await engine.process("drive", t + 2.0, [], frame_bgr=None)
    check(not cap.of("end"), "2 s of empty frames is not enough")
    await engine.process("drive", t + 6.0, [], frame_bgr=None)
    ends = cap.of("end")
    check(len(ends) == 1 and ends[0]["id"] == fid, "one 'end' once nothing has been seen for the timeout")
    check(ends and abs(ends[0]["end_time"] - (before_end - 0.5)) < 1e-6,
          "end_time is the last moment anything was seen")
    check(engine.open_event("drive") is None, "and the camera has no open event")

    print("\na short gap in one type is not a departure")
    engine, cap = make_engine()
    t = await feed(engine, 1000.0, MIN_HITS + 2, lambda i: [car(i), person(i)])
    t = await feed(engine, t, 4, lambda i: [car(i)])  # 2 s without the person
    t = await feed(engine, t, 2, lambda i: [car(i), person(i)])
    check(engine.open_event("drive", "person") is not None,
          "a person unseen for 2 s (under the timeout) never left the event")

    print("\nloitering is per type, from when THAT type arrived")
    engine, cap = make_engine(dwell_alert_seconds=60)
    t = await feed(engine, 1000.0, 65, lambda i: [car(i)], step=1.0)
    check([d[1] for d in cap.dwells] == ["car"],
          "the car that has been here 60 s is reported")
    t = await feed(engine, t, 30, lambda i: [car(i), person(i)], step=1.0)
    check([d[1] for d in cap.dwells] == ["car"],
          "a person 30 s into a long event has NOT been loitering 90 s")
    t = await feed(engine, t, 40, lambda i: [car(i), person(i)], step=1.0)
    check([d[1] for d in cap.dwells] == ["car", "person"],
          "but is reported once they have been here 60 s themselves")
    check(cap.dwells[-1][2] >= 60 and cap.dwells[-1][2] < 80,
          f"measured from the person's arrival ({cap.dwells[-1][2]} s)")


async def parked_car_checks() -> None:
    print("\nsomeone walks past a parked car (stationary objects ignored)")
    engine, cap = make_engine(ignore_stationary=True)
    parked = (300.0, 200.0, 600.0, 400.0)
    t = 1000.0
    for i in range(30):  # the car has been parked all along
        await engine.process("drive", t, [Observation("car", 2, 0.95, parked)], frame_bgr=None)
        t += 0.5
    check(not cap.of("new"), "a parked car alone opens nothing")
    for i in range(14):  # a person walks left to right, in front of it
        x = 200.0 + i * 40.0
        person_box = (x, 150.0, x + 60.0, 420.0)
        # The detector's box on the car is cut where the person covers it.
        car_box = (300.0, 200.0, 600.0, 400.0) if not 300 <= x <= 560 else (300.0, 200.0, 450.0, 400.0)
        await engine.process("drive", t, [Observation("car", 2, 0.95, car_box),
                                          Observation("person", 1, 0.85, person_box)],
                             frame_bgr=None)
        t += 0.3
    news = cap.of("new")
    check(len(news) == 1 and news[0]["label"] == "person", "the person opens the event")
    all_labels = {l for p in cap.payloads for l in p["after"].get("labels", [])}
    check("car" not in all_labels,
          f"and the parked car never joins it, even though its box was cut in half "
          f"while they passed (labels seen: {sorted(all_labels)})")


def label_checks() -> None:
    print("\nnaming order")
    check(label_rank("person") < label_rank("car") < label_rank("dog") < label_rank("kite"),
          "person, then vehicles, then animals, then everything else")
    check(primary_label(["dog", "car", "person"]) == "person", "the person wins")
    check(primary_label(["truck", "car"]) == "truck", "a tie goes to whichever came first")
    check(primary_label([]) == "", "nothing named from nothing")


class NotifySettings:
    def __init__(self):
        self.notifications = {**DEFAULT_SETTINGS["notifications"],
                              "labels": ["person", "car"], "cooldown_seconds": 60,
                              "min_score": 0.5}
        self.recognition = {"enabled": False}

    def is_private(self, _c):
        return False


def bare_pipeline():
    pipe = EventsPipeline.__new__(EventsPipeline)
    pipe._settings = NotifySettings()
    pipe._cooldowns = {}
    pipe.counts = {}
    sent: list[dict] = []

    async def friendly(camera):
        return "Driveway"

    async def send(**kw):
        sent.append(kw)

    pipe._friendly_name = friendly
    pipe._send_notification = send
    return pipe, sent


async def notify_checks() -> None:
    print("\na person joining a car's open event still alerts")
    pipe, sent = bare_pipeline()
    after = {"camera": "drive", "label": "car", "top_score": 0.9}
    state = {"notified": False, "labels": ["car"], "recognitions": [], "max_count": 1,
             "event_id": 7, "snap_time": 1.0}
    await pipe._maybe_notify_object("f", after, state)
    check(len(sent) == 1 and sent[0]["title"] == "Car detected at Driveway",
          f"the car alerts first ({[s['title'] for s in sent]})")

    # What _on_update does when the person joins.
    state["labels"] = ["car", "person"]
    wanted = pipe._settings.notifications["labels"]
    if state["notified"] and any(l in wanted and l not in state["notified_labels"]
                                 for l in state["labels"]):
        state["notified"] = False
    await pipe._maybe_notify_object("f", {**after, "label": "person"}, state)
    check(len(sent) == 2 and sent[1]["title"] == "Person also detected at Driveway",
          f"then the person, as a second alert ({[s['title'] for s in sent]})")
    check(sent[1]["tag"].endswith("-person"),
          "with its own tag, so it does not replace the car's alert on the phone")

    await pipe._maybe_notify_object("f", {**after, "label": "person"}, state)
    check(len(sent) == 2, "and nothing more for types already alerted about")

    print("\na type not on the alert list does not re-arm anything")
    pipe, sent = bare_pipeline()
    state = {"notified": False, "labels": ["dog"], "recognitions": [], "max_count": 1,
             "event_id": 8, "snap_time": 1.0}
    await pipe._maybe_notify_object("g", {"camera": "drive", "label": "dog",
                                          "top_score": 0.9}, state)
    check(sent == [], "a dog, not on the list, sends nothing")


async def presence_checks() -> None:
    print("\nHome Assistant: one sensor per type, driven from what is in view")
    pipe = EventsPipeline.__new__(EventsPipeline)
    calls: list[tuple[str, str]] = []

    async def mqtt(camera, label, etype, row):
        calls.append((label, etype))

    pipe._publish_mqtt = mqtt
    state = {"mqtt_on": set()}
    await pipe._publish_presence(state, {"camera": "drive", "label": "person",
                                         "present_labels": ["car", "person"]}, None, "new")
    check(calls == [("car", "new"), ("person", "new")],
          f"both sensors on, the event's name last ({calls})")
    calls.clear()
    await pipe._publish_presence(state, {"camera": "drive", "label": "person",
                                         "present_labels": ["car"]}, None, "update")
    check(("person", "end") in calls and ("car", "update") in calls,
          f"the person leaving turns only its sensor off ({calls})")
    calls.clear()
    await pipe._publish_presence(state, {"camera": "drive", "label": "person"}, None, "end")
    check(calls == [("car", "end")], f"the end turns off what was still on ({calls})")


async def filter_checks() -> None:
    print("\nfiltering the list by a type finds the events it was in")
    from app.db import Database

    tmp = Path(__file__).resolve().parent / ".one_event_tmp.db"
    tmp.unlink(missing_ok=True)
    db = Database(tmp)
    await db.connect()
    try:
        now = time.time()
        await db.insert_event(frigate_id="a", camera="drive", label="person", count=1,
                              score=0.9, start_time=now, zones=[], has_clip=False,
                              has_snapshot=False, box=None, labels=["car", "person"])
        await db.insert_event(frigate_id="b", camera="drive", label="dog", count=1,
                              score=0.9, start_time=now + 1, zones=[], has_clip=False,
                              has_snapshot=False, box=None, labels=["dog"])
        _, n_car = await db.list_events(label="car")
        _, n_person = await db.list_events(label="person")
        _, n_dog = await db.list_events(label="dog")
        check(n_car == 1, "'car' finds the event named 'person' that the car was in")
        check(n_person == 1 and n_dog == 1, "and names still match as before")
    finally:
        await db.close()
        tmp.unlink(missing_ok=True)


async def main() -> int:
    label_checks()
    await engine_checks()
    await parked_car_checks()
    await notify_checks()
    await presence_checks()
    await filter_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (one event per camera)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
