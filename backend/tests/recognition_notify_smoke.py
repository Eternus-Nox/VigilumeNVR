#!/usr/bin/env python3
"""The recognition notification gate, and recognition on event rows.

The gate decides whether an alert is sent, held, suppressed, or named. Every one
of those can fail silently, so each gets a test:

1. RECOGNITION OFF MUST NOT COST LATENCY. A box that never enabled the feature
   has to alert exactly as fast as before. A hold applied unconditionally would
   delay every alert on every camera for a feature nobody turned on.
2. A HELD ALERT MUST STILL FIRE. The hold is a DEFERRAL, not a drop — if
   recognition never identifies anyone, "someone was here and I could not tell
   who" is the case most worth hearing about, and an expired hold that swallowed
   it would be the worst possible bug in this feature.
3. THE HOLD MUST NOT BURN THE COOLDOWN. Same trap the crossing gate documents:
   a deferred notification that marks the cooldown means the real one, seconds
   later, is silently dropped.
4. unknown_only MUST SUPPRESS PERMANENTLY, not defer. Deferring a known subject
   would send the alert anyway the moment the hold expired — exactly the alert
   the operator asked not to receive.
5. A LABEL RECOGNITION CANNOT SPEAK TO must never wait. A dog has no face
   profile; holding its alert would delay it for an answer that never comes.

Offline-runnable; no models, no network.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.events_pipeline import EventsPipeline  # noqa: E402

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


class FakeSettings:
    """Only the surface `_recognition_gate` touches."""

    def __init__(self, **recognition):
        base = {"enabled": False, "notify_grace_seconds": 4, "notify_mode": "all"}
        base.update(recognition)
        self.recognition = base


def pipeline_with(**recognition) -> EventsPipeline:
    """An EventsPipeline with just enough wired to exercise the gate."""
    p = EventsPipeline.__new__(EventsPipeline)
    p._settings = FakeSettings(**recognition)
    p._active = {}
    p._cooldowns = {}
    # note_recognition also publishes to Home Assistant. None = the
    # integration is off, which is the default and must stay a no-op —
    # recognition working must never depend on MQTT being configured.
    p._mqtt = None
    return p


def state(*, age_s: float = 0.0, recognitions=None, notified: bool = False) -> dict:
    return {
        "opened_at": time.monotonic() - age_s,
        "recognitions": list(recognitions or []),
        "notified": notified,
        "event_id": 1,
    }


def after(label: str = "person", camera: str = "front") -> dict:
    return {"label": label, "camera": camera, "id": "native.1"}


KNOWN = {"kind": "face", "name": "Adam", "profile_id": 7, "plate": "", "score": 0.9}
UNKNOWN = {"kind": "face", "name": "", "profile_id": None, "plate": "", "score": 0.2}
PLATE = {"kind": "plate", "name": "", "profile_id": None, "plate": "7ABC123", "score": 0.0}


def gate_checks() -> None:
    print("\nrecognition disabled — the feature must cost nothing")
    p = pipeline_with(enabled=False)
    send, name = p._recognition_gate(after(), state(age_s=0.0))
    check(send is True, "a brand-new person event sends IMMEDIATELY, with no hold")
    check(name == "", "and carries no name")

    print("\nenabled, nothing identified yet — hold, then send anyway")
    p = pipeline_with(enabled=True, notify_grace_seconds=4)
    send, _ = p._recognition_gate(after(), state(age_s=0.0))
    check(send is False, "inside the hold, the alert is DEFERRED")
    st = state(age_s=0.0)
    p._recognition_gate(after(), st)
    check(st["notified"] is False,
          "...and `notified` stays False, so a later update re-decides")

    send, _ = p._recognition_gate(after(), state(age_s=4.1))
    check(send is True,
          "once the hold expires the alert fires ANYWAY — an unidentified "
          "person is the case most worth hearing about")

    p0 = pipeline_with(enabled=True, notify_grace_seconds=0)
    send, _ = p0._recognition_gate(after(), state(age_s=0.0))
    check(send is True, "a hold of 0 disables the wait entirely")

    print("\na label recognition cannot speak to never waits")
    p = pipeline_with(enabled=True, notify_grace_seconds=10)
    send, _ = p._recognition_gate(after(label="dog"), state(age_s=0.0))
    check(send is True, "a dog alert is not held for a face that will never come")
    send, _ = p._recognition_gate(after(label="car"), state(age_s=0.0))
    check(send is False, "...but a car IS held, because plates are read from it")

    print("\nmode 'all' — a known subject is NAMED")
    p = pipeline_with(enabled=True, notify_mode="all")
    send, name = p._recognition_gate(after(), state(recognitions=[KNOWN]))
    check(send is True, "a recognized person alerts immediately, without waiting out the hold")
    check(name == "Adam", "and the alert carries their name")

    print("\nmode 'unknown_only' — a known subject is SUPPRESSED, permanently")
    p = pipeline_with(enabled=True, notify_mode="unknown_only")
    st = state(recognitions=[KNOWN])
    send, name = p._recognition_gate(after(), st)
    check(send is False, "a recognized person does NOT alert")
    check(st["notified"] is True,
          "...and is marked notified so the next update cannot send it anyway — "
          "a deferral here would fire the very alert that was opted out of")

    st = state(recognitions=[UNKNOWN])
    send, _ = p._recognition_gate(after(), st)
    check(send is True, "an UNIDENTIFIED face still alerts in unknown_only mode")
    check(st["notified"] is False, "...and is not pre-marked")

    st = state(age_s=99, recognitions=[])
    send, _ = p._recognition_gate(after(), st)
    check(send is True,
          "and a person nobody could identify at all still alerts — silence "
          "there would be the worst failure this feature could have")

    print("\nmalformed settings degrade rather than break the alert")
    p = pipeline_with(enabled=True, notify_grace_seconds="soon")
    send, _ = p._recognition_gate(after(), state(age_s=99))
    check(send is True, "an unparseable grace falls back to the default and still sends")
    p = pipeline_with(enabled=True, notify_mode="nonsense")
    send, name = p._recognition_gate(after(), state(recognitions=[KNOWN]))
    check(send is True and name == "Adam",
          "an unknown notify_mode behaves as 'all' rather than silencing alerts")
    send, _ = pipeline_with(enabled=True)._recognition_gate(after(), {"recognitions": []})
    check(send is True, "a state with no opened_at sends rather than hanging forever")


def note_checks() -> None:
    print("\nnote_recognition")
    p = pipeline_with(enabled=True)
    p._spawned = []
    p._spawn = lambda coro: (p._spawned.append(coro), coro.close())  # type: ignore

    p.note_recognition("nosuch", "face", name="Adam", profile_id=1)
    check(p._spawned == [], "a recognition for an event that is not live is ignored")

    st = state()
    st["last_after"] = after()
    p._active["native.1"] = st
    p.note_recognition("native.1", "face", name="Adam", profile_id=7, score=0.9)
    check(len(st["recognitions"]) == 1, "a live event records the recognition")
    check(st["recognitions"][0]["name"] == "Adam", "...with the name")
    check(st["recognitions"][0]["profile_id"] == 7, "...and the profile it matched")
    check(len(p._spawned) == 1,
          "and re-runs the notify decision immediately — the alert may have "
          "been held for exactly this")

    st2 = state(notified=True)
    st2["last_after"] = after()
    p._active["native.2"] = st2
    p._spawned.clear()
    p.note_recognition("native.2", "face", name="Adam", profile_id=7)
    check(len(st2["recognitions"]) == 1, "an already-notified event still records it")
    check(p._spawned == [],
          "...but does not re-notify — that would be a second alert for one event")

    p._spawned.clear()
    st3 = state()
    st3["last_after"] = after(label="car")
    p._active["native.3"] = st3
    p.note_recognition("native.3", "plate", plate="7ABC123")
    check(st3["recognitions"][0]["plate"] == "7ABC123",
          "an UNMATCHED plate is still recorded — it belongs in the alert even "
          "when the vehicle is not enrolled")
    check(st3["recognitions"][0]["known"] is False
          if "known" in st3["recognitions"][0] else True,
          "...and is not claimed as a known vehicle")


def events_join_checks() -> None:
    """Recognition must never be able to take down the events list.

    `_with_recognitions` decorates GET /api/events with who was recognized. It
    is an optional feature, OFF by default, decorating the core screen of the
    product — so a fault on the recognition side has exactly one acceptable
    outcome: serve the events without it. An exception escaping here 500s the
    events list AND the timeline, leaving the operator with nothing at all.
    """
    import asyncio

    from app.routers.events import _with_recognitions

    print("\n_with_recognitions")

    rows = [{"id": 1, "frigate_id": "native.1"}, {"id": 2, "frigate_id": "native.2"}]

    class OkDB:
        async def recognitions_for(self, fids):
            return {"native.1": [KNOWN]}

    out = asyncio.run(_with_recognitions(OkDB(), [dict(r) for r in rows]))
    check(out[0]["recognitions"] == [KNOWN], "a matched event carries its recognition")
    check(out[1]["recognitions"] == [],
          "an event with none carries [] — absent and empty must not differ to a client")

    class BoomDB:
        async def recognitions_for(self, fids):
            raise RuntimeError("no such table: event_recognitions")

    try:
        out = asyncio.run(_with_recognitions(BoomDB(), [dict(r) for r in rows]))
        raised = False
    except Exception:
        raised = True
    check(not raised,
          "a FAILING recognition store does not propagate — an off-by-default "
          "feature must never 500 the events list")
    if not raised:
        check([e["recognitions"] for e in out] == [[], []],
              "...and every event is served with no recognitions rather than dropped")
        check([e["id"] for e in out] == [1, 2],
              "...with the events themselves intact, which is what the screen is for")

    check(asyncio.run(_with_recognitions(BoomDB(), [])) == [],
          "no events -> no query at all")


class FakeMqtt:
    """Records what would reach the broker."""

    def __init__(self):
        self.published = []

    async def publish_recognition(self, camera, *, kind, name, plate, known, score=0.0):
        self.published.append(
            {"camera": camera, "kind": kind, "name": name,
             "plate": plate, "known": known, "score": score}
        )


def mqtt_checks() -> None:
    """Home Assistant must hear WHO, at the moment recognition decides."""
    print("\nnote_recognition publishes to MQTT")
    p = pipeline_with(enabled=True)
    p._spawned = []
    # _spawn is what actually runs the coroutine in production; here run it
    # synchronously so the publish is observable.
    import asyncio

    def run(coro):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)

    p._spawn = run  # type: ignore
    mqtt = FakeMqtt()
    p._mqtt = mqtt

    st = state()
    st["last_after"] = after(camera="front_door")
    p._active["native.1"] = st
    p.note_recognition("native.1", "face", name="Adam", profile_id=7, score=0.91)
    hit = next((x for x in mqtt.published if x["name"] == "Adam"), None)
    check(hit is not None, "a recognized person is published")
    if hit:
        check(hit["camera"] == "front_door", "...on the camera that saw them")
        check(hit["known"] is True, "...marked known, because a profile matched")

    st2 = state()
    st2["last_after"] = after(camera="front_door")
    p._active["native.2"] = st2
    p.note_recognition("native.2", "face", name="", profile_id=None, score=0.2)
    unknown = [x for x in mqtt.published if not x["known"]]
    check(len(unknown) == 1,
          "an UNMATCHED face is published too — 'nobody we know was at the "
          "door' is the state most worth automating on, and silence there is "
          "indistinguishable from recognition being switched off")

    print("\nMQTT off is a no-op, not a failure")
    p2 = pipeline_with(enabled=True)
    p2._spawned = []
    p2._spawn = lambda coro: (p2._spawned.append(coro), coro.close())  # type: ignore
    p2._mqtt = None
    st3 = state()
    st3["last_after"] = after()
    p2._active["native.3"] = st3
    raised = False
    try:
        p2.note_recognition("native.3", "face", name="Adam", profile_id=1)
    except Exception:
        raised = True
    check(not raised,
          "recognition works with no MQTT configured — the integration is off "
          "by default and must never be a dependency")
    check(len(st3["recognitions"]) == 1, "...and the recognition is still recorded")


def wording_checks() -> None:
    from app.notify.wording import named_title, place_phrase

    print("\nwho is where — the wording")
    for friendly, want in (
        ("Front Door", "at the front door"),
        ("Driveway", "in the driveway"),
        ("Drive Way", "in the drive way"),
        ("Back Yard", "in the back yard"),
        ("Front Porch", "on the front porch"),
        ("The Garage", "in the garage"),
        ("front_door", "at the front door"),
        ("Cam 2", "at Cam 2"),
        ("Adam's Office", "in Adam's Office"),
        ("IPC-4K", "at IPC-4K"),
    ):
        check(place_phrase(friendly) == want, f"{friendly!r} -> {want!r} (got {place_phrase(friendly)!r})")
    check(named_title(["Adam"], "Front Door") == "Adam is at the front door", "one person: 'is'")
    check(named_title(["the car"], "Driveway") == "The car is in the driveway",
          "a vehicle profile's own name leads, capitalised")
    check(named_title(["Adam", "Sarah", "adam"], "Back Yard", still=True)
          == "Adam and Sarah are still in the back yard",
          "two names: 'are', duplicates dropped, 'still' for a dwell")
    check(named_title([], "Front Door") == "", "no name, no named title")


def named_pipeline(**recognition) -> EventsPipeline:
    """A pipeline that runs the real alert path and records what it sends."""
    import asyncio

    p = pipeline_with(enabled=True, **recognition)
    p._settings.notifications = {"enabled": True, "labels": ["person", "car"],
                                 "min_score": 0.5, "cooldown_seconds": 60}
    p.counts = {}
    p.sent = []

    async def friendly(camera):
        return {"front_door": "Front Door", "drive": "Driveway"}.get(camera, camera)

    async def send(**kw):
        p.sent.append(kw)

    p._friendly_name = friendly
    p._send_notification = send
    loop = asyncio.new_event_loop()
    p._spawn = lambda coro: loop.run_until_complete(coro)
    p._run = p._spawn
    return p


def live(p: EventsPipeline, fid: str, camera: str, label: str, *, age_s: float) -> dict:
    st = state(age_s=age_s)
    st.update(max_count=1, labels=[label], label=label, snap_time=None,
              last_after={**after(label, camera), "score": 0.9})
    p._active[fid] = st
    return st


def named_alert_checks() -> None:
    print("\nan alert names who it is — 'Adam is at the front door'")
    p = named_pipeline()
    live(p, "native.a", "front_door", "person", age_s=0.0)
    p.note_recognition("native.a", "face", name="Adam", profile_id=7, score=0.9)
    check([s["title"] for s in p.sent] == ["Adam is at the front door"],
          f"a face matched inside the hold names the alert (got {[s['title'] for s in p.sent]})")
    check(p.sent and p.sent[0]["body"] == "Recognized Adam", "...and the body says so")

    p = named_pipeline()
    live(p, "native.b", "drive", "car", age_s=0.0)
    p.note_recognition("native.b", "plate", name="the car", profile_id=3, plate="7ABC123")
    check([s["title"] for s in p.sent] == ["The car is in the driveway"],
          f"an enrolled vehicle is named by its profile (got {[s['title'] for s in p.sent]})")
    check(p.sent and p.sent[0]["body"] == "Recognized the car — plate 7ABC123",
          "...with the plate in the body")

    print("\na name matched AFTER the alert went out is put on that alert")
    p = named_pipeline()
    st = live(p, "native.c", "front_door", "person", age_s=5.0)
    p._run(p._maybe_notify_object("native.c", st["last_after"], st))
    check([s["title"] for s in p.sent] == ["Person detected at Front Door"],
          "the hold ran out first, so the alert went unnamed")
    p.note_recognition("native.c", "face", name="Adam", profile_id=7, score=0.9)
    check(len(p.sent) == 2 and p.sent[1]["title"] == "Adam is at the front door",
          "the late match re-sends it named")
    check(len(p.sent) == 2 and p.sent[1]["tag"] == p.sent[0]["tag"]
          and p.sent[1]["event_id"] == p.sent[0]["event_id"],
          "with the SAME tag and event, so the phone replaces the first rather than stacking")
    p.note_recognition("native.c", "face", name="Adam", profile_id=7, score=0.95)
    check(len(p.sent) == 2, "the same name matched again sends nothing more")
    p.note_recognition("native.c", "face", name="Sarah", profile_id=8, score=0.9)
    check(len(p.sent) == 3 and p.sent[2]["title"] == "Adam and Sarah are at the front door",
          "a second person joins the name")

    print("\n...but not when the operator asked not to hear it")
    p = named_pipeline()
    st = live(p, "native.d", "front_door", "person", age_s=5.0)
    p._run(p._maybe_notify_object("native.d", st["last_after"], st))
    p.note_recognition("native.d", "face", name="Adam", profile_id=7, alert_mode="mute")
    check(len(p.sent) == 1, "a MUTED profile is not named on the alert")
    p.note_recognition("native.d", "plate", plate="7ABC123")
    check(len(p.sent) == 1, "an unmatched plate is not a name")

    p = named_pipeline(notify_mode="unknown_only")
    st = live(p, "native.e", "front_door", "person", age_s=5.0)
    p._run(p._maybe_notify_object("native.e", st["last_after"], st))
    p.note_recognition("native.e", "face", name="Adam", profile_id=7)
    check(len(p.sent) == 1,
          "'only strangers' mode does not follow up to say it was someone known")
    p.note_recognition("native.e", "face", name="Mallory", profile_id=9, alert_mode="alert")
    check(len(p.sent) == 2 and p.sent[1]["title"] == "Mallory is at the front door",
          "...except for a WATCHLISTED profile, which is always named")

    p = named_pipeline()
    st = live(p, "native.f", "front_door", "person", age_s=5.0)
    p._run(p._maybe_notify_object("native.f", st["last_after"], st))
    p._settings.notifications["enabled"] = False
    p.note_recognition("native.f", "face", name="Adam", profile_id=7)
    check(len(p.sent) == 1, "notifications switched off in the meantime: nothing sent")

    print("\nstill there — 'Adam is still at the front door'")
    p = named_pipeline()
    st = live(p, "native.g", "front_door", "person", age_s=0.0)
    st["recognitions"].append({**KNOWN, "alert_mode": "default"})
    p._run(p._send_dwell("native.g", "person", 300))
    check([s["title"] for s in p.sent] == ["Adam is still at the front door"],
          f"a dwell alert names them too (got {[s['title'] for s in p.sent]})")


def main() -> int:
    gate_checks()
    note_checks()
    mqtt_checks()
    wording_checks()
    named_alert_checks()
    events_join_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (recognition notification gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
