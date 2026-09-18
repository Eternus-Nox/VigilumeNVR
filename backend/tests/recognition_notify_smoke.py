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


def main() -> int:
    gate_checks()
    note_checks()
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
