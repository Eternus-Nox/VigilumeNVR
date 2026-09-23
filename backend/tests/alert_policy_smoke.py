#!/usr/bin/env python3
"""Per-profile notification policy: mute the household, watchlist a stranger.

WHAT THIS IS FOR
================
The global switch is `recognition.notify_mode`, which is all-or-nothing: alert
on everyone, or alert only on people you have NOT enrolled. Neither expresses
the thing people actually want, which is two opposite instructions at once —
"stop telling me when it is me or my wife" AND "tell me the moment THIS person
turns up". So each profile carries its own `alert_mode`.

THE FAILURE TO FEAR IS A MISSED ALERT
-------------------------------------
Every rule below is arranged so that ambiguity resolves towards alerting, and
each of those is asserted from the side that would hurt:

  * an unknown subject can never be muted by somebody else's setting;
  * a muted resident walking in beside a stranger does not silence the
    stranger;
  * one watchlisted person in a group beats everybody else's mute;
  * an unrecognised `alert_mode` string reads as "default", never as "mute" —
    a typo or a value from a newer build must not silently swallow alerts.

`mute` is the only outcome that suppresses permanently, and it requires EVERY
recognized subject to be muted and nothing unmatched in the frame.

Offline: no models, no network, no database — it drives `_recognition_gate`
directly, which is the function that actually decides.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.events_pipeline import EventsPipeline  # noqa: E402
from app.native.recognition import (  # noqa: E402
    ALERT_MODES, Gallery, normalize_alert_mode,
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


class FakeSettings:
    def __init__(self, notify_mode: str = "all"):
        self.recognition = {
            "enabled": True,
            "notify_mode": notify_mode,
            "notify_grace_seconds": 4,
        }


def gate(recognitions, *, notify_mode="all", label="person"):
    """Run the real gate over a hand-built event state.

    `EventsPipeline.__new__` rather than a constructed one: the gate reads only
    `self._settings`, and building a real pipeline would drag in the database,
    the media provider and the push service for a pure decision function.
    """
    pipe = EventsPipeline.__new__(EventsPipeline)
    pipe._settings = FakeSettings(notify_mode)  # type: ignore[attr-defined]
    state = {"recognitions": list(recognitions), "notified": False, "opened_at": None}
    send, name = pipe._recognition_gate({"label": label, "camera": "front"}, state)
    return send, name, state


def known(name, mode="default", pid=1):
    return {"kind": "face", "name": name, "profile_id": pid, "alert_mode": mode,
            "plate": "", "score": 0.9}


def stranger():
    return {"kind": "face", "name": "", "profile_id": None, "alert_mode": "default",
            "plate": "", "score": 0.2}


def normalize_checks() -> None:
    print("\nan unknown mode reads as DEFAULT, never as mute")
    check(ALERT_MODES == ("default", "mute", "alert"), "the mode set is closed")
    for bad in (None, "", "muted", "MUTE ", "silence", 7, [], {"a": 1}):
        got = normalize_alert_mode(bad)
        expected = "mute" if str(bad).strip().lower() == "mute" else None
        if expected:
            check(got == "mute", f"{bad!r} -> mute")
        else:
            check(got == "default",
                  f"{bad!r} -> default, not a policy nobody implemented "
                  f"(got {got!r})")
    check(normalize_alert_mode(" Alert ") == "alert",
          "case and whitespace are tolerated on a value that IS a mode")


def gate_checks() -> None:
    print("\nmute: the household case")
    send, _, state = gate([known("Adam", "mute")])
    check(send is False, "a muted profile does not alert")
    check(state["notified"] is True,
          "...and is marked notified, so the next update does not re-decide it "
          "and alert a second later")

    send, name, _ = gate([known("Adam", "mute"), known("Sam", "mute", pid=2)])
    check(send is False, "two muted people together stay muted")

    print("\n...but a mute must never silence someone else")
    send, _, _ = gate([known("Adam", "mute"), stranger()])
    check(send is True,
          "a muted resident walking in BESIDE A STRANGER still alerts — the "
          "stranger is the whole reason the system exists")
    send, _, _ = gate([known("Adam", "mute"), known("Guest", "default", pid=3)])
    check(send is True,
          "a muted person beside a non-muted person alerts — mute is 'do not "
          "tell me about ME', not 'ignore this frame'")

    print("\nalert: the watchlist case")
    send, name, _ = gate([known("Trouble", "alert")])
    check(send is True and name == "Trouble", "a watchlisted profile alerts, named")
    send, name, _ = gate([known("Trouble", "alert")], notify_mode="unknown_only")
    check(send is True and name == "Trouble",
          "...even under unknown_only, which would otherwise suppress every "
          "enrolled subject. The specific instruction beats the general one")
    send, name, _ = gate([known("Adam", "mute"), known("Trouble", "alert", pid=2)])
    check(send is True and name == "Trouble",
          "one watchlisted person beats everybody else's mute, and the alert is "
          "named after THEM rather than after whoever was recognized first")

    print("\ndefault: the global setting still decides")
    send, name, _ = gate([known("Adam")])
    check(send is True and name == "Adam", "notify_mode=all alerts on a known face")
    send, _, state = gate([known("Adam")], notify_mode="unknown_only")
    check(send is False and state["notified"] is True,
          "notify_mode=unknown_only still suppresses a default profile")
    send, _, _ = gate([stranger()], notify_mode="unknown_only")
    check(send is True,
          "an unmatched face alerts under unknown_only — that is the mode's "
          "entire point")

    print("\na corrupt stored mode cannot swallow an alert")
    send, _, _ = gate([known("Adam", "muted")])  # a typo, not a mode
    check(send is True,
          "'muted' is not a mode, so it reads as default and the alert goes — "
          "the alternative is a person silently unmonitored for months")
    send, _, _ = gate([known("Adam", None)])
    check(send is True, "a NULL mode reads as default too")

    print("\nrecognition off, or a label that cannot be recognized")
    pipe = EventsPipeline.__new__(EventsPipeline)
    pipe._settings = FakeSettings()  # type: ignore[attr-defined]
    pipe._settings.recognition["enabled"] = False  # type: ignore[attr-defined]
    send, _ = pipe._recognition_gate(
        {"label": "person", "camera": "front"},
        {"recognitions": [known("Adam", "mute")], "notified": False},
    )
    check(send is True,
          "with recognition DISABLED the gate opens regardless of any stored "
          "policy — a switched-off feature must not keep muting people")
    send, _, _ = gate([known("Adam", "mute")], label="dog")
    check(send is True,
          "a label that carries no recognition is unaffected by a profile's mode")


def gallery_checks() -> None:
    """The policy must survive the trip from a DB row to a Match.

    Exercised through `match_plate` rather than by reading gallery internals:
    plates need no embedding, so this is the one matcher that can be driven
    end to end offline — and it is the real path, so it would catch the
    carry-through being dropped anywhere along it.
    """
    print("\nthe policy rides the match, not a database lookup")

    def gallery(mode):
        rows = [{"id": 7, "kind": "vehicle", "name": "Van", "enabled": 1,
                 "threshold": None, "alert_mode": mode}]
        samples = [{"id": 1, "profile_id": 7, "plate": "7ABC123",
                    "embedding": None, "dim": 0, "model_key": ""}]
        return Gallery.build(rows, samples, model_key="")

    m = gallery("mute").match_plate("7ABC123")
    check(m.matched and m.alert_mode == "mute",
          f"a matched plate carries its profile's policy out to the pipeline "
          f"(got {m.alert_mode!r}) — no database lookup on the alert path")

    m = gallery("nonsense").match_plate("7ABC123")
    check(m.alert_mode == "default",
          f"a nonsense stored value is normalized AT LOAD, so an unknown "
          f"policy can never reach the gate (got {m.alert_mode!r})")

    m = gallery("mute").match_plate("ZZZ9999")
    check(not m.matched and m.alert_mode == "default",
          "an UNMATCHED read carries 'default' — this is what stops an unknown "
          "vehicle inheriting an enrolled one's mute")


def main() -> int:
    normalize_checks()
    gate_checks()
    gallery_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (per-profile alert policy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
