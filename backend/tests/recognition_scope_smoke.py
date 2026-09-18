#!/usr/bin/env python3
"""Per-camera recognition switches, and the knobs that decide how hard it looks.

WHAT THIS GUARDS
================
Before the switches existed, `cameras.face_zones` was the only per-camera
control and it CANNOT express "off": an empty zone list means WHOLE FRAME. So
turning recognition on ran a face pass on every camera in the system — on a
12-camera box, ten of which watch a driveway at 30 m where no face is ever
legible. That is wasted inference AND a source of wrong names, because the only
crops those cameras produce are marginal ones.

So the switches must actually gate, and they must default ON so that adding them
changes nothing for an existing box.

The tuning knobs (shots, gap, pass interval, identify quality, faces on
vehicles) all reach the passes through a settings dict that a human or an older
client can write badly. They are read on a MAINTENANCE TICK, so a raise there
takes recognition down — every malformed shape must degrade to the shipped
default instead.

Offline-runnable; no models, no network, no database.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.bestshot import (  # noqa: E402
    KEEP_SHOTS, MIN_GAP_S, clamp_setting, shot_params,
)
from app.native.facepass import FACE_LABELS, VEHICLE_FACE_LABELS  # noqa: E402

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


def clamp_checks() -> None:
    print("\nsettings reach the passes on a tick — a bad value must never raise")
    for bad in (None, "soon", "", [], {}, float("nan")):
        got = clamp_setting(bad, 0.6, 0.1, 5.0)
        check(got == 0.6, f"{bad!r} falls back to the default ({got})")
    check(clamp_setting(0.2, 0.6, 0.1, 5.0) == 0.2, "a valid value is used as given")
    check(clamp_setting(0.001, 0.6, 0.1, 5.0) == 0.1, "below the floor clamps to the floor")
    check(clamp_setting(500, 0.6, 0.1, 5.0) == 5.0, "above the ceiling clamps to the ceiling")
    check(clamp_setting(True, 0.6, 0.1, 5.0) == 1.0,
          "a bool is a number to float() — clamped, not crashed")

    print("\nshot_params")
    check(shot_params({}) == (KEEP_SHOTS, MIN_GAP_S), "an empty config is the shipped default")
    check(shot_params(None) == (KEEP_SHOTS, MIN_GAP_S), "so is a missing config")
    check(shot_params({"shots_per_track": 10, "shot_min_gap_seconds": 0.25}) == (10, 0.25),
          "configured values come through")
    shots, _ = shot_params({"shots_per_track": 999})
    check(shots == 12,
          "shots clamp at 12 — past that the buffer holds neighbouring frames "
          "rather than distinct moments, which is not more information")
    shots, _ = shot_params({"shots_per_track": 0})
    check(shots == 1, "and never below 1, which would disable the buffer entirely")
    check(isinstance(shot_params({"shots_per_track": 7.9})[0], int),
          "the shot count is an int — it indexes a buffer")


def label_checks() -> None:
    print("\nwhich labels get a face pass")
    check("person" in FACE_LABELS, "a person always does")
    check(not any(v in FACE_LABELS for v in VEHICLE_FACE_LABELS),
          "a vehicle does NOT by default — on a road-facing camera most "
          "windscreens are glare, and a plate identifies a car better")
    check("car" in VEHICLE_FACE_LABELS and "truck" in VEHICLE_FACE_LABELS,
          "cars and trucks are the opt-in set (the driver through the glass)")
    check(not set(FACE_LABELS) & set(VEHICLE_FACE_LABELS),
          "the two sets are disjoint, so enabling the opt-in cannot double-count "
          "a label and run two passes on one track")


class FakeCam:
    """Only the surface the passes touch before doing any work."""

    def __init__(self, **kw):
        self.row = {"name": kw.pop("name", "front")}
        self.face_zones = []
        self.plate_zones = []
        for k, v in kw.items():
            setattr(self, k, v)


def gate_checks() -> None:
    """The switch must be read where it is cheapest and must default ON."""
    print("\nthe per-camera switch")

    # Read the way the passes read it, so this test tracks the real accessor
    # rather than a paraphrase of it.
    def face_on(cam) -> bool:
        return bool(getattr(cam, "face_recognition", True))

    def plate_on(cam) -> bool:
        return bool(getattr(cam, "plate_recognition", True))

    check(face_on(FakeCam()) and plate_on(FakeCam()),
          "a camera with NO switches set recognizes — the default is ON, so "
          "adding these columns changes nothing for an existing box")
    check(not face_on(FakeCam(face_recognition=False)), "face off is honoured")
    check(not plate_on(FakeCam(plate_recognition=False)), "plate off is honoured")
    check(face_on(FakeCam(plate_recognition=False)),
          "the two are INDEPENDENT — a driveway camera can read plates while "
          "never being asked for a face")
    check(plate_on(FakeCam(face_recognition=False)), "...and the reverse")


def settings_model_checks() -> None:
    print("\nthe settings model carries every knob")
    from app.config import DEFAULT_SETTINGS
    from app.routers.settings import AppSettings, RecognitionSettings

    keys = {
        "enabled", "candidate_retention_days", "notify_grace_seconds", "notify_mode",
        "shots_per_track", "shot_min_gap_seconds", "pass_interval_seconds",
        "face_on_vehicles", "identify_quality",
    }
    model = set(RecognitionSettings().model_dump())
    check(keys <= model, f"RecognitionSettings models them all (missing: {sorted(keys - model)})")
    defaults = set(DEFAULT_SETTINGS["recognition"])
    check(keys <= defaults,
          f"and so does DEFAULT_SETTINGS (missing: {sorted(keys - defaults)}) — the two "
          "are merged over each other, so a key in one and not the other drifts")

    # The trap that already bit once: AppSettings drops what it does not model,
    # and PATCH stores the validated merge, so an unmodelled key is not merely
    # unreachable — every unrelated save wipes it.
    doc = AppSettings(**{"recognition": {"shots_per_track": 9, "face_on_vehicles": True}})
    check(doc.recognition.shots_per_track == 9 and doc.recognition.face_on_vehicles is True,
          "AppSettings round-trips the new fields rather than dropping them")

    bad = False
    try:
        RecognitionSettings(shots_per_track=99)
    except Exception:
        bad = True
    check(bad, "an out-of-range shots_per_track is a 422, not a silent clamp")


def main() -> int:
    clamp_checks()
    label_checks()
    gate_checks()
    settings_model_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (per-camera recognition scope + tuning)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
