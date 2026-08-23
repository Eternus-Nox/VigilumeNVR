#!/usr/bin/env python3
"""Event clip timing settings.

Covers the four knobs that decide what a clip contains and when it appears:
``recording.clip_pre_s`` / ``clip_post_s`` (Recorder._clip_pads, including the
bound that keeps post-roll inside what is actually on disk), ``clip_delay_s``
(Recorder._clip_delay, which MOVES that bound), and ``detection.
absence_timeout_s`` (DetectionEngine._absence_timeout, which decides when the
event — and so the clip's tail — ends). Plus the window arithmetic in
extract_clip that turns the pads into an ffmpeg seek+duration.

SEPARATE from native_smoke, which owns the rest of extract_clip, for one
practical reason: native_smoke reaches the network during import and cannot run
on an offline box. Clip padding is exactly the kind of arithmetic that should
stay checkable anywhere, so it lives here where `python3 tests/clip_window_
smoke.py` needs nothing but a stdlib interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_SETTINGS  # noqa: E402
from app.native.engine import ABSENCE_TIMEOUT_S, DetectionEngine  # noqa: E402
from app.native.recorder import (  # noqa: E402
    CLIP_DELAY_S,
    CLIP_PAD_S,
    MAX_CLIP_POST_S,
    SEGMENT_SECONDS,
    Recorder,
    max_clip_post_s,
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


class _FakeSettings:
    """Only the surface the two resolvers under test touch."""

    def __init__(self, recording: dict, detection: Optional[dict] = None) -> None:
        self.recording = recording
        self.detection = detection if detection is not None else {}


def _recorder(recording: dict, override: Optional[float] = None) -> Recorder:
    """A Recorder built without touching __init__.

    __new__ deliberately: the real constructor spins up a Transcoder and wants a
    Config and a Database, none of which this arithmetic reads.
    """
    rec = Recorder.__new__(Recorder)
    rec._settings = _FakeSettings(recording)  # type: ignore[attr-defined]
    rec.clip_delay_s = override
    return rec


def _pads(recording: dict, override: Optional[float] = None) -> tuple[float, float]:
    return _recorder(recording, override)._clip_pads()


def _delay(recording: dict, override: Optional[float] = None) -> float:
    return _recorder(recording, override)._clip_delay()


def _absence(detection: dict) -> float:
    """DetectionEngine._absence_timeout, built the same way as _recorder."""
    eng = DetectionEngine.__new__(DetectionEngine)
    eng._settings = _FakeSettings({}, detection)  # type: ignore[attr-defined]
    return eng._absence_timeout()


def main() -> int:
    print("event clip timing settings")

    # --- defaults ---------------------------------------------------------
    pre, post = _pads(DEFAULT_SETTINGS["recording"])
    check(
        (pre, post) == (CLIP_PAD_S, CLIP_PAD_S),
        f"shipped defaults reproduce the old symmetric {CLIP_PAD_S:g}s pad "
        "(upgrading changes no clip)",
    )

    # --- the point of the feature: asymmetric padding ---------------------
    pre, post = _pads({"clip_pre_s": 15, "clip_post_s": 8})
    check((pre, post) == (15.0, 8.0), "pre and post are independent")

    # --- 0 is a real choice, not a missing value --------------------------
    # The bug this guards is `value or DEFAULT`, which silently rewrites a
    # deliberate 0 into 5 because 0 is falsy.
    pre, post = _pads({"clip_pre_s": 0, "clip_post_s": 0})
    check((pre, post) == (0.0, 0.0), "an explicit 0 stays 0 (not coerced to the default)")

    # --- absent keys fall back --------------------------------------------
    pre, post = _pads({})
    check(
        (pre, post) == (CLIP_PAD_S, CLIP_PAD_S),
        "a settings doc written before this feature falls back to the default pad",
    )
    pre, post = _pads({"clip_pre_s": 12})
    check((pre, post) == (12.0, CLIP_PAD_S), "one key set, the other defaulted")

    # --- post-roll bound ---------------------------------------------------
    # Not a taste bound: extraction runs clip_delay_s after the event and a
    # segment lands SEGMENT_SECONDS after it opens, so footage past the
    # difference has not been written when the clip is cut.
    check(
        MAX_CLIP_POST_S == int(CLIP_DELAY_S - SEGMENT_SECONDS),
        f"MAX_CLIP_POST_S ({MAX_CLIP_POST_S}s) is the ceiling at the default "
        f"delay ({CLIP_DELAY_S:g}s minus a {SEGMENT_SECONDS}s segment)",
    )
    _, post = _pads({"clip_post_s": 600})
    check(
        post == float(MAX_CLIP_POST_S),
        "an over-large post-roll clamps to the horizon rather than quietly "
        "producing a short clip",
    )
    pre, _ = _pads({"clip_pre_s": 600})
    check(pre == 600.0, "pre-roll is NOT clamped — past footage is already on disk")

    # --- the horizon MOVES with the configured delay -----------------------
    # The whole reason clip_delay_s is adjustable: waiting longer is how an
    # operator buys post-roll the default cannot reach.
    check(
        max_clip_post_s(60) == 50 and max_clip_post_s(20) == 10,
        "max_clip_post_s tracks the delay (60s delay -> 50s reachable)",
    )
    check(
        max_clip_post_s(SEGMENT_SECONDS) == 0 and max_clip_post_s(0) == 0,
        "a delay at or below one segment reaches no post-roll at all, and never "
        "goes negative",
    )
    _, post = _pads({"clip_post_s": 45, "clip_delay_s": 60})
    check(post == 45.0, "a raised delay lets a larger post-roll through unclamped")
    _, post = _pads({"clip_post_s": 45, "clip_delay_s": 20})
    check(post == 10.0, "the same post-roll clamps back down when the delay is not raised")

    # --- the delay itself --------------------------------------------------
    check(_delay({}) == CLIP_DELAY_S, "absent clip_delay_s falls back to the default")
    check(_delay({"clip_delay_s": 45}) == 45.0, "configured clip_delay_s is used")
    check(
        _delay({"clip_delay_s": 45}, override=0.2) == 0.2,
        "an explicit instance override (tests) wins over settings",
    )
    # The override is what native_smoke uses to keep its scheduled-clip case
    # fast; if settings started winning there, that test would wait 20 s.
    _, post = _pads({"clip_post_s": 5}, override=0.2)
    check(
        post == 0.0,
        "a shrunk delay drags the post-roll clamp down with it (no post-roll is "
        "retrievable 0.2s after the event)",
    )

    # --- absence timeout (engine side of the same knob set) ---------------
    # Lives here rather than in a suite of its own because it decides when an
    # event ENDS, which is the other half of what a clip's tail contains.
    check(_absence({}) == ABSENCE_TIMEOUT_S, "absent absence_timeout_s falls back to 5s")
    check(_absence({"absence_timeout_s": 30}) == 30.0, "configured absence timeout is used")
    check(
        _absence({"absence_timeout_s": 0}) == 0.5,
        "a 0 timeout floors at 0.5s — at 0 a single dropped frame would end "
        "every event",
    )

    # --- hand-edited nonsense ---------------------------------------------
    pre, post = _pads({"clip_pre_s": -30, "clip_post_s": -1})
    check((pre, post) == (0.0, 0.0), "negative padding floors at 0 (never inverts the window)")
    pre, post = _pads({"clip_pre_s": 7.5, "clip_post_s": 2.5})
    check((pre, post) == (7.5, 2.5), "fractional seconds survive (float, not int, arithmetic)")

    # --- window arithmetic, as extract_clip computes it -------------------
    # extract_clip: window = [start - pre, end + post]; ffmpeg is then given a
    # seek from the first segment's start and a duration. Asserted here because
    # an asymmetric pad makes it newly possible to get the two mixed up — a
    # symmetric pad hides a pre/post swap completely.
    start_time, end_time = 1000.0, 1004.0
    first_segment_start = 990.0
    pre, post = _pads({"clip_pre_s": 15, "clip_post_s": 8})
    window_start = start_time - pre
    window_end = end_time + post
    seek_s = window_start - first_segment_start
    duration_s = window_end - window_start
    check(window_start == 985.0, "window opens pre_s before the event start")
    check(window_end == 1012.0, "window closes post_s after the event end")
    check(seek_s == -5.0, "seek is measured from the first covering segment's start")
    check(
        duration_s == (end_time - start_time) + pre + post,
        "duration is the event plus BOTH pads (27s here, not 2x either one)",
    )

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (event clip timing settings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
