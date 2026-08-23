#!/usr/bin/env python3
"""Clip window padding — settings.recording.clip_pre_s / clip_post_s.

Covers Recorder._clip_pads (the settings -> (pre, post) resolution, including
the bound that keeps post-roll inside what is actually on disk) and the window
arithmetic in extract_clip that turns those pads into an ffmpeg seek+duration.

SEPARATE from native_smoke, which owns the rest of extract_clip, for one
practical reason: native_smoke reaches the network during import and cannot run
on an offline box. Clip padding is exactly the kind of arithmetic that should
stay checkable anywhere, so it lives here where `python3 tests/clip_window_
smoke.py` needs nothing but a stdlib interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_SETTINGS  # noqa: E402
from app.native.recorder import (  # noqa: E402
    CLIP_DELAY_S,
    CLIP_PAD_S,
    MAX_CLIP_POST_S,
    SEGMENT_SECONDS,
    Recorder,
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
    """Only the surface Recorder._clip_pads touches."""

    def __init__(self, recording: dict) -> None:
        self.recording = recording


def _pads(recording: dict) -> tuple[float, float]:
    """_clip_pads on a Recorder built without touching __init__.

    __new__ deliberately: the real constructor spins up a Transcoder and wants a
    Config and a Database, none of which this arithmetic reads.
    """
    rec = Recorder.__new__(Recorder)
    rec._settings = _FakeSettings(recording)  # type: ignore[attr-defined]
    return rec._clip_pads()


def main() -> int:
    print("clip window padding")

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
    # Not a taste bound: extraction runs CLIP_DELAY_S after the event and a
    # segment lands SEGMENT_SECONDS after it opens, so footage past the
    # difference has not been written when the clip is cut.
    check(
        MAX_CLIP_POST_S == int(CLIP_DELAY_S - SEGMENT_SECONDS),
        f"MAX_CLIP_POST_S ({MAX_CLIP_POST_S}s) is derived from the extraction "
        f"delay ({CLIP_DELAY_S:g}s) minus a segment ({SEGMENT_SECONDS}s)",
    )
    _, post = _pads({"clip_post_s": 600})
    check(
        post == float(MAX_CLIP_POST_S),
        "an over-large post-roll clamps to the horizon rather than quietly "
        "producing a short clip",
    )
    pre, _ = _pads({"clip_pre_s": 600})
    check(pre == 600.0, "pre-roll is NOT clamped — past footage is already on disk")

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
    print(f"ALL {_checks} CHECKS PASSED (clip window padding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
