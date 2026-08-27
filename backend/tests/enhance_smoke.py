#!/usr/bin/env python3
"""Night contrast boost for the detector's input frame.

The property that matters most is NEGATIVE: this must never alter the frame the
engine keeps. That frame becomes the event snapshot — the evidence — and a
detection aid has no business rewriting it. Ingest boosts a copy; these checks
prove the original survives byte-for-byte.

The rest pin the gate (off / auto / always), that a boost actually raises local
contrast, and that malformed settings degrade to OFF rather than to a boost
nobody asked for.

Offline-runnable; synthesises its own frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.enhance import (  # noqa: E402
    DEFAULT_DARK_THRESHOLD,
    DEFAULT_MODE,
    VALID_MODES,
    boost_contrast,
    maybe_boost,
    mean_luma,
    settings_for,
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


def night_frame() -> np.ndarray:
    """A dim, low-contrast scene: a faint shape on a dark ground, plus noise —
    the case this feature exists for."""
    rng = np.random.default_rng(7)
    f = np.full((120, 160, 3), 18, dtype=np.uint8)
    f[40:80, 50:110] = 34                       # a barely-there subject
    noise = rng.integers(0, 7, f.shape, dtype=np.int16)
    return np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def day_frame() -> np.ndarray:
    rng = np.random.default_rng(11)
    f = np.full((120, 160, 3), 165, dtype=np.uint8)
    f[40:80, 50:110] = 70
    noise = rng.integers(0, 7, f.shape, dtype=np.int16)
    return np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    print("night contrast boost (detector input only)")

    night, day = night_frame(), day_frame()
    check(mean_luma(night) < DEFAULT_DARK_THRESHOLD,
          f"the synthetic night frame reads as dark ({mean_luma(night):.0f} luma)")
    check(mean_luma(day) >= DEFAULT_DARK_THRESHOLD,
          f"and the day frame does not ({mean_luma(day):.0f} luma)")
    check(mean_luma(np.zeros((4, 4, 3), np.uint8)) == 0.0, "a black frame reads 0")

    # --- THE critical property: the original is never touched --------------
    before = night.copy()
    out, applied = maybe_boost(night, mode="always")
    check(applied and out is not night, "a boost returns a NEW array")
    check(
        np.array_equal(night, before),
        "and leaves the caller's frame BYTE-FOR-BYTE unchanged — that frame "
        "becomes the event snapshot, and evidence must stay what the camera saw",
    )

    # --- it does something ---------------------------------------------------
    boosted = boost_contrast(night)
    check(boosted.shape == night.shape and boosted.dtype == night.dtype,
          "shape and dtype are preserved, so the detector's preprocess is unaffected")
    check(
        float(boosted.std()) > float(night.std()),
        f"local contrast rises ({night.std():.1f} -> {boosted.std():.1f} std) — "
        "there is more for the model to separate",
    )
    # Colour must not swing: CLAHE per-BGR-channel would equalise each channel to
    # its own histogram and tint the scene. Luma-only keeps the balance.
    def channel_spread(f: np.ndarray) -> float:
        return float(np.ptp(f.reshape(-1, 3).mean(axis=0)))
    check(
        channel_spread(boosted) - channel_spread(night) < 8.0,
        "without a colour cast — CLAHE runs on LUMA only, not per channel",
    )

    # --- the gate ------------------------------------------------------------
    _, applied = maybe_boost(night, mode="off")
    check(not applied, "off never boosts, however dark the frame")
    out, applied = maybe_boost(night, mode="off")
    check(out is night, "and returns the SAME object, so the common path is free")
    _, applied = maybe_boost(night, mode="auto")
    check(applied, "auto boosts a dark frame")
    out, applied = maybe_boost(day, mode="auto")
    check(not applied and out is day,
          "auto leaves a DAYLIGHT frame completely alone — no cost, no change")
    _, applied = maybe_boost(day, mode="always")
    check(applied, "always boosts regardless of light (for A/B against off)")
    _, applied = maybe_boost(day, mode="auto", dark_threshold=255)
    check(applied, "the threshold is honoured, not hardcoded")

    # --- settings degrade safely --------------------------------------------
    check(settings_for({}) == (DEFAULT_MODE, DEFAULT_DARK_THRESHOLD),
          "an empty detection block gives the defaults")
    check(settings_for(None) == (DEFAULT_MODE, DEFAULT_DARK_THRESHOLD),
          "and so does a missing one (ingest may hold no settings store)")
    check(
        settings_for({"night_boost": "ALWAYS"}) [0] == "always",
        "the mode is case-insensitive",
    )
    check(
        settings_for({"night_boost": "enhance-please"})[0] == "off",
        "an UNKNOWN mode degrades to off, not to a boost — a typo must never "
        "silently start altering what the detector sees",
    )
    check(
        settings_for({"night_boost_threshold": "abc"})[1] == DEFAULT_DARK_THRESHOLD,
        "an unparseable threshold falls back to the default",
    )
    check(
        settings_for({"night_boost_threshold": 900})[1] == 255
        and settings_for({"night_boost_threshold": -5})[1] == 0,
        "and an out-of-range one clamps into 0-255",
    )
    check(DEFAULT_MODE == "off" and "off" in VALID_MODES,
          "the shipped default is OFF — this changes model input, so it is opt-in")

    # --- degenerate input ----------------------------------------------------
    out, applied = maybe_boost(None, mode="always")
    check(out is None and not applied, "a missing frame is a no-op, not a crash")

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (night contrast boost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
