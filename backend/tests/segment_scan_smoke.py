#!/usr/bin/env python3
"""select_segments — correctness of the window, and the cost of finding it.

A clip window is tens of seconds; a camera-day is 24 hour dirs of ~360 segments.
Pruning only by day meant every clip extraction walked the whole day to find
three or four files. These checks pin BOTH halves: the same segments come back
as before, and the walk that finds them touches far fewer directory entries.

The cost half is measured, not asserted by eye — os.scandir is wrapped in a
counter, so a future change that reintroduces the full-day walk fails here
instead of quietly costing I/O on every event.

Offline-runnable (see storage_smoke / clip_window_smoke for why that matters).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native import recorder  # noqa: E402
from app.native.recorder import SEGMENT_SECONDS, select_segments  # noqa: E402

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


class _CountingScandir:
    """os.scandir wrapper that counts the entries actually consumed."""

    def __init__(self) -> None:
        self.entries = 0
        self._real = os.scandir

    def __call__(self, path):  # noqa: ANN001
        outer = self

        class _Counted:
            def __init__(self) -> None:
                self._it = outer._real(path)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._it.__exit__(*exc)

            def __iter__(self):
                for entry in self._it:
                    outer.entries += 1
                    yield entry

        return _Counted()


def _reference_select(
    camera_dir: Path, window_start: float, window_end: float
) -> list[tuple[float, Path]]:
    """select_segments as it was BEFORE hour pruning: prune by day, then walk
    every hour dir of every candidate day. Kept deliberately naive — it is the
    oracle, so it must be obviously correct rather than fast."""
    from datetime import date as _date

    from app.native.recorder import _segment_start

    if not camera_dir.is_dir() or window_end <= window_start:
        return []
    lo = window_start - SEGMENT_SECONDS
    first_day, last_day = _date.fromtimestamp(lo), _date.fromtimestamp(window_end)
    out: list[tuple[float, Path]] = []
    for day_p in sorted(camera_dir.iterdir()):
        try:
            day = _date.fromisoformat(day_p.name)
        except ValueError:
            continue
        if day < first_day or day > last_day or not day_p.is_dir():
            continue
        for hour_p in sorted(day_p.iterdir()):
            if not hour_p.is_dir():
                continue
            for seg_p in sorted(hour_p.iterdir()):
                if not seg_p.name.endswith(".ts"):
                    continue
                ts = _segment_start(day, hour_p.name, seg_p.name)
                if ts is not None and lo <= ts < window_end:
                    out.append((ts, seg_p))
    out.sort(key=lambda item: (item[0], str(item[1])))
    return out


def build_day(cam_dir: Path, day: str, hours: range, segs_per_hour: int) -> None:
    """A realistic camera-day: every hour populated with 10 s segments."""
    for hour in hours:
        hd = cam_dir / day / f"{hour:02d}"
        hd.mkdir(parents=True, exist_ok=True)
        for i in range(segs_per_hour):
            secs = i * SEGMENT_SECONDS
            (hd / f"{secs // 60:02d}.{secs % 60:02d}.ts").write_bytes(b"\x47")


def main() -> int:
    print("select_segments: window correctness + scan cost")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-segscan-"))
    try:
        cam = tmp / "front"
        day = "2026-03-14"
        # 24 h x 360 segments = a full day at 10 s segments, as a real camera writes.
        build_day(cam, day, range(24), 360)

        noon = datetime(2026, 3, 14, 12, 0, 0).timestamp()
        window_start, window_end = noon + 30, noon + 60

        counter = _CountingScandir()
        real_scandir = os.scandir
        recorder.os.scandir = counter  # type: ignore[assignment]
        try:
            got = select_segments(cam, window_start, window_end)
        finally:
            recorder.os.scandir = real_scandir  # type: ignore[assignment]

        # --- correctness -------------------------------------------------
        # Window [12:00:30, 12:01:00) plus the SEGMENT_SECONDS lead-in, so the
        # segment starting at 12:00:20 (which still runs into the window) counts.
        names = [p.name for _, p in got]
        check(names == ["00.20.ts", "00.30.ts", "00.40.ts", "00.50.ts"],
              f"exactly the intersecting segments, in order (got {names})")
        check(all(p.parent.name == "12" for _, p in got), "all from the 12:00 hour dir")
        check([ts for ts, _ in got] == sorted(ts for ts, _ in got), "sorted by start time")

        # --- cost ----------------------------------------------------------
        # Day-level pruning alone would walk 24 hour dirs x 360 segments.
        full_day_entries = 24 * 360
        check(
            counter.entries < full_day_entries // 5,
            f"finding 4 segments touched {counter.entries} entries, not the "
            f"~{full_day_entries} a full-day walk costs",
        )

        # --- the pruning must not lose footage at boundaries ---------------
        # Opening exactly on the hour puts the lead-in (window_start -
        # SEGMENT_SECONDS) at 12:59:50, so the LAST segment of the previous hour
        # dir is in range and the scan must still reach it.
        boundary = datetime(2026, 3, 14, 13, 0, 0).timestamp()
        got_b = select_segments(cam, boundary, boundary + 15)
        hours_b = sorted({p.parent.name for _, p in got_b})
        check(
            hours_b == ["12", "13"],
            f"a window opening on the hour still reaches the previous hour dir "
            f"(got {hours_b})",
        )
        check(
            [p.name for _, p in got_b] == ["59.50.ts", "00.00.ts", "00.10.ts"],
            "including the segment that starts before the hour and runs into it",
        )

        # Midnight: the window spans two DAY dirs, which day-level pruning
        # already handled — re-checked because hour pruning must not undo it.
        build_day(cam, "2026-03-15", range(0, 2), 360)
        midnight = datetime(2026, 3, 15, 0, 0, 0).timestamp()
        got_m = select_segments(cam, midnight - 5, midnight + 15)
        days_m = sorted({p.parent.parent.name for _, p in got_m})
        check(days_m == ["2026-03-14", "2026-03-15"], "a window across midnight spans both days")

        # --- differential: identical results to the pre-optimization walk ----
        # The real guard. Hand-written expectations only cover the windows I
        # thought of; this replays the OLD algorithm (day-level pruning, every
        # hour dir scanned) over a spread of windows and demands the same answer
        # every time. Any window where hour pruning drops a segment shows up
        # here, including ones I would not have thought to write down.
        probes = [
            (noon + 30, noon + 60, "mid-hour"),
            (boundary, boundary + 15, "opening on the hour"),
            (boundary - 15, boundary + 5, "spanning the hour"),
            (boundary - 0.5, boundary + 0.5, "sub-second across the hour"),
            (midnight - 5, midnight + 15, "across midnight"),
            (noon, noon + 7200, "two whole hours"),
            (noon - 3600 * 13, noon + 3600 * 13, "wider than the day itself"),
            (noon + 5, noon + 5, "empty window"),
            (noon + 1e6, noon + 1e6 + 30, "far past every segment"),
            (noon - 1e6, noon - 1e6 + 30, "far before every segment"),
        ]
        mismatches = [
            why for start, end, why in probes
            if _reference_select(cam, start, end) != select_segments(cam, start, end)
        ]
        check(
            not mismatches,
            f"identical to the pre-optimization full-day walk on {len(probes)} "
            f"windows{'' if not mismatches else ' — differs on: ' + ', '.join(mismatches)}",
        )

        # --- degenerate input ------------------------------------------------
        check(select_segments(cam, window_end, window_start) == [], "inverted window -> []")
        check(select_segments(tmp / "nope", 0.0, 100.0) == [], "missing camera dir -> []")
        # A stray non-hour dir must be skipped, not crash the scan.
        (cam / day / "notanhour").mkdir(exist_ok=True)
        check(
            [p.name for _, p in select_segments(cam, window_start, window_end)] == names,
            "a non-numeric dir beside the hour dirs is ignored",
        )

        print()
        if _failures:
            print(f"{len(_failures)} of {_checks} CHECKS FAILED")
            for f in _failures:
                print(f"  - {f}")
            return 1
        print(f"ALL {_checks} CHECKS PASSED (select_segments window + scan cost)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
