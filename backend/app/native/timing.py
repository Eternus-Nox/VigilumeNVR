"""Per-stage timing for the recognition pipeline, measured rather than assumed.

WHY THIS EXISTS
===============
Every claim about what recognition costs — "~4 ms on a person crop", "~1.5% of
one core per person" — was measured once, on one box, on one set of settings.
The moment an operator raises the shot count, shortens the pass interval or
turns on faces-for-vehicles, those numbers stop describing their system.

"Should this move to the GPU?" is then unanswerable from the outside, and the
honest answer to it is a measurement from the box in question, not an estimate
from mine. So the stages record what they actually take, and
`/api/recognition/status` serves it.

WHAT IT COSTS TO HAVE
---------------------
One `time.perf_counter()` pair and an O(1) update per call. That is nanoseconds
against stages measured in milliseconds, so it is always on — a profiler you
have to enable is one nobody enables before the question comes up.

WHY A ROLLING WINDOW AND NOT A MEAN
-----------------------------------
A lifetime mean over a week of uptime is dominated by conditions that no longer
hold: the first call includes model warmup, and a quiet night of no detections
buries the busy hour that actually matters. A window of the last N calls tells
you what the box is doing NOW, which is the question being asked.

p95 is reported alongside the mean because the mean hides the case that hurts.
A face pass that usually takes 4 ms and occasionally takes 40 is a different
system from one that always takes 6, and they have the same mean.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Optional

#: Calls retained per stage. At a face pass every 0.6 s per track this is a few
#: minutes of history — long enough to be stable, short enough to still be about
#: the present.
WINDOW = 256


class Stage:
    """Rolling timing for one named stage. Thread-safe; never raises."""

    __slots__ = ("name", "_samples", "_lock", "_total", "_count")

    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: deque[float] = deque(maxlen=WINDOW)
        self._lock = threading.Lock()
        # Lifetime counters kept ALONGSIDE the window, not instead of it: the
        # window answers "what is it doing now", the count answers "is this
        # stage running at all", which is the first thing you check when a
        # number looks wrong.
        self._total = 0.0
        self._count = 0

    def record(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)
            self._total += ms
            self._count += 1

    def measure(self) -> "_Timer":
        """`with stage.measure():` — records the elapsed ms on exit."""
        return _Timer(self)

    def snapshot(self) -> Optional[dict[str, Any]]:
        """Stats, or None when the stage has never run.

        None rather than zeroes, deliberately: a stage that has not run and a
        stage that runs instantly are completely different findings, and zeroes
        would present the first as the second.
        """
        with self._lock:
            if not self._count:
                return None
            window = sorted(self._samples)
        if not window:
            return None
        # Nearest-rank p95. Over a window this small an interpolated percentile
        # implies a precision the sample size does not support.
        index = min(len(window) - 1, int(round(0.95 * (len(window) - 1))))
        return {
            "calls": self._count,
            "mean_ms": round(sum(window) / len(window), 2),
            "p95_ms": round(window[index], 2),
            "max_ms": round(window[-1], 2),
            "lifetime_mean_ms": round(self._total / self._count, 2),
        }


class _Timer:
    __slots__ = ("_stage", "_start")

    def __init__(self, stage: Stage) -> None:
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        # Records even when the body raised. A stage that is failing slowly is
        # precisely the one worth seeing in the numbers, and swallowing its
        # timing would hide it. Returns False so the exception still propagates.
        self._stage.record((time.perf_counter() - self._start) * 1000.0)
        return False


class Timings:
    """The recognition pipeline's stages, in one place."""

    def __init__(self) -> None:
        self.face_detect = Stage("face_detect")
        self.face_align = Stage("face_align")
        self.face_embed = Stage("face_embed")
        self.plate_localize = Stage("plate_localize")
        self.plate_ocr = Stage("plate_ocr")

    def report(self) -> dict[str, Any]:
        """Stages that have actually run, for /api/recognition/status."""
        out: dict[str, Any] = {}
        for stage in (
            self.face_detect, self.face_align, self.face_embed,
            self.plate_localize, self.plate_ocr,
        ):
            snap = stage.snapshot()
            if snap is not None:
                out[stage.name] = snap
        return out


#: One process-wide instance. A module global rather than something threaded
#: through every constructor because it is pure instrumentation: it must never
#: become a parameter that a caller can forget to pass and thereby silently
#: stop measuring.
TIMINGS = Timings()
