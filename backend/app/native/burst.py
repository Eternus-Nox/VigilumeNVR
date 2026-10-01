"""Recorded bursts: reading the full-resolution recording while an object is
still in view, starting from just BEFORE it was detected.

WHY
===
The live looks are late by construction. Detection runs on the ~704x480
substream at a few frames a second and confirms a track over three frames;
a full-resolution snapshot is asked for after that and arrives a few hundred
milliseconds later. A car crossing the view in a second and a half, or a
person walking briskly past, has often turned or gone by the time the first
look lands — which is why plates were being read when a car backed slowly out
of the driveway, and missed when one drove in.

The recorder has been writing the camera's full-resolution stream to disk the
whole time, including the seconds before anything was detected. So, starting
FIRST_BURST_AFTER_S after a track is first seen, its pass decodes the
recording from PRE_ROLL_S before that first sighting up to (almost) now, at
BURST_FPS, cropped to where the object was, and reads every frame. It repeats
every BURST_EVERY_S while the object is in view, and once more TAIL_DELAY_S
after its last sighting for the seconds as it left. Every frame is a look: a
second of burst is ten looks at full resolution, where the live path gets one.
The answer is then decided from all of them — plates by the per-character vote,
faces from their best few shots together.

REC_LAG_S is how far behind real time the recording on disk is assumed to be
(go2rtc relay + the muxer's interleaving + write buffering). A burst never
reads past it, and records how far it actually got from the frames that came
back, so an underestimate just leaves the rest to the next burst.

State per track lives in `BurstState`; the passes own the reading. Nothing
here does I/O.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

#: How far back before the first sighting a track's first burst starts.
PRE_ROLL_S = 2.0

#: How far behind real time the recording on disk is assumed to be.
REC_LAG_S = 0.8

#: The first burst, this long after the first sighting.
FIRST_BURST_AFTER_S = 1.0

#: A further live burst once at least this much new recording is available.
BURST_EVERY_S = 1.2

#: The final burst reads this far past the last sighting.
POST_ROLL_S = 0.7

#: And starts this long after the last sighting (the post-roll has to be on
#: disk by then).
TAIL_DELAY_S = POST_ROLL_S + REC_LAG_S + 0.3

#: Frames decoded per second of recording.
BURST_FPS = 10.0

#: Bursts per track, the final one included. A parked car or someone standing
#: at the door does not get decoded for as long as they stay.
MAX_BURSTS = 6

#: The longest window one burst decodes; anything beyond is left to the next.
MAX_BURST_WINDOW_S = 4.0

#: A final burst shorter than this is not worth starting.
MIN_FINAL_SPAN_S = 0.2

#: How far a recorded frame's time may be from the detect stream's clock. The
#: segment start is exact to milliseconds (platereplay.segment_starts); the two
#: streams' paths through go2rtc differ by a few hundred.
TIME_TOLERANCE_S = 0.35


@dataclass
class BurstState:
    first_seen: Optional[float] = None
    last_seen: float = 0.0
    #: The recording has been read up to here.
    covered_to: Optional[float] = None
    count: int = 0
    task: Optional["asyncio.Task[None]"] = None
    tail: Optional[asyncio.TimerHandle] = None
    #: A burst came back with no frames at all: no recording to read for this
    #: track (the camera is not recording, or it has been pruned).
    unavailable: bool = False

    def note_seen(self, t: float) -> None:
        if self.first_seen is None:
            self.first_seen = t
        self.last_seen = max(self.last_seen, t)

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def window(
        self, now: float = 0.0, *, final: bool = False, cap: float = MAX_BURST_WINDOW_S,
    ) -> Optional[tuple[float, float]]:
        """The next stretch of recording to read, or None if there is none
        worth reading yet. `now` is the detect stream's clock; a final window
        ends POST_ROLL_S after the last sighting instead. `cap` bounds its
        length (the after-the-track read may take a longer one)."""
        if self.first_seen is None or self.unavailable:
            return None
        start = self.covered_to if self.covered_to is not None else self.first_seen - PRE_ROLL_S
        end = self.last_seen + POST_ROLL_S if final else now - REC_LAG_S
        end = min(end, start + cap)
        if end - start < (MIN_FINAL_SPAN_S if final else BURST_EVERY_S):
            return None
        return start, end

    def due(self, now: float) -> Optional[tuple[float, float]]:
        """The window of a live burst to start now, if one is due."""
        if self.busy or self.count >= MAX_BURSTS or self.first_seen is None:
            return None
        if now - self.first_seen < FIRST_BURST_AFTER_S:
            return None
        return self.window(now)

    def started(self) -> None:
        self.count += 1

    def finished(self, start: float, end: float, last_frame: Optional[float], fps: float) -> None:
        """Record what a burst over [start, end] actually covered: up to just
        past its last frame. No frames at all means there is nothing to read."""
        if last_frame is None:
            self.unavailable = True
            return
        reached = min(end, last_frame + 1.0 / fps)
        self.covered_to = max(self.covered_to or reached, reached)

    def cancel(self) -> None:
        if self.tail is not None:
            self.tail.cancel()
            self.tail = None
        if self.busy:
            assert self.task is not None
            self.task.cancel()
