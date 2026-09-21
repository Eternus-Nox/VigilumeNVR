"""Which tracked objects are actually DOING something — per-track motion state.

THE PROBLEM
===========
An object detector has no notion of news. It answers "is there a car here?" on
every frame, so a car parked in the driveway is detected five times a second
for as long as it is parked. In an event model keyed on (camera, label) — one
open event per label, ended only when the label goes ABSENT (see engine.py) —
that has three consequences, and only the first is the one people notice:

1. the car's event never ends, and heartbeats an "update" every 10 s forever;
2. the same is true of anything the detector reads as an object and is wrong
   about: a wheelie bin that scores 0.6 as a person holds a person event open
   all night;
3. and worst, WHILE THAT EVENT IS OPEN NO NEW ONE CAN BE. A car that pulls
   into the drive arrives as a count change on a stale event rather than as a
   new event, so the arrival — the thing worth telling someone about — is the
   least visible thing on the screen.

So this is not noise reduction. Point 3 is a missed detection.

WHAT COUNTS AS MOTION
---------------------
Displacement is measured from an ANCHOR — the position where the track was
last judged to be moving — not between consecutive frames. This is the whole
trick. Frame-to-frame jitter is bounded and never accumulates, so box wobble
cannot add up to "moving" no matter how long you watch; genuine motion, even a
slow walk, accumulates against a fixed point and crosses the threshold in a
second or two. Comparing consecutive frames instead would need a threshold
below one frame of walking and above one frame of wobble, and on a distant
subject there is no such number.

Two things count, because either alone misses a real case:

- the box CENTRE moving, which is lateral movement across the frame; and
- the box's SIZE changing, which is movement toward or away from the camera.
  Someone walking straight down a driveway at the lens barely moves their
  centre while their box doubles. Centre-only would call that parked.

Both are measured as a fraction of the box's own diagonal, so the threshold
scales with distance automatically: a subject at 30 m has a smaller box and
smaller absolute jitter, and needs to be held to a proportionally smaller
number rather than to the one that suited a subject on the doorstep.

THE TWO STATES, AND WHY "NEVER MOVED" IS ITS OWN THING
------------------------------------------------------
`active` is the only state the event layer cares about. A track is active when
it has moved at least once AND is not dormant.

- **Never moved.** A track that has not moved since it was confirmed is
  FURNITURE: a parked car, a bin, a statue, or a car that was already parked
  when the backend restarted. It is never active, so it opens no event, sends
  no notification, and costs no recognition pass. It is still TRACKED, which is
  what lets it become news the instant it moves.
- **Moved, then stopped.** A real subject that arrived and settled. Its event
  is already open and stays open — a person standing at a door is exactly the
  sighting worth keeping — but it stops heartbeating, and after
  `stationary_after_s` it goes dormant and stops sustaining the event at all.
  Otherwise a car that pulls in and parks would hold its event open until it
  drove away, which is the original bug reached by a different route.

Dormancy is not a terminal state. Moving again clears it, and because the
label had gone absent in the meantime, moving again opens a NEW event — which
is the correct reading of a parked car pulling out.

NOTHING HERE ENDS UP IN FOOTAGE
-------------------------------
Recording is continuous (recorder.py: 10 s segments kept for
`continuous_days`), and event clips are cut from it afterwards. So a suppressed
or shortened event costs a shorter clip and never costs footage: whatever the
camera saw is still on disk to scrub back to. That is what makes it safe to be
this aggressive about what deserves an event.

This module is pure arithmetic over boxes — no models, no I/O, no clock of its
own (callers pass `frame_time`). It is testable offline and cannot fail a frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Tunables. Empirical; gathered here rather than hunted through the code.
# ---------------------------------------------------------------------------

#: Movement threshold as a fraction of the box's own diagonal.
#:
#: The floor is set by tracker/box wobble on a genuinely static object, which
#: engine.py's reject-suppression measured at "a few → ~20 px" on a 704-wide
#: frame. Against a person-sized box (~215 px diagonal) that is ~9%, so 12%
#: clears it with margin while still tripping on a subject who shifts their
#: weight. It does NOT have to catch motion in one frame — displacement
#: accumulates from the anchor, so anything genuinely moving crosses it within
#: a frame or two.
MOVE_FRACTION = 0.12

#: An absolute floor in detect-stream pixels, for boxes small enough that a
#: fraction of their diagonal is under the noise. A 20x40 box has a 45 px
#: diagonal, and 12% of that is 5 px — inside the wobble of a distant
#: detection. Genuine motion still clears this easily, because it accumulates.
MIN_MOVE_PX = 6.0

#: How long a track that HAS moved may sit still before it stops sustaining its
#: event. Three minutes: long enough that someone waiting at a door, or a
#: delivery driver filling in a form, stays one event rather than being chopped
#: into several, and short enough that a car that parks releases its label
#: within a few minutes so the next arrival gets its own event.
STATIONARY_AFTER_S = 180.0


@dataclass
class TrackState:
    """One track's motion history. Only `active` leaves this module."""

    #: Box centre when this track was last judged to be moving.
    anchor: tuple[float, float]
    #: Box diagonal at that moment — the scale every threshold is taken
    #: against, and what makes "the box grew" measurable.
    anchor_diag: float
    #: Has this track EVER moved since it was first seen? False means
    #: furniture; see the module docstring.
    ever_moved: bool = False
    #: When it stopped moving, or None while it is moving.
    still_since: Optional[float] = None
    #: Last frame this track was offered, for pruning.
    last_seen: float = 0.0

    def is_still(self) -> bool:
        return self.still_since is not None


@dataclass
class Stillness:
    """Per-camera motion state for every live track.

    One instance per camera. Keyed by tracker_id, which the engine already
    guarantees is stable for the life of a track.
    """

    stationary_after_s: float = STATIONARY_AFTER_S
    move_fraction: float = MOVE_FRACTION
    min_move_px: float = MIN_MOVE_PX
    tracks: dict[int, TrackState] = field(default_factory=dict)
    #: How many observations this camera has dropped as furniture or dormant,
    #: for /api/recognition-style status and for answering "why is my parked
    #: car not showing up" without a debugger.
    dropped_never_moved: int = 0
    dropped_dormant: int = 0

    # ---------- per-frame ----------

    def update(self, tracker_id: int, box: tuple[float, float, float, float],
               frame_time: float) -> TrackState:
        """Fold one observation in and return the track's state.

        Cheap enough to call for every confirmed observation on every frame:
        two subtractions, a hypot and a comparison.
        """
        cx, cy, diag = _centre_and_diag(box)
        st = self.tracks.get(tracker_id)
        if st is None:
            # A brand-new track starts NOT moved. It has to earn that, which is
            # precisely how something already parked when we started looking
            # never becomes an event.
            st = TrackState(anchor=(cx, cy), anchor_diag=diag, still_since=frame_time)
            self.tracks[tracker_id] = st

        st.last_seen = frame_time
        if self._moved(st, cx, cy, diag):
            st.anchor = (cx, cy)
            st.anchor_diag = diag
            st.ever_moved = True
            st.still_since = None
        elif st.still_since is None:
            st.still_since = frame_time
        return st

    def _moved(self, st: TrackState, cx: float, cy: float, diag: float) -> bool:
        """Has this track moved far enough from its anchor to count?

        Measured against the ANCHOR's diagonal rather than the current one, so
        a box that is growing is judged by the scale it started at — otherwise
        an approaching subject raises its own threshold as it approaches and
        can outrun it.
        """
        scale = max(st.anchor_diag, diag)
        threshold = max(self.move_fraction * scale, self.min_move_px)
        if math.hypot(cx - st.anchor[0], cy - st.anchor[1]) >= threshold:
            return True
        # Size change: motion straight at or away from the lens, where the
        # centre can be almost perfectly still.
        return abs(diag - st.anchor_diag) >= self.move_fraction * scale

    # ---------- reading ----------

    def is_active(self, tracker_id: int, frame_time: float) -> bool:
        """Is this track worth an event right now?

        Unknown tracks read as active. This is called from the frame loop and
        a missing entry means `update` was not reached for it — the honest
        answer there is "no opinion", and for a security system no opinion must
        mean DETECT, never suppress.
        """
        st = self.tracks.get(tracker_id)
        if st is None:
            return True
        if not st.ever_moved:
            return False
        if st.still_since is None:
            return True
        return frame_time - st.still_since < self.stationary_after_s

    def all_still(self, tracker_ids: Iterable[int]) -> bool:
        """True when every one of these tracks is currently motionless.

        Used to hold back the event heartbeat: an open event whose subjects are
        all standing still has nothing new to say every 10 seconds. A score
        improvement, a count change or a line crossing still emits, because
        those are new information regardless of who is moving.
        """
        seen = False
        for tid in tracker_ids:
            st = self.tracks.get(tid)
            if st is None:
                return False
            if not st.is_still():
                return False
            seen = True
        return seen

    def count_drop(self, tracker_id: int) -> None:
        """Record WHY an observation was held back, for the status endpoint."""
        st = self.tracks.get(tracker_id)
        if st is not None and not st.ever_moved:
            self.dropped_never_moved += 1
        else:
            self.dropped_dormant += 1

    # ---------- housekeeping ----------

    def forget(self, tracker_ids: Iterable[int]) -> None:
        for tid in tracker_ids:
            self.tracks.pop(tid, None)

    def retain(self, live: Iterable[int]) -> None:
        """Drop every track not in `live`. A safety net against leaking a dict
        entry per passing car on a road-facing camera, in case a caller ever
        forgets to `forget`."""
        keep = set(live)
        for tid in [t for t in self.tracks if t not in keep]:
            del self.tracks[tid]

    def status(self) -> dict[str, object]:
        return {
            "tracked": len(self.tracks),
            "still": sum(1 for st in self.tracks.values() if st.is_still()),
            "never_moved": sum(1 for st in self.tracks.values() if not st.ever_moved),
            "dropped_never_moved": self.dropped_never_moved,
            "dropped_dormant": self.dropped_dormant,
        }


def _centre_and_diag(
    box: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, math.hypot(w, h)


def clamp_stationary_after(value: object) -> float:
    """`settings.detection.stationary_after_s`, clamped, with every bad shape
    falling back.

    Read on the frame loop, so a missing key, None, a string or NaN must all
    degrade to the shipped default rather than raise — an exception here stops
    detection on every camera.

    The floor is deliberately not zero: a few seconds would make a subject who
    pauses mid-driveway dormant and chop them into two events. The ceiling is
    an hour, past which this stops being a feature and becomes "off".
    """
    try:
        number = float(STATIONARY_AFTER_S if value is None else value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return STATIONARY_AFTER_S
    if number != number:  # NaN compares false against both bounds
        return STATIONARY_AFTER_S
    return max(10.0, min(3600.0, number))
