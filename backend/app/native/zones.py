"""Include zones, crossing lines and track traces — the *positive* half of
per-camera geometry.

WHY THIS EXISTS ALONGSIDE THE EXEMPT ZONES IN engine.py
=======================================================
The engine already carries per-camera polygons, but they are exempt-only:
places to IGNORE. This module adds the other three shapes an NVR wants, all
built on supervision primitives:

* INCLUDE ZONES (``sv.PolygonZone``) — "only alert on the driveway, not the
  street." Where an exempt zone carves a hole out of the watched area, an
  include zone replaces it: configure one and everything OUTSIDE it stops
  producing events. This is the sharper answer to street traffic, because the
  detection is dropped before an event ever opens.

* CROSSING LINES (``sv.LineZone``) — "someone crossed the property line", with
  in/out counts. A crossing is a much more specific claim than "a person
  appeared in frame".

* TRACES — the ground path an object walked, so an event snapshot shows where
  someone came from, not just where they stood at one instant.

THE EXEMPT LOGIC IS NOT TOUCHED. It stays in engine.py with its three
deliberately-asymmetric rules (foot-center, foot-line, whole-box containment),
tuned so a real foreground subject is never masked by a zone drawn higher in
the frame. Include zones use ONE rule instead — see below — because the two
have opposite failure directions and should not share a threshold.

THE INCLUDE RULE: FOOT-CENTER, AND ONLY FOOT-CENTER
---------------------------------------------------
An object is "in" an include zone when the midpoint of its box's bottom edge
lies inside the polygon — where a person's feet or a vehicle's tyres meet the
ground. That is ``sv.Position.BOTTOM_CENTER``, which is exactly the foot-center
the exempt rules already use, so both kinds of zone mean the same thing by
"where the object is" and an operator only has to learn it once.

It is also the only rule that is safe to state in one sentence in the UI, and a
filter that silently drops detections MUST be explainable — if someone cannot
predict what an include zone will ignore, they will not trust the camera.

ORDER OF OPERATIONS (engine.process)
------------------------------------
    include zones -> exempt zones -> reject-suppression -> tracking

Include first, exempt second, so exempt zones read as HOLES punched in the
included area: "watch the driveway, but not the bit of pavement inside it."
That composition is what people expect from drawing the shapes in that order.

COORDINATES. Zones and lines are stored NORMALIZED (0..1) on the camera row so
they survive a resolution change, and are converted to detect-stream pixels
ONCE per camera-row change (see engine.reload) — never per frame.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import numpy as np
import supervision as sv
from supervision.geometry.core import Point, Position

from .coco_labels import LABEL_TO_ID

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Observation

log = logging.getLogger(__name__)

# The anchor that decides whether a box is inside an include zone / which side
# of a crossing line it is on. BOTTOM_CENTER is the foot-center the exempt-zone
# rules already use (engine.box_foot_center) — one meaning of "where the object
# is" across every shape the operator can draw.
TRIGGER_ANCHOR = Position.BOTTOM_CENTER

# Frames an object must be seen on the far side of a line before the crossing
# counts. 1 would fire on a single frame of box wobble; 2 costs ~0.4 s at the
# default 5 fps and rejects a box that merely jitters across the line. Higher
# would start missing someone who crosses fast near the frame edge.
LINE_MIN_CROSSING_FRAMES = 2

# A line shorter than this (detect-space px) is dropped rather than built:
# sv.LineZone raises ValueError on a zero-magnitude vector, and a 2 px "line"
# is a mis-click, not a property boundary.
MIN_LINE_PX = 4.0

# Points kept per tracked object for its trace. 40 points at 5 fps is the last
# ~8 seconds of walking — enough to show where someone came from, short enough
# that the path does not wrap the whole frame in spaghetti. Bounded per track,
# and tracks are forgotten with the engine's hit counters.
MAX_TRACE_POINTS = 40


@dataclass(frozen=True)
class Crossing:
    """One object crossing one line on one frame.

    ``direction`` is "in" when the object crossed to the LEFT-HAND side of the
    start->end arrow (supervision's own convention, verified against
    sv.LineZone), and "out" for the other way. Neither is inherently the alarm
    direction — the operator decides that by which way they draw the line, and
    the UI draws the arrow so they can see which is which.
    """

    line: str
    direction: str  # "in" | "out"
    label: str
    tracker_id: int


# --------------------------------------------------------------------------
# parsing camera-row geometry -> supervision objects (once per row change)
# --------------------------------------------------------------------------


def _detect_dims(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row.get("detect_width") or 0.0), float(row.get("detect_height") or 0.0))


def _name_of(item: Any, index: int, prefix: str) -> str:
    if isinstance(item, dict):
        name = str(item.get("name") or "").strip()
        if name:
            return name
    return f"{prefix}#{index}"


def _points_of(item: Any) -> Optional[Sequence[Any]]:
    """The point list from ``{"points": [[x, y], ...]}`` or a bare
    ``[[x, y], ...]`` — the same two shapes the exempt zones accept."""
    if isinstance(item, dict):
        return item.get("points")
    if isinstance(item, (list, tuple)):
        return item
    return None


def include_detect_zones(row: dict[str, Any]) -> list[tuple[str, sv.PolygonZone]]:
    """``(name, PolygonZone)`` per stored include zone, in detect-stream pixels.

    Malformed zones are SKIPPED, never raised on: a bad polygon in the database
    must not stop a camera from detecting. A zone with fewer than 3 points is
    not a polygon and is dropped the same way the exempt parser drops it.
    """
    dw, dh = _detect_dims(row)
    out: list[tuple[str, sv.PolygonZone]] = []
    if dw <= 0 or dh <= 0:
        return out
    for i, zone in enumerate(row.get("include_zones") or []):
        pts = _points_of(zone)
        if not pts or len(pts) < 3:
            continue
        try:
            poly = np.array(
                [[float(p[0]) * dw, float(p[1]) * dh] for p in pts], dtype=np.int64
            )
        except (TypeError, ValueError, IndexError):
            continue
        try:
            out.append(
                (
                    _name_of(zone, i, "include"),
                    sv.PolygonZone(polygon=poly, triggering_anchors=(TRIGGER_ANCHOR,)),
                )
            )
        except Exception:  # noqa: BLE001 — a bad polygon is data, not a crash
            log.warning("skipping malformed include zone %d on %s", i, row.get("name"))
    return out


def cross_detect_lines(row: dict[str, Any]) -> list[tuple[str, sv.LineZone]]:
    """``(name, LineZone)`` per stored crossing line, in detect-stream pixels.

    Same tolerance as the zone parser: anything malformed or degenerate is
    skipped with a log rather than raised.
    """
    dw, dh = _detect_dims(row)
    out: list[tuple[str, sv.LineZone]] = []
    if dw <= 0 or dh <= 0:
        return out
    for i, line in enumerate(row.get("cross_lines") or []):
        if not isinstance(line, dict):
            continue
        try:
            sx, sy = (float(v) for v in line["start"])
            ex, ey = (float(v) for v in line["end"])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        start = Point(x=sx * dw, y=sy * dh)
        end = Point(x=ex * dw, y=ey * dh)
        if float(np.hypot(end.x - start.x, end.y - start.y)) < MIN_LINE_PX:
            log.warning("skipping zero-length crossing line %d on %s", i, row.get("name"))
            continue
        try:
            out.append(
                (
                    _name_of(line, i, "line"),
                    sv.LineZone(
                        start=start,
                        end=end,
                        triggering_anchors=(TRIGGER_ANCHOR,),
                        minimum_crossing_threshold=LINE_MIN_CROSSING_FRAMES,
                    ),
                )
            )
        except Exception:  # noqa: BLE001
            log.warning("skipping malformed crossing line %d on %s", i, row.get("name"))
    return out


def line_in_normal(
    start: Sequence[float], end: Sequence[float]
) -> tuple[float, float]:
    """Unit vector pointing to the side a crossing counts as "in".

    supervision counts a crossing to the LEFT of the start->end arrow as "in"
    (verified empirically against sv.LineZone, not assumed from the source).
    In screen coordinates — y increasing downward — the left-hand normal of
    ``(dx, dy)`` is ``(dy, -dx)``: for a line drawn left-to-right that points
    UP the frame, which is the direction sv.LineZone reports as "in".

    The UI draws this arrow on the line so the operator can see which way is
    which before they name it. Returns (0, 0) for a degenerate line.
    """
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    mag = float(np.hypot(dx, dy))
    if mag == 0.0:
        return (0.0, 0.0)
    return (dy / mag, -dx / mag)


# --------------------------------------------------------------------------
# per-frame use
# --------------------------------------------------------------------------


def detections_from(observations: Sequence["Observation"]) -> sv.Detections:
    """Engine observations -> ``sv.Detections`` for zone/line triggering.

    Carries tracker_id and class_id because sv.LineZone keys its crossing
    history on both; an unknown label maps to -1, which is a valid dictionary
    key for that history and never collides with a real COCO class.
    """
    if not observations:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([o.box for o in observations], dtype=np.float32),
        confidence=np.array([o.score for o in observations], dtype=np.float32),
        class_id=np.array([LABEL_TO_ID.get(o.label, -1) for o in observations], dtype=int),
        tracker_id=np.array([o.tracker_id for o in observations], dtype=int),
    )


def zone_hits(
    observations: Sequence["Observation"],
    zones: Sequence[tuple[str, sv.PolygonZone]],
) -> list[list[str]]:
    """Names of the include zones each observation is standing in.

    Returns one list per observation, aligned by index. An empty inner list
    means that observation is in NO include zone — which, when the camera has
    any include zones at all, is what makes it get dropped.
    """
    if not observations or not zones:
        return [[] for _ in observations]
    detections = detections_from(observations)
    hits: list[list[str]] = [[] for _ in observations]
    for name, zone in zones:
        try:
            mask = zone.trigger(detections)
        except Exception:  # noqa: BLE001 — a zone must never break detection
            log.exception("include zone %s failed to trigger", name)
            continue
        for i, inside in enumerate(mask):
            if inside:
                hits[i].append(name)
    return hits


def crossings(
    observations: Sequence["Observation"],
    lines: Sequence[tuple[str, sv.LineZone]],
) -> list[Crossing]:
    """Line crossings completed on THIS frame.

    Every line is triggered on every frame even when nothing crosses — that is
    not optional. sv.LineZone decides a crossing by comparing the side an
    object is on now against the side it was on in previous frames, so skipping
    frames would lose the history that makes a crossing detectable at all.
    """
    if not lines:
        return []
    detections = detections_from(observations)
    out: list[Crossing] = []
    for name, line in lines:
        try:
            crossed_in, crossed_out = line.trigger(detections)
        except Exception:  # noqa: BLE001 — a line must never break detection
            log.exception("crossing line %s failed to trigger", name)
            continue
        for i, obs in enumerate(observations):
            if i < len(crossed_in) and crossed_in[i]:
                out.append(Crossing(name, "in", obs.label, obs.tracker_id))
            elif i < len(crossed_out) and crossed_out[i]:
                out.append(Crossing(name, "out", obs.label, obs.tracker_id))
    return out


def forget_tracks(
    lines: Sequence[tuple[str, sv.LineZone]], keep: Iterable[int]
) -> None:
    """Drop crossing history for tracker_ids the engine has forgotten.

    sv.LineZone keeps ``crossing_state_history`` in a defaultdict it never
    prunes. On a laptop demo that is nothing; on an NVR that runs for months
    and mints a new tracker_id for every passing car, it is an unbounded leak.
    The engine already forgets a track's hit counter after _TRACK_FORGET_S, so
    this is called with the ids it still knows about and everything else goes.
    """
    keep_set = set(keep)
    for _name, line in lines:
        history = getattr(line, "crossing_state_history", None)
        if not isinstance(history, dict):
            continue
        for key in [k for k in history if k[0] not in keep_set]:
            del history[key]


def push_trace(
    traces: dict[int, deque], tracker_id: int, box: tuple[float, float, float, float]
) -> None:
    """Append an object's current ground position to its trace."""
    path = traces.get(tracker_id)
    if path is None:
        path = deque(maxlen=MAX_TRACE_POINTS)
        traces[tracker_id] = path
    x1, _y1, x2, y2 = box
    path.append(((x1 + x2) / 2.0, y2))


def line_geometry(lines: Sequence[tuple[str, sv.LineZone]]) -> list[dict[str, Any]]:
    """Detect-space endpoints per line, for drawing on an event snapshot.

    Read off the built LineZone objects rather than re-scaling the row, so what
    gets drawn is exactly the geometry that did the counting.
    """
    return [
        {
            "name": name,
            "start": [float(line.vector.start.x), float(line.vector.start.y)],
            "end": [float(line.vector.end.x), float(line.vector.end.y)],
        }
        for name, line in lines
    ]


def zone_geometry(
    zones: Sequence[tuple[str, sv.PolygonZone]],
) -> list[dict[str, Any]]:
    """Detect-space polygons per include zone, for drawing on a snapshot."""
    return [
        {"name": name, "points": [[float(x), float(y)] for x, y in zone.polygon]}
        for name, zone in zones
    ]
