"""Where a tracked object was at any moment, from its sightings.

A track's PATH is its sightings on the detect stream: ``(wall time, box)``
with the box as FRACTIONS of the frame, so it maps onto any resolution of the
same view. Reading the recording needs the object's box at times nobody
detected it — between detect frames (the recording runs at the camera's full
frame rate, detection at ~5 fps), and BEFORE it was first detected, which is
the whole point of looking back: a car is often legible in the second before
detection has confirmed it, and gone a second later.

So: linear interpolation between sightings, and linear extrapolation beyond
the ends from the object's recent velocity, clamped to MAX_EXTRAPOLATE_S —
past that a guess about where it was is no longer worth trusting, and the box
simply stays where the extrapolation stopped.

Pure functions, no I/O: everything here is unit-testable without a camera.
"""
from __future__ import annotations

from typing import Sequence

Box = tuple[float, float, float, float]
Path = Sequence[tuple[float, Box]]

#: Furthest beyond the first/last sighting a box is extrapolated.
MAX_EXTRAPOLATE_S = 1.0

#: Span of sightings the extrapolation velocity is measured over. Short enough
#: to follow a turn, long enough that one jittery detector box does not fling
#: the estimate.
VELOCITY_SPAN_S = 0.6

#: Step at which a box is sampled across a window (see `union_over`).
SAMPLE_S = 0.05


def _lerp(a: Box, b: Box, f: float) -> Box:
    return tuple(av + (bv - av) * f for av, bv in zip(a, b))  # type: ignore[return-value]


def _velocity(path: Path, from_end: bool) -> Box:
    """Per-second change of each box coordinate near one end of the path."""
    if len(path) < 2:
        return (0.0, 0.0, 0.0, 0.0)
    if from_end:
        t1, b1 = path[-1]
        k = len(path) - 2
        while k > 0 and t1 - path[k][0] < VELOCITY_SPAN_S:
            k -= 1
        t0, b0 = path[k]
    else:
        t0, b0 = path[0]
        k = 1
        while k < len(path) - 1 and path[k][0] - t0 < VELOCITY_SPAN_S:
            k += 1
        t1, b1 = path[k]
    dt = t1 - t0
    if dt <= 1e-6:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple((v1 - v0) / dt for v0, v1 in zip(b0, b1))  # type: ignore[return-value]


def box_at(path: Path, t: float) -> Box:
    """The object's box at time `t` (fractions of the frame)."""
    if not path:
        raise ValueError("empty path")
    t_first, b_first = path[0]
    t_last, b_last = path[-1]
    if t <= t_first:
        dt = max(-MAX_EXTRAPOLATE_S, t - t_first)
        v = _velocity(path, from_end=False)
        return clamp(tuple(c + vc * dt for c, vc in zip(b_first, v)))  # type: ignore[arg-type]
    if t >= t_last:
        dt = min(MAX_EXTRAPOLATE_S, t - t_last)
        v = _velocity(path, from_end=True)
        return clamp(tuple(c + vc * dt for c, vc in zip(b_last, v)))  # type: ignore[arg-type]
    lo, hi = 0, len(path) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if path[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, ba), (tb, bb) = path[lo], path[hi]
    return _lerp(ba, bb, (t - ta) / (tb - ta) if tb > ta else 0.0)


def union(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def union_over(path: Path, start: float, end: float) -> Box:
    """Every place the object was (or is extrapolated to have been) in
    [start, end] — the area a window of frames has to cover."""
    if end < start:
        start, end = end, start
    n = max(1, int((end - start) / SAMPLE_S))
    boxes = [box_at(path, start + (end - start) * i / n) for i in range(n + 1)]
    # The sightings themselves, so a sharp turn between samples is not cut.
    boxes += [b for t, b in path if start <= t <= end]
    return union(boxes)


def box_near(path: Path, t: float, tolerance: float) -> Box:
    """Where the object was at `t`, give or take `tolerance` seconds — for
    matching a recorded frame whose time is known only to that accuracy."""
    return union_over(path, t - tolerance, t + tolerance)


def expand(box: Box, frac: float) -> Box:
    """`box` grown by `frac` of its size on every side, clamped to the frame."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return clamp((x1 - dx, y1 - dy, x2 + dx, y2 + dy))


def clamp(box: Box) -> Box:
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in box)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def to_pixels(box: Box, width: int, height: int, ox: int = 0, oy: int = 0) -> Box:
    """A fractional box in the pixels of a crop taken at (ox, oy) from a
    `width` x `height` frame."""
    return (box[0] * width - ox, box[1] * height - oy,
            box[2] * width - ox, box[3] * height - oy)


def contains(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]
