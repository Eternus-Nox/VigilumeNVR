"""Where recognition actually works on each camera.

THE QUESTION THIS ANSWERS
=========================
The ROI editor asks the operator to draw a region where faces (or plates) are
worth spending a recognition pass on. Left to intuition, almost everyone draws
the region where people WALK — which is the obvious guess and frequently the
wrong one. Whether a face is legible depends on range, lens, mounting height
and where the light is, and none of those are visible in a still frame. The
region that matters is where something READABLE has actually come from.

So each camera accumulates a coarse grid with two numbers per cell:

    count        how many times a face/plate was centred here
    quality_sum  the sum of their bestshot quality scores

`count` alone is a footfall map. `quality_sum / count` is the part that says
whether anything usable ever came out of that cell. Keeping both lets the
editor render DENSITY AS OPACITY and MEAN QUALITY AS HUE, so "busy but
unreadable" (the far pavement) and "quiet but sharp" (the doorstep) look
different at a glance — which is exactly the distinction the operator needs and
cannot otherwise see.

WHY A GRID AND NOT THE RAW POINTS
---------------------------------
Bounded cost. A grid is at most ``COLS * ROWS`` rows per (camera, kind) no
matter how long the camera runs, whereas keeping sightings would grow without
limit and then need its own retention policy. The grid is also already the
shape both renderers want.

It is deliberately COARSE (32x24). Finer would imply a precision the underlying
measurement does not have — a face's centre moves several cells between frames
— and would make the overlay noisy rather than informative.

THIS IS NOT BIOMETRIC DATA
--------------------------
A cell holds a count and a sum of quality scores. There is no embedding, no
crop, and nothing that identifies anyone — so unlike the candidate store it has
no retention window, and it survives `Clear unknown faces`. It is decayed
instead (see `decay`), so a camera that gets moved or re-aimed stops being
described by where faces used to be.

WRITE PATTERN
-------------
Accumulated IN MEMORY and flushed on the recognition maintenance tick. A face
pass runs a few times a second per track; a DB write per sighting would be
pointless churn for data whose whole purpose is to be looked at occasionally.
A flush that is lost to a crash costs a few minutes of counts on a map built
over weeks.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

#: Grid resolution. 32x24 is 4:3, matching the detect streams these cameras
#: run, and is coarse enough that one cell is a meaningful area rather than a
#: pixel-accurate claim.
COLS = 32
ROWS = 24

#: Multiplier applied to every cell on each decay pass, and how often. Together
#: these halve a cell's weight in about a month of continuous running: a camera
#: that is re-aimed, re-lensed or moved stops being described by where faces
#: used to appear, without ever discarding history abruptly.
DECAY_FACTOR = 0.98
DECAY_INTERVAL_S = 24 * 3600.0

#: Cells below this count after decay are deleted rather than kept at a
#: vanishing weight — it keeps the table small and the overlay clean.
MIN_CELL_COUNT = 0.5


def cell_index(nx: float, ny: float) -> Optional[int]:
    """Normalized (0..1) frame position -> flat cell index, or None if outside.

    Clamping would be wrong here: a coordinate outside the frame means the
    caller got its transform wrong (most likely by forgetting a crop's origin),
    and silently folding it onto the edge would paint a convincing stripe down
    the side of the heatmap instead of surfacing the bug.
    """
    if not (0.0 <= nx <= 1.0) or not (0.0 <= ny <= 1.0):
        return None
    col = min(COLS - 1, int(nx * COLS))
    row = min(ROWS - 1, int(ny * ROWS))
    return row * COLS + col


def cell_bounds(index: int) -> tuple[float, float, float, float]:
    """Flat cell index -> normalized (x1, y1, x2, y2)."""
    row, col = divmod(int(index), COLS)
    return (col / COLS, row / ROWS, (col + 1) / COLS, (row + 1) / ROWS)


class HeatmapAccumulator:
    """In-memory counts per (camera, kind, cell), flushed to SQLite in batches."""

    def __init__(self, db: Any) -> None:
        self._db = db
        # (camera, kind, cell) -> [count, quality_sum]
        self._pending: dict[tuple[str, str, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        self._last_decay = time.monotonic()

    # ---------- accumulate ----------

    def record(
        self, camera: str, kind: str, nx: float, ny: float, quality: float
    ) -> bool:
        """Note one sighting at a normalized frame position. True if counted."""
        idx = cell_index(nx, ny)
        if idx is None:
            log.debug(
                "heatmap: dropping %s sighting at (%.3f, %.3f) on %s — outside the frame",
                kind, nx, ny, camera,
            )
            return False
        slot = self._pending[(camera, kind, idx)]
        slot[0] += 1.0
        slot[1] += max(0.0, float(quality))
        return True

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ---------- persist ----------

    async def flush(self) -> int:
        """Write accumulated counts. Returns the number of cells touched."""
        if not self._pending:
            return 0
        batch, self._pending = self._pending, defaultdict(lambda: [0.0, 0.0])
        now = time.time()
        try:
            for (camera, kind, cell), (count, qsum) in batch.items():
                await self._db.conn.execute(
                    "INSERT INTO recognition_heatmap "
                    "(camera, kind, cell, count, quality_sum, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(camera, kind, cell) DO UPDATE SET "
                    "  count = count + excluded.count, "
                    "  quality_sum = quality_sum + excluded.quality_sum, "
                    "  updated_at = excluded.updated_at",
                    (camera, kind, cell, count, qsum, now),
                )
            await self._db.conn.commit()
            return len(batch)
        except Exception:
            # Put the batch back so a transient DB error costs nothing. Merged
            # rather than replaced: more sightings may have arrived meanwhile.
            for key, value in batch.items():
                slot = self._pending[key]
                slot[0] += value[0]
                slot[1] += value[1]
            log.exception("heatmap flush failed — %d cell(s) retained", len(batch))
            return 0

    async def maybe_decay(self) -> int:
        """Age the map down on a slow schedule. Returns cells removed."""
        if time.monotonic() - self._last_decay < DECAY_INTERVAL_S:
            return 0
        self._last_decay = time.monotonic()
        return await self.decay()

    async def decay(self) -> int:
        """Multiply every cell down and drop the ones that have faded out."""
        try:
            await self._db.conn.execute(
                "UPDATE recognition_heatmap SET count = count * ?, quality_sum = quality_sum * ?",
                (DECAY_FACTOR, DECAY_FACTOR),
            )
            cur = await self._db.conn.execute(
                "DELETE FROM recognition_heatmap WHERE count < ?", (MIN_CELL_COUNT,)
            )
            await self._db.conn.commit()
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        except Exception:
            log.exception("heatmap decay failed")
            return 0

    # ---------- read ----------

    async def grid(self, camera: str, kind: str) -> dict[str, Any]:
        """The stored map for one camera/kind, in the shape both UIs want.

        Two parallel flat arrays rather than a list of objects: 768 cells as
        ``[{x,y,count,quality}, ...]`` is ~40 KB of JSON for something both
        renderers immediately flatten anyway.

        `counts` are normalized to 0..1 against the busiest cell, because the
        absolute number is meaningless to a person — what they need is "where,
        relatively". `quality` is the MEAN in 0..1 and is already absolute.
        """
        empty = {
            "camera": camera, "kind": kind, "cols": COLS, "rows": ROWS,
            "counts": [], "quality": [], "samples": 0, "peak": 0.0,
            "updated_at": None,
        }
        try:
            rows = await (
                await self._db.conn.execute(
                    "SELECT cell, count, quality_sum, updated_at FROM recognition_heatmap "
                    "WHERE camera = ? AND kind = ?",
                    (camera, kind),
                )
            ).fetchall()
        except Exception:
            log.exception("heatmap read failed for %s/%s", camera, kind)
            return empty
        if not rows:
            return empty

        counts = [0.0] * (COLS * ROWS)
        quality = [0.0] * (COLS * ROWS)
        peak = 0.0
        total = 0.0
        updated = 0.0
        for r in rows:
            idx = int(r["cell"])
            if not 0 <= idx < COLS * ROWS:
                continue
            c = float(r["count"])
            counts[idx] = c
            quality[idx] = (float(r["quality_sum"]) / c) if c > 0 else 0.0
            peak = max(peak, c)
            total += c
            updated = max(updated, float(r["updated_at"] or 0.0))

        if peak > 0:
            counts = [round(c / peak, 4) for c in counts]
        return {
            "camera": camera, "kind": kind, "cols": COLS, "rows": ROWS,
            "counts": counts,
            "quality": [round(q, 4) for q in quality],
            "samples": int(round(total)),
            "peak": round(peak, 2),
            "updated_at": updated or None,
        }

    async def clear(self, camera: Optional[str] = None, kind: Optional[str] = None) -> None:
        """Forget the map — for a re-aimed camera, where history is now a lie."""
        sql = "DELETE FROM recognition_heatmap WHERE 1=1"
        params: list[Any] = []
        if camera:
            sql += " AND camera = ?"
            params.append(camera)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        try:
            await self._db.conn.execute(sql, params)
            await self._db.conn.commit()
        except Exception:
            log.exception("heatmap clear failed")
        # Drop anything buffered for the same scope, or the next flush would
        # immediately resurrect part of what was just cleared.
        for key in [
            k for k in self._pending
            if (camera is None or k[0] == camera) and (kind is None or k[1] == kind)
        ]:
            del self._pending[key]

    async def cameras(self) -> list[dict[str, Any]]:
        """Which (camera, kind) pairs have a map, for the editor's picker."""
        try:
            rows = await (
                await self._db.conn.execute(
                    "SELECT camera, kind, SUM(count) AS n, MAX(updated_at) AS t "
                    "FROM recognition_heatmap GROUP BY camera, kind"
                )
            ).fetchall()
        except Exception:
            log.exception("heatmap summary failed")
            return []
        return [
            {
                "camera": r["camera"], "kind": r["kind"],
                "samples": int(round(r["n"] or 0)), "updated_at": r["t"],
            }
            for r in rows
        ]


def suggest_zone(grid: dict[str, Any], *, min_quality: float = 0.45) -> list[list[float]]:
    """A starting polygon covering the cells worth recognizing in.

    A rectangle, not a tight hull: the bounding box of every cell that has both
    real evidence and a mean quality worth spending inference on. The operator
    is expected to drag it — the point is to put something defensible on screen
    so the editor opens with an answer instead of a blank frame.

    Returns [] when there is not enough evidence to suggest anything, which the
    UI shows as "watch for a while first" rather than a guess dressed up as a
    recommendation.
    """
    counts = grid.get("counts") or []
    quality = grid.get("quality") or []
    if not counts or len(counts) != len(quality):
        return []
    # A cell needs a tenth of the peak traffic to count as evidence at all;
    # below that one lucky sighting would define the region.
    hot = [
        i for i, c in enumerate(counts)
        if c >= 0.1 and quality[i] >= min_quality
    ]
    if len(hot) < 2:
        return []
    xs1, ys1, xs2, ys2 = [], [], [], []
    for i in hot:
        x1, y1, x2, y2 = cell_bounds(i)
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
