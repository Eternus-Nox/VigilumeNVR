#!/usr/bin/env python3
"""The recognition heatmap: accumulation, decay, and the zone suggestion.

The heatmap is what makes the ROI editor answerable instead of a guess, so the
things that would quietly ruin it get the attention:

1. THE COORDINATE TRANSLATION. A face is detected inside a PERSON CROP, so its
   box is in the crop's coordinates. Painting it without translating by the
   crop's origin does not fail — it produces a perfectly plausible map of the
   wrong places, which is worse than no map. A position outside 0..1 is
   therefore DROPPED rather than clamped, so the bug surfaces instead of
   painting a stripe down the edge.
2. COUNT vs QUALITY. Keeping both is the whole point: count alone is a footfall
   map, and footfall is exactly the wrong thing to draw a recognition zone
   around. The two must stay separable all the way to the API.
3. FLUSH DURABILITY. A failed flush must not silently discard counts.
4. THE SUGGESTION MUST ABSTAIN. With thin evidence it returns nothing rather
   than dressing a guess up as a recommendation.

Offline-runnable.
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.heatmap import (  # noqa: E402
    COLS,
    ROWS,
    HeatmapAccumulator,
    cell_bounds,
    cell_index,
    suggest_zone,
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


class FakeCursor:
    def __init__(self, rows, rowcount=0):
        self._rows = rows
        self.rowcount = rowcount

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeDB:
    def __init__(self, path, fail=False):
        self._c = sqlite3.connect(path)
        self._c.row_factory = sqlite3.Row
        self._c.executescript(
            """
            CREATE TABLE recognition_heatmap (
                camera TEXT NOT NULL, kind TEXT NOT NULL, cell INTEGER NOT NULL,
                count REAL NOT NULL DEFAULT 0, quality_sum REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (camera, kind, cell));
            """
        )
        self._c.commit()
        self.conn = self
        self.fail = fail

    async def execute(self, sql, params=()):
        if self.fail:
            raise RuntimeError("simulated database failure")
        cur = self._c.execute(sql, params)
        rows = cur.fetchall() if sql.strip().upper().startswith("SELECT") else []
        return FakeCursor(rows, cur.rowcount)

    async def commit(self):
        if self.fail:
            raise RuntimeError("simulated database failure")
        self._c.commit()

    def rows(self, sql, params=()):
        return self._c.execute(sql, params).fetchall()


def geometry_checks() -> None:
    print("\ncell geometry")
    check(cell_index(0.0, 0.0) == 0, "the top-left corner is cell 0")
    check(cell_index(0.999, 0.999) == COLS * ROWS - 1, "the bottom-right is the last cell")
    check(cell_index(1.0, 1.0) == COLS * ROWS - 1, "exactly 1.0 stays in range, not off the end")
    mid = cell_index(0.5, 0.5)
    check(mid == (ROWS // 2) * COLS + COLS // 2, "the centre lands in the middle cell")

    # OUT OF RANGE IS DROPPED, NOT CLAMPED — see the module header.
    check(cell_index(-0.01, 0.5) is None, "a negative x is refused")
    check(cell_index(0.5, 1.5) is None, "a y past the frame is refused")
    check(cell_index(1.001, 0.5) is None, "...even slightly past it")

    x1, y1, x2, y2 = cell_bounds(0)
    check((x1, y1) == (0.0, 0.0) and abs(x2 - 1 / COLS) < 1e-9,
          "cell_bounds(0) is the top-left cell in normalized coords")
    for i in (0, 5, COLS, COLS * ROWS - 1):
        bx1, by1, bx2, by2 = cell_bounds(i)
        # A point just inside the cell must map back to the same index.
        check(cell_index((bx1 + bx2) / 2, (by1 + by2) / 2) == i,
              f"cell {i} round-trips through bounds -> index")


async def accumulate_checks() -> None:
    print("\naccumulation and flush")
    tmp = Path(tempfile.mkdtemp())
    db = FakeDB(tmp / "h.db")
    acc = HeatmapAccumulator(db)

    check(acc.record("front", "face", 0.5, 0.5, 0.9) is True, "an in-frame sighting is counted")
    check(acc.record("front", "face", -0.2, 0.5, 0.9) is False,
          "an out-of-frame sighting is DROPPED, not clamped onto the edge")
    check(acc.pending == 1, "...and only the valid one is buffered")

    for _ in range(4):
        acc.record("front", "face", 0.5, 0.5, 0.8)
    acc.record("front", "face", 0.1, 0.1, 0.2)
    check(acc.pending == 2, "two distinct cells buffered")

    touched = await acc.flush()
    check(touched == 2, "flush writes both cells")
    check(acc.pending == 0, "...and empties the buffer")

    rows = db.rows("SELECT * FROM recognition_heatmap ORDER BY cell")
    check(len(rows) == 2, "two rows persisted")
    hot = [r for r in rows if r["count"] == 5][0]
    check(abs(hot["quality_sum"] - (0.9 + 0.8 * 4)) < 1e-6,
          "quality_sum accumulates alongside the count, not instead of it")

    # A second flush must ADD to the stored cell, not replace it.
    acc.record("front", "face", 0.5, 0.5, 1.0)
    await acc.flush()
    rows = db.rows("SELECT count FROM recognition_heatmap WHERE count > 1")
    check(rows[0]["count"] == 6, "a later flush accumulates onto the existing cell")

    print("\nflush durability")
    acc2 = HeatmapAccumulator(FakeDB(tmp / "h2.db", fail=True))
    acc2.record("front", "face", 0.5, 0.5, 0.9)
    check(await acc2.flush() == 0, "a failing flush reports nothing written")
    check(acc2.pending == 1, "...and RETAINS the counts rather than dropping them")
    acc2.record("front", "face", 0.5, 0.5, 0.9)
    check(acc2.pending == 1, "a retained cell merges with new sightings")


async def grid_checks() -> None:
    print("\nthe served grid")
    tmp = Path(tempfile.mkdtemp())
    db = FakeDB(tmp / "g.db")
    acc = HeatmapAccumulator(db)

    # A busy-but-unreadable region and a quiet-but-sharp one: the exact
    # distinction the heatmap exists to make visible.
    for _ in range(50):
        acc.record("front", "face", 0.9, 0.2, 0.15)   # far pavement
    # The doorstep spans a few cells, as a real one does — a region, not a
    # single lucky cell (a single cell is tested below and must abstain).
    for _ in range(5):
        for dx in (0.30, 0.34, 0.38):
            acc.record("front", "face", dx, 0.8, 0.85)
    await acc.flush()

    grid = await acc.grid("front", "face")
    check(grid["cols"] == COLS and grid["rows"] == ROWS, "grid reports its dimensions")
    check(len(grid["counts"]) == COLS * ROWS, "counts is a full flat grid")
    check(len(grid["quality"]) == COLS * ROWS, "quality is a full flat grid")
    check(grid["samples"] == 65, "samples is the real total")
    check(grid["peak"] == 50, "peak is the busiest cell's raw count")

    busy = cell_index(0.9, 0.2)
    quiet = cell_index(0.3, 0.8)
    check(grid["counts"][busy] == 1.0, "counts are normalized against the busiest cell")
    check(abs(grid["counts"][quiet] - 0.1) < 1e-6, "...so a doorstep cell reads as 0.1")
    check(abs(grid["quality"][busy] - 0.15) < 1e-3, "the busy cell's MEAN quality is low")
    check(abs(grid["quality"][quiet] - 0.85) < 1e-3, "the quiet cell's mean quality is high")
    check(
        grid["counts"][busy] > grid["counts"][quiet]
        and grid["quality"][busy] < grid["quality"][quiet],
        "density and legibility are INDEPENDENT — which is the whole point",
    )

    empty = await acc.grid("nosuchcam", "face")
    check(empty["counts"] == [] and empty["samples"] == 0,
          "a camera with no history returns an empty grid, not an error")

    print("\nzone suggestion")
    zone = suggest_zone(grid)
    check(len(zone) == 4, f"the suggestion is a 4-point rectangle (got {len(zone)})")
    xs = [p[0] for p in zone] or [0.0]
    ys = [p[1] for p in zone] or [0.0]
    check(min(xs) >= 0.0 and max(xs) <= 1.0 and min(ys) >= 0.0 and max(ys) <= 1.0,
          "...in normalized coordinates")
    # The busy cell is LOW quality, so it must not define the suggested region.
    bx1, by1, bx2, by2 = cell_bounds(busy)
    check(not (min(xs) <= bx1 and max(xs) >= bx2 and min(ys) <= by1 and max(ys) >= by2),
          "the busy-but-unreadable region is NOT included in the suggestion")

    check(suggest_zone({"counts": [], "quality": []}) == [],
          "no data suggests nothing")
    check(suggest_zone({"counts": [1.0] + [0.0] * 10, "quality": [0.9] + [0.0] * 10}) == [],
          "a single hot cell is not enough evidence to suggest a region")
    check(
        suggest_zone(await acc.grid("front", "face"), min_quality=0.99) == [],
        "an unreachable quality bar abstains rather than returning the whole frame",
    )


async def decay_checks() -> None:
    print("\ndecay and clearing")
    tmp = Path(tempfile.mkdtemp())
    db = FakeDB(tmp / "d.db")
    acc = HeatmapAccumulator(db)
    for _ in range(100):
        acc.record("front", "face", 0.5, 0.5, 0.8)
    acc.record("front", "face", 0.1, 0.1, 0.8)
    await acc.flush()

    before = db.rows("SELECT count, quality_sum FROM recognition_heatmap ORDER BY count")
    check(len(before) == 2, "both cells are stored to begin with")

    await acc.decay()
    mid = db.rows("SELECT count FROM recognition_heatmap ORDER BY count")
    check(len(mid) == 2,
          "ONE decay pass drops nothing — ageing is gradual, so a camera that is "
          "briefly quiet does not lose its map")
    check(mid[-1]["count"] < before[-1]["count"], "but every cell is scaled down")

    # Enough passes to take a single sighting under the floor (0.98^n < 0.5 at
    # n≈35). The established cell must still be there: that asymmetry is the
    # point — incidental cells fade out, real regions persist.
    for _ in range(40):
        await acc.decay()
    after = db.rows("SELECT count, quality_sum FROM recognition_heatmap ORDER BY count")
    check(len(after) == 1, "the single-sighting cell eventually fades out and is dropped")
    check(after[0]["count"] > 1.0, "...while the established cell is still well above the floor")
    check(
        abs(after[0]["quality_sum"] / after[0]["count"]
            - before[-1]["quality_sum"] / before[-1]["count"]) < 1e-6,
        "decay scales count and quality_sum together, so MEAN quality is preserved",
    )

    print("\nclearing")
    acc.record("front", "face", 0.5, 0.5, 0.8)
    acc.record("back", "face", 0.5, 0.5, 0.8)
    await acc.clear(camera="front")
    check(db.rows("SELECT * FROM recognition_heatmap WHERE camera='front'") == [],
          "clearing a camera removes its stored cells")
    await acc.flush()
    check(db.rows("SELECT * FROM recognition_heatmap WHERE camera='front'") == [],
          "...and the BUFFERED sightings for it too — otherwise the next flush "
          "would resurrect part of what was just cleared")
    check(len(db.rows("SELECT * FROM recognition_heatmap WHERE camera='back'")) == 1,
          "...while another camera's map is untouched")

    summary = await acc.cameras()
    check(any(s["camera"] == "back" for s in summary), "cameras() lists what has a map")


async def main() -> int:
    geometry_checks()
    await accumulate_checks()
    await grid_checks()
    await decay_checks()

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (recognition heatmap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
