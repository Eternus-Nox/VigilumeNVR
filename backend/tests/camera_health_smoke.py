"""CameraHealthTracker: debounce, boot suppression, interval recording, alerts.

Uses a REAL in-memory SQLite Database (so the schema/migration + the
record/query methods are exercised end to end) plus a fake push service that
records payloads. Deterministic clock so intervals are exact.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.camera_health import CameraHealthTracker  # noqa: E402
from app.db import Database  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {msg}")
    else:
        print(f"FAIL: {msg}")
        raise SystemExit(1)


class FakePush:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def send_to_all(self, payload: dict) -> None:
        self.payloads.append(payload)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


async def _db() -> Database:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(Path(path))
    await db.connect()
    return db


async def main() -> None:
    db = await _db()
    push = FakePush()
    clock = Clock()
    enabled = {"on": True}
    tracker = CameraHealthTracker(
        db, push, alerts_enabled=lambda: enabled["on"], clock=clock)
    await tracker.start()

    # ---- boot suppression: first two cycles never alert ----
    # cam "a" reachable, cam "b" NOT — but b needs debounce anyway, and even a
    # confirmed down inside the boot window must not alert.
    await tracker.observe([("a", True), ("b", False)])   # cycle 1
    clock.advance(45)
    await tracker.observe([("a", True), ("b", False)])   # cycle 2 -> b hits _FAILS_TO_DOWN
    check(push.payloads == [],
          "no alert during the boot-suppression window even when a camera is down")

    # b should now be recorded DOWN in history (suppression is alerts-only).
    ivs = await db.camera_health_intervals(0.0, clock() + 1)
    b_down = [i for i in ivs if i["camera"] == "b" and not i["online"]]
    check(len(b_down) == 1, "history IS written during boot suppression (b down interval exists)")

    # ---- past boot suppression: a real down->up->down for cam 'a' ----
    clock.advance(45)
    await tracker.observe([("a", False), ("b", False)])  # cycle 3: a fails once
    check(push.payloads == [], "one failed poll does NOT alert (debounce)")

    clock.advance(45)
    await tracker.observe([("a", False), ("b", False)])  # cycle 4: a fails twice -> DOWN
    downs = [p for p in push.payloads if p["camera"] == "a"]
    check(len(downs) == 1 and downs[0]["type"] == "camera_down",
          "a alerts DOWN only after _FAILS_TO_DOWN consecutive failures")

    # ---- recovery is immediate, and re-down alerts again ----
    clock.advance(45)
    await tracker.observe([("a", True), ("b", False)])   # a recovers in one good poll
    clock.advance(45)
    await tracker.observe([("a", False), ("b", False)])  # a fails once
    clock.advance(45)
    await tracker.observe([("a", False), ("b", False)])  # a fails twice -> DOWN again
    check(len([p for p in push.payloads if p["camera"] == "a"]) == 2,
          "a alerts again on a second genuine down (recovery reset the counter)")

    # ---- the toggle gates alerts ----
    enabled["on"] = False
    clock.advance(45)
    await tracker.observe([("a", True), ("b", False)])   # recover (no alert on up)
    clock.advance(45)
    await tracker.observe([("a", False), ("b", False)])
    clock.advance(45)
    before = len(push.payloads)
    await tracker.observe([("a", False), ("b", False)])  # would be a down
    check(len(push.payloads) == before,
          "no alert fires while the camera_down_alerts toggle is off")
    enabled["on"] = True

    # ---- uptime math: exact from the recorded intervals ----
    # Fresh tracker + camera to compute a clean uptime over a known window.
    push2 = FakePush()
    clock2 = Clock()
    t2 = CameraHealthTracker(db, push2, alerts_enabled=lambda: True, clock=clock2)
    await t2.start()
    # up for 90s, then down for 90s (needs 2 fails), then up.
    await t2.observe([("c", True)])            # t0: up
    clock2.advance(90)
    await t2.observe([("c", False)])           # t90: fail 1 (still up)
    clock2.advance(45)
    await t2.observe([("c", False)])           # t135: fail 2 -> down recorded at t135
    clock2.advance(90)
    await t2.observe([("c", True)])            # t225: up
    start = 1_000_000.0
    ivs_c = await db.camera_health_intervals(start, clock2())
    ivs_c = [i for i in ivs_c if i["camera"] == "c"]
    up = sum(i["end"] - i["start"] for i in ivs_c if i["online"])
    down = sum(i["end"] - i["start"] for i in ivs_c if not i["online"])
    # up: t0..t135 (135) + t225..now (0) ; down: t135..t225 (90)
    check(abs(up - 135.0) < 0.01, f"uptime interval sums correctly (up={up})")
    check(abs(down - 90.0) < 0.01, f"downtime interval sums correctly (down={down})")

    # ---- resume across a "restart": a new tracker loads the open state ----
    t3 = CameraHealthTracker(db, FakePush(), alerts_enabled=lambda: True, clock=clock2)
    await t3.start()
    # c's last state was up; a same-state poll must NOT open a duplicate interval.
    n_before = len(await db.camera_health_intervals(start, clock2() + 1))
    await t3.observe([("c", True)])
    n_after = len(await db.camera_health_intervals(start, clock2() + 1))
    check(n_after == n_before,
          "a restarted tracker resumes open state and writes no duplicate interval")

    # ---- clipping: a 10 s window is fully covered by c's history (the tail of
    # its down interval, which ended exactly at now), so the clipped total is 10.
    clip = await db.camera_health_intervals(clock2() - 10, clock2())
    total_clip = sum(i["end"] - i["start"] for i in clip if i["camera"] == "c")
    check(abs(total_clip - 10.0) < 0.01, "intervals clip to the query window")

    await db.close()
    print(f"\nAll {PASS} camera-health checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
