#!/usr/bin/env python3
"""Full-resolution plate reading — against the REAL pinned OCR model.

The complaint this answers: faces were recognized, plates mostly were not.
The cause was resolution. The plate pass read from the DETECT frame, which is
the camera's substream scaled to 704x480; a plate there is usually 35-55 px
wide, and the plate scorer (rightly) will not read anything under 64. So a
plate was read only when a car was almost at the lens.

What this suite demonstrates, in the order it matters:

1. THE PROBLEM IS REAL. A driveway scene whose plate is perfectly legible at the
   camera's 2304x1296 main resolution yields NOTHING from the detect frame alone.
2. THE FIX READS IT. With a snapshot source serving the full-resolution frame,
   the same scene produces the right plate — as a stored recognition and an
   enrollable candidate — even when the car has moved between the detect frame
   and the snapshot, which in real life it always has.
3. THE LOCALIZER IS SCALE-INDEPENDENT. Full-resolution crops are 2-5x wider than
   the detect-frame crops the localizer's kernels were tuned on; before the fix
   a 180-280 px plate came back as a strip the scorer vetoed as "too wide" even
   though the OCR had read it correctly.
4. IT NEVER STALLS DETECTION. The snapshot is an HTTP round trip; the pass must
   not await it inside the frame loop, and a track ending while one is in
   flight must return at once and vote later.
5. IT IS A GOOD CITIZEN. One snapshot is shared by every vehicle on a camera,
   a failing camera is left alone after a few tries, a camera whose snapshot is
   no bigger than detection is not asked again, and a vehicle that sits there
   does not cost a snapshot a second forever.

NETWORK: the first run downloads the 3.3 MB OCR (set VIGILUME_TEST_MODELS_DIR
to reuse one). Offline, the suite SKIPS (exit 0) rather than failing.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.native import platepass as pp_mod  # noqa: E402
from app.native import platesnap  # noqa: E402
from app.native.bestshot import MIN_QUALITY, score_plate  # noqa: E402
from app.native.platepass import PlatePass  # noqa: E402
from app.native.plates import PlateReader, candidate_regions, deskew  # noqa: E402
from app.native.platesnap import SnapshotSource, locate  # noqa: E402
from plates_smoke import FakeDB, Obs, fake_vehicle  # noqa: E402

PASS = 0
_failures: list[str] = []

HI_W, HI_H = 2304, 1296
DET_W, DET_H = 704, 480
PLATE = "7ABC123"


def check(cond: bool, label: str) -> None:
    global PASS
    PASS += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------


def _background(seed: int = 3) -> np.ndarray:
    """A driveway-ish background with enough texture that template matching
    has to find the car rather than the only non-flat thing in the frame."""
    rng = np.random.default_rng(seed)
    bg = np.full((HI_H, HI_W, 3), 105, np.uint8)
    bg = cv2.add(bg, rng.integers(0, 18, (HI_H, HI_W, 3), dtype=np.uint8))
    for _ in range(25):
        x, y = int(rng.integers(0, HI_W - 200)), int(rng.integers(0, HI_H // 2))
        cv2.rectangle(bg, (x, y), (x + int(rng.integers(40, 200)), y + int(rng.integers(20, 120))),
                      tuple(int(v) for v in rng.integers(40, 200, 3)), -1)
    return bg


BACKGROUND = _background()


def scene(plate_w: int = 150, shift_x: int = 0, text: str = PLATE):
    """(full-resolution frame, car box in it). The car is ~4.4x its plate's width."""
    frame = BACKGROUND.copy()
    car_w = int(plate_w * 4.4)
    car_h = int(car_w * 0.7)
    car, _ = fake_vehicle(text=text, w=car_w, h=car_h, plate_w=plate_w)
    x0 = (HI_W - car_w) // 2 + shift_x
    y0 = HI_H - car_h - 40
    frame[y0:y0 + car_h, x0:x0 + car_w] = car
    return frame, (x0, y0, x0 + car_w, y0 + car_h)


def to_detect(frame: np.ndarray, box):
    """What ingest hands the engine: the same view at 704x480 (anamorphic)."""
    det = cv2.resize(frame, (DET_W, DET_H), interpolation=cv2.INTER_AREA)
    sx, sy = DET_W / HI_W, DET_H / HI_H
    return det, (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


class Cam:
    def __init__(self, name: str = "drive"):
        self.row = {"name": name, "ip": "192.0.2.10", "username": "u", "password": "p"}
        self.plate_zones = []


class Settings:
    def __init__(self, hires: bool = True):
        self.current = {"recognition": {"enabled": True, "plate_hires": hires}}

    def get(self):
        return self.current


class Served:
    """A fake camera: serves one JPEG, counts requests, optionally slowly."""

    def __init__(self, frame: np.ndarray, delay: float = 0.0):
        self.data = jpeg(frame)
        self.delay = delay
        self.calls = 0

    async def __call__(self, row):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.data


async def drive_past(pp: PlatePass, cam: Cam, det, box, *, tid: int, fid: str,
                     frames: int = 5, t0: float = 10.0, settle: float = 0.0) -> float:
    """Feed a stationary-in-frame vehicle for `frames` frames, 1.1 s apart.

    Starts at t=10, not 0: a track's pass clock starts at 0, so a first frame
    AT 0 is throttled as "too soon after the last pass" — harmless with the
    wall-clock times the engine really passes, misleading in a test."""
    t = t0
    for _ in range(frames):
        await pp.observe(cam, [Obs(tid, box)], det, t, event_fid=fid)
        # Let the background look finish before the next frame, as it would
        # in real time between frames a second apart.
        await asyncio.sleep(settle)
        t += 1.1
    return t


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def localizer_checks(reader: PlateReader) -> None:
    print("\nthe localizer reads a plate whatever resolution the crop came from")

    def reads(car, text: str) -> bool:
        for (x1, y1, x2, y2) in candidate_regions(car):
            strip = deskew(car[y1:y2, x1:x2])
            got, _ = reader.read_blocking(strip)
            if got == text and score_plate(strip).total >= MIN_QUALITY:
                return True
        return False

    for pw in (120, 180, 220, 280, 360, 480):
        car_w = int(pw * 4.4)
        car, _ = fake_vehicle(w=car_w, h=int(car_w * 0.7), plate_w=pw)
        check(reads(car, PLATE), f"a {pw} px plate on a {car_w} px car is found, kept and read")

    # The sweep the working widths and the aspect band were chosen on. Before
    # either change 86 of these read; the floor leaves room for OpenCV version
    # drift without letting the regression back in.
    total = hits = 0
    for pw in (100, 120, 150, 180, 220, 250, 280, 320, 360, 420, 480, 560):
        for text in ("7ABC123", "XYZ8901", "4KDM772"):
            for angle in (0, 6):
                for body in (70, 30):
                    car_w = int(pw * 4.4)
                    car, _ = fake_vehicle(text=text, w=car_w, h=int(car_w * 0.7),
                                          plate_w=pw, angle=angle, body=body)
                    total += 1
                    hits += reads(car, text)
    check(hits >= 138, f"across {total} plates of every size, angle and body colour, "
                       f"{hits} are read (floor 138; 86 before this change)")


def problem_checks(reader: PlateReader) -> None:
    print("\nthe problem: a plate legible at full resolution is invisible to the detect frame")
    frame, box = scene(plate_w=150)
    det, dbox = to_detect(frame, box)
    x1, y1, x2, y2 = (int(v) for v in dbox)
    det_regions = candidate_regions(det[y1:y2, x1:x2])
    det_reads = [reader.read_blocking(deskew(det[y1 + a:y1 + c, x1 + b0:x1 + d]))[0]
                 for (b0, a, d, c) in det_regions]
    check(PLATE not in det_reads,
          f"on the {DET_W}x{DET_H} detect frame the plate is ~{int(150 * DET_W / HI_W)} px "
          f"wide and is not read (reads: {det_reads})")
    hx1, hy1, hx2, hy2 = box
    hi_crop = frame[hy1:hy2, hx1:hx2]
    hi_reads = [reader.read_blocking(deskew(hi_crop[a:c, b0:d]))[0]
                for (b0, a, d, c) in candidate_regions(hi_crop)]
    check(PLATE in hi_reads, f"at {HI_W}x{HI_H} the same plate reads as {PLATE!r}")


def locate_checks() -> None:
    print("\nfinding the vehicle again in a snapshot taken a moment later")
    frame, box = scene(plate_w=150)
    det, dbox = to_detect(frame, box)
    x1, y1, x2, y2 = (int(round(v)) for v in dbox)
    tmpl = det[y1:y2, x1:x2].copy()

    found = locate(tmpl, (x1, y1, x2, y2), det.shape[:2], frame)
    check(found is not None, "a vehicle that has not moved is found")
    if found:
        (bx1, by1, bx2, by2), score = found
        err = max(abs(bx1 - box[0]), abs(by1 - box[1]), abs(bx2 - box[2]), abs(by2 - box[3]))
        check(err < 12, f"...at the right full-resolution box (off by {err:.0f} px of {HI_W})")
        check(score > 0.8, f"...with a confident match ({score:.2f})")

    moved, mbox = scene(plate_w=150, shift_x=140)
    found = locate(tmpl, (x1, y1, x2, y2), det.shape[:2], moved)
    check(found is not None, "a vehicle that moved 140 px (full-res) since the detect frame is found")
    if found:
        (bx1, _, _, _), _ = found
        check(abs(bx1 - mbox[0]) < 15,
              f"...where it moved TO, not where it was (x {bx1:.0f} vs true {mbox[0]})")

    empty = BACKGROUND.copy()
    check(locate(tmpl, (x1, y1, x2, y2), det.shape[:2], empty) is None,
          "a snapshot the car has already left yields no box, not a guess")
    flat = np.full_like(tmpl, 120)
    check(locate(flat, (x1, y1, x2, y2), det.shape[:2], frame) is None,
          "a featureless template matches nothing rather than everything")
    tiny = tmpl[:10, :10]
    check(locate(tiny, (x1, y1, x1 + 10, y1 + 10), det.shape[:2], frame) is None,
          f"a vehicle under {platesnap.MIN_TEMPLATE_PX} px in the detect frame is not attempted")


async def source_checks() -> None:
    print("\nthe snapshot source: shared, backed off, never raising")
    frame, _ = scene()
    served = Served(frame, delay=0.05)
    src = SnapshotSource(fetch_jpeg=served)
    row = {"name": "drive"}
    a, b = await asyncio.gather(src.fetch(row), src.fetch(row))
    check(served.calls == 1, f"two vehicles asking at once cost ONE request (got {served.calls})")
    check(a is not None and b is not None and a[0].shape == (HI_H, HI_W, 3),
          "...and both get the full-resolution frame")
    await src.fetch(row)
    check(served.calls == 1, "a third asking right after shares the same frame too")
    await asyncio.sleep(platesnap.SHARE_S * 2 + 0.05)
    check("drive" not in src._recent, "the shared frame is released rather than held between vehicles")
    await src.fetch(row)
    check(served.calls == 2, "once it is stale, a new request is made")
    check(src.status()["drive"]["resolution"] == f"{HI_W}x{HI_H}",
          "the status reports the resolution the camera actually serves")

    async def refuse(row):
        raise RuntimeError("camera 192.0.2.10: HTTP 401 for /cgi-bin/snapshot.cgi")

    bad = SnapshotSource(fetch_jpeg=refuse)
    results = [await bad.fetch({"name": "gate"}) for _ in range(platesnap.FAIL_LIMIT)]
    check(all(r is None for r in results), "a refusing camera returns None, never raises")
    check(not bad.available("gate"),
          f"after {platesnap.FAIL_LIMIT} failures in a row the camera is left alone")
    st = bad.status()["gate"]
    check("username/password" in st["last_error"],
          f"with a reason a person can act on ({st['last_error']!r})")
    check(st["backing_off_s"] > 0, "and the status says for how long")

    async def garbage(row):
        return b"not a jpeg"

    junk = SnapshotSource(fetch_jpeg=garbage)
    check(await junk.fetch({"name": "x"}) is None, "a body that is not a JPEG is a failure, not a crash")
    check("JPEG" in junk.status()["x"]["last_error"], "...and is reported as such")

    old = platesnap.FETCH_TIMEOUT_S
    platesnap.FETCH_TIMEOUT_S = 0.05
    try:
        slow = SnapshotSource(fetch_jpeg=Served(frame, delay=0.5))
        t0 = time.monotonic()
        r = await slow.fetch({"name": "slow"})
        check(r is None and time.monotonic() - t0 < 0.4,
              "a snapshot slower than the timeout is abandoned — it would be a different moment")
    finally:
        platesnap.FETCH_TIMEOUT_S = old

    noip = SnapshotSource()
    check(await noip.fetch({"name": "rtsp-only", "ip": ""}) is None,
          "a camera with no IP (a bare RTSP source) is simply not asked")
    check("IP" in noip.status()["rtsp-only"]["last_error"], "...and the status says why")


async def pass_checks(reader: PlateReader) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-plate-hires-"))
    frame, box = scene(plate_w=150)
    det, dbox = to_detect(frame, box)

    print("\nthe pass WITHOUT full-resolution looks: nothing, which is the bug")
    db = FakeDB(tmp / "a.db")
    plain = PlatePass(reader, db, tmp / "crops-a")
    await plain.reload_gallery()
    await drive_past(plain, Cam(), det, dbox, tid=1, fid="native.car-1")
    await plain.finish("drive", 1)
    check(db.rows("SELECT * FROM event_recognitions") == [],
          "reading the detect frame alone records no plate for this car")
    stats = plain.status()["cameras"]["drive"]
    check(stats["passes"] > 0 and stats["reads"] == 0,
          f"and the status shows passes with no reads ({stats['passes']} passes, "
          f"median strip {stats['median_strip_px']} px)")

    print("\nthe pass WITH full-resolution looks: the plate is read")
    db = FakeDB(tmp / "b.db")
    served = Served(frame)
    src = SnapshotSource(fetch_jpeg=served)
    pp = PlatePass(reader, db, tmp / "crops-b", snapshots=src)
    await pp.reload_gallery()
    await drive_past(pp, Cam(), det, dbox, tid=1, fid="native.car-1", settle=0.3)
    await pp.finish("drive", 1)
    await pp.wait_idle()
    ev = db.rows("SELECT * FROM event_recognitions")
    check(len(ev) == 1 and ev[0]["plate"] == PLATE,
          f"the event records {PLATE!r} (got {[r['plate'] for r in ev]})")
    cands = db.rows("SELECT * FROM recognition_candidates")
    check(len(cands) == 1 and cands[0]["plate"] == PLATE,
          "and it becomes an enrollable candidate")
    crop = cv2.imread(str(tmp / "crops-b" / cands[0]["image_path"])) if cands else None
    check(crop is not None and crop.shape[1] >= 100,
          f"whose stored crop is the full-resolution one "
          f"({crop.shape[1] if crop is not None else 0} px wide), not a smudge")
    stats = pp.status()["cameras"]["drive"]
    check(stats["hires_reads"] >= 2, f"the reads came from the snapshots ({stats['hires_reads']})")
    check(stats["votes_stored"] == 1 and stats["last_plate"] == PLATE,
          "and the status shows the stored plate")
    check(served.calls <= 5, f"at most one snapshot per frame a second apart ({served.calls})")
    check(pp.status()["hires"] is True, "the status reports full-resolution looks as on")

    print("\nthe car moves between the detect frame and the snapshot")
    moved, _ = scene(plate_w=150, shift_x=160)
    db = FakeDB(tmp / "c.db")
    pp = PlatePass(reader, db, tmp / "crops-c", snapshots=SnapshotSource(fetch_jpeg=Served(moved)))
    await pp.reload_gallery()
    await drive_past(pp, Cam(), det, dbox, tid=1, fid="native.car-2", settle=0.3)
    await pp.finish("drive", 1)
    await pp.wait_idle()
    ev = db.rows("SELECT * FROM event_recognitions")
    check(len(ev) == 1 and ev[0]["plate"] == PLATE,
          "the plate is still read from where the car moved to")

    print("\nthe detection loop never waits on a camera")
    db = FakeDB(tmp / "d.db")
    slow = Served(frame, delay=0.6)
    pp = PlatePass(reader, db, tmp / "crops-d", snapshots=SnapshotSource(fetch_jpeg=slow))
    await pp.reload_gallery()
    t0 = time.monotonic()
    await pp.observe(Cam(), [Obs(1, dbox)], det, 10.0, event_fid="native.car-3")
    check(time.monotonic() - t0 < 0.4,
          f"observe() returns while a 0.6 s snapshot is still in flight "
          f"({time.monotonic() - t0:.2f} s)")
    await pp.observe(Cam(), [Obs(1, dbox)], det, 11.1, event_fid="native.car-3")
    t0 = time.monotonic()
    await pp.finish("drive", 1)
    check(time.monotonic() - t0 < 0.2,
          "and a track ending mid-snapshot returns at once instead of waiting for it")
    check(db.rows("SELECT * FROM event_recognitions") == [],
          "...with the vote deferred, not taken early without the snapshot's read")
    await pp.wait_idle()
    ev = db.rows("SELECT * FROM event_recognitions")
    check(len(ev) == 1 and ev[0]["plate"] == PLATE,
          "once the snapshot lands, the deferred vote stores the plate")

    print("\nbounded cost")
    db = FakeDB(tmp / "e.db")
    served = Served(frame)
    pp = PlatePass(reader, db, tmp / "crops-e", snapshots=SnapshotSource(fetch_jpeg=served))
    old_settled = pp_mod.SETTLED_READS
    pp_mod.SETTLED_READS = 10_000  # never settle, to exercise the cap alone
    try:
        await drive_past(pp, Cam(), det, dbox, tid=7, fid="native.car-4",
                         frames=pp_mod.HIRES_MAX_PER_TRACK + 6, settle=0.3)
    finally:
        pp_mod.SETTLED_READS = old_settled
    asked = pp.status()["cameras"]["drive"]["hires_requested"]
    check(asked == pp_mod.HIRES_MAX_PER_TRACK,
          f"a vehicle that stays is capped at {pp_mod.HIRES_MAX_PER_TRACK} full-resolution "
          f"looks over {pp_mod.HIRES_MAX_PER_TRACK + 6} frames (got {asked})")
    await pp.finish("drive", 7)
    await pp.wait_idle()

    db = FakeDB(tmp / "f.db")
    served = Served(frame)
    pp = PlatePass(reader, db, tmp / "crops-f", snapshots=SnapshotSource(fetch_jpeg=served))
    # Spaced past SHARE_S so each look is a fresh snapshot, as it is with real
    # frames a second apart; inside it, looks share one frame and add nothing.
    await drive_past(pp, Cam(), det, dbox, tid=8, fid="native.car-5", frames=10,
                     settle=platesnap.SHARE_S + 0.1)
    asked = pp.status()["cameras"]["drive"]["hires_requested"]
    check(asked <= pp_mod.SETTLED_READS + 1,
          f"and once the plate is read confidently it stops asking ({asked} looks "
          f"over 10 frames)")
    await pp.finish("drive", 8)
    await pp.wait_idle()

    print("\nthe switch")
    db = FakeDB(tmp / "g.db")
    served = Served(frame)
    pp = PlatePass(reader, db, tmp / "crops-g", snapshots=SnapshotSource(fetch_jpeg=served))
    pp._settings = Settings(hires=False)
    await drive_past(pp, Cam(), det, dbox, tid=9, fid="native.car-6", settle=0.05)
    check(served.calls == 0, "with plate_hires off, the camera is never asked")
    check(pp.status()["hires"] is False, "and the status says so")
    pp._settings = Settings(hires=True)
    await drive_past(pp, Cam(), det, dbox, tid=9, fid="native.car-6", t0=100.0, frames=1,
                     settle=0.3)
    check(served.calls == 1, "switching it on takes effect on the next frame, not in five minutes")
    await pp.finish("drive", 9)
    await pp.wait_idle()

    print("\na camera whose snapshot is no bigger than detection is not asked again")
    small = cv2.resize(frame, (DET_W, DET_H))
    db = FakeDB(tmp / "h.db")
    served = Served(small)
    src = SnapshotSource(fetch_jpeg=served)
    pp = PlatePass(reader, db, tmp / "crops-h", snapshots=src)
    await drive_past(pp, Cam(), det, dbox, tid=10, fid="native.car-7", settle=0.2)
    check(served.calls == 1, f"one look, then it stops ({served.calls})")
    reason = src.status()["drive"]["no_gain"]
    check("snapshot resolution" in reason, f"and the status says what to change ({reason[:60]}…)")
    await pp.finish("drive", 10)
    await pp.wait_idle()

    print("\nforgetting a camera cancels its looks")
    db = FakeDB(tmp / "i.db")
    pp = PlatePass(reader, db, tmp / "crops-i",
                   snapshots=SnapshotSource(fetch_jpeg=Served(frame, delay=1.0)))
    await pp.observe(Cam(), [Obs(11, dbox)], det, 10.0, event_fid="native.car-8")
    task = pp._tracks[("drive", 11)].hires_task
    pp.forget_camera("drive")
    await asyncio.sleep(0.05)
    check(task is not None and task.cancelled(), "the in-flight look is cancelled, not orphaned")
    check("drive" not in pp.status()["cameras"], "and the camera's stats go with it")


async def main() -> int:
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-plate-models-"))
    reader = PlateReader(models_dir)
    if not await reader.load():
        print("SKIP: plate OCR could not be downloaded (offline?)")
        return 0

    localizer_checks(reader)
    problem_checks(reader)
    locate_checks()
    await source_checks()
    await pass_checks(reader)

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (full-resolution plate reading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
