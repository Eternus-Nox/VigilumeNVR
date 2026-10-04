#!/usr/bin/env python3
"""Faces read from the camera's full-resolution snapshot.

The detect stream is ~704x480; a face at the door there is a few dozen pixels,
under the 40 px the pass will even try. This builds a scene where the face is
~150 px in the camera's own picture and ~30 px in the detect frame, and checks:

1. WITHOUT snapshots the face is never read — it is too small on the detect
   frame. (The problem, pinned so the fix can be seen to matter.)
2. WITH them, a full-resolution look finds the person, reads the face from the
   real pixels, and the stored candidate is a sharp, high-quality crop.
3. It never blocks the engine: observe() returns at once, the look runs in the
   background, and a track that ends mid-look still gets its answer.
4. One snapshot is shared: a second person on the same camera at the same
   moment does not cost a second request.
5. A person who moved since the detect frame is searched for around where they
   were, and only a face where THEIR head would be is taken.
6. The switch: face_hires off means no snapshot is asked for at all.

Needs the face models and a sample photo (see facepass_smoke); SKIPS without.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from app.native.facepass import FacePass  # noqa: E402
from app.native.snapshots import SnapshotSource  # noqa: E402
from app.native.recognizer import FaceRecognizer  # noqa: E402
from facepass_smoke import Cam, Obs, load_face_image, make_db  # noqa: E402

PASS = 0
_failures: list[str] = []

HW, HH = 2560, 1440      # the camera's own picture
DET = 0.2                # detect frame = 512x288: the face lands under 40 px
PX, PY = 1100, 380       # where the person stands in the full frame


def check(cond: bool, label: str) -> None:
    global PASS
    PASS += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


def scene(face: np.ndarray, x: int = PX, y: int = PY) -> np.ndarray:
    rng = np.random.default_rng(3)
    img = np.full((HH, HW, 3), 110, np.uint8)
    img = cv2.add(img, rng.integers(0, 25, (HH, HW, 3), dtype=np.uint8))
    fh, fw = face.shape[:2]
    img[y:y + fh, x:x + fw] = face
    # A "body" under the head, so the person box is person-shaped.
    cv2.rectangle(img, (x - 40, y + fh), (x + fw + 40, min(HH - 1, y + fh + 520)), (60, 70, 90), -1)
    return img


def person_box(face: np.ndarray, x: int = PX, y: int = PY) -> tuple[float, ...]:
    fh, fw = face.shape[:2]
    return tuple(v * DET for v in (x - 60, y - 30, x + fw + 60, min(HH - 1, y + fh + 520)))


class CountingSource:
    """Serves the full-resolution scene as the camera's snapshot.cgi would."""

    def __init__(self, img: np.ndarray, delay: float = 0.15) -> None:
        self.calls = 0
        self.delay = delay
        self.jpeg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()

    async def __call__(self, row):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.jpeg


class Settings:
    def __init__(self, **recognition):
        self.current = {"recognition": {"face_hires": True, **recognition}}


async def drive(fp: FacePass, det: np.ndarray, boxes, *, seconds: float = 3.0,
                tid0: int = 1) -> float:
    """Feed the pass detect frames at 5 fps; returns the slowest observe()."""
    cam = Cam()
    slowest, t = 0.0, time.time()
    for i in range(int(seconds * 5)):
        obs = [Obs(tid0 + k, b) for k, b in enumerate(boxes)]
        t0 = time.monotonic()
        await fp.observe(cam, obs, det, t + i * 0.2, event_fid="native.door")
        slowest = max(slowest, time.monotonic() - t0)
        await asyncio.sleep(0.2)
    return slowest


async def main() -> int:
    face = load_face_image()
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-face-models-"))
    rec = FaceRecognizer(models_dir)
    if face is None or not await rec.load():
        print("SKIP: needs the face models and a sample face photo")
        return 0
    face = cv2.resize(face, (300, 300), interpolation=cv2.INTER_AREA)
    full = scene(face)
    det = cv2.resize(full, (int(HW * DET), int(HH * DET)), interpolation=cv2.INTER_AREA)
    box = person_box(face)
    # Measured as the pass sees it: in the PERSON's crop, at each resolution.
    x1, y1, x2, y2 = (int(v / DET) for v in box)
    big = await rec.detect(full[y1:y2, x1:x2])
    check(bool(big) and big[0].width >= 80,
          f"the face is {int(big[0].width) if big else 0} px in the camera's own picture")
    small = await rec.detect(det[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
    check(not small, "and too small to be read at all on the detect frame")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-face-hires-"))

    print("\n1. without snapshots: too small on the detect frame")
    db = make_db(tmp / "a.db")
    fp = FacePass(rec, db, tmp / "crops-a")
    await fp.reload_gallery()
    await drive(fp, det, [box])
    await fp.finish("front", 1)
    await fp.wait_idle()
    check(db.rows("SELECT * FROM recognition_candidates") == [],
          "nothing is read: the face is under the 40 px the pass tries")

    print("\n2-3. with snapshots: read from the full-resolution pixels, without blocking")
    src = CountingSource(full)
    db = make_db(tmp / "b.db")
    fp = FacePass(rec, db, tmp / "crops-b", snapshots=SnapshotSource(fetch_jpeg=src))
    fp._settings = Settings()
    await fp.reload_gallery()
    slowest = await drive(fp, det, [box], seconds=2.0)
    check(slowest < 0.1, f"observe() never waits on the camera (slowest {slowest * 1000:.0f} ms, "
                         f"a snapshot takes {src.delay * 1000:.0f})")
    t0 = time.monotonic()
    await fp.finish("front", 1)
    check(time.monotonic() - t0 < 0.05, "finish() returns at once even with a look in flight")
    await fp.wait_idle()
    rows = db.rows("SELECT * FROM recognition_candidates")
    check(len(rows) == 1, f"the face is read and kept as a candidate (got {len(rows)})")
    check(rows and rows[0]["quality"] >= 0.6,
          f"from a sharp shot (quality {rows[0]['quality']:.2f})" if rows else "from a sharp shot")
    st = fp.status()
    check(st["hires"]["faces"] >= 1 and st["hires"]["requested"] <= 3,
          f"via full-resolution looks, about one a second ({st['hires']})")
    check(src.calls == st["hires"]["requested"], "each look is one snapshot request")
    crop = cv2.imread(str(tmp / "crops-b" / rows[0]["image_path"])) if rows else None
    check(crop is not None and crop.shape[:2] == (112, 112),
          "the stored crop is the aligned 112x112 the embedding was taken from")
    if rows and rows[0]["frame_box"]:
        import json

        fb = json.loads(rows[0]["frame_box"])
        cx = (fb[0] + fb[2]) / 2 * HW
        check(PX <= cx <= PX + 300, "and its box points at the face in the frame")

    print("\n4. one snapshot serves two people at the same moment")
    src2 = CountingSource(full, delay=0.3)
    db = make_db(tmp / "c.db")
    fp = FacePass(rec, db, tmp / "crops-c", snapshots=SnapshotSource(fetch_jpeg=src2))
    fp._settings = Settings()
    other = tuple(v * DET for v in (200, 300, 600, 1300))
    await drive(fp, det, [box, other], seconds=0.4)
    for tid in (1, 2):
        await fp.finish("front", tid)
    await fp.wait_idle()
    check(src2.calls == 1, f"two tracked people, one request ({src2.calls})")

    print("\n5. a person who moved is found around where they were — by their head")
    moved = scene(face, PX + 120, PY + 40)        # stepped since the detect frame
    src3 = CountingSource(moved)
    db = make_db(tmp / "d.db")
    fp = FacePass(rec, db, tmp / "crops-d", snapshots=SnapshotSource(fetch_jpeg=src3))
    fp._settings = Settings()
    await drive(fp, det, [box], seconds=1.2)
    await fp.finish("front", 1)
    await fp.wait_idle()
    check(len(db.rows("SELECT * FROM recognition_candidates")) == 1,
          "still read, from the snapshot, after a step to the side")
    stranger = scene(face, 2100, 200)             # a face nowhere near this person
    src4 = CountingSource(stranger)
    blank = scene(np.full_like(face, 110))
    det_blank = cv2.resize(blank, (int(HW * DET), int(HH * DET)), interpolation=cv2.INTER_AREA)
    db = make_db(tmp / "e.db")
    fp = FacePass(rec, db, tmp / "crops-e", snapshots=SnapshotSource(fetch_jpeg=src4))
    fp._settings = Settings()
    await drive(fp, det_blank, [box], seconds=1.2)
    await fp.finish("front", 1)
    await fp.wait_idle()
    check(db.rows("SELECT * FROM recognition_candidates") == [],
          "a face somewhere else in the snapshot is NOT taken as this person's")

    print("\n6. face_hires off: no snapshot is asked for")
    src5 = CountingSource(full)
    db = make_db(tmp / "f.db")
    fp = FacePass(rec, db, tmp / "crops-f", snapshots=SnapshotSource(fetch_jpeg=src5))
    fp._settings = Settings(face_hires=False)
    await drive(fp, det, [box], seconds=1.2)
    await fp.finish("front", 1)
    await fp.wait_idle()
    check(src5.calls == 0 and fp.status()["hires_enabled"] is False,
          "no requests, and the status says the looks are off")

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (faces from full-resolution snapshots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
