#!/usr/bin/env python3
"""Plate reading on a REAL photograph, not a drawn one.

Every other plate suite uses plates drawn with cv2.putText, which are useful
for testing plumbing and useless for judging accuracy: a drawn plate is black
text on flat white, and a real one is embossed, reflective, framed, dirty and
lit from the wrong side. This suite uses the photograph fast-alpr ships as its
own test image (a car with the plate 5AU 5341), downloaded and pinned by
SHA-256 exactly like the models, and checks the claims the design rests on:

1. THE DETECTOR FINDS REAL PLATES where the classical localizer does not — at
   half size the classical one finds nothing; the detector still finds it.
2. TWO READERS AND A PER-CHARACTER VOTE read what one reader gets wrong: at
   half size the small reader says 5AJ5341; the vote says 5AU5341.
3. THE WHOLE PASS WORKS ON IT: detect frame -> full-resolution snapshot ->
   detector -> both readers -> vote -> a stored plate.

NETWORK: downloads the photo (353 KB) and the plate models on first run. Set
VIGILUME_TEST_MODELS_DIR to reuse models. Offline, the suite SKIPS (exit 0).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.native.plates import PlateReader, candidate_regions, deskew  # noqa: E402
from app.native.platepass import PlatePass  # noqa: E402
from app.native.platesnap import SnapshotSource  # noqa: E402
from app.native.recognition import PlateRead, normalize_plate, vote_plate  # noqa: E402
from plates_smoke import FakeDB, Obs  # noqa: E402

PHOTO_URL = "https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png"
PHOTO_SHA256 = "c154c3e0873fa35076ecf4f611d345c092450217fe75737b2a77183018218255"
TRUTH = "5AU5341"
#: The car in the photo, (x1, y1, x2, y2) — roughly what a vehicle detector
#: returns for it.
CAR = (40, 40, 455, 275)

PASS = 0
_failures: list[str] = []


def check(cond: bool, label: str) -> None:
    global PASS
    PASS += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


def fetch_photo(cache: Path) -> np.ndarray | None:
    path = cache / "fast_alpr_test_image.png"
    if not path.is_file():
        import httpx

        try:
            data = httpx.get(PHOTO_URL, timeout=30.0, follow_redirects=True).content
        except Exception:
            return None
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PHOTO_SHA256:
        path.unlink(missing_ok=True)
        return None
    return cv2.imread(str(path))


def reads_of(reader: PlateReader, crop: np.ndarray) -> list[PlateRead]:
    return [
        PlateRead(normalize_plate(t), confidence=m, quality=1.0, char_conf=c)
        for t, m, c in reader.read_all_blocking(crop) if normalize_plate(t)
    ]


def detector_checks(reader: PlateReader, photo: np.ndarray) -> None:
    print("\nfinding the plate")
    x1, y1, x2, y2 = CAR
    car = photo[y1:y2, x1:x2]
    found = reader.detect_blocking(car)
    check(bool(found) and found[0][4] > 0.8,
          f"the detector finds the plate on the car ({found[0][4]:.2f})" if found else
          "the detector finds the plate on the car")
    whole = reader.detect_blocking(photo)
    check(bool(whole), "and in the whole frame, with no vehicle box at all")

    half = cv2.resize(car, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    classic = []
    for (a, b, c, d) in candidate_regions(half):
        t, _ = reader.read_blocking(deskew(half[b:d, a:c]))
        classic.append(normalize_plate(t))
    check(TRUTH not in classic,
          "at half size (plate ~46 px) the classical localizer does NOT find it — "
          "the reason the detector replaced it")
    dh = reader.detect_blocking(half)
    check(bool(dh), "while the detector still does")


def reader_checks(reader: PlateReader, photo: np.ndarray) -> None:
    print("\nreading it")
    check(reader.status()["readers"] == 2, "both readers are loaded")
    x1, y1, x2, y2 = CAR
    car = photo[y1:y2, x1:x2]
    for label, img in (
        ("full size (~94 px)", car),
        ("half size (~46 px)", cv2.resize(car, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
        ("dark", cv2.convertScaleAbs(car, alpha=0.35)),
        ("infrared (grey)", cv2.cvtColor(cv2.cvtColor(car, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
        ("light motion blur", cv2.filter2D(car, -1, np.ones((1, 5)) / 5)),
    ):
        det = reader.detect_blocking(img)
        if not det:
            check(False, f"{label}: plate found")
            continue
        a, b, c, d, _ = det[0]
        reads = reads_of(reader, img[b:d, a:c])
        vote = vote_plate(reads)
        texts = [r.text for r in reads]
        check(vote is not None and vote.text == TRUTH,
              f"{label}: the vote reads {TRUTH} (readers said {texts}, vote "
              f"{vote.text if vote else None!r} at {vote.confidence if vote else 0:.2f})")


async def pass_checks(reader: PlateReader, photo: np.ndarray) -> None:
    print("\nthe whole pass, on the photograph")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-plate-real-"))
    # The photo is the camera's full-resolution snapshot; the detect frame is
    # the same view at half size, where the plate is ~46 px.
    detect = cv2.resize(photo, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    async def serve(_row):
        return jpg.tobytes()

    class Cam:
        row = {"name": "drive", "ip": "192.0.2.1"}
        plate_zones: list = []

    db = FakeDB(tmp / "r.db")
    pp = PlatePass(reader, db, tmp / "crops", snapshots=SnapshotSource(fetch_jpeg=serve))
    await pp.reload_gallery()
    box = tuple(v / 2.0 for v in CAR)
    t = 10.0
    for _ in range(4):
        await pp.observe(Cam(), [Obs(1, box)], detect, t, event_fid="native.real-1")
        await asyncio.sleep(0.45)
        t += 1.1
    await pp.finish("drive", 1)
    await pp.wait_idle()
    rows = db.rows("SELECT plate FROM event_recognitions")
    check([r["plate"] for r in rows] == [TRUTH],
          f"a real plate goes in as a detect frame and comes out stored as {TRUTH} "
          f"(got {[r['plate'] for r in rows]})")
    stats = pp.status()["cameras"]["drive"]
    check(stats["hires_reads"] >= 2, f"read from the full-resolution snapshot ({stats['hires_reads']} reads)")


async def main() -> int:
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-plate-models-"))
    photo = fetch_photo(models_dir)
    reader = PlateReader(models_dir)
    if photo is None or not await reader.load():
        print("SKIP: the test photo or plate models could not be downloaded (offline?)")
        return 0
    if not reader.has_detector:
        print("SKIP: the plate detector could not be downloaded")
        return 0
    detector_checks(reader, photo)
    reader_checks(reader, photo)
    await pass_checks(reader, photo)
    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (plate reading on a real photograph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
