#!/usr/bin/env python3
"""Plate localization, OCR and the per-track pass — against the REAL pinned model.

Plates ship WITHOUT a learned detector (see native/plates.py: every accurate one
descends from GPL-3.0 or AGPL-3.0 YOLO). Localization is classical CV over
D-FINE's vehicle box and only the OCR is a model. That trade puts the weight on
two claims this suite has to actually demonstrate:

1. THE LOCALIZER FINDS PLATES. A gradient localizer bounds the plate's TEXT ROW,
   not its border — roughly 5-7:1, not the ~2:1 of a whole plate. Filtering
   those with the whole-plate aspect band silently rejected a clean, perfectly
   legible plate while writing this, which is exactly the "too precise" failure
   the design is supposed to avoid.
2. THE OCR DISCRIMINATES. The generous localizer is only acceptable because the
   model returns an EMPTY string at zero confidence for things that were never
   plates, instead of inventing characters. If that stopped being true, every
   grille slat would become a licence plate.

Plus what the pass itself must get right: one read per retained shot, voting
across independent looks, an unmatched plate becoming an enrollable candidate,
a matched one not, and a low-confidence vote being DISCARDED rather than
recorded — a wrong plate on an event is worse than no plate.

NETWORK: the first run downloads 3.3 MB. Set VIGILUME_TEST_MODELS_DIR to a
directory already holding plate_ocr.onnx to skip it. Offline, the suite SKIPS
(exit 0) rather than failing.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

from app.db import RECOGNITION_SCHEMA  # noqa: E402
from app.native.plates import (  # noqa: E402
    PLATE_MODELS,
    PlateReader,
    candidate_regions,
    deskew,
    is_vehicle,
)
from app.native.platepass import MIN_VOTE_CONFIDENCE, PlatePass  # noqa: E402
from app.native.recognition import PlateRead, vote_plate  # noqa: E402

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


# --------------------------------------------------------------------------
# Synthetic vehicles
# --------------------------------------------------------------------------


def fake_vehicle(
    text: str = "7ABC123", w: int = 420, h: int = 300,
    plate_w: int = 150, angle: float = 0, body: int = 70,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A car-ish box with a plate low on it. Returns (image, true plate box)."""
    img = np.full((h, w, 3), body, np.uint8)
    cv2.rectangle(img, (0, 0), (w, int(h * 0.45)), (95, 95, 100), -1)          # body
    cv2.rectangle(img, (30, int(h * 0.5)), (w - 30, int(h * 0.62)), (40, 40, 40), -1)  # grille
    ph = int(plate_w / 2)
    px, py = (w - plate_w) // 2, int(h * 0.70)
    plate = np.full((ph, plate_w, 3), 245, np.uint8)
    cv2.putText(plate, text, (6, int(ph * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                plate_w / 190, (20, 20, 20), 2, cv2.LINE_AA)
    if angle:
        m = cv2.getRotationMatrix2D((plate_w / 2, ph / 2), angle, 1.0)
        plate = cv2.warpAffine(plate, m, (plate_w, ph), borderMode=cv2.BORDER_REPLICATE)
    img[py:py + ph, px:px + plate_w] = plate
    return img, (px, py, px + plate_w, py + ph)


class FakeCursor:
    def __init__(self, rows, lastrowid=0):
        self._rows = rows
        self.lastrowid = lastrowid

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeDB:
    def __init__(self, path):
        self._c = sqlite3.connect(path)
        self._c.row_factory = sqlite3.Row
        # The REAL schema — see the same note in facepass_smoke.py. A local
        # copy silently diverges from db.py and turns a missing column into
        # "nothing was stored", which reads as a logic bug in the pass.
        self._c.executescript(RECOGNITION_SCHEMA)
        self._c.commit()
        self.conn = self

    async def execute(self, sql, params=()):
        cur = self._c.execute(sql, params)
        rows = cur.fetchall() if sql.strip().upper().startswith("SELECT") else []
        return FakeCursor(rows, cur.lastrowid)

    async def commit(self):
        self._c.commit()

    def rows(self, sql, params=()):
        return self._c.execute(sql, params).fetchall()


class Obs:
    def __init__(self, tracker_id, box, label="car", score=0.9):
        self.tracker_id, self.box, self.label, self.score = tracker_id, box, label, score


class Cam:
    def __init__(self, name="drive"):
        self.row = {"name": name}
        self.plate_zones = []


def label_checks() -> None:
    print("\nvehicle labels")
    check(is_vehicle("car") and is_vehicle("truck") and is_vehicle("bus"),
          "cars, trucks and buses are searched for plates")
    check(is_vehicle("CAR"), "matching is case-insensitive")
    check(not is_vehicle("person") and not is_vehicle("dog"),
          "people and animals are not")


def localizer_checks(reader: PlateReader) -> None:
    print("\nlocalization + OCR on synthetic vehicles")
    cases = [
        ("level", {}),
        ("tilted 7deg", {"angle": 7}),
        ("small plate", {"plate_w": 110}),
        ("dark vehicle", {"body": 30}),
        ("different plate", {"text": "XYZ8901"}),
    ]
    for label, kw in cases:
        img, _ = fake_vehicle(**kw)
        regions = candidate_regions(img)
        reads = []
        for (x1, y1, x2, y2) in regions:
            text, conf = reader.read_blocking(deskew(img[y1:y2, x1:x2]))
            if text:
                reads.append((text, conf))
        expected = kw.get("text", "7ABC123")
        got = reads[0][0] if reads else ""
        check(got == expected,
              f"{label}: localized and read {expected!r} (got {got!r} from "
              f"{len(regions)} region(s))")
        if reads:
            check(reads[0][1] > 0.8, f"{label}: with high confidence ({reads[0][1]:.2f})")

    print("\nthe OCR discriminates — this is what makes a generous localizer safe")
    noise = np.random.default_rng(1).integers(0, 255, (300, 420, 3), dtype=np.uint8)
    reads = [reader.read_blocking(noise[y1:y2, x1:x2])
             for (x1, y1, x2, y2) in candidate_regions(noise)]
    check(all(not t for t, _ in reads),
          "pure noise yields NO characters rather than an invented plate")
    check(all(c == 0.0 for _, c in reads), "...at zero confidence")

    flat = np.full((300, 420, 3), 120, np.uint8)
    check(candidate_regions(flat) == [],
          "a flat, featureless crop proposes no regions at all")
    check(candidate_regions(np.zeros((5, 5, 3), np.uint8)) == [],
          "a crop smaller than a plate proposes nothing")
    check(candidate_regions(None) == [], "a missing crop is handled, not raised on")

    print("\ndeskew")
    img, _ = fake_vehicle(angle=9)
    regions = candidate_regions(img)
    if regions:
        x1, y1, x2, y2 = regions[0]
        strip = img[y1:y2, x1:x2]
        check(deskew(strip).shape == strip.shape, "deskew preserves the crop size")
    check(deskew(np.zeros((0, 0, 3), np.uint8)) is not None,
          "deskew on an empty crop returns rather than raising")


async def pass_checks(reader: PlateReader) -> None:
    print("\nthe pass: an unknown plate becomes a candidate")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-plates-"))
    db = FakeDB(tmp / "t.db")
    crops = tmp / "crops"
    pp = PlatePass(reader, db, crops)
    await pp.reload_gallery()

    cam = Cam()
    img, _ = fake_vehicle()
    h, w = img.shape[:2]
    obs = Obs(1, (0.0, 0.0, float(w), float(h)))
    t = 0.0
    for _ in range(5):
        await pp.observe(cam, [obs], img, t, event_fid="native.car-1")
        t += 1.0
    await pp.finish("drive", 1)

    cands = db.rows("SELECT * FROM recognition_candidates")
    check(len(cands) == 1, f"one plate candidate stored (got {len(cands)})")
    check(cands[0]["plate"] == "7ABC123", f"with the voted text (got {cands[0]['plate']!r})")
    check(cands[0]["kind"] == "plate" and cands[0]["camera"] == "drive",
          "tagged as a plate on the right camera")
    check(cands[0]["embedding"] is None,
          "a plate candidate carries NO embedding — it is a string, not a vector")
    check(bool(cands[0]["image_path"]) and (crops / cands[0]["image_path"]).is_file(),
          "and its crop is on disk")

    ev = db.rows("SELECT * FROM event_recognitions")
    check(len(ev) == 1 and ev[0]["profile_id"] is None,
          "the event records a plate that matched no vehicle")
    check(ev[0]["plate"] == "7ABC123", "...carrying the plate text itself")
    check(ev[0]["event_fid"] == "native.car-1", "tied to the open event")

    # The accumulator buffers in memory and writes on the maintenance tick, so
    # the sightings are pending until flushed — that IS the design (a pass runs
    # several times a second and a write per sighting would be churn).
    check(pp.heatmap.pending >= 1, "plate sightings are buffered into the heatmap")
    await pp.heatmap.flush()
    heat = db.rows("SELECT * FROM recognition_heatmap WHERE kind='plate'")
    check(len(heat) >= 1, "...and reach the table on flush")
    check(all(r["quality_sum"] > 0 for r in heat),
          "carrying the legibility of those sightings, not just their count")

    print("\ndedupe: the same plate again is not a second candidate")
    obs2 = Obs(2, (0.0, 0.0, float(w), float(h)))
    for _ in range(5):
        await pp.observe(cam, [obs2], img, t, event_fid="native.car-2")
        t += 1.0
    await pp.finish("drive", 2)
    check(len(db.rows("SELECT * FROM recognition_candidates")) == 1,
          "still one candidate — the same plate string is the same vehicle")
    check(len(db.rows("SELECT * FROM event_recognitions")) == 2,
          "...but both sightings are recorded as events")

    print("\nthe pass: an enrolled plate matches")
    db._c.execute("INSERT INTO profiles (id, kind, name, enabled, created_at, updated_at) "
                  "VALUES (1,'vehicle','Truck',1,0,0)")
    db._c.execute("INSERT INTO profile_samples (profile_id, plate, created_at) "
                  "VALUES (1,'7ABC123',0)")
    db._c.commit()
    await pp.reload_gallery()

    obs3 = Obs(3, (0.0, 0.0, float(w), float(h)))
    for _ in range(5):
        await pp.observe(cam, [obs3], img, t, event_fid="native.car-3")
        t += 1.0
    await pp.finish("drive", 3)
    ev3 = db.rows("SELECT * FROM event_recognitions WHERE event_fid='native.car-3'")
    check(len(ev3) == 1 and ev3[0]["profile_id"] == 1, "the event names the matched vehicle")
    check(ev3[0]["name"] == "Truck", "...by name")
    check(len(db.rows("SELECT * FROM recognition_candidates")) == 1,
          "and no new candidate — the plate is enrolled now")

    print("\nROI gating")
    import app.native.zones as zonelib

    zoned = Cam()
    zoned.row = {"name": "drive", "detect_width": w, "detect_height": h,
                 "plate_zones": [{"name": "kerb",
                                  "points": [[0.0, 0.0], [0.05, 0.0], [0.05, 0.05]]}]}
    zoned.plate_zones = zonelib.polygon_zones(zoned.row, "plate_zones", "plate")
    before = len(pp._tracks)
    await pp.observe(zoned, [Obs(9, (0.0, 0.0, float(w), float(h)))], img, t + 50)
    check(len(pp._tracks) == before,
          "a vehicle outside the plate ROI gets no plate work at all")

    print("\nnon-vehicles are ignored")
    before = len(pp._tracks)
    await pp.observe(cam, [Obs(10, (0.0, 0.0, float(w), float(h)), label="person")], img, t + 60)
    check(len(pp._tracks) == before, "a person never triggers a plate read")


def vote_discard_checks() -> None:
    print("\na vote nobody would stand behind is discarded, not recorded")
    # Three reads disagreeing on one position: the vote's confidence is its
    # WEAKEST character, so a genuine coin flip lands below the floor.
    reads = [
        PlateRead("7ABC123", confidence=0.9, quality=0.9),
        PlateRead("7ABC923", confidence=0.9, quality=0.9),
        PlateRead("7ABC523", confidence=0.9, quality=0.9),
    ]
    v = vote_plate(reads)
    check(v is not None, "three conflicting reads still produce a vote")
    check(v.confidence < MIN_VOTE_CONFIDENCE,
          f"...whose confidence ({v.confidence:.2f}) is below the "
          f"{MIN_VOTE_CONFIDENCE} floor the pass requires")
    check(min(v.agreement) < 0.5, "the contested character is visibly weak")

    agree = [PlateRead("7ABC123", confidence=0.95, quality=0.9) for _ in range(3)]
    v2 = vote_plate(agree)
    check(v2.confidence >= MIN_VOTE_CONFIDENCE,
          "three agreeing reads clear the floor comfortably")


async def main() -> int:
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-plate-models-"))
    reader = PlateReader(models_dir)
    print(f"models: {models_dir}")
    if not await reader.load():
        print("SKIP: plate OCR could not be downloaded (offline?)")
        return 0
    check(reader.ready, "the pinned OCR loads and hash-verifies")
    path = models_dir / "plate_ocr.onnx"
    check(path.stat().st_size == PLATE_MODELS["plate_ocr"]["bytes"],
          f"on disk it is exactly the pinned {PLATE_MODELS['plate_ocr']['bytes']} bytes")
    check(PLATE_MODELS["plate_ocr"]["license"] == "MIT",
          "and it is the permissively-licensed model, not a YOLO derivative")

    label_checks()
    localizer_checks(reader)
    await pass_checks(reader)
    vote_discard_checks()

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (plate localization, OCR and the pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
