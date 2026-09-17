#!/usr/bin/env python3
"""The face pass, end to end, against the REAL pinned models.

Everything else in the recognition stack is tested without weights
(recognition_smoke). This suite is the opposite: it downloads the two pinned
models, runs a real face through detect -> align -> embed -> match -> store, and
asserts on what lands in the database.

It is the only place the following can actually be checked:

1. THE PINS ARE RIGHT. A wrong URL, a stale revision, or an LFS pointer served
   instead of the binary all fail here rather than on the operator's box. (The
   pointer case is real: raw.githubusercontent.com returns a 131-byte text file
   for these paths, which downloads "successfully" and then fails every hash.)
2. ALIGNMENT IS APPLIED ONCE. SFace embeddings are only comparable in the
   canonical geometry; re-aligning an already-aligned crop warps it twice, and
   nothing about that fails loudly.
3. AN UNMATCHED FACE BECOMES A CANDIDATE, and a matched one does not.
4. DEDUPE. One stranger seen repeatedly is one row to review, not twelve.
5. RETENTION. Purging is the biometric retention control, and a retention of 0
   must mean "keep nothing", not "keep forever".

NETWORK: the first run downloads ~37 MB. Set VIGILUME_TEST_MODELS_DIR to a
directory that already holds yunet.onnx / sface.onnx to skip that. When neither
the models nor the network are available the suite SKIPS (exit 0) rather than
failing — an offline CI must not report a red build for a missing download.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

import cv2  # noqa: E402

from app.native.bestshot import score_face  # noqa: E402
from app.native.facepass import FacePass  # noqa: E402
from app.native.recognition import to_blob  # noqa: E402
from app.native.recognizer import (  # noqa: E402
    EMBEDDING_MODEL_KEY,
    FACE_MODELS,
    FaceRecognizer,
)

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
# A face to work with. cv2's own sample ships one; synthesizing a face that
# YuNet will actually detect is not feasible, so without it we skip.
# --------------------------------------------------------------------------


def load_face_image() -> "np.ndarray | None":
    for candidate in (
        os.environ.get("VIGILUME_TEST_FACE", ""),
        "/tmp/lena.jpg",
        str(Path(cv2.__file__).parent / "data" / "lena.jpg"),
    ):
        if candidate and Path(candidate).is_file():
            img = cv2.imread(candidate)
            if img is not None:
                return img
    return None


class FakeCursor:
    def __init__(self, rows, lastrowid=0):
        self._rows = rows
        self.lastrowid = lastrowid

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeDB:
    """Just enough aiosqlite surface for FacePass, backed by real sqlite3."""

    def __init__(self, path: Path):
        import sqlite3

        self._c = sqlite3.connect(path)
        self._c.row_factory = sqlite3.Row
        self.conn = self

    async def execute(self, sql, params=()):
        cur = self._c.execute(sql, params)
        rows = cur.fetchall() if sql.strip().upper().startswith("SELECT") else []
        return FakeCursor(rows, cur.lastrowid)

    async def commit(self):
        self._c.commit()

    def rows(self, sql, params=()):
        """Synchronous read for assertions — the async path is what FacePass
        uses; the test reads back with plain sqlite3."""
        return self._c.execute(sql, params).fetchall()


def make_db(path: Path) -> FakeDB:
    import sqlite3

    c = sqlite3.connect(path)
    c.executescript(
        """
        CREATE TABLE profiles (id INTEGER PRIMARY KEY, kind TEXT, name TEXT,
            notes TEXT DEFAULT '', enabled INTEGER DEFAULT 1, threshold REAL,
            created_at REAL DEFAULT 0, updated_at REAL DEFAULT 0);
        CREATE TABLE profile_samples (id INTEGER PRIMARY KEY, profile_id INTEGER,
            embedding BLOB, dim INTEGER DEFAULT 0, plate TEXT DEFAULT '',
            model_key TEXT DEFAULT '', image_path TEXT DEFAULT '',
            quality REAL DEFAULT 0, source_fid TEXT DEFAULT '', created_at REAL DEFAULT 0);
        CREATE TABLE recognition_candidates (id INTEGER PRIMARY KEY, kind TEXT,
            camera TEXT, event_fid TEXT DEFAULT '', embedding BLOB, dim INTEGER DEFAULT 0,
            plate TEXT DEFAULT '', model_key TEXT DEFAULT '', image_path TEXT DEFAULT '',
            quality REAL DEFAULT 0, best_score REAL DEFAULT 0, best_profile_id INTEGER,
            created_at REAL DEFAULT 0);
        CREATE TABLE event_recognitions (id INTEGER PRIMARY KEY, event_fid TEXT,
            kind TEXT, profile_id INTEGER, name TEXT DEFAULT '', plate TEXT DEFAULT '',
            score REAL DEFAULT 0, quality REAL DEFAULT 0, image_path TEXT DEFAULT '',
            created_at REAL DEFAULT 0);
        """
    )
    c.commit()
    c.close()
    return FakeDB(path)


class Obs:
    """Minimal stand-in for engine.Observation."""

    def __init__(self, tracker_id, box, label="person", score=0.9):
        self.tracker_id = tracker_id
        self.box = box
        self.label = label
        self.score = score


class Cam:
    def __init__(self, name="front"):
        self.row = {"name": name}
        self.face_zones = []


async def main() -> int:
    img = load_face_image()
    if img is None:
        print("SKIP: no sample face image available (set VIGILUME_TEST_FACE)")
        return 0

    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "") or
                      tempfile.mkdtemp(prefix="vigilume-face-models-"))
    rec = FaceRecognizer(models_dir)
    print(f"\nmodels: {models_dir}")
    if not await rec.load():
        print("SKIP: face models could not be downloaded (offline?)")
        return 0
    check(rec.ready, "both pinned models load and hash-verify")
    check(rec.model_key == EMBEDDING_MODEL_KEY, f"embedding space is {EMBEDDING_MODEL_KEY}")
    for key, pin in FACE_MODELS.items():
        path = models_dir / f"{key}.onnx"
        check(path.stat().st_size == pin["bytes"],
              f"{key} on disk is exactly the pinned {pin['bytes']} bytes "
              f"(an LFS pointer would be ~131)")

    print("\ndetect -> align -> embed")
    faces = await rec.detect(img)
    check(len(faces) == 1, f"one face found in the sample image (got {len(faces)})")
    face = faces[0]
    check(len(face.landmarks) == 5, "YuNet supplies the 5 landmarks bestshot wants")
    check(face.score > 0.8, f"with a confident score ({face.score:.2f})")

    aligned = await rec.align(img, face)
    check(aligned is not None and aligned.shape[:2] == (112, 112),
          "alignCrop produces the canonical 112x112 SFace input")
    vec = await rec.feature(aligned)
    check(vec is not None and vec.shape == (128,), "SFace produces a 128-d embedding")

    print("\nalignment is applied exactly once")
    # Feeding an already-aligned crop back through align() warps it twice. The
    # embedding must come from feature() on the aligned pixels, which is what
    # facepass does — this pins the difference so a future refactor cannot
    # silently reintroduce the double warp.
    refaced = await rec.detect(aligned)
    if refaced:
        double = await rec.feature(await rec.align(aligned, refaced[0]))
        from app.native.recognition import cosine, normalize

        sim = cosine(normalize(vec), normalize(double))
        check(sim < 0.999,
              f"a double-aligned crop embeds DIFFERENTLY (cos {sim:.4f}) — "
              "so passing the wrong crop in is a real bug, not a no-op")
    else:
        check(True, "re-detection on the aligned crop found nothing (double-warp N/A)")

    q = score_face(aligned, landmarks=face.landmarks)
    check(q.total > 0.5, f"the aligned crop scores as usable ({q.total:.2f}, {q.reason!r})")

    print("\nthe pass: an unknown face becomes a candidate")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-facepass-"))
    db = make_db(tmp / "t.db")
    crops = tmp / "crops"
    fp = FacePass(rec, db, crops)
    await fp.reload_gallery()
    check(fp.model_key == EMBEDDING_MODEL_KEY, "the pass reports the active embedding space")

    cam = Cam()
    h, w = img.shape[:2]
    obs = Obs(1, (0.0, 0.0, float(w), float(h)))
    t = 0.0
    for _ in range(4):
        await fp.observe(cam, [obs], img, t, event_fid="native.test-1")
        t += 1.0
    await fp.finish("front", 1)

    rows = db.rows("SELECT * FROM recognition_candidates")
    check(len(rows) == 1, f"one candidate stored for the unknown face (got {len(rows)})")
    cand = rows[0]
    check(cand["kind"] == "face" and cand["camera"] == "front", "candidate carries its camera")
    check(cand["model_key"] == EMBEDDING_MODEL_KEY, "and the model that embedded it")
    check(cand["dim"] == 128 and cand["embedding"] is not None, "and a real embedding")
    check(cand["quality"] > 0.5, f"and the shot's quality ({cand['quality']:.2f})")
    check(bool(cand["image_path"]) and (crops / cand["image_path"]).is_file(),
          "and its crop is written to disk")

    ev = db.rows("SELECT * FROM event_recognitions")
    check(len(ev) == 1 and ev[0]["profile_id"] is None,
          "an event_recognitions row records that a face was read and matched NOBODY")
    check(ev[0]["event_fid"] == "native.test-1", "tied to the open event")

    print("\ndedupe: the same stranger again is not a second row")
    obs2 = Obs(2, (0.0, 0.0, float(w), float(h)))
    for _ in range(4):
        await fp.observe(cam, [obs2], img, t, event_fid="native.test-2")
        t += 1.0
    await fp.finish("front", 2)
    rows = db.rows("SELECT * FROM recognition_candidates")
    check(len(rows) == 1, f"still one candidate after a second sighting (got {len(rows)})")
    ev = db.rows("SELECT * FROM event_recognitions")
    check(len(ev) == 2, "...but BOTH sightings are recorded as events")

    print("\nthe pass: an enrolled face matches and does NOT become a candidate")
    db._c.execute(
        "INSERT INTO profiles (id, kind, name, enabled) VALUES (1, 'person', 'Sample', 1)"
    )
    db._c.execute(
        "INSERT INTO profile_samples (profile_id, embedding, dim, model_key) VALUES (?,?,?,?)",
        (1, to_blob(vec), 128, EMBEDDING_MODEL_KEY),
    )
    db._c.commit()
    await fp.reload_gallery()

    obs3 = Obs(3, (0.0, 0.0, float(w), float(h)))
    for _ in range(4):
        await fp.observe(cam, [obs3], img, t, event_fid="native.test-3")
        t += 1.0
    await fp.finish("front", 3)

    rows = db.rows("SELECT * FROM recognition_candidates")
    check(len(rows) == 1, "no NEW candidate — the face is enrolled now")
    ev = db.rows("SELECT * FROM event_recognitions WHERE event_fid = 'native.test-3'")
    check(len(ev) == 1 and ev[0]["profile_id"] == 1,
          "the event names the matched profile")
    check(ev[0]["name"] == "Sample", "...by name")
    check(ev[0]["score"] > 0.9, f"at a high similarity ({ev[0]['score']:.3f})")

    print("\nROI zones gate the pass")
    import app.native.zones as zonelib

    cam_zoned = Cam()
    cam_zoned.row = {"name": "front", "detect_width": w, "detect_height": h,
                     "face_zones": [{"name": "corner",
                                     "points": [[0.0, 0.0], [0.05, 0.0], [0.05, 0.05]]}]}
    cam_zoned.face_zones = zonelib.polygon_zones(cam_zoned.row, "face_zones", "face")
    check(len(cam_zoned.face_zones) == 1, "the face ROI parses into a detect-space zone")
    before = len(fp._tracks)
    await fp.observe(cam_zoned, [Obs(9, (0.0, 0.0, float(w), float(h)))], img, t + 50)
    check(len(fp._tracks) == before,
          "a person whose feet land outside the face ROI gets no pass at all")

    print("\nretention")
    removed = await fp.purge_expired(0)
    check(removed >= 1, f"a retention of 0 purges the store ({removed} row(s)) — "
                        "'keep nothing' must not read as 'keep forever'")
    rows = db.rows("SELECT * FROM recognition_candidates")
    check(len(rows) == 0, "...and the store really is empty")
    check(not any(crops.glob("*.jpg")), "...and the crops are gone from disk too")
    check(await fp.purge_expired(7) == 0, "a normal retention purges nothing here")
    check(await fp.purge_expired("nonsense") == 0,
          "an unparseable retention falls back to the default rather than wiping")

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (face pass, real models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
