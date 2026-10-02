#!/usr/bin/env python3
"""Recorded bursts: full-resolution frames from just BEFORE detection, read
while the person or vehicle is still in view.

The situation this exists for: a car or a person moving quickly, legible for a
moment that has passed by the time detection has confirmed them and a
snapshot has come back. Pinned here:

1. Track paths: interpolation between sightings, extrapolation before the
   first one (clamped), and the area a window of frames must cover.
2. Burst timing: the first burst a second after first sighting, reading from
   two seconds before it; repeats while in view; a final one after the last
   sighting; coverage taken from the frames that actually came back.
3. Segment times: the previous segment's last write pins a segment's start
   (file names are whole seconds); across a recorder gap the name is used.
   And every decoded frame carries its TRUE time — a window starting between
   keyframes, or spanning two segment files, included. (Seeking the joined
   segments landed on the NEXT keyframe: frames came back 0.8 s later than
   labelled, and the start of the window was missing.)
4. A live read drops the last frame of a segment still being written.
5. PLATES: a car whose plate is legible only BEFORE it is first detected — the
   detect frames are too small and it is covered afterwards — is read from
   the recording, while it is still in view.
6. FACES: a face turned to the camera only before the person is detected is
   recognized from the recording; a face elsewhere in the frame is not given
   to another person's track; the switch turns it off.

Sections 5 and 6 need ffmpeg and the models + photos (network on first run);
they SKIP without. Sections 1-4 always run.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from app.native import burst, trackpath  # noqa: E402
from app.native.burst import BurstState  # noqa: E402
from app.native.platereplay import RecordingReplay, ReplayFrame, segment_starts  # noqa: E402

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


def close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------- 1. paths

def path_checks() -> None:
    print("\n1. where a tracked object was at any moment")
    # Moving right at 0.2 frame-widths a second, sighted at 5 fps from t=10.
    path = [(10.0 + 0.2 * i, (0.1 + 0.04 * i, 0.4, 0.3 + 0.04 * i, 0.8)) for i in range(6)]
    check(close(trackpath.box_at(path, 10.1), (0.12, 0.4, 0.32, 0.8)),
          "between two sightings: interpolated")
    check(close(trackpath.box_at(path, 9.5), (0.0, 0.4, 0.2, 0.8)),
          "half a second BEFORE the first sighting: extrapolated back along its motion")
    check(close(trackpath.box_at(path, 5.0), trackpath.box_at(path, 9.0)),
          "and no further back than a second — a guess past that is not trusted")
    near = trackpath.box_near(path, 10.4, 0.35)
    check(near[0] < trackpath.box_at(path, 10.4)[0] < trackpath.box_at(path, 10.4)[2] < near[2],
          "a recorded frame's time is known to a few hundred ms, so it is matched to "
          "everywhere the object was in that span")
    region = trackpath.union_over(path, 8.0, 11.0)
    check(region[0] <= 0.0 + 1e-9 and region[2] >= 0.5,
          "a burst's crop covers the pre-roll positions as well as the sightings")


# ---------------------------------------------------------------- 2. timing

def timing_checks() -> None:
    print("\n2. when bursts happen and what they read")
    b = BurstState()
    b.note_seen(100.0)
    check(b.due(100.5) is None, "not in the first second")
    w = b.due(100.0 + burst.FIRST_BURST_AFTER_S)
    check(w is not None and abs(w[0] - (100.0 - burst.PRE_ROLL_S)) < 1e-9,
          f"a second after first sighting, reading from {burst.PRE_ROLL_S:g} s BEFORE it")
    check(abs(w[1] - (101.0 - burst.REC_LAG_S)) < 1e-9,
          "up to what the recording on disk can hold by now")
    b.started()
    b.finished(*w, last_frame=100.1, fps=burst.BURST_FPS)
    check(abs(b.covered_to - 100.2) < 1e-9,
          "coverage is where the frames that came back ended, not what was asked for")
    b.note_seen(101.0)
    # The next one needs BURST_EVERY_S of new recording past 100.2.
    ready = 100.2 + burst.BURST_EVERY_S + burst.REC_LAG_S
    check(b.due(ready - 0.3) is None, "no new burst until there is enough new recording")
    w2 = b.due(ready + 0.1)
    check(w2 is not None and abs(w2[0] - 100.2) < 1e-9, "the next one carries on from there")
    b.started()
    b.finished(*w2, last_frame=w2[1] - 0.15, fps=burst.BURST_FPS)
    b.note_seen(102.0)
    fin = b.window(final=True)
    check(fin is not None and abs(fin[1] - (102.0 + burst.POST_ROLL_S)) < 1e-9,
          "the final one reads past the last sighting")
    gone = BurstState()
    gone.note_seen(50.0)
    gone.started()
    gone.finished(48.0, 50.2, last_frame=None, fps=burst.BURST_FPS)
    check(gone.unavailable and gone.due(60.0) is None,
          "a burst with no frames at all (camera not recording) stops further ones")
    cap = BurstState()
    cap.note_seen(0.0)
    for _ in range(burst.MAX_BURSTS):
        cap.started()
    check(cap.due(50.0) is None, f"at most {burst.MAX_BURSTS} per track")


# ---------------------------------------------------------------- 3. segment times

def segment_time_checks(tmp: Path) -> None:
    print("\n3. when a recorded segment really started")
    d = tmp / "segs"
    d.mkdir()
    names = [1000.0, 1010.0, 1020.0, 1050.0]   # the last after a recorder gap
    paths = []
    for named in names:
        p = d / f"{int(named)}.ts"
        p.write_bytes(b"x")
        paths.append(p)
    # Each segment was cut 0.54 s after its whole-second name; the file before
    # it was last written at that instant.
    os.utime(paths[0], (1010.54, 1010.54))
    os.utime(paths[1], (1020.54, 1020.54))
    os.utime(paths[2], (1030.54, 1030.54))
    os.utime(paths[3], (1060.54, 1060.54))
    got = [t for t, _ in segment_starts(list(zip(names, paths)))]
    check(close(got[1:3], (1010.54, 1020.54), 1e-3),
          "a segment starts when the previous one was last written (not at its name's whole second)")
    check(got[0] == 1000.0, "the first, with nothing before it, keeps its name")
    check(got[3] == 1050.0, "after a gap in recording, the name is all there is")


# ---------------------------------------------------------------- 4. torn frame

async def torn_frame_checks() -> None:
    print("\n4. a segment still being written: its last frame is dropped")
    rr = RecordingReplay(lambda cam: Path("/nowhere"), "/fake/ffmpeg", hwaccel=False)
    fr = [ReplayFrame(time=float(i), crop=np.zeros((4, 4, 3), np.uint8), ox=0, oy=0,
                      width=4, height=4) for i in range(5)]
    rr._frames_blocking = lambda cam, s0, e0, regions, *a, **k: [list(fr) for _ in regions]
    live = await rr.frames("c", 0.0, 4.0, (0, 0, 1, 1), pad=False, live=True)
    after = await rr.frames("c", 0.0, 4.0, (0, 0, 1, 1), pad=False, live=False)
    check(len(live) == 4 and len(after) == 5, "live: 4 of 5 frames; after the fact: all 5")


# ---------------------------------------------------------------- helpers

def find_ffmpeg():
    from plate_replay_smoke import find_ffmpeg as f

    return f()


def write_video(ffmpeg: str, rec_dir: Path, camera: str, start: float, frame_at, seconds: float,
                fps: int, size: tuple[int, int], *, t0: float = 0.0) -> Path:
    """A recording filed where the recorder would put it (MPEG-TS; MP4 inside
    the .ts if this ffmpeg cannot read MPEG-TS back — see plate_replay_smoke)."""
    lt = time.localtime(start)
    seg = (rec_dir / camera / time.strftime("%Y-%m-%d", lt) / time.strftime("%H", lt)
           / (time.strftime("%M.%S", lt) + ".ts"))
    seg.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    for container in ("mpegts", "mp4"):
        proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", str(fps),
             "-pix_fmt", "yuv420p", "-f", container, str(seg)],
            stdin=subprocess.PIPE,
        )
        for i in range(int(round(seconds * fps))):
            proc.stdin.write(frame_at(t0 + i / fps).tobytes())
        proc.stdin.close()
        proc.wait()
        if RecordingReplay(lambda cam: rec_dir / cam, ffmpeg)._probe_dims(seg) is not None:
            return seg
    return seg


class Settings:
    def __init__(self, **recognition):
        self.current = {"recognition": {"face_hires": False, **recognition}}


# ---------------------------------------------------------------- 5. plates

async def plate_checks(tmp: Path, ffmpeg: str, models_dir: Path) -> None:
    print("\n5. a car legible only BEFORE it is detected")
    from app.native.plates import PlateReader
    from app.native.platepass import PlatePass
    from plate_real_smoke import CAR, TRUTH, fetch_photo
    from plates_smoke import FakeDB, Obs

    photo = fetch_photo(models_dir)
    reader = PlateReader(models_dir)
    if photo is None or not await reader.load() or not reader.has_detector:
        print("  SKIP: needs the plate models and the test photo")
        return
    W, H, FPS, SCALE, DET = 1920, 1080, 15, 1.5, 0.25
    car = cv2.resize(photo, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    plate = (reader.detect_blocking(car) or [None])[0]
    check(plate is not None, "the plate's position in the car photo is known (to cover it later)")
    rng = np.random.default_rng(5)
    bg = cv2.add(np.full((H, W, 3), 96, np.uint8), rng.integers(0, 14, (H, W, 3), dtype=np.uint8))
    VISIBLE, COVERED_FROM, SEEN_FROM, SEEN_TO = (0.5, 5.0), 2.4, 2.2, 4.4

    def car_x(t):
        a, b = VISIBLE
        return None if not a <= t <= b else -200.0 + (t - a) * 250.0

    def frame_at(t):
        f = bg.copy()
        x = car_x(t)
        if x is None:
            return f
        img = car.copy()
        if t >= COVERED_FROM:     # turned / glare / behind the gate: unreadable from here on
            x1, y1, x2, y2 = (int(v) for v in plate[:4])
            img[y1:y2, x1:x2] = 90
        x0, ch, cw = int(x), img.shape[0], img.shape[1]
        sx0, sx1 = max(0, -x0), min(cw, W - x0)
        if sx1 > sx0:
            f[480:480 + ch, x0 + sx0:x0 + sx1] = img[:, sx0:sx1]
        return f

    start = float(int(time.time()) - 120)
    rec = tmp / "rec-plates"
    write_video(ffmpeg, rec, "drive", start, frame_at, 8.0, FPS, (W, H))
    db = FakeDB(tmp / "plates.db")
    announced: list[tuple[float, str]] = []
    pp = PlatePass(reader, db, tmp / "crops-p",
                   replay=RecordingReplay(lambda cam: rec / cam, ffmpeg, hwaccel=False))
    pp.on_recognition = lambda fid, kind, **kw: announced.append((time.monotonic(), kw.get("plate", "")))
    await pp.reload_gallery()

    class Cam:
        row = {"name": "drive"}
        plate_zones: list = []

    t = SEEN_FROM
    while t <= SEEN_TO:      # detection only catches it from here on
        x = car_x(t)
        frame = cv2.resize(frame_at(t), (int(W * DET), int(H * DET)), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = (v * SCALE for v in CAR)
        box = tuple(v * DET for v in (x + x1, 480 + y1, x + x2, 480 + y2))
        await pp.observe(Cam(), [Obs(1, box)], frame, start + t, event_fid="native.drive")
        await asyncio.sleep(0.12)    # roughly real time, so bursts run alongside
        t += 0.2
    in_view_until = time.monotonic()
    # Let a burst already started finish, as it would with the car still there.
    for _ in range(100):
        st = pp._tracks.get(("drive", 1))
        if st is None or not st.burst.busy:
            break
        await asyncio.sleep(0.1)
    check(any(p == TRUTH for _, p in announced),
          f"read from the recording while the car is still tracked — before it is retired "
          f"(announced {[p for _, p in announced]})")
    await pp.finish("drive", 1)
    await pp.wait_idle()
    rows = db.rows("SELECT plate FROM event_recognitions")
    check([r["plate"] for r in rows] == [TRUTH], f"and stored as {TRUTH} (got {[r['plate'] for r in rows]})")
    stats = pp.status()["cameras"]["drive"]
    check(stats["early_reads"] > 0,
          f"from frames taken before the car was first detected ({stats['early_reads']} early reads)")
    check(stats["hires_frames"] == 0, "with no snapshot involved")
    del in_view_until


# ---------------------------------------------------------------- 6. faces

async def face_checks(tmp: Path, ffmpeg: str, models_dir: Path) -> None:
    print("\n6. a face turned to the camera only BEFORE the person is detected")
    from app.native.facepass import FacePass
    from app.native.recognizer import FaceRecognizer
    from facepass_smoke import Cam, Obs, load_face_image, make_db

    face = load_face_image()
    rec_models = FaceRecognizer(models_dir)
    if face is None or not await rec_models.load():
        print("  SKIP: needs the face models and a sample face photo")
        return
    W, H, FPS, DET = 1280, 720, 15, 0.25
    face = cv2.resize(face, (260, 260), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(9)
    bg = cv2.add(np.full((H, W, 3), 105, np.uint8), rng.integers(0, 20, (H, W, 3), dtype=np.uint8))
    FRONTAL, SEEN_FROM, SEEN_TO = (0.4, 2.0), 2.2, 4.4

    def head_x(t):
        return 420 + 30 * t          # walking slowly across

    def frame_at(t, *, frontal=None):
        f = bg.copy()
        x, y = int(head_x(t)), 60
        cv2.rectangle(f, (x - 40, y + 260), (x + 300, H - 1), (60, 70, 95), -1)   # body
        if (FRONTAL[0] <= t <= FRONTAL[1]) if frontal is None else frontal:
            f[y:y + 260, x:x + 260] = face
        else:                         # turned away: the back of a head
            cv2.ellipse(f, (x + 130, y + 130), (95, 125), 0, 0, 360, (45, 50, 60), -1)
        return f

    def person_box(t):
        x = head_x(t)
        return tuple(v * DET for v in (x - 50, 40, x + 310, H - 1))

    start = float(int(time.time()) - 120)
    rec = tmp / "rec-faces"
    write_video(ffmpeg, rec, "front", start, frame_at, 6.0, FPS, (W, H))
    replay = RecordingReplay(lambda cam: rec / cam, ffmpeg, hwaccel=False)

    def true_time(img, ox):
        # The body's left edge moves 30 px a second: it says when a frame is from.
        xs = np.nonzero((np.abs(img[500 - 0].astype(int) - [60, 70, 95]).sum(axis=1) < 12))[0]
        return None if xs.size == 0 else (ox + xs[0] + 40 - 420) / 30.0

    print("\n   recorded frames carry their true time")
    for a, b in ((0.2, 2.4), (1.37, 3.9)):     # both start between keyframes
        fr = await replay.frames("front", start + a, start + b, (0, 0, 1, 1), fps=10, pad=False)
        errs = [abs((f.time - start) - tt) for f in fr if (tt := true_time(f.crop, f.ox)) is not None]
        check(len(fr) >= round((b - a) * 10) - 1 and errs and max(errs) <= 0.1,
              f"window {a}-{b} s: {len(fr)} frames, each within {max(errs or [9]):.3f} s of its label")
    # Two segment files, cut at 3.0 s; the first one's last write marks the cut.
    rec2 = tmp / "rec-two"
    first = write_video(ffmpeg, rec2, "front", start, frame_at, 3.0, FPS, (W, H))
    write_video(ffmpeg, rec2, "front", start + 3.0, frame_at, 3.0, FPS, (W, H), t0=3.0)
    os.utime(first, (start + 3.0, start + 3.0))
    fr = await RecordingReplay(lambda cam: rec2 / cam, ffmpeg, hwaccel=False).frames(
        "front", start + 2.0, start + 4.5, (0, 0, 1, 1), fps=10, pad=False)
    errs = [abs((f.time - start) - tt) for f in fr if (tt := true_time(f.crop, f.ox)) is not None]
    check(len(fr) >= 24 and errs and max(errs) <= 0.1,
          f"a window across two segment files: {len(fr)} frames, each within "
          f"{max(errs or [9]):.3f} s of its label")

    print("\n   one decode serves a face read and a plate read on the same camera")
    shared = RecordingReplay(lambda cam: rec / cam, ffmpeg, hwaccel=False)
    left, right = (0.05, 0.1, 0.35, 0.9), (0.6, 0.05, 0.95, 0.95)
    a_task = asyncio.ensure_future(shared.frames("front", start + 1.0, start + 2.5, left, fps=10, pad=False))
    b_task = asyncio.ensure_future(shared.frames("front", start + 1.5, start + 3.0, right, fps=10, pad=False))
    fa, fb = await a_task, await b_task
    check(sum(shared.decodes.values()) == 1 and shared.shared == 1,
          f"two reads waiting together, one decode ({shared.decodes}, {shared.shared} shared)")
    check(len(fa) >= 14 and len(fb) >= 14 and abs(fa[0].time - (start + 1.0)) < 0.06
          and abs(fb[0].time - (start + 1.5)) < 0.06,
          f"each gets its own seconds ({len(fa)} and {len(fb)} frames)")
    from app.native.platereplay import REGION_MARGIN

    want_ox = int((right[0] - (right[2] - right[0]) * REGION_MARGIN) * W) & ~1
    check(bool(fb) and abs(fb[0].ox - want_ox) <= 2 and fb[0].crop.shape[1] <= W - want_ox,
          f"and its own crop, placed where it is in the frame (right crop at x={fb[0].ox if fb else '?'}, "
          f"expected {want_ox})")
    errs = [abs((f.time - start) - tt) for f in fa if (tt := true_time(f.crop, f.ox)) is not None]
    check(bool(errs) and max(errs) <= 0.1,
          f"the shared decode keeps exact times (left crop: each frame within {max(errs or [9]):.3f} s)")

    print("\n   the OpenCV face models are safe to call from several threads at once")
    import concurrent.futures as cf

    crops = [frame_at(1.0)[0:h, 0:w] for h, w in ((360, 480), (400, 520), (300, 640), (480, 400))]
    errors: list[str] = []

    def hammer(img):
        try:
            for _ in range(15):
                rec_models.detect_blocking(img)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    import logging

    log_rec = logging.getLogger("app.native.recognizer")
    failures: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: failures.append(r.getMessage())  # type: ignore[assignment]
    log_rec.addHandler(handler)
    try:
        with cf.ThreadPoolExecutor(4) as ex:
            list(ex.map(hammer, crops))
    finally:
        log_rec.removeHandler(handler)
    check(not errors and not [f for f in failures if "failed" in f],
          f"4 threads x 15 detections of different sizes: no failures ({len(failures)} logged)")

    async def run(*, replay_on: bool, other: bool = False):
        db = make_db(tmp / f"faces-{replay_on}-{other}.db")
        fp = FacePass(rec_models, db, tmp / f"crops-f-{replay_on}-{other}", replay=replay)
        fp._settings = Settings(face_replay=replay_on)
        await fp.reload_gallery()
        cam = Cam()
        identified_live = False
        t = SEEN_FROM
        while t <= SEEN_TO:
            det = cv2.resize(frame_at(t), (int(W * DET), int(H * DET)), interpolation=cv2.INTER_AREA)
            obs = [Obs(1, person_box(t))]
            if other:   # a second person on the far right, same seconds
                obs = [Obs(2, tuple(v * DET for v in (1000, 100, 1270, 715)))]
            await fp.observe(cam, obs, det, start + t, event_fid="native.front")
            await asyncio.sleep(0.12)
            t += 0.2
        for _ in range(100):
            st = fp._tracks.get(("front", 2 if other else 1))
            if st is None or not st.burst.busy:
                break
            await asyncio.sleep(0.1)
        st = fp._tracks.get(("front", 2 if other else 1))
        identified_live = st is not None and st.identified_from is not None
        await fp.finish("front", 2 if other else 1)
        await fp.wait_idle()
        return fp, db, identified_live

    small = await rec_models.detect(cv2.resize(frame_at(1.0), (int(W * DET), int(H * DET))))
    check(not small, "on the detect frame the face is too small to be read at all")

    fp, db, live = await run(replay_on=False)
    check(db.rows("SELECT * FROM recognition_candidates") == [],
          "without bursts nothing is read: when detection catches up, they have turned away")

    fp, db, live = await run(replay_on=True)
    rows = db.rows("SELECT * FROM recognition_candidates")
    check(len(rows) == 1 and rows[0]["quality"] >= 0.6,
          f"with bursts the face is read and kept (quality {rows[0]['quality']:.2f})"
          if rows else "with bursts the face is read and kept")
    check(live, "and identified while the person is still tracked, not after they leave")
    b = fp.status()["bursts"]
    check(b["early_faces"] > 0,
          f"from frames taken BEFORE the person was first detected ({b})")

    fp, db, _ = await run(replay_on=True, other=True)
    check(db.rows("SELECT * FROM recognition_candidates") == [],
          "a person elsewhere in the frame in the same seconds is not given that face")


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-burst-"))
    path_checks()
    timing_checks()
    segment_time_checks(tmp)
    await torn_frame_checks()

    ffmpeg = find_ffmpeg()
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-burst-models-"))
    if ffmpeg is None:
        print("\nSKIP 5-6: no ffmpeg")
    else:
        await plate_checks(tmp, ffmpeg, models_dir)
        await face_checks(tmp, ffmpeg, models_dir)

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (recorded bursts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
