#!/usr/bin/env python3
"""Reading a plate back from the RECORDING after the car has gone.

The live path gets one or two late looks at a passing car. This suite builds
a real H.264 recording — the fast-alpr test photograph (plate 5AU 5341) driven
across a 1920x1080 frame — files it where the recorder would, and feeds the
plate pass detect frames at a quarter of that size, where the plate is ~35 px
and cannot be read. Then:

1. WITHOUT the replay, nothing is stored: the live path had nothing legible.
2. WITH it, the pass decodes the seconds the car was in view, reads dozens of
   frames, and stores 5AU5341.
3. A second vehicle tracked somewhere else in the same seconds does NOT pick
   up this car's plate: each frame's plate must be on the vehicle's own
   position at that moment.
4. It never blocks: finish() returns at once and the vote lands later.

Needs ffmpeg (on PATH, or VIGILUME_TEST_FFMPEG, or the imageio-ffmpeg package)
and the plate models + photo (network on first run). SKIPS otherwise.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.native.plates import PlateReader  # noqa: E402
from app.native.platepass import PlatePass  # noqa: E402
from app.native.platereplay import PlateReplay  # noqa: E402
from plate_real_smoke import CAR, TRUTH, fetch_photo  # noqa: E402
from plates_smoke import FakeDB, Obs  # noqa: E402

PASS = 0
_failures: list[str] = []

W, H, FPS = 1920, 1080, 15
SCALE = 1.5           # the photo is pasted 1.5x — plate ~140 px at full res
DET = 0.25            # detect frames are a quarter size — plate ~35 px
CLIP_S = 12.0
DRIVE = (1.0, 6.0)    # the car crosses between these seconds


def check(cond: bool, label: str) -> None:
    global PASS
    PASS += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


def find_ffmpeg() -> str | None:
    env = os.environ.get("VIGILUME_TEST_FFMPEG")
    if env:
        return env
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def car_x(t: float) -> float | None:
    """Left edge of the pasted photo at time t, or None when out of view."""
    a, b = DRIVE
    if not a <= t <= b:
        return None
    return -300.0 + (t - a) / (b - a) * 1500.0


def background() -> np.ndarray:
    rng = np.random.default_rng(5)
    bg = np.full((H, W, 3), 96, np.uint8)
    bg = cv2.add(bg, rng.integers(0, 14, (H, W, 3), dtype=np.uint8))
    cv2.rectangle(bg, (0, 900), (W, H), (70, 72, 75), -1)
    return bg


def frame_at(t: float, bg: np.ndarray, car: np.ndarray) -> np.ndarray:
    f = bg.copy()
    x = car_x(t)
    if x is None:
        return f
    ch, cw = car.shape[:2]
    x0, y0 = int(x), 480
    sx0, sx1 = max(0, -x0), min(cw, W - x0)
    if sx1 > sx0:
        f[y0:y0 + ch, x0 + sx0:x0 + sx1] = car[:, sx0:sx1]
    return f


def car_box_at(t: float, car_shape) -> tuple[float, float, float, float] | None:
    """The CAR (not the whole photo) at time t, in full-frame pixels."""
    x = car_x(t)
    if x is None:
        return None
    x1, y1, x2, y2 = (v * SCALE for v in CAR)
    return (x + x1, 480 + y1, x + x2, 480 + y2)


def write_recording(ffmpeg: str, rec_dir: Path, start: float, bg, car) -> Path:
    """A 12 s H.264 segment filed exactly where the recorder would put it.

    Written as MPEG-TS, like the recorder. If THIS ffmpeg cannot read MPEG-TS
    back (one static build crashes in the mpegts demuxer), the same video is
    written as MP4 into the .ts path instead — ffmpeg picks the demuxer from
    the content, so the replay code under test is unchanged.
    """
    lt = time.localtime(start)
    seg = (rec_dir / "drive" / time.strftime("%Y-%m-%d", lt) / time.strftime("%H", lt)
           / (time.strftime("%M.%S", lt) + ".ts"))
    seg.parent.mkdir(parents=True, exist_ok=True)
    for container in ("mpegts", "mp4"):
        proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-f", container, str(seg)],
            stdin=subprocess.PIPE,
        )
        for i in range(int(CLIP_S * FPS)):
            proc.stdin.write(frame_at(i / FPS, bg, car).tobytes())
        proc.stdin.close()
        proc.wait()
        if PlateReplay(lambda cam: rec_dir / cam, ffmpeg)._probe_dims(seg) is not None:
            if container != "mpegts":
                print(f"  (this ffmpeg cannot read MPEG-TS back; using {container})")
            return seg
    return seg


class Cam:
    row = {"name": "drive", "ip": ""}
    plate_zones: list = []


async def drive(pp: PlatePass, start: float, bg, car, *, tid: int = 1, offset_box=None) -> None:
    """Feed the pass quarter-size detect frames at 5 fps while the car crosses."""
    t = DRIVE[0]
    while t <= DRIVE[1]:
        frame = cv2.resize(frame_at(t, bg, car), (int(W * DET), int(H * DET)),
                           interpolation=cv2.INTER_AREA)
        box = offset_box or car_box_at(t, car.shape)
        if box is not None:
            dbox = tuple(v * DET for v in box)
            await pp.observe(Cam(), [Obs(tid, dbox)], frame, start + t, event_fid="native.drive")
        t += 0.2


async def main() -> int:
    ffmpeg = find_ffmpeg()
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-plate-models-"))
    photo = fetch_photo(models_dir)
    reader = PlateReader(models_dir)
    if ffmpeg is None or photo is None or not await reader.load() or not reader.has_detector:
        print("SKIP: needs ffmpeg, the plate models and the test photo")
        return 0

    car = cv2.resize(photo, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    bg = background()
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-plate-replay-"))
    rec = tmp / "recordings"
    start = float(int(time.time()) - 120)  # a whole second, two minutes ago
    write_recording(ffmpeg, rec, start, bg, car)
    replay = PlateReplay(lambda cam: rec / cam, ffmpeg)

    print("\nwithout the replay: the live path had nothing legible")
    db = FakeDB(tmp / "a.db")
    pp = PlatePass(reader, db, tmp / "crops-a")
    await pp.reload_gallery()
    await drive(pp, start, bg, car)
    await pp.finish("drive", 1)
    await pp.wait_idle()
    check(db.rows("SELECT * FROM event_recognitions") == [],
          "a ~35 px plate on the detect frame is not read, so nothing is stored")

    print("\nwith the replay: the recording is read (in bursts, and after the car has gone)")
    db = FakeDB(tmp / "b.db")
    pp = PlatePass(reader, db, tmp / "crops-b", replay=replay)
    await pp.reload_gallery()
    await drive(pp, start, bg, car)
    t0 = time.monotonic()
    await pp.finish("drive", 1)
    check(time.monotonic() - t0 < 0.2, "finish() returns at once — the replay runs in the background")
    await pp.wait_idle()
    rows = db.rows("SELECT plate FROM event_recognitions")
    check([r["plate"] for r in rows] == [TRUTH],
          f"the plate is read from the recording and stored as {TRUTH} "
          f"(got {[r['plate'] for r in rows]})")
    stats = pp.status()["cameras"]["drive"]
    check(1 <= stats["replays"] <= 7 and stats["replay_frames"] >= 20,
          f"from {stats['replays']} read(s) of the recording, {stats['replay_frames']} frames "
          "— dozens of looks, not the one or two the live path gets")
    from app.native.platepass import SETTLED_READS

    check(stats["replay_reads"] >= SETTLED_READS,
          f"{stats['replay_reads']} reads went into the vote — reading stops once the vote "
          f"is settled ({SETTLED_READS}+ agreeing reads), not after every frame")
    cands = db.rows("SELECT * FROM recognition_candidates")
    crop = cv2.imread(str(tmp / "crops-b" / cands[0]["image_path"])) if cands else None
    check(crop is not None and crop.shape[1] >= 100,
          "and the stored candidate crop is a full-resolution look from the recording")

    print("\na different vehicle in the same seconds does not get this car's plate")
    db = FakeDB(tmp / "c.db")
    pp = PlatePass(reader, db, tmp / "crops-c", replay=replay)
    await pp.reload_gallery()
    # Tracked on the far left, near the top — nowhere near the plate.
    await drive(pp, start, bg, car, tid=2, offset_box=(20.0, 60.0, 400.0, 360.0))
    await pp.finish("drive", 2)
    await pp.wait_idle()
    check(db.rows("SELECT * FROM event_recognitions") == [],
          "its replay reads nothing: every frame's plate has to be on the "
          "vehicle's OWN position at that moment")

    print("\nGPU decode first when there is a GPU, the CPU when it cannot start")
    from app.native.platereplay import replay_attempts

    att = replay_attempts("ffmpeg", Path("seg.ts"), 1.0, 5.0, 6.0, (64, 64, 0, 0), hwaccel=True)
    check([h for h, _ in att] == ["cuda", "cuda", "cpu", "cpu"]
          and att[0][1][att[0][1].index("-hwaccel") + 1] == "cuda"
          and att[0][1].index("-hwaccel") < att[0][1].index("-i"),
          "with an NVIDIA GPU: NVDEC first (-hwaccel cuda, before the input), then the CPU")
    cpu_only = replay_attempts("ffmpeg", Path("seg.ts"), 1.0, 5.0, 6.0, (64, 64, 0, 0),
                               hwaccel=False)
    check(all("-hwaccel" not in a for _, a in cpu_only) and len(cpu_only) == 2,
          "without one: CPU only, nothing tried that cannot work")
    db = FakeDB(tmp / "g.db")
    forced = PlateReplay(lambda cam: rec / cam, ffmpeg, hwaccel=True)
    pp = PlatePass(reader, db, tmp / "crops-g", replay=forced)
    await pp.reload_gallery()
    await drive(pp, start, bg, car)
    await pp.finish("drive", 1)
    await pp.wait_idle()
    rows = db.rows("SELECT plate FROM event_recognitions")
    check([r["plate"] for r in rows] == [TRUTH],
          f"GPU decode asked for on a box with no GPU still reads the plate ({forced.decodes})")
    forced._hw_misses = 2
    await asyncio.to_thread(forced._frames_blocking, "drive", start + 1.0, start + 3.0,
                            (0.0, 0.3, 0.6, 0.9))
    check(forced.status()["gpu_decode"] is False,
          "after three GPU decodes in a row that produced nothing, it stops trying the GPU")

    print("\nno recording, no problem")
    db = FakeDB(tmp / "d.db")
    pp = PlatePass(reader, db, tmp / "crops-d",
                   replay=PlateReplay(lambda cam: tmp / "nowhere" / cam, ffmpeg))
    await drive(pp, start, bg, car)
    await pp.finish("drive", 1)
    await pp.wait_idle()
    check(pp.status()["cameras"]["drive"]["replays"] == 0,
          "a camera that is not recording simply gets no replay")

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (plate replay from the recording)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
